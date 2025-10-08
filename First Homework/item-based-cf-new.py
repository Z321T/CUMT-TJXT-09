import os
import urllib.request
import zipfile
import time
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ItemBasedCFOptimized:
    def __init__(self,
                 k_neighbors=120,
                 shrinkage=30,
                 min_common=3,
                 use_uf=True,
                 keep_neg_sims=True,
                 topn=10,
                 enable_bias=True,
                 bias_reg=10.0,
                 min_sim=0.0,
                 sim_alpha=1.0,
                 adaptive_k_percent=None,
                 fallback_min_neighbors=2):
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.use_uf = use_uf
        self.keep_neg_sims = keep_neg_sims
        self.topn = topn
        self.enable_bias = enable_bias
        self.bias_reg = bias_reg
        self.min_sim = min_sim
        self.sim_alpha = sim_alpha
        self.adaptive_k_percent = adaptive_k_percent
        self.fallback_min_neighbors = fallback_min_neighbors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        print(f"设备: {self.device}")

        self.ratings_df = None
        self.movies_df = None
        self.user_id_map = {}
        self.movie_id_map = {}
        self.inv_user_map = {}
        self.inv_movie_map = {}
        self.train_mat = None
        self.test_df = None
        self.global_mean = None
        self.user_means = None
        self.user_bias = None
        self.item_bias = None
        self.sim_item_topk = None
        self.pred_matrix = None

    # ---------------- 数据 ----------------
    def download_and_load_data(self, data_dir='./data'):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, 'ml-100k.zip')
        extract_path = os.path.join(data_dir, 'ml-100k')
        data_file = os.path.join(extract_path, 'u.data')
        if not os.path.exists(data_file):
            print("下载数据集...")
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(url, zip_path)
            print("解压数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
        cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(data_file, sep='\t', names=cols, encoding='latin-1')
        movie_file = os.path.join(extract_path, 'u.item')
        movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date',
                      'imdb_url'] + [f'genre_{i}' for i in range(19)]
        self.movies_df = pd.read_csv(movie_file, sep='|', names=movie_cols,
                                     encoding='latin-1', usecols=['movie_id', 'title'])
        print(f"加载评分: {len(self.ratings_df)}")

    def prepare_data(self, test_size=0.2, seed=42):
        users = sorted(self.ratings_df.user_id.unique())
        movies = sorted(self.ratings_df.movie_id.unique())
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.movie_id_map = {m: i for i, m in enumerate(movies)}
        self.inv_user_map = {i: u for u, i in self.user_id_map.items()}
        self.inv_movie_map = {i: m for m, i in self.movie_id_map.items()}
        self.ratings_df['u_idx'] = self.ratings_df['user_id'].map(self.user_id_map)
        self.ratings_df['m_idx'] = self.ratings_df['movie_id'].map(self.movie_id_map)

        train_df, test_df = train_test_split(self.ratings_df, test_size=test_size, random_state=seed)
        self.test_df = test_df.reset_index(drop=True)

        n_u = len(users)
        n_m = len(movies)
        self.train_mat = np.zeros((n_u, n_m), dtype=np.float32)
        for r in train_df.itertuples():
            self.train_mat[r.u_idx, r.m_idx] = r.rating

        self.global_mean = float(self.train_mat[self.train_mat > 0].mean())
        user_sum = self.train_mat.sum(axis=1)
        user_cnt = (self.train_mat > 0).sum(axis=1)
        self.user_means = np.divide(user_sum, user_cnt,
                                    out=np.full(n_u, self.global_mean, dtype=np.float32),
                                    where=user_cnt > 0)
        if self.enable_bias:
            self._compute_biases(reg=self.bias_reg)
        print(f"用户数:{n_u}  物品数:{n_m}")

    def _compute_biases(self, reg=10.0, iters=1):
        R = self.train_mat
        mask = (R > 0)
        n_u, n_m = R.shape
        b_u = np.zeros(n_u, dtype=np.float32)
        b_i = np.zeros(n_m, dtype=np.float32)
        mu = self.global_mean
        for _ in range(iters):
            item_cnt = mask.sum(axis=0)
            numer_i = (R - mu - b_u[:, None]) * mask
            b_i = numer_i.sum(axis=0) / (reg + item_cnt + 1e-6)
            user_cnt = mask.sum(axis=1)
            numer_u = (R - mu - b_i[None, :]) * mask
            b_u = numer_u.sum(axis=1) / (reg + user_cnt + 1e-6)
        self.user_bias = b_u
        self.item_bias = b_i

    # ---------------- 相似度 ----------------
    def compute_item_similarity(self):
        print("计算物品相似度(GPU)...")
        start = time.time()
        R = torch.from_numpy(self.train_mat).to(self.device, dtype=self.dtype)
        mask = (R > 0).to(dtype=self.dtype)

        if self.enable_bias:
            mu = torch.tensor(self.global_mean, device=self.device, dtype=self.dtype)
            b_u = torch.from_numpy(self.user_bias).to(self.device, dtype=self.dtype).unsqueeze(1)
            b_i = torch.from_numpy(self.item_bias).to(self.device, dtype=self.dtype).unsqueeze(0)
            residual = (R - (mu + b_u + b_i)) * mask
        else:
            user_means = torch.from_numpy(self.user_means).to(self.device, dtype=self.dtype).unsqueeze(1)
            residual = (R - user_means) * mask

        if self.use_uf:
            user_pop = mask.sum(1)
            uf = 1.0 / torch.log1p(user_pop + 1.0)
            residual = residual * uf.unsqueeze(1)

        C = residual
        norms = torch.norm(C, dim=0)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        sim = C.t() @ C
        denom = torch.outer(norms, norms)
        sim = sim / denom

        common = (mask.t() @ mask)

        if self.shrinkage > 0:
            sim = sim * (common / (common + self.shrinkage))

        sim = torch.where(common >= self.min_common, sim, torch.zeros_like(sim))

        if self.min_sim > 0:
            keep = sim.abs() >= self.min_sim
            sim = sim * keep

        if not self.keep_neg_sims:
            sim = torch.clamp(sim, min=0.0)

        sim.fill_diagonal_(0.0)

        if self.sim_alpha != 1.0:
            sign = torch.sign(sim)
            sim = sign * (sim.abs() ** self.sim_alpha)

        pruned = torch.zeros_like(sim)
        I = sim.size(0)
        if self.adaptive_k_percent is not None:
            for i in range(I):
                row = sim[i]
                nz_mask = row != 0
                nz_vals = row[nz_mask]
                if nz_vals.numel() == 0:
                    continue
                k_dyn = max(1, int(nz_vals.numel() * self.adaptive_k_percent))
                k_dyn = min(k_dyn, nz_vals.numel())
                top_vals, top_idx_local = torch.topk(nz_vals, k=k_dyn)
                idx_global = nz_mask.nonzero(as_tuple=False).squeeze(1)[top_idx_local]
                pruned[i, idx_global] = top_vals
        else:
            k = min(self.k_neighbors, I - 1)
            topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
            pruned.scatter_(1, topk_idx, topk_vals)

        self.sim_item_topk = pruned.to(self.device, dtype=self.dtype)
        print(f"物品相似度完成，用时 {time.time() - start:.2f}s")

    # ---------------- 预测 ----------------
    def predict_all(self):
        print("生成预测矩阵(GPU)...")
        if self.sim_item_topk is None:
            raise RuntimeError("需先计算相似度")
        start = time.time()
        R = torch.from_numpy(self.train_mat).to(self.device, dtype=self.dtype)
        mask = (R > 0).to(dtype=self.dtype)
        mu = torch.tensor(self.global_mean, device=self.device, dtype=self.dtype)

        if self.enable_bias:
            b_u = torch.from_numpy(self.user_bias).to(self.device, dtype=self.dtype).unsqueeze(1)
            b_i = torch.from_numpy(self.item_bias).to(self.device, dtype=self.dtype).unsqueeze(0)
            base = mu + b_u + b_i
            residual = (R - base) * mask
        else:
            user_means = torch.from_numpy(self.user_means).to(self.device, dtype=self.dtype).unsqueeze(1)
            base = user_means
            residual = (R - user_means) * mask

        S = self.sim_item_topk.to(self.device, dtype=self.dtype)
        numer = residual @ S.t()
        denom = mask @ (S.abs().t())
        denom = torch.clamp(denom, min=1e-6)
        neigh_count = (mask @ (S.abs() > 0).to(dtype=self.dtype).t())
        contribution = numer / denom
        contribution = torch.where(neigh_count >= self.fallback_min_neighbors,
                                   contribution,
                                   torch.zeros_like(contribution))
        pred = base + contribution
        pred = torch.clamp(pred, 1.0, 5.0)
        filled = pred.clone()
        filled[mask > 0] = R[mask > 0]
        self.pred_matrix = filled.detach().cpu().numpy()
        print(f"预测完成，用时 {time.time() - start:.2f}s")

    # ---------------- 评分评估 ----------------
    def evaluate_rating(self):
        if self.pred_matrix is None:
            raise RuntimeError("先调用 predict_all()")
        y_true, y_pred = [], []
        for row in self.test_df.itertuples():
            y_true.append(row.rating)
            y_pred.append(self.pred_matrix[row.u_idx, row.m_idx])
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    # ---------------- TopN 评估 ----------------
    def evaluate_topn(self, threshold=4.0):
        if self.pred_matrix is None:
            raise RuntimeError("先调用 predict_all()")
        R_train = self.train_mat
        P = self.pred_matrix
        n_users, n_items = P.shape
        test_group = self.test_df.groupby('u_idx')
        precision_list, recall_list = [], []
        coverage_items = set()
        item_pop = (R_train > 0).sum(axis=0)
        pop_list = []
        for u, grp in test_group:
            liked = set(grp[grp.rating >= threshold].m_idx.values)
            if not liked:
                continue
            seen_mask = R_train[u] > 0
            scores = P[u].copy()
            scores[seen_mask] = -1e9
            top_idx = np.argpartition(-scores, self.topn)[:self.topn]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            rec_set = set(top_idx.tolist())
            hit = liked & rec_set
            precision_list.append(len(hit) / self.topn)
            recall_list.append(len(hit) / len(liked))
            coverage_items.update(rec_set)
            pop_list.append(item_pop[list(rec_set)].mean() if rec_set else 0)
        precision = float(np.mean(precision_list)) if precision_list else 0.0
        recall = float(np.mean(recall_list)) if recall_list else 0.0
        coverage = len(coverage_items) / n_items
        popularity = float(np.mean(pop_list)) if pop_list else 0.0
        return precision, recall, coverage, popularity

    # ---------------- 运行 ----------------
    def run(self):
        self.download_and_load_data()
        self.prepare_data()
        self.compute_item_similarity()
        self.predict_all()
        rmse, mae = self.evaluate_rating()
        precision, recall, coverage, popularity = self.evaluate_topn()
        print("\n=== 指标(Item-Based 优化) ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Precision@{self.topn}: {precision:.4f}")
        print(f"Recall@{self.topn}: {recall:.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Popularity: {popularity:.2f}")


if __name__ == "__main__":
    model = ItemBasedCFOptimized(
        k_neighbors=60,
        shrinkage=23,
        min_common=2,
        use_uf=True,
        keep_neg_sims=False,
        topn=10,
        enable_bias=True,
        bias_reg=15.0,
        min_sim=0.1,
        sim_alpha=1.0,
        adaptive_k_percent=None,
        fallback_min_neighbors=2
    )
    model.run()
