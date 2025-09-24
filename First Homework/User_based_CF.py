import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import math
from collections import defaultdict
import time

warnings.filterwarnings("ignore")

# 安全导入 torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class UserBasedCF:
    """
    基于用户的协同过滤（GPU 加速相似度 + 批量推荐）
    功能:
      - Pearson / Cosine 相似度（GPU 向量化）
      - 用户/物品偏置基线
      - Top-K 邻域去偏差加权
      - Top-N 推荐与评估 (Precision/Recall/Coverage/Novelty)
    """

    def __init__(
        self,
        similarity_metric: str = "pearson",
        k: int = 60,
        min_overlap: int = 5,
        shrinkage: float = 25.0,
        normalize: bool = True,
        bias_reg: float = 15.0,
        use_gpu: bool = True,
        candidate_pool_size: int = 300,
        progress_every: int = 50
    ):
        self.similarity_metric = similarity_metric
        self.k = k
        self.min_overlap = min_overlap
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.bias_reg = bias_reg
        # GPU 可用性判断
        self.has_torch = HAS_TORCH
        self.use_gpu = bool(
            use_gpu and self.has_torch and torch.cuda.is_available()
        )
        if self.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = "cpu"
        # 矩阵与统计
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_similarity_matrix: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None
        self.user_bias: Optional[pd.Series] = None
        self.item_bias: Optional[pd.Series] = None

        # 邻居缓存
        self.sorted_neighbors: Optional[List[np.ndarray]] = None

        # 推荐阶段加速参数
        self.candidate_pool_size = candidate_pool_size
        self.progress_every = progress_every

    # ================= 数据与训练 =================
    def load_movielens_data(self, split: str = "train", dataset: str = "movielens/100k-ratings") -> pd.DataFrame:
        print("正在加载MovieLens数据集...")
        ds = tfds.load(dataset, split=split, as_supervised=False)
        rows = []
        for ex in ds:
            uid = ex["user_id"].numpy()
            mid = ex["movie_id"].numpy()
            r = float(ex["user_rating"].numpy())
            if isinstance(uid, (bytes, bytearray)):
                uid = uid.decode()
            if isinstance(mid, (bytes, bytearray)):
                mid = mid.decode()
            rows.append({"user_id": uid, "movie_id": mid, "rating": r})
        df = pd.DataFrame(rows)
        print(f"数据加载完成，共有 {len(df)} 条评分记录")
        print(f"用户数量: {df['user_id'].nunique()} | 电影数量: {df['movie_id'].nunique()}")
        return df

    def build_user_item_matrix(self, ratings_df: pd.DataFrame):
        print("正在构建用户-物品评分矩阵...")
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", aggfunc="mean"
        )
        self.global_mean = float(np.nanmean(self.user_item_matrix.values))
        self._compute_biases()
        print(f"评分矩阵构建完成，形状: {self.user_item_matrix.shape}")

    def _compute_biases(self):
        print("正在计算基线偏置（用户/物品）...")
        stacked = self.user_item_matrix.stack(dropna=True)
        mu = self.global_mean

        # 物品偏置
        item_grp = stacked.groupby(level=1)
        item_cnt = item_grp.size()
        item_resid_sum = item_grp.apply(lambda s: float((s - mu).sum()))
        b_i = item_resid_sum / (self.bias_reg + item_cnt)
        self.item_bias = b_i.reindex(self.user_item_matrix.columns).fillna(0.0)

        # 用户偏置
        stacked_item_bias = stacked.index.get_level_values(1).map(self.item_bias)
        user_resid = stacked - mu - stacked_item_bias
        user_grp = user_resid.groupby(level=0)
        user_cnt = user_grp.size()
        user_resid_sum = user_grp.sum()
        b_u = user_resid_sum / (self.bias_reg + user_cnt)
        self.user_bias = b_u.reindex(self.user_item_matrix.index).fillna(0.0)

    # ================= 相似度（GPU 向量化） =================
    def compute_user_similarity(self):
        print(f"正在计算用户相似度矩阵（{self.similarity_metric}，GPU={self.use_gpu}）...")
        if self.use_gpu:
            self._compute_user_similarity_gpu()
        else:
            self._compute_user_similarity_cpu()
        print("用户相似度矩阵计算完成")
        self._cache_sorted_neighbors()

    def _compute_user_similarity_cpu(self):
        values = self.user_item_matrix.values
        n_users = values.shape[0]
        sims = np.eye(n_users, dtype=np.float32)
        for i in range(n_users):
            xi = values[i]
            for j in range(i + 1, n_users):
                yj = values[j]
                mask = ~np.isnan(xi) & ~np.isnan(yj)
                n_common = int(mask.sum())
                if n_common < self.min_overlap:
                    continue
                x = xi[mask]
                y = yj[mask]
                if self.similarity_metric == "pearson":
                    x_c = x - x.mean()
                    y_c = y - y.mean()
                    denom = np.linalg.norm(x_c) * np.linalg.norm(y_c)
                    if denom == 0:
                        continue
                    sim = float(np.dot(x_c, y_c) / denom)
                elif self.similarity_metric == "cosine":
                    if self.normalize:
                        x = x - x.mean()
                        y = y - y.mean()
                    denom = np.linalg.norm(x) * np.linalg.norm(y)
                    if denom == 0:
                        continue
                    sim = float(np.dot(x, y) / denom)
                else:
                    raise ValueError("Unsupported similarity_metric")
                sim *= n_common / (n_common + self.shrinkage)
                sims[i, j] = sim
                sims[j, i] = sim
        self.user_similarity_matrix = sims

    def _compute_user_similarity_gpu(self):
        R_np = self.user_item_matrix.values.astype(np.float32)  # (U,I)
        R = torch.from_numpy(R_np).to(self.device)
        mask = ~torch.isnan(R)  # bool
        R_filled = torch.nan_to_num(R, nan=0.0)

        if self.similarity_metric in ("pearson", "cosine") and self.normalize:
            # 按已评分中心化
            counts = mask.sum(dim=1).clamp(min=1)
            sums = (R_filled).sum(dim=1)
            means = sums / counts
            centered = (R_filled - means.unsqueeze(1)) * mask
        else:
            centered = R_filled * mask  # 只保留评分

        if self.similarity_metric == "pearson":
            M = centered
        elif self.similarity_metric == "cosine":
            M = centered  # 已按 normalize 决定是否中心化
        else:
            raise ValueError("当前实现仅支持 pearson / cosine")

        # 范数
        norms = torch.sqrt((M * M).sum(dim=1)).clamp(min=1e-8)
        # 点积
        sim = (M @ M.T) / (norms.unsqueeze(1) * norms.unsqueeze(0))
        sim = torch.clamp(sim, -1.0, 1.0)

        # 共同评分次数
        co_counts = (mask @ mask.T).float()

        # 过滤最小共评
        sim = torch.where(co_counts >= self.min_overlap, sim, torch.zeros_like(sim))

        # 收缩
        sim = sim * (co_counts / (co_counts + self.shrinkage))

        # 对角线
        sim.fill_diagonal_(1.0)

        self.user_similarity_matrix = sim.detach().cpu().numpy().astype(np.float32)

    def _cache_sorted_neighbors(self):
        sims = self.user_similarity_matrix
        self.sorted_neighbors = []
        for i in range(sims.shape[0]):
            order = np.argsort(-np.abs(sims[i]))
            order = order[order != i]
            self.sorted_neighbors.append(order)

    # ================= 基线与单点预测 =================
    def _baseline(self, user_id: str, movie_id: str) -> float:
        mu = self.global_mean
        b_u = self.user_bias.get(user_id, 0.0)
        b_i = self.item_bias.get(movie_id, 0.0)
        return mu + b_u + b_i

    def predict_rating(self, user_id: str, movie_id: str) -> float:
        if self.user_item_matrix is None:
            raise ValueError("模型未训练")
        user_known = user_id in self.user_item_matrix.index
        item_known = movie_id in self.user_item_matrix.columns
        if not user_known and not item_known:
            return float(self.global_mean)
        if not user_known:
            return float(self.global_mean + self.item_bias.get(movie_id, 0.0))
        if not item_known:
            return float(self.global_mean + self.user_bias.get(user_id, 0.0))

        u_idx = self.user_item_matrix.index.get_loc(user_id)
        i_idx = self.user_item_matrix.columns.get_loc(movie_id)
        true_val = self.user_item_matrix.iloc[u_idx, i_idx]
        if not np.isnan(true_val):
            return float(true_val)

        sims = self.user_similarity_matrix[u_idx]
        if self.sorted_neighbors is not None:
            neighbor_order = self.sorted_neighbors[u_idx]
        else:
            neighbor_order = np.argsort(-np.abs(sims))
            neighbor_order = neighbor_order[neighbor_order != u_idx]

        baseline_ui = self._baseline(user_id, movie_id)
        num = 0.0
        den = 0.0
        used = 0
        for v_idx in neighbor_order:
            if used >= self.k:
                break
            r_vi = self.user_item_matrix.iloc[v_idx, i_idx]
            if np.isnan(r_vi):
                continue
            sim = float(sims[v_idx])
            if sim == 0.0:
                continue
            v_id = self.user_item_matrix.index[v_idx]
            dev = r_vi - (self.global_mean + self.user_bias.get(v_id, 0.0) + self.item_bias.get(movie_id, 0.0))
            num += sim * dev
            den += abs(sim)
            used += 1
        pred = baseline_ui + (num / den if den > 0 else 0.0)
        return float(np.clip(pred, 1.0, 5.0))

    # ================= 批量预测（单用户多物品） =================
    def _predict_user_items_batch(self, user_id: str, movie_ids: List[str]) -> List[Tuple[str, float]]:
        u_idx = self.user_item_matrix.index.get_loc(user_id)
        sims = self.user_similarity_matrix[u_idx]
        neighbor_order = self.sorted_neighbors[u_idx]

        # 预取邻居行引用
        k_search = min(len(neighbor_order), max(self.k * 4, 100))  # 扩大搜索以保证每个物品足够已评分邻居
        neighbor_subset = neighbor_order[:k_search]
        neighbor_ids = [self.user_item_matrix.index[n] for n in neighbor_subset]

        # 提取邻居评分子矩阵 (neighbors x items)
        sub_matrix = self.user_item_matrix.loc[neighbor_ids, movie_ids].values  # shape (N, M)
        sims_vec = sims[neighbor_subset]  # shape (N,)

        results = []
        mu = self.global_mean
        # 逐物品（M <= candidate_pool_size，开销可接受）
        for col_idx, movie_id in enumerate(movie_ids):
            col = sub_matrix[:, col_idx]
            mask = ~np.isnan(col)
            if mask.sum() == 0:
                # 没有任何邻居评分 -> 基线
                pred = self._baseline(user_id, movie_id)
                results.append((movie_id, float(np.clip(pred, 1.0, 5.0))))
                continue
            # 已评分的邻居按 |sim| 排序并截取 k
            rated_indices = np.where(mask)[0]
            rated_sims = sims_vec[rated_indices]
            order = np.argsort(-np.abs(rated_sims))
            take = order[: self.k]
            selected = rated_indices[take]
            sel_sims = sims_vec[selected]
            # 偏差 dev
            neighbor_ids_sel = [neighbor_ids[i] for i in selected]
            ratings_sel = col[selected]
            devs = []
            for ridx, r_val, v_id in zip(selected, ratings_sel, neighbor_ids_sel):
                b_v = self.user_bias.get(v_id, 0.0)
                b_i = self.item_bias.get(movie_id, 0.0)
                devs.append(r_val - (mu + b_v + b_i))
            devs = np.array(devs, dtype=np.float32)
            num = float(np.dot(sel_sims, devs))
            den = float(np.sum(np.abs(sel_sims)))
            baseline_ui = self._baseline(user_id, movie_id)
            pred = baseline_ui + (num / den if den > 0 else 0.0)
            results.append((movie_id, float(np.clip(pred, 1.0, 5.0))))
        return results

    # ================= 推荐 =================
    def recommend_movies(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        if self.user_item_matrix is None:
            raise ValueError("模型未训练")
        if user_id not in self.user_item_matrix.index:
            # 冷启动用户：返回最高基线物品
            item_scores = self.global_mean + self.item_bias
            top_items = item_scores.sort_values(ascending=False).head(n_recommendations)
            return [(mid, float(score)) for mid, score in top_items.items()]

        user_row = self.user_item_matrix.loc[user_id]
        unrated = user_row[user_row.isna()].index.tolist()
        if not unrated:
            return []

        # 候选截断（按基线）
        baselines = [(mid, self._baseline(user_id, mid)) for mid in unrated]
        if len(baselines) > self.candidate_pool_size:
            baselines.sort(key=lambda x: x[1], reverse=True)
            candidate_ids = [mid for mid, _ in baselines[: self.candidate_pool_size]]
        else:
            candidate_ids = [mid for mid, _ in baselines]

        preds = self._predict_user_items_batch(user_id, candidate_ids)
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_recommendations]

    # ================= 训练主入口 =================
    def train(self, ratings_df: pd.DataFrame):
        print("开始训练User-based协同过滤模型...")
        t0 = time.time()
        self.build_user_item_matrix(ratings_df)
        self.compute_user_similarity()
        print(f"模型训练完成！（设备: {'GPU' if self.use_gpu else 'CPU'} | 用时 {time.time()-t0:.2f}s）")

    # ================= 评分预测评估 =================
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        print("正在评估评分预测指标 (RMSE/MAE)...")
        metrics = {
            "global_mean": {"se": 0.0, "ae": 0.0, "n": 0},
            "baseline_bias": {"se": 0.0, "ae": 0.0, "n": 0},
            "user_cf": {"se": 0.0, "ae": 0.0, "n": 0},
        }
        mu = self.global_mean
        for _, row in test_df.iterrows():
            u = row["user_id"]
            i = row["movie_id"]
            y = float(row["rating"])
            pred_global = mu
            pred_base = self._baseline(u, i)
            pred_cf = self.predict_rating(u, i)
            for key, pred in [
                ("global_mean", pred_global),
                ("baseline_bias", pred_base),
                ("user_cf", pred_cf),
            ]:
                metrics[key]["se"] += (pred - y) ** 2
                metrics[key]["ae"] += abs(pred - y)
                metrics[key]["n"] += 1
        out = {}
        for k, v in metrics.items():
            n = max(1, v["n"])
            out[k] = {
                "rmse": float(math.sqrt(v["se"] / n)),
                "mae": float(v["ae"] / n)
            }
        return out

    # ================= 推荐评估 =================
    def evaluate_recommendations(
        self,
        test_df: pd.DataFrame,
        top_n: int = 10,
        relevant_threshold: float = 4.0
    ) -> Dict[str, float]:
        print("正在评估推荐列表指标 (Precision / Recall / Coverage / Novelty)...")
        train_counts = self.user_item_matrix.count(axis=0)
        total_interactions = float(train_counts.sum())
        popularity_prob = (train_counts / total_interactions).to_dict()

        user_test_ratings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for _, row in test_df.iterrows():
            user_test_ratings[row["user_id"]].append((row["movie_id"], float(row["rating"])))

        precisions, recalls = [], []
        all_recommended = set()
        novelty_vals = []

        for idx, (user_id, items) in enumerate(user_test_ratings.items()):
            if self.progress_every and idx % self.progress_every == 0:
                print(f"[Rec Eval] 进度: {idx}/{len(user_test_ratings)}")
            if user_id not in self.user_item_matrix.index:
                continue
            relevant = {mid for mid, r in items if r >= relevant_threshold}
            if not relevant:
                continue
            recs = self.recommend_movies(user_id, n_recommendations=top_n)
            rec_items = [mid for mid, _ in recs]
            hit = sum(1 for mid in rec_items if mid in relevant)
            precisions.append(hit / top_n)
            recalls.append(hit / len(relevant))
            for mid in rec_items:
                all_recommended.add(mid)
                p = popularity_prob.get(mid, 1e-12)
                novelty_vals.append(-math.log2(p) if p > 0 else 0.0)

        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls)) if recalls else 0.0
        coverage = len(all_recommended) / len(self.user_item_matrix.columns)
        novelty = float(np.mean(novelty_vals)) if novelty_vals else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "coverage": coverage,
            "novelty": novelty
        }


# ================ 交叉验证 ================
def make_user_folds(df: pd.DataFrame, n_folds: int = 8, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    parts = []
    for user, grp in df.groupby("user_id"):
        idx = np.arange(len(grp))
        rng.shuffle(idx)
        fold_ids = np.array([i % n_folds for i in range(len(grp))])
        assigned = np.empty(len(grp), dtype=int)
        assigned[idx] = fold_ids
        sub = grp.copy()
        sub["fold"] = assigned
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def cross_validate(
    full_df: pd.DataFrame,
    model_params: Dict,
    n_folds: int = 8,
    top_n: int = 10,
    relevant_threshold: float = 4.0
) -> Dict[str, Dict[str, float]]:
    folds_df = make_user_folds(full_df, n_folds=n_folds)
    agg = {
        "baseline_bias_rmse": 0.0,
        "baseline_bias_mae": 0.0,
        "user_cf_rmse": 0.0,
        "user_cf_mae": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "coverage": 0.0,
        "novelty": 0.0
    }
    for fold in range(n_folds):
        print(f"\n========== 开始第 {fold + 1}/{n_folds} 折 ==========")
        train_df = folds_df[folds_df["fold"] != fold][["user_id", "movie_id", "rating"]]
        test_df = folds_df[folds_df["fold"] == fold][["user_id", "movie_id", "rating"]]

        model = UserBasedCF(**model_params)
        model.train(train_df)

        rating_metrics = model.evaluate_model(test_df)
        rec_metrics = model.evaluate_recommendations(
            test_df, top_n=top_n, relevant_threshold=relevant_threshold
        )

        agg["baseline_bias_rmse"] += rating_metrics["baseline_bias"]["rmse"]
        agg["baseline_bias_mae"] += rating_metrics["baseline_bias"]["mae"]
        agg["user_cf_rmse"] += rating_metrics["user_cf"]["rmse"]
        agg["user_cf_mae"] += rating_metrics["user_cf"]["mae"]
        agg["precision"] += rec_metrics["precision"]
        agg["recall"] += rec_metrics["recall"]
        agg["coverage"] += rec_metrics["coverage"]
        agg["novelty"] += rec_metrics["novelty"]

        print(f"折 {fold + 1} 评分: baseline_bias RMSE={rating_metrics['baseline_bias']['rmse']:.4f} "
              f"MAE={rating_metrics['baseline_bias']['mae']:.4f} | user_cf RMSE={rating_metrics['user_cf']['rmse']:.4f} "
              f"MAE={rating_metrics['user_cf']['mae']:.4f}")
        print(f"折 {fold + 1} 推荐: P={rec_metrics['precision']:.4f} R={rec_metrics['recall']:.4f} "
              f"Coverage={rec_metrics['coverage']:.4f} Novelty={rec_metrics['novelty']:.4f}")

    for k in agg.keys():
        agg[k] /= n_folds

    print("\n========== 8 折平均结果 ==========")
    print(f"baseline_bias: RMSE={agg['baseline_bias_rmse']:.4f} MAE={agg['baseline_bias_mae']:.4f}")
    print(f"user_cf:       RMSE={agg['user_cf_rmse']:.4f} MAE={agg['user_cf_mae']:.4f}")
    print(f"Precision={agg['precision']:.4f} Recall={agg['recall']:.4f} "
          f"Coverage={agg['coverage']:.4f} Novelty={agg['novelty']:.4f}")

    return {
        "baseline_bias": {"rmse": agg["baseline_bias_rmse"], "mae": agg["baseline_bias_mae"]},
        "user_cf": {
            "rmse": agg["user_cf_rmse"],
            "mae": agg["user_cf_mae"],
            "precision": agg["precision"],
            "recall": agg["recall"],
            "coverage": agg["coverage"],
            "novelty": agg["novelty"]
        }
    }


def main():
    model_params = dict(
        similarity_metric="pearson",   # 可改 "cosine" / "pearson"
        k=35,
        min_overlap=8,
        shrinkage=75,
        normalize=True,
        bias_reg=15.0,
        use_gpu=True,                # 设 True 自动检测 GPU
        candidate_pool_size=150,
        progress_every=50
    )
    temp = UserBasedCF()
    full_df = temp.load_movielens_data(split="train", dataset="movielens/100k-ratings")

    cross_validate(
        full_df,
        model_params=model_params,
        n_folds=8,
        top_n=10,
        relevant_threshold=4.0
    )


if __name__ == "__main__":
    main()