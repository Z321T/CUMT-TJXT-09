import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import math
from collections import defaultdict
import time

warnings.filterwarnings("ignore")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class ItemBasedCF:
    """
    基于物品的协同过滤（支持 GPU）
    - 相似度: pearson / cosine
    - 可选 adjusted cosine (对 cosine 时按用户均值中心化)
    - Shrinkage 稳定相似度
    - 热门物品惩罚: 防止热门项与多数项高度相似
    - 评分预测 + Top-N 推荐
    - 评价指标: RMSE / MAE / Precision / Recall / Coverage / Novelty
    """

    def __init__(
        self,
        similarity_metric: str = "pearson",
        k: int = 40,
        min_overlap: int = 5,
        shrinkage: float = 50.0,
        normalize: bool = True,
        adjusted: bool = True,              # 仅对 cosine 生效 (adjusted cosine)
        bias_reg: float = 15.0,
        use_gpu: bool = True,
        candidate_pool_size: int = 200,
        progress_every: int = 50,
        # 热门物品惩罚
        popularity_alpha: float = 0.6,      # 惩罚强度 (0 关闭)
        penalty_mode: str = "log",          # 'log' 或 'power'
        # 相似度裁剪
        sim_prune_threshold: float = 0.02
    ):
        self.similarity_metric = similarity_metric
        self.k = k
        self.min_overlap = min_overlap
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.adjusted = adjusted
        self.bias_reg = bias_reg

        self.has_torch = HAS_TORCH
        self.use_gpu = bool(use_gpu and self.has_torch and torch.cuda.is_available())
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None
        self.user_bias: Optional[pd.Series] = None
        self.item_bias: Optional[pd.Series] = None

        self.sorted_item_neighbors: Optional[List[np.ndarray]] = None

        self.candidate_pool_size = candidate_pool_size
        self.progress_every = progress_every

        self.popularity_alpha = popularity_alpha
        self.penalty_mode = penalty_mode
        self.sim_prune_threshold = sim_prune_threshold

    # =============== 数据加载与矩阵构建 ===============
    def load_movielens_data(self, split: str = "train", dataset: str = "movielens/100k-ratings") -> pd.DataFrame:
        print("加载 MovieLens 数据集...")
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
        print(f"记录数: {len(df)} | 用户数: {df['user_id'].nunique()} | 物品数: {df['movie_id'].nunique()}")
        return df

    def build_user_item_matrix(self, ratings_df: pd.DataFrame):
        print("构建用户-物品矩阵...")
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", aggfunc="mean"
        )
        self.global_mean = float(np.nanmean(self.user_item_matrix.values))
        self._compute_biases()
        print(f"矩阵形状: {self.user_item_matrix.shape}")

    def _compute_biases(self):
        print("计算基线偏置 (用户/物品)...")
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

    # =============== 相似度计算 ===============
    def compute_item_similarity(self):
        print(f"计算物品相似度 (metric={self.similarity_metric}, GPU={self.use_gpu})...")
        if self.use_gpu:
            self._compute_item_similarity_gpu()
        else:
            self._compute_item_similarity_cpu()

        # 裁剪
        if self.sim_prune_threshold > 0:
            sims = self.item_similarity_matrix
            mask = (np.abs(sims) < self.sim_prune_threshold)
            np.fill_diagonal(mask, False)
            sims[mask] = 0.0
            self.item_similarity_matrix = sims
            print(f"相似度裁剪: |sim|<{self.sim_prune_threshold} -> 0")

        print("相似度完成")
        self._cache_sorted_neighbors()

    def _popularity_penalty_vector(self, counts: np.ndarray) -> np.ndarray:
        if self.popularity_alpha <= 0:
            return np.ones_like(counts, dtype=np.float32)
        if self.penalty_mode == "log":
            # 1 / (log(1+count)^alpha)
            return 1.0 / (np.log1p(counts) ** self.popularity_alpha)
        elif self.penalty_mode == "power":
            # count^{-alpha}
            return counts ** (-self.popularity_alpha)
        else:
            raise ValueError("penalty_mode 必须为 'log' 或 'power'")

    def _compute_item_similarity_cpu(self):
        R = self.user_item_matrix.values  # shape (U, I)
        n_users, n_items = R.shape
        sims = np.eye(n_items, dtype=np.float32)

        # 物品评分数
        item_counts = np.sum(~np.isnan(R), axis=0)
        penalty_vec = self._popularity_penalty_vector(item_counts)

        # 预处理：中心化
        if self.similarity_metric == "pearson":
            centered = R.copy()
            for j in range(n_items):
                col = centered[:, j]
                mask = ~np.isnan(col)
                if mask.sum() > 0:
                    mean_j = col[mask].mean()
                    col[mask] = col[mask] - mean_j
                centered[:, j] = col
        elif self.similarity_metric == "cosine":
            if self.adjusted:
                # adjusted cosine: 按用户均值中心化
                user_means = np.nanmean(R, axis=1)
                centered = (R - user_means[:, None])
            else:
                centered = R.copy()
            # 不对 NaN 位置填值, 后面用 mask
        else:
            raise ValueError("相似度仅支持 'pearson' 或 'cosine'")

        for i in range(n_items):
            col_i = centered[:, i]
            mask_i = ~np.isnan(col_i)
            for j in range(i + 1, n_items):
                col_j = centered[:, j]
                mask_j = ~np.isnan(col_j)
                mask = mask_i & mask_j
                overlap = int(mask.sum())
                if overlap < self.min_overlap:
                    continue
                vi = col_i[mask]
                vj = col_j[mask]
                denom = np.linalg.norm(vi) * np.linalg.norm(vj)
                if denom == 0:
                    continue
                sim = float(np.dot(vi, vj) / denom)
                # shrinkage
                sim *= overlap / (overlap + self.shrinkage)
                # 热门惩罚
                sim *= penalty_vec[i] * penalty_vec[j]
                sims[i, j] = sim
                sims[j, i] = sim

        self.item_similarity_matrix = sims

    def _compute_item_similarity_gpu(self):
        R_np = self.user_item_matrix.values.astype(np.float32)  # (U,I)
        R = torch.from_numpy(R_np).to(self.device)
        mask = ~torch.isnan(R)
        R_filled = torch.nan_to_num(R, nan=0.0)

        item_counts = mask.sum(dim=0).cpu().numpy().astype(np.float32)
        penalty_vec = self._popularity_penalty_vector(item_counts)
        penalty_t = torch.from_numpy(penalty_vec).to(self.device)

        if self.similarity_metric == "pearson":
            counts = mask.sum(dim=0).clamp(min=1)
            sums = R_filled.sum(dim=0)
            means = sums / counts
            centered = (R_filled - means.unsqueeze(0)) * mask
        elif self.similarity_metric == "cosine":
            if self.adjusted:
                user_counts = mask.sum(dim=1).clamp(min=1)
                user_means = (R_filled.sum(dim=1) / user_counts).unsqueeze(1)
                centered = (R_filled - user_means) * mask
            else:
                centered = R_filled * mask
        else:
            raise ValueError("相似度仅支持 pearson / cosine")

        M = centered  # (U,I)
        # 计算列范数
        norms = torch.sqrt((M * M).sum(dim=0)).clamp(min=1e-8)
        sim = (M.t() @ M) / (norms.unsqueeze(1) * norms.unsqueeze(0))
        sim = torch.clamp(sim, -1.0, 1.0)

        # 共同评分数
        co_counts = (mask.t() @ mask).float()

        # 过滤 min_overlap
        sim = torch.where(co_counts >= self.min_overlap, sim, torch.zeros_like(sim))

        # shrinkage
        sim = sim * (co_counts / (co_counts + self.shrinkage))

        # 热门惩罚
        penalty_outer = penalty_t.unsqueeze(1) * penalty_t.unsqueeze(0)
        sim = sim * penalty_outer

        # 对角线
        sim.fill_diagonal_(1.0)

        self.item_similarity_matrix = sim.detach().cpu().numpy().astype(np.float32)

    def _cache_sorted_neighbors(self):
        sims = self.item_similarity_matrix
        self.sorted_item_neighbors = []
        for i in range(sims.shape[0]):
            order = np.argsort(-np.abs(sims[i]))
            order = order[order != i]
            self.sorted_item_neighbors.append(order)

    # =============== 基线与评分预测 ===============
    def _baseline(self, user_id: str, item_id: str) -> float:
        mu = self.global_mean
        b_u = self.user_bias.get(user_id, 0.0)
        b_i = self.item_bias.get(item_id, 0.0)
        return mu + b_u + b_i

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if self.user_item_matrix is None:
            raise ValueError("模型未训练")
        user_known = user_id in self.user_item_matrix.index
        item_known = item_id in self.user_item_matrix.columns
        if not user_known and not item_known:
            return float(self.global_mean)
        if not user_known:
            return float(self.global_mean + self.item_bias.get(item_id, 0.0))
        if not item_known:
            return float(self.global_mean + self.user_bias.get(user_id, 0.0))

        u_row = self.user_item_matrix.loc[user_id]
        if not np.isnan(u_row.get(item_id, np.nan)):
            return float(u_row[item_id])

        i_idx = self.user_item_matrix.columns.get_loc(item_id)
        sims_row = self.item_similarity_matrix[i_idx]

        rated = u_row[~u_row.isna()]
        if rated.empty:
            return float(self._baseline(user_id, item_id))

        neighbor_order = self.sorted_item_neighbors[i_idx]
        num = 0.0
        den = 0.0
        used = 0
        baseline_ui = self._baseline(user_id, item_id)

        for j_idx in neighbor_order:
            if used >= self.k:
                break
            neighbor_item = self.user_item_matrix.columns[j_idx]
            if neighbor_item not in rated.index:
                continue
            sim = float(sims_row[j_idx])
            if sim == 0.0:
                continue
            r_uj = rated[neighbor_item]
            dev = r_uj - self._baseline(user_id, neighbor_item)
            num += sim * dev
            den += abs(sim)
            used += 1

        pred = baseline_ui + (num / den if den > 0 else 0.0)
        return float(np.clip(pred, 1.0, 5.0))

    # =============== 批量预测（单用户多候选） ===============
    def _predict_user_items_batch(self, user_id: str, candidate_items: List[str]) -> List[Tuple[str, float]]:
        u_row = self.user_item_matrix.loc[user_id]
        rated_items = u_row[~u_row.isna()]
        if rated_items.empty:
            # 全冷用户（应已在外部处理），直接基线
            return [(iid, float(np.clip(self._baseline(user_id, iid), 1.0, 5.0))) for iid in candidate_items]

        rated_indices = [self.user_item_matrix.columns.get_loc(i) for i in rated_items.index]
        rated_ratings = rated_items.values
        results = []
        mu = self.global_mean

        for item_id in candidate_items:
            i_idx = self.user_item_matrix.columns.get_loc(item_id)
            sims_vec = self.item_similarity_matrix[i_idx, rated_indices]
            # 选取 top-k
            if len(sims_vec) > self.k:
                order = np.argsort(-np.abs(sims_vec))[: self.k]
                sel_sims = sims_vec[order]
                sel_items = [rated_items.index[idx] for idx in order]
                sel_ratings = rated_ratings[order]
            else:
                sel_sims = sims_vec
                sel_items = list(rated_items.index)
                sel_ratings = rated_ratings

            devs = []
            for r_val, j_item in zip(sel_ratings, sel_items):
                devs.append(r_val - self._baseline(user_id, j_item))
            devs = np.array(devs, dtype=np.float32)
            den = float(np.sum(np.abs(sel_sims)))
            num = float(np.dot(sel_sims, devs))
            baseline_ui = self._baseline(user_id, item_id)
            pred = baseline_ui + (num / den if den > 0 else 0.0)
            results.append((item_id, float(np.clip(pred, 1.0, 5.0))))
        return results

    # =============== 推荐生成 ===============
    def recommend_items(self, user_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if self.user_item_matrix is None:
            raise ValueError("模型未训练")
        if user_id not in self.user_item_matrix.index:
            # 冷启动用户：返回最高基线
            scores = (self.global_mean + self.item_bias).sort_values(ascending=False).head(top_n)
            return [(iid, float(s)) for iid, s in scores.items()]

        u_row = self.user_item_matrix.loc[user_id]
        unrated = u_row[u_row.isna()].index.tolist()
        if not unrated:
            return []

        # 基线候选截断
        baseline_scores = [(iid, self._baseline(user_id, iid)) for iid in unrated]
        if len(baseline_scores) > self.candidate_pool_size:
            baseline_scores.sort(key=lambda x: x[1], reverse=True)
            candidate_items = [iid for iid, _ in baseline_scores[: self.candidate_pool_size]]
        else:
            candidate_items = [iid for iid, _ in baseline_scores]

        preds = self._predict_user_items_batch(user_id, candidate_items)
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:top_n]

    # =============== 训练入口 ===============
    def train(self, ratings_df: pd.DataFrame):
        print("开始训练 Item-based CF ...")
        t0 = time.time()
        self.build_user_item_matrix(ratings_df)
        self.compute_item_similarity()
        print(f"训练完成 (设备: {'GPU' if self.use_gpu else 'CPU'} | 用时 {time.time() - t0:.2f}s)")

    # =============== 评分预测评估 ===============
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        print("评估评分预测 (RMSE/MAE)...")
        metrics = {
            "global_mean": {"se": 0.0, "ae": 0.0, "n": 0},
            "baseline_bias": {"se": 0.0, "ae": 0.0, "n": 0},
            "item_cf": {"se": 0.0, "ae": 0.0, "n": 0},
        }
        mu = self.global_mean
        for _, row in test_df.iterrows():
            u = row["user_id"]
            i = row["movie_id"]
            y = float(row["rating"])
            pred_g = mu
            pred_b = self._baseline(u, i)
            pred_cf = self.predict_rating(u, i)
            for key, pred in [("global_mean", pred_g), ("baseline_bias", pred_b), ("item_cf", pred_cf)]:
                metrics[key]["se"] += (pred - y) ** 2
                metrics[key]["ae"] += abs(pred - y)
                metrics[key]["n"] += 1
        out = {}
        for k, v in metrics.items():
            n = max(1, v["n"])
            out[k] = {"rmse": float(math.sqrt(v["se"] / n)), "mae": float(v["ae"] / n)}
        return out

    # =============== 推荐指标评估 ===============
    def evaluate_recommendations(
        self,
        test_df: pd.DataFrame,
        top_n: int = 10,
        relevant_threshold: float = 4.0
    ) -> Dict[str, float]:
        print("评估推荐指标 (Precision / Recall / Coverage / Novelty)...")
        train_counts = self.user_item_matrix.count(axis=0)
        total_interactions = float(train_counts.sum())
        popularity_prob = (train_counts / total_interactions).to_dict()

        user_test: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for _, row in test_df.iterrows():
            user_test[row["user_id"]].append((row["movie_id"], float(row["rating"])))

        precisions, recalls = [], []
        all_recommended = set()
        novelty_vals = []

        for idx, (user_id, items) in enumerate(user_test.items()):
            if self.progress_every and idx % self.progress_every == 0:
                print(f"[Rec Eval] {idx}/{len(user_test)}")
            if user_id not in self.user_item_matrix.index:
                continue
            relevant = {iid for iid, r in items if r >= relevant_threshold}
            if not relevant:
                continue
            recs = self.recommend_items(user_id, top_n=top_n)
            rec_items = [iid for iid, _ in recs]
            hit = sum(1 for iid in rec_items if iid in relevant)
            precisions.append(hit / top_n)
            recalls.append(hit / len(relevant))
            for iid in rec_items:
                all_recommended.add(iid)
                p = popularity_prob.get(iid, 1e-12)
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


# =============== 交叉验证 ===============
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
        "item_cf_rmse": 0.0,
        "item_cf_mae": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "coverage": 0.0,
        "novelty": 0.0
    }
    for fold in range(n_folds):
        print(f"\n========== 折 {fold + 1}/{n_folds} ==========")
        train_df = folds_df[folds_df["fold"] != fold][["user_id", "movie_id", "rating"]]
        test_df = folds_df[folds_df["fold"] == fold][["user_id", "movie_id", "rating"]]

        model = ItemBasedCF(**model_params)
        model.train(train_df)

        rating_metrics = model.evaluate_model(test_df)
        rec_metrics = model.evaluate_recommendations(
            test_df, top_n=top_n, relevant_threshold=relevant_threshold
        )

        agg["baseline_bias_rmse"] += rating_metrics["baseline_bias"]["rmse"]
        agg["baseline_bias_mae"] += rating_metrics["baseline_bias"]["mae"]
        agg["item_cf_rmse"] += rating_metrics["item_cf"]["rmse"]
        agg["item_cf_mae"] += rating_metrics["item_cf"]["mae"]
        agg["precision"] += rec_metrics["precision"]
        agg["recall"] += rec_metrics["recall"]
        agg["coverage"] += rec_metrics["coverage"]
        agg["novelty"] += rec_metrics["novelty"]

        print(f"评分: baseline_bias RMSE={rating_metrics['baseline_bias']['rmse']:.4f} "
              f"MAE={rating_metrics['baseline_bias']['mae']:.4f} | item_cf RMSE={rating_metrics['item_cf']['rmse']:.4f} "
              f"MAE={rating_metrics['item_cf']['mae']:.4f}")
        print(f"推荐: P={rec_metrics['precision']:.4f} R={rec_metrics['recall']:.4f} "
              f"Coverage={rec_metrics['coverage']:.4f} Novelty={rec_metrics['novelty']:.4f}")

    for k in agg.keys():
        agg[k] /= n_folds

    print("\n========== 平均结果 ==========")
    print(f"baseline_bias: RMSE={agg['baseline_bias_rmse']:.4f} MAE={agg['baseline_bias_mae']:.4f}")
    print(f"item_cf:       RMSE={agg['item_cf_rmse']:.4f} MAE={agg['item_cf_mae']:.4f}")
    print(f"Precision={agg['precision']:.4f} Recall={agg['recall']:.4f} "
          f"Coverage={agg['coverage']:.4f} Novelty={agg['novelty']:.4f}")

    return {
        "baseline_bias": {"rmse": agg["baseline_bias_rmse"], "mae": agg["baseline_bias_mae"]},
        "item_cf": {
            "rmse": agg["item_cf_rmse"],
            "mae": agg["item_cf_mae"],
            "precision": agg["precision"],
            "recall": agg["recall"],
            "coverage": agg["coverage"],
            "novelty": agg["novelty"]
        }
    }


def main():
    model_params = dict(
        similarity_metric="pearson",    # 'pearson' 或 'cosine'
        k=40,
        min_overlap=6,
        shrinkage=60,
        normalize=True,
        adjusted=True,                  # 仅 cosine 时有效
        bias_reg=15.0,
        use_gpu=True,
        candidate_pool_size=180,
        progress_every=50,
        popularity_alpha=0.4,
        penalty_mode="log",             # 'log' / 'power'
        sim_prune_threshold=0.01
    )

    temp = ItemBasedCF()
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