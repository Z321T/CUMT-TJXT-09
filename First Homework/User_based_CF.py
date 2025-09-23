import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class UserBasedCF:
    """
    改进版基于用户的协同过滤推荐算法
    - 未填充NaN的评分矩阵
    - 支持 pearson / cosine，相似度收缩与最小共评过滤
    - 基线偏置模型（全局均值、用户偏置、物品偏置）
    - 冷启动友好回退
    """

    def __init__(
        self,
        similarity_metric: str = 'pearson',
        k: int = 40,
        min_overlap: int = 3,
        shrinkage: float = 10.0,
        normalize: bool = True,
        bias_reg: float = 10.0
    ):
        """
        Args:
            similarity_metric: 'pearson' 或 'cosine'
            k: 选取的最近邻用户数
            min_overlap: 相似度计算最小共评物品数
            shrinkage: 相似度收缩强度（n / (n + shrinkage)）
            normalize: 是否在相似度计算时做均值中心化
            bias_reg: 基线偏置的正则强度（越大越保守）
        """
        self.similarity_metric = similarity_metric
        self.k = k
        self.min_overlap = min_overlap
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.bias_reg = bias_reg

        # 数据结构
        self.user_item_matrix: pd.DataFrame = None
        self.user_similarity_matrix: np.ndarray = None

        # 统计量与偏置
        self.global_mean: float = None
        self.user_mean_ratings: pd.Series = None
        self.item_mean_ratings: pd.Series = None
        self.user_bias: pd.Series = None
        self.item_bias: pd.Series = None

    def load_movielens_data(self, split: str = 'train'):
        """
        加载MovieLens数据集
        """
        print("正在加载MovieLens数据集...")
        ds = tfds.load('movielens/100k-ratings', split=split, as_supervised=False)

        ratings_data = []
        for example in ds:
            user_id = example['user_id'].numpy()
            movie_id = example['movie_id'].numpy()
            rating = float(example['user_rating'].numpy())

            # 转换为字符串，避免bytes键在后续DataFrame/索引中的不便
            if isinstance(user_id, (bytes, bytearray)):
                user_id = user_id.decode('utf-8')
            if isinstance(movie_id, (bytes, bytearray)):
                movie_id = movie_id.decode('utf-8')

            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })

        df = pd.DataFrame(ratings_data)
        print(f"数据加载完成，共有 {len(df)} 条评分记录")
        print(f"用户数量: {df['user_id'].nunique()} | 电影数量: {df['movie_id'].nunique()}")
        return df

    def build_user_item_matrix(self, ratings_df: pd.DataFrame):
        """
        构建用户-物品评分矩阵（保留NaN作为未评分）
        并计算全局/用户/物品均值与正则偏置
        """
        print("正在构建用户-物品评分矩阵...")
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            aggfunc='mean'  # 一般每对(user,item)只有一条；若重复取平均
        )
        # 全局均值
        self.global_mean = float(np.nanmean(self.user_item_matrix.values))
        # 用户与物品的简单均值
        self.user_mean_ratings = self.user_item_matrix.mean(axis=1, skipna=True)
        self.item_mean_ratings = self.user_item_matrix.mean(axis=0, skipna=True)

        # 基线偏置（正则）
        self._compute_biases()

        print(f"评分矩阵构建完成，形状: {self.user_item_matrix.shape}")

    def _compute_biases(self):
        """
        计算带正则的用户/物品偏置：
        b_i = sum(r_ui - μ) / (reg + n_i)
        b_u = sum(r_ui - μ - b_i) / (reg + n_u)
        """
        print("正在计算基线偏置（用户/物品）...")
        # 压平成长格式，便于分组计算
        stacked = self.user_item_matrix.stack(dropna=True)  # MultiIndex: (user_id, movie_id) -> rating
        mu = self.global_mean

        # 物品偏置
        item_grp = stacked.groupby(level=1)
        item_count = item_grp.size()
        item_resid_sum = item_grp.apply(lambda s: float((s - mu).sum()))
        b_i = item_resid_sum / (self.bias_reg + item_count)
        # 与列索引对齐（不存在的物品偏置=0）
        self.item_bias = b_i.reindex(self.user_item_matrix.columns).fillna(0.0)

        # 用户偏置
        # 需要用到每条评分对应的物品偏置
        # 构造对齐到 stacked 的 item_bias 序列
        stacked_item_bias = stacked.copy()
        stacked_item_bias[:] = stacked.index.get_level_values(1).map(self.item_bias).values
        user_grp = stacked.groupby(level=0)
        user_count = user_grp.size()
        user_resid_sum = (stacked - mu - stacked_item_bias).groupby(level=0).sum()
        b_u = user_resid_sum / (self.bias_reg + user_count)
        self.user_bias = b_u.reindex(self.user_item_matrix.index).fillna(0.0)

    def compute_user_similarity(self):
        """
        基于共评集合计算用户相似度矩阵，支持：
        - pearson：在共评上做去均值的皮尔逊（≈中心化余弦）
        - cosine：在共评上做余弦，相对可选中心化
        使用收缩：sim *= n/(n + shrinkage)
        """
        print(f"正在计算用户相似度矩阵（{self.similarity_metric}，min_overlap={self.min_overlap}，shrinkage={self.shrinkage}）...")
        users = self.user_item_matrix.index
        n_users = len(users)
        self.user_similarity_matrix = np.eye(n_users, dtype=np.float32)

        values = self.user_item_matrix.values  # shape: [n_users, n_items]
        # 预先计算每个用户的均值（仅对其已评分项）
        user_means = np.nanmean(values, axis=1)

        for i in range(n_users):
            xi = values[i, :]
            for j in range(i + 1, n_users):
                yj = values[j, :]
                mask = ~np.isnan(xi) & ~np.isnan(yj)
                n_common = int(mask.sum())
                if n_common < self.min_overlap:
                    continue

                x = xi[mask]
                y = yj[mask]

                if self.similarity_metric == 'pearson':
                    # 皮尔逊：在共评上中心化
                    x_c = x - np.mean(x)
                    y_c = y - np.mean(y)
                    denom = (np.linalg.norm(x_c) * np.linalg.norm(y_c))
                    sim = 0.0 if denom == 0 else float(np.dot(x_c, y_c) / denom)
                elif self.similarity_metric == 'cosine':
                    # 余弦：可选中心化
                    if self.normalize:
                        x = x - np.mean(x)
                        y = y - np.mean(y)
                    denom = (np.linalg.norm(x) * np.linalg.norm(y))
                    sim = 0.0 if denom == 0 else float(np.dot(x, y) / denom)
                else:
                    raise ValueError("Unsupported similarity_metric, use 'pearson' or 'cosine'.")

                # 收缩
                sim *= n_common / (n_common + self.shrinkage)
                self.user_similarity_matrix[i, j] = sim
                self.user_similarity_matrix[j, i] = sim

        print("用户相似度矩阵计算完成")

    def _baseline(self, user_id, movie_id):
        """
        基线预测：μ + b_u + b_i（带冷启动回退）
        """
        mu = self.global_mean
        b_u = self.user_bias[user_id] if user_id in self.user_bias.index else 0.0
        b_i = self.item_bias[movie_id] if movie_id in self.item_bias.index else 0.0
        return mu + b_u + b_i

    def predict_rating(self, user_id, movie_id):
        """
        使用“基线 + 用户邻域偏差”的方式预测评分，并处理冷启动与回退
        """
        # 冷启动：用户或物品不在训练矩阵中
        user_known = user_id in self.user_item_matrix.index
        item_known = movie_id in self.user_item_matrix.columns

        if not user_known and not item_known:
            return float(self.global_mean)
        if not user_known and item_known:
            # 回退到物品均值（μ + b_i）
            return float(self.global_mean + self.item_bias.get(movie_id, 0.0))
        if user_known and not item_known:
            # 回退到用户均值（μ + b_u）
            return float(self.global_mean + self.user_bias.get(user_id, 0.0))

        # 索引
        u_idx = self.user_item_matrix.index.get_loc(user_id)
        i_idx = self.user_item_matrix.columns.get_loc(movie_id)

        # 若训练中已有评分（理论上不会出现在测试），直接返回原评分
        r_ui = self.user_item_matrix.iloc[u_idx, i_idx]
        if not np.isnan(r_ui):
            return float(r_ui)

        # 基线
        baseline_ui = self._baseline(user_id, movie_id)

        # 邻域加权偏差
        sims = self.user_similarity_matrix[u_idx]
        # 按相似度绝对值排序，排除自身
        neighbor_order = np.argsort(-np.abs(sims))
        neighbor_order = neighbor_order[neighbor_order != u_idx]

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

            # 邻居v对物品i的去偏差评分
            v_id = self.user_item_matrix.index[v_idx]
            dev = r_vi - (self.global_mean + self.user_bias.get(v_id, 0.0) + self.item_bias.get(movie_id, 0.0))
            num += sim * dev
            den += abs(sim)
            used += 1

        if den > 0:
            pred = baseline_ui + num / den
        else:
            pred = baseline_ui

        # 限制到评分区间[1,5]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_movies(self, user_id, n_recommendations=10):
        """
        为用户推荐未评分电影（按预测值排序）
        """
        if user_id not in self.user_item_matrix.index:
            print(f"用户 {user_id} 不存在（冷启动），将返回热门物品")
            # 返回物品均值最高的n个
            item_scores = self.global_mean + self.item_bias
            top_items = item_scores.sort_values(ascending=False).head(n_recommendations)
            return [(mid, float(score)) for mid, score in top_items.items()]

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings.isna()].index

        preds = []
        for mid in unrated_movies:
            preds.append((mid, self.predict_rating(user_id, mid)))
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_recommendations]

    def train(self, ratings_df: pd.DataFrame):
        """
        训练流程：构建矩阵 -> 偏置 -> 相似度
        """
        print("开始训练User-based协同过滤模型...")
        self.build_user_item_matrix(ratings_df)
        self.compute_user_similarity()
        print("模型训练完成！")

    def evaluate_model(self, test_df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        多基线与CF的评估（RMSE/MAE）
        """
        print("正在评估模型性能...")

        # 采样
        test_sample = test_df.sample(n=sample_size, random_state=42) if len(test_df) > sample_size else test_df

        metrics = {
            'global_mean': {'se': 0.0, 'ae': 0.0, 'n': 0},
            'user_mean': {'se': 0.0, 'ae': 0.0, 'n': 0},
            'item_mean': {'se': 0.0, 'ae': 0.0, 'n': 0},
            'baseline_bias': {'se': 0.0, 'ae': 0.0, 'n': 0},
            'user_cf': {'se': 0.0, 'ae': 0.0, 'n': 0},
        }

        cold_user = 0
        cold_item = 0
        cold_both = 0

        for _, row in test_sample.iterrows():
            u = row['user_id']
            i = row['movie_id']
            y = float(row['rating'])

            u_known = u in self.user_item_matrix.index
            i_known = i in self.user_item_matrix.columns
            if not u_known and not i_known:
                cold_both += 1
            elif not u_known:
                cold_user += 1
            elif not i_known:
                cold_item += 1

            # 预测
            mu = self.global_mean
            user_mean = mu + (self.user_bias[u] if u_known else 0.0)
            item_mean = mu + (self.item_bias[i] if i_known else 0.0)
            baseline = self._baseline(u, i)
            cf_pred = self.predict_rating(u, i)

            # 累计误差
            for key, pred in [
                ('global_mean', mu),
                ('user_mean', user_mean),
                ('item_mean', item_mean),
                ('baseline_bias', baseline),
                ('user_cf', cf_pred),
            ]:
                metrics[key]['se'] += (pred - y) ** 2
                metrics[key]['ae'] += abs(pred - y)
                metrics[key]['n'] += 1

        # 计算RMSE/MAE
        results = {}
        print("模型评估结果:")
        for key, acc in metrics.items():
            n = max(1, acc['n'])
            rmse = np.sqrt(acc['se'] / n)
            mae = acc['ae'] / n
            results[key] = {'rmse': float(rmse), 'mae': float(mae)}
            print(f"{key}: RMSE={rmse:.4f} | MAE={mae:.4f}")

        print(f"冷启动统计 -> 仅新用户: {cold_user}, 仅新物品: {cold_item}, 用户与物品皆新: {cold_both}")
        return results


def main():
    """
    主函数：演示改进版User-based CF的训练、评测与推荐
    """
    # 可调整的核心超参
    cf_model = UserBasedCF(
        similarity_metric='pearson',  # 可改为 'cosine'
        k=60,
        min_overlap=5,
        shrinkage=25.0,
        normalize=True,
        bias_reg=15.0
    )

    # 加载训练/测试数据
    train_data = cf_model.load_movielens_data('train[:80%]')
    test_data = cf_model.load_movielens_data('train[80%:]')

    # 训练
    cf_model.train(train_data)

    # 评测（与多种基线对比）
    cf_model.evaluate_model(test_data, sample_size=2000)

    # 推荐示例
    demo_user = train_data['user_id'].iloc[0]
    recs = cf_model.recommend_movies(demo_user, n_recommendations=5)
    print(f"\n为用户 {demo_user} 推荐的电影:")
    for idx, (mid, score) in enumerate(recs, start=1):
        print(f"{idx}. 电影ID: {mid}, 预测评分: {score:.2f}")


if __name__ == "__main__":
    main()