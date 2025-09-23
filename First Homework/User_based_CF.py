import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class UserBasedCF:
    """
    基于用户的协同过滤推荐算法
    """

    def __init__(self, similarity_metric='cosine', k=20):
        """
        初始化协同过滤模型

        Args:
            similarity_metric: 相似度度量方法，支持 'cosine' 或 'pearson'
            k: 选择的最相似用户数量
        """
        self.similarity_metric = similarity_metric
        self.k = k
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_mean_ratings = None

    def load_movielens_data(self, split='train'):
        """
        加载MovieLens数据集

        Args:
            split: 数据集分割，默认为 'train'

        Returns:
            处理后的评分数据
        """
        print("正在加载MovieLens数据集...")

        # 加载MovieLens数据集
        ds = tfds.load('movielens/100k-ratings', split=split, as_supervised=False)

        # 转换为pandas DataFrame
        ratings_data = []
        for example in ds:
            user_id = example['user_id'].numpy()
            movie_id = example['movie_id'].numpy()
            rating = example['user_rating'].numpy()
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })

        df = pd.DataFrame(ratings_data)
        print(f"数据加载完成，共有 {len(df)} 条评分记录")
        print(f"用户数量: {df['user_id'].nunique()}")
        print(f"电影数量: {df['movie_id'].nunique()}")

        return df

    def build_user_item_matrix(self, ratings_df):
        """
        构建用户-物品评分矩阵

        Args:
            ratings_df: 评分数据DataFrame
        """
        print("正在构建用户-物品评分矩阵...")

        # 创建用户-物品评分矩阵
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )

        # 计算每个用户的平均评分
        self.user_mean_ratings = self.user_item_matrix.replace(0, np.nan).mean(axis=1)

        print(f"评分矩阵构建完成，形状: {self.user_item_matrix.shape}")

    def compute_user_similarity(self):
        """
        计算用户间的相似度矩阵
        """
        print(f"正在计算用户相似度矩阵（使用{self.similarity_metric}方法）...")

        if self.similarity_metric == 'cosine':
            # 使用余弦相似度
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)

        elif self.similarity_metric == 'pearson':
            # 使用皮尔逊相关系数
            n_users = len(self.user_item_matrix)
            self.user_similarity_matrix = np.zeros((n_users, n_users))

            for i in range(n_users):
                for j in range(i, n_users):
                    if i == j:
                        self.user_similarity_matrix[i, j] = 1.0
                    else:
                        user_i_ratings = self.user_item_matrix.iloc[i].values
                        user_j_ratings = self.user_item_matrix.iloc[j].values

                        # 只考虑两个用户都评分过的物品
                        common_mask = (user_i_ratings > 0) & (user_j_ratings > 0)

                        if np.sum(common_mask) > 1:
                            correlation, _ = pearsonr(
                                user_i_ratings[common_mask],
                                user_j_ratings[common_mask]
                            )
                            if not np.isnan(correlation):
                                self.user_similarity_matrix[i, j] = correlation
                                self.user_similarity_matrix[j, i] = correlation

        print("用户相似度矩阵计算完成")

    def predict_rating(self, user_id, movie_id):
        """
        预测用户对特定电影的评分

        Args:
            user_id: 用户ID
            movie_id: 电影ID

        Returns:
            预测评分
        """
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        except KeyError:
            return self.user_mean_ratings.mean()  # 返回全局平均评分

        # 如果用户已经评分过这部电影，返回原评分
        if self.user_item_matrix.iloc[user_idx, movie_idx] > 0:
            return self.user_item_matrix.iloc[user_idx, movie_idx]

        # 获取用户的相似度向量
        user_similarities = self.user_similarity_matrix[user_idx]

        # 找到最相似的k个用户（排除自己）
        similar_users_indices = np.argsort(user_similarities)[::-1][1:self.k + 1]

        # 计算加权平均评分
        weighted_sum = 0
        similarity_sum = 0

        for similar_user_idx in similar_users_indices:
            similar_user_rating = self.user_item_matrix.iloc[similar_user_idx, movie_idx]
            similarity = user_similarities[similar_user_idx]

            if similar_user_rating > 0 and similarity > 0:
                weighted_sum += similarity * similar_user_rating
                similarity_sum += similarity

        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
        else:
            # 如果没有相似用户评分过这部电影，返回用户平均评分
            predicted_rating = self.user_mean_ratings.iloc[user_idx]
            if np.isnan(predicted_rating):
                predicted_rating = self.user_mean_ratings.mean()

        return max(1, min(5, predicted_rating))  # 限制评分在1-5之间

    def recommend_movies(self, user_id, n_recommendations=10):
        """
        为用户推荐电影

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量

        Returns:
            推荐的电影ID列表及其预测评分
        """
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            print(f"用户 {user_id} 不存在")
            return []

        # 获取用户未评分的电影
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index

        # 预测所有未评分电影的评分
        movie_predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating(user_id, movie_id)
            movie_predictions.append((movie_id, predicted_rating))

        # 按预测评分排序
        movie_predictions.sort(key=lambda x: x[1], reverse=True)

        return movie_predictions[:n_recommendations]

    def train(self, ratings_df):
        """
        训练协同过滤模型

        Args:
            ratings_df: 评分数据DataFrame
        """
        print("开始训练User-based协同过滤模型...")

        self.build_user_item_matrix(ratings_df)
        self.compute_user_similarity()

        print("模型训练完成！")

    def evaluate_model(self, test_df, sample_size=1000):
        """
        评估模型性能

        Args:
            test_df: 测试数据
            sample_size: 采样大小（避免计算时间过长）
        """
        print("正在评估模型性能...")

        # 随机采样测试数据
        if len(test_df) > sample_size:
            test_sample = test_df.sample(n=sample_size, random_state=42)
        else:
            test_sample = test_df

        predictions = []
        actual_ratings = []

        for _, row in test_sample.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            actual_rating = row['rating']

            predicted_rating = self.predict_rating(user_id, movie_id)

            predictions.append(predicted_rating)
            actual_ratings.append(actual_rating)

        # 计算RMSE
        mse = np.mean([(pred - actual) ** 2 for pred, actual in zip(predictions, actual_ratings)])
        rmse = np.sqrt(mse)

        # 计算MAE
        mae = np.mean([abs(pred - actual) for pred, actual in zip(predictions, actual_ratings)])

        print(f"模型评估结果:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return {'rmse': rmse, 'mae': mae}


# 使用示例
def main():
    """
    主函数：演示如何使用User-based协同过滤算法
    """
    # 初始化模型
    cf_model = UserBasedCF(similarity_metric='cosine', k=20)

    # 加载训练数据
    train_data = cf_model.load_movielens_data('train[:80%]')

    # 训练模型
    cf_model.train(train_data)

    # 加载测试数据进行评估
    test_data = cf_model.load_movielens_data('train[80%:]')
    cf_model.evaluate_model(test_data)

    # 为特定用户生成推荐
    user_id = train_data['user_id'].iloc[0]  # 选择第一个用户作为示例
    recommendations = cf_model.recommend_movies(user_id, n_recommendations=5)

    print(f"\n为用户 {user_id} 推荐的电影:")
    for i, (movie_id, predicted_rating) in enumerate(recommendations, 1):
        print(f"{i}. 电影ID: {movie_id}, 预测评分: {predicted_rating:.2f}")


if __name__ == "__main__":
    main()