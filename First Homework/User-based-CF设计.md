设计思想
User-based协同过滤算法的核心思想是：


用户相似度计算：找到与目标用户兴趣相似的其他用户
评分预测：基于相似用户的评分来预测目标用户对未评价物品的偏好
推荐生成：为用户推荐评分最高的物品
算法主要步骤：
加载和预处理MovieLens数据集
构建用户-物品评分矩阵
计算用户间的相似度（使用余弦相似度或皮尔逊相关系数）
预测用户对未评分物品的评分
生成推荐列表

001
代码实现包含了完整的User-based协同过滤算法，具有以下特点：


数据处理：自动加载和预处理MovieLens数据集
相似度计算：支持余弦相似度和皮尔逊相关系数
评分预测：基于相似用户的加权平均评分进行预测
推荐生成：为用户生成个性化的电影推荐列表
模型评估：提供RMSE和MAE评估指标
算法的核心优势是能够发现用户间的相似性并利用这些信息进行准确的推荐。


库安装：
pip install -U \
  tensorflow==2.15.0 \
  tensorflow-datasets \
  pandas numpy scipy scikit-learn \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn


002
下面给出一次性优化的要点与完整改造版本，并在代码中集成了改进项与评测流程。  改进点摘要：  
使用未填充缺失（NaN）的用户-物品矩阵，避免用0充当“未评分”带来的偏置。
支持 pearson 与 cosine 两种相似度；可选均值中心化与最小共评数量 min_overlap；加入相似度收缩系数 shrinkage 降低小样本噪声。
引入带正则的基线偏置模型：r̂(u,i) = μ + b_u + b_i，并在基线之上进行用户邻域加权偏差预测。
统一冷启动策略：未知用户/物品时回退到基线（用户均值、物品均值、全局均值）。
评测对比：全局均值、用户均值、物品均值、基线偏置、改进的UserCF（多指标：RMSE/MAE）。
代码说明：  
build_user_item_matrix 保留 NaN；计算全局/用户/物品均值和正则偏置。
compute_user_similarity 在共评集合上计算相似度，支持均值中心化、最小共评与收缩。
predict_rating 使用基线偏置并对邻居的“去偏差评分”做加权融合；包含全面的冷启动回退。
evaluate_model 输出多基线与CF的 RMSE/MAE 对比。