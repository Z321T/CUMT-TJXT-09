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