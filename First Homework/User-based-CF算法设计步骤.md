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


### 文件整体处理流程（`User_based_CF.py`）

1. 超参初始化  
   设定相似度类型（pearson/cosine）、邻居数 k、最小共评数、相似度收缩系数、是否中心化、偏置正则系数等。  

2. 数据加载 `load_movielens_data`  
   使用 TFDS 读取 `movielens/100k-ratings` 指定切片，转换为 DataFrame：列包含 user\_id, movie\_id, rating。  

3. 训练入口 `train`  
   调用：
   - `build_user_item_matrix`
   - `_compute_biases`
   - `compute_user_similarity`  

4. 构建评分矩阵 `build_user_item_matrix`  
   - 透视成 用户×物品 的矩阵，未评分保持 NaN。  
   - 计算全局均值 μ、用户均值、物品均值。  
   - 调 `_compute_biases` 生成带正则的用户偏置 b\_u 与物品偏置 b\_i。  

5. 偏置计算 `_compute_biases`  
   - 物品偏置：b\_i = Σ(r\_{ui} - μ)/(reg + n\_i)  
   - 用户偏置：b\_u = Σ(r\_{ui} - μ - b\_i)/(reg + n\_u)  
   - 先算物品后算用户（因为用户残差需要减去对应 b\_i）。  

6. 用户相似度矩阵 `compute_user_similarity`  
   - 双重循环遍历用户对。  
   - 取两用户共同评分的物品集合；若少于 min\_overlap 跳过。  
   - pearson：对共评子向量做中心化后余弦；cosine：可选中心化。  
   - 收缩：sim *= n\_common / (n\_common + shrinkage)。  
   - 得到一个 n\_users × n\_users 稠密矩阵（时间/空间 O(n²)）。  

7. 预测单个评分 `predict_rating`  
   - 冷启动分支：  
     - 都未知 → 返回 μ  
     - 仅物品已知 → μ + b\_i  
     - 仅用户已知 → μ + b\_u  
   - 若训练矩阵已有评分，直接返回原值。  
   - 计算基线：baseline = μ + b\_u + b\_i。  
   - 取用户相似度向量，按 |sim| 降序取前 k 个对该物品有评分的邻居。  
   - 对每个邻居 v：  
     - 取 r\_{vi}  
     - 计算偏差 dev = r\_{vi} - (μ + b\_v + b\_i)  
     - 加权累计：num += sim * dev，den += |sim|  
   - 若 den>0：pred = baseline + num/den，否则 pred = baseline。  
   - 裁剪到 [1,5]。  

8. 推荐 `recommend_movies`  
   - 对用户未评分物品调用 `predict_rating`，按预测值排序取前 N。  
   - 冷启动用户：返回 (μ + b\_i) 最高的物品。  

9. 评估 `evaluate_model`  
   - 采样测试集（或全量）。  
   - 对每条 (u,i,r) 计算：  
     - global\_mean: μ  
     - user\_mean: μ + b\_u  
     - item\_mean: μ + b\_i  
     - baseline\_bias: μ + b\_u + b\_i  
     - user\_cf: 调 `predict_rating`（即完整 CF）  
   - 累计平方误差/绝对误差，输出 RMSE/MAE。  
   - 统计冷启动出现次数。  

### user\_cf 指标的具体处理过程

`user_cf` = 使用完整协同过滤预测逻辑（`predict_rating` 返回值）：
1. 先做冷启动回退（缺用户/物品）。  
2. 得到基线预测 μ + b\_u + b\_i。  
3. 取用户的相似度向量，按绝对值排序，跳过自己。  
4. 逐个考察邻居是否对目标物品有评分，收集最多 k 个。  
5. 使用“邻居评分去偏差”加权：加权项为 sim * (r\_{vi} - (μ + b\_v + b\_i))。  
6. 归一化后加回基线，形成最终预测。  
7. 裁剪到合法评分区间。  
8. 在评估阶段作为误差统计的一类方法，与其他基线比较。  

### 核心思想总结
- 基线（偏置模型）消除全局与系统性偏差。  
- 邻域阶段只建模“相对偏差”，稳健且减轻稀疏噪声。  
- 收缩 + 最小共评防止小样本相似度虚高。  
- user\_cf 相比 baseline\_bias 的改进来源于利用相似用户的剩余偏差结构。  

### 复杂度提示
- 相似度构建 O(n\_users² * avg\_overlap)，在用户数继续增大时需改进（稀疏近邻检索或改成 item-based / 矩阵分解）。

003
整体评价（8 折平均）：

**评分预测**  
- baseline\_bias: RMSE 0.9465 → user\_cf: 0.9211（约 2.7% 降低），MAE 也同步下降。  
- 改进幅度属于典型 User-based CF 在 MovieLens-100K 上的正常水平；继续压到 <0.91 难度加大，需融合模型（如 MF/SVD++）。

**推荐指标**  
- Precision@10=0.0461，Recall@10=0.0602：中等偏低，可接受但仍有提升空间（常见目标：P≈0.05~0.06，R≈0.07+）。  
- Coverage=0.1172：覆盖约 11.7% 物品，说明有一定多样性但未充分探索长尾。  
- Novelty=9.82：偏中等，之前的高新颖性已被调参换取了精确度，当前权衡较均衡。

**结论**  
- 邻域模型工作正常；评分改进稳定，推荐质量已较初始显著提升。  
- 现阶段瓶颈更多在邻域信号有限与单模型表达能力。

**精简改进方向（按优先级）**  
1. 相似度裁剪：|sim| < 0.05 置 0，减少噪声。  
2. 动态邻域：按共同评分数加权 sim *= co\_cnt/(co\_cnt+α)。  
3. 融合 Item-based CF 与一个轻量 MF（线性加权或学习加权）。  
4. 加 NDCG@10 / MAP@10 监控排序质量。  
5. 适度提高候选多样性：重排序时加 (−λ·log(pop))，微调 λ 保持 Precision 不明显下降。  

当前结果“健康、可迭代”；若不引入新模型，后续增益会逐步变小。

