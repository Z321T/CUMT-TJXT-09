========== 8 折平均结果 ==========
baseline_bias: RMSE=0.9465 MAE=0.7522
user_cf:       RMSE=0.9211 MAE=0.7212
Precision=0.0461 Recall=0.0602 Coverage=0.1172 Novelty=9.8189


下面按执行顺序梳理当前 `UserBasedCF` 算法整体流程（训练、预测、推荐、评估、交叉验证）：

### 1. 数据加载 (`load_movielens_data`)
1. 通过 `tfds.load` 读取 `movielens/100k-ratings`。
2. 提取字段：`user_id`、`movie_id`、`user_rating`，构造成 `DataFrame`。
3. 返回整份评分记录表（长表，每行一条评分）。

### 2. 构建用户-物品矩阵 (`build_user_item_matrix`)
1. 以 `user_id` 为行、`movie_id` 为列，`rating` 聚合为平均值（稀疏矩阵，未评分为 NaN）。
2. 计算全局均值 `global_mean`。
3. 进入偏置估计 `_compute_biases`。

### 3. 基线偏置估计 (`_compute_biases`)
1. 将稀疏矩阵 stack 成 (user,movie,rating) 的长索引序列。
2. 物品偏置 `b_i`：
   - 对每个物品：`sum(r - μ)`。
   - 正则：`b_i = resid_sum / (bias_reg + count)`。
3. 用户偏置 `b_u`：
   - 给每条评分减去 `μ + b_i` 后，按用户聚合。
   - `b_u = resid_sum / (bias_reg + count)`。
4. 得到：预测的基线部分 `μ + b_u + b_i`。

### 4. 用户相似度计算 (`compute_user_similarity`)
1. 判断是否使用 GPU。
2. GPU 分支：
   - 用掩码区分已评分位置。
   - 按需中心化（均值减除，忽略 NaN）。
   - 计算向量范数与点积得到 Cosine / Pearson（在该实现里两者合并为中心化后余弦）。
   - 共同评分数矩阵 `co_counts = mask @ mask.T`。
   - 过滤：共同评分数 < `min_overlap` 置 0。
   - Shrinkage：`sim *= co_counts / (co_counts + shrinkage)`。
   - 对角线设为 1。
3. CPU 分支（双层循环）：
   - 对每对用户找共同评分列。
   - 计算 Pearson 或（中心化后的）Cosine。
   - 应用 shrinkage。
4. 缓存最终相似度矩阵 `user_similarity_matrix`。
5. 预存按 |sim| 排序的邻居索引列表 `_cache_sorted_neighbors`。

### 5. 邻居排序缓存 (`_cache_sorted_neighbors`)
1. 对每个用户行：
   - 按相似度绝对值降序排序（排除自身）。
   - 存入 `sorted_neighbors`，加速后续查询与 Top-K 选取。

### 6. 单点评分预测 (`predict_rating`)
1. 处理冷启动（未知用户 / 未知物品）：
   - 双未知：`global_mean`。
   - 只未知用户：`μ + b_i`。
   - 只未知物品：`μ + b_u`。
2. 已有真实评分：直接返回真实值（不覆盖）。
3. 需要预测：
   - 取该用户的相似度向量与已缓存的邻居序列。
   - 遍历邻居（最多 K）：
     - 只用对该物品有评分的邻居。
     - 计算评分偏差：`r_vi - (μ + b_v + b_i)`。
     - 累加：`num += sim * dev`，`den += |sim|`。
   - 预测：`baseline_ui + num / den`（若 `den` 为 0 则退化为基线）。
   - 截断到 [1,5] 区间。

### 7. 批量物品预测加速 (`_predict_user_items_batch`)
1. 针对单个用户的一批候选物品：
   - 先放大候选邻居池：`k_search = max(k*4, 100)`，保证足够覆盖。
   - 提取邻居子矩阵 (neighbors × candidate_items)。
2. 对每个物品列：
   - 过滤出有评分的邻居行。
   - 按 |sim| 排序截取 K。
   - 计算偏差并加权聚合，形成最终预测。
3. 返回 (movie_id, predicted_rating) 列表。

### 8. Top-N 推荐 (`recommend_movies`)
1. 找出该用户未评分的物品集合。
2. 先用基线打分进行候选截断（限制为 `candidate_pool_size`）。
3. 对候选集合调用批量预测。
4. 按预测得分排序取前 N。

### 9. 评分预测评估 (`evaluate_model`)
1. 遍历测试集每条 (u,i,r)：
   - 计算三种预测：`global_mean`、`baseline_bias`（μ+b_u+b_i）、`user_cf`（完整邻域）。
2. 累加平方误差与绝对误差。
3. 输出各自 RMSE / MAE。

### 10. 推荐指标评估 (`evaluate_recommendations`)
1. 计算训练集物品流行度分布（用于新颖性：`-log2(pop)`）。
2. 组织测试集中每用户的评价条目。
3. 对每个用户：
   - 过滤出相关物品集合（评分 ≥ 阈值）。
   - 调用推荐生成 Top-N。
   - 计算 Precision、Recall（命中 / 相关）。
   - 统计覆盖的物品集合与新颖性累积。
4. 汇总平均：Precision / Recall / Coverage（被推荐的物品数占全部物品数）/ Novelty。

### 11. 交叉验证 (`cross_validate`)
1. 基于用户内打散：每个用户的评分均匀分配到 n 折（防止用户泄漏）。
2. 对每折：
   - 训练：其余折 → 建矩阵、偏置、相似度、邻居缓存。
   - 评估评分指标与推荐指标。
   - 累加折结果。
3. 平均输出：baseline 与 user_cf 的 RMSE/MAE 及推荐四指标。

### 12. 主流程 (`main`)
1. 设定模型超参（相似度、K、min_overlap、shrinkage 等）。
2. 加载完整数据。
3. 调用交叉验证，打印 8 折平均指标。

### 13. 关键超参影响概述
- k：邻域大小，过小噪声大，过大稀释权重。
- min_overlap：最小共同评分，过高会稀疏导致可用邻居不足。
- shrinkage：对小共同评分的相似度收缩，稳定性与区分度权衡。
- bias_reg：控制用户 / 物品偏置的正则强度。
- candidate_pool_size：限制推荐阶段需精排的物品数，加速。
- normalize（影响相似度）：是否中心化评分，Pearson 通常需要。

### 14. 预测公式总结
最终评分估计：
pred(u,i) = μ + b_u + b_i + Σ_{v∈N_k(u,i)} sim(u,v) * (r_{v,i} - (μ + b_v + b_i)) / Σ |sim(u,v)|

### 15. 当前实现的特征
- 单一 User-based 邻域协同过滤（无 MF / 融合）。
- 相似度矩阵一次性全量计算并缓存。
- 推荐阶段使用基线预筛 + 邻域批量偏差修正。
- 指标齐全：RMSE / MAE + Precision / Recall / Coverage / Novelty。
- 交叉验证按“用户内拆分”避免用户冷启动噪声。

以上即当前代码的完整处理流程。