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






                  ========== 8 折平均结果 ==========
user_cf:       RMSE=0.9211 MAE=0.7212
Precision=0.0461 Recall=0.0602 Coverage=0.1172 Novelty=9.8189

item_cf:       RMSE=0.9493 MAE=0.7382
Precision=0.0293 Recall=0.0286 Coverage=0.1856 Novelty=10.1600









基本处理逻辑（当前实现）：

1. 数据加载与索引重映射  
   - 读取评分数据，重映射用户 / 物品 ID 为连续索引。  
   - 划分训练 / 测试集，构建用户\-物品评分矩阵。  

2. 特征预处理  
   - 计算每个用户平均评分，用于均值中心化。  
   - 生成评分存在的掩码矩阵。  
   - 可选应用 IUF（对高频物品降权）。  

3. 用户相似度计算  
   - 对中心化矩阵使用余弦相似度。  
   - 计算共同评分数并做 significance weighting（共同评分越少衰减越大）。  
   - 应用 shrinkage：sim *= n\_co / (n\_co + shrinkage)。  
   - 过滤共同评分数 < min\_common 的相似度为 0。  
   - 可选择是否保留负相似度。  
   - 对每个用户保留 Top\-K 邻居（其余置 0，形成稀疏相似度）。  

4. 批量评分预测  
   - 使用公式：ŷ = μ\_u + (S · (R \- μ)) / Σ|S|（全矩阵 GPU 乘法）。  
   - 裁剪到评分区间 [1,5]。  
   - 可将训练集中已评分位置还原为真实评分。  

5. Top\-N 推荐生成  
   - 对每个用户屏蔽训练集中已看物品。  
   - 在预测矩阵中选取得分最高的 N 个未看物品作为推荐列表。  

6. 评估指标  
   - 评分预测：RMSE、MAE（在测试集中真实条目上）。  
   - Top\-N：Precision@N、Recall@N。  
   - Coverage：被推荐的不同物品数 / 全部物品数。  
   - Popularity：推荐物品的平均流行度（训练集中评分次数均值）。  

7. 主要可调超参  
   - k\_neighbors（控制多样性与精度平衡）  
   - shrinkage（抑制小样本高相似度噪声）  
   - min\_common（提高可靠性 vs. 召回）  
   - use\_iuf（控制热门物品权重）  
   - keep_neg_sims（是否利用负相关）  
   - topn（推荐列表长度）  

8. 性能策略  
   - 全流程核心矩阵运算放在 GPU。  
   - 相似度稀疏化降低后续乘法成本。  
   - 一次性生成完整预测矩阵避免逐条调用。





### 1. 数据加载与索引重映射  
- 从本地（如未存在则下载并解压）读取 MovieLens 100K 原始评分文件 `u.data`，解析为列: user\_id, movie\_id, rating, timestamp，构造成 DataFrame。  
- 读取 `u.item` 提供的 movie\_id 与 title，便于后续输出。  
- 对所有出现过的用户与物品分别排序，建立映射: user\_id → 连续整数 u\_idx，movie\_id → 连续整数 m\_idx，便于矩阵化与张量操作。  

### 2. 构建训练/测试集与评分矩阵  
- 使用随机划分（train\_test\_split, 比例如 0.8/0.2）得到 train 与 test DataFrame。  
- 初始化形状为 (用户数, 物品数) 的稠密 numpy.float32 矩阵 train\_mat，全 0 表示未评分（此实现用 0 而非 NaN，后续用 mask 区分）。  
- 遍历训练集逐行写入 train\_mat[u\_idx, m\_idx] = rating。  
- 此处未显式构造用户\*物品的原始稀疏矩阵对象，而是直接用稠密矩阵 + 掩码；优点是后续 GPU 张量乘法简单，缺点是空间利用率低（但 100K 数据规模可接受）。  

### 3. 全局统计与用户均值（基线部分）  
- global\_mean = 训练集中所有非零评分的平均值。  
- user\_means[u] = 该用户所有已评分的算术平均；若用户没有评分则回退到 global\_mean。  
- 当前优化版本未单独估计 item bias（物品偏置）和 user bias 的正则化形式，只使用用户均值做中心化；与更完整的偏置模型相比：  
  - 简化：少一次物品聚合与正则。  
  - 影响：无法单独校正“整体被打高/低分的电影”偏差，可能稍降低评分预测精度，但减少计算与参数。  

### 4. 掩码与中心化准备  
- mask = (train\_mat > 0) → 1 表示有评分，0 表示无。  
- 将 numpy 矩阵拷贝到 GPU：R (U×I)。  
- user\_means 通过广播扩展为 (U×1)，centered = (R - user\_means) * mask：未评分位置保持为 0（避免把均值本身引入噪声）。  

### 5. IUF（可选逆用户频率缩放）  
- 若启用 use\_iuf:  
  - item\_pop[i] = mask[:, i] 求和 = 给该物品评分的用户数。  
  - iuf[i] = log(1 + U / (1 + item\_pop[i]))。热门物品用户覆盖度高 → 权重更低。  
  - centered 每一列乘以对应 iuf 以削弱流行物品对相似度的主导。  

### 6. 用户相似度计算（GPU 全量）  
- norms[u] = L2 范数 ||centered\_u||；若为 0（全空）替换为 1，避免除零。  
- 原始相似度 sim = centered @ centered.T （点积）再除以外积 denom = norms ⊗ norms，得到余弦相似度（等价于均值中心化后 Pearson）。  
- common = mask @ mask.T：两用户共同评分的物品个数矩阵（整数）。  
- shrinkage + 显著性权重： sim *= common / (common + shrinkage)。共同评分少 → 收缩趋近 0，降低偶然高相似度。  
- 共同评分过滤：common < min\_common 的位置置 0。  
- 负相似度：若 keep\_neg\_sims=False，则所有负值裁剪为 0。  
- 自相似对角线设 0（避免在 Top-K 中选到自己）。  

### 7. Top-K 邻域稀疏化  
- 对每个用户行做 topk(sim, k=k\_neighbors)。  
- 用 scatter 构造只保留 K 个最大（可含负，视 keep\_neg\_sims）相似度的稀疏行；其余置 0。  
- 得到 sim\_matrix\_topk（仍为稠密张量，但高比例 0，便于后续单次矩阵乘法）。  
- 作用：  
  - 去噪：忽略长尾极低相似度。  
  - 加速：后续乘法有效非零减少。  

### 8. 全量评分预测矩阵生成  
- centered 重用： (R - user\_means) * mask。  
- numerator = S @ centered （S 为 Top-K 相似度矩阵, U×U；结果 U×I）。  
- denominator = sum\_axis1(|S|) 形状 (U×1)，加 clamp 最小 1e-6。  
- pred = user\_means + numerator / denominator （用户均值加加权偏差）。  
- 截断 pred 到 [1,5]。  
- 为便于 Top-N 推荐时不改变用户已观列表，可将训练集中已评分位置覆盖为真实评分（filled[mask==1] = R[mask==1]）。  
- 保存到 CPU：pred\_matrix (numpy)。一次生成避免逐条调用函数。  

### 9. Top-N 推荐生成流程  
- 对每个用户 u：  
  - 获取预测行 pred\_matrix[u] 的拷贝。  
  - 将训练集中已评分位置置为极小值 (-1e9)，防止被选中。  
  - 使用 argpartition 选出前 N 个索引，再局部排序保证顺序。  
  - 得到推荐物品集合。  
- 该过程无需重新访问相似度或遍历邻居，复杂度 O(U·I) 生成后 O(U·I log N)（argpartition 近似线性）。  

### 10. 评分预测评估（RMSE, MAE）  
- 遍历测试集每条 (u,i,r)：直接取 pred\_matrix[u,i] 作为预测值。  
- 采用 test 集原始条目（无过滤），衡量模型对未见条目的评分拟合能力。  
- 计算平方误差与绝对误差平均后开根/取平均。  

### 11. 推荐评估（Precision@N, Recall@N, Coverage, Popularity）  
- 对测试集中每个出现过的用户 u：  
  - 相关集合 relevant = { i | (u,i) 在 test 且 rating ≥ 阈值(如 4) }。  
  - 推荐集合 rec\_set = Top-N 推荐结果。  
  - precision\_u = |rec ∩ rel| / N，recall\_u = |rec ∩ rel| / |rel|。  
  - coverage：收集所有 rec\_set 中物品并在结束后除以全物品总数。  
  - popularity：计算训练集中每个物品被评分次数 item\_pop，统计推荐集合平均流行度。  
- 对所有可评估用户取均值。  

### 12. 主要超参与其影响  
- k\_neighbors：邻域规模；增大 → 召回 / 覆盖上升，精度可能先升后降，计算稍增。  
- shrinkage：收缩强度；增大 → 降低小共同评分数的夸大相似度，稳健但可能削弱差异性。  
- min\_common：最小共同评分；提高 → 可信度上升，稀疏性加剧，召回下降。  
- use\_iuf：控制热门物品权重；开启可提升多样性与长尾曝光，可能轻微损失对主流物品的精度。  
- keep\_neg\_sims：是否保留负相似度；保留可利用“反偏好”信息（在某些场景帮助区分），但可能引入噪声。  
- topn：推荐列表长度；直接影响 Precision/Recall 的权衡。  

### 13. 评分预测核心公式  
对于用户 u、物品 i：  
ŷ\_{u,i} = μ\_u + Σ\_{v∈N\_u} sim(u,v) * (r\_{v,i} - μ\_v) / Σ\_{v∈N\_u} |sim(u,v)|  
- μ\_u, μ\_v 为各自用户均值。  
- N\_u 为经过 Top-K、最小共同评分、shrinkage 后剩余的邻居集合；若分母为 0，退化为 μ\_u。  

### 14. 性能优化要点  
- 所有重计算（相似度、全量预测）放在 GPU，以矩阵乘法替代嵌套循环。  
- Top-K 稀疏化减少后续乘法的有效参与元素。  
- 一次性构建预测矩阵 → 推荐与评估直接索引读取。  
- argpartition 减少排序成本（不对全列排序）。  

### 15. 与“偏置+邻域”完整版本的差异说明  
- 本实现未显式建模 item bias（物品偏置）与正则化偏置求解，而是用用户均值 + 偏差聚合；若需要更精细的基线可在步骤 3 中补充：  
  - b\_i = Σ(r\_{u,i} - μ)/ (λ + count\_i)  
  - b\_u = Σ(r\_{u,i} - μ - b\_i)/ (λ + count\_u)  
  - 公式调整：ŷ = μ + b\_u + b\_i + 邻域偏差项。  

### 16. 长尾 / 覆盖率提升（关联步骤）  
- 提高 k\_neighbors（更多不同邻居物品进入候选）。  
- 降低 min\_common 或减小 shrinkage 以放宽边缘用户连接。  
- 保持 use\_iuf=True 抑制热门物品重复出现。  
- 可额外在推荐阶段加入多样化重排（未在当前代码实现）。  

以上为当前 `user-based-cf-new.py` 实现各步骤的细化处理逻辑。