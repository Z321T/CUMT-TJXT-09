========== 平均结果 ==========
baseline_bias: RMSE=0.9465 MAE=0.7522
item_cf:       RMSE=0.9493 MAE=0.7382
Precision=0.0293 Recall=0.0286 Coverage=0.1856 Novelty=10.1600

========== 平均结果 ==========
baseline_bias: RMSE=0.9465 MAE=0.7522
item_cf:       RMSE=0.9152 MAE=0.7184
Precision=0.0558 Recall=0.0607 Coverage=0.1194 Novelty=9.3824


### 总体执行流程概览
主流程: `main()` → 载入数据 → `cross_validate()` → 每折: 构建并训练 `ItemBasedCF` → 评分预测评估 + 推荐评估 → 聚合输出平均指标。  
核心算法生命周期: 初始化参数 → `train()` → 构建用户\‑物品矩阵与偏置 → 计算物品相似度(含中心化/重叠过滤/shrinkage/热门惩罚/裁剪) → 缓存邻居顺序 → 评分预测 / Top‑N 推荐 / 指标评估。

---

### 1. `main()`
1. 设定 `model_params`：定义相似度度量、邻域规模 `k`、`min_overlap`、`shrinkage`、热门惩罚力度 `popularity_alpha`、裁剪阈值 `sim_prune_threshold`、候选池大小等。目的: 固化实验配置，便于复现实验与可控调参。
2. 先实例化一个临时 `ItemBasedCF()` 仅用于调用 `load_movielens_data()`：避免重复下载。
3. 调用 `cross_validate()`：执行折叠级别的训练/评估，输出平均表现，降低方差。

---

### 2. `load_movielens_data()`
读取 TensorFlow Datasets 的 MovieLens 100K：逐条提取 `user_id` / `movie_id` / `rating` 构造成 DataFrame。  
目的: 标准化输入格式，后续 pivot 构建稀疏评分矩阵。

---

### 3. 交叉验证 `cross_validate()`
1. `make_user_folds()`：对每个用户内部打乱分配折号，保证用户级拆分而非全局随机 → 防止同一用户部分行为进入训练、部分进入测试(保持真实冷启动/补全场景)。
2. 遍历每折:
   - 训练集 = 其他折；测试集 = 当前折。
   - 实例化模型(解耦每折状态)。
   - 执行 `train()`：构建矩阵 + 相似度。
   - `evaluate_model()`：逐测试样本评分预测 → 累积 SE/AE。
   - `evaluate_recommendations()`：用户级 Top‑N → 统计 P/R/Coverage/Novelty。
3. 聚合平均：简单算术均值(不加权)，输出汇总。  
目的: 降低偶然性，观察参数在不同采样切分下稳定性。

---

### 4. 初始化 `__init__()`
纯参数存储，不做重计算。关键设计:
- `similarity_metric`：`pearson` 或 `cosine`，决定中心化方式。
- `min_overlap`：控制最小共同评分数，过滤不可靠相似对。
- `shrinkage`：平滑相似度 (后面乘以 `overlap / (overlap + shrinkage)`)，防止小样本高相似度。
- `popularity_alpha` + `penalty_mode`：对热门物品相似度降权，缓解“头部霸权”，提升长尾覆盖/新颖性。
- `sim_prune_threshold`：后处理裁剪小绝对值相似度 → 稀疏化矩阵，减少噪声与预测开销。
- `candidate_pool_size`：控制 Top‑N 生成阶段初步候选池规模(基线打分截断)。

---

### 5. `train()`
1. `build_user_item_matrix()`：生成用户×物品稀疏矩阵 + 计算偏置。
2. `compute_item_similarity()`：生成物品相似度矩阵 + 裁剪 + 缓存邻居排序。
目的: 将离线可预计算部分一次性完成，使线上预测仅做向量读取与加权。

---

### 6. 构建矩阵 `build_user_item_matrix()`
- `pivot_table`：行=用户，列=物品，值=评分；聚合函数 = 均值(容忍重复评分)。
- `global_mean`：全局均值 μ，用作多级冷启动基线。
- `_compute_biases()`：正则化用户/物品偏置，降低系统性偏移。
作用: 分离系统性偏差(全局、用户、物品)后让协同过滤聚焦“交互残差”。

---

### 7. 偏置 `_compute_biases()`
流程:
1. 展开评分为一维 Series `stacked`。
2. 物品偏置:  
   `b_i = Σ(r_ui - μ) / (reg + count_i)` → L2 平滑形式的加性缩放 (这里是线性除法正则)；少样本物品被拉向 0。
3. 用户偏置: 先用已得 `b_i` 去除物品偏差 → `user_resid = r_ui - μ - b_i` →  
   `b_u = Σ(user_resid) / (reg + count_u)`。
结果: 三层基线 `μ + b_u + b_i`；为后续残差建模提供稳定中心。

---

### 8. 相似度计算入口 `compute_item_similarity()`
1. 根据 `use_gpu` 选择 CPU/GPU 实现。
2. 结束后执行阈值裁剪：`|sim| < sim_prune_threshold → 0`，保持主干强联系，稀疏化减少预测遍历。
3. `_cache_sorted_neighbors()`：对每个物品按绝对相似度降序缓存索引(排除自身)，预测时快速截取前 `k`。  
目的: 将 O(I log I) 排序离线化，线上仅数组切片。

---

### 9. CPU 路径 `_compute_item_similarity_cpu()`
关键步骤:
1. 统计每列评分数 `item_counts` → 生成热门惩罚向量 `penalty_vec`。
2. 中心化:
   - `pearson`: 对每个物品列内减自己列均值 → 相当于皮尔逊协方差归一化前一步。
   - `cosine + adjusted=True`: 先按用户均值中心化(Adjusted Cosine) → 抵消用户打分尺度差异。
3. 双层 for(物品 i,j):
   - 交集掩码 `mask` → 共同评分数 `overlap` < `min_overlap` 跳过(降噪)。
   - 取中心化向量 `vi,vj` 计算余弦。
   - shrinkage: `sim *= overlap / (overlap + shrinkage)` → 防过拟合。
   - 热门惩罚: `sim *= penalty_vec[i] * penalty_vec[j]` → 抑制高频物品相似度放大。
4. 写对称矩阵。  
设计权衡: 朴素 O(I^2 * U)；在小数据(ML-100K)可接受，逻辑清晰。

---

### 10. GPU 路径 `_compute_item_similarity_gpu()`
向量化流程:
1. 用 `mask` 区分缺失，不填充 NaN 位置的贡献。
2. 按选择的中心化策略构造 `centered`。
3. 直接矩阵乘 `(M.T @ M)` 得到未归一化内积；除以列范数外积 → 余弦/皮尔逊相关。
4. 计算 `co_counts = (mask.T @ mask)` 得到重叠数矩阵。
5. 重叠过滤: `< min_overlap` → 置零。
6. shrinkage: 逐元素乘 `(co_counts / (co_counts + shrinkage))`。
7. 热门惩罚: 外积形式一次性施加。
8. 对角线设 1。  
优势: 利用 GPU 张量广播减少 Python 循环，提升 I 较大时性能。

---

### 11. 热门惩罚 `_popularity_penalty_vector()`
两模式:
- `log`: `1 / (log(1 + count)^α)` → 增长缓慢，温和抑制。
- `power`: `count^{-α}` → 幂律更激进。  
目的: 降低热门项与所有项都高相似的偏差，提升长尾多样性/覆盖率。

---

### 12. 相似度裁剪 + 邻居缓存
- 裁剪后大量相似度=0 → 预测时早停(无需乘加)。
- 缓存排序列表: 预测阶段不再排序，直接遍历排好序的索引直到获取 `k` 个有效邻居。  
优化点: 降低在线延迟，代价是内存 O(I^2) 仍保留(矩阵未稀疏存储)。

---

### 13. 基线 `_baseline()`
返回 `μ + b_u + b_i`；用于:
1. 冷启动回退 (用户或物品未知)。
2. 评分预测中的偏差校正(邻域部分只建模残差)。
3. 推荐候选初筛(以基线得分截断候选池)。  
理由: 分离大规模系统性信号，邻域只解释剩余协同结构，稳定性更高。

---

### 14. 单点评分预测 `predict_rating()`
步骤:
1. 处理三类冷启动: 用户未知 / 物品未知 / 双未知。
2. 若目标物品已评分 → 直接返回原值(避免自解释失真)。
3. 取该物品相似度行 → 通过预缓存邻居序列遍历:
   - 仅考虑用户已评分的邻居物品。
   - 收集前 `k` 个非零相似度。
   - 使用残差形式: `dev = r_uj - baseline(u,j)`。
4. 加权: `pred = baseline(u,i) + Σ(sim * dev) / Σ|sim|`。
5. 截断到评分区间 `[1,5]`。  
优势: 减少偏置传播；使用绝对值归一避免负相似度抵消权重幅度。

---

### 15. 批量预测 `_predict_user_items_batch()`
用于推荐阶段对同一用户多个候选批量评分:
1. 预取用户已评分集合及索引。
2. 对每个候选:
   - 抽取与已评分项的相似度子向量。
   - 截取 top‑k（局部排序）。
   - 基线 + 残差加权。
3. 统一返回列表。  
目的: 避免对同一用户重复索引/转换，提高推荐阶段吞吐。

---

### 16. Top‑N 推荐 `recommend_items()`
1. 若用户未知: 返回基线最高的物品(偏置排序)。
2. 取用户未评分物品集合。
3. 基线预评分 + 按基线排序截断为 `candidate_pool_size` (削减后续精算成本 & 噪声)。
4. 对候选做批量预测 → 排序 → 取前 `top_n`。  
设计思想: 先用廉价信号(偏置)粗召回，再用邻域精排，模拟“两阶段召回+重排”框架。

---

### 17. 评分评估 `evaluate_model()`
逐行:
1. 计算三种预测: 全局均值 / 基线偏置 / ItemCF。
2. 累积平方误差与绝对误差。
3. 结束后计算 RMSE/MAE。  
目的: 分离各层模型增益，验证协同过滤是否真正改善偏置基线。

---

### 18. 推荐评估 `evaluate_recommendations()`
1. 构造测试集中每用户的真实 `(item, rating)` 集。
2. 仅保留含相关物品(≥ 阈值)的用户，避免除零。
3. 调用 `recommend_items()` 取 Top‑N：
   - Precision = 命中数 / N
   - Recall = 命中数 / 相关物品数
4. Coverage = 所有被推荐过的独立物品数 / 训练集中物品总数。
5. Novelty: 对每个被推荐物品计算 `-log2(p)`，其中 `p = item_train_count / 总交互数`；再取均值。  
衡量: 同时关注准确性(P/R)、多样性(Coverage)与流行度偏移(Novelty)。

---

### 19. 折分 `make_user_folds()`
对每个用户内部打散均匀分配折号，保持每个用户在所有折中都有“留出”部分 → 评估场景与真实用户后续交互预测匹配。  
避免全局随机导致某些用户完全缺失(不可评)。

---

### 20. 关键参数相互作用
- `k`: 过小 → 信息不足；过大 → 引入低质量相似度噪声(特别在裁剪后稀疏)。
- `min_overlap`: 提升可靠性；值大 → 稀疏数据下相似度稀缺。
- `shrinkage`: 控制高相似度的“过拟合峰值”；减缓小重叠放大问题。
- `popularity_alpha`: 增大 → 覆盖/新颖性↑，可能精度下降。
- `sim_prune_threshold`: 提升稀疏性与速度；过高 → 召回丢失。
- `candidate_pool_size`: 小 → 精排前召回损失；大 → 计算成本↑。
- `bias_reg`: 平滑稀疏用户/物品偏置，防止残差层吸收系统偏移。

---

### 21. 设计理念总结
- 分层建模(全局→偏置→残差邻域)减少方差。
- 相似度多重稳定化(中心化 + minOverlap + shrinkage + 热门惩罚 + 裁剪)提升泛化。
- 推荐两阶段(偏置候选截断 + 精排)权衡效率与质量。
- 评估覆盖/新颖性 → 明确调参方向(准确性 vs 多样性)。

---

### 22. 可观测输出与调优映射
- 若 RMSE 改善但 P/R 低：检查 `candidate_pool_size` 与 `sim_prune_threshold`。
- 若 Coverage 高但精度低：热门惩罚/裁剪/shrinkage 可能过强。
- 若 Novelty 低：降低 `candidate_pool_size` 依赖或增大 `popularity_alpha`。
- 若 预测偏窄(集中于均值)：可能相似度矩阵过稀(阈值过高或 shrinkage 过大)。

---

### 结论
该实现以“偏置分解 + 稳健邻域相似 + 热门抑制 + 候选截断再精排”为核心结构，平衡了可解释性、可调性与实验 reproducibility；所有关键步骤均针对评分噪声、数据稀疏、流行度偏斜与在线效率问题做了明确处理。