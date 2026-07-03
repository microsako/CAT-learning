# 精读笔记:Quality meets Diversity: A Model-Agnostic Framework for Computerized Adaptive Testing (MAAT)

> Bi, Ma, Huang, Yin, Liu, Chen, Su, Wang — USTC / 安徽大学 / 讯飞研究院,ICDM 2020(arXiv:2101.05986)
> 本文件夹内代码:官方开源实现(https://github.com/bigdata-ustc/MAAT)的核心部分,配 ASSISTments 数据

---

## 1. 论文解决了什么问题?

**一句话:让 CAT(计算机自适应测试)的选题策略不再依赖某个具体的认知诊断模型(CDM),同时兼顾"题目质量(信息量)"和"知识点多样性(覆盖度)"。**

CAT 系统 = 认知诊断模型 M(估计学生知识状态)+ 选题策略 S(每步挑下一题)。论文把 CAT 形式化为:给定新考生 e、题库 Q 和知识点集 K,设计策略 S 逐步选出长度为 N 的测试序列 Q_T,使得:

1. **Quality(质量)**:选出的题信息量最大,能最快降低对学生知识状态估计的不确定性;
2. **Diversity(多样性)**:选出的题尽可能覆盖所有知识点,给学生一个全面的诊断。

且整个策略**不依赖 CDM 的内部机制**(model-agnostic),换 CDM 不用重新设计策略。

## 2. 这个问题是由什么引发的?

- 传统 CAT 策略都是 **model-specific** 的:MFI(最大 Fisher 信息量)必须知道 IRT 的参数形式才能算 Fisher 信息;D-Optimality、MKLI 绑定 MIRT。换一个 CDM 就要重新推导整个选题策略,系统非常不灵活。
- **深度学习 CDM(如 NCDM)出现后问题更尖锐**:神经网络参数没有 IRT 那样可解释的解析形式,Fisher 信息根本没法推,导致这些强大的新模型接不进现有 CAT 系统。论文实验里 NCDM 只能和随机选题比,就是因为没有任何现存策略能配它。
- 另外,现有策略只顾信息量、**忽略知识点覆盖**:比如一直选"函数"相关的高信息题,导致"几何"完全没测过,诊断结果片面。测试通常很短(几十题),覆盖不均衡的问题很实际。

## 3. 目前有哪些其他解决方法?

论文对比的基线(都绑定具体 CDM):

| 策略 | 绑定的 CDM | 思路 |
|---|---|---|
| RAND | 任意 | 随机选题(唯一的 model-agnostic 基线) |
| MFI (Lord, 1980) | IRT | 最大 Fisher 信息,最经典 |
| KLI (Chang & Ying, 1996) | IRT | KL 散度衡量全局信息量 |
| D-Opt | MIRT | MFI 的多维推广(D-最优设计) |
| MKLI | MIRT | KLI 的多维推广 |

它们的共同缺陷:① 必须懂 CDM 内部细节;② 基本不管知识点多样性。

思想来源:**主动学习(Active Learning)**。AL 中"不看模型细节、只看模型输出/数据特征来挑样本"的做法(如 Expected Model Change、representativeness)启发了本文;知识覆盖的 **submodular(次模)优化**则借鉴自文档摘要、推荐系统里的 coverage 建模。论文自称是第一个显式把"知识点覆盖"作为 CAT 优化目标并给出次模优化算法的工作。

## 4. 作者怎么解决的?—— MAAT 框架三模块

每一步 t 的选题分两阶段,外加一个测前预计算模块:

### 4.1 Quality Module:Expected Model Change(EMC)—— 质量

不看 CDM 内部,只看"**如果把这道题的作答加进去,模型参数 θ 会变多少**":

- ΔM(r_ij) = |θ(R_i ∪ {r_ij}) − θ(R_i)|,即加入一条作答记录后参数的变化量;
- 考前不知道学生答对答错,所以按模型预测的答对概率 p = M(e_i, q_j | θ) 取期望:
  **EMC(q_j) = p·ΔM(答对) + (1−p)·ΔM(答错)** (式 1-2)
- θ 变化大 → 这题信息量大。任何梯度可训练的 CDM 都适用(用梯度近似 ΔM 提效),这就是 model-agnostic 的关键。
- 对所有未测题算 EMC,取 **top-K_C** 构成候选集 Q_C(实验里 K_C=10)。

### 4.2 Diversity Module:IWKC 覆盖分 —— 多样性

从候选集 Q_C 里挑一道让"知识覆盖分"边际增益最大的题。朴素覆盖 NKC(覆盖知识点数/总知识点数)有两个毛病:① 所有知识点同等重要;② 0/1 覆盖太硬(某知识点测 1 题和测 9 题没区别)。改进为 **Importance Weighted Knowledge Coverage(IWKC)**(式 5-7):

- IWKC(Q_T) = Σ_k w_k · IncCov(k, Q_T) / Σ_k w_k
- IncCov(k, Q_T) = cnt/(cnt+1),cnt 是 Q_T 中关联知识点 k 的题数 —— 软覆盖,0→0.5→0.67→0.75…,收益递减;
- w_k 是知识点重要性权重(由 Importance Module 给)。

选题规则(式 8-9):q* = argmax_{q∈Q_C} [IWKC(Q_T ∪ {q}) − IWKC(Q_T)],即贪心最大边际增益。

**理论保证**:证明了 IWKC 最大化问题是 **NP-hard**(规约到加权最大覆盖问题),但 IWKC 是**非负单调次模函数**,因此贪心算法有经典的 **(1 − 1/e) 近似比**保证。

### 4.3 Importance Module:知识点重要性 w_k(测前用历史数据预计算)

思想:一个知识点重要 ⇔ 它关联的题更有"代表性"。

1. **Test-Effect Embedding**(式 10-12):仿 Item2Vec,把每条历史记录编码成 2|Q| 维 one-hot(答对/答错拼接),用 Skip-Gram 负采样(SGNS)训练题目向量 —— 历史上被学生们表现相似的题,向量相近;
2. **Test-Effect Density**(式 13-14):题目 q 的密度 = 它与 K_N 个近邻的平均相似度 Sim = exp(−γ||q_i − q_j||),密度越大越有代表性;
3. **w_k**(式 15)= 关联知识点 k 的所有题的密度平均值。

### 4.4 整体流程(Algorithm 1)

测前:用历史数据 H 训练 CDM 初始参数(题目侧参数固定)、训练 Test-Effect embedding、算 w_k。测中每步:算全部未测题 EMC → 取 top-K_C → 按 IWKC 边际增益选 1 题 → 学生作答 → 用新记录更新 θ → 重复 N 步。超参 K_C 调节质量-多样性的平衡:K_C=1 退化成纯 Quality,K_C=|Q_U| 退化成纯 Diversity。

## 5. 在代码中是怎么体现的?

代码是简化版官方实现,只含 IRT 模型 + Random/MAAT 两种策略(models/ 下有 ncd checkpoint 但没有 NCD 模型代码;**Importance Module 没有实现**,详见局限)。

### 目录结构

```
pyat/
  model/irt_model.py        # IRT CDM + EMC 的实现
  strategy/maat_strategy.py # MAAT 选题策略(Quality→Diversity 两阶段)
  utils/data/               # TrainDataset / AdapTestDataset(tested/untested 集合管理)
  driver.py                 # 自适应测试模拟主循环
scripts/train.py            # 测前:训练 IRT
scripts/test.py             # 测中:模拟 CAT,对比 Random vs MAAT
datasets/assistment/        # ASSISTments 2009-2010 数据
```

### 论文-代码对照

| 论文概念 | 代码位置 | 说明 |
|---|---|---|
| 抽象 CDM M(θ) | `irt_model.py:16` `IRT` 类 | θ=学生能力 embedding,α=区分度,β=难度;预测 p = σ(α·θ + β) |
| EMC(式 1-2) | `irt_model.py:174` `expected_model_change()` | 冻结 α、β,克隆当前 θ;分别假设"答对"/"答错"各做几轮梯度更新,得 pos_weights / neg_weights;返回 `p·||θ⁺−θ|| + (1−p)·||θ⁻−θ||`。注意代码是**真的小步重训**而非论文说的单步梯度近似,每次算完把 θ 恢复原值 |
| Quality Module 取 top-K_C | `maat_strategy.py:35-37` | 对每个未测题算 EMC,`np.argsort` 降序取前 `n_candidates`(默认 10,即论文 K_C=10) |
| Diversity Module / IWKC | `maat_strategy.py:18-27` `_compute_coverage_gain()` | 对候选题算"加入后"的覆盖分:`Σ cnt/(cnt+1) / 知识点数`,正是式 6 的 IncCov 软覆盖;`maat_strategy.py:38` 用 `max(candidates, key=...)` 贪心选边际增益最大者(直接比"加入后的总分"等价于比边际增益) |
| Importance Module w_k | **未实现** | 代码里所有知识点等权(相当于 w_k≡1),即论文的 WKC 退化版,没有 Item2Vec/SGNS 部分 |
| 更新 θ(观察作答后) | `irt_model.py:96` `adaptest_update()` | 只对 `model.theta` 建 optimizer(α、β 不动),用刚测的那道题(`get_tested_dataset(last=True)`)训练 8 个 epoch |
| Algorithm 1 主循环 | `driver.py:23-36` | 每轮:策略选题 → `apply_selection`(untested→tested)→ 更新 θ → 评估 AUC/Coverage 并写 tensorboard |
| Inf(S) = AUC(式 17)、Cov(S)(式 18) | `irt_model.py:133` `adaptest_evaluate()` | 用当前 θ 预测该生**全部**有真实标签的题算 AUC;覆盖率 = 已测题涉及知识点 / 该生做过的题涉及的全部知识点 |

### 输入数据格式

- **`train_triplets.csv` / `test_triplets.csv`**:三元组 `student_id, question_id, correct`(0/1),就是论文的作答记录 r = <​e_i, q_j, a_ij>。按学生划分:1426 个学生做历史数据(训练),47 个学生模拟考生(测试),学生 ID 各自重新编号(`data_prep.py:split_data_by_student`);
- **`concept_map.json`**:`{题目id: [知识点id,...]}`,即论文的 Q-K 二元关系 G;
- **`metadata.json`**:1473 学生 / 903 题 / 22 知识点 / 58427 条记录。

### 训练了什么、怎么测的

1. **`scripts/train.py`(测前)**:用 1426 个历史学生的全部三元组训练 IRT(lr=0.002, batch=2048, 100 epochs, 1 维 θ),交叉熵损失。`adaptest_save` **只保存 α、β(题目参数)**,学生 θ 不保存 —— 因为测试时面对的是全新考生;
2. **`scripts/test.py`(模拟 CAT)**:加载 α、β,47 个测试学生的 θ 随机初始化;他们的真实作答记录当作"标准答案库",系统每选一题就"揭晓"该题对错(`data[sid][qid]`),模拟真实考试。测 50 步,每步后评估。对比 `RandomStrategy` 和 `MAATStrategy`。

### 最后得到什么

`results/<时间戳>/` 下的 tensorboard 日志,记录每一步(0~50)的两条曲线:**AUC**(诊断准确度,质量指标)和 **cov**(知识点覆盖率,多样性指标),可直观看到 MAAT 两条曲线都压过 Random。

## 6. 实验结果如何?

数据集:EXAM(讯飞私有,4307 生/527 题/31 知识点)和 ASSIST(公开 ASSISTments 2009-2010)。测长 N=50,评估第 25/50 步。

- **质量(AUC@25/@50,Table III)**:MAAT 在 IRT、MIRT、NCDM 三种 CDM 上全部胜过对应的 model-specific 基线。如 EXAM+IRT@50:MAAT 0.7319 > KLI 0.7257 > MFI 0.7207;NCDM 上(无基线可用)也明显超过 RAND(0.7868 vs 0.7566)。而且 CDM 越复杂整体 AUC 越高,印证 model-agnostic 的价值:换更强的 CDM 不用改策略,直接受益;
- **多样性(Coverage,Fig. 3)**:MAAT 覆盖率在测试前期快速逼近 1,大幅领先所有基线;同时发现 MFI/KLI 也顺带提升覆盖 —— 说明质量与多样性是相关而非对立的目标;
- **一致性验证(SEE,Fig. 4)**:在传统模拟研究(IRT 模拟参数)下,MAAT 的参数估计误差下降速度也与 MFI/KLI 相当且远快于 RAND,说明新评估方式与传统结论一致;
- **消融(K_C,Fig. 5)**:K_C 越大覆盖涨越快;K_C=10 时多样性已接近上限,而质量最好的反而不是 K_C=1(因为覆盖/重要性也间接帮助质量)—— 小的 K_C(10)即可取得最佳平衡;
- **案例(Table IV)**:同一考生前 10 题,MAAT 覆盖 9 个知识点,D-Opt 只有 5 个、MKLI 6 个,且 AUC 还略高;基线反复选"Function"相关题,MAAT 不会。

## 7. 局限性

**论文自身的局限:**

1. **EMC 计算开销大**:每步要对**每道未测题**分别模拟"答对/答错"两次参数更新。虽然论文说可用单步梯度近似,但对大题库、深度 CDM 仍是 O(|Q_U|) 次前向+反向,比 MFI 的解析式贵得多;论文没有报告运行时间;
2. **w_k 是静态的**:知识点重要性由历史数据测前算好,对所有考生一样、测试中不变,没有个性化(不同学生薄弱点不同,重要性理应自适应);
3. **贪心只有 (1−1/e) 近似**,且质量-多样性平衡靠单一超参 K_C 硬切换(先质量后多样性的两阶段串联),不是真正的联合多目标优化;
4. **评估依赖真实数据回放**:只能从"该生历史上做过的题"里选(否则没有标签),题库被人为缩小,与真实 CAT 场景有偏差;AUC 也是用"全量作答"训练出的参数当近似真值;
5. 未考虑曝光控制、题目使用均衡、作答时间等实际 CAT 运营约束;实验只有数学学科的两个数据集。

**这份代码相对论文的缺口(读代码时要注意):**

1. **Importance Module 完全缺失**:没有 Test-Effect Embedding/SGNS/密度估计,w_k 等权 —— 实现的其实是 WKC 而非 IWKC;
2. 只实现了 IRT(1 维,近似 2PL);NCDM/MIRT 模型代码不在此 repo(仅留了一个 ncd checkpoint);
3. EMC 用多轮真实重训而非论文提到的梯度近似,`num_epochs=8` 意味着每题 16 次小训练,慢;
4. `expected_model_change` 里 optimizer 是 Adam 且每次新建,冻结非 θ 参数的方式(`requires_grad=False` 后又全部置 True)与 `adaptest_update` 的注释代码有些不一致,属工程简化;
5. 基线里只有 Random,没有 MFI/KLI 的实现,复现 Table III 需自行补。

## 8. 一句话总结

MAAT 把主动学习的"期望模型变化"和次模覆盖优化引入 CAT:用 EMC 把选题策略与 CDM 解耦(黑盒也能测信息量),用 IWKC+贪心保证知识点覆盖有 (1−1/e) 理论下界,在 IRT/MIRT/NCDM 上同时赢得质量与多样性 —— 代价是每步对全题库做两次假设更新的计算开销,以及开源代码只落地了框架的一半(缺 Importance Module)。
