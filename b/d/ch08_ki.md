## 第八章：Knowledge Insulation — Train Fast, Run Fast, Generalize Better (2025.05)

**论文信息**
- 发表时间: 2025-05-29
- 会议: NeurIPS 2025 (spotlight)
- arXiv: https://arxiv.org/abs/2505.23705
- 核心作者: Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z. Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, Sergey Levine (Physical Intelligence)

### 1. 解决了什么问题

当前VLA模型面临三重困境。第一，自回归VLA（如pi0-FAST）推理速度极慢，在RTX 4090上预测1秒的动作块需要约750ms，控制频率仅约1.3Hz，无法满足高频灵巧操作需求。第二，带有连续动作输出的VLA（如pi0）虽然通过300M参数的action expert实现了10Hz的控制频率，但其随机初始化的action expert在训练时会通过梯度反传破坏预训练VLM骨干网络中的语义知识，导致模型无法正确遵循语言指令（例如被指示收拾勺子却去抓垃圾）。第三，简单冻结VLM骨干网络不可行，因为预训练VLM的表示不包含机器人控制所需的信息，冻结后性能为0%。本文提出"知识绝缘"（Knowledge Insulation）技术，通过阻断action expert到VLM骨干的梯度流，同时用离散动作token为骨干网络提供替代学习信号，实现了训练速度提升（比pi0快7.5倍收敛）、推理速度快（连续动作输出）、泛化能力更强的三重目标。

### 2. 数据来源、处理方式及其原因

训练数据分为机器人动作数据和通用VLM数据两大类。机器人数据涵盖12种机器人构型：单臂静态机械臂（ARX、UR5、Franka）、双臂静态机械臂（ARX、AgileX、Trossen、UR5）以及双臂移动操作平台（mobile Trossen、ARX slate、Galaxea G1、Hexmove H1、Fibocom），还包含开源OXE数据集。任务种类远超评估范围，包括磨咖啡豆、挂毛巾等日常操作。VLM共训数据包括：图像描述（CapsFusion、COCO）、视觉问答（Cambrian-7M、PixMo、VQAv2）和目标定位（含额外标注的室内场景和家用物品包围框数据）。

数据处理的关键设计是将动作数据同时转换为两种表示：(1) 通过FAST分词器将连续动作序列经离散余弦变换、量化和BPE编码压缩为离散token；(2) 保留原始连续动作用于flow matching训练。这种双重表示使模型能同时从离散token获得快速表示学习信号，并通过连续动作保持精确控制能力。VLM数据的引入则是为了在微调过程中保持骨干网络的语义知识，防止灾难性遗忘，实验证明移除VLM数据会显著降低语言遵循率和OOD泛化能力。

### 3. 模型结构及设计原因

模型采用双流混合专家架构，由3B参数的VLM骨干网络和300M参数的action expert组成。VLM骨干基于PaliGemma（SigLIP视觉编码器 + 2B Gemma语言模型），配置为width=2048、depth=18、mlp_dim=16384、num_heads=8、num_kv_heads=1、head_dim=256。Action expert为更小的Transformer，配置为width=1024、mlp_dim=4096，其余与骨干相同，共300M参数。

两个模块仅通过self-attention层交互，采用精心设计的注意力掩码：图像、语言token和文本状态使用完整前缀掩码；FAST离散动作token对前缀做全注意力、对自身做自回归注意力；action expert的embedding对前缀和自身做全注意力，但不能注意到FAST token（避免两种动作表示之间的信息泄露）。信息单向从VLM流向action expert，VLM的任何embedding都不注意action expert。

知识绝缘的核心实现在于修改注意力计算：在action expert的query与VLM骨干的key/value交互处插入stop-gradient算子（sg），使得action expert能读取骨干特征但其loss梯度不会回传到骨干权重。骨干网络仅通过离散动作token的交叉熵损失和VLM数据的语言建模损失进行更新，保护了预训练知识。

### 4. 训练方法（算法与工程）及其原因

训练采用联合损失函数 L_CO-VLA = 自回归语言/动作建模损失 + alpha * flow matching损失。具体而言，骨干网络通过next-token prediction同时预测语言token和FAST离散动作token，action expert通过flow matching预测连续动作的向量场。由于使用了stop-gradient，alpha可直接设为1，无需复杂的损失权重调参——两个损失作用于完全独立的权重集合。

Flow matching的时间步采样遵循pi0的设计，使用偏向低时间步的Beta分布 p(tau) = Beta((s-tau)/s; alpha=1.5, beta=1), s=0.999，而非标准均匀采样。动作预测使用H=50的action chunk horizon。噪声动作通过线性插值 a^{tau,omega} = tau * a + (1-tau) * omega 构造，模型预测目标为向量场 omega - a。

训练数据灵活混合三类样本：纯VLM数据（仅有图文标注，仅计算语言损失）、纯动作数据（条件于图像和文本的动作预测）、以及同时包含语言描述和动作标注的混合数据。这种设计使模型能从VLM数据中迁移语义知识到机器人动作生成。推理时仅使用action expert的flow matching输出生成连续动作，离散动作分支仅在训练时作为表示学习目标使用，不增加推理开销。训练同时使用连续和离散输出会增加约20%的计算成本，但由于收敛速度大幅提升，实际wall-clock训练时间反而显著减少。

### 5. 实验揭示的关键发现

**性能与收敛速度**：在table bussing任务上，本方法的收敛速度与pi0-FAST相当，而pi0需要7.5倍的训练步数才能达到类似性能。在DROID基准上，本方法得分0.55+-0.09，pi0为0.49+-0.09，pi0-FAST为0.45+-0.09。在LIBERO-90和LIBERO-Spatial上取得了新的SOTA（从泛化模型微调分别达到96.0%和98.0%）。

**语言遵循能力**：这是本文最显著的发现。pi0由于action expert的梯度干扰严重损害了语言遵循能力；stop-gradient能有效改善这一问题。在items in drawer任务中，本方法在性能和语言遵循两方面均显著超越所有基线。实验表明，如果同时使用VLM数据共训，即使不用stop-gradient的joint-training也能获得较好的语言遵循，但stop-gradient + VLM数据的组合效果最优。

**语义泛化**：在移动操作机器人的OOD物体泛化实验中，VLM数据共训对识别训练中未见过的新物体至关重要，验证了VLA从网络数据向机器人策略迁移知识的核心价值。

**推理效率**：pi0-FAST虽然语言遵循良好但推理极慢，完成table bussing任务需要的wall-clock时间是本方法的两倍。本方法通过action expert实现快速连续动作生成，兼顾了速度与精度。

### 6. 消融实验的发现

**Stop-gradient的效果**：移除stop-gradient（即joint-training）会降低语言遵循能力，尤其在不使用VLM数据共训时效果更差。在items in drawer任务中，joint-training基线因无法正确遵循语言指令而性能显著下降。但在table bussing等任务中，如果配合VLM数据共训，joint-training也能取得不错的表现，说明VLM数据共训提供了一定程度的知识保护。

**VLM数据共训的效果**：移除VLM数据（ours w/o VLM data）会略微降低任务完成率，但对joint-training的语言遵循率影响更大。论文推测VLM数据对于避免预训练表示的灾难性干扰尤为重要。在OOD泛化测试中，VLM数据共训对新物体识别至关重要。

**离散动作表示的选择**：用朴素分词替代FAST作为表示学习目标，模型仍优于仅使用连续动作训练的pi0，但劣于FAST。对朴素token做stride=5的下采样效果优于密集朴素分词，说明FAST的时间压缩特性为表示学习提供了更高效的信号。

**注意力掩码设计**：HybridVLA允许自回归token注意到连续动作token，而本文禁止两种动作表示之间的互相注意。实验表明本文的掩码设计性能显著优于HybridVLA，验证了隔离两种动作表示的必要性。

**冻结骨干网络**：完全冻结VLM骨干仅训练action expert在所有任务上性能接近0%（items in drawer、shirt folding），证明预训练VLM的表示不足以支撑机器人控制，必须微调骨干但需要保护其知识。

**状态表示**：比较了文本状态、特殊token状态和连续状态三种本体感觉表示方式。本方法在文本状态和连续状态下均表现良好，pi0在两种表示下都较差，说明pi0与本方法的性能差距不能归因于状态表示的不同。特殊token状态表现最差。

---
