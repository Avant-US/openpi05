# Physical Intelligence (π) 研究成果完整总览（补充版）

> 本文件是 `pi_docls.md` 的补充版，包含更多被遗漏的论文、博客、开源项目，以及创始人在 PI 成立之前的关键前置研究工作。
>
> Physical Intelligence 官网: <https://www.pi.website/>
>
> 旧域名: <https://www.physicalintelligence.company/>
>
> GitHub 组织: <https://github.com/Physical-Intelligence>
>
> X/Twitter: <https://x.com/physical_int>

---

## 一、Physical Intelligence 正式发表的论文 & 技术报告（由新到旧，完整版）

### 1. π₀.₇: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

- **日期**: 2026-04-16（v1），2026-04-24（v2）
- **arXiv**: <https://arxiv.org/abs/2604.15483>
- **博客**: <https://www.pi.website/blog/pi07>
- **PDF（PI 官网）**: <https://www.pi.website/download/pi07.pdf>
- **作者**: Physical Intelligence（Bo Ai, Ashwin Balakrishna, Kevin Black, Chelsea Finn, Danny Driess, Karol Hausman, Sergey Levine, Karl Pertsch 等 87 位共同作者）
- **简介**: ~5B 参数 VLA 模型（Gemma3 4B VLM backbone + 860M action expert）。支持丰富多模态 prompt（语言指令、子目标图像、视频、子任务元数据等）。展现组合泛化与零样本跨机体迁移能力（如在 UR5e 上零样本折叠衣物，尽管该机体无训练数据）。在匹配对比中，π₀.₇ 任务进度 85.6%、成功率 80%，接近专家人类遥操作者（90.9% / 80.6%）。采用训练时 RTC（Real-Time Chunking）实现平滑动作生成。训练数据包含人类干预 rollouts、开源机器人数据集、以自我为中心的人类视频、web 辅助数据等多种来源。

### 2. RL Token: Bootstrapping Online RL with Vision-Language-Action Models

- **日期**: 2026-04-24（arXiv），2026-03-19（博客首发）
- **arXiv**: <https://arxiv.org/abs/2604.23073>
- **项目页**: <https://www.pi.website/research/rlt>
- **PDF（PI 官网）**: <https://www.pi.website/download/rlt.pdf>
- **作者**: Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke（Physical Intelligence）
- **简介**: 从 VLA 模型中提取"RL Token"——一个紧凑的内部表征摘要，作为小型 actor-critic 的输入。使大型 VLA 可在 15 分钟至数小时内通过在线 RL 精调精密操作任务（拧 M3 螺丝、扎线带、插以太网线、插电源线）。在部分任务上速度甚至超越人类遥操作（如以太网线插入速度超中位人类遥操作速度）。基于 π₀.₆ 模型实现。RLT 先通过 encoder-decoder transformer 训练 bottleneck 来预测 VLA 内部 embedding，产生压缩表征（RL Token），再在此上训练轻量 actor-critic 头。冻结的 VLA 提供广泛感知理解和动作建议，轻量 actor-critic 在线适应以完成任务最难部分。

### 3. MEM: Multi-Scale Embodied Memory for Vision Language Action Models

- **日期**: 2026-03-04（v1），2026-03-08（v2）
- **arXiv**: <https://arxiv.org/abs/2603.03596>
- **项目页**: <https://www.pi.website/research/memory>
- **PDF（PI 官网）**: <https://www.pi.website/download/Mem.pdf>
- **作者**: Marcel Torne, Karl Pertsch 等 17 位作者（Physical Intelligence）
- **简介**: 双尺度记忆架构——短时视频记忆（ViT encoder 压缩最近几秒画面）+ 长时文本记忆（chain-of-thought 语言摘要记录已完成的子任务）。使机器人能执行长达 15 分钟的连续任务（厨房清洁、做三明治、整理杂货）。集成于 π₀.₆ VLA（初始化自 Gemma 3-4B）。还展现了上下文内适应能力（如抓取失败后自动调整策略）。基准任务包括：厨房清洁（15 分钟，擦台面、洗碗、收纳食物）、食谱准备（从抽屉/柜子取物品）、物流（拆包杂货并计数）。

### 4. Emergence of Human to Robot Transfer in Vision-Language-Action Models

- **日期**: 2025-12-27（arXiv），2025-12-16（博客首发）
- **arXiv**: <https://arxiv.org/abs/2512.22414>
- **项目页**: <https://www.pi.website/research/human_to_robot>
- **作者**: Simar Kareer, Karl Pertsch, James Darpinian, Judy Hoffman, Danfei Xu, Sergey Levine, Chelsea Finn, Suraj Nair（Physical Intelligence）
- **简介**: 发现当 VLA 在足够多样的机器人数据上预训练后，从人类视频到机器人操作任务的迁移能力会"涌现"出来。提出简单的协同训练方案（co-training recipe），分析表明多样化预训练产生了跨具身体的不可知表征（embodiment-agnostic representations）。

### 5. π*₀.₆: a VLA That Learns From Experience (RECAP)

- **日期**: 2025-11-18（v1），2025-11-19（v2）
- **arXiv**: <https://arxiv.org/abs/2511.14759>
- **博客**: <https://www.pi.website/blog/pistar06>
- **PDF（PI 官网）**: <https://www.pi.website/download/pistar06.pdf>
- **Model Card PDF**: <https://website.pi-asset.com/pi06star/PI06_model_card.pdf>
- **作者**: Physical Intelligence（Ali Amin, Ashwin Balakrishna, Kevin Black, Chelsea Finn, Sergey Levine 等 55 位共同作者）
- **简介**: 提出 RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）方法。核心挑战是信用分配（credit assignment）：理解哪些动作导致了好坏结果。RECAP 通过训练价值函数进行优势条件化，使 VLA 能从自主部署经验和人类纠正中进行强化学习自我改进。π*₀.₆ 在困难任务上吞吐量翻倍以上（洗衣折叠、制作浓缩咖啡），失败率降低约 50%，可连续自主部署 13 小时。π₀.₆ 基于 π₀.₅ 构建。模型能在真实家庭中折叠衣物、可靠地组装盒子、使用专业咖啡机制作浓缩咖啡饮料。

### 6. Real-Time Execution of Action Chunking Flow Policies (RTC)

- **日期**: 2025-06-09（v1），2025-12-05（v2）
- **arXiv**: <https://arxiv.org/abs/2506.07339>
- **项目页**: <https://www.pi.website/research/real_time_chunking>
- **PDF（PI 官网）**: <https://www.pi.website/download/real_time_chunking.pdf>
- **代码**: <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **作者**: Kevin Black, Manuel Y. Galliker, Sergey Levine（Physical Intelligence）
- **会议**: NeurIPS 2025 (poster)
- **简介**: 无需重训练的推理时算法，使任何 diffusion/flow-based VLA 实现实时平滑异步执行。核心思想是在执行当前动作块的同时生成下一个，"冻结"已确定要执行的动作并"修复"（inpaint）其余部分。引入 12 个高动态 Kinetix 仿真任务基准，以及 6 个真实世界双臂操作任务评估。即使注入 300ms+ 人工延迟仍能完成划火柴、插网线等精密任务。2025-12-08 发布了训练时版本 RTC 的后续论文，用于 π₀.₆ 制作咖啡的演示。

### 7. Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better

- **日期**: 2025-05-29
- **arXiv**: <https://arxiv.org/abs/2505.23705>
- **项目页**: <https://www.pi.website/research/knowledge_insulation>
- **PDF（PI 官网）**: <https://www.physicalintelligence.company/download/pi05_KI.pdf>
- **OpenReview**: <https://openreview.net/forum?id=cb0xbZ3APM>
- **作者**: Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z. Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, Sergey Levine（Physical Intelligence）
- **会议**: NeurIPS 2025 (spotlight)
- **简介**: 通过"知识绝缘"——阻断 action expert（flow matching/diffusion）的梯度回传到 VLM backbone，同时用 FAST 离散 token + web data 联合训练 backbone，使其保留语义知识。第一步是停止从 action expert 到 VLM backbone 的梯度流；第二步是同时在 π₀-FAST action tokens（用于表征学习）、通用 web 数据（用于泛化）和连续动作（通过 action expert）上训练。训练时间减少 66%，泛化能力更好。形式化了 π₀.₅ 中使用的方法，并扩展为更精炼的单阶段训练配方（π₀.₅ + KI）。

### 8. π₀.₅: a Vision-Language-Action Model with Open-World Generalization

- **日期**: 2025-04-22
- **arXiv**: <https://arxiv.org/abs/2504.16054>
- **博客**: <https://www.physicalintelligence.company/blog/pi05>（也可通过 <https://www.pi.website/blog/pi05>）
- **PDF（PI 官网）**: <https://www.pi.website/download/pi05.pdf>
- **作者**: Physical Intelligence（Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter 等 35+ 位共同作者）
- **会议**: CoRL 2025（第 9 届机器人学习年度会议）
- **简介**: 基于 π₀ 的升级模型，通过异构多任务协同训练（多机器人数据、高层语义子任务预测、物体检测、web 数据）实现开放世界泛化。使用混合多模态样本（图像观测、语言指令、物体检测、语义子任务预测、低层动作）。能在训练中从未见过的全新家庭环境中执行厨房清洁、卧室整理等任务。

### 9. Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

- **日期**: 2025-02-26（v1），2025-07-15（v2）
- **arXiv**: <https://arxiv.org/abs/2502.19417>
- **项目页**: <https://www.pi.website/research/hirobot>
- **PDF（PI 官网）**: <https://www.pi.website/download/hirobot.pdf>
- **作者**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner 等（Physical Intelligence）
- **简介**: 分层架构——VLM 作为"系统2"（慢思考）分解复杂指令为简单子步骤，π₀ 作为"系统1"（快反应）执行低层动作。类似 LLM 的"思维链"，允许机器人"自言自语"进行推理。支持实时人类反馈纠正。在单臂、双臂、双臂移动机器人三个平台上测试，能完成清洁桌面、制作三明治、杂货购物等复杂任务。知道"请勿擦除"写在白板上意味着什么、知道易碎物品需小心处理。

### 10. FAST: Efficient Action Tokenization for Vision-Language-Action Models

- **日期**: 2025-01-16
- **arXiv**: <https://arxiv.org/abs/2501.09747>
- **项目页**: <https://www.pi.website/research/fast>
- **PDF（PI 官网）**: <https://www.pi.website/download/fast.pdf>
- **作者**: Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine（Physical Intelligence / UC Berkeley / Stanford）
- **会议**: RSS 2025（Robotics: Science and Systems 2025）
- **简介**: 提出基于离散余弦变换（DCT）的频域动作序列 tokenizer（Frequency-space Action Sequence Tokenization）。解决传统逐维度逐时间步离散化在高频灵巧控制下完全失效的问题。流程：对每个动作维度应用 DCT（类似 JPEG/MP3 的压缩算法）→ 量化去除不重要系数 → 展平为一维向量 → BPE 编码压缩为密集 action tokens，实现约 10 倍压缩。使自回归 VLA 训练速度提升 5 倍。还发布了 FAST+——基于 100 万条真实机器人动作轨迹训练的通用 tokenizer，可作为黑箱用于多种机器人动作空间和控制频率。基于 FAST 训练的 π₀-FAST 首次在 DROID 数据集上训练出了能在全新环境中跟随多种指令泛化的通用策略。

### 11. π₀: A Vision-Language-Action Flow Model for General Robot Control

- **日期**: 2024-10-31
- **arXiv**: <https://arxiv.org/abs/2410.24164>
- **博客**: <https://www.pi.website/blog/pi0>（旧域名: <https://physicalintelligence.company/blog/pi0>）
- **PDF（PI 官网）**: <https://www.pi.website/download/pi0.pdf>
- **作者**: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky（Physical Intelligence）
- **简介**: Physical Intelligence 的首个通用机器人策略基础模型。基于大规模预训练和 flow matching 动作生成，50Hz 输出连续低层电机命令。在 7 个机器人平台、68 个独立任务上训练（约 10,000 小时演示数据 + OXE 开源数据集——据其所知为当时最大规模的机器人学习实验）。能折叠衣物、收拾餐桌、打包杂货、组装盒子、取回物品等。挑战性任务包括：多物体杂乱桌面清理、纸箱从展平状态组装、将食物移入外卖盒并关闭、将 6 个鸡蛋从碗中取出放入蛋盒。

---

## 二、PI 成员在公司成立前后的关键前置/关联研究（由新到旧）

> 以下论文虽非以 "Physical Intelligence" 名义发表，但核心作者均为 PI 创始成员或早期员工，构成了 π₀ 模型的直接技术前置。

### 12. Octo: An Open-Source Generalist Robot Policy

- **日期**: 2024-05-20（v1），2024-05-26（v2）
- **arXiv**: <https://arxiv.org/abs/2405.12213>
- **项目页**: <https://octo-models.github.io/>
- **会议**: RSS 2024（Delft, Netherlands）
- **RSS 论文集**: <https://www.roboticsproceedings.org/rss20/p090.pdf>
- **作者**: Octo Model Team: Dibya Ghosh\*, Homer Walke\*, Karl Pertsch\*, Kevin Black\*, Oier Mees\*, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, Sergey Levine（UC Berkeley / Stanford / CMU / Google DeepMind）
- **简介**: π₀ 的直接前身。基于 Transformer 的 diffusion policy，预训练于 Open X-Embodiment 数据集的 80 万条机器人轨迹。支持语言指令和目标图像引导，可灵活微调至新的传感器输入和动作空间。Octo-Small 27M 参数，Octo-Base 93M 参数。表现优于 RT-1-X，接近 550 亿参数的 RT-2-X。平均跨 6 个评估设置比次优基线高 52%。可在消费级 GPU 上数小时完成微调。Kevin Black, Karl Pertsch, Chelsea Finn, Sergey Levine 等作者后来加入/创立了 Physical Intelligence。

### 13. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset

- **日期**: 2024-03-19（v1），2025-04-22（v2）
- **arXiv**: <https://arxiv.org/abs/2403.12945>
- **项目页**: <https://droid-dataset.github.io/>
- **会议**: RSS 2024（Delft, Netherlands）
- **RSS 论文集**: <https://www.roboticsproceedings.org/rss20/p120.pdf>
- **作者**: Alexander Khazatsky\*, Karl Pertsch\*, Suraj Nair\*, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama 等 100+ 位作者（跨多机构大规模协作，PI 成员 Karl Pertsch、Suraj Nair、Ashwin Balakrishna 为联合第一作者/核心贡献者）
- **简介**: 分布式机器人交互数据集（Distributed Robot Interaction Dataset）。76,000 条演示轨迹 / 350 小时交互数据，横跨 564 个场景、86 个任务，由 50 名数据收集者在北美、亚洲、欧洲历时 12 个月收集。硬件：Franka Panda 7DoF 机械臂 + 双 Zed 2 立体摄像头 + 腕部 Zed Mini + Oculus Quest 2 遥操作。1,417 个摄像头视角，附内外参标定。训练于 DROID 的策略具有更高性能和更好的泛化能力。PI 的 openpi 仓库中提供了 π₀-FAST DROID 和 π₀ DROID 微调模型。

### 14. BridgeData V2: A Dataset for Robot Learning at Scale

- **日期**: 2023-08-24（v1），2024-01-17（v3）
- **arXiv**: <https://arxiv.org/abs/2308.12952>
- **项目页**: <https://rail-berkeley.github.io/bridgedata/>
- **会议**: CoRL 2023（第 7 届机器人学习会议）
- **PMLR**: <https://proceedings.mlr.press/v229/walke23a.html>
- **作者**: Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, Sergey Levine（UC Berkeley / Stanford）
- **简介**: 60,096 条轨迹，横跨 24 个环境，在公开可用的低成本机器人上收集。包含拾放、推动、扫除、堆叠、折叠等技能。CC-BY-4.0 开源。OXE 数据集的重要组成部分。Homer Walke, Kevin Black, Quan Vuong, Chelsea Finn, Sergey Levine 等后来加入/创立了 Physical Intelligence。

---

## 三、博客文章 & 公告（由新到旧，完整版）

### 1. π₀.₇: a Steerable Model with Emergent Capabilities

- **日期**: 2026-04-16
- **URL**: <https://www.pi.website/blog/pi07>
- **简介**: π₀.₇ 模型发布博客。展示装空气炸锅、折叠牛仔裤、用 Windex 清洁玻璃等演示。介绍可操控 prompt 框架与涌现能力。不同 prompt 模态使其可集成多种数据源（不同机器人、控制模态、人类视频、自主数据）。展示视觉子目标图像引导、跨机体迁移（在 UR5e 上折叠衣物无需该机体训练数据）。

### 2. Precise Manipulation with Efficient Online RL (RL Token)

- **日期**: 2026-03-19
- **URL**: <https://www.pi.website/research/rlt>
- **简介**: RL Token 方法首发博客。展示如何在几小时甚至几分钟内通过在线 RL 精调精密操作。评估了四个高精度操作任务：螺丝刀拧 M3 螺丝、扎线带固定、以太网线插入、电源线插入。

### 3. MEM: VLAs with Long and Short-Term Memory

- **日期**: 约 2026-03-03
- **URL**: <https://www.pi.website/research/memory>
- **简介**: 多尺度具身记忆（MEM）方法博客，使机器人具备 15 分钟长时任务执行能力。展示厨房清洁、做饭等长时任务。

### 4. The Physical Intelligence Layer（合作伙伴）

- **日期**: 2026-02-24
- **URL**: <https://www.pi.website/blog/partner>
- **简介**: 介绍与合作伙伴 Weave、Ultra 在真实仓库中的部署。模型在实际客户仓库中打包真实订单。展示 π₀→π₀.₅→π₀.₆ 逐代在智能、吞吐量和可靠性上的显著提升。阐述"物理智能层"的愿景——像 LLM API 一样让任何人使用机器人基础模型。人机交互系统在模型出错时介入，确保跨数千订单的正确性，同时持续生成改进下一代模型的新数据。

### 5. Moravec's Paradox and the Robot Olympics ★ 新增

- **日期**: 2025-12-22
- **URL**: <https://www.pi.website/blog/olympics>
- **简介**: 讨论 Moravec 悖论——下棋对机器"容易"但搬棋子"难"。通过微调 π₀.₆，在 Benjie Holson 提出的"机器人奥运会"挑战任务中取得金牌：开门、钥匙操作、清洗油锅、涂花生酱、翻袜子等。π₀.₆ 在 5 个类别中的 3 个获得金牌，每个任务微调数据不超过 9 小时。基线模型（无专用机器人预训练）仅达 9% 任务进度，而微调模型达 72% 进度和 52% 平均成功率。Moravec 悖论可视为数据稀疏性的声明——如果无法从网上数据学到所需，就不得不通过编程实现，效果不佳。

### 6. Emergence of Human to Robot Transfer

- **日期**: 2025-12-16
- **URL**: <https://www.pi.website/research/human_to_robot>
- **简介**: 探索如何通过规模化使人类视频到机器人操作的迁移能力涌现。

### 7. Real-Time Action Chunking (RTC)

- **日期**: 首发约 2025-06，更新于 2025-12-08
- **URL**: <https://www.pi.website/research/real_time_chunking>
- **简介**: 实时动作分块方法博客。2025-12-08 更新发布了训练时 RTC 版本后续论文，用于 π₀.₆ 制作咖啡演示。

### 8. π*₀.₆: A VLA that Learns from Experience

- **日期**: 2025-11-17
- **URL**: <https://www.pi.website/blog/pistar06>
- **简介**: RECAP 方法与 π*₀.₆ / π₀.₆ 模型发布博客。两种从"坏"经验数据中获得好训练信号的方式：教练式纠正（coaching）和强化学习（机器人自行判断哪些行为好坏）。

### 9. Knowledge Insulation: VLAs that Train Fast, Run Fast, and Generalize Better

- **日期**: 约 2025-05（与论文同步）
- **URL**: <https://www.pi.website/research/knowledge_insulation>
- **简介**: 知识绝缘方法博客，介绍如何在不损失预训练知识的前提下高效训练 VLA。

### 10. π₀.₅: A VLA with Open-World Generalization

- **日期**: 约 2025-04
- **URL**: <https://www.pi.website/blog/pi05>（也可通过旧域名: <https://www.physicalintelligence.company/blog/pi05>）
- **简介**: π₀.₅ 模型发布博客，介绍开放世界泛化能力。能控制移动操作机器人清理全新厨房或卧室。

### 11. Hi Robot: Teaching Robots to Listen and Think Harder

- **日期**: 2025-02-26
- **URL**: <https://www.pi.website/research/hirobot>
- **简介**: 分层推理 + 人机交互博客。介绍"系统1 + 系统2"架构。机器人如同获得了"内心独白"。

### 12. Open Sourcing π0

- **日期**: 2025-02-04
- **URL**: <https://www.pi.website/blog/openpi>
- **简介**: 宣布开源 π₀ 和 π₀-FAST 的权重与代码。发布 openpi 仓库，含预训练 checkpoint、ALOHA/DROID 微调模型、推理与微调代码。HuggingFace 同步发布 PyTorch 移植版。发现 1 到 20 小时数据即足以微调到多种任务。希望像开源 LLM/VLM 引发"寒武纪大爆发"一样，openpi 能激发机器人基础模型的新应用。

### 13. FAST: Efficient Robot Action Tokenization ★ 项目页新增

- **日期**: 约 2025-01-16
- **URL**: <https://www.pi.website/research/fast>
- **简介**: FAST tokenizer 发布。基于 DCT 的压缩式 tokenization，相比传统离散化实现约 10 倍压缩。π₀-FAST 训练速度比 π₀ 快 5 倍。首次在 DROID 数据集上训练出可泛化到新环境的通用策略。发布 FAST+ 通用 tokenizer。

### 14. π₀: Our First Generalist Policy

- **日期**: 2024-10-31
- **URL**: <https://www.pi.website/blog/pi0>（旧域名: <https://physicalintelligence.company/blog/pi0>）
- **简介**: Physical Intelligence 首个通用策略模型发布博客。介绍 π₀ 如何结合大规模多任务多机器人数据和新网络架构，成为最强大最灵巧的通用机器人策略。

---

## 四、开源代码 & 模型（完整版）

### 1. openpi（主仓库）

- **URL**: <https://github.com/Physical-Intelligence/openpi>
- **简介**: Physical Intelligence 的核心开源仓库（JAX 实现）。包含：
  - **π₀ 模型**: flow-based VLA，预训练于 10,000+ 小时机器人数据
  - **π₀-FAST 模型**: 基于 FAST tokenizer 的自回归 VLA，语言跟随性能略好，但推理成本约 4-5 倍
  - **π₀.₅ 模型**: 增强开放世界泛化（Knowledge Insulation 训练）
  - 多个预训练 checkpoint:
    - **Pi0 base**: 标准预训练模型，在 OXE + PI 的 7 个机器人平台上训练
    - **Pi0-FAST base**: 使用 FAST tokenizer
    - **Pi0-FAST DROID / Pi0 DROID**: 微调到 DROID 数据集，首个能在全新环境中跟随指令的 DROID 模型
    - **Pi0 ALOHA**: 微调到 ALOHA 平台任务（毛巾折叠、食物舀取等）
  - 远程推理支持（WebSocket 连接，可使用 off-robot GPU）
  - 无需机器人的推理测试脚本（生成随机观测输入）
  - Docker 安装支持
  - 已测试于 Ubuntu 22.04
  - 2025-06 新增：使用 openpi 在完整 DROID 数据集上训练 VLA 的说明（近似复现 pi0-FAST-DROID 训练管线）
  - 2025-09 新增：PyTorch 支持、pi05 发布、改进的 DROID 训练空闲过滤器

### 2. real-time-chunking-kinetix

- **URL**: <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **简介**: RTC 论文的 Kinetix 仿真实验代码。12 个高动态仿真任务基准。

### 3. pi-data-sharing

- **URL**: <https://github.com/Physical-Intelligence/pi-data-sharing>
- **简介**: 数据共享相关工具与规范。

### 4. aloha

- **URL**: <https://github.com/Physical-Intelligence/aloha>
- **简介**: ALOHA 机器人平台相关代码。ALOHA 是一个低成本双臂灵巧操作系统。

### 5. augmax

- **URL**: <https://github.com/Physical-Intelligence/augmax>
- **简介**: 数据增强库，附 X-embodiment 数据集转换的 RLDS 数据集构建器示例。

### 6. mujoco (fork)

- **URL**: <https://github.com/Physical-Intelligence/mujoco>
- **简介**: MuJoCo（Multi-Joint dynamics with Contact）物理仿真器的 fork。

### 7. HuggingFace / LeRobot 模型

- **π₀.₅ base**: <https://huggingface.co/lerobot/pi05_base>
- **HuggingFace 博客**: <https://huggingface.co/blog/pi0>
- **π₀-FAST 文档**: <https://huggingface.co/docs/lerobot/pi0fast>
- **简介**: HuggingFace 团队的 PyTorch 移植版本，遵循原始参考代码以保持兼容。已集成入 LeRobot 框架。

### 8. 社区第三方复现

- **open-pi-zero** (allenzren): <https://github.com/allenzren/open-pi-zero> — 基于论文的 π₀ 独立复现（PaliGemma VLM 2.291B + action expert 0.315B，MoE-like 架构）
- **pi-zero-pytorch** (lucidrains): <https://github.com/lucidrains/pi-zero-pytorch> — π₀ 架构的 PyTorch 实现
- **contact-openpi** (gist-ailab): <https://github.com/gist-ailab/contact-openpi> — 基于 openpi 的接触感知扩展
- **openpi_robotv** (chancharikmitra): <https://github.com/chancharikmitra/openpi_robotv> — openpi 的机器人视觉扩展

---

## 五、PI 官网提供的 PDF 下载列表

| 文件 | URL |
|------|-----|
| π₀ 论文 | <https://www.pi.website/download/pi0.pdf> |
| π₀.₅ 论文 | <https://www.pi.website/download/pi05.pdf> |
| π*₀.₆ 论文 | <https://www.pi.website/download/pistar06.pdf> |
| π₀.₆ Model Card | <https://website.pi-asset.com/pi06star/PI06_model_card.pdf> |
| π₀.₇ 论文 | <https://www.pi.website/download/pi07.pdf> |
| FAST 论文 | <https://www.pi.website/download/fast.pdf> |
| Hi Robot 论文 | <https://www.pi.website/download/hirobot.pdf> |
| RTC 论文 | <https://www.pi.website/download/real_time_chunking.pdf> |
| Knowledge Insulation 论文 | <https://www.physicalintelligence.company/download/pi05_KI.pdf> |
| MEM 论文 | <https://www.pi.website/download/Mem.pdf> |
| RL Token 论文 | <https://www.pi.website/download/rlt.pdf> |

---

## 六、PI 官网 Research 项目页列表

| 项目 | URL |
|------|-----|
| FAST | <https://www.pi.website/research/fast> |
| Hi Robot | <https://www.pi.website/research/hirobot> |
| Real-Time Chunking (RTC) | <https://www.pi.website/research/real_time_chunking> |
| Knowledge Insulation | <https://www.pi.website/research/knowledge_insulation> |
| Human to Robot Transfer | <https://www.pi.website/research/human_to_robot> |
| MEM (Memory) | <https://www.pi.website/research/memory> |
| RL Token (RLT) | <https://www.pi.website/research/rlt> |

---

## 七、会议发表汇总

| 论文 | 会议 | 年份 |
|------|------|------|
| BridgeData V2 | CoRL 2023 | 2023 |
| DROID | RSS 2024 | 2024 |
| Octo | RSS 2024 | 2024 |
| FAST | RSS 2025 | 2025 |
| π₀.₅ | CoRL 2025 | 2025 |
| Knowledge Insulation | NeurIPS 2025 (spotlight) | 2025 |
| RTC | NeurIPS 2025 (poster) | 2025 |

---

## 八、模型演化路线（完整版）

```
前置研究（PI 成员在 Google/Berkeley/Stanford 时期）:
  BridgeData V2 (2023.08, CoRL 2023)
  └── DROID (2024.03, RSS 2024)
  └── Octo (2024.05, RSS 2024)

Physical Intelligence 正式成果:
  π₀ (2024.10)
    │
    ├── FAST tokenizer (2025.01, RSS 2025)
    │     └── π₀-FAST 模型
    │     └── π₀-FAST DROID（首个可泛化的 DROID 策略）
    │
    ├── OpenPI 开源 (2025.02) ─── 开源 π₀ + π₀-FAST 权重与代码
    │
    ├── Hi Robot (2025.02) ─── 分层推理（系统1 + 系统2）
    │
    ├── π₀.₅ (2025.04, CoRL 2025) ─── 开放世界泛化
    │
    ├── Knowledge Insulation (2025.05, NeurIPS 2025 spotlight) ─── 高效训练方法
    │     └── π₀.₅ + KI
    │
    ├── RTC (2025.06, NeurIPS 2025 poster) ─── 实时动作分块
    │     └── 训练时 RTC (2025.12 后续)
    │
    ├── π*₀.₆ / RECAP (2025.11) ─── 经验学习 + RL 自我改进
    │     └── π₀.₆ ─── π*₀.₆ 的基础模型（基于 π₀.₅ 构建）
    │
    ├── Moravec's Paradox & Robot Olympics (2025.12) ─── π₀.₆ 微调解决挑战任务
    │
    ├── Human Video Transfer (2025.12) ─── 人类视频迁移涌现
    │
    ├── MEM (2026.03) ─── 多尺度具身记忆（15 分钟长时任务）
    │
    ├── RL Token (2026.03/04) ─── 高精度在线 RL 微调
    │
    └── π₀.₇ (2026.04) ─── 可操控基础模型 + 涌现组合泛化能力
```

**核心趋势**: 从单一通用策略 (π₀) → 开放世界泛化 (π₀.₅) → 经验自我改进 (π*₀.₆) → 可操控+涌现能力 (π₀.₇)，同时在推理效率 (RTC, FAST)、记忆 (MEM)、精密操作 (RLT)、人类视频利用、挑战性任务（Robot Olympics）等方向全面推进。

---

## 九、公司背景

- **全名**: Physical Intelligence, Inc.（简称 PI 或 π）
- **总部**: 美国旧金山
- **成立**: 2024 年
- **创始人**:
  - **Karol Hausman**（CEO）: 前 Google DeepMind Staff Research Scientist、Stanford 兼职教授。SayCan、RT-1、RT-2、RT-X、PaLM-E 核心作者。
  - **Sergey Levine**（Chief Scientist）: 深度强化学习应用于机器人领域的奠基人之一，UC Berkeley RAIL Lab 创建者。
  - **Chelsea Finn**（Research Lead）: MAML 元学习论文作者（过去十年最有影响力的 AI 论文之一），Stanford IRIS Lab 负责人。
  - **Brian Ichter**: 前 Google DeepMind。RT-1、RT-2 核心作者。
  - **Lachy Groom**: 前 Stripe 高管。
  - **Adnan Esmail**: 联合创始人。
  - **Quan Vuong**: 联合创始人，前 Google DeepMind。
- **估值**: 约 $11B（据 TechCrunch 报道正以此估值融资约 $1B，较前轮 ~$5.6B 翻倍）
- **总融资**: 约 $1.07B
  - 2025 年底完成 $600M 融资轮
- **投资方**: Bond, Jeff Bezos, Khosla Ventures, Lux Capital, OpenAI, Redpoint Ventures, Sequoia Capital, CapitalG, Thrive Capital
- **使命**: 开发通用物理智能基础模型，使任何机器人能执行任何任务

---

## 十、相比 pi_docls.md 新增/补充的内容

1. **Moravec's Paradox and the Robot Olympics** 博客（2025-12-22）— 完全遗漏
2. **DROID 数据集**（2024-03, RSS 2024）— PI 核心成员主导的大规模数据集
3. **Octo 模型**（2024-05, RSS 2024）— π₀ 的直接前身，PI 核心成员主导
4. **BridgeData V2**（2023-08, CoRL 2023）— PI 核心成员主导的早期数据集
5. **FAST 项目页** URL 补充: <https://www.pi.website/research/fast>
6. **会议发表信息补充**: FAST→RSS 2025, π₀.₅→CoRL 2025, KI→NeurIPS 2025 spotlight, RTC→NeurIPS 2025 poster
7. **PI 官网 PDF 下载完整列表**: 11 个 PDF 文件
8. **PI 官网 Research 项目页完整列表**: 7 个项目页
9. **创始人详细背景与前置研究谱系**: SayCan → RT-1 → RT-2 → RT-X → Octo → AutoRT → π₀
10. **各论文的详细作者列表、具体作者单位**补充完善
11. **openpi 仓库更新记录补充**: 2025-09 PyTorch 支持、pi05 发布、DROID 训练等
12. **HuggingFace π₀-FAST 文档页** URL 补充: <https://huggingface.co/docs/lerobot/pi0fast>
