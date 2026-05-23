# Physical Intelligence (π) 研究成果完整总览

> Physical Intelligence 官网: <https://www.pi.website/>
>
> 旧域名: <https://www.physicalintelligence.company/>
>
> GitHub 组织: <https://github.com/Physical-Intelligence>
>
> X/Twitter: <https://x.com/physical_int>

---

## 一、论文 & 技术报告（由新到旧）

### 1. π₀.₇: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

- **日期**: 2026-04-16（v1），2026-04-24（v2）
- **arXiv**: <https://arxiv.org/abs/2604.15483>
- **博客**: <https://www.pi.website/blog/pi07>
- **PDF**: <https://www.pi.website/download/pi07.pdf>
- **作者**: Physical Intelligence（Bo Ai, Ashwin Balakrishna, Kevin Black, Chelsea Finn, Danny Driess, Karol Hausman, Sergey Levine, Karl Pertsch 等 87 位共同作者）
- **简介**: ~5B 参数 VLA 模型（Gemma3 4B VLM backbone + 860M action expert）。支持丰富多模态 prompt（语言指令、子目标图像、视频、子任务元数据等）。展现组合泛化与零样本跨机体迁移能力（如在 UR5e 上零样本折叠衣物，尽管该机体无训练数据）。在匹配对比中，π₀.₇ 任务进度 85.6%、成功率 80%，接近专家人类遥操作者（90.9% / 80.6%）。采用训练时 RTC（Real-Time Chunking）实现平滑动作生成。

### 2. RL Token: Bootstrapping Online RL with Vision-Language-Action Models

- **日期**: 2026-04-24（arXiv），2026-03-19（博客首发）
- **arXiv**: <https://arxiv.org/abs/2604.23073>
- **项目页**: <https://www.pi.website/research/rlt>
- **PDF**: <https://www.pi.website/download/rlt.pdf>
- **作者**: Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke（Physical Intelligence）
- **简介**: 从 VLA 模型中提取"RL Token"——一个紧凑的内部表征摘要，作为小型 actor-critic 的输入。使大型 VLA 可在 15 分钟至数小时内通过在线 RL 精调精密操作任务（拧 M3 螺丝、扎线带、插以太网线、插电源线）。在部分任务上速度甚至超越人类遥操作。基于 π₀.₆ 模型实现。

### 3. MEM: Multi-Scale Embodied Memory for Vision Language Action Models

- **日期**: 2026-03-04（v1），2026-03-08（v2）
- **arXiv**: <https://arxiv.org/abs/2603.03596>
- **项目页**: <https://www.pi.website/research/memory>
- **PDF**: <https://www.pi.website/download/Mem.pdf>
- **作者**: Marcel Torne, Karl Pertsch 等 17 位作者（Physical Intelligence）
- **简介**: 双尺度记忆架构——短时视频记忆（ViT encoder 压缩最近几秒画面）+ 长时文本记忆（chain-of-thought 语言摘要记录已完成的子任务）。使机器人能执行长达 15 分钟的连续任务（厨房清洁、做三明治、整理杂货）。集成于 π₀.₆ VLA（初始化自 Gemma 3-4B）。还展现了上下文内适应能力（如抓取失败后自动调整策略）。

### 4. Emergence of Human to Robot Transfer in Vision-Language-Action Models

- **日期**: 2025-12-27（arXiv），2025-12-16（博客）
- **arXiv**: <https://arxiv.org/abs/2512.22414>
- **项目页**: <https://www.pi.website/research/human_to_robot>
- **作者**: Simar Kareer, Karl Pertsch, James Darpinian, Judy Hoffman, Danfei Xu, Sergey Levine, Chelsea Finn, Suraj Nair（Physical Intelligence）
- **简介**: 发现当 VLA 在足够多样的机器人数据上预训练后，从人类视频到机器人操作任务的迁移能力会"涌现"出来。提出简单的协同训练方案（co-training recipe），分析表明多样化预训练产生了跨具身体的不可知表征。

### 5. π*₀.₆: a VLA That Learns From Experience

- **日期**: 2025-11-18（v1），2025-11-19（v2）
- **arXiv**: <https://arxiv.org/abs/2511.14759>
- **博客**: <https://www.pi.website/blog/pistar06>
- **PDF**: <https://www.pi.website/download/pistar06.pdf>
- **Model Card PDF**: <https://website.pi-asset.com/pi06star/PI06_model_card.pdf>
- **作者**: Physical Intelligence（Ali Amin, Ashwin Balakrishna, Kevin Black, Chelsea Finn, Sergey Levine 等 55 位共同作者）
- **简介**: 提出 RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）方法，通过优势条件化使 VLA 能从自主部署经验和人类纠正中进行强化学习自我改进。π*₀.₆ 在困难任务上吞吐量翻倍以上（洗衣折叠、制作浓缩咖啡），失败率降低约 50%，可连续自主部署 13 小时。π₀.₆ 基于 π₀.₅ 构建。

### 6. Real-Time Execution of Action Chunking Flow Policies (RTC)

- **日期**: 2025-06-09（v1），2025-12-05（v2）
- **arXiv**: <https://arxiv.org/abs/2506.07339>
- **项目页**: <https://www.pi.website/research/real_time_chunking>
- **PDF**: <https://www.pi.website/download/real_time_chunking.pdf>
- **代码**: <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **作者**: Kevin Black 等（Physical Intelligence）
- **会议**: NeurIPS 2025 (poster)
- **简介**: 无需重训练的推理时算法，使任何 diffusion/flow-based VLA 实现实时平滑异步执行。核心思想是在执行当前动作块的同时生成下一个，"冻结"已确定要执行的动作并"修复"其余部分。即使注入 300ms+ 人工延迟仍能完成划火柴、插网线等精密任务。2025-12-08 发布了训练时版本 RTC 的后续论文，用于 π₀.₆ 制作咖啡的演示。

### 7. Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better

- **日期**: 2025-05-29
- **arXiv**: <https://arxiv.org/abs/2505.23705>
- **项目页**: <https://www.pi.website/research/knowledge_insulation>
- **PDF（PI 官网）**: <https://www.physicalintelligence.company/download/pi05_KI.pdf>
- **OpenReview**: <https://openreview.net/forum?id=cb0xbZ3APM>
- **作者**: Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z. Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, Sergey Levine（Physical Intelligence）
- **会议**: NeurIPS 2025 (spotlight)
- **简介**: 通过"知识绝缘"——阻断 action expert（flow matching/diffusion）的梯度回传到 VLM backbone，同时用 FAST 离散 token + web data 联合训练 backbone，使其保留语义知识。训练时间减少 66%，泛化能力更好。形式化了 π₀.₅ 中使用的方法，并扩展为更精炼的单阶段训练配方。

### 8. π₀.₅: a Vision-Language-Action Model with Open-World Generalization

- **日期**: 2025-04-22
- **arXiv**: <https://arxiv.org/abs/2504.16054>
- **博客**: <https://www.physicalintelligence.company/blog/pi05>
- **作者**: Physical Intelligence
- **简介**: 基于 π₀ 的升级模型，通过异构多任务协同训练（多机器人数据、高层语义子任务预测、物体检测、web 数据）实现开放世界泛化。使用混合多模态样本（图像观测、语言指令、物体检测、语义子任务预测、低层动作）。能在训练中从未见过的全新家庭环境中执行厨房清洁、卧室整理等任务。

### 9. Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

- **日期**: 2025-02-26（v1），2025-07-15（v2）
- **arXiv**: <https://arxiv.org/abs/2502.19417>
- **项目页**: <https://www.pi.website/research/hirobot>
- **PDF**: <https://www.pi.website/download/hirobot.pdf>
- **作者**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner 等（Physical Intelligence）
- **简介**: 分层架构——VLM 作为"系统2"（慢思考）分解复杂指令为简单子步骤，π₀ 作为"系统1"（快反应）执行低层动作。类似 LLM 的"思维链"，允许机器人"自言自语"进行推理。支持实时人类反馈纠正。在单臂、双臂、双臂移动机器人三个平台上测试，能完成清洁桌面、制作三明治、购物等复杂任务。

### 10. FAST: Efficient Action Tokenization for Vision-Language-Action Models

- **日期**: 2025-01-16
- **arXiv**: <https://arxiv.org/abs/2501.09747>
- **作者**: Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine（Physical Intelligence）
- **简介**: 提出基于离散余弦变换（DCT）的频域动作序列 tokenizer（Frequency-space Action Sequence Tokenization）。解决传统逐维度逐时间步离散化在高频灵巧控制下完全失效的问题。使自回归 VLA 训练速度提升 5 倍。还发布了 FAST+——基于 100 万条真实机器人动作轨迹训练的通用 tokenizer。

### 11. π₀: A Vision-Language-Action Flow Model for General Robot Control

- **日期**: 2024-10-31
- **arXiv**: <https://arxiv.org/abs/2410.24164>
- **博客**: <https://www.pi.website/blog/pi0>（旧域名: <https://physicalintelligence.company/blog/pi0>）
- **作者**: Kevin Black, Noah Brown, Danny Driess, Michael Equi, Chelsea Finn, Karol Hausman, Brian Ichter, Sergey Levine 等（Physical Intelligence）
- **简介**: Physical Intelligence 的首个通用机器人策略基础模型。基于大规模预训练和 flow matching 动作生成，50Hz 输出连续低层电机命令。在 7 个机器人平台、68 个独立任务上训练（约 10,000 小时演示数据 + OXE 开源数据集）。能折叠衣物、收拾餐桌、打包杂货、组装盒子、取回物品等。

---

## 二、博客文章 & 公告（由新到旧）

### 1. π₀.₇: a Steerable Model with Emergent Capabilities

- **日期**: 2026-04-16
- **URL**: <https://www.pi.website/blog/pi07>
- **简介**: π₀.₇ 模型发布博客。展示装空气炸锅、折叠牛仔裤、用 Windex 清洁玻璃等演示。介绍可操控 prompt 框架与涌现能力。

### 2. Precise Manipulation with Efficient Online RL (RL Token)

- **日期**: 2026-03-19
- **URL**: <https://www.pi.website/research/rlt>
- **简介**: RL Token 方法首发博客，介绍如何在几小时甚至几分钟内通过在线 RL 精调精密操作。

### 3. MEM: VLAs with Long and Short-Term Memory

- **日期**: 2026-03（约 2026-03-03）
- **URL**: <https://www.pi.website/research/memory>
- **简介**: 多尺度具身记忆（MEM）方法博客，使机器人具备 15 分钟长时任务执行能力。

### 4. The Physical Intelligence Layer（合作伙伴）

- **日期**: 2026-02-24
- **URL**: <https://www.pi.website/blog/partner>
- **简介**: 介绍与合作伙伴 Weave、Ultra 在真实仓库中的部署。模型在实际客户仓库中打包真实订单。展示 π₀→π₀.₅→π₀.₆ 逐代在智能、吞吐量和可靠性上的显著提升。阐述"物理智能层"的愿景——像 LLM API 一样让任何人使用机器人基础模型。

### 5. Emergence of Human to Robot Transfer

- **日期**: 2025-12-16
- **URL**: <https://www.pi.website/research/human_to_robot>
- **简介**: 探索如何通过规模化使人类视频到机器人操作的迁移能力涌现。

### 6. Real-Time Action Chunking (RTC)

- **日期**: 首发约 2025-06，更新于 2025-12-08
- **URL**: <https://www.pi.website/research/real_time_chunking>
- **简介**: 实时动作分块方法博客。2025-12-08 更新发布了训练时 RTC 版本后续论文，用于 π₀.₆ 制作咖啡演示。

### 7. π*₀.₆: A VLA that Learns from Experience

- **日期**: 2025-11-17
- **URL**: <https://www.pi.website/blog/pistar06>
- **简介**: RECAP 方法与 π*₀.₆ / π₀.₆ 模型发布博客。介绍如何通过经验学习和人类纠正进行 RL 自我改进。

### 8. Knowledge Insulation: VLAs that Train Fast, Run Fast, and Generalize Better

- **日期**: 2025-05（约与论文同步）
- **URL**: <https://www.pi.website/research/knowledge_insulation>
- **简介**: 知识绝缘方法博客，介绍如何在不损失预训练知识的前提下高效训练 VLA。

### 9. π₀.₅: A VLA with Open-World Generalization

- **日期**: 2025-04
- **URL**: <https://www.physicalintelligence.company/blog/pi05>
- **简介**: π₀.₅ 模型发布博客，介绍开放世界泛化能力。

### 10. Hi Robot: Teaching Robots to Listen and Think Harder

- **日期**: 2025-02-26
- **URL**: <https://www.pi.website/research/hirobot>
- **简介**: 分层推理 + 人机交互博客。介绍"系统1 + 系统2"架构。

### 11. Open Sourcing π0

- **日期**: 2025-02-04
- **URL**: <https://www.pi.website/blog/openpi>
- **简介**: 宣布开源 π₀ 和 π₀-FAST 的权重与代码。发布 openpi 仓库，含预训练 checkpoint、ALOHA/DROID 微调模型、推理与微调代码。HuggingFace 同步发布 PyTorch 移植版。

### 12. FAST: Efficient Robot Action Tokenization

- **日期**: 2025-01-16
- **URL**: 随论文发布（参见 arXiv: <https://arxiv.org/abs/2501.09747>）
- **简介**: FAST tokenizer 发布，训练速度提升 5 倍。

### 13. π₀: Our First Generalist Policy

- **日期**: 2024-10-31
- **URL**: <https://www.pi.website/blog/pi0>（旧域名: <https://physicalintelligence.company/blog/pi0>）
- **简介**: Physical Intelligence 首个通用策略模型发布博客。

---

## 三、开源代码 & 模型

### 1. openpi（主仓库）

- **URL**: <https://github.com/Physical-Intelligence/openpi>
- **简介**: Physical Intelligence 的核心开源仓库（JAX 实现）。包含：
  - **π₀ 模型**: flow-based VLA，预训练于 10,000+ 小时机器人数据
  - **π₀-FAST 模型**: 基于 FAST tokenizer 的自回归 VLA，语言跟随性能略好，但推理成本约 4-5 倍
  - **π₀.₅ 模型**: 增强开放世界泛化（Knowledge Insulation 训练）
  - 多个预训练 checkpoint（base 模型 + ALOHA/DROID 等平台的 expert 微调模型）
  - 远程推理支持（WebSocket 连接，可使用 off-robot GPU）
  - 无需机器人的推理测试脚本（生成随机观测输入）
  - Docker 安装支持
  - 2025-06 新增：使用 openpi 在完整 DROID 数据集上训练 VLA 的说明

### 2. real-time-chunking-kinetix

- **URL**: <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **简介**: RTC 论文的 Kinetix 仿真实验代码。12 个高动态仿真任务基准。

### 3. HuggingFace / LeRobot 模型

- **π₀.₅ base**: <https://huggingface.co/lerobot/pi05_base>
- **HuggingFace 博客**: <https://huggingface.co/blog/pi0>
- **简介**: HuggingFace 团队的 PyTorch 移植版本，遵循原始参考代码以保持兼容。已集成入 LeRobot 框架。

### 4. Physical Intelligence GitHub 组织其他仓库

- **组织页**: <https://github.com/Physical-Intelligence>
- 其他仓库包括:
  - **pi-data-sharing**: 数据共享相关
  - **aloha**: ALOHA 机器人平台相关
  - **augmax**: 数据增强库
  - **mujoco**: MuJoCo 物理仿真器 fork

### 5. 社区第三方复现

- **open-pi-zero** (allenzren): <https://github.com/allenzren/open-pi-zero> — 基于论文的 π₀ 独立复现（PaliGemma VLM 2.291B + action expert 0.315B）
- **pi-zero-pytorch** (lucidrains): <https://github.com/lucidrains/pi-zero-pytorch> — π₀ 架构的 PyTorch 实现
- **contact-openpi** (gist-ailab): <https://github.com/gist-ailab/contact-openpi> — 基于 openpi 的接触感知扩展
- **openpi_robotv** (chancharikmitra): <https://github.com/chancharikmitra/openpi_robotv> — openpi 的机器人视觉扩展

---

## 四、模型演化路线

```
π₀ (2024.10)
  │
  ├── FAST tokenizer (2025.01)
  │
  ├── OpenPI 开源 (2025.02) ─── 开源 π₀ + π₀-FAST 权重与代码
  │
  ├── Hi Robot (2025.02) ─── 分层推理（系统1 + 系统2）
  │
  ├── π₀.₅ (2025.04) ─── 开放世界泛化
  │
  ├── Knowledge Insulation (2025.05) ─── 高效训练方法
  │
  ├── RTC (2025.06) ─── 实时动作分块
  │
  ├── π*₀.₆ / RECAP (2025.11) ─── 经验学习 + RL 自我改进
  │     └── π₀.₆ ─── π*₀.₆ 的基础模型（基于 π₀.₅ 构建）
  │
  ├── Human Video Transfer (2025.12) ─── 人类视频迁移涌现
  │
  ├── MEM (2026.03) ─── 多尺度具身记忆（15 分钟长时任务）
  │
  ├── RL Token (2026.03/04) ─── 高精度在线 RL 微调
  │
  └── π₀.₇ (2026.04) ─── 可操控基础模型 + 涌现组合泛化能力
```

**核心趋势**: 从单一通用策略 (π₀) → 开放世界泛化 (π₀.₅) → 经验自我改进 (π*₀.₆) → 可操控+涌现能力 (π₀.₇)，同时在推理效率 (RTC, FAST)、记忆 (MEM)、精密操作 (RLT)、人类视频利用等方向全面推进。

---

## 五、公司背景

- **总部**: 美国旧金山
- **估值**: 约 $11B（据报道正在以此估值融资约 $1B，较前轮 ~$5.6B 翻倍）
- **投资方**: Bond, Jeff Bezos, Khosla Ventures, Lux Capital, OpenAI, Redpoint Ventures, Sequoia Capital, CapitalG, Thrive Capital
- **2025 年底完成 $600M 融资轮**
- **使命**: 开发通用物理智能基础模型，使任何机器人能执行任何任务
