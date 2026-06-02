# Pi0 架构详解

> 本文档专注于 **π₀（Pi0）** 的实现思路，按数据流向组织：
>
> **模型架构 → 输入 → 前向传播 → Loss → 推理采样 → 输出**

---

## 一、模型架构

Pi0 是一个 **Vision-Language-Action（VLA）** 模型，整体由 **3 个网络模块** 拼成：

```
┌──────────────────────────────────────────────────────────────┐
│  ① 视觉编码器  SigLIP ViT (So400m/14)                          │
│     224×224 RGB  →  256 个视觉 token (×N_cam)                 │
│                       │                                       │
│                       ▼                                       │
│  ② 双流 Transformer 骨干（两个 expert 逐层联合注意力）            │
│     ┌─────────────────────────────┬────────────────────────┐ │
│     │ Expert 1：Gemma 2B 主干      │ Expert 2：Action       │ │
│     │ （来自 PaliGemma 预训练）    │  Expert 300M（新增）    │ │
│     │ width=2048, depth=18         │ width=1024, depth=18   │ │
│     │ 输入 token：图像 + 文本       │ 输入 token：state + 带噪动作│ │
│     │ Norm：RMSNorm                │ Norm：RMSNorm           │ │
│     │                              │  (时间步通过 MLP 与动作  │ │
│     │                              │   token 拼接融合，无 adaRMS)│ │
│     └─────────────────────────────┴────────────────────────┘ │
│         每一层 attention 时，两个 expert 的 Q/K/V 在序列维拼接   │
│         一起做 softmax → 跨流信息融合                          │
│                       │                                       │
│                       ▼                                       │
│  ③ 动作输出头（线性投影）                                      │
│     Expert 2 最后一层的 1024-dim 输出  →  action_dim 维速度场  │
└──────────────────────────────────────────────────────────────┘
```

### 1.1 术语澄清：两个 expert 是什么

| 名词 | 是什么 | 数量 |
|------|--------|------|
| **PaliGemma** | 一个外部的 VLM = SigLIP + Gemma 2B | 1 套 |
| **Expert 1**（"主干"） | PaliGemma 内的那个 Gemma 2B（被 Pi0 论文称为 expert） | 1 个 |
| **Expert 2**（"Action Expert"） | Pi0 论文新加的 300M Gemma 风格小网络 | 1 个 |
| **expert 总数** | Expert 1 + Expert 2 | **共 2 个** |

> "双流" 指 Expert 1 和 Expert 2 是**两套独立的参数**（各自有自己的 Q/K/V、FFN、Norm），但每一层 attention 时把双方 token 拼起来一起算，实现信息互通。**全模型里只有这 2 个 expert，不存在"两套 Gemma"或"多个 Action Expert"。**

### 1.2 三大设计选择

| 设计 | 作用 |
|------|------|
| **双 expert + 联合注意力** | Expert 1 复用 PaliGemma 预训练权重负责"理解"（视觉+语言）；Expert 2 从零训负责"生成"（动作 chunk）；逐层把两边 Q/K/V 拼起来做一次 attention，让动作生成能看到全部感知信息 |
| **Flow Matching（连续动作）** | 输出连续值的速度场而非离散 token，比自回归方式快，天然适配机器人连续控制 |
| **State 作为连续 token** | 机器人当前状态经线性投影成 1 个 Action Expert token，与动作 token 一起送入 Expert 2；prompt 里只保留任务文本 |

### 1.3 Pi0 与 Pi0.5 的核心区别（设计层面）

- **State 注入位置**：Pi0 把 state 作为独立的连续 token 喂给 Expert 2；Pi0.5 把 state 离散化后拼进文本 prompt，由 Expert 1 处理
- **时间步注入方式**：Pi0 把 timestep 拼到动作 token 上过 MLP，**只在进入骨干前融合一次**；Pi0.5 把 timestep 作为 **adaRMSNorm 的条件**注入 Expert 2 的每一层
- **Prompt 长度**：Pi0 是 48 tokens（只放任务文本），Pi0.5 扩到 200 tokens 以容纳离散化的 state 字符串

---

## 二、输入：从机器人观测到模型张量

### 2.1 原始观测格式

机器人侧采集到的原始输入（以 ALOHA 双臂为例）：

| 字段 | 形状 | 含义 |
|------|------|------|
| `cam_high` | uint8 [H, W, 3] | 顶视相机图像 |
| `cam_left_wrist` | uint8 [H, W, 3] | 左腕相机图像 |
| `cam_right_wrist` | uint8 [H, W, 3] | 右腕相机图像 |
| `state` | float32 [14] | 当前关节/夹爪状态 |
| `prompt` | str | 任务文本，如 "Fold the towel…" |

### 2.2 输入变换链（数据流向）

```
原始观测 dict
   │
   ▼ ① 键名映射（机器人特异）
       cam_high        → base_0_rgb
       cam_left_wrist  → left_wrist_0_rgb
       cam_right_wrist → right_wrist_0_rgb
       state（14 维）  → state
   │
   ▼ ② 注入默认 prompt（若客户端未传 prompt）
   │
   ▼ ③ Normalize（Z-score 归一化）
       state, actions 用 (x - mean) / std 映射到归一化空间
       图像不参与归一化（在 model 入口单独 *2/255-1）
   │
   ▼ ④ Resize 图像到 224×224 + pad
   │
   ▼ ⑤ + ⑥ 两件事各取所需（互不影响）
   │
   ├── ⑤ TokenizePrompt：仅编码任务文本（state 不参与）
   │     SentencePiece 分词（任务文本 + "\n" 作 "start-of-answer"）
   │     约 20–30 个 token，pad 到 48
   │     产物：tokenized_prompt (int32[48]) ← 给 Expert 1 用
   │
   └── ⑥ PadStatesAndActions：state 张量末尾补 0
         state(14 维) → state(32 维)，最后 18 维全 0
         actions(14 维) → actions(32 维)，最后 18 维全 0
         产物：state (float32[32]) ← **真实进入模型**（喂给 Expert 2 的线性投影）
                 actions (float32[50, 32]) ← 模型输出/loss 计算用
   │
模型可接受的张量字典
```

### 2.3 进入模型的 4 类张量

经过变换链后，模型实际看到的输入是：

| 张量 | 形状 | 说明 |
|------|------|------|
| `images` | dict[str, float32[H, W, 3]] | 多视角图像，[-1, 1] |
| `image_masks` | dict[str, bool] | 该视角是否有效 |
| `state` | float32[32] | 归一化、补零后的 state，**Pi0 会真实读取它** |
| `tokenized_prompt` | int32[48] | prompt token 序列，**纯任务文本，不含 state** |
| `tokenized_prompt_mask` | bool[48] | 有效 token 掩码 |

---

## 三、前向传播：从张量到速度场

模型把输入序列拆成**前缀（prefix）+ 后缀（suffix）** 两段，分别送给 **Expert 1（Gemma 2B 主干）** 和 **Expert 2（Action Expert 300M）**，再通过联合注意力融合。

### 3.1 前缀（送给 Expert 1）

```
图像 ×N (256 × N tokens)        语言 (~30 tokens)
        │                              │
        ▼                              ▼
   SigLIP ViT 编码              PaliGemma Embedding 表
        │                              │
        └─────────── 拼接 ──────────────┘
                       │
                       ▼
       前缀序列：~800 个 2048 维 token
```

前缀里所有 token 之间**双向注意力**，互相能看到。

### 3.2 后缀（送给 Expert 2）

后缀由 **1 个 state token + 50 个动作 token** 组成，共 51 个 1024 维 token：

```
连续 state (32 维)         带噪动作 x_t (50 步 × 32 维)     timestep t ∈ [0, 1]
       │                            │                              │
       ▼ 线性投影                  ▼ 线性投影                     ▼ sin-cos 位置编码
[1 个 state token]          [50 个动作 token, 1024]          time_emb (1024)
                                    │                              │
                                    │  ◄────── tile 到 50 步 ──────┘
                                    ▼
                          沿特征维拼接 → (50, 2048)
                                    │
                                    ▼ 两层 MLP + swish
                          [50 个动作+时间融合 token, 1024]
       │                            │
       └────────── 拼接 ────────────┘
                       │
                       ▼
       后缀序列：1 + 50 = 51 个 1024 维 token
```

动作 token 之间**互相能看**，且**能看到全部前缀 + state token**；但前缀**看不到 state 与动作**（防止训练时动作真值泄露）。

### 3.3 时间步注入方式（Pi0 vs Pi0.5）

> Pi0 把时间编码与动作 token **沿特征维拼接后过 MLP，只在进入 18 层骨干之前融合一次**；Pi0.5 把时间作为 **adaRMSNorm 的条件**，在 Expert 2 的**每一层**都被注入。
>
> 因此 Pi0 的 Expert 2 仍然用普通 RMSNorm，无 adaRMS 条件，无 gate 门控。

### 3.4 双 Expert 联合注意力（18 层重复）

每一层 Block 的数据流：

```
   Expert 1（处理前缀）          Expert 2（处理后缀）
[前缀 token, 2048-dim]        [state+动作 token, 1024-dim]
        │                              │
     RMSNorm                       RMSNorm  ← Pi0：普通 RMSNorm（无时间条件）
        │                              │
        ▼                              ▼
 Q,K,V 投影 (Expert 1 权重)   Q,K,V 投影 (Expert 2 权重)
        │                              │
        └────────── 沿序列维拼接 ──────┘
                       │
                       ▼
               RoPE + GQA Attention
               （受 attn_mask 约束）
                       │
                 按段切回各自维度
        ┌──────────┴──────────┐
        ▼                      ▼
   [前缀输出]                 [后缀输出]
        │                      │
   普通残差 (x + y)          普通残差 (x + y)  ← Pi0 无 gate 门控
        │                      │
   FFN (Expert 1 权重)      FFN (Expert 2 权重)
        │                      │
        ▼                      ▼
   下一层                  下一层
```

整个过程中：Expert 1 和 Expert 2 **共用**一次 softmax（联合注意力实现跨流信息交流），但 Q/K/V 投影、FFN、Norm 等**全部参数都是各自独立**的。

### 3.5 注意力掩码（防止动作真值泄露）

```
谁能看谁:
                 前缀 (图像+语言)   state (1)    动作 (50)
前缀（双向）            ✓              ✗            ✗
state（看前缀）         ✓              ✓            ✗
动作（看全部条件）       ✓              ✓        ✓（互相可见）
```

### 3.6 输出：速度场

```
Expert 1 最终输出 (~800 × 2048-dim)        Expert 2 最终输出 (51 × 1024-dim)
        │                                          │
        ▼                                          ▼ 取末尾 50 个（跳过 state token 输出）
    🗑️ 直接丢弃                                   ▼ 线性投影
                                                  速度场 v_t (50 × 32)
```

> 这里输出的**不是关节角度本身**，而是 Flow Matching 中的"速度场"——指引当前噪声样本 $x_t$ 朝着真实动作 $x_0$ 移动的方向向量。

### 3.7 关于 Expert 1 的输出与梯度（重要澄清）

Expert 1 最终一层的 ~800 个输出向量**被直接丢弃**，不计算 loss、不输出任何东西。乍看上去 Expert 1 是不是"白跑了"？**不是**——它的价值不在最终输出，而在 attention 过程中给 Expert 2 提供的 K/V。

#### 前向：Expert 1 每一层都给 Expert 2 提供 K/V

```
                Layer 1（双 Expert 联合 attention）
   ┌─────────────────────────────────────────────────────┐
   │  Expert 1 算出 Q₁ᴱ¹, K₁ᴱ¹, V₁ᴱ¹                     │
   │  Expert 2 算出 Q₁ᴱ², K₁ᴱ², V₁ᴱ²                     │
   │  K, V 沿序列维拼起来：K = [K₁ᴱ¹ ‖ K₁ᴱ²], V 同理      │
   │  attention：Expert 2 的 Q₁ᴱ² 会查 K₁ᴱ¹  ◄────────────┼── 关键点
   └─────────────────────────────────────────────────────┘
                  │
                  ▼ Expert 1 当层输出 → 下一层的输入
   ┌─────────────────────────────────────────────────────┐
   │  Layer 2: Expert 1 更新表征 → 算出 K₂ᴱ¹, V₂ᴱ¹        │
   │           Expert 2 的 Q₂ᴱ² 会查 K₂ᴱ¹                 │
   └─────────────────────────────────────────────────────┘
                  ...
   ┌─────────────────────────────────────────────────────┐
   │  Layer 18:  Expert 1 最终输出 prefix_out  →  🗑️ 丢弃 │
   │             Expert 2 最终输出 suffix_out             │
   │                  ↓ 取末 50 个                        │
   │             线性投影  →  v_t                         │
   │                  ↓                                   │
   │             Loss = MSE(v_t, u_t)                     │
   └─────────────────────────────────────────────────────┘
```

每一层都重复同样的模式：**Expert 1 提供 K/V，Expert 2 用 Q 去查**。Loss 因此间接依赖 Expert 1 全部 18 层的所有参数。

#### 反向：梯度沿 K/V 链路从 Expert 2 流回 Expert 1

```
        Loss
         │
         ▼
       v_t
         │
         ▼ (梯度)
      Expert 2 第 18 层输出
         │
         ▼ 反向穿过 attention
   ┌─────┴───────┐
   ▼              ▼
Expert 2          Expert 1 第 18 层的 K, V   ◄── 梯度流到 Expert 1
参数 (FFN/Q)         │
                    ▼ 反向穿过第 18 层的 FFN + Norm
                Expert 1 第 18 层的 Q, K, V 投影参数   ◄── 这里更新
                    │
                    ▼ 反向传到第 17 层输出
                ... (一路反向到第 1 层) ...
                    │
                    ▼
                SigLIP 输出
                    │
                    ▼ (梯度继续传)
                SigLIP 27 层 Transformer + Patch Embedding 也更新
```

只要在前向计算中**被用到**，反向传播就能把梯度送回去；变量的最终值有没有被使用并不重要。

#### 结论

- ✅ Expert 1 的**最终输出向量**被丢弃
- ✅ Expert 1 的**全部 18 层参数**（包括 Q/K/V 投影、FFN、Norm、embedding）仍然参与梯度更新
- ✅ 连上游的 SigLIP 27 层也会一起更新（全量微调模式下）

这两件事并不矛盾——Expert 1 的价值不在"它输出了什么"，而在"它在 attention 中为 Expert 2 提供了什么"。

---

## 四、Loss：Flow Matching 训练目标

### 4.1 直观理解

Flow Matching 把"从噪声生成动作"建模成一条**直线轨迹**：

```
t=0 (干净动作)               t=1 (纯噪声)
     │←─────────────────────────│
     ●─────────────────────────►●
   actions                    noise
        速度场 u_t = noise - actions
            （时刻无关，恒为常量）
```

训练时让模型学会**预测这条直线的方向**：从任意时刻 $t$ 的混合点 $x_t$ 出发，输出应当是 $u_t = \varepsilon - a$。

### 4.2 训练时一个 batch 的数据流

```
真实动作 actions (B, 50, 32)
      │
      ├──► 采样噪声 ε ~ N(0, I)
      │
      ├──► 采样时间 t ~ Beta(1.5, 1)（偏向 t≈1，难度更高）
      │
      ▼
  线性插值：x_t = t·ε + (1-t)·actions     （加噪后的动作）
  目标速度： u_t = ε - actions             （真值方向）
      │
      ▼
  模型前向：v_t = Pi0(x_t, t, observation)
      │
      ▼
  Loss = MSE(v_t, u_t)，沿 action_dim 取均值
```

### 4.3 Loss 公式

$$
\mathcal{L} = \mathbb{E}_{t,\varepsilon,(o,a)} \big\| v_\theta(x_t,\, t,\, o) - (\varepsilon - a) \big\|^2
$$

其中 $x_t = t\varepsilon + (1-t)a$，$o$ 是观测（图像 + prompt + state），$a$ 是真实动作。

> Loss 在 32 维的全部输出上计算（包括 padding 出来的零分量）。padding 部分目标 $u_t = \varepsilon - 0$ 近似标准正态，模型容易拟合，对总体收敛影响可忽略。

### 4.4 训练时哪些参数会被更新

默认配置下 Pi0 是**全量微调**：SigLIP、Expert 1（Gemma 2B 主干）、Expert 2（Action Expert 300M）、各投影层与时间-动作 MLP 的全部权重都参与梯度更新。

> Expert 1 虽然最终输出会被丢弃，但梯度通过联合 attention 中的 K/V 链路传回到它每一层（详见 §3.7），所以 Expert 1 的参数仍然会被持续更新。

LoRA 模式（可选）下，只训练挂在 Attention Q/K/V 和 FFN 上的低秩矩阵 (A, B)，原始权重冻结。

---

## 五、推理：从噪声采样出动作

推理时不再"学速度场"，而是从纯噪声出发，沿模型预测的速度场反向积分回干净动作。

### 5.1 推理时的数据流

```
机器人当前观测 (images, state, prompt)
      │
      ▼ 输入变换链（同训练，但不破坏 state、不加图像增广）
      ▼
  前缀 token（图像 + 纯任务文本）
      │
      ▼ ① 一次前向，缓存 prefix KV
      ▼
  prefix_kv_cache (整段推理只算一次)
      │
      ▼ ② Euler 积分（默认 10 步，t: 1 → 0）
      │
       ┌───────── 初始化：x = randn(50, 32) ─────────┐
       │                                              │
       │      while t > 0:                            │
       │          后缀 = [state token]                │
       │                + MLP(投影(x) ⊕ time_emb(t))   │
       │          v = Expert 2(后缀, kv_cache=prefix) │
       │          v = 取末 50 个动作 token的输出        │
       │          x = x + (-0.1) * v        ◄─ Euler 步
       │          t = t - 0.1                         │
       │                                              │
       └──────────────────────────────────────────────┘
                       │
                       ▼
       动作序列 x_0 (50, 32) —— 模型输出的归一化动作
                       │
      ▼ ③ 输出变换链
      │    Unnormalize (反归一化回原始物理量纲)
      │    截取前 N 维（机器人实际控制维度，如 ALOHA 14 维）
      ▼
  最终动作 (50 步 × N 维) → 机器人执行
```

> Pi0 后缀首位是 state token，其输出不参与生成，因此每步只取末尾 50 个动作 token 过线性投影得到速度场。

### 5.2 推理效率优化（设计要点）

- **prefix KV 缓存**：图像 + 语言这部分占了序列的 ~94%，但整个采样过程中完全不变 → Expert 1 只前向一次，10 步采样都复用同一份 KV
- **Expert 2 独立计算**：每步只把后缀的 51 个 token 走一遍 18 层 attention（参数量 300M，远小于 Expert 1 的 2B），延迟低
- **Action Chunk**：一次出 50 步动作而非 1 步，机器人执行端可以做异步推理 + 动作集成（HATO/RTG 等），用一次推理覆盖多个控制周期

---

## 六、输出：从模型张量回到机器人指令

### 6.1 输出变换链

```
模型输出 actions (50, 32)，归一化空间
        │
        ▼ Unnormalize（Z-score 反变换）
        │   x_real = x * std + mean
        │   把归一化值拉回真实物理量纲（弧度、米/秒等）
        │
        ▼ 机器人特异截取
        │   只取前 N 维（如 ALOHA 取 14、LIBERO 取 7）
        │   丢弃 padding 的尾部维度
        │
        ▼
最终动作 (50, N)：每一行就是一帧机器人指令
```

### 6.2 几种常见机器人的维度切片

| 机器人 | 维度 | 切片含义 |
|--------|------|----------|
| ALOHA | 14 | 左臂 7 + 右臂 7（含夹爪角度映射） |
| LIBERO | 7 | 单臂末端 6 + 夹爪 1 |
| R1 Pro Chassis | 23 | 左臂 7 + 右臂 7 + 夹爪×2 + 躯干 4 + 底盘 3 |

---

## 七、整体数据流总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            训练 Pipeline                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  LeRobot 数据集（episode 序列）                                            │
│     │                                                                     │
│     ▼ 数据加载 + repack                                                    │
│     ▼ 机器人适配（键名映射 + state/actions 重组）                            │
│     ▼ Z-score 归一化                                                       │
│     ▼ Resize 224 + Tokenize（纯任务文本，pad 到 48） + Pad 32                │
│     │                                                                     │
│     ▼ 加噪：x_t = t·ε + (1-t)·a    ;    u_t = ε - a                       │
│     │                                                                     │
│     ▼ SigLIP → 视觉 token   |   PaliGemma embed → 文本 token               │
│     ▼ state → 1 个 state token                                            │
│     ▼ 动作 token 与 sin-cos 时间编码拼接 → MLP → 50 个融合 token             │
│     ▼ 双 Expert 18 层联合注意力（两边均为普通 RMSNorm，普通残差）             │
│     ▼ 取后缀末 50 个 → 线性投影 → v_t                                       │
│     │                                                                     │
│     ▼ Loss = MSE(v_t, u_t)                                                │
│     ▼ AdamW + Cosine LR + Gradient Clip(1.0)                              │
│     ▼ EMA(0.99) + Orbax Checkpoint                                        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                            推理 Pipeline                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  机器人观测（多路图像 + state + prompt）                                    │
│     │                                                                     │
│     ▼ 输入变换链（同训练，无增广）                                          │
│     │                                                                     │
│     ▼ prefix 前向 1 次 → 缓存 KV                                           │
│     │                                                                     │
│     ▼ x = randn(50, 32), t = 1.0                                          │
│     ▼ for 10 steps:                                                       │
│         suffix = [state token] + MLP(投影(x) ⊕ time_emb(t))                │
│         v = Expert 2(suffix, kv_cache=prefix_kv)                          │
│         v = 取末 50 个 → 线性投影                                          │
│         x = x - 0.1·v ; t -= 0.1                                          │
│     │                                                                     │
│     ▼ Unnormalize + 截取前 N 维                                           │
│     ▼ 50 步动作 chunk → 机器人异步执行（配合动作集成）                       │
└──────────────────────────────────────────────────────────────────────────┘
```
