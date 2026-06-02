# 双流 Gemma 详解

## 先搞清楚三个组件的关系

```
PaliGemma（Google 预训练的多模态模型）
├── SigLIP (ViT)     ← 视觉编码器，27层 Encoder，双向 attention
└── Gemma (2B)       ← 语言模型，18层 Decoder，因果 attention
```

Pi0.5 在 PaliGemma 基础上**额外新增**了一个小 Gemma：
```
Pi0.5 完整模型
├── SigLIP (ViT)         ← 预训练好的，编码图像，单独先跑
├── Gemma 主干 (2B)      ← 来自 PaliGemma，接收视觉+语言token    ← 流1
└── 动作专家 Gemma (300M) ← 新加的，接收噪声动作token             ← 流2
```

**"双流"= 流1 + 流2 = 两个 Gemma Decoder**，SigLIP 不参与双流，它在上游先跑完。

## 完整数据流（时间顺序）

```
┌───────────────────────────────────────────────────────────────┐
│ 第一步：SigLIP 单独编码图像                                      │
│   图像(224×224×3) → Patch → 27层ViT → 线性投影 → 256×2048     │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 第二步：双流 Gemma 逐层联合运行 (18层)                                  │
│                                                                      │
│   流1 输入: [256个视觉token, L个语言token]  ← 维度2048               │
│   流2 输入: [H个噪声动作token]              ← 维度1024               │
│                                                                      │
│   每一层:                                                            │
│     流1 token → Q₁,K₁,V₁                                           │
│     流2 token → Q₂,K₂,V₂                                           │
│     K拼接=[K₁;K₂], V拼接=[V₁;V₂]                                   │
│     流1输出 = Attn(Q₁, K拼接, V拼接)  ← 能看到动作token              │
│     流2输出 = Attn(Q₂, K拼接, V拼接)  ← 能看到视觉+语言token         │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 第三步：流2 输出 → 预测去噪后的动作                               │
└───────────────────────────────────────────────────────────────┘
```

## 参数表详解

| 参数 | Gemma 主干 (2B) | 动作专家 (300M) | 说明 |
|------|----------------|----------------|------|
| width | 2048 | 1024 | token 向量维度。主干大是因为已预训练，要保持原始能力 |
| mlp_dim | 16384 | 4096 | FFN 中间维度。通常为 width 的 4~8 倍 |
| depth | 18 | 18 | **必须相同**——逐层配对做联合注意力 |
| num_heads | 8 | 8 | Q 的头数 |
| num_kv_heads | 1 | 1 | KV 头数（GQA：8个Q头共享1组KV） |
| head_dim | 256 | 256 | **必须相同**——联合注意力要求 Q/K/V 在同一空间 |

为什么 width 可以不同但 head_dim 必须相同？
- width 只影响各自内部的表示维度
- 联合注意力时，Q×K^T 要求 Q 和 K 维度一致 → head_dim 必须一致

## 每层 Block 的 4 步

### 1. RMSNorm / adaRMSNorm

普通 RMSNorm（流1）：
$$\hat{x} = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2}} \cdot \gamma$$

adaRMSNorm（流2，动作分支）：
```
时间步 t → sinusoidal编码 → Dense → 输出3个向量: scale, shift, gate
x_norm = RMSNorm(x) * (1 + scale) + shift
```
作用：让模型知道当前去噪进行到了哪一步（t=0 全噪声，t=1 完全去噪）。

### 2. 联合注意力（核心创新）

```python
# 流1 投影 (维度 2048 → head_dim*num_heads = 256*8 = 2048)
Q1 = x1 @ W_Q1    # shape: [seq1, 8, 256]
K1 = x1 @ W_K1    # shape: [seq1, 1, 256]  (GQA: 只有1个KV头)
V1 = x1 @ W_V1    # shape: [seq1, 1, 256]

# 流2 投影 (维度 1024 → head_dim*num_heads = 256*8 = 2048)
Q2 = x2 @ W_Q2    # shape: [seq2, 8, 256]
K2 = x2 @ W_K2    # shape: [seq2, 1, 256]
V2 = x2 @ W_V2    # shape: [seq2, 1, 256]

# 序列维拼接
K_all = concat([K1, K2], dim=seq)  # shape: [seq1+seq2, 1, 256]
V_all = concat([V1, V2], dim=seq)  # shape: [seq1+seq2, 1, 256]

# 加 RoPE 位置编码
K_all = apply_rope(K_all)
Q1 = apply_rope(Q1)
Q2 = apply_rope(Q2)

# 各自用自己的Q对合并的KV做attention
out1 = softmax(Q1 @ K_all^T / sqrt(256)) @ V_all  # 流1的输出
out2 = softmax(Q2 @ K_all^T / sqrt(256)) @ V_all  # 流2的输出

# 投影回各自维度
y1 = out1 @ W_O1  # → 2048
y2 = out2 @ W_O2  # → 1024
```

**关键点**：两个流 width 不同（2048 vs 1024），但投影到相同的 head_dim=256 空间后做联合 attention，算完再投影回各自维度。

### 3. 门控残差

```
x = x + attention_output * gate
```
gate 来自 adaRMSNorm 的第三个输出，取值 0~1。
- gate≈0：跳过这一层（模型觉得这层没什么用）
- gate≈1：正常加上

### 4. 门控 FFN

```
hidden = GeLU(x @ W_gate) * (x @ W_up)   ← 门控：一路做非线性，一路做门
output = hidden @ W_down                   ← 压回原维度
```
可以在 W_gate / W_up / W_down 上挂 LoRA 做参数高效微调。

## GQA（Grouped Query Attention）

```
标准 MHA:  8个Q头, 8个K头, 8个V头  → KV cache = 8 × seq × 256
GQA:      8个Q头, 1个K头, 1个V头  → KV cache = 1 × seq × 256  (省8倍)
```
8 个 Q 头各自算出不同的注意力权重，但它们查询的是同一份 K/V。
效果损失很小，推理时 KV cache 省 8 倍显存。

## 为什么要这样设计？

| 设计选择 | 原因 |
|---------|------|
| 双流而非单流 | 预训练的 Gemma 2B 不动（或少动），新加小 expert 学动作，避免灾难性遗忘 |
| 联合注意力而非独立 | 动作 expert 需要看到视觉/语言信息才能决策 |
| 动作 expert 更小 | 动作空间比语言简单，不需要那么大的容量 |
| adaRMSNorm | 流匹配(flow matching)需要知道时间步，条件注入最自然的位置 |
| GQA | 推理效率，机器人需要实时出动作 |
| 门控残差 | 时间步控制信息流动强度 |
