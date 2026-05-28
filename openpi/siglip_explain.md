## SigLIP 详解

### 一句话
SigLIP 是一个纯视觉编码器，把一张图像"翻译"成一串向量序列，让后面的语言模型能像读文字一样"读"图像。

### 模型架构
SigLIP 的网络结构就是标准的 ViT（Vision Transformer），由三部分组成：

**1. Patch Embedding（切片+嵌入）**
- 用一个 14×14 的卷积核（stride=14）把图像切成不重叠的小块
- 每块 14×14×3 像素 → 映射为 1152 维向量
- 224÷14 = 16，横竖各 16 块，共 256 块
- 结果：256 个 1152 维向量

**2. Position Embedding（位置编码）**
- 给每个 patch 加一个可学习的位置向量
- 告诉模型每块在原图的空间位置（左上/右下/中间...）

**3. Transformer Encoder（特征提取）**
- 27 层 Block，每层包含：
  - LayerNorm → 16 头自注意力 → 残差连接
  - LayerNorm → MLP (1152→4304→1152, GELU) → 残差连接
- 每层让所有 patch 互相交换信息
- 经过 27 层后，每个 patch token 已融合了整张图的全局语义

**4. 输出投影**
- Linear(1152 → 2048)
- 把视觉 token 维度对齐到 Gemma 语言模型的维度

### 关键参数（So400m/14 变体）
| 参数 | 值 |
|------|-----|
| 输入分辨率 | 224×224 |
| patch 大小 | 14×14 |
| patch 数量 | 256 |
| 隐藏维度 (width) | 1152 |
| 层数 (depth) | 27 |
| 注意力头数 | 16 |
| MLP 中间维度 | 4304 |
| 输出投影维度 | 2048 |
| 池化方式 | none（保留全部 token） |

### 输入
- 一张 224×224×3 的 RGB 图像
- 像素值范围 [-1, 1]（uint8 图像会在 Observation.from_dict 中做 $$x \times \frac{2}{255} - 1$$ 转换）

### 输出
- 256 个 2048 维的向量（即 256 个"视觉 token"）
- 这些 token 和语言 token 格式完全一样，可以直接拼接送入 Gemma

### 在 OpenPI 中的角色

```
相机图像(3路)
  │
  ├─ head_rgb ────→ SigLIP ──→ 256 个 token ─┐
  ├─ left_wrist ──→ SigLIP ──→ 256 个 token ─┼─→ 共 768 个视觉 token
  └─ right_wrist ─→ SigLIP ──→ 256 个 token ─┘         │
                                                         ├─ 拼接 → Gemma
  语言指令 "拿起杯子" ──→ Token Embedding ──→ L 个语言 token ─┘
```

- 三个相机共用同一个 SigLIP（参数共享）
- SigLIP 的权重来自 PaliGemma 预训练，在 fine-tune 时通常冻结或用 LoRA
- SigLIP 不做任何文字理解，它只负责把图像压缩成 token 序列
- 真正的图文融合发生在 Gemma 里（视觉 token 和语言 token 一起做 attention）

### 源码位置
- 实现：src/openpi/models/siglip.py（整个文件就是完整的 SigLIP）
- 被调用处：src/openpi/models/pi0.py 第 82 行和 src/openpi/models/pi0_fast.py 第 148 行
- 调用方式：`_siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none")`

### 为什么叫 SigLIP
| 名称 | 含义 |
|------|------|
| Sig | Sigmoid（训练用 sigmoid 对比损失，区别于 CLIP 的 softmax） |
| LI | Language-Image（语言-图像对比学习） |
| P | Pre-training（预训练） |

网络结构 = 标准 ViT，但预训练方式特殊：用 4 亿张"图+文字描述"的配对数据，训练图像编码器让它学会"这张图对应什么语义"。训练完后只保留视觉编码器部分用于下游任务。
