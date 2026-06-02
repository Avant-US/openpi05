## Transformer 总结

### 一句话
一种让序列中所有元素同时互相交换信息的神经网络结构。

### 核心公式
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}}) \cdot V$$
其中：
- $$Q = X \cdot W_Q$$ （每个token变出一个"提问向量"）
- $$K = X \cdot W_K$$ （每个token变出一个"被匹配向量"）
- $$V = X \cdot W_V$$ （每个token变出一个"内容向量"）
- $$QK^T$$ （所有token两两算点积 → 相关度矩阵 $$[n \times n]$$）
- $$\div \sqrt{d}$$ （缩放，防止数值太大）
- softmax （每行归一化成权重，和=1）
- $$\times V$$ （按权重加权求和 → 每个token的新表示）

### 一个 Block 做的事
输入 x (n个token，各d维)
$$x' = x + MultiHeadAttention(LayerNorm(x))$$ ← 收集其他token的信息
$$y = x' + MLP(LayerNorm(x'))$$ ← 独立消化每个token的信息
输出 y (n个token，各d维，但信息更丰富了)

### 多头
把 d 维拆成 h 份，每份 $$d/h$$ 维，每份独立算一组 attention（独立的 $$W_Q, W_K, W_V$$），最后拼回 d 维。
$$MultiHead(Q, K, V) = Concat(head_1, head_2, \dots, head_h) \cdot W_O$$
$$head_i = Attention(X \cdot W_{Q_i},\ X \cdot W_{K_i},\ X \cdot W_{V_i})$$
好处：允许同一层里存在多组不同的 attention 权重分布。

### 堆叠 N 层
- 第 1 层：每个 token 收集了邻近 token 的直接信息
- 第 2 层：每个 token 收集的信息里已经包含了别人第1层收集的信息 → 间接关联
- 第 N 层：每个 token 已经融合了整个序列的全局信息

### 和 LSTM 的本质区别
| | LSTM | Transformer |
|--|------|-------------|
| 信息传递 | 串行，一步步传 | 并行，一步直达任意位置 |
| 每步能看到 | 只有上一步的隐状态 | 所有位置（通过 attention 权重选择性关注） |
| 计算量 | $$O(n \cdot d^2)$$，但必须串行 | $$O(n^2 \cdot d)$$，但可以全部并行 |
| 长序列 | 前面的信息逐步衰减 | 距离无关，attention 决定一切 |

### 在 OpenPI 中的具体使用
图片 → SigLIP (27层 Transformer, 双向) → 256个视觉token (各2048维)
语言 → Token Embedding → L个语言token (各2048维)
两者拼接成一个长序列 → Gemma (18层 Transformer) → 动作 token 输出 → 机器人动作

### 总结
Transformer 的全部精髓就是：用可学习的 Q/K/V 矩阵计算 attention 权重，让每个 token 按需从所有其他 token 取信息，多头允许多种取法，堆多层逐步精炼。
