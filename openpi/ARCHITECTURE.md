# OpenPI 架构详解：模型、策略、训练
---
## 一、模型架构
> 源码位置：`src/openpi/models/`

### 1.1 文件总览
| 文件 | 职责 |
|------|------|
| `model.py` | 公共类型：`ModelType`、`Observation`、`Actions`、`BaseModelConfig`/`BaseModel`、`preprocess_observation`、`restore_params` |
| `pi0_config.py` | Pi0/Pi05 的 `Pi0Config`：变体选择、冻结策略、LoRA |
| `pi0.py` | Pi0/Pi05 主模型：前缀/后缀嵌入、双流 Gemma、flow matching loss/sampling |
| `pi0_fast.py` | Pi0FAST：单流 Gemma + 自回归 token 预测（CE loss） |
| `gemma.py` | 双流 Gemma（Pi0/Pi05 用）：GQA、RoPE、adaRMSNorm、跨 expert 联合注意力 |
| `gemma_fast.py` | 单流 Gemma（FAST 用）：带 KV cache 的解码 |
| `siglip.py` | SigLIP ViT 视觉编码器（So400m/14） |
| `lora.py` | LoRA 配置与低秩分解实现 |
| `tokenizer.py` | PaliGemma/FAST tokenizer：prompt/state 拼装 |
---
### 1.2 视觉编码器：SigLIP
> 📁 **源码**：`src/openpi/models/siglip.py`

```
输入: 224×224×3 RGB 图像
    ↓ Patch Embedding (14×14 patch → 16×16 grid = 256 tokens)
    ↓ Learned Position Embedding
    ↓ 27 层 Transformer Encoder Block
        - LayerNorm → Multi-Head Self-Attention (16 heads)
        - LayerNorm → MLP (GELU, 4304 hidden)
    ↓ Linear Projection (1152 → 2048)
输出: 256 × 2048 视觉 token 序列
```
**关键参数（So400m/14 变体）**：
- width = 1152, depth = 27, mlp_dim = 4304, heads = 16
- patch_size = 14, 无池化（保留全部 patch token）
- 输出投影到 LM 隐藏维度 2048
---
### 1.3 语言+动作骨干：双流 Gemma
> 📁 **源码**：`src/openpi/models/gemma.py`
> 配置入口：`gemma.get_config(variant)` 定义两套参数

SigLIP 编码图像后，其输出 token 与语言 token 一起送入下面的双流结构。
Pi0/Pi05 使用**两套 Gemma Decoder 逐层共享注意力**（注意：这里指 PaliGemma 内部的 Gemma，不含 SigLIP）：
| 参数 | Gemma 主干（PaliGemma 内, 2B） | 动作专家（新增, 300M） |
|------|---------------------|-----------------|
| width | 2048 | 1024 |
| mlp_dim | 16384 | 4096 |
| depth | 18 层 | 18 层 |
| num_heads | 8 | 8 |
| num_kv_heads | 1 (GQA) | 1 (GQA) |
| head_dim | 256 | 256 |

**每层 Block 结构**（见 `gemma.py` 中 `Block` 类）：
1. **RMSNorm**（Pi05 动作分支用 **adaRMSNorm**：时间条件 → Dense(3×dim) → scale/shift/gate）
2. **联合注意力**（`Attention` 类）：两个 expert 分别投影 Q/K/V → **序列维拼接** → RoPE → softmax → 按段拆回各自维度的输出
3. **门控残差**：`x + y * gate`
4. **门控 FFN**（`lora.FeedForward`）：GeLU gating × 线性，可挂 LoRA

**词汇表大小**：257,152（PaliGemma）
---
### 1.4 三种模型变体
> 📁 **源码**：
> - Pi0 / Pi05：`src/openpi/models/pi0.py`（`Pi0` 类，通过 `config.pi05` 标志区分）
> - Pi0FAST：`src/openpi/models/pi0_fast.py`（`Pi0FAST` 类）
> - 配置：`src/openpi/models/pi0_config.py`（`Pi0Config`、`Pi0FASTConfig`）

#### Pi0（Flow Matching）
```
前缀 (prefix):                          ← pi0.py: embed_prefix()
  [SigLIP tokens × N_cameras] + [语言 token embeddings]
  - 内部双向注意力（ar_mask=0）

后缀 (suffix):                          ← pi0.py: embed_suffix()
  [1 个 state token] + [action_horizon 个动作 token]
  - state: action_dim → expert_width 线性映射 (state_proj)
  - 动作: noisy_actions 经 action_in_proj → 与 sin-cos 时间编码拼接
           → action_time_mlp 得到每步一个 token
  - 注意力: 前缀不能 attend 后缀; 动作 token 之间因果
```

#### Pi05（Flow Matching + 离散 State）
与 Pi0 的区别：
- **State 离散化**：state 不再是单独的连续 token，而是在 tokenizer 里**离散化后嵌入 prompt**（见 `tokenizer.py: PaligemmaTokenizer`）
- **无 state token**：后缀只有 `action_horizon` 个动作 token
- **时间注入方式不同**：timestep → sin-cos → time_mlp → 作为 **adaRMSNorm 的条件**注入第二层 expert（见 `gemma.py: RMSNorm` 的 `ada` 分支）
- **更长的 prompt**：`max_token_len = 200`（Pi0 为 48）

#### Pi0FAST（自回归 Token 预测）
```
单流 Gemma 2B (18层, 2048-wide, 无额外 expert)   ← gemma_fast.py
前缀: [SigLIP tokens] + [语言 token embeddings]
    ↓ 自回归 next-token 预测
输出: 离散 token ID → FAST tokenizer 解码 → 连续动作
Loss: Cross-Entropy（非 MSE）              ← pi0_fast.py: compute_loss()
```
---
### 1.5 注意力掩码机制（attn_mask）
> 📁 **源码**：`src/openpi/models/pi0.py` 中的 `make_attn_mask()` 函数

Pi0/Pi05 使用 `mask_ar`（自回归掩码）控制 token 之间的注意力可见性，通过**累积和比较**生成最终的注意力矩阵。

**核心算法**（4 行代码）：
```python
cumsum = cumsum(mask_ar)                         # 对 mask_ar 做累积和
attn_mask[i][j] = (cumsum[j] <= cumsum[i])       # j 能被 i 看到的条件
valid_mask = input_mask[i] AND input_mask[j]     # 排除 padding
final_mask = attn_mask AND valid_mask            # 合并
```

**规则**：`mask_ar=0` 表示"与前一个共享可见性"，`mask_ar=1` 表示"切一刀，前面看不到我"。

**Pi0.5 的实际掩码布局**：
```
token 序列:  [图像×768, 语言×~20, 动作×50]
mask_ar:     [0,0,...0, 0,0,...0,  1,0,...0]
cumsum:      [0,0,...0, 0,0,...0,  1,1,...1]
```
生成的注意力矩阵效果：
| 谁在看 (i) → | 图像/语言 (cumsum=0) | 动作 (cumsum=1) |
|--------------|---------------------|----------------|
| **图像/语言** (cumsum=0) | ✓ 双向互看 | ✗ 看不到动作 |
| **动作** (cumsum=1) | ✓ 能看图像/语言 | ✓ 动作之间互看 |

**与 Transformer Attention 的关系**：
```
scores = Q × K^T / √d                   ← 原始相关度分数
scores[attn_mask == False] = -∞          ← mask 禁止的位置设为负无穷
weights = softmax(scores)                ← -∞ 经 softmax 变为 0（完全忽略）
output = weights × V                     ← 只融合 mask=True 位置的信息
```

**设计意图**：
- 图像和语言互相看 → 视觉-语言融合理解
- 动作能看所有条件 → 基于感知做决策
- 条件看不到动作 → 防止信息泄露（训练时动作含真值信息）
---
### 1.6 Flow Matching 机制（Pi0/Pi05）
> 📁 **源码**：`src/openpi/models/pi0.py`
> - 训练：`Pi0.compute_loss()` 方法
> - 推理：`Pi0.sample_actions()` 方法

**训练**：
```python
t ~ Beta(1.5, 1) × 0.999 + 0.001          # 时间采样
ε ~ N(0, I)                                # 噪声
x_t = t · ε + (1-t) · actions              # 线性插值
u_t = ε - actions                          # 目标速度场
v_t = model(x_t, t, observation)           # 预测速度场
loss = mean((v_t - u_t)², axis=-1)         # MSE loss
```
**推理（Euler 积分，t: 1→0）**：
```python
x = randn(action_horizon, action_dim)      # 从纯噪声开始
for i in range(N_steps):
    dt = -1/N
    v = model(x, t, observation)
    x = x + dt * v
    t = t + dt
```
---
### 1.7 Observation / Actions 数据结构
> 📁 **源码**：`src/openpi/models/model.py`
> - 定义：`Observation` dataclass
> - 构建：`Observation.from_dict()` 静态方法

**Observation**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `images` | `dict[str, float[B,H,W,3]]` | 多视角图像，范围 [-1,1] |
| `image_masks` | `dict[str, bool[B]]` | 各视图是否有效 |
| `state` | `float[B, action_dim]` | 机器人状态（维度=动作维度） |
| `tokenized_prompt` | `int[B, L]` | 分词后的语言指令 |
| `tokenized_prompt_mask` | `bool[B, L]` | prompt 有效位 |
| `token_ar_mask` | `int[B, L]` | FAST 用：前缀 attention 样式 |
| `token_loss_mask` | `bool[B, L]` | FAST 用：哪些位计算 CE loss |

**Actions**：`float[B, action_horizon, action_dim]`
---
### 1.8 LoRA
> 📁 **源码**：`src/openpi/models/lora.py`
> - 配置类：`LoRAConfig`
> - 挂载点：`lora.Einsum`（Attention）、`lora.FeedForward`（FFN）
> 冻结逻辑见：`src/openpi/models/pi0_config.py` → `Pi0Config.get_freeze_filter()`

- **结构**：在 Attention 的 Q/K/V 和 FFN 上各挂一对低秩矩阵 (A, B)
- **默认 rank**：16
- **缩放**：`alpha / sqrt(rank)`（rsLoRA）
- **冻结策略**：带 LoRA 变体时冻结所有原始 LLM 权重，只训练 LoRA 参数
---
### 1.9 PyTorch 实现差异
> 📁 **源码**：`src/openpi/models_pytorch/`
> - 主模型：`pi0_pytorch.py`（`PI0Pytorch` 类）
> - 预处理：`_preprocessing.py`
> - HF 补丁：`transformers_replace/`（需拷入已安装的 transformers 包）

| 维度 | JAX | PyTorch |
|------|-----|---------|
| 骨干 | 自研 Flax NNX Gemma + SigLIP | HuggingFace `PaliGemmaForConditionalGeneration` + 补丁 |
| 损失 | `jnp.square` | `F.mse_loss(reduction="none")` |
| 加速 | — | `torch.compile(sample_actions)` |
| 精度 | bfloat16 全程 | 部分层强制 fp32（patch emb、norm） |
| 注意力 | bool mask + cumsum | 大负数 mask (-2.38e38) |
---
## 二、策略层
> 源码位置：`src/openpi/policies/`

### 2.1 Policy 类核心流程
> 📁 **源码**：`src/openpi/policies/policy.py` → `Policy` 类的 `infer()` 方法

```
┌─────────────────────────────────────────────────────┐
│                  Policy.infer(obs)                    │
├─────────────────────────────────────────────────────┤
│  1. 深拷贝输入字典                                    │
│  2. 输入变换链（按顺序）:                              │
│     repack_transforms.inputs                         │
│     → InjectDefaultPrompt                            │
│     → data_transforms.inputs (机器人适配)             │
│     → Normalize (z-score 或 quantile)                │
│     → model_transforms.inputs (resize/tokenize/pad)  │
│  3. 添加 batch 维度                                   │
│  4. Observation.from_dict(inputs)                    │
│     - uint8 图像 → float [-1, 1]                     │
│  5. model.sample_actions(observation)                │
│     - Flow matching Euler 积分 (Pi0/Pi05)            │
│     - 或自回归解码 (FAST)                             │
│  6. 移除 batch 维度                                   │
│  7. 输出变换链（按顺序）:                              │
│     model_transforms.outputs (FAST解码)              │
│     → Unnormalize                                    │
│     → data_transforms.outputs (回到机器人格式)        │
│     → repack_transforms.outputs                      │
│  8. 返回 actions + timing 信息                        │
└─────────────────────────────────────────────────────┘
```
---
### 2.2 Policy 创建流程
> 📁 **源码**：`src/openpi/policies/policy_config.py` → `create_trained_policy()`

```python
create_trained_policy(train_config, checkpoint_dir):
    1. 判断后端: model.safetensors 存在 → PyTorch，否则 JAX
    2. 加载模型权重
    3. DataConfig = train_config.data.create(assets_dirs, model_config)
    4. norm_stats 从 checkpoint/assets/<asset_id> 加载
    5. 组装输入/输出变换链
    6. 返回 Policy 实例
```
---
### 2.3 DataTransformFn 协议
> 📁 **源码**：`src/openpi/transforms.py` → `DataTransformFn` Protocol

```python
class DataTransformFn(Protocol):
    def __call__(self, data: dict) -> dict: ...
```
每个机器人平台提供一对实现：
- **`XxxInputs`**：原始观测 → 标准格式（图像键名映射、state 重组）
- **`XxxOutputs`**：标准动作 → 机器人可执行动作（维度截取、坐标变换）
---
### 2.4 具体 Policy 实现示例

#### ALOHA (双臂 14 维)
> 📁 **源码**：`src/openpi/policies/aloha_policy.py`

```python
class AlohaInputs:
    # 相机映射:
    #   cam_high → base_0_rgb
    #   cam_left_wrist → left_wrist_0_rgb
    #   cam_right_wrist → right_wrist_0_rgb
    # State: 14维 (左臂7 + 右臂7)，含关节符号翻转 + 夹爪角度映射
    # 缺失相机: 黑图 + mask=False

class AlohaOutputs:
    # actions[:, :14] → 关节翻转逆变换 + 夹爪角度逆映射
```

#### R1 Pro Chassis (23 维)
> 📁 **源码**：`src/openpi/policies/r1pro_chassis_policy.py`

```python
class R1ProChassisInputs:
    # 相机映射:
    #   head_rgb → base_0_rgb
    #   left_wrist_rgb → left_wrist_0_rgb
    #   right_wrist_rgb → right_wrist_0_rgb
    # State: 23维 (左臂7 + 右臂7 + 左夹爪1 + 右夹爪1 + 躯干4 + 底盘3)
    # 三个 mask 全 True（无缺失相机）

class R1ProChassisOutputs:
    # actions[:, :23]
```

#### LIBERO (仿真 7 维)
> 📁 **源码**：`src/openpi/policies/libero_policy.py`

```python
class LiberoInputs:
    # 相机映射:
    #   observation/image → base_0_rgb
    #   observation/wrist_image → left_wrist_0_rgb
    #   right_wrist_0_rgb → 零填充
    # State: observation/state (7维)

class LiberoOutputs:
    # actions[:, :7]
```
---
### 2.5 归一化机制
> 📁 **源码**：
> - 变换实现：`src/openpi/transforms.py` → `Normalize` / `Unnormalize` 类
> - 统计量定义：`src/openpi/shared/normalize.py` → `NormStats`
> - 预计算脚本：`scripts/compute_norm_stats.py`

| 类型 | 公式 | 使用场景 |
|------|------|----------|
| Z-score | `(x - mean) / (std + 1e-6)` | Pi0 |
| Quantile | `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1` | Pi05, FAST |

- 归一化作用于 `state` 和 `actions`，不作用于图像
- 图像在 `Observation.from_dict` 中独立处理: `x * (2/255) - 1`（见 `models/model.py`）
- 统计量与 checkpoint 绑定，从 `checkpoint/assets/<asset_id>/norm_stats.json` 加载
---
## 三、训练系统
> 源码位置：`src/openpi/training/`

### 3.1 文件总览
| 文件 | 职责 |
|------|------|
| `config.py` | `TrainConfig`、`DataConfig`、`DataConfigFactory`、配置注册表 `_CONFIGS` |
| `data_loader.py` | LeRobot / RLDS 数据加载、batch、sharding |
| `optimizer.py` | AdamW + cosine/rsqrt schedule + 梯度裁剪 |
| `checkpoints.py` | Orbax checkpoint 管理：保存/恢复/EMA |
| `sharding.py` | JAX mesh、FSDP 分片策略 |
| `weight_loaders.py` | 预训练权重加载协议 |
| `utils.py` | TrainState 定义 |
---
### 3.2 TrainConfig 关键字段
> 📁 **源码**：`src/openpi/training/config.py` → `TrainConfig` dataclass

```python
@dataclass
class TrainConfig:
    name: str                    # 配置唯一标识 (如 "pi05_r1pro_chassis_clean")
    project_name: str            # WandB 项目名, 默认 "openpi"
    exp_name: str                # 实验名, 决定 checkpoint 子目录
    # 模型
    model: BaseModelConfig       # Pi0Config / Pi0FASTConfig
    weight_loader: WeightLoader  # 预训练权重加载器
    freeze_filter: nnx.Filter    # 冻结哪些参数
    # 数据
    data: DataConfigFactory      # 数据管线工厂
    # 优化
    lr_schedule: LRSchedule      # 默认 CosineDecaySchedule
    optimizer: Optimizer         # 默认 AdamW
    batch_size: int
    num_train_steps: int
    ema_decay: float = 0.99      # EMA 衰减率 (None=不用)
    # 路径
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    # 并行
    fsdp_devices: int = 1        # FSDP 轴宽度
    # 日志/存盘
    log_interval: int
    save_interval: int
    seed: int
    resume: bool = False
```
---
### 3.3 配置注册表
> 📁 **源码**：`src/openpi/training/config.py` → `_CONFIGS` 列表 + `get_config()` / `cli()`

```python
_CONFIGS = [
    TrainConfig(name="pi0_aloha", ...),
    TrainConfig(name="pi05_aloha", ...),
    TrainConfig(name="pi0_libero", ...),
    TrainConfig(name="pi0_fast_libero", ...),
    TrainConfig(name="pi05_r1pro_chassis_clean", ...),
    TrainConfig(name="pi0_aloha_sim", ...),
    TrainConfig(name="debug", ...),
    ...
]
# 使用方式
config = get_config("pi05_r1pro_chassis_clean")
# CLI 方式 (tyro 子命令)
uv run python scripts/train.py pi05_r1pro_chassis_clean --exp_name my_exp
```
---
### 3.4 数据管线架构
> 📁 **源码**：
> - 工厂与配置：`src/openpi/training/config.py` → `DataConfigFactory` / `DataConfig`
> - 数据加载：`src/openpi/training/data_loader.py` → `create_data_loader()`
> - 变换定义：`src/openpi/transforms.py`

```
DataConfigFactory.create(assets_dirs, model_config)
    │
    ├── repo_id → LeRobot 数据集路径
    ├── asset_id → norm stats 子目录名
    ├── norm_stats → 加载归一化统计量
    │
    └── 三组 Transform:
         ├── repack_transforms   → 键名重整 (LeRobot 扁平路径 → 标准结构)
         ├── data_transforms     → 机器人特异变换 (Inputs/Outputs)
         └── model_transforms    → 模型通用变换 (resize/tokenize/pad)
```
**数据加载分支**（见 `data_loader.py: create_data_loader()`）：
| 条件 | 路径 | 支持框架 |
|------|------|----------|
| `rlds_data_dir != None` | RLDS (tf.data + dlimp) | 仅 JAX |
| 否则 | LeRobot + PyTorch DataLoader | JAX + PyTorch |

**样本变换顺序（训练时）**：
```
原始样本 → repack → data_transforms → Normalize → model_transforms → (Observation, Actions)
```
---
### 3.5 优化器配置
> 📁 **源码**：`src/openpi/training/optimizer.py`
> - Schedule：`CosineDecaySchedule` / `RsqrtDecaySchedule`
> - Optimizer：`AdamW` / `SGD`
> - 组装：`create_optimizer()`

**JAX**：
```python
optax.chain(
    optax.clip_by_global_norm(max_norm),   # 梯度裁剪
    optax.adamw(
        learning_rate=warmup_cosine_schedule,
        weight_decay=1e-10,
        b1=0.9, b2=0.95
    )
)
```
**学习率 Schedule（默认 Cosine Decay）**：
```
warmup: 0 → peak_lr (线性, warmup_steps 步)
decay:  peak_lr → 0 (余弦, 剩余步数)
```
**PyTorch**（见 `scripts/train_pytorch.py`）：
```python
torch.optim.AdamW(params, lr=peak_lr, betas=(0.9, 0.95), weight_decay=1e-10)
# + 手写 warmup + cosine schedule
# + torch.nn.utils.clip_grad_norm_(max_norm)
```
---
### 3.6 训练循环

#### JAX 训练循环
> 📁 **源码**：`scripts/train.py`

```python
# 1. 初始化
mesh = make_mesh(fsdp_devices)
data_loader = create_data_loader(data_config, framework="jax")
train_state = init_train_state(model, optimizer, weight_loader)
# 2. 可选恢复
if resume:
    train_state = restore_state(checkpoint_manager)
# 3. JIT 编译训练步
@jax.jit
def train_step(rng, train_state, batch):
    def loss_fn(model):
        obs = Observation.from_dict(batch)
        actions = batch["actions"]
        return mean(model.compute_loss(rng, obs, actions))
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    train_state.apply_updates(grads)
    ema_update(train_state)
    return loss, grad_norm, param_norm
# 4. 主循环
for step, batch in enumerate(data_loader):
    loss, info = train_step(rng, train_state, batch)
    if step % log_interval == 0: wandb.log(info)
    if step % save_interval == 0: save_state(train_state)
```

#### PyTorch 训练循环
> 📁 **源码**：`scripts/train_pytorch.py`

```python
# 1. DDP 初始化
init_process_group(backend="nccl")
model = DDP(PI0Pytorch(...).to(device))
# 2. 优化器
optimizer = AdamW(model.parameters(), ...)
# 3. 主循环
for step, batch in enumerate(data_loader):
    observation = Observation.from_dict(batch).to(device)
    actions = batch["actions"].to(device)
    loss = model(observation, actions).mean()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad()
    if step % save_interval == 0:
        save_safetensors(model, step)
```
---
### 3.7 Checkpoint 系统
> 📁 **源码**：`src/openpi/training/checkpoints.py`
> - 初始化：`initialize_checkpoint_dir()`
> - 保存/恢复：`save_state()` / `restore_state()`

**JAX (Orbax)**：
```
checkpoint_dir/
├── <step>/
│   ├── params/           # 模型参数 (EMA 时存 EMA 版本)
│   ├── train_state/      # 优化器状态 + step
│   └── assets/
│       └── <asset_id>/
│           └── norm_stats.json
```
**PyTorch**（见 `scripts/train_pytorch.py` 中的保存逻辑）：
```
checkpoint_dir/
├── <step>/
│   ├── model.safetensors  # 模型参数
│   ├── optimizer.pt       # 优化器状态
│   ├── metadata.pt        # 配置快照
│   └── assets/
│       └── <asset_id>/
│           └── norm_stats.json
```
---
### 3.8 预训练权重加载
> 📁 **源码**：`src/openpi/training/weight_loaders.py`

| Loader | 用途 |
|--------|------|
| `PaliGemmaWeightLoader` | 从 Google Cloud 下载官方 PaliGemma 权重，合并到随机初始化的完整模型 |
| `CheckpointWeightLoader` | 从已有 checkpoint 加载（fine-tune），LoRA 参数保留初始化 |
| `NoOpWeightLoader` | 不加载任何预训练权重（从头训练/调试） |
---
### 3.9 FSDP 分片策略 (JAX)
> 📁 **源码**：`src/openpi/training/sharding.py`

```python
mesh = Mesh(devices.reshape(B, F), axis_names=("batch", "fsdp"))
# 数据: 沿 batch+fsdp 两轴切分
data_sharding = PartitionSpec(("batch", "fsdp"))
# 参数: 大矩阵沿最大且能被 F 整除的维做 FSDP 切分
#        小参数/标量 → replicate
param_sharding = PartitionSpec(..., "fsdp", ...)
```
PyTorch 侧仅使用 DDP（不做 FSDP）。
---
## 四、端到端数据流图

### 训练 Pipeline
```
┌─────────────────────────────────────────────────────────────────────┐
│  LeRobot/RLDS 数据集                                                  │
│       │                                                              │
│       ▼                                                              │
│  RepackTransform (键名标准化)          ← transforms.py               │
│       │                                                              │
│       ▼                                                              │
│  DataTransforms.inputs                 ← policies/*_policy.py        │
│  (机器人适配: AlohaInputs / R1ProChassisInputs)                       │
│       │                                                              │
│       ▼                                                              │
│  Normalize                             ← transforms.py               │
│  (state/actions → z-score 或 quantile)                               │
│       │                                                              │
│       ▼                                                              │
│  ModelTransforms.inputs                ← transforms.py               │
│  (ResizeImages 224×224, TokenizePrompt, Pad)                         │
│       │                                                              │
│       ▼                                                              │
│  Observation.from_dict + Actions       ← models/model.py             │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────┐                    │
│  │         Model Forward               ← models/pi0.py              │
│  │  Images → SigLIP (256 tokens/cam × 2048)     │                    │
│  │  Prompt → Embedder (token → 2048)            │                    │
│  │  State  → Linear (action_dim → 1024)         │                    │
│  │  Noisy Actions + Time → MLP → tokens         │                    │
│  │       ↓                                       │                    │
│  │  18层 双流 Gemma (联合注意力)    ← models/gemma.py                │
│  │       ↓                                       │                    │
│  │  预测速度场 v_t                                │                    │
│  │       ↓                                       │                    │
│  │  MSE Loss: |v_t - (noise - actions)|²        │                    │
│  └──────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼                                                              │
│  Backward + AdamW + Gradient Clip      ← training/optimizer.py       │
│       │                                                              │
│       ▼                                                              │
│  Checkpoint                            ← training/checkpoints.py     │
│  (params + optimizer + norm_stats)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 推理 Pipeline
```
┌─────────────────────────────────────────────────────────────────────┐
│  机器人原始观测                                                        │
│  {"head_rgb": ..., "state": [...], "prompt": "..."}                  │
│       │                                                              │
│       ▼                                                              │
│  输入变换链                             ← policies/policy.py          │
│  (同训练, 含 InjectDefaultPrompt)                                     │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────┐                    │
│  │    Model: sample_actions              ← models/pi0.py             │
│  │  x = randn(action_horizon, action_dim)       │                    │
│  │  for step in range(N):                       │                    │
│  │      v = model(x, t, observation)            │                    │
│  │      x = x + (-1/N) * v                     │                    │
│  │      t = t - 1/N                             │                    │
│  └──────────────────────────────────────────────┘                    │
│       │                                                              │
│       ▼                                                              │
│  输出变换链                             ← policies/policy.py          │
│  (Unnormalize → 机器人格式)                                           │
│       │                                                              │
│       ▼                                                              │
│  actions[action_horizon, action_dim] → 机器人执行                     │
└─────────────────────────────────────────────────────────────────────┘
```
---
## 五、添加新机器人平台的步骤
以 R1 Pro Chassis 为例：

1. **创建 Policy 适配文件**（`src/openpi/policies/r1pro_chassis_policy.py`）：
   - `R1ProChassisInputs`：映射相机名 + 组装 state
   - `R1ProChassisOutputs`：截取 actions 前 N 维

2. **在配置中注册**（`src/openpi/training/config.py`）：
   - 创建 `DataConfigFactory` 子类或用 `SimpleDataConfig`
   - 指定 `repo_id`、`data_transforms=Group(inputs=[R1ProChassisInputs()], outputs=[R1ProChassisOutputs()])`
   - 添加 `TrainConfig(name="pi05_r1pro_chassis_clean", ...)` 到 `_CONFIGS`

3. **预计算 norm stats**（`scripts/compute_norm_stats.py`）：
   ```bash
   python scripts/compute_norm_stats.py pi05_r1pro_chassis_clean
   ```
   生成 `assets/pi05_r1pro_chassis_clean/<asset_id>/norm_stats.json`

4. **训练**（`scripts/train.py` 或 `scripts/train_pytorch.py`）：
   ```bash
   uv run python scripts/train.py pi05_r1pro_chassis_clean --exp_name v1
   ```

5. **部署推理**（`scripts/serve_policy.py`）：
   ```bash
   uv run python scripts/serve_policy.py \
       --port 8000 policy:checkpoint \
       --policy.config pi05_r1pro_chassis_clean \
       --policy.dir checkpoints/.../12500
   ```
