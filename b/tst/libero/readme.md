# Pi0.5 LIBERO 8-GPU Training Example

基于 Physical Intelligence 的 OpenPI 框架，使用 JAX 版本的 π₀.₅ (pi0.5) 模型在 LIBERO 数据集上进行微调训练。
本例启用所有可用优化，使用 8 张 NVIDIA H200 GPU，全局 batch size 128，训练 1000 步，每 200 步保存 checkpoint。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.11（本例使用 3.11.14） |
| JAX | 0.5.3（含 CUDA 12 支持） |
| Flax | 0.10.2 |
| GPU | 8x NVIDIA H200（141GB HBM3e） |
| 虚拟环境 | `/mnt/r/VENV/openpi_venv/` |
| 数据集 | `physical-intelligence/libero`（LeRobot 格式） |

## 文件说明

```
b/tst/libero/
├── readme.md                    # 本文件
├── compute_norm_stats_local.py  # 基于本地缓存数据计算归一化统计
├── train_pi05_libero.py         # 训练包装器（patch 数据加载，仅用本地缓存 episode）
├── run_norm_stats.sh            # 运行 norm stats 计算的 shell 脚本
└── run_train.sh                 # 8-GPU 训练的 shell 脚本
```

## Step-by-Step 使用说明

### Step 1: 确认虚拟环境就绪

```bash
/mnt/r/VENV/openpi_venv/bin/python --version   # 应输出 Python 3.11.x
/mnt/r/VENV/openpi_venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
```

确认 JAX 0.5.3 和 8 张 CUDA 设备均可见。

### Step 2: 确认数据集已缓存

训练使用 HuggingFace 上的 LeRobot 数据集 `physical-intelligence/libero`。首次使用会自动下载到
`~/.cache/huggingface/lerobot/physical-intelligence/libero/`。

检查本地缓存状态：

```bash
ls ~/.cache/huggingface/lerobot/physical-intelligence/libero/data/
# 应看到 chunk-000/, chunk-001/ 等目录
find ~/.cache/huggingface/lerobot/physical-intelligence/libero/data/ -name "*.parquet" | wc -l
# 完整数据集约 994 个 episode
```

### Step 3: 计算归一化统计（可选）

如果 `assets/pi05_libero/physical-intelligence/libero/norm_stats.json` 已存在，可跳过此步。

```bash
bash b/tst/libero/run_norm_stats.sh
```

或使用官方脚本：

```bash
UV_PROJECT_ENVIRONMENT=/mnt/r/VENV/openpi_venv uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### Step 4: 启动训练

```bash
bash b/tst/libero/run_train.sh
```

该脚本等价于：

```bash
export UV_PROJECT_ENVIRONMENT=/mnt/r/VENV/openpi_venv
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jax_compilation_cache"

cd /home/physical/SRC/Robot/openpi05

uv run python b/tst/libero/train_pi05_libero.py pi05_libero \
    --exp-name=pi05_libero_8gpu_1k \
    --batch-size=128 \
    --num-train-steps=1000 \
    --save-interval=200 \
    --log-interval=50 \
    --fsdp-devices=8 \
    --ema-decay=0.999 \
    --lr-schedule.warmup-steps=100 \
    --lr-schedule.peak-lr=5e-5 \
    --lr-schedule.decay-steps=1000 \
    --lr-schedule.decay-lr=5e-6 \
    --overwrite
```

### Step 5: 训练产出

训练完成后，checkpoint 保存在：

```
checkpoints/pi05_libero/pi05_libero_8gpu_1k/999/
├── assets/          # norm_stats 等元数据
├── params/          # EMA 模型权重（用于推理）
└── train_state/     # 训练状态（optimizer state 等）
```

W&B 日志同步到 `wandb.ai` 的 `openpi` 项目。

### Step 6: 使用训练好的模型推理（可选）

```bash
UV_PROJECT_ENVIRONMENT=/mnt/r/VENV/openpi_venv uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/pi05_libero_8gpu_1k/999
```

## 启用的优化与技巧

本例启用了 OpenPI 框架中所有可用的优化：

| 优化 | 配置 | 说明 |
|------|------|------|
| **FSDP** | `--fsdp-devices=8` | 8 卡 Fully Sharded Data Parallelism，模型参数分片到所有 GPU |
| **EMA** | `--ema-decay=0.999` | 指数移动平均，推理时使用 EMA 权重，效果更平滑稳定 |
| **bfloat16** | 默认启用 | 激活值使用 bfloat16，权重/梯度 float32，节省显存 |
| **梯度裁剪** | 默认 `clip_gradient_norm=1.0` | 全局梯度范数裁剪，防止梯度爆炸 |
| **Cosine LR + Warmup** | warmup=100, decay 1000 步 | 预热后余弦衰减学习率 |
| **AdamW** | b1=0.9, b2=0.95, wd=1e-10 | 带权重衰减的 Adam 优化器 |
| **XLA 内存优化** | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | 分配 90% GPU 显存给 JAX |
| **XLA Latency Hiding** | `xla_gpu_enable_latency_hiding_scheduler` | 重叠计算与通信 |
| **JAX 编译缓存** | `JAX_COMPILATION_CACHE_DIR` | 缓存编译结果避免重复编译 |
| **Activation Remat** | 内置 `nn.remat + nn.scan` | 激活值重计算节省显存 |
| **图像增强** | 内置 RandomCrop/Rotate/ColorJitter | 训练时数据增强提升泛化 |
| **Buffer Donation** | 内置 `donate_argnums` | JIT 中捐赠输入 buffer，减少显存拷贝 |
| **分位数归一化** | π₀.₅ 默认 `use_quantile_norm=True` | 使用 Q01/Q99 分位数归一化，对异常值鲁棒 |

## 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | `Pi0Config(pi05=True, action_horizon=10)` | π₀.₅ 模型，10 步动作预测 |
| 预训练权重 | `gs://openpi-assets/checkpoints/pi05_base/params` | Pi Intelligence 官方预训练基座 |
| 全局 batch size | 128 | 8 GPU，每卡 local batch = 16 |
| 训练步数 | 1000 | 短期微调演示 |
| Checkpoint 间隔 | 200 步 | 步 200/400/600/800/999 各保存一次 |
| 日志间隔 | 50 步 | 每 50 步记录 loss/grad_norm |
| 学习率 | peak 5e-5, decay to 5e-6 | 100 步 warmup + cosine decay |
| EMA decay | 0.999 | 接近 1 则 EMA 更稳定 |
| FSDP 设备 | 8 | 全部 GPU 参与模型分片 |
| Checkpoint 保留策略 | `max_to_keep=1` | 仅保留最新 checkpoint（节省磁盘） |

## 训练结果

成功完成 1000 步训练，在 8x H200 上耗时约 **16.5 分钟**（~1.3 it/s）。

| Step | Loss | Grad Norm | Param Norm |
|------|------|-----------|------------|
| 0 | 0.0893 | 0.9612 | 1802.39 |
| 100 | 0.0318 | 0.1353 | 1802.40 |
| 200 | 0.0280 | 0.0887 | 1802.49 |
| 500 | 0.0245 | 0.0767 | 1802.72 |
| 800 | 0.0211 | 0.0690 | 1802.79 |
| 950 | 0.0197 | 0.0560 | 1802.80 |

Loss 从初始 0.089 稳定下降至 0.020，梯度范数收敛，表明训练正常进行。

## 遇到的问题与解决方案

### 问题 1: XLA_FLAGS 不兼容

**错误信息**:
```
F external/xla/xla/parse_flags_from_env.cc:233] Unknown flag in XLA_FLAGS: --xla_gpu_enable_triton_softmax_fusion=true
```

**原因**: JAX 0.5.3 的 XLA 版本不支持 `xla_gpu_enable_triton_softmax_fusion` 和 `xla_gpu_triton_gemm_any` 标志。这些是更新版本 XLA 中的实验性功能。

**解决方案**: 将 XLA_FLAGS 替换为当前版本支持的优化标志：
```bash
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
```

`xla_gpu_enable_latency_hiding_scheduler` 通过重叠计算和通信来隐藏延迟，是当前版本可用的有效优化。

### 问题 2: GPU 显存被其他进程占用导致 OOM

**错误信息**:
```
NCCL operation ncclGroupEnd() failed: unhandled cuda error ... 'Cuda failure 2 "out of memory"'
```

**原因**: 之前的训练进程仍在后台运行，占用了所有 8 张 GPU 的显存（每张 133GB / 143GB）。新进程启动时没有足够显存。

**解决方案**:
1. 使用 `nvidia-smi` 检查 GPU 占用情况
2. 识别并终止占用 GPU 的旧进程：
   ```bash
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
   kill <old_pid>
   ```
3. 等待 GPU 显存完全释放后重新启动训练

**预防措施**: 在训练前添加 GPU 占用检查，或在脚本中加入确认步骤。

### 问题 3: LeRobot 数据集版本警告

**警告信息**:
```
The dataset you requested (physical-intelligence/libero) is in 2.0 format.
```

**原因**: 本地缓存的数据集使用 LeRobot v2.0 格式（全局统计），而当前 LeRobot 库使用 v2.1 格式（按 episode 统计）。

**解决方案**: 这只是警告，不影响训练。LeRobot 保持向后兼容。如需消除警告，可运行：
```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py --repo-id=physical-intelligence/libero
```

## 设计说明

### 为什么不直接用官方脚本？

官方 `scripts/train.py` 会尝试从 HuggingFace Hub 获取完整数据集元数据，可能触发网络限流。
`train_pi05_libero.py` 通过 monkey-patch `create_torch_dataset` 函数，限制只使用本地已缓存的 episode，避免网络依赖。

### Checkpoint 管理策略

OpenPI 使用 Orbax CheckpointManager，配置为 `max_to_keep=1`（仅保留最新），`keep_period=5000`。
每 200 步保存时，旧 checkpoint 会被删除以节省磁盘空间（单个 checkpoint 约 42GB）。
最终保留的是 step 999 的 checkpoint，包含 EMA 权重和完整训练状态。

### FSDP 分片策略

使用 `fsdp_devices=8` 创建 mesh shape `(1, 8)`：
- Batch 轴大小 = 1（无数据并行复制）
- FSDP 轴大小 = 8（模型参数 8 路分片）
- 数据沿 `DATA_AXIS = (BATCH, FSDP)` 分配，即 128 samples 均匀分布在 8 GPU 上

大于 4MB 的参数张量沿最大可整除维度切分到 FSDP 轴，小参数复制到所有设备。

### LR Schedule 适配

官方 `pi05_libero` 配置设定 warmup_steps=10000、decay_steps=1000000（适用于 30k 步训练）。
本例训练仅 1000 步，因此调整为：
- warmup_steps=100（前 10% 预热）
- decay_steps=1000（匹配总步数）
- peak_lr=5e-5（保持与官方一致）
- decay_lr=5e-6（衰减到 peak 的 1/10）
