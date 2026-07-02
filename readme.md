# OpenPI · R1 Pro (pi0.5) 服务端部署与训练

本仓库基于 Physical Intelligence 官方 [openpi](https://github.com/Physical-Intelligence/openpi) 定制，面向 Galaxea R1 Pro 双臂带底盘机器人，提供 pi0.5 VLA 模型的训练与 WebSocket 服务端推理能力，已在 H200 / A100 / RTX 5090 等 GPU 上部署验证。

代码相对官方主要做了 R1 Pro 数据适配、推理延迟优化、关键帧推理链路、以及针对开门任务的训练策略改进，详见下面的「相对官方的主要改进」。

## 相对官方的主要改进

下面是本仓库相对官方 openpi 的改动概览。

R1 Pro 机器人适配

- 新增 `src/openpi/policies/r1pro_policy.py`：20 维状态/动作（左臂7 + 右臂7 + 左爪1 + 右爪1 + 躯干4），三路相机（head / left_wrist / right_wrist）映射到模型的 base / left_wrist / right_wrist。
- 新增 `src/openpi/policies/r1pro_chassis_policy.py`：23 维状态/动作（在上面基础上 + 底盘3），即 85 / 151 两台生产机器使用的格式。
- 新增 `src/openpi/policies/r1pro_keyframe_policy.py`：关键帧增强版输入输出。
- `src/openpi/training/config.py` 新增 8 个 R1 Pro 训练配置（见下方「训练」章节）。

训练策略：StateCorruptionTransform（打破 state→action 捷径）

- 位置：`r1pro_chassis_policy.py` 中的 `StateCorruptionTransform`。
- 解决的问题：pi0.5 在开门任务上容易学到「当前 state 像推门姿态就继续输出推门 action」这条捷径，导致「门没开却一直戳门」的失败模式。
- 做法：训练时按概率破坏 state 输入，强迫模型从视觉判断真实环境（门是否已打开），而不是只对 state 做模式匹配。
- 三种模式：normal（原样透传，默认，baseline 行为不变）、zero（按概率把 state 置零，思路接近 State-free Policy, arXiv:2509.18644）、swap（从缓冲区随机取历史 state 替换，更激进）。
- 仅训练生效（按是否含 actions 键区分训练/推理），默认 prob=0.15、mode=normal。

推理延迟优化

- 新增 `scripts/serve_policy_latency.py`，相对官方 serve_policy 增加两点。
- JPEG 直传（方案C）：客户端上传原始 JPEG 字节，服务端用 OpenCV 解码并按各相机宽高比做中心裁剪，显著降低上行带宽，并保证与训练输入像素级一致。
- 服务端时间戳：返回 t2_recv_ns / t5_send_ns / infer_ms，配合客户端的 t1/t6，可精确拆分「上传 / 推理 / 下载」三段延迟。

关键帧（Keyframe）推理链路

- 新增 `scripts/serve_keyframe_policy.py` 与 `r1pro_keyframe_policy.py`，并大幅改写 `src/openpi/serving/websocket_keyframe_server.py`。
- 在 head 相机序列中注入历史关键帧，帮助模型判断任务阶段；prefix token 由 768 增至 1024 / 1536，对应 batch size 相应缩小以避免 OOM。
- 对应训练配置 `pi05_r1pro_chassis_keyframe`。

模型与视觉编码器层改动

- `src/openpi/models/pi0.py`、`siglip.py`、`model.py`、`models_pytorch/preprocessing_pytorch.py` 有改动（详见提交说明）。

## 环境要求

- 操作系统：Ubuntu 22.04（官方仅测试此版本）。
- Python：3.11（由 `.python-version` 指定，uv 自动准备）。
- GPU 显存参考见下表。

| 用途 | 显存需求 | 示例 GPU |
| --- | --- | --- |
| 推理部署 | 大于 8 GB | RTX 4090 / RTX 5090 |
| LoRA 微调 | 大于 22.5 GB | RTX 4090 |
| 全量微调 | 大于 70 GB | A100 80G / H100 / H200 |

pi0.5 权重约 12 GB（bfloat16），加上 JIT 编译与推理中间变量，推理实际占用约 16 至 20 GB。

## 安装

依赖通过 uv 管理，JAX / PyTorch / CUDA 运行库都由 uv 自动安装，系统层不需要预装 CUDA toolkit。

一键安装：

```bash
bash install.sh
```

或手动执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

其中 `GIT_LFS_SKIP_SMUDGE=1` 用于跳过 LeRobot 依赖的 LFS 大文件拉取。

RTX 5090（Blackwell）提示：若 JAX 报 sm_120 不支持，请把 NVIDIA 驱动升级到支持 CUDA 12.8 及以上的版本后重试。

## 目录结构

| 路径 | 作用 |
| --- | --- |
| src/openpi/policies/ | R1 Pro 输入输出适配（r1pro / chassis / keyframe） |
| src/openpi/training/config.py | 训练配置，含 8 个 R1 Pro 配置 |
| src/openpi/serving/ | WebSocket 服务端（含 keyframe server） |
| src/openpi/models/ | pi0 / pi0.5 模型与视觉编码器 |
| scripts/serve_policy.py | 标准服务端启动脚本 |
| scripts/serve_policy_latency.py | 带延迟统计与 JPEG 直传的服务端 |
| scripts/serve_keyframe_policy.py | 关键帧服务端 |
| scripts/train.py | 训练入口 |
| scripts/compute_norm_stats.py | 计算归一化参数 |
| packages/openpi-client/ | 客户端库（uv workspace 成员） |
| install.sh | 一键安装脚本 |

模型权重 checkpoints/、归一化参数 assets/、数据 data/ 不随仓库提交，需另行准备，详见「模型权重与归一化参数」。

## 训练

config.py 中可用的 R1 Pro 训练配置如下。

| 配置名 | 状态维度 | 说明 |
| --- | --- | --- |
| pi05_r1pro_open_door | 20 | 开门基础版 |
| pi05_r1pro_open_door_lora | 20 | 开门 LoRA 微调版 |
| pi05_r1pro_chassis | 23 | 含底盘单数据集版 |
| pi05_r1pro_chassis_all | 23 | 含底盘全量数据集版 |
| pi05_r1pro_chassis_clean | 23 | 含底盘清洗数据版 |
| pi05_r1pro_chassis_roll | 23 | 85 机器使用 |
| pi05_r1pro_open_door_0530 | 23 | 151 机器使用，0530 开门模型 |
| pi05_r1pro_chassis_keyframe | 23 | 关键帧增强版 |

先计算归一化参数：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_r1pro_chassis_roll
```

再启动训练（按需指定空闲 GPU 与实验名）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_r1pro_chassis_roll \
    --exp-name=my_experiment --overwrite
```

## 服务端部署（推理）

启动后通过 WebSocket（默认 8000 端口）对外提供推理服务。首次推理会触发 JAX JIT 编译，通常需要 1 至 2 分钟。

务必用 CUDA_VISIBLE_DEVICES 指定空闲 GPU，否则 JAX 可能选到被占用的卡导致 OOM。

85 机器（pi05_r1pro_chassis_roll）固化权重启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy_latency.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_r1pro_chassis_roll \
    --policy.dir checkpoints/pi05_r1pro_chassis_roll/0521_chassis_roll_normal/20000
```

151 机器（pi05_r1pro_open_door_0530）固化权重启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy_latency.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_r1pro_open_door_0530 \
    --policy.dir checkpoints/pi05_r1pro_open_door_0530/10000
```

上面 `--policy.dir` 为示例路径，按实际部署机器上的 checkpoint 位置调整。健康检查：

```bash
curl http://localhost:8000/healthz
```

## 输入 / 输出格式

以 23 维 chassis 版（85 / 151）为例。输入为一个 dict。

| 字段 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- |
| head_rgb | uint8 / float | (H,W,3) 或 (3,H,W) | 头部相机 |
| left_wrist_rgb | uint8 / float | (H,W,3) 或 (3,H,W) | 左腕相机 |
| right_wrist_rgb | uint8 / float | (H,W,3) 或 (3,H,W) | 右腕相机 |
| state | float32 | (23,) | 当前状态 |
| prompt | str | - | 任务指令，可选 |

state 与 action 的 23 维顺序为：左臂 [0:7]、右臂 [7:14]、左爪 [14]、右爪 [15]、躯干 [16:20]、底盘 [20:23]。图像支持 HWC / CHW 两种排布与 uint8 / float 两种值域，服务端会自动处理。

## 模型权重与归一化参数

为保持仓库轻量，以下大文件不随仓库提交，需另行准备。

| 内容 | 目录 | 说明 |
| --- | --- | --- |
| 模型权重 | checkpoints/ | 各任务的 JAX checkpoint |
| 归一化参数 | checkpoint 内 assets/ | 每个 checkpoint 自带对应 norm_stats |
| 训练数据 | data/ | LeRobot 格式数据集 |

服务端启动时通过 `--policy.dir` 指向具体 checkpoint 目录，归一化参数从该 checkpoint 的 assets 自动加载。

## 常见问题

显存不足 OOM：用 `CUDA_VISIBLE_DEVICES=0` 指定空闲卡。

首次推理很慢：正常现象，JAX 首次需要 JIT 编译。

客户端连不上：确认 `curl http://<IP>:8000/healthz` 返回 OK，确认端口未被防火墙拦截，跨机部署不要写 localhost。

更详细的官方说明（libero/aloha 训练、JAX 转 PyTorch 等）见 `docs/README_openpi_upstream.md`。
