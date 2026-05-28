# π₀.₅ on R1 Pro

基于 [OpenPI](https://github.com/Physical-Intelligence/openpi) 在 **R1 Pro** 机器人上做 π₀.₅ 微调与部署。动作空间 **23 维**（双臂 14 + 双夹爪 2 + torso 4 + 底盘速度 3）。

---

## 1. 安装

```bash
git clone --recurse-submodules <REPO_URL> && cd <REPO_NAME>/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

> 需要 [uv](https://docs.astral.sh/uv/)、CUDA 12、显存：推理 8G+ / LoRA 22G+ / 全量 70G+。

### 关于 `HF_LEROBOT_HOME`

后续命令都带 `HF_LEROBOT_HOME=/path/to/data`，**它是一个目录路径**（建议磁盘空间大的盘），用来存放所有 LeRobot 格式数据集。

- **首次使用**：随便指一个空目录（不存在会自动创建），转换脚本会把结果写到 `<这个目录>/<output_repo>/`
- **后续使用**：必须用同一个目录 —— 训练 / 计算 norm stats 都要从这里读数据，否则找不到

转换后的目录结构（LeRobot v3.0 格式）：
```
/path/to/data/
└── r1_pro_data_convert_chassis_all/      <- 由 --output_repo 决定
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── tasks.jsonl
    │   └── episodes_stats.jsonl
    └── data/chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

---

## 2. 数据转换

把 R1 Pro 原始数据转成 OpenPI 能直接加载的 LeRobot 数据集。脚本会：

- 把分散字段拼成 **23 维 state / action**（双臂 + 双夹爪 + torso + 底盘速度）
- 只保留 3 路相机（`head_rgb / left_wrist_rgb / right_wrist_rgb`）
- 输出到 `HF_LEROBOT_HOME/<output_repo>/`

```bash
cd openpi
HF_LEROBOT_HOME=/path/to/data uv run python ../scripts/convert_r1pro_chassis_data_newr1pro.py \
    --data_dirs /path/to/raw/dataset_1/lerobot_video \
    --output_repo r1_pro_data_convert_chassis_all \
    --overwrite
```

追加新数据：把 `--overwrite` 换成 `--append`（追加后必须重算 norm stats）。

---

## 3. 计算 Norm Stats

计算数据集的 quantile / mean / std，训练时用于归一化。

```bash
cd openpi
HF_LEROBOT_HOME=/path/to/data uv run python scripts/compute_norm_stats.py \
    --config-name=pi05_r1pro_chassis_all
```

**输出路径**：`openpi/assets/<config_name>/<repo_id>/norm_stats.json`（脚本自动创建，例如 `openpi/assets/pi05_r1pro_chassis_all/r1_pro_data_convert_chassis_all/norm_stats.json`）。训练时会从这里读取，文件不存在会报错。

**什么时候要重跑：**

| 情况 | 重跑？ |
|------|------|
| 首次训练新 config | ✅ |
| 用 `--append` 加了新数据 | ✅ |
| 换了 config（即使数据相同） | ✅ |
| 只改 batch_size / exp_name 等训练超参 | ❌ |
| 只改 `StateCorruptionTransform` 模式 | ❌ |

---

## 4. 训练

```bash
cd openpi
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_LEROBOT_HOME=/path/to/data \
uv run python scripts/train.py pi05_r1pro_chassis_all \
    --exp_name my_run_v1 \
    --batch_size 128 \
    --num_train_steps 30000 \
    --save_interval 500 \
    --keep_period 2500
```

恢复训练加 `--resume`。建议在 tmux 里跑。

Checkpoint 存到 `openpi/checkpoints/<config>/<exp_name>/<step>/`。

---

## 5. 离线评测

```bash
cd openpi
CUDA_VISIBLE_DEVICES=0 uv run python ../scripts/eval_pi05.py \
    --config pi05_r1pro_chassis_all \
    --checkpoint checkpoints/pi05_r1pro_chassis_all/my_run_v1/15000 \
    --test_data /path/to/test_data \
    --action_horizon 50 \
    --sample_interval 10
```

---

## 6. 部署

**启动服务端：**

```bash
cd openpi
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_r1pro_chassis_all \
    --policy.dir checkpoints/pi05_r1pro_chassis_all/my_run_v1/10000
```

**客户端调用：**

```python
from openpi_client import websocket_client_policy as wcp

client = wcp.WebsocketClientPolicy("localhost", 8000)
result = client.infer({
    "head_rgb":        head_image,    # (H,W,3) uint8
    "left_wrist_rgb":  left_wrist,
    "right_wrist_rgb": right_wrist,
    "state":           state_23d,     # (23,) float32
    "prompt":          "Open the door ...",
})
actions = result["actions"]  # (50, 23) float32
```

action 拆分：`[0:7]` 左臂 / `[7:14]` 右臂 / `[14]` 左夹爪 / `[15]` 右夹爪 / `[16:20]` torso / `[20:23]` 底盘 xyz 速度。

---



完整定义见 [openpi/src/openpi/training/config.py](openpi/src/openpi/training/config.py)；架构说明见 [doc/PI05_架构说明.md](doc/PI05_架构说明.md)。
