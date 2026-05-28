# 将 KC-VLA 关键帧策略移植到 Pi0.5 的实施方案

## 目标
将 KC-VLA 的关键帧链（Keyframe Chaining）策略应用到 Pi0.5，使其具备历史感知能力，解决 Non-Markovian 任务（如开门）中"不知道之前发生了什么"的问题。

---
## 整体方案
```
KC-VLA 的策略:
  KSM(关键帧选择) + 稀疏历史(≤6帧) + VLA(Eagle2.5)

移植到 Pi0.5:
  KSM(可直接复用) + 稀疏历史(≤K帧) + Pi0.5(SigLIP + 双流Gemma)
```
需要改动的层次：
| 层次 | 改什么 | 难度 |
|------|--------|------|
| 数据层 | DataTransformFn 中加载历史帧 | 低 |
| 模型层 | embed_prefix 支持多帧输入 | 中 |
| 训练层 | 数据集提供 keyframes.json | 低 |
| 推理层 | 在线累积关键帧 | 中 |
| KSM | 直接复用 KC-VLA 的 KSM 或自训一个 | 独立 |

---
## 第一步：准备关键帧标注

整体流程：**人工标注少量(30-50条) → 训练 KSM → KSM 自动标注剩余所有数据**

### 1.1 你的数据流程
```
原始数据: /home/likaixin/kaizhe_ws/data/r1_pro_data_raw/locomanipulation
     ↓
转换脚本: /home/likaixin/kaizhe_ws/pi0.5/scripts/convert_r1pro_chassis_data.py
     ↓
LeRobot数据集: ~/.cache/huggingface/lerobot/r1_pro_pull_door_chassis/
```
转换后的目录结构：
```
r1_pro_pull_door_chassis/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   └── keyframes.json              ← 你要在这里加上关键帧标注
├── data/
│   └── chunk-000/
│       └── file-000.parquet        (state 23维, actions 23维)
└── videos/
    ├── head_rgb/
    │   └── chunk-000/
    │       ├── file-000.mp4        ← episode 0 (头部摄像头)
    │       ├── file-001.mp4        ← episode 1
    │       └── ...
    ├── left_wrist_rgb/
    │   └── chunk-000/...
    └── right_wrist_rgb/
        └── chunk-000/...
```
**注意**：你的视频命名是 `file-XXX.mp4`（LeRobot 默认格式），不是 KC-VLA 的 `episode_XXXXXX.mp4`。后面标注和 KSM 代码需要适配这个路径格式。
摄像头3个：head_rgb, left_wrist_rgb, right_wrist_rgb。
关键帧标注主要看 **head_rgb**（全局视角，最容易看到门的状态变化）。

### 1.2 人工标注 keyframes.json（第一步，只标 30-50 条轨迹）
**工具**：用下面的 Python 脚本逐帧翻看视频，按 k 标记关键帧。

标注辅助脚本（三摄像头同时显示，直接在 raw 目录视频上标注）：
```python
import cv2
import json
import argparse
from pathlib import Path

# 在 raw 目录上标注，标注结果存到 raw 目录的 meta/keyframes.json
# 用法: python annotate_keyframes.py --data_dir /path/to/push_door_data0402/open_door/lerobot_video

def get_cameras(data_dir: Path) -> dict:
    """自动识别 raw 目录中的摄像头视频路径"""
    videos_dir = data_dir / "videos"
    return {
        "head":       videos_dir / "observation.images.head_rgb" / "chunk-000",
        "left_wrist": videos_dir / "observation.images.left_wrist_rgb" / "chunk-000",
        "right_wrist": videos_dir / "observation.images.right_wrist_rgb" / "chunk-000",
    }

def annotate_episode(cameras: dict, episode_id: int):
    """人工标注一条轨迹的关键帧，同时显示三个摄像头"""
    # 打开三个视频
    caps = {}
    for cam_name, cam_dir in cameras.items():
        video_path = cam_dir / f"file-{episode_id:03d}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  警告: 无法打开 {video_path}")
            return None
        caps[cam_name] = cap

    total = int(caps["head"].get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = [0]
    idx = 0
    paused = True

    print(f"\n=== Episode {episode_id} | 总帧数: {total} ===")
    print(f"  操作: k=标记关键帧, 空格=播放/暂停, f=前进1帧, d=后退1帧, q=完成")

    def read_all_frames():
        """从三个摄像头同时读取当前帧"""
        frames = {}
        for cam_name, cap in caps.items():
            ret, frame = cap.read()
            frames[cam_name] = frame if ret else None
        return frames

    def seek_all(frame_idx):
        """三个摄像头同时 seek 到指定帧"""
        for cap in caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def build_display(frames):
        """拼接三个摄像头画面 + 标注信息"""
        # 头部摄像头放上面（大图），左右手放下面
        head = frames.get("head")
        left = frames.get("left_wrist")
        right = frames.get("right_wrist")
        if head is None:
            return None

        # 统一高度
        h_target = 360
        head_resized = cv2.resize(head, (640, h_target))
        left_resized = cv2.resize(left, (320, 240)) if left is not None else np.zeros((240, 320, 3), dtype=np.uint8)
        right_resized = cv2.resize(right, (320, 240)) if right is not None else np.zeros((240, 320, 3), dtype=np.uint8)

        # 左右手横向拼接
        wrists = cv2.hconcat([left_resized, right_resized])

        # 上下拼接：头部 + 手腕
        display = cv2.vconcat([head_resized, wrists])

        # 写标注信息
        info = f"Ep:{episode_id} Frame:{idx}/{total} KF:{keyframes}"
        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "HEAD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, "LEFT WRIST", (10, h_target + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, "RIGHT WRIST", (330, h_target + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return display

    # 读第一帧
    seek_all(0)
    frames = read_all_frames()

    while True:
        display = build_display(frames)
        if display is not None:
            cv2.imshow("Annotate Keyframes (3 cameras)", display)

        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord('k'):
            if idx not in keyframes:
                keyframes.append(idx)
                print(f"  ✓ 标记关键帧: 第{idx}帧")
        elif key == ord(' '):
            paused = not paused
        elif key == ord('f'):  # 前进1帧
            frames = read_all_frames()
            if frames["head"] is not None:
                idx += 1
            paused = True
        elif key == ord('d'):  # 后退1帧
            idx = max(0, idx - 1)
            seek_all(idx)
            frames = read_all_frames()
            paused = True
        elif key == ord('q'):
            break
        elif not paused:
            frames = read_all_frames()
            if frames["head"] is None:
                break
            idx += 1

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    return sorted(keyframes)

def main():
    import numpy as np  # build_display 里用到

    parser = argparse.ArgumentParser(description="在 raw 视频上标注关键帧")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="raw 数据的 lerobot_video 目录路径")
    parser.add_argument("--num_annotate", type=int, default=50,
                        help="标注多少条轨迹 (默认50)")
    args = parser.parse_args()

    data_dir = args.data_dir
    cameras = get_cameras(data_dir)

    # 找到所有 episode 视频
    head_videos = sorted(cameras["head"].glob("file-*.mp4"))
    print(f"数据目录: {data_dir}")
    print(f"找到 {len(head_videos)} 条轨迹")
    print(f"三个摄像头同时显示：上方=头部(全局), 下方左=左手, 下方右=右手\n")

    # 加载已有标注（支持断点续标）
    save_path = data_dir / "meta" / "keyframes.json"
    if save_path.exists():
        with open(save_path) as f:
            all_keyframes = json.load(f)
        print(f"加载已有标注: {len(all_keyframes)} 条")
    else:
        all_keyframes = {}

    num_to_annotate = min(args.num_annotate, len(head_videos))

    for i in range(num_to_annotate):
        if str(i) in all_keyframes:
            print(f"  Episode {i}: 已标注，跳过")
            continue

        kfs = annotate_episode(cameras, episode_id=i)
        if kfs is not None:
            all_keyframes[str(i)] = kfs
            print(f"  Episode {i} 完成: {kfs}")

        # 每标5条保存一次
        if (i + 1) % 5 == 0:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(all_keyframes, f, indent=2)
            print(f"  [自动保存] 已标注 {len(all_keyframes)} 条")

    # 最终保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_keyframes, f, indent=2)
    print(f"\n标注完成！共 {len(all_keyframes)} 条轨迹")
    print(f"保存到: {save_path}")
    print(f"\n之后运行 convert 脚本时会自动携带这些标注到转换后的数据集。")

if __name__ == "__main__":
    main()
```

### 1.3 标注规则（开门任务）
标注时关注**状态发生不可逆变化**的瞬间：
```
关键帧1: 第0帧（初始状态，自动标）
关键帧2: 手接触到门把手的瞬间
关键帧3: 门把手被完全按下的瞬间
关键帧4: 门开始移动的瞬间（门缝出现）
关键帧5: 门完全打开
```
**不要标的**：中间的过渡帧（手在移动但还没碰到把手、门在慢慢打开中）。
**最终 keyframes.json 示例**：
```json
{
    "0": [0, 35, 52, 61, 78],
    "1": [0, 40, 55, 68, 85],
    "2": [0, 28, 45, 55, 72],
    "3": [0, 33, 50, 60, 80]
}
```
- key：轨迹 ID（字符串）
- value：该轨迹的关键帧**帧索引**列表（升序，第一个必须是 0）
- 每条轨迹通常 3-6 个关键帧

### 1.4 训练 KSM（用你标注的 30-50 条数据）
**KSM = Keyframe Selection Module**，是一个独立的小模型，专门判断"当前帧是不是关键帧"。
它和 Pi0.5 是**分开训练的**，训好后用来给剩余数据自动打关键帧标签。
```
人工标注50条 → 训练KSM → KSM自动标注剩余几百条 → 用全量标注训练Pi0.5
```
**但是**：KC-VLA 的 KSM 代码期望的视频路径格式是 `episode_XXXXXX.mp4`，而你的是 `file-XXX.mp4`。
需要适配一下路径逻辑。最简单的方式是建个软链接或改 KSM 的 `_find_video_subdir` 函数。

#### 适配方案：建软链接把你的视频映射成 KSM 期望的格式
```bash
# 创建一个 KSM 能识别的目录结构
mkdir -p /tmp/ksm_door_data/videos/chunk-000/observation.images.image/
mkdir -p /tmp/ksm_door_data/meta/

# 软链接视频（file-000.mp4 → episode_000000.mp4）
LEROBOT_DIR=~/.cache/huggingface/lerobot/r1_pro_pull_door_chassis
for f in $LEROBOT_DIR/videos/head_rgb/chunk-000/file-*.mp4; do
    num=$(basename "$f" | sed 's/file-\([0-9]*\).mp4/\1/')
    ln -sf "$f" "/tmp/ksm_door_data/videos/chunk-000/observation.images.image/episode_$(printf '%06d' $num).mp4"
done

# 复制你标注好的 keyframes.json
cp $LEROBOT_DIR/meta/keyframes.json /tmp/ksm_door_data/meta/keyframes.json
```

#### Stage1：对比学习（让模型学会区分"不同阶段的帧看起来不一样"）
修改 `KC-VLA/keyframe_selection_module/train_stage1.py` 中的配置：
```python
TASKS_CONFIG = {
    0: "/tmp/ksm_door_data"   # 指向软链接目录
}
```
运行：
```bash
cd /home/likaixin/kaizhe_ws/pi0.5/KC-VLA
python keyframe_selection_module/train_stage1.py
# 输出: checkpoints/stage1/best_backbone.pth
```
#### Stage2：二分类（让模型判断"这一帧是不是关键帧"）
修改 `KC-VLA/keyframe_selection_module/train_stage2.py` 中的配置：
```python
TASKS_CONFIG = {
    0: "/tmp/ksm_door_data"
}
STAGE1_WEIGHTS = "./checkpoints/stage1/best_backbone.pth"
```
运行：
```bash
python keyframe_selection_module/train_stage2.py
# 输出: checkpoint/stage2/best_model_stage2.pth
```
#### Stage3：搜阈值 + 对剩余数据自动标注
```bash
python keyframe_selection_module/threshold_finding.py \
    --ckpt checkpoint/stage2/best_model_stage2.pth \
    --data_root /tmp/ksm_door_data
```
完成后把生成的 keyframes.json 拷回你的 LeRobot 数据集：
```bash
cp /tmp/ksm_door_data/meta/keyframes.json \
   ~/.cache/huggingface/lerobot/r1_pro_pull_door_chassis/meta/keyframes.json
```

### 1.5 验证标注质量
标完后跑一个可视化脚本确认 KSM 标得对不对：
```python
import cv2
import json
from pathlib import Path

DATASET_ROOT = Path("~/.cache/huggingface/lerobot/r1_pro_pull_door_chassis").expanduser()
VIDEO_DIR = DATASET_ROOT / "videos" / "head_rgb" / "chunk-000"

def visualize_keyframes(video_path, keyframes, output_path=None):
    """可视化某条轨迹的关键帧，横向拼接"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for kf_idx in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf_idx)
        ret, frame = cap.read()
        if ret:
            # 缩小方便拼接
            frame = cv2.resize(frame, (320, 180))
            cv2.putText(frame, f"KF@{kf_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frames.append(frame)
    cap.release()
    if frames:
        concat = cv2.hconcat(frames)
        if output_path:
            cv2.imwrite(str(output_path), concat)
            print(f"  已保存: {output_path}")
        return concat

with open(DATASET_ROOT / "meta" / "keyframes.json") as f:
    kf_data = json.load(f)

# 随机抽几条 KSM 自动标注的看看（ep >= 50 是 KSM 标的）
check_dir = Path("keyframe_check_results")
check_dir.mkdir(exist_ok=True)
for ep_id in ["50", "51", "52", "60", "70"]:
    if ep_id not in kf_data:
        continue
    video = VIDEO_DIR / f"file-{int(ep_id):03d}.mp4"
    if video.exists():
        visualize_keyframes(video, kf_data[ep_id], check_dir / f"check_ep{ep_id}.png")
```

---
## 第二步：修改数据层（DataTransformFn）
> 📁 改动文件：新建 `src/openpi/policies/your_door_policy.py`

Pi0.5 的数据流：`原始obs → DataTransformFn → model输入格式`。
需要让 transform 把历史关键帧也塞进 `images` 字典里。

```python
import json
import numpy as np
from openpi import transforms as _transforms

class DoorKeyframeTransform(_transforms.DataTransformFn):
    """把关键帧图像加入 obs 的 images 字典"""

    def __init__(self, keyframes_path: str, max_keyframes: int = 3):
        with open(keyframes_path) as f:
            self.keyframes = json.load(f)
        self.max_keyframes = max_keyframes

    def __call__(self, data: dict) -> dict:
        traj_id = str(data["trajectory_id"])
        current_idx = data["frame_index"]

        # 取 ≤ 当前帧的关键帧
        all_kfs = self.keyframes.get(traj_id, [0])
        valid_kfs = [k for k in all_kfs if k < current_idx]

        # 最多取 max_keyframes 帧
        selected_kfs = valid_kfs[-self.max_keyframes:]

        # 把关键帧图像加到 images 字典中
        # Pi0.5 的 embed_prefix 会遍历 obs.images 的所有 key
        for i, kf_idx in enumerate(selected_kfs):
            kf_image = data["all_frames"][kf_idx]  # 需要数据集支持按索引取帧
            data["images"][f"keyframe_{i}_rgb"] = kf_image

        # 关键帧不足时用黑图 padding，设置 mask=False
        for i in range(len(selected_kfs), self.max_keyframes):
            data["images"][f"keyframe_{i}_rgb"] = np.zeros_like(
                data["images"]["base_0_rgb"]
            )

        return data
```

---
## 第三步：修改模型层（embed_prefix）
> 📁 改动文件：`src/openpi/models/pi0.py` 中的 `embed_prefix`

**核心改动**：让 SigLIP 也处理关键帧图像，把关键帧的视觉 token 拼在当前帧前面。

```python
@at.typecheck
def embed_prefix(
    self, obs: _model.Observation
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    input_mask = []
    ar_mask = []
    tokens = []

    # ===== 新增：先处理关键帧（按名称排序保证顺序） =====
    keyframe_names = sorted([k for k in obs.images if k.startswith("keyframe_")])
    for name in keyframe_names:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
        )
        # 关键帧之间互相可见（bidirectional）
        ar_mask += [False] * image_tokens.shape[1]

    # ===== 原有逻辑：处理当前帧 =====
    current_names = [k for k in obs.images if not k.startswith("keyframe_")]
    for name in current_names:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
        )
        ar_mask += [False] * image_tokens.shape[1]

    # 语言 token
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask
```

改动后的输入序列：
```
原来:
  [当前base₂₅₆, 当前left₂₅₆, 当前right₂₅₆, 语言L]

改后:
  [KF0₂₅₆, KF1₂₅₆, KF2₂₅₆, 当前base₂₅₆, 当前left₂₅₆, 当前right₂₅₆, 语言L]
   \______关键帧______/     \_____________当前帧______________/
```

---
## 第四步：修改 Observation 和 IMAGE_KEYS
> 📁 改动文件：`src/openpi/models/model.py`

```python
# 原来固定3个 key
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

# 改成动态的（或扩展固定的）
IMAGE_KEYS = (
    "keyframe_0_rgb",
    "keyframe_1_rgb",
    "keyframe_2_rgb",
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
```
同时确保 `preprocess_observation` 能处理这些新 key（它遍历 `image_keys`，只要名字匹配就行）。

---
## 第五步：修改推理逻辑（在线关键帧管理）
> 📁 新建：推理时的关键帧 buffer

```python
import numpy as np
import torch
from keyframe_selection_module.model.network import TransformerKeyframeSelector

class KeyframeManager:
    """推理时在线管理关键帧"""

    def __init__(self, ksm_path: str, threshold: float = 0.5, max_keyframes: int = 3):
        # 加载 KSM 模型
        self.ksm = TransformerKeyframeSelector()
        self.ksm.load_state_dict(torch.load(ksm_path))
        self.ksm.eval()
        self.threshold = threshold
        self.max_keyframes = max_keyframes

        self.keyframes = []       # 存储关键帧图像
        self.prev_frames = []     # 滑动窗口（KSM 需要 3 帧窗口）

    def update(self, current_image: np.ndarray) -> list[np.ndarray]:
        """每步调用，返回当前应该喂给模型的关键帧列表"""
        self.prev_frames.append(current_image)
        if len(self.prev_frames) > 3:
            self.prev_frames.pop(0)

        # KSM 判断当前帧是否为关键帧
        if len(self.prev_frames) == 3:
            score = self._run_ksm(self.prev_frames)
            if score > self.threshold:
                self.keyframes.append(current_image.copy())
                # 超过上限则丢弃最旧的（但保留第一帧）
                if len(self.keyframes) > self.max_keyframes:
                    self.keyframes.pop(1)

        # 返回关键帧列表（不足则 padding 黑图）
        result = list(self.keyframes[-self.max_keyframes:])
        while len(result) < self.max_keyframes:
            result = [np.zeros_like(current_image)] + result
        return result

    def _run_ksm(self, window: list[np.ndarray]) -> float:
        with torch.no_grad():
            # 图像预处理 + 推理
            frames = torch.stack([self._preprocess(f) for f in window])
            logit = self.ksm(frames.unsqueeze(0))
            return torch.sigmoid(logit).item()

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img
```

推理循环：
```python
from openpi_client import websocket_client

# 初始化
policy = load_pi05_policy(checkpoint_path)
kf_manager = KeyframeManager(ksm_path="ksm_stage2.pth", threshold=0.5)

while not done:
    # 获取当前观测
    obs = env.get_observation()

    # 更新关键帧 buffer
    keyframes = kf_manager.update(obs["images"]["base_0_rgb"])

    # 把关键帧塞入 obs
    for i, kf in enumerate(keyframes):
        obs["images"][f"keyframe_{i}_rgb"] = kf

    # 推理
    action = policy.infer(obs)
    env.step(action)
```

---
## 第六步：训练流程
### 数据集准备
```
your_dataset/
├── meta/
│   └── keyframes.json          # 步骤一生成的
├── videos/
│   └── episode_000000.mp4
├── states.parquet
└── actions.parquet
```
### 训练命令
和原来一样，只是 policy 配置指向你新写的 transform：
```bash
python scripts/train.py \
    --config your_door_config \
    --overrides "data.keyframes_path=meta/keyframes.json"
```
### 训练配置改动
在 `src/openpi/training/config.py` 中新增一个训练配置，指定：
- 使用 `DoorKeyframeTransform`
- `IMAGE_KEYS` 包含 `keyframe_*_rgb`
- `max_token_len` 增大（因为序列变长了）

---
## 计算开销评估
| 配置 | prefix token 数 | 相对原来 | 内存影响 |
|------|----------------|---------|---------|
| 原始 Pi0.5 (3摄像头) | 256×3 = 768 | 1× | baseline |
| +1 关键帧 | 256×4 = 1024 | 1.8× attention | +~33% |
| +2 关键帧 | 256×5 = 1280 | 2.8× attention | +~67% |
| +3 关键帧 | 256×6 = 1536 | 4× attention | +~100% |

**优化手段**：
1. 关键帧只用 base 摄像头（不用腕部），减少 token
2. 用 KV cache（推理时关键帧不变，缓存其 KV，只计算新帧）
3. 关键帧过一个 pooling 压缩 token 数（256→64）

---
## 实施优先级

### Phase 1（1-2天）：验证可行性
- [ ] 用启发式规则生成 keyframes.json
- [ ] 修改 embed_prefix 支持 keyframe_* 图像
- [ ] 修改 IMAGE_KEYS
- [ ] 训练一个小实验看 loss 下降

### Phase 2（3-5天）：完整流程
- [ ] 训练 KSM（复用 KC-VLA 代码）
- [ ] 写完整的 DoorKeyframeTransform
- [ ] 全量训练 Pi0.5 + 关键帧
- [ ] 推理时接入 KeyframeManager

### Phase 3（后续优化）
- [ ] 关键帧 token 压缩（pooling 256→64）
- [ ] KV cache 优化
- [ ] 对比实验：无关键帧 vs 1帧 vs 3帧

---
## 文件改动清单
| 文件 | 改动 |
|------|------|
| `src/openpi/models/model.py` | IMAGE_KEYS 扩展 |
| `src/openpi/models/pi0.py` | embed_prefix 支持 keyframe 图像 |
| `src/openpi/policies/door_policy.py` | **新建**，DoorKeyframeTransform |
| `src/openpi/training/config.py` | 新增 door 训练配置 |
| `scripts/inference_with_keyframes.py` | **新建**，KeyframeManager + 推理循环 |
| `meta/keyframes.json` | **生成**，关键帧标注 |

---
## 关键区别：KC-VLA vs 我们的做法

| | KC-VLA | 我们（Pi0.5 + KC策略） |
|--|--------|----------------------|
| 关键帧输入方式 | 6帧×多视角，Eagle2.5 一次处理 | K帧各过 SigLIP → token 拼接 |
| 骨干 | Eagle2.5-VL (3B) | PaliGemma (2B) + 动作专家 (300M) |
| 历史帧如何影响动作 | 骨干 cross-attention | 双流 Gemma 的联合注意力 |
| KSM | ResNet+Transformer 两阶段 | 可复用 KC-VLA 的 KSM |
| Flow Matching | DiT action head | 动作专家 Gemma 内置 |
| 框架 | PyTorch | JAX (或 PyTorch) |
