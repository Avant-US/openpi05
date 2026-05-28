"""R1 Pro 带关键帧链（Keyframe Chaining）的 Policy adapter.

在 R1ProChassisInputs 基础上扩展：把 <= 当前帧的最近 K 个关键帧（默认 K=3）
作为额外图像 keyframe_0_rgb / keyframe_1_rgb / keyframe_2_rgb 注入到 obs.images。

数据流：
1. LeRobot 加载器输出当前帧样本（含 head_rgb/left_wrist_rgb/right_wrist_rgb/state/actions
   以及 episode_index、frame_index 等元信息）
2. DoorKeyframeTransform 读取 dataset 根目录下 meta/keyframes.json
   - key 为 str(episode_index) -> 关键帧帧索引列表（升序）
3. 找出 <= 当前 frame_index 的关键帧，取最近 K 个
4. 用 cv2 解码对应 episode 的 head_rgb 视频，按帧索引取出关键帧图像
5. 不足 K 帧时用黑图 padding，对应 image_mask=False

视频路径解析复用 LeRobotDatasetMetadata.get_video_file_path，自动兼容 v2.1（episode_XXXXXX.mp4）
和 v3.0（file-XXX.mp4 多 episode 合并）格式。
"""

import dataclasses
import functools
import json
import logging
from pathlib import Path

import cv2
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)

ACTION_DIM = 23
DEFAULT_NUM_KEYFRAMES = 3
KEYFRAME_FEATURE = "head_rgb"  # 关键帧只用头部相机，节省 token


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@functools.lru_cache(maxsize=128)
def _open_capture(video_path: str) -> cv2.VideoCapture | None:
    """缓存 cv2.VideoCapture，避免每次 transform 都重新打开。注意：worker fork 后会复制副本。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    return cap


def _read_frame(video_path: Path, frame_idx: int, target_hw: tuple[int, int]) -> np.ndarray:
    """从视频读取指定帧；失败时返回黑图。"""
    h, w = target_hw
    cap = _open_capture(str(video_path))
    if cap is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame


@dataclasses.dataclass
class _KeyframeIndex:
    """关键帧索引（持有 keyframes.json 内容 + LeRobot 元信息）。"""

    repo_id: str
    dataset_root: Path
    # episode_index(int) -> sorted frame indices
    keyframes: dict[int, list[int]]
    # 用于路径解析的 LeRobotDatasetMetadata 引用，惰性加载
    _meta: object = None

    @classmethod
    def load(cls, repo_id: str) -> "_KeyframeIndex":
        # 这里晚导入以避免循环依赖；LeRobot 已在 data_loader 中使用，环境一致
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore

        meta = LeRobotDatasetMetadata(repo_id)
        dataset_root = Path(meta.root)
        kf_file = dataset_root / "meta" / "keyframes.json"
        if not kf_file.exists():
            logger.warning("keyframes.json not found at %s, falling back to empty index", kf_file)
            return cls(repo_id=repo_id, dataset_root=dataset_root, keyframes={}, _meta=meta)
        with kf_file.open() as f:
            raw = json.load(f)
        keyframes = {int(k): sorted(v) for k, v in raw.items()}
        return cls(repo_id=repo_id, dataset_root=dataset_root, keyframes=keyframes, _meta=meta)

    def get_recent(self, episode_idx: int, current_frame: int, k: int) -> list[int]:
        """返回 <= current_frame 的最近 k 个关键帧索引（按时间升序）。"""
        kfs = self.keyframes.get(episode_idx, [0])
        valid = [f for f in kfs if f <= current_frame]
        if not valid:
            valid = [0]
        return valid[-k:]

    def video_path(self, episode_idx: int, vid_key: str) -> Path:
        """返回视频文件绝对路径。"""
        if self._meta is None:
            # fallback to v2.1 layout
            chunk = episode_idx // 1000
            return self.dataset_root / "videos" / f"chunk-{chunk:03d}" / vid_key / f"episode_{episode_idx:06d}.mp4"
        rel = self._meta.get_video_file_path(episode_idx, vid_key)  # type: ignore
        rel = Path(rel)
        return rel if rel.is_absolute() else self.dataset_root / rel


@dataclasses.dataclass(frozen=True)
class R1ProKeyframeInputs(transforms.DataTransformFn):
    """在 R1ProChassisInputs 基础上加入关键帧字段。

    Expected inputs (LeRobot 输出 + DoorKeyframeTransform 注入的关键帧):
    - head_rgb / left_wrist_rgb / right_wrist_rgb: 当前帧三摄像头图像
    - keyframe_<i>_rgb (i in [0, num_keyframes)): 历史关键帧头部图像
    - keyframe_<i>_mask (i in [0, num_keyframes)): bool，True=有效，False=padding
    - state: [23]
    - actions: [horizon, 23] (训练时)
    - prompt: str
    """

    model_type: _model.ModelType = _model.ModelType.PI05
    num_keyframes: int = DEFAULT_NUM_KEYFRAMES

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["head_rgb"])
        left_wrist = _parse_image(data["left_wrist_rgb"])
        right_wrist = _parse_image(data["right_wrist_rgb"])

        images: dict[str, np.ndarray] = {}
        masks: dict[str, np.bool_] = {}

        # 关键帧顺序排在前面，让 embed_prefix 把它们放在 prefix 序列开头。
        for i in range(self.num_keyframes):
            kf_img_key = f"keyframe_{i}_rgb"
            kf_mask_key = f"keyframe_{i}_mask"
            if kf_img_key in data:
                images[kf_img_key] = _parse_image(data[kf_img_key])
                m = data.get(kf_mask_key, np.True_)
                masks[kf_img_key] = np.bool_(m)
            else:
                # 缺失则塞黑图 + mask=False
                images[kf_img_key] = np.zeros_like(base_image)
                masks[kf_img_key] = np.False_

        images.update({
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        })
        masks.update({
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        })

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": images,
            "image_mask": masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class R1ProKeyframeOutputs(transforms.DataTransformFn):
    """裁掉 padding 维度，只返回前 ACTION_DIM 维。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


@dataclasses.dataclass
class DoorKeyframeTransform:
    """训练阶段把历史关键帧图像注入到样本里。

    行为：
    - 在样本 dict 里读 episode_index、frame_index
    - 通过 _KeyframeIndex 查询当前 episode 的关键帧列表
    - 取 <= 当前帧的最近 num_keyframes 个关键帧
    - 用 cv2 从对应 episode 的 head_rgb 视频抽帧（resize 到当前 head_rgb 同形状）
    - 写入 data["keyframe_<i>_rgb"] / data["keyframe_<i>_mask"]
    - 不足的位置用黑图 + mask=False 填充

    注意：本类是 stateful（持有 _KeyframeIndex 缓存），不能 dataclass(frozen=True)。
    多 worker DataLoader 下，每个 worker 会拿自己的副本，cv2.VideoCapture lru_cache 也是
    进程级的，不会跨 worker 冲突。
    """

    repo_id: str
    num_keyframes: int = DEFAULT_NUM_KEYFRAMES
    keyframe_feature: str = KEYFRAME_FEATURE
    # 延迟初始化以避免在 config 注册阶段触发数据集加载/网络请求
    _index: _KeyframeIndex | None = dataclasses.field(default=None, init=False, repr=False)

    def _get_index(self) -> _KeyframeIndex:
        if self._index is None:
            idx = _KeyframeIndex.load(self.repo_id)
            if not idx.keyframes:
                logger.warning(
                    "DoorKeyframeTransform: empty keyframe index for %s. "
                    "All samples will get black-image padding for keyframes.",
                    self.repo_id,
                )
            self._index = idx
        return self._index

    def __call__(self, data: dict) -> dict:
        ep_raw = data.get("episode_index", 0)
        frame_raw = data.get("frame_index", 0)
        episode_idx = int(np.asarray(ep_raw).item()) if hasattr(ep_raw, "item") else int(ep_raw)
        current_frame = int(np.asarray(frame_raw).item()) if hasattr(frame_raw, "item") else int(frame_raw)

        index = self._get_index()
        kf_indices = index.get_recent(episode_idx, current_frame, self.num_keyframes)

        # 当前帧 head_rgb 的形状用作目标尺寸（确保 keyframe 与当前帧分辨率一致）
        head = data.get(self.keyframe_feature)
        if head is None:
            target_hw = (224, 224)
        else:
            head_arr = np.asarray(head)
            if head_arr.ndim == 3 and head_arr.shape[0] == 3:
                target_hw = (head_arr.shape[1], head_arr.shape[2])
            elif head_arr.ndim == 3:
                target_hw = (head_arr.shape[0], head_arr.shape[1])
            else:
                target_hw = (224, 224)

        video_path = index.video_path(episode_idx, self.keyframe_feature)

        # padding 在前，最近的关键帧在后（更接近当前帧）
        num_pad = self.num_keyframes - len(kf_indices)
        for i in range(num_pad):
            data[f"keyframe_{i}_rgb"] = np.zeros((*target_hw, 3), dtype=np.uint8)
            data[f"keyframe_{i}_mask"] = np.False_

        for j, kf_frame in enumerate(kf_indices):
            slot = num_pad + j
            img = _read_frame(video_path, kf_frame, target_hw)
            data[f"keyframe_{slot}_rgb"] = img
            data[f"keyframe_{slot}_mask"] = np.True_

        return data


def make_r1pro_keyframe_example(num_keyframes: int = DEFAULT_NUM_KEYFRAMES) -> dict:
    """Creates a random input example for the R1 Pro keyframe policy."""
    example = {
        "state": np.random.rand(ACTION_DIM).astype(np.float32),
        "head_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "left_wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "right_wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "open the door",
    }
    for i in range(num_keyframes):
        example[f"keyframe_{i}_rgb"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        example[f"keyframe_{i}_mask"] = np.True_
    return example
