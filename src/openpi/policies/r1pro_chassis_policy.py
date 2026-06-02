"""R1 Pro robot policy adapter for OpenPI (with chassis).

Maps R1 Pro data fields to pi0/pi0.5 model expected format.

State/Action: 23-dim = left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1) + torso(4) + chassis(3)
Cameras: head_rgb -> base_0_rgb, left_wrist_rgb -> left_wrist_0_rgb, right_wrist_rgb -> right_wrist_0_rgb
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 23


def make_r1pro_chassis_example() -> dict:
    """Creates a random input example for the R1 Pro chassis policy."""
    return {
        "state": np.random.rand(ACTION_DIM).astype(np.float32),
        "head_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "left_wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "right_wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class R1ProChassisInputs(transforms.DataTransformFn):
    """Converts R1 Pro inputs (with chassis) to model expected format.

    Expected inputs:
    - head_rgb: image [C, H, W] or [H, W, C]
    - left_wrist_rgb: image
    - right_wrist_rgb: image
    - state: [23]
    - actions: [action_horizon, 23] (training only)
    - prompt: str
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["head_rgb"])
        left_wrist = _parse_image(data["left_wrist_rgb"])
        right_wrist = _parse_image(data["right_wrist_rgb"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class R1ProChassisOutputs(transforms.DataTransformFn):
    """Converts model outputs back to R1 Pro action format (with chassis).

    Returns only the first 23 dims (strips zero-padding).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


@dataclasses.dataclass
class StateCorruptionTransform:
    """训练时以概率 prob 破坏 state 输入，打破 state->action 同构捷径。

    在开门等任务中，pi0.5 容易学到 "当前 state 形如某个推门姿态 -> 输出继续推门
    的 action" 这条 state->action 捷径，导致 "门没开继续戳门" 的失败模式。本
    transform 通过在训练时按概率破坏 state，让模型必须从视觉中获取真正的环境
    信息（比如 "门是否已经开了"），而不是只看 state 模式匹配。

    模式（mode 字段）：
    - "normal": 原样透传，no-op（**类的默认值**；用于 baseline / A-B 对比的
      "对照组"。仓库默认 4 个 R1Pro chassis 系列 config 也都用这个值，使加完
      代码后 baseline 行为完全不变。要做扰动实验时手动改成 "zero" 或 "swap"
      并换新 --exp_name 启动。）
    - "zero": 以 prob 概率把 state 整体置 0。语义干净 —— "信息缺失"。pi0.5
      会把 state 离散化进 prompt 文本，全 0 落到固定 bin，模型会学到 "该 bin =
      state 不可信，看视觉"。最接近 State-free Policy (arXiv:2509.18644) 思路。
    - "swap": 以 prob 概率从最近见过的 state reservoir 缓冲区随机抽一个替换。
      更激进，但腕部图像会与 state 不一致；冲突信号更强。

    仅训练时启用：通过判定 data 里是否含有 "actions" 键来区分训练 / 推理。
    推理时（policy_config.create_trained_policy 路径）输入 dict 不含 actions，
    会直接 return data 跳过破坏。norm_stats 计算路径虽然有 actions，但本
    transform 默认 mode="normal"，对 norm_stats 无影响；即便切到 "zero"/"swap"
    也是从同一分布采样，统计上几乎等价（按文档约定不重算 norm_stats）。

    本类是 stateful（swap 模式下的 reservoir 与 RNG），不能 dataclass(frozen=True)。
    多 worker DataLoader 下每个 worker 拥有自己的副本，互不冲突。
    """

    prob: float = 0.15
    mode: str = "normal"  # "normal" | "zero" | "swap"
    buffer_size: int = 1024  # 仅 swap 模式使用
    _buf: list = dataclasses.field(default_factory=list, init=False, repr=False)
    _rng: object = dataclasses.field(default=None, init=False, repr=False)

    def __call__(self, data: dict) -> dict:
        # mode="normal" 是 no-op，最快路径直接返回，避免任何额外开销。
        if self.mode == "normal":
            return data

        # 每个 worker 进程第一次调用时才初始化 RNG，确保 fork 后各 worker
        # 各自从 OS 熵源拿种子，互不重复。
        if self._rng is None:
            self._rng = np.random.default_rng()

        # train gate: 推理路径不含 "actions" 键，直接放行。
        if "state" not in data or "actions" not in data:
            return data

        state = np.asarray(data["state"], dtype=np.float32)

        if self.mode == "zero":
            if self._rng.random() < self.prob:
                data = {**data, "state": np.zeros_like(state)}
            return data

        if self.mode == "swap":
            # 冷启动期：缓冲区填满前先攒数据，不替换，避免初期把 state
            # 替换成同一帧自身。
            if len(self._buf) < self.buffer_size:
                self._buf.append(state.copy())
                return data
            if self._rng.random() < self.prob:
                idx = int(self._rng.integers(0, self.buffer_size))
                data = {**data, "state": self._buf[idx].copy()}
            # reservoir 滚动更新：随机一个槽换成当前 state，保持缓冲区
            # 始终反映训练流的最近样本分布。
            self._buf[int(self._rng.integers(0, self.buffer_size))] = state.copy()
            return data

        raise ValueError(
            f"Unknown StateCorruptionTransform mode: {self.mode!r}. "
            f"Expected one of: 'normal', 'zero', 'swap'."
        )
