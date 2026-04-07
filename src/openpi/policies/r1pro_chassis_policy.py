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
