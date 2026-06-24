"""Compute normalization statistics using locally cached LIBERO data.

This script works around HuggingFace rate-limiting by restricting the dataset
to only locally-available episodes. It applies the same transforms as the
official pi05_libero config.
"""

import sys
from pathlib import Path
from unittest.mock import patch

OPENPI_ROOT = Path("/home/physical/SRC/Robot/openpi05")
sys.path.insert(0, str(OPENPI_ROOT / "src"))

import numpy as np
import tqdm

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def get_local_episode_count(root: Path) -> int:
    """Count locally available parquet episode files."""
    count = 0
    data_dir = root / "data"
    if data_dir.exists():
        for chunk_dir in sorted(data_dir.iterdir()):
            if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                count += sum(1 for f in chunk_dir.iterdir() if f.suffix == ".parquet")
    return count


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main():
    config_name = "pi05_libero"
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    from lerobot.common.constants import HF_LEROBOT_HOME

    local_root = HF_LEROBOT_HOME / data_config.repo_id
    available_episodes = get_local_episode_count(local_root)
    print(f"Found {available_episodes} locally cached episodes at {local_root}")

    if available_episodes == 0:
        print("ERROR: No local data found. Please download the dataset first.")
        sys.exit(1)

    original_create = _data_loader.create_torch_dataset

    def patched_create_torch_dataset(data_cfg, action_horizon, model_cfg):
        from lerobot.common.datasets import lerobot_dataset

        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_cfg.repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            data_cfg.repo_id,
            episodes=list(range(available_episodes)),
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_cfg.action_sequence_keys
            },
        )

        if data_cfg.prompt_from_task:
            dataset = _data_loader.TransformedDataset(
                dataset, [transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        return dataset

    with patch.object(_data_loader, "create_torch_dataset", patched_create_torch_dataset):
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, max_frames=10000
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing norm stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")


def create_torch_dataloader(data_config, action_horizon, batch_size, model_config, max_frames=None):
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=2,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


if __name__ == "__main__":
    main()
