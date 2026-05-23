# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openpi is Physical Intelligence's open-source robotics framework providing vision-language-action (VLA) models:
- **π₀**: Flow-matching VLA (PaliGemma vision-language + 300M action expert)
- **π₀-FAST**: Autoregressive VLA using FSQ action tokenizer
- **π₀.₅**: Upgraded π₀ with knowledge insulation and discrete state input (`pi05=True` flag on `Pi0Config`)

All models support both JAX (Flax NNX) and PyTorch backends. Base model checkpoints are pre-trained on 10k+ hours of robot data.

## Common Commands

```bash
# Install (requires uv)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Lint
uv run ruff check . --fix
uv run ruff format .

# Test (excludes manual-tagged tests)
uv run pytest --strict-markers -m "not manual"

# Run a single test
uv run pytest src/openpi/models/pi0_test.py
uv run pytest src/openpi/models/pi0_test.py::test_function_name

# Train (JAX)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name>

# Train (PyTorch, multi-GPU)
uv run torchrun --nproc_per_node=<N> scripts/train_pytorch.py <config_name> --exp-name=<name>

# Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Serve policy over websocket
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir>
```

Config names (e.g., `pi0_base`, `pi05_libero`, `pi0_fast_droid`) are defined in `src/openpi/training/config.py` in the `_CONFIGS` list. Use `get_config(name)` to load them.

## Architecture

### Data Pipeline

```
Raw Data (LeRobot/RLDS) → repack_transforms → data_transforms → normalization → model_transforms → Model
```

- **repack_transforms**: Adapt dataset-specific formats to a common schema
- **data_transforms**: Robot-specific I/O mapping (e.g., `DroidInputs`, `LiberoInputs`, `AlohaInputs`)
- **model_transforms**: Model-specific preprocessing (tokenization, image resizing)

Transforms are composed via `Group` dataclass with `inputs`/`outputs` lists in `src/openpi/transforms.py`.

### Key Source Layout

- `src/openpi/models/` — Model definitions: `pi0.py` (π₀), `pi0_fast.py` (π₀-FAST), `pi0_config.py` (configs), `model.py` (BaseModel, Observation/Actions), `gemma.py`, `siglip.py`, `tokenizer.py`
- `src/openpi/models_pytorch/` — PyTorch implementations (includes `transformers_replace/` patches)
- `src/openpi/policies/` — Policy wrappers and robot-specific transforms (`droid_policy.py`, `libero_policy.py`, `aloha_policy.py`)
- `src/openpi/training/` — Training pipeline: `config.py` (all named configs), `optimizer.py`, `weight_loaders.py`, `checkpoints.py`
- `src/openpi/serving/` — WebSocket server for remote inference (msgpack serialization)
- `src/openpi/shared/` — Utilities: `normalize.py` (NormStats, RunningStats), `download.py`, array typing
- `packages/openpi-client/` — Minimal-dependency client package with `Runtime`, `Environment`, `Agent` abstractions
- `scripts/` — Entry points for training, serving, norm stats computation
- `examples/` — Per-robot examples (libero, droid, aloha, aloha_sim, simple_client)

### Model Architecture

All models use a dual-stream design: PaliGemma (SigLIP vision encoder + Gemma LLM, 2-3B params) fused with a separate Gemma-based action expert (~300M params). π₀/π₀.₅ use a flow matching head for continuous actions; π₀-FAST uses autoregressive token generation.

### Configuration System

`TrainConfig` in `src/openpi/training/config.py` is the master config dataclass containing model config, data config, optimizer config, weight loader, and sharding settings. CLI uses `tyro` for automatic argument parsing. The first positional argument to training scripts is the config name.

### Inference Path

```python
config = get_config("pi05_droid")
policy = create_trained_policy(config, checkpoint_dir)
output = policy.infer(obs_dict)  # returns {"actions": ..., "policy_timing": ...}
```

`Policy` (in `src/openpi/policies/policy.py`) wraps a model with its transforms and handles normalization/denormalization.

## Code Style

- Python 3.11+, line length 120
- Ruff for linting and formatting (config in `pyproject.toml`)
- Imports: force single-line, sorted within sections (isort via ruff)
- `transformers_replace/` files are excluded from ruff
- Tests use pytest; mark slow/hardware tests with `@pytest.mark.manual`
- Test paths: `src/`, `scripts/`, `packages/`

## Important Details

- JAX models use Flax NNX (stateful modules), JIT-compiled via `module_jit()`
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for JAX GPU training
- PyTorch backend requires patched transformers files in `src/openpi/models_pytorch/transformers_replace/`
- Normalization stats are pre-computed per robot platform (`asset_id`) and stored in checkpoint `assets/` dir
- Images: 224×224 RGB, uint8 or float32 in [-1, 1]
- The `openpi-client` package is a workspace member (`packages/openpi-client/`) with its own dependencies
- LeRobot is pinned to a specific git revision; RLDS support requires Python 3.11 and `uv sync --group rlds`
- `uv` workspace is configured in `pyproject.toml` with `packages/*` as members
