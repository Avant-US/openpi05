# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openpi is Physical Intelligence's open-source framework for training and deploying Vision-Language-Action (VLA) models for robot manipulation. It supports three model families: π₀ (flow-based diffusion), π₀-FAST (autoregressive with action tokenization), and π₀.₅ (enhanced π₀). Primary framework is JAX/Flax with secondary PyTorch support for inference and basic finetuning.

## Commands

### Setup
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync          # install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds  # include RLDS/TensorFlow deps (requires Python 3.11)
```

### Testing
```bash
uv run pytest --strict-markers -m "not manual"   # all CI tests
uv run pytest --strict-markers src/openpi/models/pi0_test.py  # single file
uv run pytest --strict-markers -k "test_pi0_full_finetune"    # single test
```
Tests use the `_test.py` suffix convention (not `test_` prefix on files). The `manual` marker excludes tests that require GPU or manual setup from CI.

### Linting & Formatting
```bash
uv run ruff check --fix .    # lint with auto-fix
uv run ruff format .         # format
```
Pre-commit hooks run uv-lock sync check, ruff lint, and ruff format.

### Training
```bash
uv run python scripts/train.py --config <config_name>  # JAX training (tyro CLI)
uv run python scripts/train_pytorch.py --config <config_name>  # PyTorch training
```

### Inference Server
```bash
uv run python scripts/serve_policy.py --config <config_name>
```

## Architecture

### Source Layout
- `src/openpi/models/` — JAX model implementations (Flax NNX modules)
- `src/openpi/models_pytorch/` — PyTorch model implementations
- `src/openpi/policies/` — Policy wrappers connecting models to robot platforms (ALOHA, DROID, LIBERO)
- `src/openpi/training/` — Training loop, data loading, optimization, checkpointing, sharding
- `src/openpi/serving/` — WebSocket inference server
- `src/openpi/shared/` — Utilities: array typing, normalization, downloads, image tools
- `src/openpi/transforms.py` — Composable data transform pipeline
- `packages/openpi-client/` — Minimal client library (Python ≥3.7, few deps)
- `scripts/` — Entry points: train.py, train_pytorch.py, serve_policy.py, compute_norm_stats.py

### Key Abstractions

**Models** inherit from `BaseModel` (nnx.Module) with two core methods: `compute_loss()` and `sample_actions()`. Configured via `BaseModelConfig` frozen dataclasses.

**Data pipeline** is a chain of composable transforms: raw dataset → repack transforms → data transforms (robot-specific) → normalization → model transforms (tokenization, resizing) → `Observation`/`Actions` objects. Transforms implement the `DataTransformFn` protocol.

**Training configs** are frozen `TrainConfig` dataclasses registered in `src/openpi/training/config.py` `_CONFIGS` list. Each config bundles model, data, optimizer, checkpoint, and hyperparameter settings. Selected via `--config <name>` CLI arg (tyro).

**Policies** wrap a model with input/output transforms for inference. `Policy.infer(obs_dict) → actions_dict` is the main interface, shared by both JAX and PyTorch backends.

### Configuration System
All configs live in `src/openpi/training/config.py`. CLI parsing uses `tyro.extras.overridable_config_cli` — you pick a named config and override individual fields. Named configs include: `pi0_aloha`, `pi0_droid`, `pi0_fast_droid`, `pi05_*`, `pi0_libero`, `pi0_fast_libero`, `debug`, etc.

### Type System
Uses `jaxtyping` with `beartype` for runtime array shape/dtype checking. The `@typecheck` decorator and `Array` union type (JAX + PyTorch) are defined in `src/openpi/shared/array_typing.py`.

## Code Style

- Line length: 120
- Python target: 3.11
- Imports: force-single-line, force-sort-within-sections (ruff isort)
- `third_party/` and `src/openpi/models_pytorch/transformers_replace/` are excluded from linting
- `F722` is ignored (conflicts with jaxtyping annotations)
- `T201` is ignored (print statements are used intentionally)

## Workspace Structure

This is a uv workspace. `packages/openpi-client` is a workspace member with its own pyproject.toml and minimal dependencies (Python ≥3.7). The main package depends on it via `openpi-client = { workspace = true }`.

External sources pinned to specific git revisions: LeRobot (HuggingFace) and dlimp.
