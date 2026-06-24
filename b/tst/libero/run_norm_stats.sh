#!/bin/bash
set -euo pipefail

# Compute normalization statistics for pi0.5 LIBERO training.
# Uses locally cached episodes to avoid HuggingFace rate-limiting.

export UV_PROJECT_ENVIRONMENT=/mnt/r/VENV/openpi_venv
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /home/physical/SRC/Robot/openpi05

echo "=== Computing LIBERO norm stats (pi05_libero, local data) ==="
uv run python b/tst/libero/compute_norm_stats_local.py

echo "=== Done. Stats saved to ./assets/ ==="
