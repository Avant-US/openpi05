#!/bin/bash
set -euo pipefail

# Pi0.5 LIBERO 8-GPU training script.
# Global batch size: 128, 1000 steps, checkpoint every 200 steps.
# All optimizations enabled: FSDP, EMA, bfloat16, gradient clipping, cosine LR warmup.

export UV_PROJECT_ENVIRONMENT=/mnt/r/VENV/openpi_venv
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jax_compilation_cache"

cd /home/physical/SRC/Robot/openpi05

echo "=== Pi0.5 LIBERO Training ==="
echo "GPUs: 8, Batch: 128, Steps: 1000, FSDP: 8, EMA: 0.999"
echo "Checkpoint interval: 200 steps"
echo "LR: cosine decay, warmup=100, peak=5e-5, decay to 5e-6 over 1000 steps"
echo ""

uv run python b/tst/libero/train_pi05_libero.py pi05_libero \
    --exp-name=pi05_libero_8gpu_1k \
    --batch-size=128 \
    --num-train-steps=1000 \
    --save-interval=200 \
    --log-interval=50 \
    --fsdp-devices=8 \
    --ema-decay=0.999 \
    --lr-schedule.warmup-steps=100 \
    --lr-schedule.peak-lr=5e-5 \
    --lr-schedule.decay-steps=1000 \
    --lr-schedule.decay-lr=5e-6 \
    --overwrite
