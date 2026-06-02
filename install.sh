#!/usr/bin/env bash
#
# openpi (R1 Pro / pi0.5) 一键安装脚本
#
# 适用环境：
#   - 操作系统：Ubuntu 22.04（官方仅测试此版本）
#   - GPU：NVIDIA RTX 5090 / H200 / A100 / RTX 4090 等
#   - Python：3.11（由 .python-version 指定，uv 会自动准备）
#
# 用法：
#   bash install.sh            # 安装运行 / 训练所需依赖
#
# 说明：
#   - 依赖通过 uv 管理，JAX / PyTorch / CUDA 运行库都由 uv 自动安装，
#     系统层不需要预装 CUDA toolkit。
#   - 模型权重（checkpoints/）和归一化参数（assets/）不在本仓库内，
#     需另行获取后放到对应目录，详见 README.md。

set -euo pipefail

cd "$(dirname "$0")"

echo "==================================================="
echo " openpi (R1 Pro / pi0.5) 安装"
echo "==================================================="

# 1. 准备 uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/4] 未检测到 uv，开始安装 ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # 安装后 uv 默认在 ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[1/4] 已检测到 uv: $(uv --version)"
fi

# 2. 同步依赖到 .venv
#    GIT_LFS_SKIP_SMUDGE=1 用于跳过 LeRobot 依赖的 LFS 大文件拉取
echo "[2/4] uv sync 安装依赖（首次较慢，需下载 JAX/Torch/CUDA 运行库）..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 3. 以可编辑模式安装 openpi 本体
echo "[3/4] 以 editable 模式安装 openpi ..."
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4. 校验 JAX 能否识别 GPU
echo "[4/4] 校验 JAX GPU 可见性 ..."
uv run python -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

echo ""
echo "==================================================="
echo " 安装完成"
echo "==================================================="
echo "下一步："
echo "  1) 准备模型权重到 checkpoints/，归一化参数到 checkpoint 内的 assets/"
echo "  2) 启动服务端推理，见 README.md 的【服务端部署】章节"
echo ""
echo "RTX 5090 (Blackwell) 提示：如果 JAX 报 sm_120 不支持，"
echo "请升级 NVIDIA 驱动到支持 CUDA 12.8+ 的版本后重试。"
