# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""从 big_vision 简化而来的 ViT 实现，用于 Pi 模型。"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """2D 正弦余弦位置编码，参考 MoCo v3。"""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "width 必须是 4 的倍数才能用 sincos 位置编码"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"未知的位置编码类型: {typ}")


class MlpBlock(nn.Module):
    """Transformer 的 MLP / 前馈网络模块。"""

    mlp_dim: int | None = None  # 默认为输入维度的 4 倍
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """前馈网络：Linear(放大) → GELU → Dropout → Linear(压回)"""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """单个 Transformer 编码器 Block（自注意力 + MLP）。"""

    mlp_dim: int | None = None  # 默认为输入维度的 4 倍
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y  # 残差连接

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y  # 残差连接
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer 编码器：堆叠多层 Encoder1DBlock。"""

    depth: int
    mlp_dim: int | None = None  # 默认为输入维度的 4 倍
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            # 使用 scan 模式：共享参数结构，节省编译时间
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # 普通模式：逐层实例化
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """多头注意力池化（Multihead Attention Pooling）。"""

    mlp_dim: int | None = None  # 默认为输入维度的 4 倍
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT 主模型。"""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # 默认为输入维度的 4 倍
    num_heads: int = 12
    posemb: str = "learn"  # 也可以是 "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # 也可以是 "map"、"tok"、"none"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"  # 或 "dots_with_no_batch_dims_saveable" 可加速但更耗显存
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Patch 提取和位置编码在 float32 下进行，数值更安全
        image = jnp.asarray(image, jnp.float32)

        # Patch 提取：用卷积把图像切成 patch 并映射到 width 维
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])  # 展平为序列: [batch, num_patches, width]

        # 加入位置编码
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            # 使用 CLS token 模式时，在序列前面加一个可学习的 token
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # 转换为计算精度（可能是半精度 bfloat16）
        x = x.astype(self.dtype_mm)

        # 通过 Transformer 编码器（堆叠 depth 层 Block）
        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        # 池化策略选择
        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)  # 全局平均池化
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]  # 取第一个 token
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]  # 取 CLS token
            encoded = encoded[:, 1:]  # 去掉 CLS token
        elif self.pool_type == "none":
            pass  # 不做池化，保留所有 token（OpenPI 中使用此模式）
        else:
            raise ValueError(f"未知的池化类型: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            # 输出投影：将 token 维度映射到目标维度（如 2048，与 Gemma 对齐）
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """工厂函数：根据 variant 字符串创建对应配置的 ViT 模型。"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """将变体字符串（如 "So400m/14"）转换为参数字典。"""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # 参考: https://arxiv.org/abs/2106.04560 Table 2
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        **patch,
    }
