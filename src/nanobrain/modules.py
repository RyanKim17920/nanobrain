# References:
# capi: https://github.com/facebookresearch/capi/blob/main/model.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/vision_transformer.py

import math
from functools import lru_cache, partial
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from timm.layers import DropPath

Layer = Type[nn.Module]


# Transformer modules adapted from capi (but removed the efficient residual)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        causal: bool = False,
        qk_norm: bool = False,
        qk_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        context_dim = context_dim or dim

        # qk norm caps the logits at ~sqrt(head_dim), so a learned per head scale (init
        # qk_scale_init) lets them grow back
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LayerNorm(self.head_dim)
            self.k_norm = LayerNorm(self.head_dim)
            self.qk_scale = nn.Parameter(torch.full((num_heads, 1, 1), float(qk_scale_init)))

        # using separate q, k, v weights so that xavier init uses the correct dim.
        # although perhaps technically it should be initialized wrt the head dim..
        # but this is what original mae does.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def extra_repr(self):
        return f"num_heads={self.num_heads}, causal={self.causal}"

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
        vgate: Float[Tensor, "B M"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        if context is None:
            context = x
        B, N, D = x.shape
        _, M, _ = context.shape
        h = self.num_heads

        q = self.q(x).reshape(B, N, h, D // h).transpose(1, 2)
        k = self.k(context).reshape(B, M, h, D // h).transpose(1, 2)
        v = self.v(context).reshape(B, M, h, D // h).transpose(1, 2)
        if vgate is not None:
            # per-key value gate (rtd suspicion gating): a gated key still takes attention mass but
            # contributes a shrunk value. None = the plain attention
            v = v * vgate[:, None, :, None].to(v.dtype)
        if self.qk_norm:
            q = self.q_norm(q) * self.qk_scale.to(q.dtype)
            k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    # lt_gelu (training speed, NUMERICS-CHANGING: tanh GELU; set by train.py native_lt_gelu): fc1 + bias + GELU
    # and fc2's input grad + GELU' as cuBLASLt epilogues (cublaslt.LtMlp)
    lt_gelu = False

    def __init__(
        self,
        dim: int,
        mlp_ratio: int | float = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        if self.lt_gelu and self.training:
            from nanobrain.cublaslt import LtMlp

            bf = torch.bfloat16
            fc1, fc2 = self.fc1, self.fc2
            y = LtMlp.apply(
                x.reshape(-1, x.shape[-1]).to(bf), fc1.weight.to(bf), fc1.bias.to(bf), fc2.weight.to(bf), fc2.bias.to(bf)
            )
            return y.view(*x.shape[:-1], -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# timm default eps=1e-6
LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        mlp_ratio: int | float = 4,
        drop_path: float = 0.0,
        norm_layer: Layer = LayerNorm,
        qk_norm: bool = False,
        qk_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            context_dim=context_dim,
            qk_norm=qk_norm,
            qk_scale_init=qk_scale_init,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=proj_bias,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
        vgate: Float[Tensor, "B M"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        # should the context also be normalized? capi doesn't, so I guess not
        x = x + self.drop_path1(self.attn(self.norm1(x), context=context, vgate=vgate))
        return x + self._mlp_branch(x)

    def _mlp_branch(self, x: Tensor) -> Tensor:
        te_mlp = self.__dict__.get("te_mlp")
        if te_mlp is not None and self.training and TE_ON:
            # native_fp8_mlp (NUMERICS-CHANGING): norm2 + mlp as transformer engine's fp8 LayerNormMLP
            return _te_mlp(te_mlp, x)
        return self.drop_path2(self.mlp(self.norm2(x)))

    def forward_last_rows(
        self, x: Float[Tensor, "B N D"], num_rows: int, vgate: Float[Tensor, "B N"] | None = None
    ) -> Float[Tensor, "B M D"]:
        # forward(x)[:, -num_rows:]: queries and mlp for those rows only, keys / values from all
        h = self.norm1(x)
        x = x[:, -num_rows:] + self.drop_path1(self.attn(h[:, -num_rows:], context=h, vgate=vgate))
        return x + self._mlp_branch(x)


TE_RECIPE = None  # native_fp8_mlp: the transformer engine fp8 recipe (train.py)
TE_ON = True  # dev in-run A/B switch


@torch.compiler.disable
def _te_mlp(mod: nn.Module, x: Tensor) -> Tensor:
    # outside the compiled graphs (a graph break per block): transformer engine runs its own fp8 cast / amax kernels
    import transformer_engine.pytorch as te

    with te.fp8_autocast(enabled=True, fp8_recipe=TE_RECIPE):
        return mod(x)


def te_layernorm_mlp(block: "Block") -> nn.Module:
    # a transformer engine LayerNormMLP (tanh GELU) on the block's own norm2 / fc1 / fc2 parameters (shared, so the
    # state dict, the optimizers and the eval path see the same tensors); not registered as a submodule
    import transformer_engine.pytorch as te

    fc1, fc2, ln = block.mlp.fc1, block.mlp.fc2, block.norm2
    mod = te.LayerNormMLP(
        fc1.in_features, fc1.out_features, eps=ln.eps, bias=fc1.bias is not None, activation="gelu",
        params_dtype=torch.float32, device=fc1.weight.device,
    )
    for name, p in [("layer_norm_weight", ln.weight), ("layer_norm_bias", ln.bias), ("fc1_weight", fc1.weight),
                    ("fc1_bias", fc1.bias), ("fc2_weight", fc2.weight), ("fc2_bias", fc2.bias)]:
        setattr(mod, name, p)
    block.__dict__["te_mlp"] = mod
    return mod


# Position embedding


class SeparablePosEmbed3D(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        embed_dim: int,
        unit: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.unit = unit

        X, Y, Z = grid_size
        self.weight_x = nn.Parameter(torch.empty(X, embed_dim))
        self.weight_y = nn.Parameter(torch.empty(Y, embed_dim))
        self.weight_z = nn.Parameter(torch.empty(Z, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight_x, std=0.02)
        nn.init.trunc_normal_(self.weight_y, std=0.02)
        nn.init.trunc_normal_(self.weight_z, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B N D"]:
        if coord.is_floating_point():
            # continuous coordinates (native resolution token centres): linear interpolation
            # between rows, clamped to the table. integer values give exactly the rows
            coord = coord / self.unit
            for axis, weight in enumerate((self.weight_x, self.weight_y, self.weight_z)):
                pos = coord[:, :, axis].clamp(0, weight.shape[0] - 1)
                lo = pos.floor().clamp(max=weight.shape[0] - 2)
                frac = (pos - lo)[..., None]
                lo = lo.to(torch.int64)
                x = x + (weight[lo] * (1 - frac) + weight[lo + 1] * frac)
            return x
        pos_ids = torch.floor(coord / self.unit).to(torch.int64)
        x = x + self.weight_x[pos_ids[:, :, 0]]
        x = x + self.weight_y[pos_ids[:, :, 1]]
        x = x + self.weight_z[pos_ids[:, :, 2]]
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}, unit={self.unit}"


# Native resolution patch embedding and head. Kernels live in mm on cells x cells x cells cells
# over one token: the canonical 1 mm voxel weight plus a learned residual (zero mean over every
# 1 mm cell, zero at init), integrated over each native voxel's extent. 1 mm voxels get exactly
# the canonical weight, so 1 mm images and the evals are a plain patch_size model.


class NativeEmbed(nn.Linear):
    def __init__(self, patch_size: int, out_features: int, cells: int = 16):
        super().__init__(patch_size**3, out_features)
        assert cells % patch_size == 0
        self.patch_size = patch_size
        self.cells = cells
        self.residual = nn.Parameter(torch.zeros(out_features, cells, cells, cells))

    def native_weights(
        self, spacings: list[tuple[float, ...]], patches: list[tuple[int, ...]]
    ) -> list[Tensor]:
        # [D, P] weight per spacing, all spacings in one pass
        p = self.patch_size
        with torch.autocast(self.weight.device.type, enabled=False):
            density = fine_kernel(self.weight.view(-1, p, p, p), self.residual)
            axes = [
                separable_overlaps(q, sp, self.cells, p / 2, density.device, False)
                for q, sp in zip(patches, spacings)
            ]
            return [weight.t() for weight in separable_weights(density, axes)]

    def native_weight(self, spacing: tuple[float, ...], patch: tuple[int, ...]) -> Tensor:
        return self.native_weights([spacing], [patch])[0]

    def native(
        self,
        x: Float[Tensor, "... P"],
        spacing: tuple[float, ...],
        patch: tuple[int, ...],
        weight: Tensor | None = None,
    ) -> Float[Tensor, "... D"]:
        if spacing == (1.0, 1.0, 1.0):
            return self(x)
        if weight is None:
            weight = self.native_weight(spacing, patch)
        return F.linear(x, weight, self.bias)


class NativeHead(nn.Linear):
    # predicts each native voxel as the mean over its extent of the prediction field
    def __init__(self, in_features: int, patch_size: int, cells: int = 16):
        super().__init__(in_features, patch_size**3)
        assert cells % patch_size == 0
        self.patch_size = patch_size
        self.cells = cells
        self.residual = nn.Parameter(torch.zeros(in_features, cells, cells, cells))
        self.bias_residual = nn.Parameter(torch.zeros(cells, cells, cells))

    def native_weights(
        self, spacings: list[tuple[float, ...]], patches: list[tuple[int, ...]]
    ) -> list[tuple[Tensor, Tensor]]:
        # ([P, D] weight, [P] bias) per spacing, all spacings in one pass
        p = self.patch_size
        with torch.autocast(self.weight.device.type, enabled=False):
            field = fine_kernel(self.weight.t().reshape(-1, p, p, p), self.residual)
            bias = fine_kernel(self.bias.view(p, p, p), self.bias_residual)
            # bias as one more field channel
            field = torch.cat([field, bias[None]])
            # overlaps per unit length average the field over each voxel
            axes = [
                separable_overlaps(q, sp, self.cells, p / 2, field.device, True)
                for q, sp in zip(patches, spacings)
            ]
            weights = [w.split([w.shape[1] - 1, 1], dim=1) for w in separable_weights(field, axes)]
            return [(weight, bias.view(-1)) for weight, bias in weights]

    def native_weight(
        self, spacing: tuple[float, ...], patch: tuple[int, ...]
    ) -> tuple[Tensor, Tensor]:
        return self.native_weights([spacing], [patch])[0]

    def native(
        self,
        x: Float[Tensor, "... D"],
        spacing: tuple[float, ...],
        patch: tuple[int, ...],
        weight: tuple[Tensor, Tensor] | None = None,
    ) -> Float[Tensor, "... P"]:
        if spacing == (1.0, 1.0, 1.0):
            return self(x)
        weight, bias = self.native_weight(spacing, patch) if weight is None else weight
        return F.linear(x, weight, bias)


def separable_weights(
    density: Float[Tensor, "C n n n"], axes: list[tuple[Tensor, Tensor]]
) -> list[Float[Tensor, "P C"]]:
    # per (oxy [qx qy, n n], oz [qz, n]): sum_abc density[:, a, b, c] ox[i, a] oy[j, b] oz[k, c]
    # as [qx qy qz, C]. the z sums of all spacings in one matmul, then one matmul per spacing
    C, n = density.shape[0], density.shape[-1]
    dens = density.permute(1, 2, 3, 0).reshape(n * n, n, C)
    oz = torch.cat([oz for _, oz in axes])
    t = torch.bmm(oz.expand(n * n, -1, -1), dens)
    parts = torch.split(t, [oz.shape[0] for _, oz in axes], dim=1)
    return [(oxy @ part.reshape(n * n, -1)).view(-1, C) for (oxy, _), part in zip(axes, parts)]


@lru_cache(maxsize=4096)
def separable_overlaps(
    patch: tuple[int, ...],
    spacing: tuple[float, ...],
    cells: int,
    half_mm: float,
    device: torch.device,
    per_mm: bool,
) -> tuple[Tensor, Tensor]:
    # voxel_overlaps per axis (per unit length when per_mm) as the x, y outer product and z
    overlaps = [voxel_overlaps(q, sp, cells, half_mm, device) for q, sp in zip(patch, spacing)]
    if per_mm:
        overlaps = [overlap / sp for overlap, sp in zip(overlaps, spacing)]
    ox, oy, oz = overlaps
    oxy = (ox[:, None, :, None] * oy[None, :, None, :]).reshape(len(ox) * len(oy), -1)
    return oxy, oz


def fine_kernel(coarse: Tensor, residual: Tensor) -> Tensor:
    # [..., p, p, p] per 1 mm cell -> [..., n, n, n] density on the fine cells: the cell's value
    # plus the residual minus its mean over that cell
    p, n = coarse.shape[-1], residual.shape[-1]
    sub = n // p
    up = coarse
    for axis in (-3, -2, -1):
        up = up.repeat_interleave(sub, dim=axis)
    cells = residual.reshape(*residual.shape[:-3], p, sub, p, sub, p, sub)
    cells = cells - cells.mean(dim=(-5, -3, -1), keepdim=True)
    return up + cells.reshape(residual.shape)


@lru_cache(maxsize=4096)
def voxel_overlaps(
    patch: int, spacing: float, cells: int, half_mm: float, device: torch.device
) -> Tensor:
    # [patch, cells] mm of overlap between a token's voxels (centred) and the kernel cells on
    # [-half_mm, half_mm]. the outer cells extend outward, for tokens wider than the kernel
    voxel = torch.arange(patch, dtype=torch.float64, device=device)
    lo, hi = (voxel - patch / 2) * spacing, (voxel + 1 - patch / 2) * spacing
    edges = torch.linspace(-half_mm, half_mm, cells + 1, dtype=torch.float64, device=device)
    cell_lo, cell_hi = edges[:-1].clone(), edges[1:].clone()
    # fill_, not item assignment, which syncs
    cell_lo[:1].fill_(-math.inf)
    cell_hi[-1:].fill_(math.inf)
    overlap = torch.minimum(hi[:, None], cell_hi) - torch.maximum(lo[:, None], cell_lo)
    return overlap.clamp_min(0).float()
