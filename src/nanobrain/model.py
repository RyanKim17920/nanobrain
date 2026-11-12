import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainmarks_smri.models import EmbeddingOutput, ImageInput, register_model
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from jaxtyping import Float, Int

from .fino import FinoHeads
from .modules import Block, LayerNorm, NativeEmbed, NativeHead, SeparablePosEmbed3D
from .data import (
    crop_to_mask,
    fit_to_shape,
    native_box,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)


class ViTMAE3D(nn.Module):
    name = "vit_mae_3d"

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        target_spacing: float = 1.0,
        patch_size: int = 8,
        # native resolution (forward_native): images keep their voxel grids, patches of about
        # patch_size mm per axis. grid_size / patch_size stay the 1 mm grid of the evals
        native: bool = False,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        decoder_depth: int = 4,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_tokens: int = 1,
        drop_path_rate: float = 0.0,
        clip_quantiles: tuple[float, float] = (0.005, 0.995),
        qk_norm: bool = False,
        decoder_qk_norm: bool = False,
        qk_scale_init: float = 1.0,
        mask_cube: int = 0,
        # training only: hard-patch loss weight (detached per-token error ** gamma, capped) and the
        # fraction of encoder outputs the decoder sees
        hard_gamma: float = 0.0,
        hard_max: float = 4.0,
        decoder_keep: float = 1.0,
        # zero-init linear path from the pooled encoder mean to every mask token
        decoder_global: bool = False,
        # linear head on the visible encoder tokens reconstructing their own patch (weight; 0 = off)
        self_recon_weight: float = 0.0,
        # sequences only use patches with at least this brain fraction (unless a scan has too few)
        min_mask_frac: float = 0.25,
        # low band of the coarse-to-fine target curriculum, block mean size in mm (0 = off)
        target_lp_mm: float = 0.0,
        # linear head on the visible encoder tokens: BCE to the in-mask fraction of each level_pool^3
        # sub-block of their own patch above each level threshold (weight; 0 = off)
        level_weight: float = 0.0,
        level_thresholds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5),
        level_pool: int = 2,
        # the nanobrain-v3 loss stack, grafted from v3/src/nanobrain/model.py:
        # z-score each target sequence before the mse; foreground-weighted mse; and the
        # xyz gradient penalty on the target-ref patches (dense_neighbor_pool needs
        # encoder plumbing and is not part of this graft)
        norm_target: bool = False,
        grad_loss_weight: float = 0.0,
        fg_loss_weight: float = 0.0,
        # fino3d config block (nanobrain.fino), training only heads on the global embedding;
        # None = off (no params, same init)
        fino3d: dict | None = None,
        # cross-subject replaced-token detection (ELECTRA / CIM RevDet, with REAL replacements):
        # in native training a rtd_ratio share of each sequence's visible patches is swapped for
        # the patch at the same token coordinate of another image of its spacing group, quantile
        # matched (rtd_quantiles levels) to the host image's in-brain intensities; a zero-init
        # linear head on the visible encoder tokens detects them (BCE, weight rtd_weight; 0 = off).
        # the MAE targets stay the clean host patches. training only
        rtd_weight: float = 0.0,
        rtd_ratio: float = 0.15,
        rtd_quantiles: int = 32,
        # per-token orientation detection (PatchRot / RotNet, patch level, 3D): an orient_ratio
        # share of each sequence's visible patches is rotated 180 degrees about one of the three
        # voxel axes (a voxel permutation that keeps any anisotropic patch shape: no interpolation);
        # a zero-init 4-way linear head on the visible encoder tokens names the rotation (CE over
        # all visible tokens, weight orient_weight; 0 = off). MAE targets stay clean. training only
        orient_weight: float = 0.0,
        orient_ratio: float = 0.15,
        # within-image token-shuffle detection (ELECTRA / CIM RevDet with in-distribution donors):
        # in native training a shuffle_ratio share of each sequence's visible patches is replaced
        # by ANOTHER visible patch of the same sequence (same scan, contrast, site, bias field;
        # only its anatomical position is wrong), and a zero-init linear head on the visible
        # encoder tokens detects the displaced ones (BCE, weight shuffle_weight; 0 = off). the MAE
        # targets stay the clean masked patches. runs before rtd; a token rtd then replaces
        # counts as rtd only. training only
        shuffle_weight: float = 0.0,
        shuffle_ratio: float = 0.15,
        # rtd suspicion gating (architecture, training only): every MAE decoder attention scales
        # the value of visible token j by g_j = min(1, 2 sigmoid(-rtd_gate_scale * rtd_logit_j)), so
        # tokens the encoder flags as foreign are muted for reconstruction and the MAE loss reaches
        # the detection features. g = 1 at the zero-init head (identity at step 0); 0 = off
        rtd_gate_scale: float = 0.0,
        # ---- ms-aux pretext heads (native training only, all training-only, 0 = off) ----
        # where-am-I: zero-init Linear(D, 3) on the visible encoder tokens regresses each token's
        # centre in brain-normalized RAS coordinates ((centre - brain-mask centroid) / brain
        # half-extent, per axis; smooth-L1). a brainpos_hide_frac share of the visible tokens get a
        # learned "unknown position" embedding instead of their position embedding
        brainpos_weight: float = 0.0,
        brainpos_hide_frac: float = 0.25,
        # same-scan matching: mean visible token -> MLP (D, D, scanmatch_dim) -> l2; the
        # num_samples maskings of one scan are positives, every other scan (minus same-subject
        # scans) a negative; symmetric InfoNCE at scanmatch_temp
        scanmatch_weight: float = 0.0,
        scanmatch_temp: float = 0.1,
        scanmatch_dim: int = 256,
        # hemisphere swap detection: a hemiswap_ratio share of the visible tokens is replaced by
        # the left-right mirror of the contralateral block (mirrored across the brain-mask
        # centroid x, RAS axis 0); zero-init Linear(D, 1), BCE. MAE targets stay clean
        hemiswap_weight: float = 0.0,
        hemiswap_ratio: float = 0.1,
        # corruption-type detection: a corrupt_ratio share of the visible tokens gets a bias ramp
        # (x0.7-1.3), a sigma 1 voxel blur or Rician-like noise (10% of the in-brain patch std);
        # zero-init Linear(D, 4) over {clean, bias, blur, noise}, CE. MAE targets stay clean
        corrupt_weight: float = 0.0,
        corrupt_ratio: float = 0.1,
        # edge level-set: the level head construction on the gradient magnitude (central
        # differences per mm on the patch's own voxel grid), thresholds k x the row's in-brain
        # median gradient, level_pool sub-blocks, BCE weighted by brain fraction
        edgelvl_weight: float = 0.0,
        edgelvl_thresholds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    ):
        super().__init__()
        self.brainpos_weight = float(brainpos_weight)
        self.brainpos_hide_frac = float(brainpos_hide_frac)
        self.scanmatch_weight = float(scanmatch_weight)
        self.scanmatch_temp = float(scanmatch_temp)
        self.hemiswap_weight = float(hemiswap_weight)
        self.hemiswap_ratio = float(hemiswap_ratio)
        self.corrupt_weight = float(corrupt_weight)
        self.corrupt_ratio = float(corrupt_ratio)
        self.edgelvl_weight = float(edgelvl_weight)
        self.aux_stats = {}
        self.orient_weight = float(orient_weight)
        self.orient_ratio = float(orient_ratio)
        self.shuffle_weight = float(shuffle_weight)
        self.shuffle_ratio = float(shuffle_ratio)
        self.rtd_gate_scale = float(rtd_gate_scale)
        self.rtd_weight = float(rtd_weight)
        self.rtd_ratio = float(rtd_ratio)
        self.rtd_quantiles = int(rtd_quantiles)
        self.norm_target = norm_target
        self.grad_loss_weight = float(grad_loss_weight)
        self.fg_loss_weight = float(fg_loss_weight)
        self.grid_size = grid_size
        self.target_spacing = target_spacing
        self.native = native
        assert not native or target_spacing == 1.0, "native kernels are defined on 1 mm voxels"
        # tuple for the brainmarks-smri model contract
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.class_tokens = class_tokens
        patch_grid_size = tuple(size // patch_size for size in grid_size)
        self.patch_grid_size = patch_grid_size
        self.clip_quantiles = clip_quantiles
        self.mask_cube = mask_cube
        self.hard_gamma = hard_gamma
        self.hard_max = hard_max
        self.decoder_keep = decoder_keep

        patch_dim = patch_size**3
        self.register_buffer("coord_grid", make_coord_grid(grid_size, patch_size), persistent=False)

        # encoder
        if native:
            self.patch_embed = NativeEmbed(patch_size, embed_dim)
        else:
            self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = SeparablePosEmbed3D(patch_grid_size, embed_dim)
        if class_tokens:
            self.cls_token = nn.Parameter(torch.empty(1, class_tokens, embed_dim))
        else:
            self.cls_token = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                    qk_norm=qk_norm,
                    qk_scale_init=qk_scale_init,
                )
                for ii in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.global_proj = nn.Linear(embed_dim, decoder_embed_dim) if decoder_global else None
        self.self_recon_weight = self_recon_weight
        self.self_recon_head = nn.Linear(embed_dim, patch_dim) if self_recon_weight else None
        self.min_mask_frac = min_mask_frac
        # coarse-to-fine target curriculum (native path): block size in mm of the low band (0 =
        # off); target_band (0..1) is set by the training loop from the wall-clock progress
        self.target_lp_mm = float(target_lp_mm)
        self.target_band = 1.0
        self.level_weight = level_weight
        self.level_pool = level_pool
        self.level_head = None
        if level_weight:
            levels = [float(t) for t in level_thresholds]
            self.level_head = nn.Linear(embed_dim, len(levels) * (patch_size // level_pool) ** 3)
            self.register_buffer("level_thresholds", torch.tensor(levels), persistent=False)
        # training speed, set by train.py (not a model kwarg): see decode_features
        self.decoder_prune_last = False
        self.decoder_pos_embed = SeparablePosEmbed3D(patch_grid_size, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    qk_norm=decoder_qk_norm,
                    qk_scale_init=qk_scale_init,
                )
                for ii in range(decoder_depth)
            ]
        )

        self.decoder_norm = LayerNorm(decoder_embed_dim)
        if native:
            self.decoder_head = NativeHead(decoder_embed_dim, patch_size)
        else:
            self.decoder_head = nn.Linear(decoder_embed_dim, patch_dim)
        self.init_weights()
        # built after the base init, so every base parameter is initialized as with fino3d off
        self.fino_heads = None
        self.fino_stats = {}
        if fino3d is not None:
            assert native, "fino3d: native training only"
            self.fino_heads = FinoHeads(embed_dim, fino3d)
            self.fino_heads.apply(_init_weights)
        # built last, so every other parameter is initialized as with rtd off
        self.rtd_head = None
        self.rtd_stats = {}
        if self.rtd_weight > 0.0:
            assert native, "rtd: native training only"
            self.rtd_head = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.rtd_head.weight)
            nn.init.zeros_(self.rtd_head.bias)
        self.orient_head = None
        if self.orient_weight > 0.0:
            assert native, "orient: native training only"
            self.orient_head = nn.Linear(embed_dim, 4)
            nn.init.zeros_(self.orient_head.weight)
            nn.init.zeros_(self.orient_head.bias)
        # built after rtd, so every other parameter is initialized as with shuffle detection off
        self.shuffle_head = None
        self.shuffle_stats = {}
        if self.shuffle_weight > 0.0:
            assert native, "shuffle detection: native training only"
            self.shuffle_head = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.shuffle_head.weight)
            nn.init.zeros_(self.shuffle_head.bias)

        # ms-aux heads, built last so every other parameter is initialized as with them off
        def zero_linear(dout: int) -> nn.Linear:
            head = nn.Linear(embed_dim, dout)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            return head

        self.brainpos_head = None
        if self.brainpos_weight > 0.0:
            assert native, "brainpos: native training only"
            self.brainpos_head = zero_linear(3)
            self.brainpos_unk = nn.Parameter(torch.zeros(embed_dim))
            nn.init.trunc_normal_(self.brainpos_unk, std=0.02)
        self.scanmatch_proj = None
        if self.scanmatch_weight > 0.0:
            assert native, "scanmatch: native training only"
            self.scanmatch_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, int(scanmatch_dim))
            )
            self.scanmatch_proj.apply(_init_weights)
        self.hemiswap_head = None
        if self.hemiswap_weight > 0.0:
            assert native, "hemiswap: native training only"
            self.hemiswap_head = zero_linear(1)
        self.corrupt_head = None
        if self.corrupt_weight > 0.0:
            assert native, "corrupt: native training only"
            self.corrupt_head = zero_linear(4)
        self.edgelvl_head = None
        if self.edgelvl_weight > 0.0:
            assert native, "edgelvl: native training only"
            edges = [float(t) for t in edgelvl_thresholds]
            self.edgelvl_head = zero_linear(len(edges) * (patch_size // level_pool) ** 3)
            self.register_buffer("edgelvl_thresholds", torch.tensor(edges), persistent=False)

    @torch.no_grad()
    def brain_frame(
        self, mask_patches: Tensor, grid: tuple[int, int, int], patch: tuple[int, int, int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per image brain frame on the padded box, in voxels: token centres [num, 3], brain-mask
        centroid [b, 3] (voxel-count weighted token centres) and half-extent [b, 3] (outermost
        brain-touching token centres +- half a patch)."""
        dev = mask_patches.device
        axes = [(torch.arange(g, device=dev, dtype=torch.float32) + 0.5) * p for g, p in zip(grid, patch)]
        centres = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        counts = mask_patches.float().sum(-1)  # [b, num]
        cen = counts @ centres / counts.sum(1, keepdim=True).clamp_min(1.0)
        has = (counts > 0)[..., None]
        hi = torch.where(has, centres[None], -math.inf).amax(1)
        lo = torch.where(has, centres[None], math.inf).amin(1)
        half = ((hi - lo) / 2 + torch.tensor(patch, device=dev, dtype=torch.float32) / 2)
        half = torch.where(torch.isfinite(half), half, torch.ones_like(half)).clamp_min(1.0)
        return centres, cen, half

    @torch.no_grad()
    def hemiswap_replace(
        self, vis_patches: Tensor, vis_mask: Tensor, raw: Tensor, vis_order: Tensor,
        grid: tuple[int, int, int], patch: tuple[int, int, int], cen: Tensor, img_of_row: Tensor,
        exclude: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replace a hemiswap_ratio share of the visible tokens by the x-flipped block at the
        contralateral position (mirrored across the brain centroid x, RAS axis 0) of the same
        image. raw [2, b, X, Y, Z] is the padded (image, mask) box. Eligible: the mirror block is
        >= 50% brain, the token is >= one patch off the midline, and not in `exclude` (rtd)."""
        px, py, pz = patch
        gx, gy, gz = grid
        X, Y, Z = raw.shape[-3:]
        dev = vis_patches.device
        t = vis_order
        ix, iy, iz = t // (gy * gz), (t // gz) % gy, t % gz
        sx = (ix * px).float()
        cx = cen[img_of_row, 0][:, None]
        start = torch.round(2 * cx - sx - px).long().clamp(0, X - px)
        far = ((sx + px / 2) - cx).abs() >= px
        ax, ay, az = torch.meshgrid(
            torch.arange(px, device=dev), torch.arange(py, device=dev), torch.arange(pz, device=dev),
            indexing="ij",
        )
        ax, ay, az = ax.reshape(-1), ay.reshape(-1), az.reshape(-1)
        xs = start[..., None] + (px - 1 - ax)  # the x flip
        ys = (iy * py)[..., None] + ay
        zs = (iz * pz)[..., None] + az
        flat = ((img_of_row[:, None, None] * X + xs) * Y + ys) * Z + zs
        m_patches = raw[0].reshape(-1)[flat].to(vis_patches.dtype)
        m_mask = raw[1].reshape(-1)[flat].to(vis_mask.dtype)
        ok = (m_mask.float().mean(-1) >= 0.5) & far
        if exclude is not None:
            ok = ok & ~exclude
        hit = (torch.rand(t.shape, device=dev) < self.hemiswap_ratio) & ok
        sel = hit[..., None]
        return torch.where(sel, m_patches, vis_patches), torch.where(sel, m_mask, vis_mask), hit

    @torch.no_grad()
    def corrupt_transform(
        self, vis_patches: Tensor, vis_mask: Tensor, patch: tuple[int, int, int], exclude: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Corrupt a corrupt_ratio share of the visible (>= 50% brain) tokens with 1 = a smooth
        multiplicative bias ramp (x[0.7, 1.3] across the patch, applied above the row's in-brain
        minimum), 2 = a sigma 1 voxel gaussian blur (normalized inside the mask), 3 = Rician-like
        noise at 10% of the patch's in-brain std. Returns (patches, labels [rows, V] long)."""
        rows, nv, P = vis_patches.shape
        px, py, pz = patch
        dev = vis_patches.device
        x = vis_patches.float()
        m = vis_mask.float()
        ok = m.mean(-1) >= 0.5
        if exclude is not None:
            ok = ok & ~exclude
        hit = (torch.rand(rows, nv, device=dev) < self.corrupt_ratio) & ok
        labels = torch.where(hit, torch.randint(1, 4, (rows, nv), device=dev), 0)
        lo = torch.where(m > 0, x, math.inf).amin(dim=(1, 2), keepdim=True)
        lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
        cnt = m.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(-1, keepdim=True) / cnt
        std = (((x - mean) ** 2 * m).sum(-1, keepdim=True) / cnt).sqrt()
        # (1) bias ramp along a random direction: |u . r| <= 1 with r in [-1, 1]^3
        axes = [((torch.arange(p, device=dev) + 0.5) / p * 2 - 1) for p in (px, py, pz)]
        r = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        u = torch.randn(rows, nv, 3, device=dev)
        u = u / u.abs().sum(-1, keepdim=True).clamp_min(1e-6)
        amp = 0.2 + 0.1 * torch.rand(rows, nv, 1, device=dev)
        x_bias = (x - lo) * (1 + amp * (u @ r.T)) + lo
        # (2) separable gaussian blur, sigma 1 voxel, normalized convolution inside the mask
        k = torch.exp(-torch.arange(-2, 3, device=dev, dtype=torch.float32) ** 2 / 2)
        k = k / k.sum()
        vol = torch.cat([(x * m).reshape(-1, 1, px, py, pz), m.reshape(-1, 1, px, py, pz)], 1)
        vol = vol.reshape(-1, 1, px, py, pz)
        for axis, size in zip((2, 3, 4), (px, py, pz)):
            if size < 2:
                continue
            shape = [1, 1, 1, 1, 1]
            shape[axis] = 5
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - axis)] = pad[2 * (4 - axis) + 1] = 2
            vol = F.conv3d(F.pad(vol, pad, mode="replicate"), k.view(shape))
        vol = vol.reshape(-1, 2, P)
        x_blur = (vol[:, 0] / vol[:, 1].clamp_min(1e-3)).reshape(rows, nv, P)
        # (3) Rician-like noise on the magnitude above the row minimum
        s = 0.1 * std
        n1 = torch.randn_like(x) * s
        n2 = torch.randn_like(x) * s
        x_noise = ((x - lo + n1) ** 2 + n2**2).sqrt() + lo
        out = x
        for c, xc in ((1, x_bias), (2, x_blur), (3, x_noise)):
            out = torch.where((labels == c)[..., None], xc, out)
        out = out * m
        return out.to(vis_patches.dtype), labels

    def edgelvl_terms(
        self, tokens: Tensor, patches: Tensor, mask: Tensor, patch, spacing
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Edge level-set head: level_terms on the gradient magnitude (central differences per
        mm, only where both neighbours are in the mask), thresholds edgelvl_thresholds x the
        row's in-brain median gradient. Returns (weighted bce sum, normalizer, weighted abs err)."""
        px, py, pz = patch
        rows, nv = patches.shape[:2]
        with torch.no_grad():
            x = patches.float().reshape(-1, 1, px, py, pz)
            m = mask.float().reshape(-1, 1, px, py, pz)
            g2 = torch.zeros_like(x)
            for axis, size, sp in zip((2, 3, 4), (px, py, pz), spacing):
                if size < 3:
                    continue
                d = (x.narrow(axis, 2, size - 2) - x.narrow(axis, 0, size - 2)) / (2.0 * float(sp))
                both = m.narrow(axis, 2, size - 2) * m.narrow(axis, 0, size - 2)
                pad = [0, 0, 0, 0, 0, 0]
                pad[2 * (4 - axis)] = pad[2 * (4 - axis) + 1] = 1
                g2 = g2 + F.pad((d * both) ** 2, pad)
            g = g2.sqrt() * m
            # per-row in-brain median gradient
            flat = torch.where(m > 0, g, math.inf).reshape(rows, -1)
            cnt = (m > 0).reshape(rows, -1).sum(1)
            srt = flat.sort(dim=1).values
            med = srt.gather(1, (cnt // 2).clamp(max=flat.shape[1] - 1)[:, None]).squeeze(1)
            med = torch.where(torch.isfinite(med) & (med > 0), med, torch.ones_like(med))
            thr = med[:, None] * self.edgelvl_thresholds[None]  # [rows, L]
            thr = thr.repeat_interleave(nv, 0).view(-1, thr.shape[1], 1, 1, 1)
            above = (g > thr).float() * m
            sub = self.patch_size[0] // self.level_pool
            weight = F.adaptive_avg_pool3d(m, sub)
            target = F.adaptive_avg_pool3d(above, sub) / weight.clamp_min(1e-6)
        logits = self.edgelvl_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        err = (weight * (torch.sigmoid(logits.detach()) - target).abs()).sum()
        return (weight * bce).sum(), weight.sum() * target.shape[1], err

    @torch.no_grad()
    def orient_transform(
        self, vis_patches: Tensor, vis_mask: Tensor, patch: tuple[int, int, int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Rotate an orient_ratio share of the visible patches by 180 degrees about x, y or z
        (classes 1 / 2 / 3; 0 = untouched). Returns (patches, mask, labels [rows, V] long)."""
        rows, nv, _ = vis_patches.shape
        px, py, pz = patch
        dev = vis_patches.device
        hit = torch.rand(rows, nv, device=dev) < self.orient_ratio
        labels = torch.where(hit, torch.randint(1, 4, (rows, nv), device=dev), 0)
        x = vis_patches.view(rows, nv, px, py, pz)
        m = vis_mask.view(rows, nv, px, py, pz)
        # rotation by 180 degrees about an axis flips the two other axes
        flips = {1: (3, 4), 2: (2, 4), 3: (2, 3)}
        out_x, out_m = x, m
        for c, dims in flips.items():
            sel = (labels == c)[..., None, None, None]
            out_x = torch.where(sel, x.flip(dims), out_x)
            out_m = torch.where(sel, m.flip(dims), out_m)
        return out_x.reshape_as(vis_patches), out_m.reshape_as(vis_mask), labels

    @torch.no_grad()
    def shuffle_replace(self, vis_patches: Tensor, vis_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Replace a shuffle_ratio share of every row's visible patches by another visible patch
        of the same row (a random within-row permutation gives the donors). Host and donor must
        both be mostly brain (>= half their voxels in the mask), so the cue is anatomical
        inconsistency, not background. Returns (patches, mask, replaced [rows, V] bool)."""
        rows, num_vis = vis_patches.shape[:2]
        device = vis_patches.device
        perm = torch.rand(rows, num_vis, device=device).argsort(dim=1)
        brain = vis_mask.float().mean(-1) >= 0.5
        donor_brain = brain.gather(1, perm)
        not_self = perm != torch.arange(num_vis, device=device)[None]
        replaced = (torch.rand(rows, num_vis, device=device) < self.shuffle_ratio) & brain
        replaced = replaced & donor_brain & not_self
        idx = perm[..., None].expand_as(vis_patches)
        d_patches = vis_patches.gather(1, idx)
        d_mask = vis_mask.gather(1, perm[..., None].expand_as(vis_mask))
        sel = replaced[..., None]
        return (
            torch.where(sel, d_patches, vis_patches),
            torch.where(sel, d_mask, vis_mask),
            replaced,
        )

    @torch.no_grad()
    def rtd_replace(
        self, vis_patches: Tensor, vis_mask: Tensor, order: Tensor, vols: Tensor, b: int, num: int,
        num_samples: int, num_visible: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Swap a rtd_ratio share of the visible patches of every row for the patch at the same
        token coordinate of the next image of the group (rolled by one), quantile matched to the
        host row's in-brain voxels. Returns (patches, mask, replaced [rows, num_visible] bool)."""
        rows = b * num_samples
        vis_order = order[:, :num_visible]  # [rows, V], token ids of the host
        partner_img = (torch.arange(b, device=order.device) + 1) % b
        partner_img = partner_img.repeat_interleave(num_samples)  # [rows]
        flat = vis_order + (partner_img * num)[:, None]
        p_patches = vols[0].flatten(0, 1).index_select(0, flat.flatten()).view_as(vis_patches)
        p_mask = vols[1].flatten(0, 1).index_select(0, flat.flatten()).view_as(vis_patches)
        # the partner patch must be brain tissue too (half its voxels in its brain mask)
        ok = p_mask.float().mean(-1) >= 0.5
        replaced = (torch.rand(vis_order.shape, device=order.device) < self.rtd_ratio) & ok
        # per-row in-brain quantiles of the host's visible patches and of the partner image's
        # visible patches in its own row (row of image j+1, same sample index)
        nq = self.rtd_quantiles + 1
        qs = torch.linspace(0, 1, nq, device=order.device)

        def row_quantiles(x: Tensor, m: Tensor) -> Tensor:
            x = x.float().flatten(1)
            m = m.flatten(1) > 0.5
            cnt = m.sum(1).clamp_min(1)
            srt = torch.where(m, x, torch.full_like(x, float("inf"))).sort(dim=1).values
            idx = (qs[None] * (cnt[:, None] - 1).float()).round().long()
            q = srt.gather(1, idx)
            q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
            # strictly increasing for the interpolation
            return q + 1e-5 * torch.arange(nq, device=x.device)[None]

        q_host = row_quantiles(vis_patches, vis_mask)
        row_of = torch.arange(rows, device=order.device)
        partner_row = (row_of // num_samples + 1) % b * num_samples + row_of % num_samples
        q_part = q_host[partner_row]
        x = p_patches.float().flatten(1)
        pos = torch.searchsorted(q_part.contiguous(), x.contiguous()).clamp(1, nq - 1)
        lo, hi = q_part.gather(1, pos - 1), q_part.gather(1, pos)
        t = ((x - lo) / (hi - lo)).clamp(0, 1)
        mapped = q_host.gather(1, pos - 1) + t * (q_host.gather(1, pos) - q_host.gather(1, pos - 1))
        mapped = mapped.view_as(p_patches).to(vis_patches.dtype)
        sel = replaced[..., None]
        return (
            torch.where(sel, mapped, vis_patches),
            torch.where(sel, p_mask, vis_mask),
            replaced,
        )

    @classmethod
    def from_config(cls, args: dict) -> "ViTMAE3D":
        return cls(
            grid_size=tuple(args["grid_size"]),
            target_spacing=args["target_spacing"],
            clip_quantiles=tuple(args["clip_quantiles"]),
            patch_size=args["patch_size"],
            native=args["native"],
            fino3d=fino if (fino := args.get("fino3d")) and fino.get("enabled") else None,
            **args["model_kwargs"],
        )

    def extra_repr(self):
        return (
            f"{self.grid_size}, {self.patch_size}, class_tokens={self.class_tokens}, "
            f"native={self.native}"
        )

    def init_weights(self):
        self.apply(_init_weights)
        if self.class_tokens:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # initialize the decoder head to zero to prevent hockey stick
        self.decoder_head.weight.data.mul_(0.0)
        if self.level_head is not None:
            nn.init.zeros_(self.level_head.weight)
            nn.init.zeros_(self.level_head.bias)
        if self.self_recon_head is not None:
            nn.init.zeros_(self.self_recon_head.weight)
            nn.init.zeros_(self.self_recon_head.bias)
        if self.global_proj is not None:
            nn.init.zeros_(self.global_proj.weight)
            nn.init.zeros_(self.global_proj.bias)

    def forward_encoder(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B R+N D"]:
        B, N, P = patches.shape
        assert coord.shape == (B, N, 3)
        return self.encode_tokens(self.patch_embed(patches), coord)

    def encode_tokens(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
        pos_hide: Tensor | None = None,
    ) -> Float[Tensor, "B R+N D"]:
        # forward_encoder after the patch embedding. pos_hide [B, N] bool (brainpos, training
        # only): those tokens get the learned unknown-position embedding instead of their own
        B = x.shape[0]
        if pos_hide is not None:
            x = torch.where(pos_hide[..., None], x + self.brainpos_unk.to(x.dtype), self.pos_embed(x, coord))
        else:
            x = self.pos_embed(x, coord)

        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_decoder(
        self,
        embeds: Float[Tensor, "B R+N D"],
        target_coord: Float[Tensor, "B M 3"],
    ) -> Float[Tensor, "B M P"]:
        return self.decoder_head(self.decode_features(embeds, target_coord))

    def decode_features(
        self,
        embeds: Float[Tensor, "B R+N D"],
        target_coord: Float[Tensor, "B M 3"],
        vis_gate: Float[Tensor, "B N"] | None = None,
    ) -> Float[Tensor, "B M D"]:
        # forward_decoder before the head. vis_gate (rtd suspicion gating, training only): a per
        # visible token value gate in every decoder attention; class and mask tokens stay at 1
        B = embeds.shape[0]
        M = target_coord.shape[1]
        assert target_coord.shape == (B, M, 3)

        if self.global_proj is not None:
            pooled = self.global_proj(embeds[:, self.class_tokens :].mean(dim=1, keepdim=True))
        if self.training and self.decoder_keep < 1.0:
            # visible order is a random shuffle, so a prefix is a uniform subset
            keep = int((embeds.shape[1] - self.class_tokens) * self.decoder_keep)
            embeds = embeds[:, : self.class_tokens + keep]
            if vis_gate is not None:
                vis_gate = vis_gate[:, :keep]
        embeds = self.decoder_proj(embeds)

        mask_tokens = self.mask_token.expand(B, M, -1)
        mask_tokens = self.decoder_pos_embed(mask_tokens, target_coord)
        if self.global_proj is not None:
            mask_tokens = mask_tokens + pooled

        x = torch.cat([embeds, mask_tokens], dim=1)
        gate = None
        if vis_gate is not None:
            ones_c = vis_gate.new_ones(B, self.class_tokens)
            ones_m = vis_gate.new_ones(B, M)
            gate = torch.cat([ones_c, vis_gate, ones_m], dim=1)

        if self.decoder_prune_last and len(self.decoder_blocks):
            # the head reads only the mask tokens, so the last block computes just their rows
            for block in self.decoder_blocks[:-1]:
                x = block(x, vgate=gate)
            x = self.decoder_blocks[-1].forward_last_rows(x, M, vgate=gate)
        else:
            for block in self.decoder_blocks:
                x = block(x, vgate=gate)
            x = x[:, -M:]
        x = self.decoder_norm(x)
        return x

    def forward(
        self,
        images: Float[Tensor, "B X Y Z"] | list[Float[Tensor, "X Y Z"]],
        mask: Float[Tensor, "B X Y Z"] | list[Float[Tensor, "X Y Z"]],
        num_visible: int,
        num_predict: int | None = None,
        num_samples: int = 1,
        with_state: bool = True,
        spacing: list[tuple[float, float, float]] | None = None,
        fino_meta: Tensor | None = None,
        fino_gamma: float = 0.0,
        subject_ids: Tensor | None = None,
    ):
        if isinstance(images, list):
            # native resolution batch, images on their own voxel grids
            return self.forward_native(
                images, mask, spacing, num_visible, num_predict, num_samples, with_state,
                fino_meta=fino_meta, fino_gamma=fino_gamma, subject_ids=subject_ids,
            )
        seqs, order = self.sample_patches(images, mask, num_visible, num_predict, num_samples)
        if not with_state:
            return self.patch_loss(*seqs)
        loss, embeds, pred_patches = self.patch_loss(*seqs, with_outputs=True)
        vis_patches, vis_mask_patches, _, target_patches, target_mask_patches, _ = seqs
        num_predict = target_patches.shape[1]
        B = images.shape[0]

        patch_ids = order.flatten(0, 1)
        image_ids = torch.arange(B, device=images.device).repeat_interleave(num_samples)
        vis_ids, target_ids = torch.split(patch_ids, [num_visible, num_predict], dim=1)

        state = {
            "vis_patches": vis_patches,
            "target_patches": target_patches,
            "vis_mask_patches": vis_mask_patches.to(images.dtype),
            "target_mask_patches": target_mask_patches.to(images.dtype),
            "embeds": embeds,
            "pred_patches": pred_patches,
            "vis_ids": vis_ids,
            "target_ids": target_ids,
            "image_ids": image_ids,
        }
        return loss, state

    def sample_patches(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
        num_visible: int,
        num_predict: int | None = None,
        num_samples: int = 1,
    ):
        # forward's sequence sampling and patch gather: (vis patches, vis mask patches, vis coords,
        # target patches, target mask patches, target coords), order
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)

        patches = patchify3d(images, self.patch_size[0])
        mask_patches = patchify3d(mask, self.patch_size[0])
        coord = self.coord_grid.expand(B, -1, -1)

        # sample num_samples sequences per image, then flatten them into the batch
        # num_predict None predicts all remaining patches (dense decoding)
        seq_length = None if num_predict is None else num_visible + num_predict
        cube = (self.patch_grid_size, self.mask_cube) if self.mask_cube else None
        order = sample_sequences(
            mask_patches, num_samples, seq_length, min_mask_frac=self.min_mask_frac, cube=cube
        )
        num_predict = order.shape[2] - num_visible
        batch_ids = torch.arange(B, device=images.device)[:, None, None]
        patches = patches[batch_ids, order].flatten(0, 1)
        # uint8 masks from the fast prep, cast in patch_loss
        mask_patches = mask_patches[batch_ids, order].flatten(0, 1)
        coord = coord[batch_ids, order].flatten(0, 1)

        vis_patches, target_patches = torch.split(patches, [num_visible, num_predict], dim=1)
        vis_mask_patches, target_mask_patches = torch.split(
            mask_patches, [num_visible, num_predict], dim=1
        )
        vis_coord, target_coord = torch.split(coord, [num_visible, num_predict], dim=1)
        seqs = (vis_patches, vis_mask_patches, vis_coord, target_patches, target_mask_patches, target_coord)
        return seqs, order

    def sample_patches_contiguous(self, *args):
        # cuda graph inputs: separate dense buffers, uint8 masks stay uint8. the class method, as
        # compile_cudagraphs shadows self.sample_patches with this
        seqs, order = ViTMAE3D.sample_patches(self, *args)
        return tuple(t.contiguous() for t in seqs), order

    def patch_loss(
        self,
        vis_patches: Float[Tensor, "B N P"],
        vis_mask_patches: Tensor,
        vis_coord: Tensor,
        target_patches: Float[Tensor, "B M P"],
        target_mask_patches: Tensor,
        target_coord: Tensor,
        with_outputs: bool = False,
    ):
        # forward's loss from sampled sequences
        vis_mask_patches = vis_mask_patches.to(vis_patches.dtype)
        target_mask_patches = target_mask_patches.to(target_patches.dtype)
        embeds = self.forward_encoder(vis_patches, vis_coord)
        pred_patches = self.forward_decoder(embeds, target_coord)
        # graft: the v3 loss stack. norm_target z-scores each target sequence; the
        # mse and the gradient penalty run on target_ref, the fg mask weights the mse
        if self.norm_target:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_ref = (target_patches - mean) / (var + 1e-6) ** 0.5
        else:
            target_ref = target_patches
        loss = F.mse_loss(pred_patches, target_ref, reduction="none")
        if self.fg_loss_weight > 0.0:
            fg_mask = (target_patches.abs() > 0.05).float()
            fg_weights = 1.0 + self.fg_loss_weight * fg_mask
            loss = (target_mask_patches * fg_weights * loss).sum() / (target_mask_patches * fg_weights).sum().clamp(min=1.0)
        else:
            loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()
        if self.grad_loss_weight > 0.0:
            p = self.patch_size[0]
            BM = pred_patches.shape[0] * pred_patches.shape[1]
            pred_vol = pred_patches.reshape(BM, p, p, p)
            tgt_vol = target_ref.reshape(BM, p, p, p)
            mask_vol = target_mask_patches.reshape(BM, p, p, p)
            dx_loss = F.mse_loss(
                pred_vol[:, 1:, :, :] - pred_vol[:, :-1, :, :],
                tgt_vol[:, 1:, :, :] - tgt_vol[:, :-1, :, :],
                reduction="none",
            )
            dy_loss = F.mse_loss(
                pred_vol[:, :, 1:, :] - pred_vol[:, :, :-1, :],
                tgt_vol[:, :, 1:, :] - tgt_vol[:, :, :-1, :],
                reduction="none",
            )
            dz_loss = F.mse_loss(
                pred_vol[:, :, :, 1:] - pred_vol[:, :, :, :-1],
                tgt_vol[:, :, :, 1:] - tgt_vol[:, :, :, :-1],
                reduction="none",
            )
            mx = mask_vol[:, 1:, :, :] * mask_vol[:, :-1, :, :]
            my = mask_vol[:, :, 1:, :] * mask_vol[:, :, :-1, :]
            mz = mask_vol[:, :, :, 1:] * mask_vol[:, :, :, :-1]
            grad_loss = (
                (mx * dx_loss).sum() / mx.sum().clamp(min=1.0)
                + (my * dy_loss).sum() / my.sum().clamp(min=1.0)
                + (mz * dz_loss).sum() / mz.sum().clamp(min=1.0)
            ) / 3.0
            loss = loss + self.grad_loss_weight * grad_loss

        if self.hard_gamma:
            vox = target_mask_patches.sum(-1)
            tok = (target_mask_patches * loss).sum(-1) / vox.clamp_min(1)
            has = (vox > 0).to(tok.dtype)
            h = tok.detach()
            w = (h / ((h * has).sum() / has.sum()).clamp_min(1e-8)).clamp_min(1e-8)
            w = w.pow(self.hard_gamma).clamp(max=self.hard_max)
            loss = (w * vox * tok).sum() / (w * vox).sum()
        else:
            loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()
        if self.self_recon_head is not None:
            self_pred = self.self_recon_head(embeds[:, self.class_tokens :])
            self_loss = F.mse_loss(self_pred, vis_patches, reduction="none")
            self_loss = (vis_mask_patches * self_loss).sum() / vis_mask_patches.sum()
            loss = loss + self.self_recon_weight * self_loss
        if self.level_head is not None:
            loss = loss + self.level_weight * self.level_loss(
                embeds[:, self.class_tokens :], vis_patches, vis_mask_patches
            )
        if with_outputs:
            return loss, embeds, pred_patches
        return loss

    def level_terms(self, tokens: Tensor, patches: Tensor, mask: Tensor, patch) -> tuple[Tensor, Tensor]:
        # level_loss on native patches (voxels px x py x pz): targets on the same mm sub-blocks via
        # adaptive pooling (thick-slice axes replicate); weighted sum and its normalizer
        px, py, pz = patch
        x = patches.float().reshape(-1, 1, px, py, pz)
        m = mask.float().reshape(-1, 1, px, py, pz)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        sub = self.patch_size[0] // self.level_pool
        weight = F.adaptive_avg_pool3d(m, sub)
        target = F.adaptive_avg_pool3d(above, sub) / weight.clamp_min(1e-6)
        logits = self.level_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (weight * bce).sum(), weight.sum() * target.shape[1]

    def level_loss(self, tokens: Tensor, patches: Tensor, mask: Tensor) -> Tensor:
        # level head BCE, sub-blocks weighted by their in-mask fraction
        p, q = self.patch_size[0], self.level_pool
        x = patches.float().reshape(-1, 1, p, p, p)
        m = mask.float().reshape(-1, 1, p, p, p)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        weight = F.avg_pool3d(m, q)
        target = F.avg_pool3d(above, q) / weight.clamp_min(1e-6)
        logits = self.level_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (weight * bce).sum() / (weight.sum() * target.shape[1]).clamp_min(1e-6)

    def forward_native(
        self,
        images: list[Float[Tensor, "X Y Z"]],
        mask: list[Float[Tensor, "X Y Z"]],
        spacing: list[tuple[float, float, float]],
        num_visible: int,
        num_predict: int,
        num_samples: int = 1,
        with_state: bool = True,
        fino_meta: Tensor | None = None,
        fino_gamma: float = 0.0,
        subject_ids: Tensor | None = None,
    ):
        # images (float) and masks (bool) on their own voxel grids (sample_transform native), fit
        # here to the grid_size mm box in their voxels. images with the same spacing share one
        # token grid, sequence sampling and embedding / head weight. each predicted token counts
        # by its in-mask fraction whatever its voxel count; with 1 mm images only this is
        # forward's loss, bitwise
        assert self.native and num_predict is not None
        seq_length = num_visible + num_predict
        token_mm = self.patch_size[0] * self.target_spacing
        groups: dict[tuple, list[int]] = {}
        for ii, sp in enumerate(spacing):
            groups.setdefault(tuple(sp), []).append(ii)

        device = images[0].device
        geoms = {sp: native_geometry(sp, tuple(self.grid_size), token_mm, device) for sp in groups}
        # embedding and head weights of all non 1 mm spacings in one pass each
        native_sps = [sp for sp in groups if sp != (1.0, 1.0, 1.0)]
        embed_ws, head_ws = {}, {}
        if native_sps:
            sp_patches = [geoms[sp][0] for sp in native_sps]
            embed_ws = self.patch_embed.native_weights(native_sps, sp_patches)
            head_ws = self.decoder_head.native_weights(native_sps, sp_patches)
            embed_ws, head_ws = dict(zip(native_sps, embed_ws)), dict(zip(native_sps, head_ws))

        parts = []
        for sp, ids in groups.items():
            patch, box, grid, _ = geoms[sp]
            # images and masks fit to the box (fit_to_shape) and padded up to whole patches, tiling
            # the box from its low corner, in one buffer
            padded = tuple(num * p for num, p in zip(grid, patch))
            vols = images[ids[0]].new_zeros((2, len(ids)) + padded)
            for jj, ii in enumerate(ids):
                src, dst = fit_slices(images[ii].shape, box)
                vols[0, jj][dst] = images[ii][src]
                vols[1, jj][dst] = mask[ii][src]
            raw = vols if (self.hemiswap_head is not None and self.training) else None
            vols = patchify3d(vols.flatten(0, 1), patch).unflatten(0, (2, len(ids)))
            cube = (grid, self.mask_cube) if self.mask_cube else None
            scores = sequence_scores(vols[1], num_samples, self.min_mask_frac, cube=cube).flatten(0, 1)
            parts.append({"spacing": sp, "patch": patch, "ids": ids, "vols": vols, "grid": grid})
            parts[-1]["scores"] = scores
            if raw is not None:
                parts[-1]["raw"] = raw

        # one argsort for all groups, rows padded with 3.0 (after every score). sorts this long are
        # stable, so these are the per group argsorts
        num_tokens = [part["scores"].shape[1] for part in parts]
        num_rows = [part["scores"].shape[0] for part in parts]
        assert min(num_tokens) >= seq_length
        if len(parts) == 1:
            scores = parts[0]["scores"]
        else:
            scores = parts[0]["scores"].new_full((sum(num_rows), max(num_tokens)), 3.0)
            row = 0
            for part, num, rows in zip(parts, num_tokens, num_rows):
                scores[row : row + rows, :num] = part["scores"]
                row += rows
        orders = scores.argsort(dim=1, stable=True)[:, :seq_length].split(num_rows)

        for part, order, num in zip(parts, orders, num_tokens):
            sp, patch, vols = part["spacing"], part["patch"], part.pop("vols")
            del part["scores"]
            b = len(part["ids"])
            part["order"] = order.unflatten(0, (b, num_samples))
            # images and masks gathered together by flat token ids
            flat_ids = part["order"] + gather_offsets(b, num, device)
            patches = vols.flatten(0, 2).index_select(0, flat_ids.flatten())
            patches, mask_patches = patches.view(2, b * num_samples, seq_length, -1)
            coord = geoms[sp][3][0][order]
            vis_patches, part["target_patches"] = patches.split([num_visible, num_predict], 1)
            part["target_mask_patches"] = mask_patches[:, num_visible:]
            vis_mask = mask_patches[:, :num_visible]
            if self.shuffle_head is not None and self.training:
                # displaced visible tokens first (donors are clean same-scan patches); rtd below
                # may then overwrite some of them with a foreign patch, which clears the label
                vis_patches, vis_mask, part["shuffled"] = self.shuffle_replace(vis_patches, vis_mask)
            if self.rtd_head is not None and self.training:
                if b >= 2:
                    vis_patches, vis_mask, part["rtd_label"] = self.rtd_replace(
                        vis_patches, vis_mask, order, vols, b, num, num_samples, num_visible
                    )
                else:
                    part["rtd_label"] = torch.zeros(
                        vis_patches.shape[:2], dtype=torch.bool, device=device
                    )
                if "shuffled" in part:
                    part["shuffled"] = part["shuffled"] & ~part["rtd_label"]
            # ms-aux: brain frame, hemisphere swap, corruption (each token carries one label)
            if self.training and (self.brainpos_head is not None or self.hemiswap_head is not None):
                vis_order = order[:, :num_visible]
                img_of_row = torch.arange(b, device=device).repeat_interleave(num_samples)
                centres, cen, half = self.brain_frame(vols[1], part["grid"], patch)
                if self.brainpos_head is not None:
                    rel = (centres[vis_order] - cen[img_of_row][:, None]) / half[img_of_row][:, None]
                    part["bp_target"] = rel
                    sp_t = torch.tensor(sp, device=device, dtype=torch.float32)
                    part["bp_scale"] = (half * sp_t)[img_of_row][:, None].expand_as(rel)
                    part["bp_hide"] = torch.rand(vis_order.shape, device=device) < self.brainpos_hide_frac
                if self.hemiswap_head is not None:
                    vis_patches, vis_mask, part["hemi_label"] = self.hemiswap_replace(
                        vis_patches, vis_mask, part.pop("raw"), vis_order, part["grid"], patch, cen,
                        img_of_row, part.get("rtd_label"),
                    )
            if self.corrupt_head is not None and self.training:
                excl = part.get("rtd_label")
                if "hemi_label" in part:
                    excl = part["hemi_label"] if excl is None else excl | part["hemi_label"]
                vis_patches, part["corrupt_label"] = self.corrupt_transform(vis_patches, vis_mask, patch, excl)
            if self.orient_head is not None and self.training:
                vis_patches, vis_mask, part["orient_label"] = self.orient_transform(
                    vis_patches, vis_mask, patch
                )
            part["vis_coord"], part["target_coord"] = coord.split([num_visible, num_predict], 1)
            part["embeds"] = self.patch_embed.native(vis_patches, sp, patch, embed_ws.get(sp))
            if self.level_head is not None or (self.edgelvl_head is not None and self.training):
                part["vis_patches"], part["vis_mask"] = vis_patches, vis_mask

        def cat(key: str) -> Tensor:
            # contiguous for one group too, the compiled encoder would recompile on view strides
            if len(parts) == 1:
                return parts[0][key].contiguous()
            return torch.cat([part[key] for part in parts])

        def flat_cat(tensors: list[Tensor]) -> Tensor:
            if len(tensors) == 1:
                return tensors[0].flatten()
            return torch.cat([t.flatten() for t in tensors])

        bp_on = self.brainpos_head is not None and self.training and "bp_target" in parts[0]
        embeds = self.encode_tokens(cat("embeds"), cat("vis_coord"), cat("bp_hide") if bp_on else None)
        rtd_on = self.rtd_head is not None and self.training and "rtd_label" in parts[0]
        rtd_logits, vis_gate = None, None
        if rtd_on:
            rtd_logits = self.rtd_head(embeds[:, self.class_tokens :]).squeeze(-1).float()
            if self.rtd_gate_scale > 0.0:
                vis_gate = (2.0 * torch.sigmoid(-self.rtd_gate_scale * rtd_logits)).clamp(max=1.0)
                vis_gate = vis_gate.to(embeds.dtype)
        feats = self.decode_features(embeds, cat("target_coord"), vis_gate)
        feats = torch.split(feats, [len(part["ids"]) * num_samples for part in parts])
        # graft: the v3 loss stack, extended to per-spacing native patches. norm_target
        # z-scores each token's voxels (spacing groups differ in patch size), the fg mask
        # weights the mse per voxel, and the xyz gradient penalty runs per group on its own
        # patch geometry; the global loss is the weight-masked mean across groups
        losses, wts = [], []
        grad_num, grad_den = [], []
        for part, feat in zip(parts, feats):
            sp = part["spacing"]
            pred = self.decoder_head.native(feat, sp, part["patch"], head_ws.get(sp))
            part["pred_patches"] = pred
            px, py, pz = part["patch"]
            tgt = part["target_patches"]
            if self.target_lp_mm > 0.0 and self.target_band < 1.0:
                # coarse-to-fine target bandwidth curriculum: blend the target toward its
                # target_lp_mm block-mean (low band) projection; band 0 = low band only, 1 = full
                low = lowpass_patches(tgt, (px, py, pz), sp, self.target_lp_mm)
                tgt = low + self.target_band * (tgt - low)
            if self.norm_target:
                tgt = (tgt - tgt.mean(dim=-1, keepdim=True)) / (tgt.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
            mse = F.mse_loss(pred, tgt, reduction="none")
            mask = part["target_mask_patches"]
            w = mask / pred.shape[-1]
            if self.fg_loss_weight > 0.0:
                fg_w = 1.0 + self.fg_loss_weight * (part["target_patches"].abs() > 0.05).to(w.dtype)
                w = w * fg_w
            losses.append((w * mse).flatten())
            wts.append(w.flatten())
            if self.grad_loss_weight > 0.0:
                BM = pred.shape[0] * pred.shape[1]
                pv = pred.reshape(BM, px, py, pz)
                tv = tgt.reshape(BM, px, py, pz)
                mv = mask.reshape(BM, px, py, pz).to(w.dtype)
                dx = F.mse_loss(pv[:, 1:], tv[:, 1:] - tv[:, :-1], reduction="none")
                dy = F.mse_loss(pv[:, :, 1:], tv[:, :, 1:] - tv[:, :, :-1], reduction="none")
                dz = F.mse_loss(pv[:, :, :, 1:], tv[:, :, :, 1:] - tv[:, :, :, :-1], reduction="none")
                mx = mv[:, 1:] * mv[:, :-1]
                my = mv[:, :, 1:] * mv[:, :, :-1]
                mz = mv[:, :, :, 1:] * mv[:, :, :, :-1]
                grad_num += [(mx * dx).sum(), (my * dy).sum(), (mz * dz).sum()]
                grad_den += [mx.sum(), my.sum(), mz.sum()]
        loss = flat_cat(losses).sum() / flat_cat(wts).sum()
        if self.grad_loss_weight > 0.0:
            loss = loss + self.grad_loss_weight * (
                grad_num[0] / grad_den[0].clamp_min(1.0)
                + grad_num[1] / grad_den[1].clamp_min(1.0)
                + grad_num[2] / grad_den[2].clamp_min(1.0)
            ) / 3.0
        # an all 1 mm batch leaves the residuals without a gradient, and ddp needs one every step
        residuals = (self.patch_embed.residual, self.decoder_head.residual)
        residuals += (self.decoder_head.bias_residual,)
        loss = loss + 0.0 * sum(r.sum() for r in residuals)
        if self.level_head is not None:
            num, den = 0.0, 0.0
            rows = torch.split(embeds, [len(part["ids"]) * num_samples for part in parts])
            for part, tokens in zip(parts, rows):
                n, d = self.level_terms(
                    tokens[:, self.class_tokens :], part["vis_patches"], part["vis_mask"],
                    part["patch"],
                )
                num, den = num + n, den + d
            loss = loss + self.level_weight * num / den.clamp_min(1e-6)
        if self.fino_heads is not None and fino_meta is not None:
            # the global embedding of each sequence: mean of its visible encoder tokens (eval:
            # mean of all brain-touching tokens); rows are part by part, image-major
            pooled = embeds[:, self.class_tokens :].mean(dim=1)
            # fino_meta stays on the cpu: one pinned non blocking copy, no sync with the gpu queue
            rows = torch.tensor([ii for part in parts for ii in part["ids"]]).repeat_interleave(num_samples)
            meta = fino_meta.cpu()[rows]
            if device.type == "cuda":
                meta = meta.pin_memory().to(device, non_blocking=True)
            fino_loss, self.fino_stats = self.fino_heads(pooled, meta, fino_gamma)
            loss = loss + fino_loss
        if rtd_on:
            labels = cat("rtd_label").float()
            logits = rtd_logits
            rtd_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + self.rtd_weight * rtd_loss
            with torch.no_grad():
                pos = labels > 0.5
                acc_pos = ((logits > 0) & pos).sum() / pos.sum().clamp_min(1)
                acc_neg = ((logits <= 0) & ~pos).sum() / (~pos).sum().clamp_min(1)
            self.rtd_stats = {"rtd_loss": rtd_loss.detach(), "rtd_tpr": acc_pos, "rtd_tnr": acc_neg}
            if vis_gate is not None:
                with torch.no_grad():
                    g = vis_gate.float()
                    self.rtd_stats["rtd_gate_pos"] = (g * pos).sum() / pos.sum().clamp_min(1)
                    self.rtd_stats["rtd_gate_neg"] = (g * ~pos).sum() / (~pos).sum().clamp_min(1)
            self.fino_stats = {**self.fino_stats, **self.rtd_stats}
        if self.orient_head is not None and self.training and "orient_label" in parts[0]:
            labels = cat("orient_label").flatten()
            logits = self.orient_head(embeds[:, self.class_tokens :]).float().flatten(0, 1)
            orient_loss = F.cross_entropy(logits, labels)
            loss = loss + self.orient_weight * orient_loss
            with torch.no_grad():
                pred = logits.argmax(-1)
                rot = labels > 0
                acc_rot = ((pred == labels) & rot).sum() / rot.sum().clamp_min(1)
                acc_id = ((pred == 0) & ~rot).sum() / (~rot).sum().clamp_min(1)
            self.fino_stats = {
                **self.fino_stats, "orient_loss": orient_loss.detach(),
                "orient_acc_rot": acc_rot, "orient_acc_id": acc_id,
            }
        if self.shuffle_head is not None and self.training:
            # displaced-token detection on the visible encoder tokens (rows part by part, as embeds)
            labels = torch.cat([part.pop("shuffled") for part in parts]).float()
            logits = self.shuffle_head(embeds[:, self.class_tokens :]).float().squeeze(-1)
            # balanced BCE: positives (~ratio) and negatives weigh equally, so the head cannot
            # win by predicting "clean" everywhere
            pos = labels.sum().clamp_min(1.0)
            neg = (1.0 - labels).sum().clamp_min(1.0)
            weight = labels / pos + (1.0 - labels) / neg
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = loss + self.shuffle_weight * 0.5 * (weight * bce).sum()
            with torch.no_grad():
                pred = logits > 0
                self.shuffle_stats = {
                    "shuf_frac": labels.mean().detach(),
                    "shuf_recall": ((pred & (labels > 0)).sum() / pos).detach(),
                    "shuf_fpr": ((pred & (labels == 0)).sum() / neg).detach(),
                }

        aux_on = (self.brainpos_head, self.scanmatch_proj, self.hemiswap_head, self.corrupt_head, self.edgelvl_head)
        if self.training and any(h is not None for h in aux_on):
            loss = loss + self.aux_losses(parts, embeds, num_samples, subject_ids, device)
        for part in parts:
            part.pop("vis_patches", None)
            part.pop("vis_mask", None)

        if not with_state:
            return loss
        return loss, {"parts": parts, "embeds": embeds}

    def aux_losses(
        self, parts: list[dict], embeds: Tensor, num_samples: int, subject_ids: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        # the ms-aux pretext losses (training only); stats (detached tensors) in self.aux_stats
        loss = embeds.new_zeros((), dtype=torch.float32)
        stats = {}
        tokens = embeds[:, self.class_tokens :]

        def cat(key: str) -> Tensor:
            return torch.cat([part[key] for part in parts])

        if self.brainpos_head is not None and "bp_target" in parts[0]:
            pred = self.brainpos_head(tokens).float()
            tgt = cat("bp_target")
            bp_loss = F.smooth_l1_loss(pred, tgt, beta=0.1)
            loss = loss + self.brainpos_weight * bp_loss
            with torch.no_grad():
                err = ((pred - tgt) * cat("bp_scale")).norm(dim=-1)
                hide = cat("bp_hide")
                stats["bp_loss"] = bp_loss.detach()
                stats["bp_err_mm_hidden"] = (err * hide).sum() / hide.sum().clamp_min(1)
                stats["bp_err_mm_shown"] = (err * ~hide).sum() / (~hide).sum().clamp_min(1)
        if self.scanmatch_proj is not None:
            img = torch.tensor([ii for part in parts for ii in part["ids"]]).repeat_interleave(num_samples)
            pooled = tokens.mean(dim=1)
            with torch.autocast(device_type=device.type, enabled=False):
                z = F.normalize(self.scanmatch_proj(pooled.float()), dim=-1)
                sim = z @ z.T / self.scanmatch_temp
            img = img.to(device, non_blocking=True)
            same = img[:, None] == img[None]
            eye = torch.eye(len(img), dtype=torch.bool, device=device)
            pos = same & ~eye
            block = eye
            if subject_ids is not None:
                subj = subject_ids.to(device, non_blocking=True)[img]
                same_subj = (subj[:, None] == subj[None]) & ~same
                block = block | same_subj
                stats["sm_subj_masked"] = same_subj.float().sum(1).mean()
            sim = sim.masked_fill(block, -1e4)
            lse_all = torch.logsumexp(sim, dim=1)
            lse_pos = torch.logsumexp(sim.masked_fill(~pos, -1e4), dim=1)
            sm_loss = (lse_all - lse_pos).mean()
            loss = loss + self.scanmatch_weight * sm_loss
            with torch.no_grad():
                stats["sm_loss"] = sm_loss.detach()
                stats["sm_acc"] = pos.gather(1, sim.argmax(1, keepdim=True)).float().mean()
        if self.hemiswap_head is not None and "hemi_label" in parts[0]:
            labels = cat("hemi_label").float()
            logits = self.hemiswap_head(tokens).float().squeeze(-1)
            hs_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + self.hemiswap_weight * hs_loss
            with torch.no_grad():
                p = labels > 0.5
                stats["hs_loss"] = hs_loss.detach()
                stats["hs_frac"] = labels.mean()
                stats["hs_tpr"] = ((logits > 0) & p).sum() / p.sum().clamp_min(1)
                stats["hs_tnr"] = ((logits <= 0) & ~p).sum() / (~p).sum().clamp_min(1)
        if self.corrupt_head is not None and "corrupt_label" in parts[0]:
            labels = cat("corrupt_label").flatten()
            logits = self.corrupt_head(tokens).float().flatten(0, 1)
            cr_loss = F.cross_entropy(logits, labels)
            loss = loss + self.corrupt_weight * cr_loss
            with torch.no_grad():
                pred = logits.argmax(-1)
                bad = labels > 0
                stats["cr_loss"] = cr_loss.detach()
                stats["cr_acc_corrupt"] = ((pred == labels) & bad).sum() / bad.sum().clamp_min(1)
                stats["cr_acc_clean"] = ((pred == 0) & ~bad).sum() / (~bad).sum().clamp_min(1)
        if self.edgelvl_head is not None and "vis_patches" in parts[0]:
            num, den, err = 0.0, 0.0, 0.0
            rows = torch.split(tokens, [len(part["ids"]) * num_samples for part in parts])
            for part, tok in zip(parts, rows):
                n, d, e = self.edgelvl_terms(
                    tok, part["vis_patches"], part["vis_mask"], part["patch"], part["spacing"]
                )
                num, den, err = num + n, den + d, err + e
            el_loss = num / den.clamp_min(1e-6)
            loss = loss + self.edgelvl_weight * el_loss
            stats["el_loss"] = el_loss.detach()
            stats["el_mae"] = (err / den.clamp_min(1e-6)).detach()
        self.aux_stats = stats
        return loss

    def compile_native(self, **kwargs) -> None:
        # the fixed-shape transformer parts. per-spacing tokenization, embedding and head stay eager
        self.encode_tokens = torch.compile(self.encode_tokens, **kwargs)
        self.decode_features = torch.compile(self.decode_features, **kwargs)

    def compile_cudagraphs(self) -> None:
        # training speed, same math: sampling + gather compiled as usual, the encoder / decoder / loss
        # fwd and bwd replayed as cuda graphs (one launch each instead of ~800 python launches)
        self.sample_patches = torch.compile(self.sample_patches_contiguous, dynamic=False)
        self.patch_loss = torch.compile(self.patch_loss, mode="reduce-overhead", dynamic=False)

    # the methods below implement the brainmarks model contract.
    #   - forward_embeddings: computes a global and dense embedding per input sample
    #   - transform: cpu data transform, runs in loader workers (optional)
    #   - transform_cuda: cuda data transform. you call it inside forward_embeddings (optional)

    def transform(self, sample: ImageInput) -> dict[str, Tensor]:
        image = torch.from_numpy(sample["image"].get_fdata(dtype=np.float32))
        mask = torch.from_numpy(np.asarray(sample["brain_mask"].dataobj) > 0)
        affine = sample["image"].affine
        image, mask, affine = crop_to_mask(image, mask, affine)
        return {"values": image[mask], "mask": mask, "affine": affine}

    def transform_cuda(self, sample: dict[str, Tensor]) -> dict:
        values = sample["values"]
        mask = sample["mask"]
        affine = sample["affine"]
        values = normalize_values(values, self.clip_quantiles)
        image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
        image, mask, affine = reorient_to_ras(image, mask, affine)
        image, mask, affine = resample_to_spacing(image, mask, affine, self.target_spacing)
        image, mask, affine = fit_to_shape(image, mask, affine, self.grid_size)
        return {"image": image, "mask": mask, "affine": affine}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        device = self.coord_grid.device
        patch_grid_size = [size // self.patch_size[0] for size in self.grid_size]

        # one forward per image since the number of patches in the mask varies
        outputs = []
        for sample in samples:
            sample = {
                "values": sample["values"].to(device),
                "mask": sample["mask"].to(device),
                "affine": sample["affine"],
            }
            sample = self.transform_cuda(sample)
            patches = patchify3d(sample["image"][None], self.patch_size[0])
            in_mask = (patchify3d(sample["mask"][None], self.patch_size[0]) > 0).any(dim=2)[0]
            # cudnn attention builds a new graph (~45 ms) for each new sequence length
            with (
                sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                tokens = self.forward_encoder(patches[:, in_mask], self.coord_grid[:, in_mask])
            tokens = tokens[0, self.class_tokens :].float()

            output = EmbeddingOutput(
                global_embedding=tokens.mean(dim=0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dense = tokens.new_zeros(len(in_mask), self.embed_dim)
                dense[in_mask] = tokens
                output["dense_embedding"] = dense.reshape(*patch_grid_size, self.embed_dim)
                output["dense_affine"] = sample["affine"]
                output["dense_mask"] = in_mask.reshape(patch_grid_size)
            outputs.append(output)
        return outputs


@register_model
def vit_mae_3d(checkpoint: str, random_weights: bool = False) -> ViTMAE3D:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ViTMAE3D.from_config(ckpt["args"])
    if not random_weights:
        model.load_state_dict(ckpt["model"])
    return model


def patchify3d(x: Tensor, patch_size: int | tuple[int, int, int] = 8) -> Tensor:
    px, py, pz = (patch_size,) * 3 if isinstance(patch_size, int) else patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=px, py=py, pz=pz)
    return x


@lru_cache(maxsize=4096)
def native_geometry(
    spacing: tuple[float, float, float],
    box_mm: tuple[int, int, int],
    token_mm: float,
    device: torch.device | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Float[Tensor, "1 N 3"]]:
    # per axis: patch voxels (~token_mm), the box in voxels, the token grid tiling the box from
    # its low corner, and token centres in position table units, centred like fit_to_shape. at
    # 1 mm these are make_coord_grid's coordinates
    patch = tuple(max(1, round(token_mm / sp)) for sp in spacing)
    box = native_box(spacing, box_mm)
    grid = tuple(math.ceil(size / p) for size, p in zip(box, patch))
    centres = []
    for sp, p, size, num, mm in zip(spacing, patch, box, grid, box_mm):
        voxel = (torch.arange(num, dtype=torch.float64, device=device) + 0.5) * p
        centres.append((((voxel - size / 2) * sp + mm / 2) / token_mm - 0.5).float())
    coord = torch.stack(torch.meshgrid(*centres, indexing="ij"), dim=-1).reshape(1, -1, 3)
    return patch, box, grid, coord


def unpatchify3d(x: Tensor, grid_size: tuple[int, int, int], patch_size: int = 8) -> Tensor:
    p = patch_size
    gx, gy, gz = [size // patch_size for size in grid_size]
    x = rearrange(
        x,
        "b (gx gy gz) (px py pz) -> b (gx px) (gy py) (gz pz)",
        gx=gx,
        gy=gy,
        gz=gz,
        px=p,
        py=p,
        pz=p,
    )
    return x


def patches_to_volume(
    patches: Float[Tensor, "B L P"],
    patch_ids: Int[Tensor, "B L"],
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
) -> Float[Tensor, "B X Y Z"]:
    B, L, P = patches.shape
    gx, gy, gz = [size // patch_size for size in grid_size]
    grid = patches.new_zeros(B, gx * gy * gz, P)
    batch_ids = torch.arange(B, device=patches.device)[:, None]
    grid[batch_ids, patch_ids] = patches
    images = unpatchify3d(grid, grid_size=grid_size, patch_size=patch_size)
    return images


def make_coord_grid(
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
    device: torch.device | None = None,
) -> Tensor:
    patch_grid_size = tuple(size // patch_size for size in grid_size)
    grid_ids = [torch.arange(size, device=device) for size in patch_grid_size]
    coords = torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1)
    coords = coords.reshape(1, -1, 3)
    return coords


def sample_sequences(
    mask_patches: torch.Tensor,
    num_samples: int,
    seq_length: int | None = None,
    min_mask_frac: float = 0.25,
    cube: tuple[tuple[int, int, int], int] | None = None,
) -> torch.Tensor:
    scores = sequence_scores(mask_patches, num_samples, min_mask_frac, cube)
    if seq_length is None:
        # one host sync to get the longest mask sequence in the batch
        mask_frac = mask_patches.float().mean(dim=2)
        seq_length = (mask_frac >= min_mask_frac).sum(dim=1).max().item()
    order = scores.argsort(dim=2)[:, :, :seq_length]
    return order


def sequence_scores(
    mask_patches: torch.Tensor,
    num_samples: int,
    min_mask_frac: float = 0.25,
    cube: tuple[tuple[int, int, int], int] | None = None,
) -> torch.Tensor:
    # sample_sequences' sort keys, [B, num_samples, N]
    batch_size, num_patches, _ = mask_patches.shape
    mask_frac = mask_patches.float().mean(dim=2)
    # one independent shuffle per sequence, with patches outside the mask sorted last so
    # they are only used when a volume has too few mask patches
    scores = torch.rand(batch_size, num_samples, num_patches, device=mask_patches.device)
    if cube is not None:
        # cube: (patch grid, edge in patches). one uniform score per cube, shared by its patches,
        # so they are ordered (visible / predicted) together; the white noise only breaks ties
        # inside a cube. The cube grid is shifted by a random offset per sequence, so no two
        # patches are always in the same cube
        grid, edge = cube
        num_seq = batch_size * num_samples
        coords = grid_coords(tuple(grid), scores.device)
        offset = torch.randint(0, edge, (num_seq, 1, 3), device=scores.device)
        cells = (coords + offset) // edge
        cx, cy, cz = [size // edge + 2 for size in grid]
        cell_ids = (cells[..., 0] * cy + cells[..., 1]) * cz + cells[..., 2]
        cell_scores = torch.rand(num_seq, cx * cy * cz, device=scores.device)
        cell_scores = torch.gather(cell_scores, 1, cell_ids).reshape(batch_size, num_samples, num_patches)
        scores = cell_scores + 1e-6 * scores
    return torch.where(mask_frac[:, None, :] >= min_mask_frac, scores, 2.0)


def lowpass_patches(
    tgt: Tensor, patch: tuple[int, int, int], spacing: tuple[float, ...], block_mm: float
) -> Tensor:
    """Project every [.., px*py*pz] voxel patch onto its low band: the mean of each block of
    ~block_mm (per axis round(block_mm / spacing), at least 1 voxel; edge blocks partial), held
    over the block's voxels. A box-filter + decimation + hold, i.e. the coarse approximation of a
    Haar pyramid; the high band (tgt - low) carries most of the per-voxel thermal noise."""
    px, py, pz = patch
    ks = tuple(max(1, int(round(block_mm / float(s)))) for s in spacing)
    ks = tuple(min(k, p) for k, p in zip(ks, (px, py, pz)))
    if ks == (1, 1, 1):
        return tgt
    lead = tgt.shape[:-1]
    vol = tgt.reshape(-1, 1, px, py, pz).float()
    low = F.avg_pool3d(vol, kernel_size=ks, stride=ks, ceil_mode=True)
    for dim, k, p in zip((2, 3, 4), ks, (px, py, pz)):
        low = low.repeat_interleave(k, dim=dim).narrow(dim, 0, p)
    return low.reshape(*lead, px * py * pz).to(tgt.dtype)


def fit_slices(
    shape: tuple[int, ...], box: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    # fit_to_shape's centre pad / crop as (source, destination) slices
    src, dst = [], []
    for size, target in zip(shape, box):
        before = (target - size) // 2
        num = min(size, target)
        src.append(slice(max(0, -before), max(0, -before) + num))
        dst.append(slice(max(0, before), max(0, before) + num))
    return tuple(src), tuple(dst)


@lru_cache(maxsize=4096)
def gather_offsets(batch_size: int, num_patches: int, device: torch.device) -> Tensor:
    # [2, B, 1, 1] flat row offsets of (image | mask, image) in a [2, B, N, P] patch tensor
    return torch.arange(2 * batch_size, device=device).view(2, batch_size, 1, 1) * num_patches


@lru_cache(maxsize=4096)
def grid_coords(grid: tuple[int, int, int], device: torch.device) -> Tensor:
    axes = [torch.arange(size, device=device) for size in grid]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)


# JAX ViT xavier uniform init
# https://github.com/facebookresearch/capi/blob/main/model.py
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
