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
        # extended corruption kinds (corrupt head and unified head): + 4 = gamma / contrast change
        # (patch-range gamma, |log g| in [0.2, 0.5]) and 5 = 2x downsample-upsample (masked 2-voxel
        # block mean held over the block; axes under 4 voxels untouched)
        corrupt_extended: bool = False,
        # second extension (needs corrupt_extended): + 6 = linear contrast change about the patch's
        # in-brain mean (x mean + c (x - mean), |log c| in [0.2, 0.5], floored at the row minimum; the
        # gamma variant of a "contrast shift" is class 4 already) and 7 = motion ghost (the patch plus a
        # copy shifted half the patch along one random axis, circular like a k-space ghost, amplitude
        # a in [0.2, 0.5], renormalized by 1 + a x ghost-mask so the mean level stays put)
        corrupt_extended2: bool = False,
        # unified corruption: ONE shared budget and ONE multi-class token head replacing the rtd and
        # corrupt heads (both must be off). a unified_ratio share of the visible (>= 50% brain)
        # tokens is corrupted, the class uniform among {cross-subject swap (rtd_replace), bias,
        # blur, noise (+ gamma, downsample if corrupt_extended) (+ within-scan shuffle if
        # unified_shuffle)}; a token whose class is ineligible (no brain partner / donor) stays
        # clean. zero-init Linear(D, 1 + classes), CE weight unified_weight. MAE targets stay clean
        unified_weight: float = 0.0,
        unified_ratio: float = 0.25,
        unified_shuffle: bool = False,
        # edge level-set: the level head construction on the gradient magnitude (central
        # differences per mm on the patch's own voxel grid), thresholds k x the row's in-brain
        # median gradient, level_pool sub-blocks, BCE weighted by brain fraction
        edgelvl_weight: float = 0.0,
        edgelvl_thresholds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
        # ---- ms-syn (ported from r24_syn, Labless run_sub_463a0d7b5e; all off by default) ----
        # level_mirror (r18_mir_dec, native path; () = off): extra level channels on a separate
        # zero-init head: in-mask fraction of each 2 mm sub-block above each threshold AND above
        # margin + the max within +-2-4 mm of its x(L-R)-mirrored 2 mm cell
        level_mirror: tuple[float, ...] = (),
        level_mirror_margin: float = 0.5,
        # training only: share of images given 1 to blob_max synthetic focal hyperintensities,
        # radius log-uniform in blob_radius mm, + blob_contrast z, soft edge blob_edge mm
        blob_prob: float = 0.0,
        blob_max: int = 3,
        blob_radius: tuple[float, float] = (2.0, 9.0),
        blob_contrast: tuple[float, float] = (1.0, 3.0),
        blob_edge: float = 1.5,
        # training only: share of images given one lobulated mass, radius log-uniform in tumor_radius
        # mm: a bright rim on a mildly darker core (share tumor_ring) or a diffuse bright mass, +
        # tumor_contrast z, 10-90% edge width tumor_edge mm; a tumor_nodule share are solid bright
        # 3-10 mm nodules instead. rims and nodules +2..+5 z, every image clipped to its own in-mask
        # range. pasted into the clean volumes before patchify and before any token corruption
        tumor_prob: float = 0.0,
        tumor_radius: tuple[float, float] = (8.0, 30.0),
        tumor_ring: float = 0.5,
        tumor_contrast: tuple[float, float] = (0.8, 2.5),
        tumor_edge: tuple[float, float] = (3.0, 8.0),
        tumor_nodule: float = 0.25,
        # h-tneg: share of masses whose amplitude is sign-flipped (rim / nodule / diffuse mass darker than
        # the tissue instead of brighter; a ring's core darkening keeps its sign, so a flipped ring is a
        # dark mass with a darker core). 0 = off: no extra draw, the paste and its RNG stream unchanged
        tumor_neg_prob: float = 0.0,
        # zero-init linear head on the visible encoder tokens: BCE to the inserted lesion (blobs,
        # tumours) fraction of each level_pool^3 sub-block, in-mask weighted (weight; 0 = off).
        # tokens a corruption replaced or altered (rtd / unified / hemiswap / corrupt / orient /
        # shuffle) are out of this loss and of the mirror level loss
        lesion_mask_weight: float = 0.0,
        # aux schedules over training progress t in [0, 1] (the LR schedule's clock; set each step by
        # the training loop through set_aux_progress; None = off, scale 1). [start, end] fractions:
        # *_ramp scales linearly 0 -> 1 between them, unified_decay 1 -> 0. unified_ramp / unified_decay
        # scale unified_weight, lesion_ramp lesion_mask_weight, tumor_prob_ramp the paste probability
        unified_ramp: tuple[float, float] | None = None,
        unified_decay: tuple[float, float] | None = None,
        lesion_ramp: tuple[float, float] | None = None,
        tumor_prob_ramp: tuple[float, float] | None = None,
        # h-tdecay: the paste probability scales 1 -> 0 between [start, end] (no paste after end)
        tumor_prob_decay: tuple[float, float] | None = None,
        # multi-level feature fusion (MFF, Liu et al. ICCV 2023; port of loop n0119): training-only
        # zero-init per-layer projections (after a LayerNorm) of these encoder levels (0 = the
        # position-embedded patch embedding, l = block l output) summed, scaled by fuse_scale, and
        # added to the projected encoder tokens at the decoder input. the evaluated encoder output
        # is unchanged. empty / scale 0 = off (no params)
        fuse_layers: tuple[int, ...] = (),
        fuse_scale: float = 1.0,
        # tumours in ONE view only: the num_samples sequences of a pasted image gather the pasted
        # volume in view 0 and the clean (pre-paste) volume in every other view, whose lesion
        # targets are 0 (training only; False = every view sees the paste)
        tumor_one_view: bool = False,
    ):
        super().__init__()
        self.tumor_one_view = bool(tumor_one_view)
        assert not tumor_one_view or tumor_prob, "tumor_one_view: needs tumor_prob"
        assert not (tumor_one_view and level_mirror), "tumor_one_view: mirror refs come from the pasted volume"
        assert not (tumor_one_view and blob_prob), "tumor_one_view: blobs are pasted into every view"
        self.fuse_layers = tuple(int(l) for l in fuse_layers) if fuse_scale else ()
        self.fuse_scale = float(fuse_scale)
        assert all(0 <= l < depth for l in self.fuse_layers), "fuse_layers: 0 (embed) .. depth-1"
        assert len(set(self.fuse_layers)) == len(self.fuse_layers), "fuse_layers: duplicates"
        self.aux_schedules = {
            k: (float(v[0]), float(v[1]))
            for k, v in (("unified_ramp", unified_ramp), ("unified_decay", unified_decay),
                         ("lesion_ramp", lesion_ramp), ("tumor_prob_ramp", tumor_prob_ramp),
                         ("tumor_prob_decay", tumor_prob_decay))
            if v is not None
        }
        for k, (a, b) in self.aux_schedules.items():
            assert 0.0 <= a <= b <= 1.0, f"{k}: need 0 <= start <= end <= 1, got {[a, b]}"
        # multipliers the schedules drive (exactly 1.0 when off: x * 1.0 == x, the flags-off path is unchanged)
        self.unified_scale = 1.0
        self.lesion_scale = 1.0
        self.tumor_scale = 1.0
        self.brainpos_weight = float(brainpos_weight)
        self.brainpos_hide_frac = float(brainpos_hide_frac)
        self.scanmatch_weight = float(scanmatch_weight)
        self.scanmatch_temp = float(scanmatch_temp)
        self.hemiswap_weight = float(hemiswap_weight)
        self.hemiswap_ratio = float(hemiswap_ratio)
        self.corrupt_weight = float(corrupt_weight)
        self.corrupt_ratio = float(corrupt_ratio)
        assert not corrupt_extended2 or corrupt_extended, "corrupt_extended2: needs corrupt_extended"
        self.corrupt_kinds = 7 if corrupt_extended2 else 5 if corrupt_extended else 3
        self.unified_weight = float(unified_weight)
        self.unified_ratio = float(unified_ratio)
        self.unified_shuffle = bool(unified_shuffle)
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
            self.corrupt_head = zero_linear(1 + self.corrupt_kinds)
        self.unified_head = None
        if self.unified_weight > 0.0:
            assert native, "unified corruption: native training only"
            assert self.rtd_weight == 0.0 and self.corrupt_weight == 0.0, (
                "unified corruption replaces the rtd and corrupt heads: set rtd_weight = corrupt_weight = 0"
            )
            # classes: 0 clean, 1 swap, 2 .. 1 + corrupt_kinds corruptions, (last) shuffle
            self.unified_classes = 2 + self.corrupt_kinds + int(self.unified_shuffle)
            self.unified_head = zero_linear(self.unified_classes)
        self.edgelvl_head = None
        if self.edgelvl_weight > 0.0:
            assert native, "edgelvl: native training only"
            edges = [float(t) for t in edgelvl_thresholds]
            self.edgelvl_head = zero_linear(len(edges) * (patch_size // level_pool) ** 3)
            self.register_buffer("edgelvl_thresholds", torch.tensor(edges), persistent=False)

        # ms-syn heads (r24_syn): created under fork_rng and zero-initialized, so every other weight
        # and the rng stream after the constructor are as with them off
        self.level_mirror = tuple(float(t) for t in level_mirror)
        self.level_mirror_margin = float(level_mirror_margin)
        self.level_mirror_head = None
        self.level_mirror_stats = {}
        if self.level_mirror:
            assert level_weight and native, "level_mirror: needs the level head, native path"
            assert patch_size == 8 and level_pool == 2, "level_mirror: 2 mm cells of 8 mm tokens"
            # background (0) is in the mirror window at the mask edge: thresholds >= margin keep it inert
            assert min(self.level_mirror) >= self.level_mirror_margin, "level_mirror: thresholds >= margin"
            with torch.random.fork_rng(devices=[]):
                self.level_mirror_head = nn.Linear(embed_dim, len(self.level_mirror) * 64)
            nn.init.zeros_(self.level_mirror_head.weight)
            nn.init.zeros_(self.level_mirror_head.bias)
            self.register_buffer("level_mirror_thresholds", torch.tensor(self.level_mirror), persistent=False)
        assert not blob_prob or native, "blob_prob: native path only"
        self.blob_prob = float(blob_prob)
        self.blob_max = int(blob_max)
        self.blob_radius = tuple(float(v) for v in blob_radius)
        self.blob_contrast = tuple(float(v) for v in blob_contrast)
        self.blob_edge = float(blob_edge)
        assert not tumor_prob or native, "tumor_prob: native path only"
        self.tumor_prob = float(tumor_prob)
        self.tumor_radius = tuple(float(v) for v in tumor_radius)
        self.tumor_ring = float(tumor_ring)
        self.tumor_contrast = tuple(float(v) for v in tumor_contrast)
        self.tumor_edge = tuple(float(v) for v in tumor_edge)
        self.tumor_nodule = float(tumor_nodule)
        self.tumor_neg_prob = float(tumor_neg_prob)
        assert 0.0 <= self.tumor_neg_prob <= 1.0, "tumor_neg_prob: in [0, 1]"
        self.lesion_mask_weight = float(lesion_mask_weight)
        self.lesion_head = None
        if self.lesion_mask_weight:
            assert native and (blob_prob or tumor_prob), "lesion_mask_weight: native path, blobs or tumours on"
            with torch.random.fork_rng(devices=[]):
                self.lesion_head = nn.Linear(embed_dim, (patch_size // level_pool) ** 3)
            nn.init.zeros_(self.lesion_head.weight)
            nn.init.zeros_(self.lesion_head.bias)
        self.fuse_norms = self.fuse_projs = None
        if self.fuse_layers:
            assert native, "fuse_layers: native path only"
            with torch.random.fork_rng(devices=[]):
                self.fuse_norms = nn.ModuleList([LayerNorm(embed_dim) for _ in self.fuse_layers])
                self.fuse_projs = nn.ModuleList([nn.Linear(embed_dim, decoder_embed_dim) for _ in self.fuse_layers])
            for proj in self.fuse_projs:
                # zero init: the decoder input equals the unfused one at step 0
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

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
        noise at 10% of the patch's in-brain std (+ 4 = gamma, 5 = 2x downsample-upsample with
        corrupt_extended). Returns (patches, labels [rows, V] long)."""
        rows, nv, P = vis_patches.shape
        dev = vis_patches.device
        ok = vis_mask.float().mean(-1) >= 0.5
        if exclude is not None:
            ok = ok & ~exclude
        # a fixed k candidates per row (static shapes, no host sync): the corruptions run on k / nv of
        # the tokens; each candidate is hit with probability corrupt_ratio * nv / k (= ratio overall)
        cand, hit = self.corrupt_candidates(ok, self.corrupt_ratio)
        kind = torch.where(hit, torch.randint(1, 1 + self.corrupt_kinds, hit.shape, device=dev), 0)
        patches = self.corrupt_apply(vis_patches, vis_mask, patch, cand, kind)
        labels = torch.zeros(rows, nv, dtype=torch.long, device=dev).scatter(1, cand, kind)
        return patches, labels

    @torch.no_grad()
    def corrupt_candidates(self, ok: Tensor, ratio: float) -> tuple[Tensor, Tensor]:
        """k = ceil(2 ratio nv) random candidate tokens per row and which of them are hit (each
        with probability ratio * nv / k, and only where ok): (cand [rows, k] long, hit bool)."""
        rows, nv = ok.shape
        dev = ok.device
        k = min(nv, max(1, math.ceil(2 * ratio * nv)))
        cand = torch.rand(rows, nv, device=dev).topk(k, dim=1, largest=False).indices
        hit = (torch.rand(rows, k, device=dev) < ratio * nv / k) & ok.gather(1, cand)
        return cand, hit

    @torch.no_grad()
    def corrupt_apply(
        self, vis_patches: Tensor, vis_mask: Tensor, patch: tuple[int, int, int], cand: Tensor, kind: Tensor
    ) -> Tensor:
        """Apply corruption kind[r, j] (0 = none, 1 bias, 2 blur, 3 noise, 4 gamma, 5 downsample,
        6 linear contrast, 7 motion ghost)
        to the candidate tokens cand [rows, k] of vis_patches. Returns the patches."""
        rows, nv, P = vis_patches.shape
        k = cand.shape[1]
        px, py, pz = patch
        dev = vis_patches.device
        hit = kind > 0
        m_all = vis_mask.float()
        # row minimum over the in-brain visible voxels (the magnitude floor of the z-scored scan)
        lo = torch.where(m_all > 0, vis_patches.float(), math.inf).amin(dim=(1, 2), keepdim=True)
        lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
        idx = cand[..., None].expand(rows, k, P)
        x = vis_patches.gather(1, idx).float()
        m = vis_mask.gather(1, idx).float()
        cnt = m.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(-1, keepdim=True) / cnt
        std = (((x - mean) ** 2 * m).sum(-1, keepdim=True) / cnt).sqrt()
        # (1) bias ramp along a random direction: |u . r| <= 1 with r in [-1, 1]^3
        axes = [((torch.arange(p, device=dev) + 0.5) / p * 2 - 1) for p in (px, py, pz)]
        r = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        u = torch.randn(rows, k, 3, device=dev)
        u = u / u.abs().sum(-1, keepdim=True).clamp_min(1e-6)
        amp = 0.2 + 0.1 * torch.rand(rows, k, 1, device=dev)
        x_bias = (x - lo) * (1 + amp * (u @ r.T)) + lo
        # (2) separable gaussian blur, sigma 1 voxel, normalized convolution inside the mask
        kern = torch.exp(-torch.arange(-2, 3, device=dev, dtype=torch.float32) ** 2 / 2)
        kern = kern / kern.sum()
        vol = torch.cat([(x * m).reshape(-1, 1, px, py, pz), m.reshape(-1, 1, px, py, pz)], 1)
        vol = vol.reshape(-1, 1, px, py, pz)
        for axis, size in zip((2, 3, 4), (px, py, pz)):
            if size < 2:
                continue
            shape = [1, 1, 1, 1, 1]
            shape[axis] = 5
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - axis)] = pad[2 * (4 - axis) + 1] = 2
            vol = F.conv3d(F.pad(vol, pad, mode="replicate"), kern.view(shape))
        vol = vol.reshape(-1, 2, P)
        x_blur = (vol[:, 0] / vol[:, 1].clamp_min(1e-3)).reshape(rows, k, P)
        # (3) Rician-like noise on the magnitude above the row minimum
        s = 0.1 * std
        x_noise = ((x - lo + torch.randn_like(x) * s) ** 2 + (torch.randn_like(x) * s) ** 2).sqrt() + lo
        variants = [(1, x_bias), (2, x_blur), (3, x_noise)]
        if self.corrupt_kinds >= 5:
            # (4) gamma / contrast change on the patch's in-brain range: g = exp(+-U[0.2, 0.5])
            big = torch.where(m > 0, x, -math.inf).amax(-1, keepdim=True)
            small = torch.where(m > 0, x, math.inf).amin(-1, keepdim=True)
            span = (big - small).clamp_min(1e-6)
            span = torch.where(torch.isfinite(span), span, torch.ones_like(span))
            small = torch.where(torch.isfinite(small), small, torch.zeros_like(small))
            sign = torch.where(torch.rand(rows, k, 1, device=dev) < 0.5, -1.0, 1.0)
            g = torch.exp(sign * (0.2 + 0.3 * torch.rand(rows, k, 1, device=dev)))
            x_gamma = small + span * ((x - small) / span).clamp(0, 1) ** g
            # (5) 2x downsample-upsample: masked 2-voxel block mean held over its block (axes of
            # fewer than half the nominal patch_size voxels, i.e. coarser than ~2 mm, keep full
            # resolution: 4 voxels at patch_size 8, 2 at patch_size 4)
            min_vox = max(2, self.patch_size[0] // 2)
            ks = tuple(2 if p >= min_vox else 1 for p in (px, py, pz))
            vx = torch.cat([(x * m).reshape(-1, 1, px, py, pz), m.reshape(-1, 1, px, py, pz)], 1)
            low = F.avg_pool3d(vx, kernel_size=ks, stride=ks, ceil_mode=True)
            for dim, kk, p in zip((2, 3, 4), ks, (px, py, pz)):
                low = low.repeat_interleave(kk, dim=dim).narrow(dim, 0, p)
            low = low.reshape(rows, k, 2, P)
            x_down = low[:, :, 0] / low[:, :, 1].clamp_min(1e-3)
            variants += [(4, x_gamma), (5, x_down)]
        if self.corrupt_kinds >= 7:
            # (6) linear contrast change about the patch's in-brain mean: c = exp(+-U[0.2, 0.5]),
            # floored at the row minimum (the magnitude floor)
            sign = torch.where(torch.rand(rows, k, 1, device=dev) < 0.5, -1.0, 1.0)
            c = torch.exp(sign * (0.2 + 0.3 * torch.rand(rows, k, 1, device=dev)))
            x_contrast = torch.maximum(mean + c * (x - mean), lo)
            # (7) motion ghost: + a x the magnitude (above the row minimum) circularly shifted by half
            # the patch along one random axis of >= 2 voxels, renormalized by 1 + a x shifted mask
            sizes = (px, py, pz)
            elig = [d for d, p in enumerate(sizes) if p >= 2]
            a = 0.2 + 0.3 * torch.rand(rows, k, 1, device=dev)
            pick = torch.randint(0, max(1, len(elig)), (rows, k, 1), device=dev)
            xm = ((x - lo) * m).reshape(rows, k, px, py, pz)
            mm = m.reshape(rows, k, px, py, pz)
            gx, gm = torch.zeros_like(x), torch.zeros_like(m)
            for j, d in enumerate(elig):
                sx = xm.roll(sizes[d] // 2, dims=2 + d).reshape(rows, k, P)
                sm = mm.roll(sizes[d] // 2, dims=2 + d).reshape(rows, k, P)
                gx = torch.where(pick == j, sx, gx)
                gm = torch.where(pick == j, sm, gm)
            x_ghost = (x - lo + a * gx) / (1 + a * gm) + lo
            if not elig:
                x_ghost = x
            variants += [(6, x_contrast), (7, x_ghost)]
        out = x
        for c, xc in variants:
            out = torch.where((kind == c)[..., None], xc, out)
        out = (out * m).to(vis_patches.dtype)
        return vis_patches.scatter(1, idx, torch.where(hit[..., None], out, vis_patches.gather(1, idx)))

    @torch.no_grad()
    def corrupt_apply_batched(self, items: list[tuple], compact: bool = False) -> list[Tensor]:
        """corrupt_apply over every spacing group at once (speed: ~200 kernel launches in total instead of
        ~150 per group; the per-group python loop was ~half the main thread's launches). items: one
        (vis_patches, vis_mask, patch, cand, kind) per group, as corrupt_apply takes them; returns the
        patches per group. Same per-candidate math as corrupt_apply on one flat buffer of every voxel of
        every candidate token (ragged over the groups' patch shapes); the random draws are laid out per
        candidate instead of per group (same distributions), and the bf16 autocast rounding of the
        matmul / conv steps is reproduced explicitly."""
        dev = items[0][0].device
        ac = torch.is_autocast_enabled(dev.type)
        cdt = torch.get_autocast_dtype(dev.type) if ac else torch.float32

        def rnd(t: Tensor) -> Tensor:
            # the autocast compute dtype's rounding (identity without autocast)
            return t.to(cdt).float() if ac else t

        xs, ms, kinds, los, meta = [], [], [], [], []
        for vis, vmask, patch, cand, kind in items:
            rows, nv, P = vis.shape
            k = cand.shape[1]
            lo = torch.where(vmask.float() > 0, vis.float(), math.inf).amin(dim=(1, 2))
            los.append(lo[:, None].expand(rows, k).reshape(-1))
            idx = cand[..., None].expand(rows, k, P)
            xs.append(vis.gather(1, idx).float().reshape(-1))
            ms.append(vmask.gather(1, idx).float().reshape(-1))
            kinds.append(kind.reshape(-1))
            meta.append((rows * k, P, *patch))
        X, M, kind, lo = (torch.cat(t) if len(t) > 1 else t[0] for t in (xs, ms, kinds, los))
        Mfull = M
        lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
        # per candidate (P, px, py, pz) and the 2x downsample block grid, built on the cpu (known sizes)
        min_vox = max(2, self.patch_size[0] // 2)
        cpu = []
        for n, P, px, py, pz in meta:
            ks = [2 if q >= min_vox else 1 for q in (px, py, pz)]
            nb = [-(-q // kk) for q, kk in zip((px, py, pz), ks)]
            cpu.append([n, P, px, py, pz, *ks, *nb, nb[0] * nb[1] * nb[2]])
        C, V = sum(r[0] for r in cpu), sum(r[0] * r[1] for r in cpu)
        Btot = sum(r[0] * r[11] for r in cpu)
        # the per group table to the gpu, expanded there (cpu repeat_interleave was ~10 ms)
        tab = torch.tensor(cpu, dtype=torch.int64)
        tab = (tab.pin_memory() if dev.type == "cuda" else tab).to(dev, non_blocking=True)
        per = torch.repeat_interleave(tab[:, 1:], tab[:, 0], dim=0, output_size=C)
        i32 = torch.int32
        # random draws per candidate (all C, the same draws with or without compact)
        u = torch.randn(C, 3, device=dev)
        u = rnd(u / u.abs().sum(-1, keepdim=True).clamp_min(1e-6)).t().contiguous()
        amp = 0.2 + 0.1 * torch.rand(C, device=dev)
        draws = {"amp": amp}
        if self.corrupt_kinds >= 5:
            draws["g"] = torch.exp(torch.where(torch.rand(C, device=dev) < 0.5, -1.0, 1.0) * (0.2 + 0.3 * torch.rand(C, device=dev)))
        if self.corrupt_kinds >= 7:
            draws["c"] = torch.exp(torch.where(torch.rand(C, device=dev) < 0.5, -1.0, 1.0) * (0.2 + 0.3 * torch.rand(C, device=dev)))
            draws["a"] = 0.2 + 0.3 * torch.rand(C, device=dev)
            draws["pick"] = torch.rand(C, device=dev)
        Xfull = X
        if compact:
            # corrupt_compact (prep threads, where a sync only waits for the prep stream): just the hit candidates,
            # sorted by kind, so each corruption runs on its own candidates' voxels only (~1/10 of the work)
            order = torch.argsort(kind, stable=True)
            nk = self.corrupt_kinds
            counts = torch.bincount(kind, minlength=nk + 1)
            vox = torch.zeros(nk + 1, device=dev, dtype=torch.int64).index_add_(0, kind, per[:, 0])
            counts, vox = torch.stack([counts, vox]).tolist()
            sel = order[counts[0]:]
            start_full = (per[:, 0].cumsum(0) - per[:, 0])[sel]
            per, lo, kind, u = per[sel], lo[sel], kind[sel], u[:, sel]
            draws = {k_: v_[sel] for k_, v_ in draws.items()}
            C, V = sum(counts[1:]), sum(vox[1:])
            if C == 0:
                return [vis for vis, *_ in items]
            Btot = int(per[:, 10].sum())
        # per candidate columns (int32, contiguous): voxels, patch dims, downsample factors, block grid
        Pc, p0, p1, p2, k0, k1, k2, n0, n1, n2, nblk = per.t().to(i32).contiguous().unbind(0)
        seg = torch.repeat_interleave(torch.arange(C, device=dev, dtype=i32), per[:, 0], output_size=V)
        flat = torch.arange(V, device=dev, dtype=i32)
        q = flat - (Pc.cumsum(0, dtype=i32) - Pc)[seg]
        if compact:
            vsel = start_full[seg] + q
            X, M = Xfull[vsel], M[vsel]
        P0, P1, P2 = p0[seg], p1[seg], p2[seg]
        S0 = P1 * P2
        c0 = q // S0
        rem = q - c0 * S0
        c1 = rem // P2
        c2 = rem - c1 * P2
        coord, dims, strides = (c0, c1, c2), (P0, P1, P2), (S0, P2, None)
        LO = lo[seg]
        XM = X * M

        def seg_sum(v: Tensor, sg: Tensor) -> Tensor:
            return torch.zeros(C, device=dev, dtype=v.dtype).index_add_(0, sg, v)

        memo = {}

        def stats(sl):
            # in-mask count, mean and std of the candidates of voxel slice sl
            if sl == slice(None) and "all" in memo:
                return memo["all"]
            sg, x, m = seg[sl], X[sl], M[sl]
            cnt = seg_sum(m, sg).clamp_min(1.0)
            mean = seg_sum(x * m, sg) / cnt
            res = mean, (seg_sum((x - mean[sg]) ** 2 * m, sg) / cnt).sqrt()
            if sl == slice(None):
                memo["all"] = res
            return res

        def bias(sl):
            # (1) bias ramp along a random direction (u @ r.T in the autocast dtype, as corrupt_apply)
            sg = seg[sl]
            ur = None
            for ax in range(3):
                r = rnd((coord[ax][sl].float() + 0.5) / dims[ax][sl].float() * 2 - 1)
                term = u[ax][sg] * r
                ur = term if ur is None else ur + term
            return (X[sl] - LO[sl]) * (1 + draws["amp"][sg] * rnd(ur)) + LO[sl]

        kern = torch.exp(-torch.arange(-2, 3, dtype=torch.float32) ** 2 / 2)
        kern = rnd(kern / kern.sum()).tolist()

        def blur(sl):
            # (2) separable gaussian blur, sigma 1 voxel, replicate padded, normalized inside the mask; the
            # conv's autocast rounding per pass; axes of < 2 voxels skipped
            vw = torch.stack([rnd(XM[sl]), rnd(M[sl])], 1)
            loc = torch.arange(vw.shape[0], device=dev, dtype=i32)
            for ax in range(3):
                ca, pa, sa = coord[ax][sl], dims[ax][sl], strides[ax]
                sa = None if sa is None else sa[sl]
                acc = None
                for t in range(-2, 3):
                    d = torch.minimum((ca + t).clamp_min(0), pa - 1) - ca
                    term = kern[t + 2] * vw.index_select(0, loc + (d * sa if sa is not None else d))
                    acc = term if acc is None else acc + term
                vw = torch.where((pa >= 2)[:, None], rnd(acc), vw)
            return (vw[:, 0].to(cdt) / vw[:, 1].to(cdt).clamp_min(1e-3)).float()

        def noise(sl):
            # (3) Rician-like noise on the magnitude above the row minimum
            x, lo_ = X[sl], LO[sl]
            s_ = (0.1 * stats(sl)[1])[seg[sl]]
            return ((x - lo_ + torch.randn_like(x) * s_) ** 2 + (torch.randn_like(x) * s_) ** 2).sqrt() + lo_

        def gamma(sl):
            # (4) gamma on the patch's in-brain range
            sg, x = seg[sl], X[sl]
            inb = M[sl] > 0
            big = torch.full((C,), -math.inf, device=dev).scatter_reduce_(0, sg.long(), torch.where(inb, x, -math.inf), "amax")
            small = torch.full((C,), math.inf, device=dev).scatter_reduce_(0, sg.long(), torch.where(inb, x, math.inf), "amin")
            span = (big - small).clamp_min(1e-6)
            span = torch.where(torch.isfinite(span), span, torch.ones_like(span))
            small = torch.where(torch.isfinite(small), small, torch.zeros_like(small))
            sm_, sp_ = small[sg], span[sg]
            return sm_ + sp_ * ((x - sm_) / sp_).clamp(0, 1) ** draws["g"][sg]

        def down(sl):
            # (5) 2x downsample-upsample: masked block mean (avg_pool3d, ceil_mode) held over its block
            sg = seg[sl]
            c0_, c1_, c2_ = (c[sl] for c in coord)
            bid = (nblk.cumsum(0, dtype=i32) - nblk)[sg] + ((c0_ // k0[sg]) * n1[sg] + c1_ // k1[sg]) * n2[sg] + c2_ // k2[sg]
            m = M[sl]
            sums = torch.zeros(Btot, 3, device=dev).index_add_(0, bid, torch.stack([XM[sl], m, torch.ones_like(m)], 1))
            low = (sums[:, :2] / sums[:, 2:]).index_select(0, bid)
            return low[:, 0] / low[:, 1].clamp_min(1e-3)

        def contrast(sl):
            # (6) linear contrast about the patch's in-brain mean, floored at the row minimum
            sg = seg[sl]
            mean = stats(sl)[0][sg]
            return torch.maximum(mean + draws["c"][sg] * (X[sl] - mean), LO[sl])

        def ghost(sl):
            # (7) motion ghost along one random axis of >= 2 voxels (circular half-patch shift)
            elig = torch.stack([p0, p1, p2], 1) >= 2
            ne = elig.sum(1)
            pick = torch.minimum((draws["pick"] * ne.clamp_min(1)).long(), (ne - 1).clamp_min(0))
            sg = seg[sl]
            axis = (elig & (elig.long().cumsum(1) == (pick + 1)[:, None])).long().argmax(1)[sg]
            src = torch.arange(axis.shape[0], device=dev, dtype=i32)
            for ax in range(3):
                ca, pa = coord[ax][sl], dims[ax][sl]
                d = (ca - pa // 2) % pa - ca
                src = torch.where(axis == ax, src + (d * strides[ax][sl] if strides[ax] is not None else d), src)
            x, lo_, m = X[sl], LO[sl], M[sl]
            XL = (x - lo_) * m
            av = draws["a"][sg]
            out = (x - lo_ + av * XL[src]) / (1 + av * m[src]) + lo_
            return torch.where(ne[sg] > 0, out, x)

        fns = [bias, blur, noise, gamma, down, contrast, ghost][: self.corrupt_kinds]
        if compact:
            out = X.clone()
            v0 = 0
            for cls, fn in enumerate(fns, start=1):
                n_ = vox[cls]
                if n_:
                    out[v0 : v0 + n_] = fn(slice(v0, v0 + n_))
                v0 += n_
            out = Xfull.clone().index_copy_(0, vsel, out)
            M = Mfull
        else:
            kv = kind[seg]
            out = X
            for cls, fn in enumerate(fns, start=1):
                out = torch.where(kv == cls, fn(slice(None)), out)
        out = out * M
        res, off = [], 0
        for (vis, vmask, patch, cand, kd), (n, P, *_) in zip(items, meta):
            rows, nv, _ = vis.shape
            k = cand.shape[1]
            o = out[off : off + n * P].view(rows, k, P).to(vis.dtype)
            off += n * P
            idx = cand[..., None].expand(rows, k, P)
            res.append(vis.scatter(1, idx, torch.where((kd > 0)[..., None], o, vis.gather(1, idx))))
        return res

    @torch.no_grad()
    def unified_transform(
        self, vis_patches: Tensor, vis_mask: Tensor, patch: tuple[int, int, int], order: Tensor,
        vols: Tensor, b: int, num: int, num_samples: int, num_visible: int, defer: list | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Unified corruption: a unified_ratio share of the visible (>= 50% brain) tokens, class
        uniform among swap / corruptions / (shuffle). Each token carries one label; a class that
        is ineligible for its token (no brain partner / donor, a single-image group) leaves it
        clean. Shuffle donors are taken from the clean patches first, then the cross-subject swap,
        then the corruptions. Returns (patches, mask, labels [rows, V] long)."""
        rows, nv, _ = vis_patches.shape
        dev = vis_patches.device
        ok = vis_mask.float().mean(-1) >= 0.5
        cand, hit = self.corrupt_candidates(ok, self.unified_ratio)
        cls = torch.where(hit, torch.randint(1, self.unified_classes, hit.shape, device=dev), 0)
        labels = torch.zeros(rows, nv, dtype=torch.long, device=dev).scatter(1, cand, cls)
        if self.unified_shuffle:
            shuf = self.unified_classes - 1
            vis_patches, vis_mask, done = self.shuffle_replace(vis_patches, vis_mask, select=labels == shuf)
            labels = torch.where((labels == shuf) & ~done, 0, labels)
        if b >= 2:
            vis_patches, vis_mask, done = self.rtd_replace(
                vis_patches, vis_mask, order, vols, b, num, num_samples, num_visible, select=labels == 1
            )
            labels = torch.where((labels == 1) & ~done, 0, labels)
        else:
            labels = torch.where(labels == 1, 0, labels)
        kind = torch.where((cls >= 2) & (cls < 2 + self.corrupt_kinds), cls - 1, 0)
        if defer is not None:
            # batched_corrupt: corrupt_apply_batched applies it later, all groups at once
            defer.append((vis_patches, vis_mask, patch, cand, kind))
            return vis_patches, vis_mask, labels
        vis_patches = self.corrupt_apply(vis_patches, vis_mask, patch, cand, kind)
        return vis_patches, vis_mask, labels

    def edgelvl_terms(
        self, tokens: Tensor, patches: Tensor, mask: Tensor, patch, spacing
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Edge level-set head: level_terms on the gradient magnitude (central differences per
        mm, only where both neighbours are in the mask), thresholds edgelvl_thresholds x the
        row's in-brain median gradient. Returns (weighted bce sum, normalizer, weighted abs err)."""
        rows, nv = patches.shape[:2]
        sub = self.patch_size[0] // self.level_pool
        weight, target = edge_targets(
            patches, mask, self.edgelvl_thresholds, tuple(patch), tuple(float(v) for v in spacing), rows, nv, sub
        )
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
    def shuffle_replace(
        self, vis_patches: Tensor, vis_mask: Tensor, select: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replace a shuffle_ratio share of every row's visible patches by another visible patch
        of the same row (a random within-row permutation gives the donors). Host and donor must
        both be mostly brain (>= half their voxels in the mask), so the cue is anatomical
        inconsistency, not background. select [rows, V] bool (unified corruption) replaces the
        shuffle_ratio draw. Returns (patches, mask, replaced [rows, V] bool)."""
        rows, num_vis = vis_patches.shape[:2]
        device = vis_patches.device
        perm = torch.rand(rows, num_vis, device=device).argsort(dim=1)
        brain = vis_mask.float().mean(-1) >= 0.5
        donor_brain = brain.gather(1, perm)
        not_self = perm != torch.arange(num_vis, device=device)[None]
        if select is None:
            select = torch.rand(rows, num_vis, device=device) < self.shuffle_ratio
        replaced = select & brain
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
        num_samples: int, num_visible: int, select: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Swap a rtd_ratio share of the visible patches of every row for the patch at the same
        token coordinate of the next image of the group (rolled by one), quantile matched to the
        host row's in-brain voxels. select [rows, V] bool (unified corruption) replaces the
        rtd_ratio draw. Returns (patches, mask, replaced [rows, num_visible] bool)."""
        rows = b * num_samples
        vis_order = order[:, :num_visible]  # [rows, V], token ids of the host
        partner_img = (torch.arange(b, device=order.device) + 1) % b
        partner_img = partner_img.repeat_interleave(num_samples)  # [rows]
        flat = vis_order + (partner_img * num)[:, None]
        p_patches = vols[0].flatten(0, 1).index_select(0, flat.flatten()).view_as(vis_patches)
        p_mask = vols[1].flatten(0, 1).index_select(0, flat.flatten()).view_as(vis_patches)
        # the partner patch must be brain tissue too (half its voxels in its brain mask)
        ok = p_mask.float().mean(-1) >= 0.5
        if select is None:
            select = torch.rand(vis_order.shape, device=order.device) < self.rtd_ratio
        replaced = select & ok
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

        if getattr(self, "bf16_residual", False) and self.training:
            # bf16_residual (speed, CHANGES NUMERICS: the residual stream in bf16, not fp32; training only)
            x = x.to(torch.bfloat16)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def encode_tokens_fused(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
        pos_hide: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # encode_tokens (same final output) plus the multi-level skip for the decoder input:
        # fuse_scale * sum_l proj_l(norm_l(h_l)) over the fuse_layers levels (training only)
        B = x.shape[0]
        if pos_hide is not None:
            x = torch.where(pos_hide[..., None], x + self.brainpos_unk.to(x.dtype), self.pos_embed(x, coord))
        else:
            x = self.pos_embed(x, coord)

        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        index = {l: jj for jj, l in enumerate(self.fuse_layers)}
        skip = None
        if 0 in index:
            skip = self.fuse_projs[index[0]](self.fuse_norms[index[0]](x))
        for ii, block in enumerate(self.blocks):
            x = block(x)
            jj = index.get(ii + 1)
            if jj is not None:
                term = self.fuse_projs[jj](self.fuse_norms[jj](x))
                skip = term if skip is None else skip + term
        x = self.norm(x)
        return x, self.fuse_scale * skip

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
        skip: Float[Tensor, "B R+N E"] | None = None,
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
            if skip is not None:
                skip = skip[:, : self.class_tokens + keep]
        embeds = self.decoder_proj(embeds)
        if skip is not None:
            # multi-level fusion (fuse_layers): the shallow-level skip, decoder width, per encoder token
            embeds = embeds + skip

        mask_tokens = self.mask_token.expand(B, M, -1)
        mask_tokens = self.decoder_pos_embed(mask_tokens, target_coord)
        if self.global_proj is not None:
            mask_tokens = mask_tokens + pooled

        x = torch.cat([embeds, mask_tokens], dim=1)
        if getattr(self, "bf16_residual", False) and self.training:
            x = x.to(torch.bfloat16)
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
        prepared: dict | None = None,
    ):
        if isinstance(images, list):
            # native resolution batch, images on their own voxel grids
            return self.forward_native(
                images, mask, spacing, num_visible, num_predict, num_samples, with_state,
                fino_meta=fino_meta, fino_gamma=fino_gamma, subject_ids=subject_ids, prepared=prepared,
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

    def level_targets(self, patches: Tensor, mask: Tensor, patch) -> tuple[Tensor, Tensor]:
        # level_terms' (weight [R, 1, s, s, s], target [R, T, s, s, s]) of one group, R = rows x tokens
        px, py, pz = patch
        x = patches.float().reshape(-1, 1, px, py, pz)
        m = mask.float().reshape(-1, 1, px, py, pz)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        sub = self.patch_size[0] // self.level_pool
        # one pass of sep_pool over (mask, levels) (adaptive_avg_pool3d's bins as 3 small fp32 matmuls)
        pooled = sep_pool(torch.cat([m, above], 1), sub)
        weight = pooled[:, :1]
        return weight, pooled[:, 1:] / weight.clamp_min(1e-6)

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

    @staticmethod
    def _to_cells8(t: Tensor, patch) -> Tensor:
        # native token voxels [R, px*py*pz] -> [R, 1, 8, 8, 8] (nearest; identity at 1 mm), so 2 mm
        # sub-blocks are 2^3 voxel blocks
        if tuple(patch) != (8, 8, 8):
            # F.interpolate's nearest (src = floor(dst * in / out)) as one gather: upsample_nearest3d
            # with ~1e5 batch rows is ~30x slower
            t = t.reshape(-1, math.prod(patch)).index_select(1, cells8_index(tuple(patch), t.device))
        return t.reshape(-1, 1, 8, 8, 8)

    def mirror_ref(self, vols: Tensor, patch, box, grid, sp) -> Tensor:
        # per token [b, N, 64]: the max within +-1 cell (2 mm cells) of each sub-block's x-mirrored cell
        # (mirror about the box centre, the fit_to_shape centre); 1e4 where that cell is out of the
        # mask, out of the box, or within 4 mm of the midline. at 1 mm = r18's mirror_map
        img, msk = vols[0], vols[1]
        b, N = img.shape[:2]
        gx, gy, gz = grid

        def cellvol(t: Tensor) -> Tensor:
            c = self._to_cells8(t, patch).view(b, N, 4, 2, 4, 2, 4, 2).amax((3, 5, 7))
            c = c.view(b, gx, gy, gz, 4, 4, 4).permute(0, 1, 4, 2, 5, 3, 6)
            return c.reshape(b, gx * 4, gy * 4, gz * 4)

        cells = cellvol(img)
        valid = cellvol(msk.to(img.dtype)) > 0
        hmax = F.max_pool3d(cells.float()[:, None], 3, 1, 1)[:, 0]
        # x mirror index per cell, in voxels of this spacing: cell width px / 4
        Cx, w = gx * 4, patch[0] / 4
        centre = (torch.arange(Cx, device=img.device, dtype=torch.float64) + 0.5) * w
        j = torch.floor((box[0] - centre) / w).long()
        ok = (j >= 0) & (j < Cx) & ((centre - box[0] / 2).abs() * sp[0] >= 4.0)
        j = j.clamp(0, Cx - 1)
        mvalid = valid.index_select(1, j) & ok.view(1, -1, 1, 1)
        ref = torch.where(mvalid, hmax.index_select(1, j), 1e4)
        ref = ref.view(b, gx, 4, gy, 4, gz, 4).permute(0, 1, 3, 5, 2, 4, 6)
        return ref.reshape(b, N, 64)

    def mirror_terms(self, tokens: Tensor, patches: Tensor, mask: Tensor, patch, ref: Tensor, clean: Tensor | None):
        # mirror level channels: (weighted bce sum, valid weight sum, level weight sum, weighted positives [M]).
        # clean [rows, V] bool: tokens a corruption altered are out of the loss (ms-syn; None = all in)
        x = self._to_cells8(patches.float(), patch)
        m = self._to_cells8(mask.float(), patch)
        weight = F.avg_pool3d(m, 2)
        mh = ref.float().view(-1, 1, 4, 4, 4)
        thr = torch.maximum(self.level_mirror_thresholds.view(1, -1, 1, 1, 1), mh + self.level_mirror_margin)
        thr = thr.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
        target = F.avg_pool3d((x > thr).float() * m, 2) / weight.clamp_min(1e-6)
        m_weight = weight * (mh < 1e3).float()
        if clean is not None:
            m_weight = m_weight * clean.reshape(-1, 1, 1, 1, 1).float()
        logits = self.level_mirror_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (m_weight * bce).sum(), m_weight.sum(), weight.sum(), (m_weight * target).sum((0, 2, 3, 4))

    def lesion_targets(self, lesion: Tensor | None, mask: Tensor, patch, pool=None) -> tuple[Tensor, Tensor]:
        # level_terms' sub-blocks of the visible tokens: in-mask fraction, lesion fraction of it.
        # pool: the adaptive average pool (default F.adaptive_avg_pool3d; batched_level: sep_pool)
        pool = pool or F.adaptive_avg_pool3d
        px, py, pz = patch
        m = mask.float().reshape(-1, 1, px, py, pz)
        sub = self.patch_size[0] // self.level_pool
        weight = pool(m, sub).flatten(1)
        if lesion is None:
            return weight, torch.zeros_like(weight)
        target = pool(lesion.float().reshape(m.shape) * m, sub).flatten(1)
        return weight, target / weight.clamp_min(1e-6)

    @staticmethod
    def altered_tokens(part: dict) -> Tensor | None:
        # [rows, V] bool: visible tokens a corruption replaced or altered (ms-syn: out of the lesion and
        # mirror level losses, as r24 excludes rtd-replaced tokens); None when nothing corrupts
        out = None
        for key, fn in (
            ("rtd_label", lambda t: t), ("unified_label", lambda t: t > 0), ("hemi_label", lambda t: t),
            ("corrupt_label", lambda t: t > 0), ("orient_label", lambda t: t > 0), ("shuffled", lambda t: t),
        ):
            if key in part:
                hit = fn(part[key]).bool()
                out = hit if out is None else out | hit
        return out

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
        prepared: dict | None = None,
    ):
        # images (float) and masks (bool) on their own voxel grids (sample_transform native), fit
        # here to the grid_size mm box in their voxels. images with the same spacing share one
        # token grid, sequence sampling and embedding / head weight. each predicted token counts
        # by its in-mask fraction whatever its voxel count; with 1 mm images only this is
        # forward's loss, bitwise
        assert self.native and num_predict is not None
        # prepared (native_prep_tokens): native_prepare's weight-free tokenization, done ahead in a prep
        # thread; else here, between the weights below and the embeddings (the same ops in the same order)
        if prepared is None:
            groups, geoms = self.native_groups(spacing, images[0].device)
        else:
            groups, geoms = prepared["groups"], prepared["geoms"]
        # embedding and head weights of all non 1 mm spacings in one pass each
        native_sps = [sp for sp in groups if sp != (1.0, 1.0, 1.0)]
        embed_ws, head_ws = {}, {}
        if native_sps:
            sp_patches = [geoms[sp][0] for sp in native_sps]
            embed_ws = self.patch_embed.native_weights(native_sps, sp_patches)
            head_ws = self.decoder_head.native_weights(native_sps, sp_patches)
            embed_ws, head_ws = dict(zip(native_sps, embed_ws)), dict(zip(native_sps, head_ws))

        if prepared is None:
            parts = self.native_tokens(images, mask, groups, geoms, num_visible, num_predict, num_samples)
        else:
            parts = prepared["parts"]
        device = parts[0]["vis_coord"].device
        for part in parts:
            sp = part["spacing"]
            part["embeds"] = self.patch_embed.native(part.pop("vis_tok"), sp, part["patch"], embed_ws.get(sp))

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
        fuse_skip = None
        if self.fuse_layers and self.training:
            embeds, fuse_skip = self.encode_tokens_fused(
                cat("embeds"), cat("vis_coord"), cat("bp_hide") if bp_on else None
            )
        else:
            embeds = self.encode_tokens(cat("embeds"), cat("vis_coord"), cat("bp_hide") if bp_on else None)
        rtd_on = self.rtd_head is not None and self.training and "rtd_label" in parts[0]
        rtd_logits, vis_gate = None, None
        if rtd_on:
            rtd_logits = self.rtd_head(embeds[:, self.class_tokens :]).squeeze(-1).float()
            if self.rtd_gate_scale > 0.0:
                vis_gate = (2.0 * torch.sigmoid(-self.rtd_gate_scale * rtd_logits)).clamp(max=1.0)
                vis_gate = vis_gate.to(embeds.dtype)
        if fuse_skip is None:
            feats = self.decode_features(embeds, cat("target_coord"), vis_gate)
        else:
            feats = self.decode_features(embeds, cat("target_coord"), vis_gate, fuse_skip)
        feats = torch.split(feats, [len(part["ids"]) * num_samples for part in parts])
        # graft: the v3 loss stack, extended to per-spacing native patches. norm_target
        # z-scores each token's voxels (spacing groups differ in patch size), the fg mask
        # weights the mse per voxel, and the xyz gradient penalty runs per group on its own
        # patch geometry; the global loss is the weight-masked mean across groups
        losses, wts = [], []
        grad_num, grad_den = [], []
        if "flat_w" in parts[0]:
            # flat_loss: the same elementwise (w * mse) and sums as below, on the flat group-order buffers
            preds = []
            for part, feat in zip(parts, feats):
                sp = part["spacing"]
                part["pred_patches"] = self.decoder_head.native(feat, sp, part["patch"], head_ws.get(sp))
                preds.append(part["pred_patches"].flatten())
            w = parts[0].pop("flat_w")
            losses.append(w * F.mse_loss(flat_rows(preds), parts[0].pop("flat_t"), reduction="none"))
            wts.append(w)
            feats = ()
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
                # finite differences of the prediction vs of the target (the non-native reference);
                # was pv[:, 1:] (the raw prediction) against the target difference
                dx = F.mse_loss(pv[:, 1:] - pv[:, :-1], tv[:, 1:] - tv[:, :-1], reduction="none")
                dy = F.mse_loss(pv[:, :, 1:] - pv[:, :, :-1], tv[:, :, 1:] - tv[:, :, :-1], reduction="none")
                dz = F.mse_loss(pv[:, :, :, 1:] - pv[:, :, :, :-1], tv[:, :, :, 1:] - tv[:, :, :, :-1], reduction="none")
                mx = mv[:, 1:] * mv[:, :-1]
                my = mv[:, :, 1:] * mv[:, :, :-1]
                mz = mv[:, :, :, 1:] * mv[:, :, :, :-1]
                grad_num += [(mx * dx).sum(), (my * dy).sum(), (mz * dz).sum()]
                grad_den += [mx.sum(), my.sum(), mz.sum()]
        loss = flat_cat(losses).sum() / flat_cat(wts).sum()
        if self.grad_loss_weight > 0.0:
            # per axis over every spacing group (grad_num holds x, y, z per group; only the first
            # group's three terms were used before)
            loss = loss + self.grad_loss_weight * sum(
                sum(grad_num[a::3]) / sum(grad_den[a::3]).clamp_min(1.0) for a in range(3)
            ) / 3.0
        # an all 1 mm batch leaves the residuals without a gradient, and ddp needs one every step
        residuals = (self.patch_embed.residual, self.decoder_head.residual)
        residuals += (self.decoder_head.bias_residual,)
        loss = loss + 0.0 * sum(r.sum() for r in residuals)
        if self.level_head is not None:
            num, den = 0.0, 0.0
            rows = torch.split(embeds, [len(part["ids"]) * num_samples for part in parts])
            if "lvl_w" in parts[0]:
                # batched_level: every group's targets, one head pass (level_terms' sums over all rows)
                weight = flat_rows([part.pop("lvl_w") for part in parts])
                target = flat_rows([part.pop("lvl_t") for part in parts])
                logits = self.level_head(embeds[:, self.class_tokens :]).float().reshape(target.shape)
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                num, den = (weight * bce).sum(), weight.sum() * target.shape[1]
            else:
                for part, tokens in zip(parts, rows):
                    n, d = self.level_terms(
                        tokens[:, self.class_tokens :], part["vis_patches"], part["vis_mask"],
                        part["patch"],
                    )
                    num, den = num + n, den + d
            loss = loss + self.level_weight * num / den.clamp_min(1e-6)
            if self.level_mirror_head is not None:
                mterms = [
                    self.mirror_terms(
                        tokens[:, self.class_tokens :], part["vis_patches"], part["vis_mask"], part["patch"],
                        part.pop("mirror_ref"), part["syn_clean"],
                    )
                    for part, tokens in zip(parts, rows)
                ]
                # summed over groups: (weighted bce, valid weight, level weight, weighted positives [M])
                mb, mw, lw, mp = [sum(t[k] for t in mterms) for k in range(4)]
                T = len(self.level_thresholds)
                # same per-channel weight as the level channels (r18: / (valid weight * T))
                loss = loss + self.level_weight * mb / (mw * T).clamp_min(1e-6)
                M = len(self.level_mirror)
                self.level_mirror_stats = {
                    "bce": (mb / (mw * M).clamp_min(1e-6)).detach(),
                    "valid": (mw / lw.clamp_min(1e-6)).detach(),
                    **{f"pos{t:g}": v for t, v in zip(self.level_mirror, (mp / mw.clamp_min(1e-6)).detach())},
                }
        if self.lesion_head is not None and self.training:
            if "les_w" in parts[0]:
                weight = flat_rows([part.pop("les_w") for part in parts])
                target = flat_rows([part.pop("les_t") for part in parts])
            else:
                pairs = [self.lesion_targets(part.pop("lesion"), part.pop("lesion_mask"), part["patch"]) for part in parts]
                weight, target = [torch.cat(v) for v in zip(*pairs)]
            logits = self.lesion_head(embeds[:, self.class_tokens :]).float().reshape(target.shape)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            lw = weight.sum().clamp_min(1e-6)
            les_loss = (weight * bce).sum() / lw
            loss = loss + self.lesion_mask_weight * self.lesion_scale * les_loss
            self.level_mirror_stats = {
                **self.level_mirror_stats, "les_bce": les_loss.detach(),
                **({"les_pos": ((weight * target).sum() / lw).detach()} if getattr(self, "want_stats", True) else {}),
            }
        for part in parts:
            part.pop("syn_clean", None)
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

        aux_on = (
            self.brainpos_head, self.scanmatch_proj, self.hemiswap_head, self.corrupt_head, self.edgelvl_head,
            self.unified_head,
        )
        if self.training and any(h is not None for h in aux_on):
            loss = loss + self.aux_losses(parts, embeds, num_samples, subject_ids, device)
        for part in parts:
            part.pop("vis_patches", None)
            part.pop("vis_mask", None)

        if not with_state:
            return loss
        return loss, {"parts": parts, "embeds": embeds}

    def flat_loss_ok(self) -> bool:
        # flat_loss applies: plain masked mse (no target curriculum / normalization / fg or gradient terms)
        return bool(getattr(self, "flat_loss", False)) and self.training and not (
            self.target_lp_mm > 0.0 or self.norm_target or self.fg_loss_weight > 0.0 or self.grad_loss_weight > 0.0
        )

    def native_groups(self, spacing: list, device: torch.device) -> tuple[dict, dict]:
        # images by spacing (insertion order) and each spacing's token geometry
        token_mm = self.patch_size[0] * self.target_spacing
        groups: dict[tuple, list[int]] = {}
        for ii, sp in enumerate(spacing):
            groups.setdefault(tuple(sp), []).append(ii)
        geoms = {sp: native_geometry(sp, tuple(self.grid_size), token_mm, device) for sp in groups}
        return groups, geoms

    @torch.no_grad()
    def native_prepare(
        self, images: list[Tensor], mask: list[Tensor], spacing: list, num_visible: int, num_predict: int,
        num_samples: int, lims: list | None = None,
    ) -> dict:
        # native_prep_tokens: forward_native's weight-free tokenization of a batch (run in a prep thread,
        # under the training autocast), passed back as forward(prepared=...). lims: per image in-mask
        # (min, max) if known (native_worker_clip), else plan_tumors computes them
        groups, geoms = self.native_groups(spacing, images[0].device)
        parts = self.native_tokens(images, mask, groups, geoms, num_visible, num_predict, num_samples, lims=lims)
        return {"groups": groups, "geoms": geoms, "parts": parts}

    def native_tokens(
        self, images: list[Tensor], mask: list[Tensor], groups: dict, geoms: dict, num_visible: int,
        num_predict: int, num_samples: int, lims: list | None = None,
    ) -> list[dict]:
        # forward_native's tokenization (no weights): tumour paste, patchify, sequences, token corruptions,
        # per group; each part keeps its visible patches as "vis_tok" for the patch embedding
        seq_length = num_visible + num_predict
        device = images[0].device
        tumors = {}
        if self.training and self.tumor_prob * self.tumor_scale > 0:
            tumors = plan_tumors(
                images, mask, groups, geoms, self.tumor_prob * self.tumor_scale, self.tumor_radius, self.tumor_ring,
                self.tumor_contrast, self.tumor_edge, self.tumor_nodule, neg=self.tumor_neg_prob, lims=lims,
            )
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
            # ms-syn: blobs / tumours into the clean volumes before patchify (so before every token
            # corruption): they enter the inputs and every target, shared by an image's sequences
            lesion = None
            if self.training and self.blob_prob > 0:
                lesion = insert_blobs(
                    vols[0], vols[1], patch, sp, self.blob_prob, self.blob_max, self.blob_radius,
                    self.blob_contrast, self.blob_edge, with_mask=self.lesion_head is not None,
                )
            clean_img = None
            if sp in tumors and self.tumor_one_view:
                clean_img = vols[0].clone()
            if sp in tumors:
                tumor = paste_tumors(vols[0], vols[1], tumors[sp], with_mask=self.lesion_head is not None)
                if tumor is not None:
                    lesion = tumor if lesion is None else torch.maximum(lesion, tumor)
            raw = vols if (self.hemiswap_head is not None and self.training) else None
            cube = (grid, self.mask_cube) if self.mask_cube else None
            # lazy_patchify (speed): the token rows are gathered from the padded volumes on demand (only the
            # ~1.3k sampled of ~10k+ tokens per image are ever read), the same values as patchify3d's rows
            lazy = getattr(self, "lazy_patchify", False) and self.level_mirror_head is None and (
                self.brainpos_head is None and self.hemiswap_head is None  # brain_frame reads whole patch rows
            )
            patchify = (lambda t: LazyTokens(t, patch, grid)) if lazy else (lambda t: patchify3d(t, patch))
            mask_frac = None
            if lazy:
                vols = LazyTokens(vols, patch, grid)
                mask_frac = vols[1].frac()
                scores = sequence_scores(None, num_samples, self.min_mask_frac, cube=cube, mask_frac=mask_frac)
            else:
                vols = patchify3d(vols.flatten(0, 1), patch).unflatten(0, (2, len(ids)))
                scores = sequence_scores(vols[1], num_samples, self.min_mask_frac, cube=cube)
            scores = scores.flatten(0, 1)
            parts.append({"spacing": sp, "patch": patch, "ids": ids, "vols": vols, "grid": grid})
            parts[-1]["scores"] = scores
            if mask_frac is not None:
                parts[-1]["mask_frac"] = mask_frac
            if raw is not None:
                parts[-1]["raw"] = raw
            if self.lesion_head is not None and self.training:
                parts[-1]["lesion"] = None if lesion is None else patchify(lesion)
            if clean_img is not None:
                parts[-1]["clean_img"] = patchify(clean_img)

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
        order_all = scores.argsort(dim=1, stable=True)[:, :seq_length]
        orders = order_all.split(num_rows)
        uni = None
        if (
            getattr(self, "batched_unified", False) and getattr(self, "batched_corrupt", False)
            and self.unified_head is not None and self.training
            and not self.unified_shuffle and self.shuffle_head is None and self.rtd_head is None
            and all("mask_frac" in part for part in parts)
        ):
            # batched_unified (speed): unified_transform's candidate / class draws for every group's rows at once
            # (the eligibility from the lazy mask fractions = the visible mask rows' means); rtd swaps stay per group
            frac = parts[0]["mask_frac"].new_zeros((sum(len(part["ids"]) for part in parts), max(num_tokens)))
            img = 0
            for part, num in zip(parts, num_tokens):
                frac[img : img + len(part["ids"]), :num] = part.pop("mask_frac")
                img += len(part["ids"])
            vis_order = order_all[:, :num_visible]
            row_img = torch.arange(order_all.shape[0], device=device) // num_samples
            ok = frac[row_img].gather(1, vis_order) >= 0.5
            cand, hit = self.corrupt_candidates(ok, self.unified_ratio)
            cls = torch.where(hit, torch.randint(1, self.unified_classes, hit.shape, device=device), 0)
            labels = torch.zeros_like(vis_order).scatter(1, cand, cls)
            kind = torch.where((cls >= 2) & (cls < 2 + self.corrupt_kinds), cls - 1, 0)
            uni = [t.split(num_rows) for t in (cand, labels, kind)]
        for part in parts:
            part.pop("mask_frac", None)

        def finish(part, order, num, vols, coord, b, vis_patches, vis_mask):
            # a group's tokenization after its unified corruption (split out so batched_corrupt can apply
            # the corruptions of every group first; inline otherwise, the same ops in the same order)
            sp, patch = part["spacing"], part["patch"]
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
                if "unified_label" in part:
                    excl = part["unified_label"] > 0
                if "hemi_label" in part:
                    excl = part["hemi_label"] if excl is None else excl | part["hemi_label"]
                vis_patches, part["corrupt_label"] = self.corrupt_transform(vis_patches, vis_mask, patch, excl)
            if self.orient_head is not None and self.training:
                vis_patches, vis_mask, part["orient_label"] = self.orient_transform(
                    vis_patches, vis_mask, patch
                )
            part["vis_coord"], part["target_coord"] = coord.split([num_visible, num_predict], 1)
            part["vis_tok"] = vis_patches
            lvl_pre = self.level_head is not None and self.training and getattr(self, "batched_level", False)
            if lvl_pre:
                # batched_level: the level targets here (weight-free), one level head pass over all groups
                part["lvl_w"], part["lvl_t"] = self.level_targets(vis_patches, vis_mask, patch)
            if (self.level_head is not None and not (lvl_pre and self.level_mirror_head is None)) or (
                self.edgelvl_head is not None and self.training
            ):
                part["vis_patches"], part["vis_mask"] = vis_patches, vis_mask
            if self.lesion_head is not None or self.level_mirror_head is not None:
                # ms-syn: visible token ids into the [b * num] token grid; tokens a corruption altered
                # show the donor / a transform, not the host: out of the lesion and mirror losses
                vis_ids = (part["order"][..., :num_visible] + (torch.arange(b, device=device) * num).view(-1, 1, 1))
                vis_ids = vis_ids.flatten()
                altered = self.altered_tokens(part) if self.training else None
                part["syn_clean"] = None if altered is None else ~altered
                if "lesion" in part:
                    if part["lesion"] is not None:
                        part["lesion"] = part["lesion"].flatten(0, 1).index_select(0, vis_ids)
                        if self.tumor_one_view:
                            # views 1.. saw the clean volume: no lesion there (blobs are off with this flag)
                            les = part["lesion"].view(b, num_samples, num_visible, -1)
                            part["lesion"] = torch.cat([les[:, :1], torch.zeros_like(les[:, 1:])], 1).flatten(0, 2)
                    lm = vis_mask
                    if altered is not None:
                        lm = vis_mask * ~altered[..., None]
                    part["lesion_mask"] = lm
                    if lvl_pre:
                        part["les_w"], part["les_t"] = self.lesion_targets(
                            part.pop("lesion"), part.pop("lesion_mask"), patch, pool=sep_pool
                        )
                if self.level_mirror_head is not None:
                    # mirror references of the visible tokens [b * S * num_visible, 64], from the
                    # clean (lesion-pasted, uncorrupted) volumes
                    ref = self.mirror_ref(vols, patch, geoms[sp][1], geoms[sp][2], sp)
                    part["mirror_ref"] = ref.flatten(0, 1).index_select(0, vis_ids)

        # batched_corrupt (speed): the unified corruptions of all groups in one corrupt_apply_batched
        batched = self.training and self.unified_head is not None and getattr(self, "batched_corrupt", False)
        deferred, pending = [], []
        for part, order, num in zip(parts, orders, num_tokens):
            sp, patch, vols = part["spacing"], part["patch"], part.pop("vols")
            del part["scores"]
            b = len(part["ids"])
            part["order"] = order.unflatten(0, (b, num_samples))
            # images and masks gathered together by flat token ids
            flat_ids = part["order"] + gather_offsets(b, num, device)
            patches = vols.flatten(0, 2).index_select(0, flat_ids.flatten())
            patches, mask_patches = patches.view(2, b * num_samples, seq_length, -1)
            if "clean_img" in part:
                # tumor_one_view: views 1.. of every image read the clean (pre-paste) volume
                # flat_ids [2, b, S, seq] (image | mask rows): the image half indexes the [b * num] clean rows
                clean = part.pop("clean_img").flatten(0, 1).index_select(0, flat_ids[0].flatten())
                clean = clean.view(b * num_samples, seq_length, -1)
                later = (torch.arange(b * num_samples, device=device) % num_samples > 0).view(-1, 1, 1)
                patches = torch.where(later, clean, patches)
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
            if uni is not None:
                gi = len(deferred)
                cand, labels, kind = (t[gi] for t in uni)
                if b >= 2:
                    vis_patches, vis_mask, done = self.rtd_replace(
                        vis_patches, vis_mask, order, vols, b, num, num_samples, num_visible, select=labels == 1
                    )
                    labels = torch.where((labels == 1) & ~done, 0, labels)
                else:
                    labels = torch.where(labels == 1, 0, labels)
                part["unified_label"] = labels
                pending.append((vis_patches, vis_mask, patch, cand, kind))
            elif self.unified_head is not None and self.training:
                vis_patches, vis_mask, part["unified_label"] = self.unified_transform(
                    vis_patches, vis_mask, patch, order, vols, b, num, num_samples, num_visible,
                    defer=pending if batched else None,
                )
            if batched:
                deferred.append((part, order, num, vols, coord, b, vis_patches, vis_mask))
                continue
            finish(part, order, num, vols, coord, b, vis_patches, vis_mask)
        if batched:
            compact = getattr(self, "corrupt_compact", False)
            for item, corrupted in zip(deferred, self.corrupt_apply_batched(pending, compact=compact)):
                finish(*item[:6], corrupted, item[7])
        if self.flat_loss_ok():
            # flat_loss: the reconstruction loss' weights (in-mask fraction / voxels) and targets of every group,
            # flat in group order, so forward_native's loss is one pass over the cat of the predictions
            parts[0]["flat_w"] = flat_rows([(p.pop("target_mask_patches") / p["target_patches"].shape[-1]).flatten() for p in parts])
            parts[0]["flat_t"] = flat_rows([p.pop("target_patches").flatten() for p in parts])
        return parts

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
                if self.corrupt_kinds > 3:
                    for c, name in enumerate(["bias", "blur", "noise", "gamma", "down", "contrast", "ghost"][: self.corrupt_kinds], start=1):
                        sel = labels == c
                        stats[f"cr_rec_{name}"] = ((pred == c) & sel).sum() / sel.sum().clamp_min(1)
        if self.unified_head is not None and "unified_label" in parts[0]:
            labels = cat("unified_label").flatten()
            logits = self.unified_head(tokens).float().flatten(0, 1)
            uc_loss = F.cross_entropy(logits, labels)
            loss = loss + self.unified_weight * self.unified_scale * uc_loss
            stats["uc_loss"] = uc_loss.detach()
            # want_stats (native_log_stats_only: only on log steps): the logged accuracies, not in the loss
            if getattr(self, "want_stats", True):
                with torch.no_grad():
                    pred = logits.argmax(-1)
                    bad = labels > 0
                    stats["uc_frac"] = bad.float().mean()
                    stats["uc_acc_corrupt"] = ((pred == labels) & bad).sum() / bad.sum().clamp_min(1)
                    stats["uc_acc_clean"] = ((pred == 0) & ~bad).sum() / (~bad).sum().clamp_min(1)
                    names = ["swap", "bias", "blur", "noise", "gamma", "down", "contrast", "ghost"][: 1 + self.corrupt_kinds]
                    if self.unified_shuffle:
                        names.append("shuf")
                    for c, name in enumerate(names, start=1):
                        sel = labels == c
                        stats[f"uc_rec_{name}"] = ((pred == c) & sel).sum() / sel.sum().clamp_min(1)
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

    def set_aux_progress(self, t: float) -> dict[str, float]:
        # the aux schedules at training progress t (no-op without schedules); returns the scales
        def lin(k):
            a, b = self.aux_schedules[k]
            return 1.0 if t >= b else (0.0 if t <= a else (t - a) / (b - a))
        sched = self.aux_schedules
        if not sched:
            return {}
        unified = 1.0
        if "unified_ramp" in sched:
            unified *= lin("unified_ramp")
        if "unified_decay" in sched:
            unified *= 1.0 - lin("unified_decay")
        self.unified_scale = unified
        self.lesion_scale = lin("lesion_ramp") if "lesion_ramp" in sched else 1.0
        tumor = 1.0
        if "tumor_prob_ramp" in sched:
            tumor *= lin("tumor_prob_ramp")
        if "tumor_prob_decay" in sched:
            tumor *= 1.0 - lin("tumor_prob_decay")
        self.tumor_scale = tumor
        return {"unified_scale": self.unified_scale, "lesion_scale": self.lesion_scale, "tumor_scale": self.tumor_scale}

    def compile_native(self, **kwargs) -> None:
        # the fixed-shape transformer parts. per-spacing tokenization, embedding and head stay eager
        self.encode_tokens = torch.compile(self.encode_tokens, **kwargs)
        if self.fuse_layers:
            self.encode_tokens_fused = torch.compile(self.encode_tokens_fused, **kwargs)
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


def insert_blobs(
    images: Tensor,
    mask: Tensor,
    patch: tuple[int, int, int],
    sp: tuple[float, float, float],
    prob: float,
    num: int,
    radius: tuple[float, float],
    contrast: tuple[float, float],
    edge: float,
    box_mm: float = 40.0,
    with_mask: bool = False,
) -> Tensor | None:
    # in place on a spacing group's padded volumes [B, X, Y, Z] (voxels of sp mm): per image (share
    # prob) 1 to num soft edged ellipsoids with a perturbed boundary, sized in mm, each in a box_mm
    # box and centred in a patch >= 90% in the mask. with_mask: returns the soft blob mask [B, X, Y, Z]
    B, X, Y, Z = images.shape
    dev = images.device
    dims = torch.tensor([X, Y, Z], device=dev)
    pt = torch.tensor(patch, device=dev)
    mm = torch.tensor(sp, device=dev, dtype=torch.float32)
    gx, gy, gz = X // patch[0], Y // patch[1], Z // patch[2]
    ok = mask.view(B, gx, patch[0], gy, patch[1], gz, patch[2]).float().mean((2, 4, 6)).view(B, 1, -1) >= 0.9
    # uniform over the qualifying patches (any patch when none qualify)
    idx = (torch.rand(B, num, ok.shape[2], device=dev) + ok.float()).argmax(-1)
    cell = torch.stack([idx // (gy * gz), idx // gz % gy, idx % gz], -1)
    centre = (cell * pt).float() + torch.rand(B, num, 3, device=dev) * pt
    box = [min(d, math.ceil(box_mm / s)) for d, s in zip((X, Y, Z), sp)]
    bt = torch.tensor(box, device=dev)
    lo = torch.minimum((centre.long() - bt // 2).clamp_min(0), dims - bt)
    r = torch.exp(torch.rand(B, num, 1, device=dev) * math.log(radius[1] / radius[0]) + math.log(radius[0]))
    axes = r * torch.exp(torch.rand(B, num, 3, device=dev) * 0.4 - 0.2)
    knots = 0.1 * torch.randn(B * num, 1, 4, 4, 4, device=dev)
    pert = F.interpolate(knots, size=box, mode="trilinear", align_corners=True).view(B, num, -1)
    grid = torch.meshgrid(*[torch.arange(n, device=dev) for n in box], indexing="ij")
    pos = lo[:, :, None] + torch.stack(grid, -1).view(1, 1, -1, 3)
    d = ((pos - centre[:, :, None]) * mm / axes[:, :, None]).norm(dim=-1) * (1 + pert)
    on = (torch.rand(B, 1, device=dev) < prob) & (
        torch.arange(num, device=dev) < torch.randint(1, num + 1, (B, 1), device=dev)
    )
    amp = (torch.rand(B, num, 1, device=dev) * (contrast[1] - contrast[0]) + contrast[0]) * on[..., None]
    flat = (pos[..., 0] * Y + pos[..., 1]) * Z + pos[..., 2]
    flat = (flat + torch.arange(B, device=dev).view(-1, 1, 1) * (X * Y * Z)).view(-1)
    soft = torch.sigmoid((1 - d) * r / edge)
    inside = mask.reshape(-1)[flat].float()
    val = (amp * soft).view(-1) * inside
    images.view(-1).index_add_(0, flat, val.to(images.dtype))
    if not with_mask:
        return None
    out = torch.zeros_like(images)
    out.view(-1).scatter_reduce_(0, flat, ((soft * on[..., None]).view(-1) * inside).to(out.dtype), "amax")
    return out


def paste_tumors(images: Tensor, mask: Tensor, plan: tuple, with_mask: bool = False) -> Tensor | None:
    # in place on a spacing group's padded volumes [B, X, Y, Z], in the mask only, clipped to each
    # image's own in-mask range (a saturating enhancement). returns the soft lesion mask if asked
    flat, val, soft, lim = plan
    inside = mask.reshape(-1)[flat] > 0
    old = images.view(-1)[flat]
    new = torch.minimum(torch.maximum(old + val, lim[:, 0]), lim[:, 1]).to(old.dtype)
    images.view(-1).index_copy_(0, flat, torch.where(inside, new, old))
    if not with_mask:
        return None
    out = torch.zeros_like(images)
    out.view(-1).index_copy_(0, flat, (soft * inside).to(out.dtype))
    return out


def plan_tumors(
    images: list[Tensor],
    masks: list[Tensor],
    groups: dict[tuple, list[int]],
    geoms: dict[tuple, tuple],
    prob: float,
    radius: tuple[float, float],
    ring: float,
    contrast: tuple[float, float],
    edge: tuple[float, float],
    nodule: float,
    neg: float = 0.0,
    lims: list | None = None,
) -> dict[tuple, tuple[Tensor, Tensor, Tensor, Tensor]]:
    # one lobulated mass sized in mm in a prob share of the batch, centred >= 6 mm inside the mask: a
    # solid bright 3-10 mm nodule (share nodule), else a bright 3-8 mm rim of uneven thickness on a
    # mildly darker core (share ring) or a diffuse bright mass. all spacing groups in one pass over
    # the masses' own boxes; per spacing: flat voxel indices into its padded [B, X, Y, Z] volumes, the
    # value to add, the soft lesion mask and the image's in-mask (min, max) there
    dev = masks[0].device
    k = min(len(masks), int(prob * len(masks) + torch.rand(()).item()))
    if k == 0:
        return {}
    chosen = set(torch.randperm(len(masks))[:k].tolist())
    geo, owners, counts = [], [], {}
    for sp, ids in groups.items():
        patch, box, grid, _ = geoms[sp]
        dims = [num * p for num, p in zip(grid, patch)]
        for jj, ii in enumerate(ids):
            if ii not in chosen:
                continue
            src, dst = fit_slices(masks[ii].shape, box)
            span = [d.stop - d.start for d in dst]
            geo.append([*sp, *dims, *(d.start for d in dst), *span, *(d.start - c.start for c, d in zip(src, dst)),
                        *masks[ii].shape, jj * math.prod(dims)])
            owners.append(ii)
            counts[sp] = counts.get(sp, 0) + 1
    # per mass draws and box sizes on the cpu stream (known sizes, no gpu sync)
    geo = torch.tensor(geo, dtype=torch.float64)
    spc, dims, dst0, span, shift, shape = geo[:, :18].split(3, 1)
    u = torch.rand(k, 10, dtype=torch.float64)
    is_nodule = u[:, 9] < nodule
    is_ring = ~is_nodule & (u[:, 4] < ring)
    hot = is_nodule | is_ring
    r = torch.where(is_nodule, 3.0 * (10.0 / 3.0) ** u[:, 0], radius[0] * (radius[1] / radius[0]) ** u[:, 0])
    axes = r[:, None] * torch.exp(u[:, 1:4] * 0.4 - 0.2)
    # ring and nodule edges 2 mm wide (10-90%), diffuse edges at most r wide
    width = torch.where(hot, 2.0, torch.minimum(edge[0] + u[:, 5] * (edge[1] - edge[0]), r))
    rim = torch.minimum(3.0 + 5.0 * u[:, 6], 0.6 * r)
    amp = torch.where(hot, 2.0 + 3.0 * u[:, 7], contrast[0] + u[:, 7] * (contrast[1] - contrast[0]))
    if neg > 0:
        # h-tneg: a neg share of the masses darker than the tissue (drawn only when on: the default stream is unchanged)
        amp = torch.where(torch.rand(k, dtype=torch.float64) < neg, -amp, amp)
    # box per axis: its semi-axis at the largest boundary bump (d = 1 / 0.75), + the edge
    num = torch.minimum(dims, torch.ceil(2 * (axes / 0.75 + width[:, None]) / spc)).long()
    size = num.prod(1)
    start = size.cumsum(0) - size
    total = int(size.sum())
    ints = torch.cat([
        dims, dst0, span, shift, shape, (6.0 / spc).round().clamp_min(1), num, start[:, None], geo[:, 18:],
    ], 1).long().to(dev)
    dims, dst0, span, shift, shape, probe, num = ints[:, :21].split(3, 1)
    start, base = ints[:, 21], ints[:, 22]
    # columns: mm per voxel, 1 / semi-axes, 1 / bump scale, r, 1 / logistic scale, rim, ring, contrasts, knot row
    cols = torch.cat([
        spc, 1 / axes, 1 / (1.6 * r[:, None]), r[:, None], 4.4 / width[:, None], rim[:, None], is_ring[:, None],
        amp[:, None], 0.2 + 0.6 * u[:, 8:9], 5.0 * torch.arange(k)[:, None],
    ], 1).float().to(dev)

    # centre: of 64 voxels in the image, the first with itself and its 6 mm axis neighbours in the mask
    cand = torch.minimum((torch.rand(k, 64, 3, device=dev) * span[:, None]).long(), span[:, None] - 1) + dst0[:, None]
    star = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], device=dev
    )
    pts = (cand - shift[:, None])[:, :, None] + star * probe[:, None, None]
    pts = torch.minimum(pts.clamp_min(0), shape[:, None, None] - 1)
    pts = (pts[..., 0] * shape[:, 1, None, None] + pts[..., 1]) * shape[:, 2, None, None] + pts[..., 2]
    hits = torch.stack([masks[ii].reshape(-1)[pts[t].reshape(-1)] for t, ii in enumerate(owners)])
    best = hits.view(k, 64, 7).sum(-1).argmax(1)
    centre = torch.take_along_dim(cand, best.view(k, 1, 1).expand(k, 1, 3), 1)[:, 0] + torch.rand(k, 3, device=dev)
    lo = torch.minimum((centre.long() - num // 2).clamp_min(0), dims - num)
    # lims (native_worker_clip): the same in-mask (min, max), known from the normalization
    lim = torch.stack([
        lims[ii] if lims is not None and lims[ii] is not None else torch.stack([
            torch.where(masks[ii].bool(), images[ii], math.inf).amin(),
            torch.where(masks[ii].bool(), images[ii], -math.inf).amax(),
        ])
        for ii in owners
    ]).float()

    # every box voxel of every mass in one ragged row: owner, then its box and volume indices
    owner = torch.zeros(total, dtype=torch.int32, device=dev)
    owner[start[1:]] = 1
    owner = owner.cumsum(0, dtype=torch.int32)
    per = torch.stack([
        start, num[:, 1] * num[:, 2], num[:, 1], num[:, 2], dims[:, 1] * dims[:, 2], dims[:, 2],
    ], 1).int().index_select(0, owner)
    q = torch.arange(total, dtype=torch.int32, device=dev) - per[:, 0]
    loc = torch.stack([q // per[:, 1], q // per[:, 3] % per[:, 2], q % per[:, 3]], 1)
    flat = (base + (lo[:, 0] * dims[:, 1] + lo[:, 1]) * dims[:, 2] + lo[:, 2]).index_select(0, owner)
    flat = flat + (loc[:, 0] * per[:, 4] + loc[:, 1] * per[:, 5] + loc[:, 2])
    par = torch.cat([cols, centre - lo], 1).index_select(0, owner)
    rel = (loc - par[:, 14:17]) * par[:, 0:3]  # mm from the centre
    d = (rel * par[:, 3:6]).square().sum(1).sqrt()
    # boundary bumps and rim thickness: trilinear noise on 5^3 knots spanning +-1.6 r of the mass,
    # masses stacked on depth
    knots = (0.15 * torch.randn(1, 2, 5 * k, 5, 5, device=dev)).clamp(-0.25, 0.25)
    g = (rel * par[:, 6:7]).clamp(-1, 1)
    row = (par[:, 13] + (g[:, 0] + 1) * 2) * (2 / (5 * k - 1)) - 1
    grid = torch.stack([g[:, 2], g[:, 1], row], -1).view(1, total, 1, 1, 3)
    bump, thick = F.grid_sample(knots, grid, mode="bilinear", padding_mode="border", align_corners=True).view(2, total)
    depth = (1 - d * (1 + bump)) * par[:, 7]  # ~mm inside the boundary
    outer = torch.sigmoid(depth * par[:, 8])
    # rim 0.5-1.5x its thickness around the mass, 4 mm wide transition to the core
    core = torch.sigmoid((depth - par[:, 9] * (1 + 2 * thick)) * (4.4 / 4.0)) * par[:, 10]
    val = par[:, 11] * (outer - core) - par[:, 12] * core
    lim = lim.index_select(0, owner)
    out, t = {}, 0
    size, start = size.tolist(), start.tolist()
    for sp, n in counts.items():
        a, b = start[t], start[t + n - 1] + size[t + n - 1]
        out[sp] = (flat[a:b], val[a:b], outer[a:b], lim[a:b])
        t += n
    return out


@lru_cache(maxsize=1024)
def pool_kron(shape: tuple[int, int, int], out: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # adaptive average pooling's bins (floor(i size / out) .. ceil((i + 1) size / out) per axis) as a 0/1
    # [out^3, X Y Z] indicator matrix, and the bin lengths per axis
    ind, lens = [], []
    for size in shape:
        a = torch.zeros(out, size, dtype=torch.float32)
        for i in range(out):
            a[i, (i * size) // out : -((-(i + 1) * size) // out)] = 1.0
        ind.append(a)
        lens.append(a.sum(1))
    k = torch.einsum("ai,bj,ck->abcijk", *ind).reshape(out**3, -1)
    return tuple(_cached(t.to(device)) for t in (k, *lens))


def sep_pool(v: Tensor, out: int) -> Tensor:
    # F.adaptive_avg_pool3d(v, out) for v [N, C, X, Y, Z]: the bin sums as one fp32 GEMM with a 0/1 indicator
    # (exact for 0/1 inputs), then / len_x / len_y / len_z as the pool kernel divides (~10x faster than its
    # kernel on ~1e5 rows of small patches)
    n, c, x, y, z = v.shape
    k, lx, ly, lz = pool_kron((x, y, z), out, v.device)
    with torch.autocast(v.device.type, enabled=False):
        s = (v.float().reshape(n * c, x * y * z) @ k.t()).view(n, c, out, out, out)
    return s / lx.view(-1, 1, 1) / ly.view(1, -1, 1) / lz.view(1, 1, -1)


# native_prep_tokens: the lru-cached device constants below may be built in one prep thread (on its side
# stream) and read by another before that stream ran; with this set, a cache miss waits for its stream
CACHE_SYNC = False


def _cached(t: Tensor) -> Tensor:
    if CACHE_SYNC and t.is_cuda:
        torch.cuda.current_stream(t.device).synchronize()
    return t


class LazyTokens:
    """patchify3d(vol.flatten(0, k), patch).unflatten(...) of a padded volume [*lead, X, Y, Z] without the copy:
    flatten(0, j) over every leading dim + the token dim, then index_select(0, rows) gathers just those token rows
    (the same values), [i] selects a leading index; frac() is the per token mask fraction (the mean of the rows)."""

    def __init__(self, vol: Tensor, patch, grid):
        self.vol, self.patch, self.grid = vol, tuple(patch), tuple(grid)
        self.num = math.prod(grid)

    def __getitem__(self, i: int) -> "LazyTokens":
        return LazyTokens(self.vol[i], self.patch, self.grid)

    def flatten(self, start: int, end: int) -> "LazyTokens":
        assert start == 0 and end == self.vol.dim() - 3, "LazyTokens: flatten every leading dim and the tokens"
        return self

    def index_select(self, dim: int, rows: Tensor) -> Tensor:
        # advanced indexing on the patch-blocked view [lead, gx, gy, gz, px, py, pz] (no voxel index tensor)
        assert dim == 0
        (gx, gy, gz), (px, py, pz) = self.grid, self.patch
        v = self.vol.reshape(-1, gx, px, gy, py, gz, pz).permute(0, 1, 3, 5, 2, 4, 6)
        t = rows % self.num
        return v[rows // self.num, t // (gy * gz), t // gz % gy, t % gz].reshape(len(rows), -1)

    def frac(self) -> Tensor:
        # [lead, N]: the mean of each token's voxels (exact counts; the same division as Tensor.mean)
        lead = self.vol.shape[:-3]
        v = self.vol.reshape(-1, 1, *self.vol.shape[-3:]).float()
        k = F.avg_pool3d(v, self.patch, stride=self.patch, divisor_override=1)
        return (k.reshape(*lead, -1) / math.prod(self.patch))


def flat_rows(tensors: list[Tensor]) -> Tensor:
    # torch.cat for several, the tensor itself for one
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors)


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
    return patch, box, grid, _cached(coord)


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
    mask_patches: torch.Tensor | None,
    num_samples: int,
    min_mask_frac: float = 0.25,
    cube: tuple[tuple[int, int, int], int] | None = None,
    mask_frac: torch.Tensor | None = None,
) -> torch.Tensor:
    # sample_sequences' sort keys, [B, num_samples, N]. mask_frac [B, N]: the patches' mask fractions, if known
    if mask_frac is None:
        mask_frac = mask_patches.float().mean(dim=2)
    batch_size, num_patches = mask_frac.shape
    # one independent shuffle per sequence, with patches outside the mask sorted last so
    # they are only used when a volume has too few mask patches
    scores = torch.rand(batch_size, num_samples, num_patches, device=mask_frac.device)
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


@torch.no_grad()
def edge_targets(
    patches: Tensor, mask: Tensor, levels: Tensor, patch: tuple, spacing: tuple, rows: int, nv: int, sub: int
) -> tuple[Tensor, Tensor]:
    # ViTMAE3D.edgelvl_terms' targets (eager: torch.compile recompiles per spacing group and stalled a
    # smoke at <100 steps). gradient magnitude by central differences per mm where both neighbours are in the mask;
    # thresholds levels x the row's in-brain median (a fixed 1/16 voxel subsample); level_pool blocks
    px, py, pz = patch
    x = patches.float().reshape(-1, 1, px, py, pz)
    m = mask.float().reshape(-1, 1, px, py, pz)
    g2 = torch.zeros_like(x)
    for axis, size, sp in zip((2, 3, 4), (px, py, pz), spacing):
        if size < 3:
            continue
        d = (x.narrow(axis, 2, size - 2) - x.narrow(axis, 0, size - 2)) / (2.0 * sp)
        both = m.narrow(axis, 2, size - 2) * m.narrow(axis, 0, size - 2)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - axis)] = pad[2 * (4 - axis) + 1] = 1
        g2 = g2 + F.pad((d * both) ** 2, pad)
    g = g2.sqrt() * m
    flat = torch.where(m > 0, g, math.inf).reshape(rows, -1)[:, ::16]
    cnt = torch.isfinite(flat).sum(1)
    srt = flat.sort(dim=1).values
    med = srt.gather(1, (cnt // 2).clamp(max=flat.shape[1] - 1)[:, None]).squeeze(1)
    med = torch.where(torch.isfinite(med) & (med > 0), med, torch.ones_like(med))
    thr = med[:, None] * levels[None]
    thr = thr.repeat_interleave(nv, 0).view(-1, levels.shape[0], 1, 1, 1)
    above = (g > thr).float() * m
    weight = block_mean(m, sub)
    target = block_mean(above, sub) / weight.clamp_min(1e-6)
    return weight, target


def block_mean(x: Tensor, sub: int) -> Tensor:
    # adaptive_avg_pool3d(x, sub); a reshape mean when every axis divides (the 1 mm patches; the
    # adaptive pool kernel is ~1 ms per call at training size)
    n, c, px, py, pz = x.shape
    if px % sub or py % sub or pz % sub:
        return F.adaptive_avg_pool3d(x, sub)
    return x.reshape(n, c, sub, px // sub, sub, py // sub, sub, pz // sub).mean(dim=(3, 5, 7))


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
    return _cached(torch.arange(2 * batch_size, device=device).view(2, batch_size, 1, 1) * num_patches)


@lru_cache(maxsize=4096)
def grid_coords(grid: tuple[int, int, int], device: torch.device) -> Tensor:
    axes = [torch.arange(size, device=device) for size in grid]
    return _cached(torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3))


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


@lru_cache(maxsize=4096)
def cells8_index(patch: tuple[int, int, int], device: torch.device) -> Tensor:
    # flat voxel index in a px x py x pz patch of each 8^3 nearest-resampled voxel
    ax = [torch.div(torch.arange(8) * p, 8, rounding_mode="floor") for p in patch]
    i, j, k = torch.meshgrid(*ax, indexing="ij")
    return _cached(((i * patch[1] + j) * patch[2] + k).flatten().to(device))
