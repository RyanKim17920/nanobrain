import copy
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
        # I-JEPA / data2vec style latent auxiliary on top of the pixel loss (weight; 0 = off, no
        # params): a linear head on the decoder features of the predicted tokens regresses the
        # layer-normed last-layer features of an EMA teacher encoder at those positions. the teacher
        # encodes the predicted tokens only (jepa_view "target") or visible + predicted ("full");
        # its momentum ramps jepa_momentum -> 1 (cosine) over training progress (ema_update)
        jepa_weight: float = 0.0,
        jepa_momentum: float = 0.996,
        jepa_view: str = "target",
        # pixel_weight scales the pixel stack (mse + grad + level-set loss); 0 = off (the decoder
        # still runs when jepa needs its features, else it is skipped). 1.0 = the base, bitwise
        pixel_weight: float = 1.0,
        # iBOT self-distillation (weight; 0 = off, no params). the student encoder sees the whole
        # sequence (visible tokens + ibot_mask_token at the predicted positions) behind a cls token;
        # ibot_head (shared mlp, l2 bottleneck, weight-normed cls / patch prototypes) on the student
        # cls and predicted-position tokens matches the EMA teacher's centered, sharpened
        # distributions (teacher sees the whole sequence unmasked): patch loss on the predicted
        # tokens of the same sequence, cls loss across the image's two sequences. the teacher is the
        # jepa teacher (momentum jepa_momentum -> 1 cosine). natinit.head loads the pretrained head
        ibot_weight: float = 0.0,
        ibot_cls_weight: float = 1.0,
        ibot_dim: int = 8192,
        ibot_temp_s: float = 0.1,
        ibot_temp_t: float = 0.07,
        ibot_center: float = 0.9,
    ):
        super().__init__()
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
        # built last for the same reason. the teacher is not a submodule (no state dict / optimizer
        # entry): built lazily from the student on the first training forward (also after a resume)
        self.jepa_weight = float(jepa_weight)
        self.jepa_momentum = float(jepa_momentum)
        assert jepa_view in ("target", "full"), jepa_view
        self.jepa_view = jepa_view
        self.jepa_head = None
        self.jepa_stats = {}
        self._teacher: list[nn.ModuleDict] = []
        if self.jepa_weight:
            assert native and not class_tokens, "jepa: native, no class tokens"
            self.jepa_head = nn.Linear(decoder_embed_dim, embed_dim)
            self.jepa_head.apply(_init_weights)
        self.pixel_weight = float(pixel_weight)
        self.ibot_weight = float(ibot_weight)
        self.ibot_cls_weight = float(ibot_cls_weight)
        self.ibot_temp_s, self.ibot_temp_t, self.ibot_center_m = ibot_temp_s, ibot_temp_t, ibot_center
        self.ibot_head = None
        self.ibot_stats = {}
        self._teacher_keys = self._TEACHER_KEYS
        if self.ibot_weight:
            assert native and not class_tokens, "ibot: native, no class tokens"
            assert not self.jepa_weight, "ibot and jepa share the teacher inputs: one at a time"
            self.ibot_head = IBOTHead(embed_dim, ibot_dim)
            self.ibot_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.ibot_mask_token, std=0.02)
            self.register_buffer("ibot_center", torch.zeros(1, ibot_dim))
            self.register_buffer("ibot_center2", torch.zeros(1, ibot_dim))
            self._teacher_keys = self._TEACHER_KEYS + ("ibot_head",)

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
    ) -> Float[Tensor, "B R+N D"]:
        # forward_encoder after the patch embedding
        B = x.shape[0]
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
    ) -> Float[Tensor, "B M D"]:
        # forward_decoder before the head
        B = embeds.shape[0]
        M = target_coord.shape[1]
        assert target_coord.shape == (B, M, 3)

        if self.global_proj is not None:
            pooled = self.global_proj(embeds[:, self.class_tokens :].mean(dim=1, keepdim=True))
        if self.training and self.decoder_keep < 1.0:
            # visible order is a random shuffle, so a prefix is a uniform subset
            keep = int((embeds.shape[1] - self.class_tokens) * self.decoder_keep)
            embeds = embeds[:, : self.class_tokens + keep]
        embeds = self.decoder_proj(embeds)

        mask_tokens = self.mask_token.expand(B, M, -1)
        mask_tokens = self.decoder_pos_embed(mask_tokens, target_coord)
        if self.global_proj is not None:
            mask_tokens = mask_tokens + pooled

        x = torch.cat([embeds, mask_tokens], dim=1)

        if self.decoder_prune_last and len(self.decoder_blocks):
            # the head reads only the mask tokens, so the last block computes just their rows
            for block in self.decoder_blocks[:-1]:
                x = block(x)
            x = self.decoder_blocks[-1].forward_last_rows(x, M)
        else:
            for block in self.decoder_blocks:
                x = block(x)
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
    ):
        if isinstance(images, list):
            # native resolution batch, images on their own voxel grids
            return self.forward_native(
                images, mask, spacing, num_visible, num_predict, num_samples, with_state,
                fino_meta=fino_meta, fino_gamma=fino_gamma,
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

    _TEACHER_KEYS = ("patch_embed", "pos_embed", "blocks", "norm")

    def teacher(self) -> nn.ModuleDict:
        if not self._teacher:
            t = nn.ModuleDict({k: copy.deepcopy(getattr(self, k)) for k in self._teacher_keys})
            t.requires_grad_(False)
            self._teacher.append(t)
        return self._teacher[0]

    def teacher_encode(self, x: Tensor, coord: Tensor) -> Tensor:
        # encode_tokens with the teacher's weights (no class tokens)
        t = self._teacher[0]
        x = t["pos_embed"](x, coord)
        for block in t["blocks"]:
            x = block(x)
        return t["norm"](x)

    @property
    def uses_teacher(self) -> bool:
        return self.jepa_head is not None or self.ibot_head is not None

    def ibot_encode(self, x: Tensor, coord: Tensor) -> Tensor:
        # encode_tokens behind the ibot cls token (no position): [B, 1 + N, D]
        x = self.pos_embed(x, coord)
        x = torch.cat([self.ibot_head.cls_token.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def teacher_encode_cls(self, x: Tensor, coord: Tensor) -> Tensor:
        t = self._teacher[0]
        x = t["pos_embed"](x, coord)
        x = torch.cat([t["ibot_head"].cls_token.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        for block in t["blocks"]:
            x = block(x)
        return t["norm"](x)

    @torch.no_grad()
    def ema_update(self, progress: float) -> None:
        # teacher <- m teacher + (1 - m) student, m ramped jepa_momentum -> 1 (cosine)
        if not self._teacher:
            return
        m0 = self.jepa_momentum
        m = 1.0 - (1.0 - m0) * (math.cos(math.pi * min(max(progress, 0.0), 1.0)) + 1.0) / 2.0
        t = self._teacher[0]
        t_params, s_params = [], []
        for k in self._teacher_keys:
            t_params += list(t[k].parameters())
            s_params += list(getattr(self, k).parameters())
        torch._foreach_lerp_(t_params, s_params, 1.0 - m)
        self.jepa_stats["m"] = m

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
            vols = patchify3d(vols.flatten(0, 1), patch).unflatten(0, (2, len(ids)))
            cube = (grid, self.mask_cube) if self.mask_cube else None
            scores = sequence_scores(vols[1], num_samples, self.min_mask_frac, cube=cube).flatten(0, 1)
            parts.append({"spacing": sp, "patch": patch, "ids": ids, "vols": vols})
            parts[-1]["scores"] = scores

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
            part["vis_coord"], part["target_coord"] = coord.split([num_visible, num_predict], 1)
            part["embeds"] = self.patch_embed.native(vis_patches, sp, patch, embed_ws.get(sp))
            if self.level_head is not None:
                part["vis_patches"], part["vis_mask"] = vis_patches, mask_patches[:, :num_visible]
            if self.uses_teacher and self.training:
                part["seq_patches"], part["coord"] = patches, coord

        def cat(key: str) -> Tensor:
            # contiguous for one group too, the compiled encoder would recompile on view strides
            if len(parts) == 1:
                return parts[0][key].contiguous()
            return torch.cat([part[key] for part in parts])

        def flat_cat(tensors: list[Tensor]) -> Tensor:
            if len(tensors) == 1:
                return tensors[0].flatten()
            return torch.cat([t.flatten() for t in tensors])

        ibot_loss = None
        if self.ibot_head is not None and self.training:
            # student: visible tokens + mask tokens at the predicted positions, behind the cls token
            vis = cat("embeds")
            x = torch.cat([vis, self.ibot_mask_token.to(vis.dtype).expand(vis.shape[0], num_predict, -1)], 1)
            out = self.ibot_encode(x, torch.cat([cat("vis_coord"), cat("target_coord")], 1))
            embeds = out[:, 1 : 1 + num_visible].contiguous()
            ibot_loss = self.ibot_aux(parts, out[:, 0], out[:, 1 + num_visible :], num_visible, num_samples)
        else:
            embeds = self.encode_tokens(cat("embeds"), cat("vis_coord"))
        jepa_loss = None
        if self.pixel_weight or self.jepa_head is not None:
            feats = self.decode_features(embeds, cat("target_coord"))
            if self.jepa_head is not None and self.training:
                jepa_loss = self.jepa_aux(parts, feats, num_visible)
        if self.pixel_weight:
            loss = self.pixel_weight * self.pixel_loss(parts, feats, embeds, head_ws, num_samples)
        else:
            loss = embeds.new_zeros((), dtype=torch.float32)
            for part in parts:
                part.pop("vis_patches", None), part.pop("vis_mask", None)
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

        if jepa_loss is not None:
            loss = loss + self.jepa_weight * jepa_loss
        if ibot_loss is not None:
            loss = loss + self.ibot_weight * ibot_loss

        if not with_state:
            return loss
        return loss, {"parts": parts, "embeds": embeds}

    def pixel_loss(self, parts: list[dict], feats: Tensor, embeds: Tensor, head_ws: dict, num_samples: int) -> Tensor:
        def flat_cat(tensors: list[Tensor]) -> Tensor:
            if len(tensors) == 1:
                return tensors[0].flatten()
            return torch.cat([t.flatten() for t in tensors])

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
                    tokens[:, self.class_tokens :], part.pop("vis_patches"), part.pop("vis_mask"),
                    part["patch"],
                )
                num, den = num + n, den + d
            loss = loss + self.level_weight * num / den.clamp_min(1e-6)
        return loss

    def jepa_aux(self, parts: list[dict], feats: Tensor, num_visible: int) -> Tensor:
        # latent prediction loss on the predicted tokens, weighted by their in-mask fraction
        teacher = self.teacher()
        full = self.jepa_view == "full"
        with torch.no_grad():
            sps = [part["spacing"] for part in parts if part["spacing"] != (1.0, 1.0, 1.0)]
            ws = {}
            if sps:
                ws = teacher["patch_embed"].native_weights(sps, [part["patch"] for part in parts
                                                                if part["spacing"] in sps])
                ws = dict(zip(sps, ws))
            xs, cs = [], []
            for part in parts:
                seq, coord = part.pop("seq_patches"), part.pop("coord")
                if not full:
                    seq, coord = seq[:, num_visible:], coord[:, num_visible:]
                sp = part["spacing"]
                xs.append(teacher["patch_embed"].native(seq, sp, part["patch"], ws.get(sp)))
                cs.append(coord)
            x = xs[0].contiguous() if len(xs) == 1 else torch.cat(xs)
            c = cs[0].contiguous() if len(cs) == 1 else torch.cat(cs)
            target = self.teacher_encode(x, c)
            if full:
                target = target[:, num_visible:]
            target = F.layer_norm(target.float(), target.shape[-1:])
        pred = self.jepa_head(feats).float()
        w = torch.cat([part["target_mask_patches"].float().mean(-1) for part in parts])
        err = (pred - target).pow(2).mean(-1)
        loss = (w * err).sum() / w.sum().clamp_min(1e-6)
        self.jepa_stats["loss"] = loss.detach()
        self.jepa_stats["pred_std"] = pred.detach().std(dim=0).mean()
        return loss

    def ibot_aux(self, parts: list[dict], s_cls: Tensor, s_patch: Tensor, num_visible: int, num_samples: int) -> Tensor:
        # iBOT: patch self-distillation on the predicted tokens (teacher sees them unmasked) +
        # cls self-distillation across the two sequences of each image (rows 2i, 2i + 1)
        assert num_samples == 2, "ibot pairs the two sequences of each image for the cls loss"
        teacher = self.teacher()
        with torch.no_grad():
            sps = [part["spacing"] for part in parts if part["spacing"] != (1.0, 1.0, 1.0)]
            ws = {}
            if sps:
                ws = teacher["patch_embed"].native_weights(sps, [part["patch"] for part in parts
                                                                if part["spacing"] in sps])
                ws = dict(zip(sps, ws))
            xs, cs = [], []
            for part in parts:
                seq, coord = part.pop("seq_patches"), part.pop("coord")
                sp = part["spacing"]
                xs.append(teacher["patch_embed"].native(seq, sp, part["patch"], ws.get(sp)))
                cs.append(coord)
            x = xs[0].contiguous() if len(xs) == 1 else torch.cat(xs)
            c = cs[0].contiguous() if len(cs) == 1 else torch.cat(cs)
            t = self.teacher_encode_cls(x, c)
            t_cls, t_patch = teacher["ibot_head"](t[:, 0], t[:, 1 + num_visible :])
            t_cls, t_patch = t_cls.float(), t_patch.float()
            p_cls = F.softmax((t_cls - self.ibot_center) / self.ibot_temp_t, dim=-1)
            p_patch = F.softmax((t_patch - self.ibot_center2) / self.ibot_temp_t, dim=-1)
            m = self.ibot_center_m
            self.ibot_center.mul_(m).add_(t_cls.mean(dim=0, keepdim=True), alpha=1.0 - m)
            self.ibot_center2.mul_(m).add_(t_patch.mean(dim=(0, 1)).unsqueeze(0), alpha=1.0 - m)
        s_cls, s_patch = self.ibot_head(s_cls, s_patch)
        logp_patch = F.log_softmax(s_patch.float() / self.ibot_temp_s, dim=-1)
        w = torch.cat([part["target_mask_patches"].float().mean(-1) for part in parts])
        patch = (w * -(p_patch * logp_patch).sum(-1)).sum() / w.sum().clamp_min(1e-6)
        swap = torch.arange(s_cls.shape[0], device=s_cls.device) ^ 1
        cls = -(p_cls[swap] * F.log_softmax(s_cls.float() / self.ibot_temp_s, dim=-1)).sum(-1).mean()
        self.ibot_stats["patch"] = patch.detach()
        self.ibot_stats["cls"] = cls.detach()
        # teacher entropy (collapse: -> 0 one prototype, -> log(dim) uniform)
        self.ibot_stats["t_ent"] = -(p_patch * p_patch.clamp_min(1e-12).log()).sum(-1).mean()
        return patch + self.ibot_cls_weight * cls

    def compile_native(self, **kwargs) -> None:
        # the fixed-shape transformer parts. per-spacing tokenization, embedding and head stay eager
        self.encode_tokens = torch.compile(self.encode_tokens, **kwargs)
        self.decode_features = torch.compile(self.decode_features, **kwargs)
        if self.jepa_head is not None:
            self.teacher_encode = torch.compile(self.teacher_encode, **kwargs)
        if self.ibot_head is not None:
            self.ibot_encode = torch.compile(self.ibot_encode, **kwargs)
            self.teacher_encode_cls = torch.compile(self.teacher_encode_cls, **kwargs)

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


class IBOTHead(nn.Module):
    # iBOT head (shared_head): 3-layer gelu mlp -> l2-normalized 256 bottleneck -> weight-normalized
    # prototypes with g = 1 (norm_last_layer): last_v = cls prototypes, last2_v = patch prototypes.
    # also holds the ibot cls token (prepended to student and teacher sequences, no position)
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.mlp.apply(_init_weights)
        self.last_v = nn.Parameter(torch.empty(out_dim, bottleneck))
        self.last2_v = nn.Parameter(torch.empty(out_dim, bottleneck))
        nn.init.normal_(self.last_v, std=0.02)
        nn.init.normal_(self.last2_v, std=0.02)

    def forward(self, cls: Tensor, patch: Tensor) -> tuple[Tensor, Tensor]:
        zc = F.normalize(self.mlp(cls), dim=-1)
        zp = F.normalize(self.mlp(patch), dim=-1)
        wc, wp = F.normalize(self.last_v, dim=-1), F.normalize(self.last2_v, dim=-1)
        return F.linear(zc, wc.to(zc.dtype)), F.linear(zp, wp.to(zp.dtype))


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
