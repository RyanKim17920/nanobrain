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
        # extra level channels (1 mm path, num_samples 2; () = off): levels of each visible patch minus
        # the decoder prediction of it by the image's other sequence, where that sequence predicts it
        level_resid: tuple[float, ...] = (),
        # extra level channels on the in-patch gradient magnitude (central differences, z per mm;
        # 1 mm path; () = off): where the edges are in each sub-block
        level_grad: tuple[float, ...] = (),
        # each level threshold's BCE over its batch base-rate entropy (rare levels weigh like common)
        level_balance: bool = False,
        # second level head on level_pool2^3 sub-blocks (global thresholds; 1 mm path; 0 = off),
        # averaged with the first so the total level weight stays level_weight
        level_pool2: int = 0,
        # extra level channels (1 mm path, level_pool 2; () = off): in-mask fraction of each sub-block
        # above each threshold and above margin + the max within +-2-4 mm of its x-mirrored 2 mm cell
        level_mirror: tuple[float, ...] = (),
        level_mirror_margin: float = 0.5,
        # dark analogue (channels after level_mirror's; () = off): fraction below each threshold and
        # below the min within +-2-4 mm of the x-mirrored 2 mm cell - level_mirror_margin
        level_mirror_dark: tuple[float, ...] = (),
        # zero-init linear head on the decoder features: the level BCE on the predicted patches
        # (level_thresholds / level_pool; weight; 0 = off)
        decoder_level_weight: float = 0.0,
        # training only (1 mm path; 0 = off): share of images given 1 to blob_max synthetic focal
        # hyperintensities, radius log-uniform in blob_radius mm, + blob_contrast z, soft edge blob_edge mm
        blob_prob: float = 0.0,
        blob_max: int = 3,
        blob_radius: tuple[float, float] = (2.0, 9.0),
        blob_contrast: tuple[float, float] = (1.0, 3.0),
        blob_edge: float = 1.5,
    ):
        super().__init__()
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
        assert not (blob_prob and (native or target_spacing != 1.0)), "blob_prob: 1 mm path only"
        self.blob_prob = blob_prob
        self.blob_max = blob_max
        self.blob_radius = tuple(float(v) for v in blob_radius)
        self.blob_contrast = tuple(float(v) for v in blob_contrast)
        self.blob_edge = blob_edge

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
        self.level_resid = tuple(float(t) for t in level_resid)
        self.level_grad = tuple(float(t) for t in level_grad)
        self.level_balance = level_balance
        self.level_pool2 = level_pool2
        self.level_mirror = tuple(float(t) for t in level_mirror)
        self.level_mirror_margin = level_mirror_margin
        self.level_mirror_dark = tuple(float(t) for t in level_mirror_dark)
        # last training forward's mirror stats, named by level_mirror_stat_names
        self.level_mirror_stats = None
        self.level_mirror_stat_names = []
        if self.level_mirror:
            self.level_mirror_stat_names += ["mirror_bce", "mirror_valid"]
            self.level_mirror_stat_names += [f"mirror_pos{t:g}" for t in self.level_mirror]
        if self.level_mirror_dark:
            self.level_mirror_stat_names += [] if self.level_mirror else ["mirror_valid"]
            self.level_mirror_stat_names += ["mirror_dark_bce"]
            self.level_mirror_stat_names += [f"mirror_dark_pos{t:g}" for t in self.level_mirror_dark]
        if self.level_mirror or self.level_mirror_dark:
            assert level_weight > 0 and not native, "level_mirror: level head, 1 mm path only"
            # 2 mm cells, an even number of them along x so each pairs with its mirror cell
            assert level_pool == 2 and grid_size[0] % 4 == 0, "level_mirror: level_pool 2, grid x % 4 == 0"
            # background (0) is in the mirror window at the mask edge: these bounds keep it from moving a threshold
            mg = level_mirror_margin
            assert min(self.level_mirror, default=mg) >= mg, "level_mirror: thresholds >= level_mirror_margin"
            assert max(self.level_mirror_dark, default=-mg) <= -mg, "level_mirror_dark: thresholds <= -margin"
        self.level_head2 = None
        if level_weight and level_pool2:
            assert not native, "level_pool2: 1 mm path only"
            n_levels = len(level_thresholds)
            self.level_head2 = nn.Linear(embed_dim, n_levels * (patch_size // level_pool2) ** 3)
        self.level_head = None
        if level_weight:
            assert not (native and self.level_resid), "level_resid: 1 mm path only"
            levels = [float(t) for t in level_thresholds]
            sub = (patch_size // level_pool) ** 3
            assert not (native and self.level_grad), "level_grad: 1 mm path only"
            # channels: levels, resid, grad, mirror, mirror dark
            n_out = len(levels) + len(self.level_resid) + len(self.level_grad) + len(self.level_mirror)
            n_out += len(self.level_mirror_dark)
            self.level_head = nn.Linear(embed_dim, n_out * sub)
            self.register_buffer("level_grad_thresholds", torch.tensor(self.level_grad), persistent=False)
            self.register_buffer("level_mirror_thresholds", torch.tensor(self.level_mirror), persistent=False)
            dark = torch.tensor(self.level_mirror_dark)
            self.register_buffer("level_mirror_dark_thresholds", dark, persistent=False)
            self.register_buffer("level_thresholds", torch.tensor(levels), persistent=False)
            self.register_buffer("level_resid_thresholds", torch.tensor(self.level_resid), persistent=False)
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
        self.decoder_level_weight = decoder_level_weight
        self.decoder_level_head = None
        if decoder_level_weight:
            assert level_weight and not native, "decoder_level_weight: needs level_weight, 1 mm path only"
            sub = (patch_size // level_pool) ** 3
            # zero init after init_weights, off the cpu rng stream: the other weights init as without it
            with torch.random.fork_rng(devices=[]):
                self.decoder_level_head = nn.Linear(decoder_embed_dim, len(level_thresholds) * sub)
            nn.init.zeros_(self.decoder_level_head.weight)
            nn.init.zeros_(self.decoder_level_head.bias)

    @classmethod
    def from_config(cls, args: dict) -> "ViTMAE3D":
        return cls(
            grid_size=tuple(args["grid_size"]),
            target_spacing=args["target_spacing"],
            clip_quantiles=tuple(args["clip_quantiles"]),
            patch_size=args["patch_size"],
            native=args["native"],
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
        if self.level_head2 is not None:
            nn.init.zeros_(self.level_head2.weight)
            nn.init.zeros_(self.level_head2.bias)
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
    ):
        if isinstance(images, list):
            # native resolution batch, images on their own voxel grids
            return self.forward_native(
                images, mask, spacing, num_visible, num_predict, num_samples, with_state
            )
        assert not self.level_resid or num_samples == 2, "level_resid pairs sequences 2i, 2i + 1"
        seqs, order = self.sample_patches(images, mask, num_visible, num_predict, num_samples)
        if not with_state:
            if self.level_mirror_stat_names:
                loss, self.level_mirror_stats = self.patch_loss(*seqs)
                return loss
            return self.patch_loss(*seqs)
        loss, embeds, pred_patches = self.patch_loss(*seqs, with_outputs=True)[:3]
        vis_patches, vis_mask_patches, _, target_patches, target_mask_patches, _ = seqs[:6]
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

        if self.training and self.blob_prob > 0:
            # before patchify: blobs enter the inputs and every target, shared by an image's sequences
            images = insert_blobs(
                images, mask, self.patch_size[0], self.blob_prob, self.blob_max, self.blob_radius,
                self.blob_contrast, self.blob_edge,
            )
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
        if self.level_mirror_stat_names:
            # the visible patches' mirror references (patch_loss vis_aux)
            vis_order = order[:, :, :num_visible]
            aux = [mp[batch_ids, vis_order] for mp in self.mirror_map(images, mask)]
            seqs += ((aux[0] if len(aux) == 1 else torch.cat(aux, -1)).flatten(0, 1),)
        return seqs, order

    def mirror_map(self, images: Tensor, mask: Tensor) -> list[Tensor]:
        # mirror references per 2 mm cell as patches [B, N, sub^3]: the max (level_mirror), then the min
        # (level_mirror_dark) within +-2-4 mm of the x-mirrored cell, from one pass over the 2 mm cells;
        # 1e4 (max) and -1e4 (min) where that cell is out of the mask or within 4 mm of the midline
        B, X, Y, Z = images.shape
        cells = images.float().reshape(B, X // 2, 2, Y // 2, 2, Z // 2, 2)
        valid = mask.reshape(B, X // 2, 2, Y // 2, 2, Z // 2, 2).amax((2, 4, 6)).flip(1) > 0
        valid[:, X // 4 - 2 : X // 4 + 2] = False
        maps = []
        if self.level_mirror:
            hmax = F.max_pool3d(cells.amax((2, 4, 6))[:, None], 3, 1, 1)[:, 0]
            maps.append(patchify3d(torch.where(valid, hmax.flip(1), 1e4), self.patch_size[0] // 2))
        if self.level_mirror_dark:
            # min pooling as -max_pool(-x)
            hmin = -F.max_pool3d(-cells.amin((2, 4, 6))[:, None], 3, 1, 1)[:, 0]
            maps.append(patchify3d(torch.where(valid, hmin.flip(1), -1e4), self.patch_size[0] // 2))
        return maps

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
        vis_aux: Tensor | None = None,
        with_outputs: bool = False,
    ):
        # forward's loss from sampled sequences
        vis_mask_patches = vis_mask_patches.to(vis_patches.dtype)
        target_mask_patches = target_mask_patches.to(target_patches.dtype)
        embeds = self.forward_encoder(vis_patches, vis_coord)
        if self.decoder_level_head is not None:
            feats = self.decode_features(embeds, target_coord)
            pred_patches = self.decoder_head(feats)
        else:
            pred_patches = self.forward_decoder(embeds, target_coord)
        loss = F.mse_loss(pred_patches, target_patches, reduction="none")
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
            resid = None
            if self.level_resid:
                resid = cross_residual(vis_patches, vis_coord, target_coord, pred_patches)
            level = self.level_loss(embeds[:, self.class_tokens :], vis_patches, vis_mask_patches, resid, vis_aux)
            if vis_aux is not None:
                level, mirror_stats = level
            loss = loss + self.level_weight * level
        if self.decoder_level_head is not None:
            target, weight = self.level_targets(target_patches, target_mask_patches)
            logits = self.decoder_level_head(feats).float().reshape(target.shape)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            dec_loss = (weight * bce).sum() / (weight.sum() * target.shape[1]).clamp_min(1e-6)
            loss = loss + self.decoder_level_weight * dec_loss
        if vis_aux is not None:
            # the mirror stats last, for the train log
            return (loss, embeds, pred_patches, mirror_stats) if with_outputs else (loss, mirror_stats)
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
        if self.level_balance:
            # per-threshold sums, balanced once all spacing groups are summed
            sums = torch.stack([(weight * bce).sum((0, 2, 3, 4)), (weight * target).sum((0, 2, 3, 4))])
            return sums, weight.sum()
        return (weight * bce).sum(), weight.sum() * target.shape[1]

    def level_targets(self, patches: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # level_loss's targets and sub-block weights (decoder_level_head: on the predicted patches)
        p, q = self.patch_size[0], self.level_pool
        x = patches.float().reshape(-1, 1, p, p, p)
        m = mask.float().reshape(-1, 1, p, p, p)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        weight = F.avg_pool3d(m, q)
        return F.avg_pool3d(above, q) / weight.clamp_min(1e-6), weight

    def level_loss(self, tokens: Tensor, patches: Tensor, mask: Tensor, resid=None, aux=None):
        # level head BCE, sub-blocks weighted by their in-mask fraction
        p, q = self.patch_size[0], self.level_pool
        x = patches.float().reshape(-1, 1, p, p, p)
        m = mask.float().reshape(-1, 1, p, p, p)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        weight = F.avg_pool3d(m, q)
        target = F.avg_pool3d(above, q) / weight.clamp_min(1e-6)
        T = target.shape[1]
        logits = self.level_head(tokens).float().reshape(len(x), -1, *target.shape[2:])
        bce = F.binary_cross_entropy_with_logits(logits[:, :T], target, reduction="none")
        if self.level_balance:
            sums = torch.stack([(weight * bce).sum((0, 2, 3, 4)), (weight * target).sum((0, 2, 3, 4))])
            loss = balanced_level_loss(sums, weight.sum())
        else:
            loss = (weight * bce).sum() / (weight.sum() * T).clamp_min(1e-6)
        if self.level_grad:
            # gradient magnitude inside the patch: per axis the mean squared difference over the valid
            # (in-brain, in-patch) neighbour pairs, one-sided at faces
            g2 = x.new_zeros(x.shape)
            for axis in (2, 3, 4):
                both = m.narrow(axis, 1, p - 1) * m.narrow(axis, 0, p - 1)
                d2 = torch.diff(x, dim=axis).square() * both
                fwd, bwd = [0, 0] * (4 - axis) + [0, 1], [0, 0] * (4 - axis) + [1, 0]
                cnt = F.pad(both, fwd) + F.pad(both, bwd)
                g2 = g2 + (F.pad(d2, fwd) + F.pad(d2, bwd)) / cnt.clamp_min(1)
            g_above = (g2.sqrt() > self.level_grad_thresholds.view(1, -1, 1, 1, 1)).float() * m
            g_target = F.avg_pool3d(g_above, q) / weight.clamp_min(1e-6)
            G0 = T + len(self.level_resid)
            g_bce = F.binary_cross_entropy_with_logits(
                logits[:, G0 : G0 + len(self.level_grad)], g_target, reduction="none"
            )
            loss = loss + (weight * g_bce).sum() / (weight.sum() * T).clamp_min(1e-6)
        if resid is not None:
            # residual channels at the same per-channel weight, on the tokens the other sequence predicts
            r, hit = resid
            r_above = (r.reshape(-1, 1, p, p, p) > self.level_resid_thresholds.view(1, -1, 1, 1, 1)).float() * m
            r_target = F.avg_pool3d(r_above, q) / weight.clamp_min(1e-6)
            r_weight = weight * hit.reshape(-1, 1, 1, 1, 1)
            r_bce = F.binary_cross_entropy_with_logits(
                logits[:, T : T + len(self.level_resid)], r_target, reduction="none"
            )
            loss = loss + (r_weight * r_bce).sum() / (r_weight.sum() * T).clamp_min(1e-6)
        if aux is not None:
            # mirror channels (after resid, grad) at the same per-channel weight, on sub-blocks with a valid
            # mirror cell; aux holds the bright then the dark reference (mirror_map)
            maps = bool(self.level_mirror) + bool(self.level_mirror_dark)
            ref = aux.float().reshape(-1, maps, *weight.shape[2:])
            M0 = T + len(self.level_resid) + len(self.level_grad)
            stats = []
        if aux is not None and self.level_mirror:
            mh = ref[:, :1]
            thr = torch.maximum(self.level_mirror_thresholds.view(1, -1, 1, 1, 1), mh + self.level_mirror_margin)
            thr = thr.repeat_interleave(q, 2).repeat_interleave(q, 3).repeat_interleave(q, 4)
            m_target = F.avg_pool3d((x > thr).float() * m, q) / weight.clamp_min(1e-6)
            m_weight = weight * (mh < 1e3).float()
            M = len(self.level_mirror)
            m_bce = F.binary_cross_entropy_with_logits(logits[:, M0 : M0 + M], m_target, reduction="none")
            loss = loss + (m_weight * m_bce).sum() / (m_weight.sum() * T).clamp_min(1e-6)
            # per-channel mean bce, valid share of the level weight, positive rate per channel
            den = m_weight.sum().clamp_min(1e-6)
            stats += [
                ((m_weight * m_bce).sum() / (den * M)).view(1),
                (den / weight.sum().clamp_min(1e-6)).view(1),
                (m_weight * m_target).sum((0, 2, 3, 4)) / den,
            ]
            M0 += M
        if aux is not None and self.level_mirror_dark:
            # dark: below each threshold and below the mirror min - margin
            ml = ref[:, -1:]
            dark = self.level_mirror_dark_thresholds.view(1, -1, 1, 1, 1)
            thr = torch.minimum(dark, ml - self.level_mirror_margin)
            thr = thr.repeat_interleave(q, 2).repeat_interleave(q, 3).repeat_interleave(q, 4)
            d_target = F.avg_pool3d((x < thr).float() * m, q) / weight.clamp_min(1e-6)
            d_weight = weight * (ml > -1e3).float()
            D = len(self.level_mirror_dark)
            d_bce = F.binary_cross_entropy_with_logits(logits[:, M0 : M0 + D], d_target, reduction="none")
            loss = loss + (d_weight * d_bce).sum() / (d_weight.sum() * T).clamp_min(1e-6)
            den = d_weight.sum().clamp_min(1e-6)
            stats += [] if self.level_mirror else [(den / weight.sum().clamp_min(1e-6)).view(1)]
            stats += [
                ((d_weight * d_bce).sum() / (den * D)).view(1),
                (d_weight * d_target).sum((0, 2, 3, 4)) / den,
            ]
        if aux is not None:
            stats = torch.cat(stats).detach()
        if self.level_head2 is not None:
            q2 = self.level_pool2
            weight2 = F.avg_pool3d(m, q2)
            target2 = F.avg_pool3d(above, q2) / weight2.clamp_min(1e-6)
            logits2 = self.level_head2(tokens).float().reshape(target2.shape)
            bce2 = F.binary_cross_entropy_with_logits(logits2, target2, reduction="none")
            loss2 = (weight2 * bce2).sum() / (weight2.sum() * T).clamp_min(1e-6)
            loss = (loss + loss2) / 2
        if aux is not None:
            return loss, stats
        return loss

    def forward_native(
        self,
        images: list[Float[Tensor, "X Y Z"]],
        mask: list[Float[Tensor, "X Y Z"]],
        spacing: list[tuple[float, float, float]],
        num_visible: int,
        num_predict: int,
        num_samples: int = 1,
        with_state: bool = True,
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

        def cat(key: str) -> Tensor:
            # contiguous for one group too, the compiled encoder would recompile on view strides
            if len(parts) == 1:
                return parts[0][key].contiguous()
            return torch.cat([part[key] for part in parts])

        def flat_cat(tensors: list[Tensor]) -> Tensor:
            if len(tensors) == 1:
                return tensors[0].flatten()
            return torch.cat([t.flatten() for t in tensors])

        embeds = self.encode_tokens(cat("embeds"), cat("vis_coord"))
        feats = self.decode_features(embeds, cat("target_coord"))
        feats = torch.split(feats, [len(part["ids"]) * num_samples for part in parts])
        preds, targets, weights = [], [], []
        for part, feat in zip(parts, feats):
            sp = part["spacing"]
            pred = self.decoder_head.native(feat, sp, part["patch"], head_ws.get(sp))
            part["pred_patches"] = pred
            preds.append(pred)
            targets.append(part["target_patches"])
            # mean over each token's voxels
            weights.append(part["target_mask_patches"] / pred.shape[-1])
        # all groups' voxels in one loss
        weights = flat_cat(weights)
        loss = F.mse_loss(flat_cat(preds), flat_cat(targets), reduction="none")
        loss = (weights * loss).sum() / weights.sum()
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
            if self.level_balance:
                loss = loss + self.level_weight * balanced_level_loss(num, den)
            else:
                loss = loss + self.level_weight * num / den.clamp_min(1e-6)

        if not with_state:
            return loss
        return loss, {"parts": parts, "embeds": embeds}

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


def balanced_level_loss(sums: Tensor, total: Tensor) -> Tensor:
    # sums [2, T] (weighted BCE, weighted target): per-threshold mean BCE over its base-rate entropy,
    # rescaled by the mean entropy
    total = total.clamp_min(1e-6)
    per_level, rate = sums[0] / total, (sums[1] / total).clamp(5e-3, 1 - 5e-3)
    entropy = -(rate * rate.log() + (1 - rate) * torch.log1p(-rate))
    return (per_level / entropy).mean() * entropy.mean()


def cross_residual(
    vis_patches: Tensor, vis_coord: Tensor, target_coord: Tensor, pred_patches: Tensor
) -> tuple[Tensor, Tensor]:
    # each visible patch minus the detached decoder prediction of it by the image's other sequence
    # (rows 2i, 2i + 1), and whether that sequence predicts it
    B, N, P = vis_patches.shape
    other = torch.arange(B, device=vis_patches.device).view(-1, 2).flip(1).reshape(-1)
    same = (vis_coord[:, :, None] == target_coord[other][:, None]).all(-1)  # [B, N, M]
    idx = same.int().argmax(-1, keepdim=True).expand(-1, -1, P)
    pred = torch.gather(pred_patches.detach()[other].float(), 1, idx)
    return vis_patches.float() - pred, same.any(-1)


def insert_blobs(
    images: Tensor,
    mask: Tensor,
    patch: int,
    prob: float,
    num: int,
    radius: tuple[float, float],
    contrast: tuple[float, float],
    edge: float,
    box: int = 40,
) -> Tensor:
    # in place, fixed shapes, no host sync: per image (share prob) 1 to num slots on, each a soft edged
    # ellipsoid with a perturbed boundary in a box^3 voxel box, centred in a patch >= 90% in the mask
    B, X, Y, Z = images.shape
    dev = images.device
    gy, gz = Y // patch, Z // patch
    ok = mask.view(B, X // patch, patch, gy, patch, gz, patch).float().mean((2, 4, 6)).view(B, 1, -1) >= 0.9
    # uniform over the qualifying patches (any patch when none qualify)
    idx = (torch.rand(B, num, ok.shape[2], device=dev) + ok.float()).argmax(-1)
    cell = torch.stack([idx // (gy * gz), idx // gz % gy, idx % gz], -1)
    centre = (cell * patch).float() + torch.rand(B, num, 3, device=dev) * patch
    # box clamped into the grid, distances from the true centre
    lo = (centre.long() - box // 2).clamp_min(0)
    lo = torch.minimum(lo, torch.tensor([X - box, Y - box, Z - box], device=dev))
    r = torch.exp(torch.rand(B, num, 1, device=dev) * math.log(radius[1] / radius[0]) + math.log(radius[0]))
    axes = r * torch.exp(torch.rand(B, num, 3, device=dev) * 0.4 - 0.2)
    knots = 0.1 * torch.randn(B * num, 1, 4, 4, 4, device=dev)
    pert = F.interpolate(knots, size=(box,) * 3, mode="trilinear", align_corners=True).view(B, num, -1)
    ax = torch.arange(box, device=dev)
    pos = lo[:, :, None] + torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).view(1, 1, -1, 3)
    d = ((pos - centre[:, :, None]) / axes[:, :, None]).norm(dim=-1) * (1 + pert)
    on = (torch.rand(B, 1, device=dev) < prob) & (
        torch.arange(num, device=dev) < torch.randint(1, num + 1, (B, 1), device=dev)
    )
    amp = (torch.rand(B, num, 1, device=dev) * (contrast[1] - contrast[0]) + contrast[0]) * on[..., None]
    flat = (pos[..., 0] * Y + pos[..., 1]) * Z + pos[..., 2]
    flat = (flat + torch.arange(B, device=dev).view(-1, 1, 1) * (X * Y * Z)).view(-1)
    val = (amp * torch.sigmoid((1 - d) * r / edge)).view(-1) * mask.view(-1)[flat].to(images.dtype)
    return images.view(-1).index_add_(0, flat, val).view(B, X, Y, Z)


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
