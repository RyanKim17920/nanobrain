# mni_coord: auxiliary MNI coordinate regression on the visible encoder tokens (training only).
#
# Inputs stay native (not resampled). Each visible token regresses the MNI152 mm coordinate of its
# patch centre: voxel (token centre in the image's RAS voxels, minus the fit_to_shape pad / crop
# offset) -> image world mm (RAS affine) -> MNI mm (the npz's mni_affine, ANTs 12 dof, the same
# field the mni branch reads). Coordinates are normalized as (mni - CENTRE) / SCALE; images without
# an affine (or with a failed registration, images.parquet template_dice < min_dice) are masked.
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger()

# MNI152 brain bounding box centre (x midline) and a scale putting the brain in about [-1, 1]
CENTRE = (0.0, -17.0, 5.0)
SCALE = 80.0


def build_ok(paths: list[str], images_parquet: str | Path, min_dice: float) -> np.ndarray:
    # per dataset index: registration usable (template_dice >= min_dice), joined like fino.build_meta
    table = pd.read_parquet(images_parquet, columns=["path", "template_dice"])
    key = "derivatives/" + table["path"].str.removeprefix("source/").str.removesuffix(".nii.gz") + ".npz"
    dice = table.set_index(key)["template_dice"].reindex(paths).to_numpy(dtype=np.float64)
    ok = np.nan_to_num(dice, nan=-1.0) >= float(min_dice)
    logger.info(f"mni_coord: {ok.mean():.4f} of {len(paths)} images with template_dice >= {min_dice}")
    return ok


class MniCoordHead(nn.Module):
    def __init__(self, embed_dim: int, cfg):
        super().__init__()
        self.weight = float(cfg["weight"])
        self.beta = float(cfg.get("beta", 0.05))
        self.head = nn.Linear(embed_dim, 3)
        self.register_buffer("centre", torch.tensor(CENTRE), persistent=False)

    def targets(self, vox_to_mni: Tensor, before: Tensor, grid, patch, order: Tensor) -> tuple[Tensor, Tensor]:
        # vox_to_mni [b, 4, 4] (RAS voxel -> MNI mm), before [b, 3] (fit_slices pad offset in voxels),
        # order [b, s, V] flat token ids -> normalized targets [b * s, V, 3], keep [b * s]
        b, s, v = order.shape
        dev = order.device
        axes = [torch.arange(n, device=dev, dtype=torch.float32) * p + (p - 1) / 2 for n, p in zip(grid, patch)]
        centres = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        ids = order.reshape(b, s * v)
        vox = centres[ids] - before[:, None, :]  # [b, s * v, 3] image voxels
        mni = torch.einsum("bij,bnj->bni", vox_to_mni[:, :3, :3], vox) + vox_to_mni[:, None, :3, 3]
        keep = torch.isfinite(vox_to_mni).flatten(1).all(1)
        target = torch.nan_to_num((mni - self.centre) / SCALE)
        return target.reshape(b * s, v, 3), keep.repeat_interleave(s)

    def forward(self, tokens: Tensor, target: Tensor, keep: Tensor) -> tuple[Tensor, dict]:
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            pred = self.head(tokens.float())
            err = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none").mean(dim=(1, 2))
            k = keep.float()
            n = k.sum().clamp_min(1.0)
            term = (err * k).sum() / n
            mm = ((pred.detach() - target).norm(dim=-1).mean(dim=1) * k).sum() / n * SCALE
        return self.weight * term, {"loss": term.detach(), "err_mm": mm, "keep": k.mean()}
