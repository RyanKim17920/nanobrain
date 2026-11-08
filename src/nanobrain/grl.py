# GRL: DANN domain-adversarial collection classifier on the global embedding.
#
# A linear probe on fb's pooled embedding recovers the FOMO source collection at 0.98 balanced
# accuracy (K=25), i.e. near-perfect site encoding. Here a small classifier reads the same pooled
# vector the FINO heads read (mean of the visible encoder tokens) through a gradient reversal layer:
# the classifier trains at full strength on cross-entropy over the collection id, the encoder gets
# -lambda * weight * grad. lambda ramps 0 -> lam_max with DANN's sigmoid over the first ramp_frac of
# the (time-keyed) training clock. Training only: the eval never calls the head.
#
# Classes: collections with >= min_count train images, plus one "other" class for the rest; a
# missing collection is ignored (NaN -> masked). Not masked on fino3d.exclude_collections: the
# target is the site, no eval label is used.
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger()


def build_collection_ids(cfg, paths: list[str], images_parquet: str | Path) -> Tensor:
    """Per-dataset-index collection class id [N] float32 (NaN = missing, ignored). Fills
    cfg.collection_classes (kept in the checkpoint args; the last class is "other")."""
    table = pd.read_parquet(images_parquet, columns=["path", "collection"])
    key = "derivatives/" + table["path"].str.removeprefix("source/").str.removesuffix(".nii.gz") + ".npz"
    coll = table.set_index(key)["collection"].reindex(paths)
    if cfg.get("collection_classes") is None:
        counts = coll.value_counts()
        cfg["collection_classes"] = sorted(counts[counts >= int(cfg["min_count"])].index.tolist())
    index = {name: float(ii) for ii, name in enumerate(cfg["collection_classes"])}
    other = float(len(index))
    ids = coll.map(lambda v: index.get(v, other) if isinstance(v, str) and v else np.nan)
    ids = ids.to_numpy(dtype=np.float64)
    n_other = int((ids == other).sum())
    logger.info(
        f"grl: {len(paths)} images, {len(index)} collections with >= {cfg['min_count']} images + "
        f"other ({n_other} images), {int(np.isnan(ids).sum())} missing (ignored)"
    )
    return torch.from_numpy(ids.astype(np.float32))


def num_classes(cfg) -> int:
    return len(cfg["collection_classes"]) + 1


def lam_schedule(lam_max: float, progress: float, ramp_frac: float) -> float:
    # DANN: lam_max * (2 / (1 + exp(-10 p)) - 1), p = progress / ramp_frac clipped to [0, 1]
    p = min(max(progress / max(ramp_frac, 1e-8), 0.0), 1.0)
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# identity forward, -lam * grad backward
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GrlHead(nn.Module):
    def __init__(self, embed_dim: int, cfg):
        super().__init__()
        self.weight = float(cfg["weight"])
        hidden = int(cfg["hidden"])
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes(cfg)),
        )

    def forward(self, pooled: Tensor, ids: Tensor, lam: float) -> tuple[Tensor, dict]:
        # pooled [R, D], ids [R] float (NaN = ignored). returns weight * ce and detached stats
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            logits = self.head(GradReverse.apply(pooled.float(), lam))
            keep = ~torch.isnan(ids)
            target = torch.where(keep, ids, torch.zeros_like(ids)).long()
            n = keep.sum().clamp_min(1)
            ce = (F.cross_entropy(logits, target, reduction="none") * keep).sum() / n
            acc = ((logits.argmax(dim=1) == target) & keep).sum() / n
        return self.weight * ce, {"ce": ce.detach(), "acc": acc.detach()}
