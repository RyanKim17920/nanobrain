# FINO3D: auxiliary metadata objectives on the global embedding (port of nanopath's FINO).
#
# Heads read the mean of the visible encoder tokens (the training-time version of the eval's
# global embedding: the mean of the brain-touching last-layer tokens). The backbone sees each
# head's gradient times sign * gamma, with gamma ramped 0 -> gamma_max over training progress
# (FINO's sigmoid ramp). A negative sign is gradient reversal (invariance). The heads are training
# only: the eval rebuilds them from the checkpoint args and never calls them.
#
# Contamination gate (scratchpad fino/contamination.md): PT030_OpenNeuro/ds001510 shares subjects
# with OpenBHB, so every target is masked on exclude_collections.
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

# column order of the per-image metadata tensor; NaN = missing (masked)
COLUMNS = ("age", "sex", "group", "field_strength", "manufacturer", "brain_volume")
COLL_COL = len(COLUMNS)  # optional 7th column: collection index (coll_bias), never NaN
CONTROL = {"control", "controls", "normal", "hc", "cn", "healthy", "healthy control",
           "healthy controls", "typically developing", "td", "nc"}
MANUFACTURERS = ("siemens", "ge", "philips")


def head_dims(cfg) -> dict[str, int]:
    # output width of each enabled head (multiplier != 0)
    dims = {"age": 1, "sex": 1, "group": len(cfg["group_classes"] or []),
            "field_strength": 3, "manufacturer": len(MANUFACTURERS) + 1, "brain_volume": 1}
    return {k: d for k, d in dims.items() if float(cfg.get(k) or 0.0) != 0.0}


def normalize_group(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = " ".join(value.strip().lower().split())
    return "control" if value in CONTROL else value


def build_meta(cfg, paths: list[str], images_parquet: str | Path) -> Tensor:
    """Per-dataset-index metadata [N, 6] float32, joined on the sampler path
    (derivatives/<collection>/.../<stem>.npz <-> images.parquet source/<...>.nii.gz). Fills
    cfg.group_classes and cfg.age_mean / cfg.age_std (from the train filelist rows)."""
    cols = ["path", "collection", "age", "sex", "group", "field_strength", "manufacturer", "brain_volume"]
    table = pd.read_parquet(images_parquet, columns=cols)
    key = "derivatives/" + table["path"].str.removeprefix("source/").str.removesuffix(".nii.gz") + ".npz"
    table = table.set_index(key).reindex(paths)
    joined = int(table["collection"].notna().sum())
    assert joined == len(paths), f"fino3d: {len(paths) - joined} filelist paths not in images.parquet"
    excluded = table["collection"].isin(list(cfg.get("exclude_collections") or [])).to_numpy()

    age = table["age"].to_numpy(dtype=np.float64, copy=True)
    age[excluded] = np.nan
    if cfg.get("age_mean") is None:
        cfg["age_mean"] = float(np.nanmean(age))
        cfg["age_std"] = float(np.nanstd(age))
    age = (age - cfg["age_mean"]) / cfg["age_std"]

    sex = table["sex"].map({"F": 0.0, "M": 1.0}).to_numpy(dtype=np.float64)

    group = table["group"].map(normalize_group)
    if cfg.get("group_classes") is None:
        counts = group[~excluded].value_counts()
        cfg["group_classes"] = sorted(counts[counts >= int(cfg["group_min_count"])].index.tolist())
    index = {name: float(ii) for ii, name in enumerate(cfg["group_classes"])}
    group = group.map(index).to_numpy(dtype=np.float64)  # rare / missing -> NaN (masked)

    fs = table["field_strength"].to_numpy(dtype=np.float64)
    field = np.where(np.isnan(fs), np.nan, np.where(fs < 2.0, 0.0, np.where(fs < 5.0, 1.0, 2.0)))

    manu = table["manufacturer"].map(
        lambda v: (next((float(ii) for ii, m in enumerate(MANUFACTURERS) if str(v).lower().startswith(m)),
                        float(len(MANUFACTURERS))) if isinstance(v, str) and v.strip() else np.nan)
    ).to_numpy(dtype=np.float64)

    # brain volume (litres, our QC mask), standardized like age
    bv = table["brain_volume"].to_numpy(dtype=np.float64, copy=True)
    bv[excluded] = np.nan
    if cfg.get("bv_mean") is None:
        cfg["bv_mean"] = float(np.nanmean(bv))
        cfg["bv_std"] = float(np.nanstd(bv))
    bv = (bv - cfg["bv_mean"]) / cfg["bv_std"]

    meta = np.stack([age, sex, group, field, manu, bv], axis=1)
    meta[excluded] = np.nan
    if cfg.get("coll_bias"):
        # collection -> index: sorted collections of the train filelist (deterministic, stored in
        # the checkpoint args); missing / unseen collection -> the dedicated last index
        if cfg.get("collections") is None:
            cfg["collections"] = sorted(table["collection"].dropna().unique().tolist())
        cmap = {c: float(ii) for ii, c in enumerate(cfg["collections"])}
        cidx = table["collection"].map(cmap).fillna(float(len(cmap))).to_numpy(dtype=np.float64)
        meta = np.concatenate([meta, cidx[:, None]], axis=1)
        logger.info(
            f"fino3d coll_bias: {len(cmap)} collections (+1 unknown index), "
            f"known-collection fraction {float(np.mean(cidx < len(cmap))):.4f}"
        )
    known = {c: float(np.mean(~np.isnan(meta[:, ii]))) for ii, c in enumerate(COLUMNS)}
    logger.info(
        f"fino3d: {len(paths)} images, {int(excluded.sum())} masked by exclude_collections, "
        f"known fraction {known}, age mean {cfg['age_mean']:.1f} sd {cfg['age_std']:.1f}, "
        f"{len(cfg['group_classes'])} group classes {cfg['group_classes']}"
    )
    return torch.from_numpy(meta.astype(np.float32))


# identity forward; the backward scales the gradient into the backbone
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def ramp(gamma_max: float, progress: float) -> float:
    # FINO's ramp: 0 at the start, ~0.76 gamma_max at 20%, ~gamma_max from ~50% of training
    return gamma_max * (2.0 / (1.0 + math.exp(-10.0 * min(max(progress, 0.0), 1.0))) - 1.0)


class FinoHeads(nn.Module):
    def __init__(self, embed_dim: int, cfg):
        super().__init__()
        self.weight = float(cfg["weight"])
        reverse = cfg.get("nuisance_mode", "predict") == "reverse"
        self.dims = head_dims(cfg)
        self.signs = {}
        for k in self.dims:
            sign = float(cfg[k])
            self.signs[k] = -abs(sign) if reverse and k in ("field_strength", "manufacturer") else sign
        hidden = int(cfg["hidden"])
        # per-collection output offsets (training only): label offsets of a source dataset go
        # into b[collection], so the pooled embedding has no incentive to encode the site
        self.coll_bias = None
        if cfg.get("coll_bias"):
            n_coll = len(cfg["collections"]) + 1
            self.coll_bias = nn.ModuleDict({k: nn.Embedding(n_coll, d) for k, d in self.dims.items()})
            for emb in self.coll_bias.values():
                nn.init.zeros_(emb.weight)
        self.heads = nn.ModuleDict({
            k: nn.Sequential(nn.LayerNorm(embed_dim, elementwise_affine=False),
                             nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, d))
            for k, d in self.dims.items()
        })

    def forward(self, pooled: Tensor, meta: Tensor, gamma: float) -> tuple[Tensor, dict]:
        # pooled [R, D] global embeddings, meta [R, 6] (NaN = masked)
        pooled = pooled.float()
        loss = pooled.new_zeros(())
        stats = {}
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            for k, head in self.heads.items():
                target = meta[:, COLUMNS.index(k)]
                keep = ~torch.isnan(target)
                # every head runs on every row so the graph (and ddp) sees all params
                out = head(GradScale.apply(pooled, self.signs[k] * gamma))
                if self.coll_bias is not None:
                    out = out + self.coll_bias[k](meta[:, COLL_COL].long())
                n = keep.sum().clamp_min(1)
                t = torch.where(keep, target, torch.zeros_like(target))
                if k in ("age", "brain_volume"):
                    term = (F.mse_loss(out[:, 0], t, reduction="none") * keep).sum() / n
                elif k == "sex":
                    term = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * keep).sum() / n
                else:
                    term = (F.cross_entropy(out, t.long(), reduction="none") * keep).sum() / n
                loss = loss + self.weight * term
                stats[k] = term.detach()
        return loss, stats
