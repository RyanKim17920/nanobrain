# NUIS_DEC: nuisance-conditioned MAE decoder (Moyer et al. 2018, "explaining away").
#
# Learned embeddings of the acquisition nuisance of each image (manufacturer, field strength,
# contrast / sequence, voxel spacing bucket, collection) are summed and added to every decoder
# token (after decoder_proj, before the decoder blocks). Reconstruction can read scanner style
# from this side channel, so the encoder need not carry it. Training only in effect: the encoder
# and the eval path (forward_embeddings) never touch it.
#
# Per field: index 0 = unknown (missing metadata), 1..K = vocab values (>= min_count images in the
# train filelist), K + 1 = other (rare value). Tables init to zero (init_std 0), so step 0 is the
# base model's forward. The vocabularies are filled at train time and kept in the checkpoint args.
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger()

# column order of the per-image nuisance index tensor
FIELDS = ("manufacturer", "field_strength", "contrast", "spacing", "collection")
MANUFACTURER_PREFIX = (
    ("siemens", "siemens"), ("philips", "philips"), ("ge", "ge"), ("general electric", "ge"),
    ("toshiba", "canon"), ("canon", "canon"), ("hitachi", "hitachi"), ("ningbo", "ningbo"),
    ("brucker", "bruker"), ("bruker", "bruker"),
)


def _manufacturer(v) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    v = v.strip().lower()
    return next((name for prefix, name in MANUFACTURER_PREFIX if v.startswith(prefix)), v)


def _field_strength(v) -> str | None:
    if v is None or not np.isfinite(v):
        return None
    for edge, name in ((1.0, "<1T"), (2.0, "1.5T"), (3.5, "3T"), (5.0, "4T"), (8.0, "7T")):
        if v < edge:
            return name
    return ">=9T"


def _contrast(suffix, dwi_type) -> str | None:
    if not isinstance(suffix, str) or not suffix.strip():
        return None
    if not suffix.startswith("dwi"):
        return suffix
    if isinstance(dwi_type, str) and dwi_type:
        return f"dwi_{dwi_type}"
    if suffix.startswith("dwi_bval"):
        try:
            bval = float(suffix.removeprefix("dwi_bval"))
        except ValueError:
            return "dwi"
        return "dwi_b0" if bval < 50 else f"dwi_b{int(round(bval / 500.0) * 500)}"
    return "dwi"


def _spacing(v) -> str | None:
    try:
        sp = np.asarray(v, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if sp.size != 3 or not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        return None
    inplane, thick = sp.min(), sp.max()
    a = next(ii for ii, e in enumerate((0.8, 1.1, 1.6, np.inf)) if inplane < e)
    b = next(ii for ii, e in enumerate((1.1, 1.6, 2.6, 4.1, np.inf)) if thick < e)
    return f"i{a}t{b}"


def nuisance_values(table: pd.DataFrame) -> pd.DataFrame:
    # string value of each nuisance field per row (None = unknown)
    return pd.DataFrame({
        "manufacturer": table["manufacturer"].map(_manufacturer),
        "field_strength": table["field_strength"].map(_field_strength),
        "contrast": [_contrast(s, d) for s, d in zip(table["suffix"], table["dwi_type"])],
        "spacing": table["spacing"].map(_spacing),
        "collection": table["collection"].where(table["collection"].notna(), None),
    }, index=table.index)


def build_nuis(cfg, paths: list[str], images_parquet: str | Path) -> Tensor:
    """Per-dataset-index nuisance ids [N, len(FIELDS)] int64, joined on the sampler path like
    nanobrain.fino.build_meta. Fills cfg.vocab (per field list of values) when null."""
    cols = ["path", "collection", "suffix", "dwi_type", "field_strength", "manufacturer", "spacing"]
    table = pd.read_parquet(images_parquet, columns=cols)
    key = "derivatives/" + table["path"].str.removeprefix("source/").str.removesuffix(".nii.gz") + ".npz"
    table = table.set_index(key).reindex(paths)
    joined = int(table["collection"].notna().sum())
    assert joined == len(paths), f"nuis_dec: {len(paths) - joined} filelist paths not in images.parquet"
    values = nuisance_values(table)

    if cfg.get("vocab") is None:
        min_count = int(cfg["min_count"])
        vocab = {}
        for f in FIELDS:
            counts = values[f].dropna().value_counts()
            vocab[f] = sorted(counts[counts >= min_count].index.tolist())
        cfg["vocab"] = vocab
    ids = np.zeros((len(paths), len(FIELDS)), dtype=np.int64)
    report = {}
    for jj, f in enumerate(FIELDS):
        vocab = list(cfg["vocab"][f])
        index = {v: ii + 1 for ii, v in enumerate(vocab)}
        col = values[f]
        known = col.notna().to_numpy()
        ids[:, jj] = [0 if v is None or (isinstance(v, float) and np.isnan(v)) else index.get(v, len(vocab) + 1)
                      for v in col]
        report[f] = {
            "known_frac": round(float(known.mean()), 4),
            "distinct": int(col.dropna().nunique()),
            "vocab": len(vocab),
            "other_frac": round(float((ids[:, jj] == len(vocab) + 1).mean()), 4),
        }
    logger.info(f"nuis_dec: {len(paths)} images, coverage {report}")
    for f in FIELDS:
        if len(cfg["vocab"][f]) <= 16:
            logger.info(f"nuis_dec: {f} vocab {list(cfg['vocab'][f])}")
    return torch.from_numpy(ids)


class NuisanceEmbed(nn.Module):
    """Sum of one learned embedding per enabled nuisance field: ids [R, len(FIELDS)] -> [R, D]."""

    def __init__(self, dim: int, cfg):
        super().__init__()
        self.fields = [f for f in FIELDS if cfg.get(f, True)]
        assert self.fields, "nuis_dec: every field is off"
        self.tables = nn.ModuleDict({
            f: nn.Embedding(len(cfg["vocab"][f]) + 2, dim) for f in self.fields
        })
        init_std = float(cfg.get("init_std") or 0.0)
        for table in self.tables.values():
            if init_std > 0.0:
                nn.init.normal_(table.weight, std=init_std)
            else:
                nn.init.zeros_(table.weight)

    def forward(self, ids: Tensor) -> Tensor:
        out = None
        for f in self.fields:
            e = self.tables[f](ids[:, FIELDS.index(f)])
            out = e if out is None else out + e
        return out
