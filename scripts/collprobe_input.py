"""Input-statistics baseline for collprobe: collection decodability from cheap INPUT features, no encoder.

Same volumes (same select_volumes args as collprobe.py), same eval preprocessing (nanobrain.data:
crop_to_mask -> normalize_values -> reorient_to_ras -> resample_to_spacing -> fit_to_shape, with the
run's clip_quantiles / target_spacing / grid_size from --config). Feature groups:
  hist_z : 32-bin density histogram of the normalized (clip+z-score) brain values after resample (range -4..6)
  stats_z: mean/std/skew/kurt/percentiles(1,5,25,50,75,95,99) of those values
  raw    : same stats on the RAW (pre-normalization) in-mask values, plus raw max and log-scale
  geom   : in-mask voxel count (native and after resample), native shape (3), native spacing (3), cropped shape (3)
  all    : concatenation
Output <out-dir>/inputstats_<name>.npz (one matrix per group + `npz` paths) and the probe/curve JSON
(<name>_inputstats.json) using collprobe.probe (C1, Ccv) and collprobe_curves.curves per group.

    PYTHONPATH=<worktree>/src $PY scripts/collprobe_input.py --config <run>/config.yaml --name k25 [selection args]
"""
import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collprobe as cp  # noqa: E402
import collprobe_curves as cc  # noqa: E402

PCT = (1, 5, 25, 50, 75, 95, 99)


def moments(v: torch.Tensor) -> list:
    v = v.double()
    m, s = v.mean(), v.std(correction=0).clamp_min(1e-12)
    zs = (v - m) / s
    q = torch.quantile(v[:: max(1, len(v) // 200000)], torch.tensor([p / 100 for p in PCT], dtype=torch.float64))
    return [m.item(), s.item(), (zs ** 3).mean().item(), (zs ** 4).mean().item()] + q.tolist()


class Feats(torch.utils.data.Dataset):
    def __init__(self, paths, cfg, device):
        self.vol = cp.NpzVolumes(paths)
        self.cfg, self.device = cfg, device

    def __len__(self):
        return len(self.vol)

    def __getitem__(self, i):
        from nanobrain.data import brle_to_dense, crop_to_mask, fit_to_shape, normalize_values, reorient_to_ras, resample_to_spacing
        row = self.vol.members.iloc[i]
        with open(cp.DATA / row["shard"], "rb") as f:
            f.seek(int(row["offset"]))
            data = f.read(int(row["size"]))
        with np.load(io.BytesIO(data)) as npz:
            scale = float(npz["scale"])
            values = torch.from_numpy(npz["values"]).float() * scale
            shape = tuple(npz["shape"].tolist())
            mask = brle_to_dense(torch.from_numpy(npz["mask_rle"]), shape=shape)
            affine = npz["affine"]
        nat_spacing = np.linalg.norm(affine[:3, :3], axis=0).tolist()
        image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
        image, mask, affine = crop_to_mask(image, mask, affine)
        values = image[mask]
        n_nat = float(len(values))
        raw = moments(values) + [values.max().item(), float(np.log(abs(scale) + 1e-12))]
        v = normalize_values(values, tuple(self.cfg["clip_quantiles"]))
        img = v.new_zeros(mask.shape).masked_scatter_(mask, v)
        cshape = list(mask.shape)
        img, m, aff = reorient_to_ras(img, mask, affine)
        img, m, aff = resample_to_spacing(img, m, aff, float(self.cfg["target_spacing"]))
        img, m, aff = fit_to_shape(img, m, aff, tuple(self.cfg["grid_size"]))
        z = img[m > 0]
        hist = torch.histc(z.clamp(-4, 6), bins=32, min=-4, max=6)
        hist = (hist / hist.sum().clamp_min(1)).tolist()
        stz = moments(z)
        geom = [n_nat, float(len(z)), *map(float, shape), *nat_spacing, *map(float, cshape)]
        return {"hist_z": hist, "stats_z": stz, "raw": raw, "geom": geom}


def main():
    warnings.filterwarnings("ignore")
    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--filelist", default=str(cp.DEFAULT_FILELIST))
    ap.add_argument("--n-collections", type=int, default=25)
    ap.add_argument("--per-collection", type=int, default=80)
    ap.add_argument("--min-count", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N volumes only")
    ap.add_argument("--out-dir", default=str(HERE.parent / "output" / "collprobe"))
    args = ap.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = cp.select_volumes(args)
    if args.limit:
        sel = sel.iloc[:: max(1, len(sel) // args.limit)]
    ds = Feats(sel["npz"], cfg, "cpu")
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.workers, shuffle=False)
    rows = []
    for i, r in enumerate(loader):
        rows.append(r)
        if i % 200 == 0:
            print(f"feats {i}/{len(sel)}", flush=True)
    groups = {g: np.array([r[g] for r in rows], dtype=np.float64) for g in ("hist_z", "stats_z", "raw", "geom")}
    groups["all"] = np.concatenate([groups[g] for g in ("hist_z", "stats_z", "raw", "geom")], axis=1)
    np.savez(out / f"inputstats_{args.name}.npz", npz=sel["npz"].to_numpy(), **groups)
    sel.to_csv(out / f"inputstats_{args.name}_volumes.tsv", sep="\t", index=False)
    y = sel["collection"].to_numpy(dtype=object)
    res = {"name": args.name, "selection": {k: getattr(args, k) for k in ("n_collections", "per_collection", "min_count", "seed")}}
    for g, X in groups.items():
        Xs = np.sign(X) * np.log1p(np.abs(X)) if g in ("geom",) else X  # counts span orders of magnitude
        res[g] = {"probe": cp.probe(Xs, y), "curves": cc.curves(Xs, y.astype(str))}
        p = res[g]["probe"]
        print(f"inputstats {args.name} {g}: dim={X.shape[1]} C1 {p['C1']['bacc_mean']:.3f}+-{p['C1']['bacc_sd']:.3f} "
              f"Ccv {p['Ccv']['bacc_mean']:.3f}", flush=True)
    (out / f"{args.name}_inputstats.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
