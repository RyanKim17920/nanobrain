"""Linear decodability of FOMO collection identity (source dataset) from the eval's global embedding.

Diagnostic, no training. One checkpoint per call; results -> <script dir>/../output/collprobe/<run>.json

    PYTHONPATH= $PY scripts/collprobe.py --ckpt <run>/checkpoint-last.pth --config <run>/config.yaml \
        --code-root <worktree that trained the run>

Encoder: exactly the eval's model (brainmarks_smri create_model("vit_mae_3d", checkpoint=...) after
importing nanobrain.model from --code-root/src), and the eval's forward_embeddings: its
transform_cuda (clip + z-score, RAS, 1 mm resample, center fit to grid_size; deterministic, no
augmentation) and global_embedding = mean of the last-layer brain-touching tokens (the readout the
eval's classification/regression probes use).

Input: FOMO training volumes (the train filelist, fomo300k_images_qc.txt) read from the npz
derivatives like nanobrain.data.Fomo300kDataset, then crop_to_mask and values = image[mask], as the
eval's ViTMAE3D.transform does on its nifti input. Collection = images.parquet `collection` (the
column nanobrain.fino.build_meta joins on); manufacturer / field strength binned as in fino.build_meta.

Volume list: deterministic from (filelist, parquet, --seed, --n-collections, --per-collection,
--min-count); the same for every checkpoint. Saved next to the results.

Probe: StandardScaler + multinomial LogisticRegression, stratified 5-fold (seed 0), balanced
accuracy per fold; C fixed at 1 and C tuned by inner 3-fold CV over LOGISTIC_CS (neg log loss, like
the eval's probes).
"""

import argparse
import hashlib
import importlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
DATA = Path("/data/brainmarks-smri/fomo300k")
DEFAULT_FILELIST = HERE.parent / "src/nanobrain/config/fomo300k_images_qc.txt"
EXCLUDE = ("PT030_OpenNeuro/ds001510",)  # fino3d contamination gate (shares subjects with OpenBHB)
MANUFACTURERS = ("siemens", "ge", "philips")
LOGISTIC_CS = (1e-3, 1e-2, 1e-1, 1.0)


def select_volumes(args) -> pd.DataFrame:
    paths = Path(args.filelist).read_text().split()
    cols = ["path", "collection", "suffix", "field_strength", "manufacturer"]
    table = pd.read_parquet(DATA / "tables" / "images.parquet", columns=cols)
    key = "derivatives/" + table["path"].str.removeprefix("source/").str.removesuffix(".nii.gz") + ".npz"
    table = table.set_index(key).reindex(paths)
    assert table["collection"].notna().all(), "filelist paths missing from images.parquet"
    table = table[~table["collection"].isin(EXCLUDE)]
    counts = table["collection"].value_counts()
    eligible = sorted(counts[counts >= args.min_count].index)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(eligible, size=min(args.n_collections, len(eligible)), replace=False))
    rows = []
    for ii, coll in enumerate(chosen):
        sub = table[table["collection"] == coll]
        take = np.random.default_rng(args.seed * 1000 + ii).choice(
            len(sub), size=min(args.per_collection, len(sub)), replace=False)
        rows.append(sub.iloc[np.sort(take)])
    sel = pd.concat(rows)
    sel.index.name = "npz"
    print(f"{len(eligible)} collections with >= {args.min_count} volumes, chose {len(chosen)}, "
          f"{len(sel)} volumes", flush=True)
    return sel.reset_index()


class NpzVolumes(torch.utils.data.Dataset):
    def __init__(self, paths):
        members = pd.read_parquet(DATA / "tables" / "shards.parquet", columns=["path", "shard", "offset", "size"])
        self.members = members.set_index("path").loc[list(paths)]
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from nanobrain.data import brle_to_dense, crop_to_mask

        row = self.members.iloc[index]
        with open(DATA / row["shard"], "rb") as f:
            f.seek(int(row["offset"]))
            data = f.read(int(row["size"]))
        with np.load(io.BytesIO(data)) as npz:
            values = torch.from_numpy(npz["values"]).float() * float(npz["scale"])
            mask = brle_to_dense(torch.from_numpy(npz["mask_rle"]), shape=tuple(npz["shape"].tolist()))
            affine = npz["affine"]
        image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
        image, mask, affine = crop_to_mask(image, mask, affine)
        return {"values": image[mask], "mask": mask, "affine": affine}


def embed(args, sel: pd.DataFrame) -> np.ndarray:
    sys.path.insert(0, str(Path(args.code_root) / "src"))
    importlib.import_module("nanobrain.model")  # registers vit_mae_3d, like the eval's imports=
    from brainmarks_smri.models import create_model
    import nanobrain.model as nm

    print(f"nanobrain.model from {nm.__file__}", flush=True)
    torch.manual_seed(0)
    model = create_model("vit_mae_3d", checkpoint=args.ckpt).to("cuda").eval()
    loader = torch.utils.data.DataLoader(NpzVolumes(sel["npz"]), batch_size=8, num_workers=args.workers,
                                         collate_fn=lambda batch: batch)
    out, start = [], time.time()
    for ii, batch in enumerate(loader):
        out += [o["global_embedding"].cpu().numpy() for o in model.forward_embeddings(batch)]
        if ii % 25 == 0:
            print(f"embedded {len(out)}/{len(sel)} ({time.time() - start:.0f}s)", flush=True)
    emb = np.stack(out).astype(np.float32)
    assert np.isfinite(emb).all()
    return emb


def probe(emb: np.ndarray, labels: np.ndarray, min_per_class: int = 10) -> dict:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = pd.notna(labels)
    vals, counts = np.unique(labels[keep].astype(str), return_counts=True)
    ok = set(vals[counts >= min_per_class])
    keep &= np.array([str(v) in ok for v in labels])
    X, y = emb[keep], labels[keep].astype(str)
    classes, n = np.unique(y, return_counts=True)
    res = {"n": int(len(y)), "K": int(len(classes)), "chance": 1.0 / len(classes),
           "n_per_class": dict(zip(classes.tolist(), n.tolist()))}
    if len(classes) < 2:
        return res
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for tag, make in [
        ("C1", lambda: LogisticRegression(C=1.0, max_iter=3000)),
        ("Ccv", lambda: LogisticRegressionCV(Cs=LOGISTIC_CS, cv=3, scoring="neg_log_loss", max_iter=3000)),
    ]:
        accs, Cs = [], []
        for tr, te in folds.split(X, y):
            clf = make_pipeline(StandardScaler(), make()).fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
            if tag == "Ccv":
                Cs.append(float(clf[-1].C_[0]))
        res[tag] = {"bacc_mean": float(np.mean(accs)), "bacc_sd": float(np.std(accs, ddof=1)),
                    "folds": [float(a) for a in accs]}
        if Cs:
            res[tag]["C_per_fold"] = Cs
    return res


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the run's config.yaml (checked against ckpt args)")
    ap.add_argument("--code-root", required=True, help="worktree whose src/nanobrain trained the run")
    ap.add_argument("--name", help="result name, default: the run dir name")
    ap.add_argument("--filelist", default=str(DEFAULT_FILELIST))
    ap.add_argument("--n-collections", type=int, default=25)
    ap.add_argument("--per-collection", type=int, default=80)
    ap.add_argument("--min-count", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=str(HERE.parent / "output" / "collprobe"))
    args = ap.parse_args()
    name = args.name or Path(args.ckpt).resolve().parent.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    ckpt_args = torch.load(args.ckpt, map_location="cpu", weights_only=True)["args"]
    for k in ("grid_size", "patch_size", "clip_quantiles", "target_spacing", "native"):
        assert list(np.atleast_1d(cfg[k])) == list(np.atleast_1d(ckpt_args[k])), f"config/ckpt mismatch {k}"

    sel = select_volumes(args)
    sig = hashlib.sha1("\n".join(sel["npz"]).encode()).hexdigest()[:10]
    sel.to_csv(out_dir / f"volumes_{sig}.tsv", sep="\t", index=False)

    emb = embed(args, sel)
    np.savez(out_dir / f"{name}_emb.npz", emb=emb, npz=sel["npz"].to_numpy())

    fs = sel["field_strength"].to_numpy(dtype=np.float64)
    field = np.where(np.isnan(fs), None, np.where(fs < 2.0, "<2T", np.where(fs < 5.0, "2-5T", ">=5T")))
    manu = sel["manufacturer"].map(
        lambda v: (next((m for m in MANUFACTURERS if str(v).lower().startswith(m)), "other")
                   if isinstance(v, str) and v.strip() else None)).to_numpy(dtype=object)
    coll = sel["collection"].to_numpy(dtype=object)
    t1 = (sel["suffix"] == "T1w").to_numpy()
    coll_t1 = np.where(t1, coll, None)

    res = {
        "name": name, "ckpt": str(Path(args.ckpt).resolve()), "config": str(Path(args.config).resolve()),
        "code_root": str(Path(args.code_root).resolve()), "volumes": f"volumes_{sig}.tsv", "volume_sig": sig,
        "selection": {k: getattr(args, k) for k in ("n_collections", "per_collection", "min_count", "seed", "filelist")},
        "readout": "eval global_embedding = mean of last-layer brain-touching tokens (vit_mae_3d.forward_embeddings)",
        "probe": "StandardScaler + multinomial LogisticRegression, StratifiedKFold(5, shuffle, seed 0), balanced accuracy",
        "suffix_counts": sel["suffix"].value_counts().to_dict(),
        "collection": probe(emb, coll),
        "collection_T1w_only": probe(emb, coll_t1),
        "manufacturer": probe(emb, manu),
        "field_strength": probe(emb, field),
        "suffix": probe(emb, sel["suffix"].str.replace(r"^dwi_bval\d+$", "dwi", regex=True).to_numpy(dtype=object)),
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(res, indent=1))
    for k in ("collection", "collection_T1w_only", "manufacturer", "field_strength", "suffix"):
        r = res[k]
        if "C1" not in r:
            print(f"{name} {k}: K={r['K']} n={r['n']} (skipped)")
            continue
        print(f"{name} {k}: K={r['K']} n={r['n']} chance={r['chance']:.3f} "
              f"bacc C1 {r['C1']['bacc_mean']:.3f}+-{r['C1']['bacc_sd']:.3f} "
              f"Ccv {r['Ccv']['bacc_mean']:.3f}+-{r['Ccv']['bacc_sd']:.3f}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
