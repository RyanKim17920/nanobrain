"""Non-saturated collection-decodability measurements on a feature matrix (embeddings or input stats).

CPU only. Given an npz with `emb` (N,D) and a volumes tsv (column `collection`), reports balanced
accuracy (mean, sd over folds/draws) for: PCA-k -> LogisticRegression (PCA fit inside each fold),
strong-regularization C sweep, learning curve (n_train per class, 10 random draws, rest = test),
and cosine kNN (k=5). Same folds for every feature set (StratifiedKFold(5, shuffle, seed 0)).

    python scripts/collprobe_curves.py --feat output/collprobe/fb-s7338_emb.npz --volumes output/collprobe/volumes_X.tsv \
        --out output/collprobe/fb-s7338_curves.json
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PCA_KS = (2, 4, 8, 16, 32)
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
NTRAIN = (2, 5, 10, 20)


def _ms(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "vals": [float(v) for v in a]}


def _fit_eval(make, Xtr, ytr, Xte, yte):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(*make()).fit(Xtr, ytr)
    return balanced_accuracy_score(yte, clf.predict(Xte))


def curves(X, y, seed=0, n_draws=10, only=None):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler, normalize
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    res = {"n": int(len(y)), "K": int(len(np.unique(y))), "chance": 1.0 / len(np.unique(y))}
    D = X.shape[1]
    LR = lambda C: LogisticRegression(C=C, max_iter=3000)
    sc = StandardScaler
    full = {}
    full["pca"] = {k: _ms([_fit_eval(lambda: (sc(), PCA(k, random_state=0), LR(1.0)), X[a], y[a], X[b], y[b])
                          for a, b in folds]) for k in PCA_KS if k < D}
    full["C"] = {c: _ms([_fit_eval(lambda: (sc(), LR(c)), X[a], y[a], X[b], y[b]) for a, b in folds]) for c in CS}
    full["knn5_cos"] = _ms([_fit_eval(
        lambda: (sc(), KNeighborsClassifier(5, metric="cosine", algorithm="brute")), X[a], y[a], X[b], y[b])
        for a, b in folds])
    lc = {}
    classes = np.unique(y)
    idx_by = {c: np.where(y == c)[0] for c in classes}
    for n in NTRAIN:
        accs = []
        for d in range(n_draws):
            rng = np.random.default_rng(1000 * n + d)
            tr = np.concatenate([rng.choice(idx_by[c], size=n, replace=False) for c in classes])
            te = np.setdiff1d(np.arange(len(y)), tr)
            accs.append(_fit_eval(lambda: (sc(), LR(1.0)), X[tr], y[tr], X[te], y[te]))
        lc[n] = _ms(accs)
    full["learning_curve_C1"] = lc
    res.update(full)
    return res


def paired(a, b):
    """fb - r12t per matched fold/draw (same folds and draws for both): mean, sd of the paired diff, t-stat."""
    d = np.array(a["vals"]) - np.array(b["vals"])
    sd = d.std(ddof=1)
    return {"diff": float(d.mean()), "sd": float(sd), "se": float(sd / np.sqrt(len(d))), "n": len(d)}


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="emb")
    ap.add_argument("--group", help="column subset name stored in the npz as <key>__<group> (input stats)")
    args = ap.parse_args()
    z = np.load(args.feat, allow_pickle=True)
    X = z[args.key]
    vol = pd.read_csv(args.volumes, sep="\t")
    assert len(vol) == len(X)
    y = vol["collection"].to_numpy().astype(str)
    r = curves(X, y)
    Path(args.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: (v["mean"] if "mean" in v else {kk: round(vv["mean"], 4) for kk, vv in v.items()})
                      for k, v in r.items() if isinstance(v, dict) and k not in ()}, default=str))


if __name__ == "__main__":
    main()
