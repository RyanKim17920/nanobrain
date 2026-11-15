# Objective round 3 (training only, every arm default off; no params and the flags-off path unchanged when off).
#   siteadv: gradient-reversal training-collection (site) classifier on the pooled visible-token embedding, CONDITIONAL
#            on biological covariates (bio_cov), so the encoder only loses site information biology does not explain
#   vicreg:  VICReg variance hinge + off-diagonal covariance on the pooled embedding (no invariance term)
#   diffmae: a fraction of the mask tokens get a noised copy of their target (+ its noise level) as a hint
#   xcon:    cross-contrast MAE: same-session other-contrast pairs exchange their second sequence's targets
#   travel:  SimSiam between one subject's scans from two scanners (same contrast; pure scanner variation)
# the biological anchors (age, brain volume, tissue fractions) and the FINO site head (fino3d.site) live in fino.py
import math
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Sampler

from .fino import NCOV, SITE_COL, GradScale


def _table(images_parquet, cols):
    t = pd.read_parquet(images_parquet, columns=["path", *cols])
    return t.set_index("derivatives/" + t["path"].str.removeprefix("source/").str.removesuffix(".nii.gz") + ".npz")


def site_codes(paths: list[str], images_parquet, min_count: int, exclude: list[str]) -> tuple[Tensor, list]:
    # per filelist image its training collection index (collections with >= min_count images; rarer ones and the
    # fino3d exclude_collections are NaN = masked). Training-set metadata only
    coll = _table(images_parquet, ["collection"])["collection"].reindex(paths)
    counts = coll[~coll.isin(exclude)].value_counts()
    names = sorted(counts[counts >= min_count].index.tolist())
    return torch.from_numpy(coll.map({n: float(i) for i, n in enumerate(names)}).to_numpy(dtype=np.float32)), names


def _contrast(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0].rsplit("_", 1)[-1]


def pair_partners(paths: list[str], images_parquet, kind: str) -> list[np.ndarray]:
    # xcon: the other anat images of the session on the same grid (orientation + spacing) with another contrast.
    # travel: the subject's (data.subject_counts key) images of the same contrast from another scanner
    # (manufacturer | model | field strength, both known) whose ages (when both known) differ by <= 1 year
    t = _table(images_parquet, ["spacing", "orientation", "manufacturer", "model_name", "field_strength", "age"]).reindex(paths)
    groups: dict[str, list[int]] = {}
    if kind == "xcon":
        grid = [f"{o}|{np.round(np.asarray(s, np.float64), 3).tolist()}" for s, o in zip(t["spacing"], t["orientation"])]
        for ii, p in enumerate(paths):
            d = p.rsplit("/", 1)[0]
            if d.endswith("/anat"):
                groups.setdefault(f"{d}|{grid[ii]}", []).append(ii)
        ok = lambda i, j: _contrast(paths[i]) != _contrast(paths[j])
    else:
        assert kind == "travel", kind
        from .data import subject_counts

        known = (t["manufacturer"].notna() & t["model_name"].notna()).to_numpy()
        scan = (t["manufacturer"].astype(str) + "|" + t["model_name"].astype(str) + "|" + t["field_strength"].astype(str)).to_numpy()
        age = t["age"].to_numpy(dtype=np.float64)
        for ii, k in enumerate(subject_counts(paths)[0]):
            if known[ii]:
                groups.setdefault(f"{k}|{_contrast(paths[ii])}", []).append(ii)
        ok = lambda i, j: scan[i] != scan[j] and not abs(age[i] - age[j]) > 1.0
    out = [np.zeros(0, np.int64)] * len(paths)
    for ids in groups.values():
        if len(ids) > 1:
            for ii in ids:
                out[ii] = np.array([jj for jj in ids if jj != ii and ok(ii, jj)], np.int64)
    return out


class PairMixSampler(Sampler[int]):
    """xcon / travel: each batch_size batch starts with n_pairs (anchor, partner) pairs at positions 2k, 2k + 1.
    Anchors are the base stream's next draws that have a partner (uniform among them); every other draw fills the
    rest of the batch in stream order, so no draw is dropped. Set base.start to start * (batch_size - n_pairs) //
    batch_size."""

    def __init__(self, base: Sampler, partners: list[np.ndarray], n_pairs: int, batch_size: int, seed: int = 0):
        assert 0 < 2 * n_pairs <= batch_size
        self.base, self.partners, self.n, self.bs, self.seed = base, partners, n_pairs, batch_size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 7919 * int(getattr(self.base, "start", 0)))
        it, anchors, rest, n_rest = iter(self.base), deque(), deque(), self.bs - 2 * self.n
        while True:
            while len(anchors) < self.n or len(rest) < n_rest:
                ii = next(it)
                (anchors if len(anchors) < self.n and len(self.partners[ii]) else rest).append(ii)
            for _ in range(self.n):
                a = anchors.popleft()
                p = self.partners[a]
                yield a
                yield int(p[rng.integers(len(p))])
            for _ in range(n_rest):
                yield rest.popleft()


def level_fractions(weight: Tensor, target: Tensor, R: int) -> Tensor:
    # per sequence the in-mask fraction of its visible voxels above each level threshold [R, T] (the level head's
    # targets pooled: an image-derived tissue / csf composition). weight [R V, 1, ...], target [R V, T, ...]
    T = target.shape[1]
    num = (weight.float() * target.float()).reshape(R, weight.shape[0] // R, T, -1).sum((1, 3))
    return (num / weight.float().reshape(R, -1).sum(1, keepdim=True).clamp_min(1e-6)).detach()


def bio_cov(meta: Tensor, frac: Tensor) -> Tensor:
    # [R, 9] biological covariates, no gradient: age z (FINO training labels; the leak-excluded collections NaN) +
    # known flag, brain volume z (QC mask) + known flag, the 5 tissue level fractions (image-derived)
    age, bv = meta[:, 0], meta[:, 5]
    flags = [age.nan_to_num(0.0), (~age.isnan()).float(), bv.nan_to_num(0.0), (~bv.isnan()).float()]
    return torch.cat([torch.stack(flags, 1), frac.float()], 1).detach()


class Obj3(nn.Module):
    def __init__(self, embed_dim: int, decoder_embed_dim: int, siteadv_weight=0.0, siteadv_classes=0, siteadv_grl=0.1,
                 vicreg_weight=0.0, vicreg_cov=0.04, vicreg_gamma=1.0, diffmae_frac=0.0, xcon_pairs=0,
                 travel_weight=0.0, travel_pairs=0):
        super().__init__()
        self.siteadv_weight, self.siteadv_grl = float(siteadv_weight), float(siteadv_grl)
        self.vicreg_weight, self.vicreg_cov, self.vicreg_gamma = float(vicreg_weight), float(vicreg_cov), float(vicreg_gamma)
        self.diffmae_frac, self.xcon_pairs = float(diffmae_frac), int(xcon_pairs)
        self.travel_weight, self.travel_pairs = float(travel_weight), int(travel_pairs)
        assert not (self.xcon_pairs and self.travel_pairs), "xcon / travel: one pair sampler"
        assert bool(self.travel_weight) == bool(self.travel_pairs), "travel: travel_weight and travel_pairs"
        self.site_head = None
        if self.siteadv_weight:
            assert siteadv_classes > 1, "siteadv: siteadv_classes (train.py sets it from the training collections)"
            # conditional discriminator: LN(GRL(z)) W1 + bio_cov Wc -> GELU -> classes
            self.site_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.site_head = nn.ModuleList([nn.Linear(embed_dim, 256), nn.Linear(NCOV, 256), nn.Linear(256, int(siteadv_classes))])
        if self.diffmae_frac:
            self.diff_hint = nn.Linear(embed_dim, decoder_embed_dim)
            self.diff_t = nn.Linear(32, decoder_embed_dim)
        if self.travel_weight:
            self.travel_pred = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, embed_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        if self.diffmae_frac:
            # zero init: the decoder input equals the plain mask token at step 0
            nn.init.zeros_(self.diff_hint.weight)
            nn.init.zeros_(self.diff_t.weight)
        self.stats = {}

    def pair_rows(self, parts: list[dict], S: int, n: int, same_part: bool) -> list[tuple[int, int]]:
        # row pairs (image 2k sample s, image 2k + 1 sample s) for k < n, s < S; rows part by part, image-major
        row = {}
        for p in parts:
            for ii in p["ids"]:
                for s in range(S):
                    row[(ii, s)] = len(row)
        part_of = {ii: pi for pi, p in enumerate(parts) for ii in p["ids"]}
        return [(row[(2 * k, s)], row[(2 * k + 1, s)]) for k in range(n) for s in range(S)
                if 2 * k + 1 in part_of and (not same_part or part_of[2 * k] == part_of[2 * k + 1])]

    def xcon_perm(self, parts: list[dict], S: int, device) -> Tensor | None:
        # row involution: sequence 1 of batch images 2k <-> 2k + 1 (k < xcon_pairs) when both sit in one spacing group
        if not self.xcon_pairs or S < 2:
            return None
        perm = list(range(sum(len(p["ids"]) for p in parts) * S))
        swaps = self.pair_rows(parts, S, self.xcon_pairs, True)[1::S]
        for a, b in swaps:
            perm[a], perm[b] = b, a
        self.stats["xcon_rows"] = float(2 * len(swaps))
        return torch.tensor(perm).pin_memory().to(device, non_blocking=True) if swaps else None

    def diffmae_hint(self, parts: list[dict], targets: list[Tensor], embed) -> tuple[Tensor, Tensor]:
        # targets: per part its clean target patches [rows, NP, P]. A diffmae_frac fraction of the target tokens get
        # x_t = sqrt(a) x0 + sqrt(1 - a) eps, a = cos^2(pi t / 2), t ~ U(0, 1) per token, patch-embedded (shared
        # native patch embedding) + a sinusoid of t, both through zero-init projections, added to their mask token
        hints, keeps = [], []
        freqs = torch.exp(torch.arange(16, device=targets[0].device) * (-math.log(1000.0) / 16))
        for p, x0 in zip(parts, targets):
            x0 = x0.float()
            t = torch.rand(x0.shape[:2], device=x0.device)
            keep = torch.rand(x0.shape[:2], device=x0.device) < self.diffmae_frac
            a = torch.cos(0.5 * math.pi * t).pow(2)[..., None]
            xt = a.sqrt() * x0 + (1.0 - a).sqrt() * torch.randn_like(x0)
            ang = 1000.0 * t[..., None] * freqs
            h = self.diff_hint(embed(xt, p)) + self.diff_t(torch.cat([ang.sin(), ang.cos()], -1))
            hints.append(h * keep[..., None])
            keeps.append(keep)
        return torch.cat(hints), torch.cat(keeps)

    def global_losses(self, pooled: Tensor, meta: Tensor | None, cov: Tensor | None, pairs) -> Tensor:
        # pooled [R, D]: each sequence's mean visible encoder token (the eval's global embedding, training version)
        loss, z = pooled.new_zeros((), dtype=torch.float32), pooled.float()
        if self.site_head is not None:
            site = meta[:, SITE_COL]
            keep = ~torch.isnan(site)
            w1, wc, w2 = self.site_head
            logits = w2(F.gelu(w1(self.site_norm(GradScale.apply(z, -self.siteadv_grl))) + wc(cov)))
            ce = (F.cross_entropy(logits, torch.where(keep, site, 0.0).long(), reduction="none") * keep).sum()
            ce = ce / keep.sum().clamp_min(1)
            loss = loss + self.siteadv_weight * ce
            self.stats["site_ce"] = ce.detach()
            self.stats["site_acc"] = ((logits.argmax(-1) == site) & keep).sum().detach() / keep.sum().clamp_min(1)
        if self.vicreg_weight:
            zc = z - z.mean(0, keepdim=True)
            std = (zc.var(0) + 1e-4).sqrt()
            var = F.relu(self.vicreg_gamma - std).mean()
            cm = zc.T @ zc / (z.shape[0] - 1)
            off = (cm.pow(2).sum() - cm.diagonal().pow(2).sum()) / z.shape[1]
            loss = loss + self.vicreg_weight * (var + self.vicreg_cov * off)
            self.stats.update(vic_var=var.detach(), vic_cov=off.detach(), vic_std=std.mean().detach())
        if self.travel_weight and pairs:
            # SimSiam both ways on the batch-centred pooled embeddings of the travelling pairs (stop-grad targets)
            a, b = (torch.tensor(x).pin_memory().to(z.device, non_blocking=True) for x in zip(*pairs))
            zc = z - z.mean(0, keepdim=True).detach()
            tr = 1.0 - 0.5 * (F.cosine_similarity(self.travel_pred(zc[a]), zc[b].detach(), dim=-1)
                              + F.cosine_similarity(self.travel_pred(zc[b]), zc[a].detach(), dim=-1)).mean()
            loss = loss + self.travel_weight * tr
            self.stats.update(travel_loss=tr.detach(), travel_cos=F.cosine_similarity(zc[a], zc[b], dim=-1).mean().detach())
        return loss


@torch.no_grad()
def token_mse(parts: list[dict], pred: Tensor, tgt: Tensor, w: Tensor, S: int, NP: int) -> tuple[Tensor, Tensor]:
    # the flat loss' per target token weighted squared error and weight sums [R, NP] (rows in cat order)
    num, den, o = [], [], 0
    for p in parts:
        P = int(np.prod(p["patch"]))
        n = len(p["ids"]) * S * NP * P
        num.append((w[o : o + n] * (pred[o : o + n].float() - tgt[o : o + n]).pow(2)).view(-1, NP, P).sum(-1))
        den.append(w[o : o + n].view(-1, NP, P).sum(-1))
        o += n
    return torch.cat(num), torch.cat(den)
