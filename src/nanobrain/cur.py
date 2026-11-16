# Data curation round (ms-cur; training only, every arm default off: no state and the flags-off path unchanged).
# Only training-time signals (the model's own reconstruction loss on training images, training-set paths): no eval
# labels, no eval-set performance, no dataset upweighting by resemblance to an eval set.
#   cur_dom_alpha: online domain importance sampling. A domain is a training collection x contrast (path-derived,
#       ~1.3k). Running per-domain reconstruction loss (decayed sums, shrunk toward the global mean by cur_shrink
#       draws); each image's draw weight is multiplied by (L_domain / L_global) ^ alpha, clipped to [1/clip, clip]
#   cur_img_alpha: online image importance sampling: x (L_image / L_its_domain) ^ (alpha exp(-age / cur_tau)) for the
#       images already drawn (age = steps since their last loss, so a stale loss decays back to weight 1)
#   cur_focal: hard-token reconstruction weighting: each target token's mse weight x (mse_token / mse_batch) ^ gamma
#       (detached, clipped at cur_focal_max; the loss stays the weight-normalized mean)
import logging
import os

import numpy as np
import torch
from torch import Tensor


logger = logging.getLogger(__name__)


def domain_codes(paths: list[str]) -> tuple[np.ndarray, list[str]]:
    # per image: its collection (derivatives/<collection...>/sub-*) x contrast (the first stem entity without a "-")
    names = []
    for path in paths:
        parts = path.split("/")
        cut = next((ii for ii, part in enumerate(parts) if part.startswith("sub")), 2)
        stem = parts[-1].rsplit(".", 1)[0]
        con = next((t for t in stem.split("_") if "-" not in t), "?")
        names.append("/".join(parts[1:cut]) + "|" + con)
    index: dict[str, int] = {}
    codes = np.array([index.setdefault(n, len(index)) for n in names], dtype=np.int64)
    return codes, list(index)


def _tok_eager(pred: Tensor, t: Tensor, w: Tensor, P: int) -> tuple[Tensor, Tensor]:
    # one spacing group's per token masked squared error / weight sums (obj3.token_mse's math, one fused pass)
    return (w * (pred.float() - t).pow(2)).view(-1, P).sum(-1), w.view(-1, P).sum(-1)


def _focal_eager(w: Tensor, f: Tensor, P: int) -> Tensor:
    return (w.view(-1, P) * f[:, None]).view(-1)


_tok = torch.compile(_tok_eager, dynamic=True)
_focal = torch.compile(_focal_eager, dynamic=True)


def token_stats(model, parts: list[dict], pred: Tensor, t: Tensor, w: Tensor, S: int) -> Tensor | None:
    """flat loss hook (no host sync): with model.cur_track the per batch image (num, den) of the masked mse into
    model.cur_last (detached); with model.cur_focal returns the focal voxel weights (else None)."""
    NP = parts[0]["target_coord"].shape[1]
    gamma = float(getattr(model, "cur_focal", 0.0))
    with torch.no_grad():
        nums, dens, sizes, o = [], [], [], 0
        pred = pred.detach()
        for p in parts:
            P = int(np.prod(p["patch"]))
            n = len(p["ids"]) * S * NP * P
            num, den = _tok(pred[o : o + n], t[o : o + n], w[o : o + n], P)
            nums.append(num)
            dens.append(den)
            sizes.append(P)
            o += n
        num, den = (nums[0], dens[0]) if len(nums) == 1 else (torch.cat(nums), torch.cat(dens))
        if getattr(model, "cur_track", False):
            rows = num.view(-1, NP).sum(1), den.view(-1, NP).sum(1)
            ids = torch.tensor([ii for p in parts for ii in p["ids"] for _ in range(S)], dtype=torch.int64)
            img = ids.pin_memory().to(num.device, non_blocking=True)
            B = 1 + max(max(p["ids"]) for p in parts)
            model.cur_last = tuple(r.new_zeros(B).index_add_(0, img, r) for r in rows)
        if not gamma:
            return None
        f = (num / den.clamp_min(1e-12)) / (num.sum() / den.sum().clamp_min(1e-12)).clamp_min(1e-12)
        f = f.clamp_min(1e-6).pow(gamma).clamp(max=float(getattr(model, "cur_focal_max", 4.0)))
        out, r, o = [], 0, 0
        for p, P in zip(parts, sizes):
            k = len(p["ids"]) * S * NP
            out.append(_focal(w[o : o + k * P], f[r : r + k], P))
            r += k
            o += k * P
        return out[0] if len(out) == 1 else torch.cat(out)


class Curator:
    """Online loss-driven sampling weights (cur_dom_alpha / cur_img_alpha) for data.InfiniteSampler.mult."""

    def __init__(self, args, paths: list[str], sampler, device):
        self.dom_alpha = float(args.get("cur_dom_alpha") or 0.0)
        self.img_alpha = float(args.get("cur_img_alpha") or 0.0)
        self.every = int(args.get("cur_every") or 200)
        self.start = int(args.get("cur_start") or 500)
        self.decay = float(args.get("cur_decay") or 0.8)
        self.shrink = float(args.get("cur_shrink") or 32.0)
        self.clip = float(args.get("cur_clip") or 3.0)
        self.tau = float(args.get("cur_tau") or 5000.0)
        self.debug = bool(os.environ.get("NB_CUR_DEBUG"))
        codes, self.names = domain_codes(paths)
        self.n_dom = len(self.names)
        self.dom_cpu = torch.from_numpy(codes)
        self.dom = self.dom_cpu.to(device)
        N = len(paths)
        self.sum = torch.zeros(self.n_dom, device=device, dtype=torch.float64)
        self.cnt = torch.zeros(self.n_dom, device=device, dtype=torch.float64)
        self.img_loss = torch.zeros(N, device=device)
        self.img_step = torch.full((N,), -1, device=device, dtype=torch.int64)
        self.S = np.zeros(self.n_dom)
        self.C = np.zeros(self.n_dom)
        self.sampler = sampler
        sampler.adaptive = True
        logger.info(
            f"curation sampling: {self.n_dom} domains, dom_alpha {self.dom_alpha} img_alpha {self.img_alpha}, "
            f"every {self.every} from step {self.start}, clip {self.clip}"
        )

    def observe(self, model, didx: Tensor, step: int) -> None:
        last = getattr(model, "cur_last", None)
        if last is None:
            return
        model.cur_last = None
        num, den = last
        d = didx.to(num.device, non_blocking=True)[: len(num)]
        ok = den > 0
        loss = torch.where(ok, num / den.clamp_min(1e-12), 0.0)
        dom = self.dom[d]
        self.sum.index_add_(0, dom, loss.double())
        self.cnt.index_add_(0, dom, ok.double())
        if self.img_alpha:
            self.img_loss[d] = torch.where(ok, loss, self.img_loss[d])
            self.img_step[d] = torch.where(ok, step, self.img_step[d])
        if (step + 1) % self.every == 0:
            self.update(step)

    def update(self, step: int) -> None:
        # one sync per cur_every steps
        self.S = self.decay * self.S + self.sum.cpu().numpy()
        self.C = self.decay * self.C + self.cnt.cpu().numpy()
        self.sum.zero_()
        self.cnt.zero_()
        if step < self.start or self.C.sum() <= 0:
            return
        g = self.S.sum() / self.C.sum()
        Ld = (self.S + self.shrink * g) / (self.C + self.shrink)
        lo, hi = 1.0 / self.clip, self.clip
        mult = np.ones(len(self.dom_cpu))
        dcode = self.dom_cpu.numpy()
        if self.dom_alpha:
            mult = mult * np.clip((Ld / g) ** self.dom_alpha, lo, hi)[dcode]
        if self.img_alpha:
            li = self.img_loss.cpu().numpy().astype(np.float64)
            age = step - self.img_step.cpu().numpy()
            seen = (self.img_step.cpu().numpy() >= 0) & (li > 0)
            expo = self.img_alpha * np.exp(-age / self.tau)
            ratio = np.where(seen, li / Ld[dcode], 1.0)
            mult = mult * np.where(seen, np.clip(ratio ** expo, lo, hi), 1.0)
        # one reference swap: the sampler reads it per chunk of draws (loader thread)
        self.sampler.mult = torch.from_numpy(mult)
        if self.debug:
            w = self.sampler.weights if self.sampler.weights is not None else torch.ones(len(mult), dtype=torch.float64)
            p = w * self.sampler.mult
            p = p / p.sum()
            q = w / w.sum()
            logger.info(
                f"cur step {step + 1}: global {g:.4f}, dom loss p5/p50/p95 "
                f"{np.percentile(Ld[self.C > 1], [5, 50, 95]).round(4).tolist()}, mult p5/p50/p95 "
                f"{np.percentile(mult, [5, 50, 95]).round(3).tolist()}, tv(p, base) {0.5 * float((p - q).abs().sum()):.3f}"
            )
            dm = np.bincount(dcode, weights=p.numpy(), minlength=self.n_dom) / np.maximum(
                np.bincount(dcode, weights=q.numpy(), minlength=self.n_dom), 1e-12)
            top = np.argsort(-dm)[:5]
            logger.info("cur top domains (draw mass ratio): " + ", ".join(f"{self.names[i]} {dm[i]:.2f}" for i in top))
