# DINOv2-style self-distillation for native 3D token sets (model_kwargs.dino, enabled: false = off,
# no params). Continued pretraining of a DINOv2 init with its own objective:
#   - EMA teacher (patch embed, pos tables, blocks, final norm, heads) initialised from the same
#     (pretrained) student weights, momentum ramped momentum -> 1 (cosine over training progress)
#   - DINO loss on the mean-pooled encoder tokens (the eval's global embedding; the encoder has no
#     cls token) of every student view vs the teacher's pooled embedding of the image's token pool
#     (global views: the two MAE sequences of the image, disjoint 512-token subsets of one
#     1280-token pool; local views: n_local compact regions of local_region brain tokens, local_size
#     of them sampled)
#   - iBOT loss: ibot_num mask tokens appended to each global student sequence at predicted
#     positions; their outputs match the teacher's tokens at those positions (teacher sees the pool
#     unmasked)
#   - KoLeo on the student's pooled global embeddings (per view)
#   - Sinkhorn-Knopp centering of the teacher (3 iterations, as DINOv2), teacher temp temp_t
# DINOv2's projection heads are not public (hub ships backbones only): heads start fresh, with
# head_warmup_steps of frozen backbone (train.py zeroes the lr of the pretrained groups).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULTS = {
    "enabled": False,
    "dino_weight": 1.0,
    "ibot_weight": 1.0,
    "koleo_weight": 0.1,
    "dino_dim": 8192,
    "ibot_dim": 8192,
    "hidden": 2048,
    "bottleneck": 256,
    "temp_s": 0.1,
    "temp_t": 0.07,
    "sk_iters": 3,
    "momentum": 0.994,
    "ibot_num": 256,
    "n_local": 2,
    "local_region": 384,
    "local_size": 128,
    "head_warmup_steps": 500,
    "eval_teacher": True,
}


def dino_config(cfg: dict | None) -> dict | None:
    if cfg is None or not cfg.get("enabled"):
        return None
    unknown = set(cfg) - set(DEFAULTS)
    assert not unknown, f"dino: unknown keys {unknown}"
    return {**DEFAULTS, **dict(cfg)}


class DinoHead(nn.Module):
    # DINOv2 DINOHead: 3-layer gelu mlp -> l2-normalized bottleneck -> weight-normalized prototypes
    # (g = 1, no bias), so logits are cosines in [-1, 1]
    def __init__(self, in_dim: int, out_dim: int, hidden: int, bottleneck: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.prototypes = nn.Parameter(torch.empty(out_dim, bottleneck))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        z = F.normalize(self.mlp(x), dim=-1, eps=1e-6)
        w = F.normalize(self.prototypes, dim=-1)
        return F.linear(z, w.to(z.dtype))


class Dino3D(nn.Module):
    # student-side dino params: the two heads and the iBOT mask token
    def __init__(self, embed_dim: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.dino_head = DinoHead(embed_dim, cfg["dino_dim"], cfg["hidden"], cfg["bottleneck"])
        self.ibot_head = DinoHead(embed_dim, cfg["ibot_dim"], cfg["hidden"], cfg["bottleneck"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)


@torch.no_grad()
def sinkhorn_knopp(logits: Tensor, temp: float, iters: int = 3) -> Tensor:
    # DINOv2 SinkhornKnoppTeacher (single gpu): rows = samples, returns per-row distributions
    q = logits.float() / temp
    q = (q - q.max()).exp().t()  # [K, B]
    k, b = q.shape
    q /= q.sum()
    for _ in range(iters):
        q /= q.sum(dim=1, keepdim=True).clamp_min(1e-30)
        q /= k
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-30)
        q /= b
    return (q * b).t()


def koleo(x: Tensor) -> Tensor:
    # DINOv2 KoLeoLoss: -mean log distance to the nearest other sample (l2-normalized)
    x = F.normalize(x.float(), dim=-1, eps=1e-8)
    with torch.no_grad():
        dots = x @ x.t()
        dots.fill_diagonal_(-2.0)
        nn_idx = dots.argmax(dim=1)
    d = (x - x[nn_idx]).norm(dim=-1)
    return -torch.log(d + 1e-8).mean()


def cross_entropy(p_t: Tensor, s_logits: Tensor, temp: float) -> Tensor:
    return -(p_t * F.log_softmax(s_logits.float() / temp, dim=-1)).sum(-1).mean()


@torch.no_grad()
def entropy_stats(logits: Tensor, temp: float) -> tuple[Tensor, Tensor]:
    # raw (pre-centering) teacher softmax: mean per-sample entropy and entropy of the batch mean
    # (collapse: both -> 0; uniform: both -> log K)
    p = F.softmax(logits.float() / temp, dim=-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    pm = p.mean(0)
    marg = -(pm * pm.clamp_min(1e-12).log()).sum()
    return ent, marg


def pool_order(order: Tensor, b: int, num_samples: int, num_visible: int) -> Tensor:
    # the two sequences of an image share one token pool (sequence 0's order): sequence 1 sees the
    # LAST num_visible pool tokens and predicts the rest, so its visible set is disjoint from
    # sequence 0's and the teacher encodes the pool once per image
    assert num_samples == 2, "dino: two sequences per image"
    L = order.shape[1]
    assert 2 * num_visible <= L, "dino: disjoint visible sets need 2 x num_visible <= sequence"
    o = order.view(b, num_samples, L)[:, 0]
    o2 = torch.cat([o[:, L - num_visible :], o[:, : L - num_visible]], dim=1)
    return torch.stack([o, o2], dim=1).flatten(0, 1)


@torch.no_grad()
def local_crops(
    vols: Tensor, b: int, num: int, coord: Tensor, cfg: dict, min_mask_frac: float
) -> tuple[Tensor, Tensor]:
    # n_local compact regions per image: the local_region brain tokens nearest (chebyshev, token
    # units, jittered) a random brain token, local_size of them sampled. vols [2, b, N, P]
    nl, reg, k = cfg["n_local"], cfg["local_region"], cfg["local_size"]
    device = vols.device
    brain = vols[1].float().mean(-1) >= min_mask_frac  # [b, N]
    r = torch.rand(b, nl, num, device=device) * brain[:, None].float()
    center = r.argmax(-1)  # [b, nl]
    cc = coord[center]  # [b, nl, 3]
    dist = (coord[None, None] - cc[:, :, None]).abs().amax(-1) + 0.5 * torch.rand(b, nl, num, device=device)
    dist = torch.where(brain[:, None], dist, torch.full_like(dist, 1e4))
    region = dist.topk(min(reg, num), dim=-1, largest=False).indices  # [b, nl, R]
    pick = torch.rand(region.shape, device=device).argsort(-1)[..., :k]
    order = region.gather(-1, pick)  # [b, nl, k]
    flat = order + (torch.arange(b, device=device) * num)[:, None, None]
    patches = vols[0].flatten(0, 1).index_select(0, flat.flatten()).view(b * nl, k, -1)
    return patches, coord[order].reshape(b * nl, k, 3)


def momentum_at(m0: float, progress: float) -> float:
    return 1.0 - (1.0 - m0) * (math.cos(math.pi * min(max(progress, 0.0), 1.0)) + 1.0) / 2.0
