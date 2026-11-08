# Pluggable self-supervised objectives for native-resolution training (train.py `objective`).
#
#   mae       ViTMAE3D.forward (the default; not routed through this file at all)
#   dino3d    DINOv2/v3 self-distillation: EMA teacher, 3D multi-crop (2 global + N local token
#             crops), DINO head on the mean-pooled tokens (the eval's global readout), iBOT masked
#             token loss, KoLeo, Sinkhorn-Knopp or softmax centering, late-phase Gram anchoring
#   ijepa     I-JEPA: context encoder on a context block, narrow predictor regressing the EMA target
#             encoder's layer-normed features of multi-block 3D targets
#   lejepa    LeJEPA: multi-view prediction of the global-view mean embedding + SIGReg (sliced
#             Epps-Pulley test against an isotropic Gaussian); no teacher, no stop-gradient
#   data2vec2 data2vec 2.0: masked tokens regress the EMA teacher's instance-normed top-K block
#             average; several masks per image share one teacher pass
#
# Every objective keeps the level head (model.level_weight) on the student's visible tokens and can
# add the MAE pixel loss (aux_mae_weight) through the model's own decoder, on extra target tokens of
# the same view. Views are token subsets of the native token grid (forward_native's geometry):
# crops are "the R brain tokens nearest a random brain centre", subsampled to a fixed token count,
# so every view of a type has the same length and batches across spacing groups.
#
# The model (ViTMAE3D) never owns any of this: heads, teachers and predictors live in the objective
# module, saved as ckpt["objective"], so the brainmarks contract (ckpt["model"] + model_kwargs) is
# unchanged.

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import fit_slices, native_geometry, patchify3d
from .modules import Block, LayerNorm, SeparablePosEmbed3D

OBJECTIVES = ("mae", "dino3d", "ijepa", "lejepa", "data2vec2")

# hyperparameters per objective (args.ssl overrides). sources in the comments; adapted to 3D tokens
# and a 1 h budget (~30k steps)
COMMON = dict(
    global_tokens=512,  # tokens per global view (base MAE: 512 visible per sequence)
    global_scale=(0.35, 1.0),  # crop region as a fraction of the brain tokens (DINOv2 0.32-1.0)
    num_local=6,
    local_tokens=64,  # nearest brain tokens to a random centre: 64 x 512 mm^3 ~ a 40 mm ball
    mae_tokens=768,  # aux MAE targets per global/context view (base num_predict 768)
    momentum=(0.994, 1.0),  # EMA teacher, cosine ramp over training progress
    pool="mean",  # global embedding: mean of tokens (= the eval's global readout) or "cls"
    print_rank=True,
)
DEFAULTS = {
    "dino3d": dict(
        COMMON,
        dino_dim=8192,
        head_hidden=2048,
        head_bottleneck=256,
        head_layers=3,
        ibot_separate_head=True,
        ibot_dim=8192,
        temp_s=0.1,
        temp_t=(0.04, 0.07),  # warmup over temp_t_warmup of training progress
        temp_t_warmup=0.3,
        centering="sinkhorn",  # or "center" (DINO v1 EMA centre, momentum 0.9)
        sk_iters=3,
        center_momentum=0.9,
        dino_weight=1.0,
        ibot_weight=1.0,
        ibot_prob=0.5,  # fraction of global views that are masked
        ibot_ratio=(0.1, 0.5),
        koleo_weight=0.1,
        gram_weight=0.0,  # DINOv3 Gram anchoring (2.0 there); 0 = off
        gram_start=0.6,  # training progress at which the Gram teacher is frozen and the term starts
        cross_view_global=True,  # with cross_sequence_views: global view 1 is the partner sequence
    ),
    "ijepa": dict(
        COMMON,
        momentum=(0.996, 1.0),
        masks=2,  # independent maskings per image (base num_samples 2)
        context_tokens=512,
        target_tokens=384,  # predicted target tokens per mask (subsampled from the 4 blocks)
        num_blocks=4,
        block_scale=(0.15, 0.2),  # each target block, fraction of the brain tokens
        context_scale=(0.85, 1.0),
        pred_depth=6,
        pred_dim=384,
        pred_heads=12,
        loss="smooth_l1",
        target_layers=1,  # mean of the last K block outputs (1 = the final output, I-JEPA)
        target_norm="layer",
    ),
    "data2vec2": dict(
        COMMON,
        momentum=(0.9998, 0.99999),
        masks=2,
        context_tokens=512,
        target_tokens=768,
        num_blocks=0,  # 0 = random masking
        block_scale=(0.15, 0.2),
        context_scale=(1.0, 1.0),
        pred_depth=4,
        pred_dim=512,
        pred_heads=8,
        loss="mse",
        target_layers=10,  # average_top_k_layers 10 (ViT-B, base_images_only_task.yaml)
        target_norm="instance",
    ),
    "lejepa": dict(
        COMMON,
        num_local=6,
        proj_hidden=2048,
        proj_dim=64,  # LeJEPA Table 1d: 64 best
        proj_layers=3,
        lamb=0.05,
        slices=256,
        knots=17,
        t_max=5.0,
    ),
}


def ssl_config(args) -> dict:
    cfg = dict(DEFAULTS[args.objective])
    for key, value in dict(args.get("ssl") or {}).items():
        assert key in cfg, f"unknown ssl key {key} for {args.objective}"
        cfg[key] = tuple(value) if isinstance(cfg[key], tuple) else value
    return cfg


def build_objective(args, model) -> nn.Module:
    cfg = ssl_config(args)
    cls = {"dino3d": DINO3D, "ijepa": IJEPA, "data2vec2": IJEPA, "lejepa": LeJEPA}[args.objective]
    return cls(model, cfg, args.objective, float(args.get("aux_mae_weight", 0.0) or 0.0))


# ---------------------------------------------------------------------------------------------
# metrics


@torch.no_grad()
def rankme(z: Tensor) -> Tensor:
    # RankMe (Garrido et al. 2023): exp entropy of the normalized singular values
    s = torch.linalg.svdvals(z.float())
    p = s / s.sum().clamp_min(1e-12) + 1e-7
    return torch.exp(-(p * p.log()).sum())


@torch.no_grad()
def feature_stats(z: Tensor) -> dict[str, Tensor]:
    z = z.float()
    zn = F.normalize(z, dim=-1)
    return {"rankme": rankme(z), "feat_std": zn.std(dim=0).mean(), "raw_std": z.std(dim=0).mean()}


# ---------------------------------------------------------------------------------------------
# losses


def koleo_loss(x: Tensor, eps: float = 1e-8) -> Tensor:
    # DINOv2 KoLeoLoss: -mean log distance to the nearest neighbour (l2-normalized features)
    x = F.normalize(x.float(), eps=eps, p=2, dim=-1)
    with torch.no_grad():
        dots = x @ x.t()
        n = x.shape[0]
        dots.view(-1)[:: n + 1].fill_(-1)
        nn_ids = dots.argmax(dim=1)
    dist = F.pairwise_distance(x, x[nn_ids], p=2, eps=eps)
    return -torch.log(dist + eps).mean()


@torch.no_grad()
def sinkhorn_knopp(logits: Tensor, temp: float, iters: int = 3) -> Tensor:
    # DINOv2 SinkhornKnoppTeacher (single gpu): rows = samples
    q = torch.exp((logits.float() - logits.float().max()) / temp).t()
    k, b = q.shape
    q /= q.sum()
    for _ in range(iters):
        q /= q.sum(dim=1, keepdim=True)
        q /= k
        q /= q.sum(dim=0, keepdim=True)
        q /= b
    q *= b
    return q.t()


def sigreg(z: Tensor, generator: torch.Generator, slices: int, knots: int, t_max: float) -> Tensor:
    # LeJEPA SIGReg: Epps-Pulley statistic of 1-d projections on random unit directions against
    # N(0, 1), trapezoid quadrature of |ecf - exp(-t^2/2)|^2 exp(-t^2/2) over [-t_max, t_max], x N
    z = z.float()
    n, d = z.shape
    a = torch.randn(d, slices, device=z.device, generator=generator)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, knots, device=z.device)
    phi = torch.exp(-0.5 * t.square())
    xt = (z @ a).unsqueeze(-1) * t  # n, slices, knots
    re, im = torch.cos(xt).mean(0), torch.sin(xt).mean(0)
    err = ((re - phi).square() + im.square()) * phi
    return (torch.trapezoid(err, t, dim=-1) * n).mean()


# ---------------------------------------------------------------------------------------------
# modules


class ProjHead(nn.Module):
    # DINO head: mlp -> l2-normalized bottleneck -> prototypes with unit-norm rows (weight norm, g = 1)
    def __init__(self, in_dim: int, out_dim: int, hidden: int, bottleneck: int, layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [bottleneck]
        mods = []
        for ii in range(layers):
            mods.append(nn.Linear(dims[ii], dims[ii + 1]))
            if ii < layers - 1:
                mods.append(nn.GELU())
        self.mlp = nn.Sequential(*mods)
        self.last = nn.Linear(bottleneck, out_dim, bias=False)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        return F.linear(x, F.normalize(self.last.weight, dim=-1))


class MLPProj(nn.Module):
    # LeJEPA projector: linear-BN-GELU x (layers - 1), linear
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        mods = []
        for ii in range(layers):
            mods.append(nn.Linear(dims[ii], dims[ii + 1]))
            if ii < layers - 1:
                mods += [nn.BatchNorm1d(dims[ii + 1]), nn.GELU()]
        self.net = nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TeacherEncoder(nn.Module):
    # frozen EMA copy of the student encoder (after obj-dino's TeacherEncoder) plus copies of the
    # objective's projection heads, under heads.<name>
    def __init__(self, model, heads: dict[str, nn.Module] | None = None):
        super().__init__()
        self.class_tokens = model.class_tokens
        self.patch_embed = copy.deepcopy(model.patch_embed)
        self.pos_embed = copy.deepcopy(model.pos_embed)
        self.cls_token = copy.deepcopy(model.cls_token) if model.class_tokens else None
        self.blocks = copy.deepcopy(model.blocks)
        self.norm = copy.deepcopy(model.norm)
        self.heads = nn.ModuleDict({k: copy.deepcopy(v) for k, v in (heads or {}).items()})
        self.requires_grad_(False)

    def forward(self, x: Tensor, coord: Tensor, layers: int = 0, norm: str = "layer"):
        # (final normed tokens, target = mean of the last `layers` block outputs each normed), cls
        # dropped. layers 0: no target
        B = x.shape[0]
        x = self.pos_embed(x, coord)
        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        outs = []
        for ii, block in enumerate(self.blocks):
            x = block(x)
            if layers and ii >= len(self.blocks) - layers:
                y = x.float()
                if norm == "instance":
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
                else:
                    y = F.layer_norm(y, y.shape[-1:])
                outs.append(y)
        final = self.norm(x)[:, self.class_tokens :]
        if not layers:
            return final
        target = sum(outs) / len(outs)
        if norm == "instance":
            # data2vec 2.0: IN per layer -> average -> LN (layer_norm_targets)
            target = F.layer_norm(target, target.shape[-1:])
        if layers == 1 and norm == "layer":
            # I-JEPA: layer norm of the final normed output
            target = F.layer_norm(final.float(), final.shape[-1:])
        return final, target[:, self.class_tokens :]


class Predictor(nn.Module):
    # narrow transformer predictor (I-JEPA): context tokens + positioned mask tokens -> targets
    def __init__(self, grid, in_dim: int, dim: int, depth: int, heads: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.pos = SeparablePosEmbed3D(grid, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, qkv_bias=True, proj_bias=True, mlp_ratio=4) for _ in range(depth)]
        )
        self.norm = LayerNorm(dim)
        self.out = nn.Linear(dim, in_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ctx: Tensor, ctx_coord: Tensor, tgt_coord: Tensor) -> Tensor:
        B, M = tgt_coord.shape[:2]
        x = self.pos(self.inp(ctx), ctx_coord)
        m = self.pos(self.mask_token.expand(B, M, -1).to(x.dtype), tgt_coord)
        x = torch.cat([x, m], dim=1)
        for block in self.blocks[:-1]:
            x = block(x)
        x = self.blocks[-1].forward_last_rows(x, M)
        return self.out(self.norm(x))


# ---------------------------------------------------------------------------------------------
# native tokens and views


class NativeTokens:
    # every token of every image, per spacing group (forward_native's geometry). the image order of
    # all views is the concatenation of the groups' image ids (self.order)

    def __init__(self, model, images, masks, spacing, partners=None):
        token_mm = model.patch_size[0] * model.target_spacing
        groups: dict[tuple, list[int]] = {}
        for ii, sp in enumerate(spacing):
            groups.setdefault(tuple(sp), []).append(ii)
        device = images[0].device
        self.device = device
        self.groups = []
        nch = 2 if partners is None else 4
        for sp, ids in groups.items():
            patch, box, grid, coord = native_geometry(sp, tuple(model.grid_size), token_mm, device)
            padded = tuple(num * p for num, p in zip(grid, patch))
            vols = images[ids[0]].new_zeros((nch, len(ids)) + padded)
            for jj, ii in enumerate(ids):
                src, dst = fit_slices(images[ii].shape, box)
                vols[0, jj][dst] = images[ii][src]
                vols[1, jj][dst] = masks[ii][src]
                if partners is not None:
                    vols[2, jj][dst] = partners[0][ii][src]
                    vols[3, jj][dst] = partners[1][ii][src]
            pt = patchify3d(vols.flatten(0, 1), patch).unflatten(0, (nch, len(ids)))
            g = {"sp": sp, "patch": patch, "ids": ids, "grid": grid, "coord": coord[0]}
            g["img"], g["msk"] = pt[0], pt[1]
            g["valid"] = pt[1].mean(dim=-1) >= model.min_mask_frac
            if partners is not None:
                g["pimg"], g["pmsk"] = pt[2], pt[3]
                g["pvalid"] = pt[3].mean(dim=-1) >= model.min_mask_frac
            self.groups.append(g)
        self.order = [ii for g in self.groups for ii in g["ids"]]
        self.B = len(self.order)
        self.native_sps = [g["sp"] for g in self.groups if g["sp"] != (1.0, 1.0, 1.0)]

    def embed_weights(self, patch_embed) -> dict:
        if not self.native_sps:
            return {}
        patches = [g["patch"] for g in self.groups if g["sp"] != (1.0, 1.0, 1.0)]
        return dict(zip(self.native_sps, patch_embed.native_weights(self.native_sps, patches)))

    def head_weights(self, head) -> dict:
        if not self.native_sps:
            return {}
        patches = [g["patch"] for g in self.groups if g["sp"] != (1.0, 1.0, 1.0)]
        return dict(zip(self.native_sps, head.native_weights(self.native_sps, patches)))

    def gather(self, idx: list[Tensor], src: str = "img") -> dict:
        # idx per group [b, V, k] token ids -> view: per group patches / mask patches [b V, k, P],
        # coords [B V, k, 3] (group order, image major, view minor)
        out = {"img": [], "msk": [], "coord": [], "rows": []}
        for g, ii in zip(self.groups, idx):
            b = ii.shape[0]
            bi = torch.arange(b, device=ii.device)[:, None, None]
            img = g["img" if src == "img" else "pimg"]
            msk = g["msk" if src == "img" else "pmsk"]
            out["img"].append(img[bi, ii].flatten(0, 1))
            out["msk"].append(msk[bi, ii].flatten(0, 1))
            out["coord"].append(g["coord"][ii].flatten(0, 1))
            out["rows"].append(b * ii.shape[1])
        out["coord"] = torch.cat(out["coord"]) if len(idx) > 1 else out["coord"][0].contiguous()
        return out

    def embed(self, patch_embed, view: dict, weights: dict) -> Tensor:
        xs = [
            patch_embed.native(x, g["sp"], g["patch"], weights.get(g["sp"]))
            for g, x in zip(self.groups, view["img"])
        ]
        return torch.cat(xs) if len(xs) > 1 else xs[0].contiguous()


def _rand_centres(valid: Tensor, num: int) -> Tensor:
    # [b, N] -> [b, num] random valid token ids
    r = torch.rand(valid.shape[0], num, valid.shape[1], device=valid.device)
    return (r * valid[:, None].float()).argmax(dim=-1)


def _dist(coord: Tensor, centres: Tensor, valid: Tensor) -> Tensor:
    # [b, V, N] distances (token units) to the centres; invalid tokens pushed past every valid one
    c = coord[centres]  # b V 3
    d = (coord[None, None] - c[:, :, None]).norm(dim=-1)
    return d + (~valid)[:, None].float() * 1e4


def _region(d: Tensor, valid: Tensor, scale: tuple[float, float]) -> tuple[Tensor, Tensor]:
    # the nearest scale x (brain tokens) tokens to each centre: (in region [b V N], rank [b V N])
    rank = d.argsort(dim=-1).argsort(dim=-1)
    nval = valid.sum(dim=-1).float()[:, None]
    frac = torch.empty(d.shape[:2], device=d.device).uniform_(*scale)
    return rank < (frac * nval)[..., None], rank


def crop_ids(g: dict, num: int, k: int, scale: tuple[float, float], valid_key="valid") -> Tensor:
    # global crop: k tokens drawn at random from the region; past the region the nearest first
    valid = g[valid_key]
    d = _dist(g["coord"], _rand_centres(valid, num), valid)
    inreg, rank = _region(d, valid, scale)
    score = torch.where(inreg, torch.rand_like(d), 1.0 + rank.float() / d.shape[-1])
    return score.topk(k, dim=-1, largest=False).indices


def local_ids(g: dict, num: int, k: int) -> Tensor:
    # local crop: the k brain tokens nearest a random brain centre (a dense ~ball)
    valid = g["valid"]
    d = _dist(g["coord"], _rand_centres(valid, num), valid)
    return (d + 1e-3 * torch.rand_like(d)).topk(k, dim=-1, largest=False).indices


def block_mask_ids(
    g: dict, num: int, k_ctx: int, k_tgt: int, blocks: int, block_scale, ctx_scale
) -> tuple[Tensor, Tensor]:
    # I-JEPA multi-block masking on the 3D token grid: targets = k_tgt random tokens of the union of
    # `blocks` balls (each block_scale of the brain); context = k_ctx random tokens of a ctx_scale
    # region outside the targets. blocks 0: random masking (data2vec / MAE)
    valid = g["valid"]
    b, n = valid.shape
    if blocks:
        d = _dist(g["coord"], _rand_centres(valid, num * blocks), valid)
        inb, _ = _region(d, valid, block_scale)
        in_t = inb.view(b, num, blocks, n).any(dim=2) & valid[:, None]
    else:
        in_t = valid[:, None].expand(b, num, n)
    rnd = torch.rand(b, num, n, device=valid.device)
    tgt = torch.where(in_t, rnd, 2.0 + rnd).topk(k_tgt, dim=-1, largest=False).indices
    taken = torch.zeros(b, num, n, dtype=torch.bool, device=valid.device).scatter_(-1, tgt, True)
    d = _dist(g["coord"], _rand_centres(valid, num), valid)
    in_c, rank = _region(d, valid, ctx_scale)
    rnd = torch.rand(b, num, n, device=valid.device)
    score = torch.where(in_c & ~in_t, rnd, torch.where(valid[:, None] & ~in_t, 1.0 + rank / n, 3.0 + rnd))
    score = torch.where(in_t & ~taken & valid[:, None], 2.5 + rnd, score)
    score = score.masked_fill(taken, 10.0)
    ctx = score.topk(k_ctx, dim=-1, largest=False).indices
    return ctx, tgt


def view_ball_mask(coord: Tensor, prob: float, ratio: tuple[float, float]) -> Tensor:
    # iBOT mask on a view [R, k, 3]: in a `prob` fraction of the rows, the ratio x k tokens nearest
    # a random token of the row (a 3D block), else nothing
    R, k = coord.shape[:2]
    c = coord[torch.arange(R, device=coord.device), torch.randint(0, k, (R,), device=coord.device)]
    d = (coord - c[:, None]).norm(dim=-1)
    rank = d.argsort(dim=-1).argsort(dim=-1)
    r = torch.empty(R, 1, device=coord.device).uniform_(*ratio)
    on = torch.rand(R, 1, device=coord.device) < prob
    return (rank < (r * k)) & on


# ---------------------------------------------------------------------------------------------
# objectives


class Objective(nn.Module):
    uses_partner = False

    def __init__(self, model, cfg: dict, name: str, aux_mae_weight: float):
        super().__init__()
        self.cfg, self.name, self.aux_mae_weight = cfg, name, aux_mae_weight
        self.progress = 0.0
        self.teacher: TeacherEncoder | None = None
        self._ema_pairs = None
        self.logs: dict[str, Tensor] = {}

    # ---- teacher
    def momentum(self) -> float:
        m0, m1 = self.cfg["momentum"]
        return m1 - (m1 - m0) * (math.cos(math.pi * min(max(self.progress, 0.0), 1.0)) + 1) / 2

    @torch.no_grad()
    def ema_update(self, model) -> None:
        if self.teacher is None:
            return
        if self._ema_pairs is None:
            student = dict(model.named_parameters())
            tp, sp = [], []
            for name, p in self.teacher.named_parameters():
                if name.startswith("heads."):
                    _, head, rest = name.split(".", 2)
                    src = dict(getattr(self, head).named_parameters())[rest]
                else:
                    src = student[name]
                tp.append(p)
                sp.append(src)
            self._ema_pairs = (tp, sp)
        tp, sp = self._ema_pairs
        torch._foreach_lerp_(tp, [p.detach() for p in sp], 1.0 - self.momentum())

    def compile(self, **kwargs):
        if self.teacher is not None:
            self.teacher.forward = torch.compile(self.teacher.forward, **kwargs)

    # ---- shared student pieces
    def level_terms(self, model, toks: NativeTokens, view: dict, out: Tensor, keep: Tensor | None):
        # level head BCE on the student's (unmasked) view tokens, all groups: (num, den)
        if model.level_head is None:
            return None
        num = den = 0.0
        rows = torch.split(out, view["rows"])
        keeps = torch.split(keep, view["rows"]) if keep is not None else [None] * len(rows)
        for g, tok, x, m, kp in zip(toks.groups, rows, view["img"], view["msk"], keeps):
            if kp is not None:
                m = m * kp[..., None].to(m.dtype)
            n, d = model.level_terms(tok, x, m, g["patch"])
            num, den = num + n, den + d
        return num / den.clamp_min(1e-6)

    def mae_loss(self, model, toks: NativeTokens, enc: Tensor, tview: dict, head_ws: dict) -> Tensor:
        # MAE pixel loss: the model's decoder from encoder outputs `enc` to the target view tokens
        feats = model.decode_features(enc, tview["coord"])
        num, den = [], []
        for g, f, tgt, m in zip(toks.groups, torch.split(feats, tview["rows"]), tview["img"], tview["msk"]):
            pred = model.decoder_head.native(f, g["sp"], g["patch"], head_ws.get(g["sp"]))
            w = m / pred.shape[-1]
            num.append((w * (pred.float() - tgt.float()).square()).sum())
            den.append(w.sum())
        loss = sum(num) / sum(den).clamp_min(1e-6)
        residuals = (model.patch_embed.residual, model.decoder_head.residual, model.decoder_head.bias_residual)
        return loss + 0.0 * sum(r.sum() for r in residuals)

    def pooled(self, out: Tensor) -> Tensor:
        return out.float().mean(dim=1)

    def add_log(self, **kw):
        for k, v in kw.items():
            self.logs[k] = v.detach() if torch.is_tensor(v) else torch.tensor(float(v))


class DINO3D(Objective):
    def __init__(self, model, cfg, name, aux_mae_weight):
        super().__init__(model, cfg, name, aux_mae_weight)
        D = model.embed_dim
        c = cfg
        self.dino_head = ProjHead(D, c["dino_dim"], c["head_hidden"], c["head_bottleneck"], c["head_layers"])
        heads = {"dino_head": self.dino_head}
        if c["ibot_weight"]:
            if c["ibot_separate_head"]:
                self.ibot_head = ProjHead(D, c["ibot_dim"], c["head_hidden"], c["head_bottleneck"], c["head_layers"])
                heads["ibot_head"] = self.ibot_head
            self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("center", torch.zeros(1, c["dino_dim"]))
        self.register_buffer("ibot_center", torch.zeros(1, c["ibot_dim"]))
        self.teacher = TeacherEncoder(model, heads)
        self.gram_teacher = TeacherEncoder(model) if c["gram_weight"] else None
        self.register_buffer("gram_on", torch.zeros((), dtype=torch.bool))

    def compile(self, **kwargs):
        super().compile(**kwargs)
        if self.gram_teacher is not None:
            self.gram_teacher.forward = torch.compile(self.gram_teacher.forward, **kwargs)

    def ibot_head_(self):
        return self.ibot_head if self.cfg["ibot_separate_head"] else self.dino_head

    def teacher_temp(self) -> float:
        t0, t1 = self.cfg["temp_t"]
        w = self.cfg["temp_t_warmup"]
        return t1 if self.progress >= w else t0 + (t1 - t0) * self.progress / max(w, 1e-6)

    @torch.no_grad()
    def teacher_probs(self, logits: Tensor, center_name: str) -> Tensor:
        tt = self.teacher_temp()
        if self.cfg["centering"] == "sinkhorn":
            return sinkhorn_knopp(logits, tt, self.cfg["sk_iters"])
        center = getattr(self, center_name)
        probs = F.softmax((logits.float() - center) / tt, dim=-1)
        m = self.cfg["center_momentum"]
        center.mul_(m).add_(logits.float().mean(dim=0, keepdim=True), alpha=1 - m)
        return probs

    @torch.no_grad()
    def maybe_start_gram(self):
        if self.gram_teacher is None or bool(self.gram_on) or self.progress < self.cfg["gram_start"]:
            return
        state = {k: v for k, v in self.teacher.state_dict().items() if not k.startswith("heads.")}
        self.gram_teacher.load_state_dict(state)
        self.gram_on.fill_(True)

    def forward(self, model, images, masks, spacing, partners=None, log=False):
        c = self.cfg
        self.logs = {}
        toks = NativeTokens(model, images, masks, spacing, partners)
        B, kg, kl, V = toks.B, c["global_tokens"], c["num_local"] and c["local_tokens"], c["num_local"]
        aux = self.aux_mae_weight > 0
        km = c["mae_tokens"] if aux else 0
        cross = partners is not None and c["cross_view_global"]
        # global views (+ aux MAE targets from the same region), then local views
        g_ids = []
        for g in toks.groups:
            if cross:
                a = crop_ids(g, 1, kg + km, c["global_scale"])
                p = crop_ids(g, 1, kg + km, c["global_scale"], valid_key="pvalid")
                g_ids.append((a, p))
            else:
                g_ids.append(crop_ids(g, 2, kg + km, c["global_scale"]))
        if cross:
            va = toks.gather([a[..., :kg] for a, _ in g_ids])
            vp = toks.gather([p[..., :kg] for _, p in g_ids], src="pimg")
            gview = interleave_views(va, vp)
        else:
            gview = toks.gather([ii[..., :kg] for ii in g_ids])
        s_w = toks.embed_weights(model.patch_embed)
        x = toks.embed(model.patch_embed, gview, s_w)  # [2B, kg, D]
        mask = None
        if c["ibot_weight"]:
            mask = view_ball_mask(gview["coord"], c["ibot_prob"], c["ibot_ratio"])
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        s_glob = model.encode_tokens(x, gview["coord"])[:, model.class_tokens :]
        s_loc = None
        if V:
            lview = toks.gather([local_ids(g, V, kl) for g in toks.groups])
            s_loc = model.encode_tokens(toks.embed(model.patch_embed, lview, s_w), lview["coord"])
            s_loc = s_loc[:, model.class_tokens :]

        with torch.no_grad():
            t_w = toks.embed_weights(self.teacher.patch_embed)
            t_glob = self.teacher(toks.embed(self.teacher.patch_embed, gview, t_w), gview["coord"])
        # DINO: every student view vs the teacher globals (never the same global view)
        t_logits = self.teacher.heads["dino_head"](self.pooled(t_glob))  # [2B, K] (b major, view minor)
        t_probs = self.teacher_probs(t_logits, "center").view(B, 2, -1)
        s_g = self.dino_head(self.pooled(s_glob)).float().view(B, 2, -1)
        logp_g = F.log_softmax(s_g / c["temp_s"], dim=-1)
        terms = [-(t_probs[:, 1] * logp_g[:, 0]).sum(-1).mean(), -(t_probs[:, 0] * logp_g[:, 1]).sum(-1).mean()]
        if V:
            s_l = self.dino_head(self.pooled(s_loc)).float().view(B, V, -1)
            logp_l = F.log_softmax(s_l / c["temp_s"], dim=-1)
            for t in range(2):
                terms.append(-(t_probs[:, t, None] * logp_l).sum(-1).mean() * V)
            n_terms = 2 + 2 * V
        else:
            n_terms = 2
        dino = sum(terms) / n_terms
        loss = c["dino_weight"] * dino
        self.add_log(dino=dino)
        if c["ibot_weight"]:
            sel = mask.flatten()
            n_mask = sel.sum()
            t_tok = self.teacher.heads["ibot_head" if c["ibot_separate_head"] else "dino_head"](
                t_glob.flatten(0, 1)[sel]
            )
            t_p = self.teacher_probs(t_tok, "ibot_center")
            s_tok = self.ibot_head_()(s_glob.flatten(0, 1)[sel]).float()
            ibot = -(t_p * F.log_softmax(s_tok / c["temp_s"], dim=-1)).sum(-1)
            ibot = ibot.sum() / n_mask.clamp_min(1)
            loss = loss + c["ibot_weight"] * ibot
            self.add_log(ibot=ibot)
        if c["koleo_weight"]:
            pooled = self.pooled(s_glob).view(B, 2, -1)
            koleo = (koleo_loss(pooled[:, 0]) + koleo_loss(pooled[:, 1])) / 2
            loss = loss + c["koleo_weight"] * koleo
            self.add_log(koleo=koleo)
        if self.gram_teacher is not None:
            self.maybe_start_gram()
            if bool(self.gram_on):
                with torch.no_grad():
                    gw = toks.embed_weights(self.gram_teacher.patch_embed)
                    gt = self.gram_teacher(toks.embed(self.gram_teacher.patch_embed, gview, gw), gview["coord"])
                    gt = F.normalize(gt.float(), dim=-1)
                    g_t = gt @ gt.transpose(1, 2)
                sn = F.normalize(s_glob.float(), dim=-1)
                gram = (sn @ sn.transpose(1, 2) - g_t).square().mean()
                loss = loss + c["gram_weight"] * gram
                self.add_log(gram=gram)
        keep = None if mask is None else ~mask
        lvl = self.level_terms(model, toks, gview, s_glob, keep)
        if lvl is not None:
            loss = loss + model.level_weight * lvl
            self.add_log(level=lvl)
        if aux:
            if cross:
                ta = toks.gather([a[..., kg:] for a, _ in g_ids])
                tp = toks.gather([p[..., kg:] for _, p in g_ids], src="pimg")
                tview = interleave_views(ta, tp)
            else:
                tview = toks.gather([ii[..., kg:] for ii in g_ids])
            mae = self.mae_loss(model, toks, s_glob, tview, toks.head_weights(model.decoder_head))
            loss = loss + self.aux_mae_weight * mae
            self.add_log(mae=mae)
        if log:
            self.add_log(**feature_stats(self.pooled(s_glob).view(B, 2, -1)[:, 0]))
            self.add_log(t_rankme=rankme(self.pooled(t_glob).view(B, 2, -1)[:, 0]), mom=self.momentum(),
                         t_temp=self.teacher_temp())
        return loss


def interleave_views(va: dict, vp: dict) -> dict:
    # two single-view gathers (anchor, partner) -> one 2-view gather, image major, view minor
    out = {"img": [], "msk": [], "rows": []}
    for a, p, ma, mp, r in zip(va["img"], vp["img"], va["msk"], vp["msk"], va["rows"]):
        out["img"].append(torch.stack([a, p], 1).flatten(0, 1))
        out["msk"].append(torch.stack([ma, mp], 1).flatten(0, 1))
        out["rows"].append(2 * r)
    out["coord"] = torch.stack([va["coord"], vp["coord"]], 1).flatten(0, 1)
    return out


class IJEPA(Objective):
    # also data2vec2 (random masks, instance-normed top-K targets, mse)
    def __init__(self, model, cfg, name, aux_mae_weight):
        super().__init__(model, cfg, name, aux_mae_weight)
        D = model.embed_dim
        self.predictor = Predictor(model.patch_grid_size, D, cfg["pred_dim"], cfg["pred_depth"], cfg["pred_heads"])
        self.teacher = TeacherEncoder(model)

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.predictor.forward = torch.compile(self.predictor.forward, **kwargs)

    def forward(self, model, images, masks, spacing, partners=None, log=False):
        c = self.cfg
        self.logs = {}
        toks = NativeTokens(model, images, masks, spacing)
        S, kc, kt = c["masks"], c["context_tokens"], c["target_tokens"]
        ids = [
            block_mask_ids(g, S, kc, kt, c["num_blocks"], c["block_scale"], c["context_scale"])
            for g in toks.groups
        ]
        cview = toks.gather([cc for cc, _ in ids])
        tview = toks.gather([tt for _, tt in ids])
        s_w = toks.embed_weights(model.patch_embed)
        ctx = model.encode_tokens(toks.embed(model.patch_embed, cview, s_w), cview["coord"])
        ctx = ctx[:, model.class_tokens :]
        with torch.no_grad():
            # the teacher encodes context + targets together (the masked image, sparsely sampled)
            t_w = toks.embed_weights(self.teacher.patch_embed)
            both = {
                "img": [torch.cat([a, b], 1) for a, b in zip(cview["img"], tview["img"])],
                "rows": cview["rows"],
                "coord": torch.cat([cview["coord"], tview["coord"]], 1),
            }
            final, target = self.teacher(
                toks.embed(self.teacher.patch_embed, both, t_w), both["coord"], c["target_layers"],
                c["target_norm"],
            )
            target = target[:, kc:]
        pred = self.predictor(ctx, cview["coord"], tview["coord"]).float()
        # tokens weighted by their in-mask fraction (voxel counts differ between groups)
        w = torch.cat([m.float().mean(-1) for m in tview["msk"]])
        if c["loss"] == "smooth_l1":
            err = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
        else:
            err = (pred - target).square().mean(-1)
        jepa = (w * err).sum() / w.sum().clamp_min(1e-6)
        loss = jepa
        self.add_log(jepa=jepa)
        lvl = self.level_terms(model, toks, cview, ctx, None)
        if lvl is not None:
            loss = loss + model.level_weight * lvl
            self.add_log(level=lvl)
        if self.aux_mae_weight > 0:
            # MAE on the same targets (subset of mae_tokens) through the model's pixel decoder
            mae = self.mae_loss(model, toks, ctx, tview, toks.head_weights(model.decoder_head))
            loss = loss + self.aux_mae_weight * mae
            self.add_log(mae=mae)
        if log:
            B = toks.B
            self.add_log(**feature_stats(self.pooled(ctx).view(B, S, -1)[:, 0]))
            self.add_log(t_rankme=rankme(self.pooled(final[:, :kc]).view(B, S, -1)[:, 0]), mom=self.momentum(),
                         pred_std=pred.std(dim=(0, 1)).mean(), tgt_std=target.std(dim=(0, 1)).mean())
        return loss


class LeJEPA(Objective):
    def __init__(self, model, cfg, name, aux_mae_weight):
        super().__init__(model, cfg, name, aux_mae_weight)
        self.proj = MLPProj(model.embed_dim, cfg["proj_hidden"], cfg["proj_dim"], cfg["proj_layers"])
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self._gen = None

    def forward(self, model, images, masks, spacing, partners=None, log=False):
        c = self.cfg
        self.logs = {}
        toks = NativeTokens(model, images, masks, spacing, partners)
        B, kg, V = toks.B, c["global_tokens"], c["num_local"]
        km = c["mae_tokens"] if self.aux_mae_weight > 0 else 0
        cross = partners is not None
        if cross:
            g_ids = [(crop_ids(g, 1, kg + km, c["global_scale"]),
                      crop_ids(g, 1, kg + km, c["global_scale"], valid_key="pvalid")) for g in toks.groups]
            gview = interleave_views(toks.gather([a[..., :kg] for a, _ in g_ids]),
                                     toks.gather([p[..., :kg] for _, p in g_ids], src="pimg"))
        else:
            g_ids = [crop_ids(g, 2, kg + km, c["global_scale"]) for g in toks.groups]
            gview = toks.gather([ii[..., :kg] for ii in g_ids])
        s_w = toks.embed_weights(model.patch_embed)
        s_glob = model.encode_tokens(toks.embed(model.patch_embed, gview, s_w), gview["coord"])
        s_glob = s_glob[:, model.class_tokens :]
        # one batch through the projector (BN statistics over all views)
        pooled = [self.pooled(s_glob)]
        if V:
            lview = toks.gather([local_ids(g, V, c["local_tokens"]) for g in toks.groups])
            s_loc = model.encode_tokens(toks.embed(model.patch_embed, lview, s_w), lview["coord"])
            pooled.append(self.pooled(s_loc[:, model.class_tokens :]))
        z = self.proj(torch.cat(pooled)).float()
        zg = z[: 2 * B].view(B, 2, -1)
        za = [zg[:, 0], zg[:, 1]]
        if V:
            zl = z[2 * B :].view(B, V, -1)
            za += [zl[:, v] for v in range(V)]
        za = torch.stack(za)  # views, B, K
        centre = zg.mean(dim=1)
        pred = (centre[None] - za).square().mean()
        if self._gen is None or self._gen.device != z.device:
            self._gen = torch.Generator(device=z.device)
        self._gen.manual_seed(int(self.step))
        self.step += 1
        sig = torch.stack([sigreg(zv, self._gen, c["slices"], c["knots"], c["t_max"]) for zv in za]).mean()
        loss = (1 - c["lamb"]) * pred + c["lamb"] * sig
        self.add_log(pred=pred, sigreg=sig)
        lvl = self.level_terms(model, toks, gview, s_glob, None)
        if lvl is not None:
            loss = loss + model.level_weight * lvl
            self.add_log(level=lvl)
        if km:
            if cross:
                tview = interleave_views(toks.gather([a[..., kg:] for a, _ in g_ids]),
                                         toks.gather([p[..., kg:] for _, p in g_ids], src="pimg"))
            else:
                tview = toks.gather([ii[..., kg:] for ii in g_ids])
            mae = self.mae_loss(model, toks, s_glob, tview, toks.head_weights(model.decoder_head))
            loss = loss + self.aux_mae_weight * mae
            self.add_log(mae=mae)
        if log:
            self.add_log(**feature_stats(self.pooled(s_glob).view(B, 2, -1)[:, 0]))
            self.add_log(z_rankme=rankme(zg[:, 0]))
        return loss


class TrainParams(nn.Module):
    # the optimizer's view of model + objective (names model.* / objective.*); checkpoints still save
    # the model and the objective separately
    def __init__(self, model, objective):
        super().__init__()
        self.model = model
        self.objective = objective
