# Natural-image pretrained init for the 3D encoder (operator ruling 2026-11-08: "natural image
# model is ok"; the README bans weights pretrained on brain MRI only, this is the operator's
# reading of that rule). Attention and MLP act on token vectors, so the 12 blocks and the final
# norm of a 2D ViT-B (width 768, 12 heads) copy over as is. The 2D patch embedding inflates
# I3D-style into the 1 mm canonical NativeEmbed kernel. Position embeddings stay random.
#
# Supported checkpoints (local safetensors, compute nodes have no network):
#   mae:    facebook/vit-mae-base (HF ViTMAEForPreTraining, separate q/k/v)
#   dinov2: timm/vit_base_patch14_dinov2.lvd142m (fused qkv, LayerScale folded into proj / fc2)
#   mmpretrain: OpenMMLab mmpretrain / mmselfsup ViT-B/16 pretrain .pth (Apache-2.0: MAE 1600e,
#           PixMIM 800e, MFF 800e). backbone.layers.N.{ln1, attn.qkv (fused), attn.proj, ln2,
#           ffn.layers.0.0 (fc1), ffn.layers.1 (fc2)}, final backbone.ln1; the neck decoder,
#           cls token, sin-cos pos embed and MFF's proj_layers / proj_weights are not used.
#           natinit.path: the .pth file, or a dir holding exactly one .pth (and no model.safetensors)

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# mmengine training checkpoints pickle their message_hub (HistoryBuffer + numpy arrays) next to
# the state dict. torch.load(weights_only=True) refuses those globals; these inert stand-ins
# (no code of mmengine / numpy runs) let the weights-only unpickler skip over them. Only these
# names are allowed; any other unknown global still fails the load.
_MM_INERT_GLOBALS = (
    "mmengine.logging.history_buffer.HistoryBuffer",
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy.ndarray",
    "numpy.dtype",
    "numpy.dtypes.Float64DType",
    "numpy.dtypes.Int64DType",
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
)


class _Inert:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _load_pth(path: Path) -> dict:
    inert = [(type(n.rsplit(".", 1)[-1], (_Inert,), {}), n) for n in _MM_INERT_GLOBALS]
    with torch.serialization.safe_globals(inert):
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def _load_state_dict(path: Path) -> dict:
    if path.is_dir():
        if (path / "model.safetensors").exists():
            return load_file(str(path / "model.safetensors"))
        pths = sorted(path.glob("*.pth"))
        if len(pths) != 1:
            raise ValueError(f"natinit: {path} has no model.safetensors and {len(pths)} .pth files")
        return _load_pth(pths[0])
    if path.suffix in (".pth", ".pt"):
        return _load_pth(path)
    return load_file(str(path))


def _family(sd: dict) -> str:
    if "vit.encoder.layer.0.attention.attention.query.weight" in sd:
        return "mae"
    if "blocks.0.attn.qkv.weight" in sd:
        return "dinov2"
    if "backbone.layers.0.attn.qkv.weight" in sd:
        return "mmpretrain"
    raise ValueError("natinit: unknown checkpoint layout")


# per family: block key prefix, index of the block number in a dotted key, final norm prefix,
# 2D patch embed prefix
_LAYOUT = {
    "mae": ("vit.encoder.layer.", 3, "vit.layernorm.", "vit.embeddings.patch_embeddings.projection."),
    "dinov2": ("blocks.", 1, "norm.", "patch_embed.proj."),
    "mmpretrain": ("backbone.layers.", 2, "backbone.ln1.", "backbone.patch_embed.projection."),
}


def _block_tensors(sd: dict, family: str, ii: int) -> dict[str, torch.Tensor]:
    # our Block's state dict keys -> tensors
    if family == "mae":
        p = f"vit.encoder.layer.{ii}."
        return {
            "norm1.weight": sd[p + "layernorm_before.weight"],
            "norm1.bias": sd[p + "layernorm_before.bias"],
            "attn.q.weight": sd[p + "attention.attention.query.weight"],
            "attn.q.bias": sd[p + "attention.attention.query.bias"],
            "attn.k.weight": sd[p + "attention.attention.key.weight"],
            "attn.k.bias": sd[p + "attention.attention.key.bias"],
            "attn.v.weight": sd[p + "attention.attention.value.weight"],
            "attn.v.bias": sd[p + "attention.attention.value.bias"],
            "attn.proj.weight": sd[p + "attention.output.dense.weight"],
            "attn.proj.bias": sd[p + "attention.output.dense.bias"],
            "norm2.weight": sd[p + "layernorm_after.weight"],
            "norm2.bias": sd[p + "layernorm_after.bias"],
            "mlp.fc1.weight": sd[p + "intermediate.dense.weight"],
            "mlp.fc1.bias": sd[p + "intermediate.dense.bias"],
            "mlp.fc2.weight": sd[p + "output.dense.weight"],
            "mlp.fc2.bias": sd[p + "output.dense.bias"],
        }
    if family == "mmpretrain":
        p = f"backbone.layers.{ii}."
        # fused qkv: rows [q; k; v], each heads x head_dim (mmpretrain reshapes (B, N, 3, H, d))
        q, k, v = sd[p + "attn.qkv.weight"].chunk(3, dim=0)
        qb, kb, vb = sd[p + "attn.qkv.bias"].chunk(3, dim=0)
        pw, pb = sd[p + "attn.proj.weight"], sd[p + "attn.proj.bias"]
        f2w, f2b = sd[p + "ffn.layers.1.weight"], sd[p + "ffn.layers.1.bias"]
        # LayerScale (absent in these checkpoints; layer_scale_init_value 0) folds like dinov2
        if p + "attn.gamma1.weight" in sd:
            g1 = sd[p + "attn.gamma1.weight"]
            pw, pb = pw * g1[:, None], pb * g1
        if p + "ffn.gamma2.weight" in sd:
            g2 = sd[p + "ffn.gamma2.weight"]
            f2w, f2b = f2w * g2[:, None], f2b * g2
        return {
            "norm1.weight": sd[p + "ln1.weight"],
            "norm1.bias": sd[p + "ln1.bias"],
            "attn.q.weight": q, "attn.q.bias": qb,
            "attn.k.weight": k, "attn.k.bias": kb,
            "attn.v.weight": v, "attn.v.bias": vb,
            "attn.proj.weight": pw,
            "attn.proj.bias": pb,
            "norm2.weight": sd[p + "ln2.weight"],
            "norm2.bias": sd[p + "ln2.bias"],
            "mlp.fc1.weight": sd[p + "ffn.layers.0.0.weight"],
            "mlp.fc1.bias": sd[p + "ffn.layers.0.0.bias"],
            "mlp.fc2.weight": f2w,
            "mlp.fc2.bias": f2b,
        }
    p = f"blocks.{ii}."
    q, k, v = sd[p + "attn.qkv.weight"].chunk(3, dim=0)
    qb, kb, vb = sd[p + "attn.qkv.bias"].chunk(3, dim=0)
    g1, g2 = sd[p + "ls1.gamma"], sd[p + "ls2.gamma"]
    # x + ls1 * proj(a) = x + (g1 W) a + g1 b
    return {
        "norm1.weight": sd[p + "norm1.weight"],
        "norm1.bias": sd[p + "norm1.bias"],
        "attn.q.weight": q, "attn.q.bias": qb,
        "attn.k.weight": k, "attn.k.bias": kb,
        "attn.v.weight": v, "attn.v.bias": vb,
        "attn.proj.weight": sd[p + "attn.proj.weight"] * g1[:, None],
        "attn.proj.bias": sd[p + "attn.proj.bias"] * g1,
        "norm2.weight": sd[p + "norm2.weight"],
        "norm2.bias": sd[p + "norm2.bias"],
        "mlp.fc1.weight": sd[p + "mlp.fc1.weight"],
        "mlp.fc1.bias": sd[p + "mlp.fc1.bias"],
        "mlp.fc2.weight": sd[p + "mlp.fc2.weight"] * g2[:, None],
        "mlp.fc2.bias": sd[p + "mlp.fc2.bias"] * g2,
    }


def inflate_patch_kernel(w2: torch.Tensor, p: int) -> torch.Tensor:
    """[D, C, k, k] 2D patch kernel -> [D, p*p*p] canonical 1 mm kernel (px py pz flattened).

    RGB summed into one channel (a grey image fed as R=G=B gives the same response), the k x k
    kernel sum-pooled to p x p in-plane cells (x <- kernel rows, y <- columns; each 1 mm voxel
    gathers the weights of the pixels it covers, so a smooth input keeps its response), then
    replicated along z and divided by p: an input constant along z gives the 2D response."""
    D, _, k, _ = w2.shape
    w = w2.float().sum(dim=1, keepdim=True)
    # adaptive average pooling handles k not a multiple of p (14 -> 8); x (k/p)^2 keeps the sum
    w = F.adaptive_avg_pool2d(w, (p, p)) * (k * k) / (p * p)
    w = w[:, 0, :, :, None].expand(D, p, p, p) / p
    return w.reshape(D, p * p * p).contiguous()


@torch.no_grad()
def apply_natural_init(model, cfg) -> dict:
    """Copy blocks + final norm (and with cfg.embed the inflated patch embedding) into model."""
    path = Path(cfg.path)
    sd = _load_state_dict(path)
    family = _family(sd)
    report = {"family": family, "path": str(path), "copied": [], "mismatch": [], "skipped": []}
    lprefix, lidx, nprefix, pprefix = _LAYOUT[family]

    n_src = 1 + max(int(k.split(".")[lidx]) for k in sd if k.startswith(lprefix))
    if n_src != len(model.blocks):
        report["mismatch"].append(f"depth {n_src} vs ours {len(model.blocks)}")
    for ii, block in enumerate(model.blocks[: min(n_src, len(model.blocks))]):
        ours = block.state_dict()
        src = _block_tensors(sd, family, ii)
        for key, t in src.items():
            if key not in ours or ours[key].shape != t.shape:
                report["mismatch"].append(
                    f"blocks.{ii}.{key}: src {tuple(t.shape)} ours "
                    f"{tuple(ours[key].shape) if key in ours else None}"
                )
                continue
            ours[key].copy_(t.to(ours[key].dtype))
        extra = sorted(set(ours) - set(src))
        if extra:
            report["skipped"].append(f"blocks.{ii} kept random: {extra}")
        block.load_state_dict(ours)
    report["copied"].append(f"{min(n_src, len(model.blocks))} blocks (q k v proj fc1 fc2 norm1 norm2)")

    nw, nb = nprefix + "weight", nprefix + "bias"
    if model.norm.weight.shape == sd[nw].shape:
        model.norm.weight.copy_(sd[nw])
        model.norm.bias.copy_(sd[nb])
        report["copied"].append("final norm")
    else:
        report["mismatch"].append(f"final norm {tuple(sd[nw].shape)} vs {tuple(model.norm.weight.shape)}")

    if cfg.embed:
        w2, b2 = sd[pprefix + "weight"], sd[pprefix + "bias"]
        p = model.patch_embed.patch_size if hasattr(model.patch_embed, "patch_size") else model.patch_size[0]
        w3 = inflate_patch_kernel(w2, p)
        if w3.shape != model.patch_embed.weight.shape:
            report["mismatch"].append(f"patch embed {tuple(w3.shape)} vs {tuple(model.patch_embed.weight.shape)}")
        else:
            model.patch_embed.weight.copy_(w3)
            model.patch_embed.bias.copy_(b2)
            report["copied"].append(
                f"patch embed inflated {tuple(w2.shape)} -> {tuple(w3.shape)} (rgb sum, pool to {p}x{p}, /{p} along z)"
            )
    else:
        report["skipped"].append("patch embed kept random")

    if cfg.pos_scale != 1.0:
        for wt in (model.pos_embed.weight_x, model.pos_embed.weight_y, model.pos_embed.weight_z):
            wt.mul_(cfg.pos_scale)
    report["skipped"].append(f"pos embed random (x{cfg.pos_scale}), decoder / heads random")
    if family == "mmpretrain":
        n_used = min(n_src, len(model.blocks))
        used = {f"{lprefix}{ii}.{s}" for ii in range(n_used) for s in (
            "ln1.weight", "ln1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
            "ln2.weight", "ln2.bias", "ffn.layers.0.0.weight", "ffn.layers.0.0.bias", "ffn.layers.1.weight",
            "ffn.layers.1.bias", "attn.gamma1.weight", "ffn.gamma2.weight")}
        used |= {nw, nb} | ({pprefix + "weight", pprefix + "bias"} if cfg.embed else set())
        unused = sorted({k.split(".")[1] for k in sd if k.startswith("backbone.") and k not in used})
        report["skipped"].append(
            f"src unused: backbone {unused}, " + ", ".join(sorted({k.split('.')[0] for k in sd} - {'backbone'}))
        )
    else:
        report["skipped"].append(
            "src unused: cls token, pos embed" + (", mae decoder" if family == "mae" else "")
        )
    logger.info(f"natinit: {report}")
    return report


def is_natinit_param(name: str) -> bool:
    # the pretrained encoder parameters whose lr natinit.block_lr_mult scales
    return name.startswith("blocks.") or name in ("norm.weight", "norm.bias")
