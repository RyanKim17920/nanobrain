# Natural-image pretrained init for the 3D encoder (operator ruling 2026-11-08: "natural image
# model is ok"; the README bans weights pretrained on brain MRI only, this is the operator's
# reading of that rule). Attention and MLP act on token vectors, so the 12 blocks and the final
# norm of a 2D ViT-B (width 768, 12 heads) copy over as is. The 2D patch embedding inflates
# I3D-style into the 1 mm canonical NativeEmbed kernel. Position embeddings stay random.
#
# Supported checkpoints (local safetensors, compute nodes have no network):
#   mae:  facebook/vit-mae-base (HF ViTMAEForPreTraining, separate q/k/v)
#   timm: any timm-layout ViT-B (fused qkv): dinov2 (+reg4), dinov3, siglip, clip, ibot (converted
#         from the bytedance teacher .pth, "backbone." prefix stripped).
#         LayerScale (ls1.gamma / gamma_1) folds into proj / fc2; a missing qkv bias (dinov3) is
#         zero. Not carried over (reported): registers, cls token, pos embed / RoPE (dinov3 is a
#         RoPE model, its attention was trained with rotary positions; ours get additive random
#         pos), clip norm_pre (ln_pre before block 0). Activation: ours is exact GELU, as are the
#         timm siglip v2 / clip laion2b / ibot / dino ones (openai clip uses quick_gelu: not used).
#         Checked 11-09 on CPU: every block and the final norm match timm (rope=None for dinov3)
#         to <=1e-4 relative; mae and dinov2 loads are bitwise those of a039eaf.

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def _family(sd: dict) -> str:
    if "vit.encoder.layer.0.attention.attention.query.weight" in sd:
        return "mae"
    if "blocks.0.attn.qkv.weight" in sd:
        return "timm"
    raise ValueError("natinit: unknown checkpoint layout")


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
    p = f"blocks.{ii}."
    q, k, v = sd[p + "attn.qkv.weight"].chunk(3, dim=0)
    qkvb = sd.get(p + "attn.qkv.bias")
    if qkvb is None:  # dinov3: no qkv bias in the checkpoint
        qkvb = torch.zeros(sd[p + "attn.qkv.weight"].shape[0], dtype=q.dtype)
    qb, kb, vb = qkvb.chunk(3, dim=0)
    # LayerScale (dinov2 ls1.gamma, dinov3 gamma_1); absent (siglip, clip, ibot) = 1
    one = torch.ones(sd[p + "attn.proj.weight"].shape[0], dtype=q.dtype)
    g1 = sd.get(p + "ls1.gamma", sd.get(p + "gamma_1", one))
    g2 = sd.get(p + "ls2.gamma", sd.get(p + "gamma_2", one))
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
    sd = load_file(str(path / "model.safetensors") if path.is_dir() else str(path))
    if any(k.startswith("backbone.") for k in sd):  # ibot / dino teacher
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    family = _family(sd)
    report = {"family": family, "path": str(path), "copied": [], "mismatch": [], "skipped": []}

    n_src = 1 + max(
        int(k.split(".")[3 if family == "mae" else 1])
        for k in sd
        if k.startswith("vit.encoder.layer." if family == "mae" else "blocks.")
    )
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
        qk_spec = cfg.get("qk_spec")
        if qk_spec is not None:
            # rescale q (and k) so the block's spectral norm is the target: q.k logits shrink by
            # the product of the two factors (dinov3: rope-trained q/k ~3x dinov2's, blew up at peak lr)
            for nm, tgt in (("q", qk_spec[0]), ("k", qk_spec[1])):
                w = ours[f"attn.{nm}.weight"]
                f = float(tgt) / torch.linalg.matrix_norm(w.float(), 2).item()
                w.mul_(f)
                if f"attn.{nm}.bias" in ours:
                    ours[f"attn.{nm}.bias"].mul_(f)
                report.setdefault("qk_scale", []).append(f"{ii}{nm}:{f:.3f}")
        block.load_state_dict(ours)
    report["copied"].append(f"{min(n_src, len(model.blocks))} blocks (q k v proj fc1 fc2 norm1 norm2)")

    nw, nb = ("vit.layernorm.weight", "vit.layernorm.bias") if family == "mae" else ("norm.weight", "norm.bias")
    if model.norm.weight.shape == sd[nw].shape:
        model.norm.weight.copy_(sd[nw])
        model.norm.bias.copy_(sd[nb])
        report["copied"].append("final norm")
    else:
        report["mismatch"].append(f"final norm {tuple(sd[nw].shape)} vs {tuple(model.norm.weight.shape)}")

    if cfg.embed:
        pw = "vit.embeddings.patch_embeddings.projection." if family == "mae" else "patch_embed.proj."
        w2 = sd[pw + "weight"]
        b2 = sd.get(pw + "bias", torch.zeros(w2.shape[0], dtype=w2.dtype))  # clip: no patch bias
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
    unused = ["cls token", "pos embed" if family == "mae" or "pos_embed" in sd else "RoPE (no pos embed)"]
    if family == "mae":
        unused.append("mae decoder")
    unused += [n for n, key in (("registers", "reg_token"), ("norm_pre (ln_pre)", "norm_pre.weight"),
                                ("attn_pool head", "attn_pool.q.weight"), ("head", "head.weight")) if key in sd]
    if family == "timm" and "blocks.0.attn.qkv.bias" not in sd:
        unused.append("(no qkv bias in src: zero)")
    report["skipped"].append("src unused: " + ", ".join(unused))
    logger.info(f"natinit: {report}")
    return report


def is_natinit_param(name: str) -> bool:
    # the pretrained encoder parameters whose lr natinit.block_lr_mult scales
    return name.startswith("blocks.") or name in ("norm.weight", "norm.bias")
