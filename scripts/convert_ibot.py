"""Convert the public iBOT release checkpoint (bytedance/ibot, e.g. vitb_16_rand_mask/checkpoint.pth,
which holds student + teacher + iBOT head + loss centers) into natinit's local safetensors layout:

  model.safetensors  teacher backbone, timm key layout (blocks.N.attn.qkv ..., norm, patch_embed.proj)
  head.safetensors   teacher iBOT head (mlp.{0,2,4}, last_layer / last_layer2 weight_v; weight_g is
                     1 = norm_last_layer), backbone cls_token, student masked_embed, ibot_loss centers

Loads with weights_only=True (argparse.Namespace + a numpy scalar allow-listed, no code execution).
usage: python -I convert_ibot.py CKPT OUT_DIR
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def main(ckpt: str, out: str) -> None:
    allow = [argparse.Namespace, (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
             type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allow):
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    t, s = ck["teacher"], ck["student"]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = {k[len("backbone."):]: v.clone().contiguous() for k, v in t.items() if k.startswith("backbone.")}
    save_file(backbone, str(out_dir / "model.safetensors"))
    head = {k[len("head."):]: v.clone().contiguous() for k, v in t.items() if k.startswith("head.")}
    for k in [k for k in head if k.endswith("weight_g")]:
        g = head.pop(k)
        assert torch.allclose(g, torch.ones_like(g)), f"{k}: weight_g != 1 (not norm_last_layer)"
    head["cls_token"] = t["backbone.cls_token"].clone().contiguous()
    head["masked_embed"] = s["module.backbone.masked_embed"].clone().contiguous()
    head["center"] = ck["ibot_loss"]["center"].reshape(1, -1).clone().contiguous()
    head["center2"] = ck["ibot_loss"]["center2"].reshape(1, -1).clone().contiguous()
    save_file(head, str(out_dir / "head.safetensors"))
    print("backbone", len(backbone), "head", {k: tuple(v.shape) for k, v in head.items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
