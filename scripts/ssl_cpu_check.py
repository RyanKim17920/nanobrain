# CPU checks for the ssl objectives. usage: python cpu_check.py MODE [objective]
#   MODE mae_ref    : 4 MAE steps (loss hex) with whatever nanobrain is on PYTHONPATH
#   MODE obj NAME   : forward/backward/no-collapse + checkpoint contract for objective NAME
import os
import sys
import tempfile

os.environ.setdefault("BRAINMARKS_SMRI_DATA", "/tmp")
import numpy as np
import torch
from omegaconf import OmegaConf

import nanobrain.misc as misc
import nanobrain.train as T
from nanobrain.model import ViTMAE3D

torch.manual_seed(0)
GRID = (64, 72, 64)


def make_args(objective="mae", ssl=None, aux=0.0):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(T.__file__), "config/default_pretrain.yaml"))
    cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(os.path.join(os.path.dirname(T.__file__), "config/r12_nat_lvl_mf.yaml")))
    over = dict(grid_size=list(GRID), num_visible=48, num_predict=64, lr=1e-3, compile=False)
    over["model_kwargs"] = dict(depth=2, embed_dim=96, num_heads=4, decoder_depth=2, decoder_embed_dim=64,
                                decoder_num_heads=4, class_tokens=0, min_mask_frac=0.05, level_weight=0.5)
    cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.create(over))
    if "objective" in cfg or objective != "mae":
        cfg.objective = objective
        cfg.aux_mae_weight = aux
        cfg.ssl = ssl or {}
        cfg.cross_sequence_views = bool(os.environ.get("CROSS"))
    return cfg


def batch(seed, n=6):
    g = torch.Generator().manual_seed(seed)
    imgs, masks, sps = [], [], []
    for ii in range(n):
        sp = (1.0, 1.0, 1.0) if ii % 2 == 0 else (1.0, 1.0, 3.0)
        shape = tuple(int(s / q) for s, q in zip((60, 70, 60), sp))
        ax = [torch.linspace(-1, 1, s) for s in shape]
        X, Y, Z = torch.meshgrid(*ax, indexing="ij")
        m = (X**2 / 0.8 + Y**2 / 0.9 + Z**2 / 0.8) < 1
        img = torch.randn(shape, generator=g) * 0.5 + (X + 2 * Y).sin() * 2
        imgs.append(img * m)
        masks.append(m)
        sps.append(sp)
    return imgs, masks, sps


def mae_ref():
    args = make_args()
    model = ViTMAE3D.from_config(args)
    opt = T.make_optimizer(args, model)
    out = []
    for step in range(4):
        imgs, masks, sps = batch(step)
        torch.manual_seed(100 + step)
        if step == 2 and os.environ.get("WITH_STATE"):
            loss, _ = model(imgs, masks, num_visible=48, num_predict=64, num_samples=2, with_state=True, spacing=sps)
        else:
            loss = model(imgs, masks, num_visible=48, num_predict=64, num_samples=2, with_state=False, spacing=sps)
        misc.backward_step(loss, opt, max_norm=1.0)
        out.append(loss.item().hex())
    print("MAE_LOSSES", " ".join(out))
    print("PARAM_SUM", sum(p.double().sum().item() for p in model.parameters()).hex())


def obj(name, ssl, aux):
    from nanobrain.objectives import TrainParams, build_objective, feature_stats

    args = make_args(name, ssl, aux)
    model = ViTMAE3D.from_config(args)
    objective = build_objective(args, model)
    opt = T.make_optimizer(args, TrainParams(model, objective))
    n_muon = sum(len(g["params"]) for g in opt.optimizers[1].param_groups)
    logs = []
    for step in range(8):
        imgs, masks, sps = batch(step)
        objective.progress = step / 8
        partners = None
        if args.get("cross_sequence_views"):
            partners = ([i.roll(1, 0) for i in imgs], masks)
        loss = objective(model, imgs, masks, sps, partners, log=True)
        assert torch.isfinite(loss), loss
        gn = misc.backward_step(loss, opt, max_norm=1.0)
        objective.ema_update(model)
        logs.append({k: round(float(v), 4) for k, v in objective.logs.items()} | {"loss": round(loss.item(), 4), "gn": round(float(gn), 3)})
    for l in logs[:: 3] + logs[-1:]:
        print(name, l)
    last = logs[-1]
    assert last["feat_std"] > 1e-3 and last["rankme"] > 1.5, f"collapse {last}"
    # teacher really EMA-tracks (differs from student, but moved)
    if objective.teacher is not None:
        d = (objective.teacher.blocks[0].attn.q.weight - model.blocks[0].attn.q.weight).abs().max().item()
        print(name, "teacher-student max diff", round(d, 6))
        assert d > 0
    # checkpoint: save with the objective, reload through the brainmarks contract
    with tempfile.TemporaryDirectory() as tmp:
        args.output_dir = tmp
        misc.save_model(args, 8, model, opt, None, 1.0, objective=objective)
        path = os.path.join(tmp, "checkpoint-last.pth")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        assert "objective" in ck and not any(k.startswith(("teacher", "objective")) for k in ck["model"])
        from nanobrain.model import vit_mae_3d

        ev = vit_mae_3d(path)  # strict load_state_dict
        ev.eval()
        img, m, _ = batch(99, 1)
        affine = np.eye(4)
        out = ev.forward_embeddings([{"values": img[0][m[0]], "mask": m[0], "affine": affine}], return_dense=True)
        ge = out[0]["global_embedding"]
        assert torch.isfinite(ge).all() and ge.shape == (96,)
        # resume path: objective state round trip
        model2 = ViTMAE3D.from_config(args)
        obj2 = build_objective(args, model2)
        opt2 = T.make_optimizer(args, TrainParams(model2, obj2))
        args.resume = True
        misc.load_model(args, model2, opt2, None, objective=obj2)
        assert all(torch.equal(a, b) for a, b in zip(obj2.state_dict().values(), objective.state_dict().values()))
    print(name, f"OK (muon matrices {n_muon}, eval global {tuple(ge.shape)} dense {tuple(out[0]['dense_embedding'].shape)})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mae_ref":
        mae_ref()
    else:
        name = sys.argv[2]
        small = dict(global_tokens=48, local_tokens=12, num_local=2, mae_tokens=32)
        if name in ("ijepa", "data2vec2"):
            small = dict(context_tokens=48, target_tokens=40, mae_tokens=32, pred_dim=32, pred_depth=2, pred_heads=4)
        if name == "dino3d":
            small.update(dino_dim=256, ibot_dim=256, head_hidden=128, head_bottleneck=32, gram_weight=2.0, gram_start=0.5)
        if name == "lejepa":
            small.update(proj_hidden=128, proj_dim=16)
        aux = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        obj(name, small, aux)
