import argparse
import datetime
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

import nanobrain.misc as misc
from nanobrain.data import (
    Fomo300kDataset,
    InfiniteSampler,
    brle_to_dense,
    fit_to_shape,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)
from nanobrain.model import ViTMAE3D, patches_to_volume
from nanobrain.visualization import plot_mask_pred

logger = logging.getLogger()

ROOT = Path(__file__).parents[2]
# brainmarks-smri data folder, shared with the eval
DATA_ROOT = Path(os.environ["BRAINMARKS_SMRI_DATA"])
DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"


def main(args: DictConfig):
    # setup
    process_start_time = time.monotonic()
    assert int(os.environ.get("WORLD_SIZE", 1)) == 1, "distributed training not supported"
    device = torch.device(args.device)
    misc.random_seed(args.seed)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    misc.setup_logging(logger, output_dir / "log.txt")

    logger.info("pretraining nanobrain")
    logger.info(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"git: {misc.git_info()}")
    if device.type == "cuda":
        logger.info(f"gpu: {torch.cuda.get_device_name(device)}")
    logger.info(f"torch: {torch.__version__} (cuda {torch.version.cuda})")
    logger.info(f"config:\n{OmegaConf.to_yaml(args)}")

    summary_path = output_dir / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["done"]:
        logger.info(f"training already done, see {summary_path}")
        return

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    misc.snapshot_source(ROOT / "src", output_dir / "source.tar.gz")

    # model
    model = ViTMAE3D.from_config(args)
    model.to(device)
    logger.info(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"num params: {num_params / 1e6:.1f}M")

    batch_flops = count_batch_flops(args, model, device)
    logger.info(f"flops per batch (fwd + bwd): {batch_flops / 1e12:.2f} TFLOP")

    if args.compile:
        # in place compile keeps the state dict keys
        model.compile()

    # optimizer
    # batch size counts sequences, following mae_st which scales by repeated samples
    total_batch_size = args.batch_size * args.num_samples * args.accum_iter
    logger.info(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} images x {args.num_samples} samples x {args.accum_iter} accum"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        logger.info(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        logger.info(f"lr: {args.lr:.2e}")

    param_groups = misc.get_param_groups(model)
    misc.update_lr(param_groups, args.lr)
    misc.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    optimizer = torch.optim.AdamW(param_groups, betas=betas, fused=args.fused_adamw)

    lr_schedule = make_lr_schedule(
        args.lr, args.min_lr, args.total_steps, args.warmup_steps, args.decay_steps
    )
    logger.info(
        f"lr schedule: total steps = {args.total_steps}, warmup = {args.warmup_steps}, "
        f"decay = {args.decay_steps}"
    )

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    start_step, prev_wall_time = misc.load_model(args, model, optimizer, loss_scaler)

    # data loader. one endless pass over shuffled epochs, resumed at the next unseen image
    dataset = Fomo300kDataset(DATA_ROOT / "fomo300k", args.train_filelist)
    logger.info(f"train dataset: {len(dataset)} images")
    sampler = InfiniteSampler(
        len(dataset), seed=args.seed, start=start_step * args.accum_iter * args.batch_size
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=list,  # no collate, variable shapes
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=ignore_sigterm,
    )
    batch_transform = make_batch_transform(args)

    # on SIGTERM (e.g. preemption) finish the current step, save and exit. resumes when relaunched
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())

    stats = train(
        args,
        model,
        train_loader,
        batch_transform,
        optimizer,
        loss_scaler,
        lr_schedule,
        start_step,
        prev_wall_time,
        process_start_time,
        device,
        batch_flops,
        stop_event,
    )

    step = stats["step"]
    images = step * args.accum_iter * args.batch_size
    summary = {
        "step": step,
        "images": images,
        "epochs": images / len(dataset),
        "train_flops": step * args.accum_iter * batch_flops,
        "wall_seconds": stats["wall_time"],
        "start_step": start_step,
        "done": stats["done"],
        "stopped_by_signal": stop_event.is_set(),
        "final_loss": stats["loss"],
        "num_params": num_params,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cpus": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git": misc.git_info(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary:\n{json.dumps(summary, indent=2)}")
    wall_time_str = datetime.timedelta(seconds=int(stats["wall_time"]))
    if stats["done"]:
        logger.info(f"done! step {step}, wall time: {wall_time_str}")
    else:
        logger.info(f"stopped by signal at step {step}. wall time: {wall_time_str}")


def train(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    batch_transform: Callable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: np.ndarray,
    start_step: int,
    prev_wall_time: float,
    start_time: float,
    device: torch.device,
    batch_flops: float,
    stop_event: threading.Event,
) -> dict:
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", misc.SmoothedValue())
    metric_logger.add_meter("loss", misc.SmoothedValue())
    metric_logger.add_meter(
        "vol/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "tflop/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "watts", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )

    throughput = misc.ThroughputMeter(
        {"vol": args.batch_size, "tflop": batch_flops / 1e12}, device=device
    )

    # steps count optimizer updates, batches count loader batches
    total_steps = args.total_steps if not args.debug else start_step + 10
    print_freq = args.print_freq if not args.debug else 1
    print_freq_batches = print_freq * args.accum_iter
    num_batches = total_steps * args.accum_iter
    start_batch = start_step * args.accum_iter

    amp_dtype = getattr(torch, args.amp_dtype)
    use_cuda = device.type == "cuda"
    output_dir = Path(args.output_dir)
    log_path = output_dir / "log.json"
    # time limit on wall time from the start of main, summed over restarts
    time_limit = args.time_limit * 60 if args.time_limit else float("inf")
    step = start_step
    done = False
    # wall time at the end of the first step of this process (excludes compile), for time_schedule
    # persisted in the output dir so a requeued/preempted run keeps the ORIGINAL origin (wall time
    # is summed over restarts via the checkpoint); re-keying at resume would restart the anneal
    t0_path = output_dir / "time_schedule_t0.json"
    t_step1 = None
    if start_step > 0:
        if t0_path.exists():
            t_step1 = float(json.loads(t0_path.read_text())["t_step1"])
        else:
            t_step1 = min(prev_wall_time, 70.0)
        logger.info(f"resumed at step {start_step}, wall {prev_wall_time:.0f} s: t_step1 {t_step1:.0f} s")

    optimizer.zero_grad()

    batches = metric_logger.log_every(
        data_loader,
        print_freq_batches,
        "Train:",
        total_steps=num_batches,
        start_step=start_batch,
    )
    for batch_idx, batch in enumerate(batches, start=start_batch):
        need_update = (batch_idx + 1) % args.accum_iter == 0
        log_step = (batch_idx + 1) % print_freq_batches == 0
        if args.time_schedule:
            elapsed = (prev_wall_time + time.monotonic() - start_time) - t_step1 if t_step1 else 0.0
            budget = time_limit - (t_step1 or 0.0) - args.time_margin
            lr = time_lr(
                step, args.lr, args.min_lr, args.warmup_steps, elapsed, budget, args.time_decay_frac
            )
        else:
            lr = float(lr_schedule[step])
        if need_update:
            misc.update_lr(optimizer.param_groups, lr)

        if use_cuda:
            batch = misc.send_data(batch, device)

        # run volume processing on gpu, not cpu workers, ~25x faster
        batch = batch_transform(batch)
        images = batch["image"]
        masks = batch["mask"]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
            )

        loss_for_log = loss.detach()
        torch._assert_async(torch.isfinite(loss_for_log), "non-finite loss")

        grad_norm = misc.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        # log tensors without syncing, values are only read on log steps
        metric_logger.update(loss=loss_for_log)
        if need_update:
            metric_logger.update(lr=lr, grad=grad_norm)

        throughput.step()
        if log_step:
            metric_logger.update(**throughput.compute())
            log_stats = {
                "step": step + 1,
                "loss": metric_logger.loss.value,
                "lr": lr,
                "grad": metric_logger.grad.value,
                "vol_per_sec": metric_logger.meters["vol/s"].value,
                "tflop_per_sec": metric_logger.meters["tflop/s"].value,
                "watts": metric_logger.meters["watts"].value,
            }
            with log_path.open("a") as f:
                print(json.dumps(log_stats), file=f)
            if args.wandb:
                wandb.log({f"train/{k}": v for k, v in log_stats.items()}, step=step + 1)

        if not need_update:
            continue
        step += 1
        wall_time = prev_wall_time + time.monotonic() - start_time
        if t_step1 is None:
            t_step1 = wall_time
            if misc.is_main_process():
                t0_path.write_text(json.dumps({"t_step1": t_step1, "step": step}))
        done = step == total_steps or wall_time >= time_limit
        is_last = done or stop_event.is_set()

        # plot a few images from the current batch
        num_examples = args.plot_nrow * args.plot_ncol
        # not on stop by signal, just save and exit
        is_plot_step = args.plot_period and (step % args.plot_period == 0 or done)
        if num_examples and is_plot_step:
            plot_paths = plot_examples(args, model, images, masks, step, amp_dtype, device)
            if args.wandb:
                wandb.log(
                    {f"train/{name}": wandb.Image(str(path)) for name, path in plot_paths.items()},
                    step=step,
                )

        is_save_step = step % args.checkpoint_period == 0 or step in (args.checkpoint_steps or [])
        if is_save_step or is_last:
            misc.save_model(args, step, model, optimizer, loss_scaler, wall_time)

        if is_last:
            break

    logger.info(f"Averaged stats: {metric_logger}")
    return {
        "step": step,
        "done": done,
        "wall_time": prev_wall_time + time.monotonic() - start_time,
        "loss": metric_logger.loss.median,
    }


def ignore_sigterm(worker_id: int):
    # `timeout` signals the whole process group. workers keep loading until the main process exits
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def make_lr_schedule(
    lr: float, min_lr: float, total_steps: int, warmup_steps: int, decay_steps: int
) -> np.ndarray:
    warmup = np.linspace(0.0, lr, warmup_steps, endpoint=False)
    stable = np.full(total_steps - warmup_steps - decay_steps, lr)
    decay = np.linspace(lr, min_lr, decay_steps)
    return np.concatenate([warmup, stable, decay])


def time_lr(
    step: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    elapsed: float,
    budget: float,
    decay_frac: float,
) -> float:
    """Wall-clock schedule: step-based linear warmup, constant peak, then linear decay to min_lr
    over the last decay_frac of the time budget (elapsed/budget in seconds, from end of step 1)."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = min(max(elapsed / max(budget, 1e-9), 0.0), 1.0)
    start = 1.0 - decay_frac
    if progress <= start or decay_frac <= 0:
        return lr
    return lr + (min_lr - lr) * (progress - start) / decay_frac


def make_batch_transform(args: DictConfig):
    def fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        batch = [
            sample_transform(
                sample,
                target_spacing=args.target_spacing,
                grid_size=tuple(args.grid_size),
                clip_quantiles=tuple(args.clip_quantiles),
            )
            for sample in batch
        ]
        images = torch.stack([sample["image"] for sample in batch])
        masks = torch.stack([sample["mask"] for sample in batch])
        return {"image": images, "mask": masks}

    return fn


def sample_transform(
    sample: dict[str, Tensor],
    *,
    target_spacing: float = 1.0,
    grid_size: tuple[int, int, int] = (192, 240, 192),
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
) -> dict[str, Tensor]:
    # pretraining preprocessing, on gpu. eval preprocessing is ViTMAE3D.transform_cuda, keep in sync
    values = sample["values"].float() * sample["scale"]
    values = normalize_values(values, clip_quantiles)
    mask = brle_to_dense(sample["mask_rle"], shape=sample["shape"])
    image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
    image, mask, affine = reorient_to_ras(image, mask, sample["affine"])
    image, mask, affine = resample_to_spacing(image, mask, affine, target_spacing)
    image, mask, affine = fit_to_shape(image, mask, affine, grid_size)
    return {"image": image, "mask": mask, "affine": affine}


def count_batch_flops(args: DictConfig, model: nn.Module, device: torch.device) -> float:
    # fwd + bwd flops of one training batch, on dummy images where every patch is in the mask
    amp_dtype = getattr(torch, args.amp_dtype)
    images = torch.zeros(args.batch_size, *args.grid_size, device=device)
    masks = torch.ones(args.batch_size, *args.grid_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
            )
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def plot_examples(
    args: DictConfig,
    model: nn.Module,
    images: Tensor,
    masks: Tensor,
    step: int,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Path]:
    num_examples = args.plot_nrow * args.plot_ncol
    images = images[:num_examples]
    masks = masks[:num_examples]

    model.eval()
    # eager, the plot shapes differ from training and compiling them isn't worth it
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp),
        torch.compiler.set_stance("force_eager"),
    ):
        _, state = model(
            images,
            masks,
            num_visible=args.num_visible,
            num_predict=args.plot_num_predict,
            num_samples=1,
        )
    model.train()

    grid_size = tuple(args.grid_size)
    patch_size = args.patch_size
    vis_ids = state["vis_ids"]
    target_ids = state["target_ids"]
    pred = patches_to_volume(state["pred_patches"].float(), target_ids, grid_size, patch_size)
    visible_mask = patches_to_volume(
        torch.ones_like(state["vis_patches"]), vis_ids, grid_size, patch_size
    )
    pred_mask = patches_to_volume(
        torch.ones_like(state["target_patches"]), target_ids, grid_size, patch_size
    )

    images = images.float().cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()
    visible_mask = visible_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    plot_dir = Path(args.output_dir) / "plots"
    plot_paths = {}
    for view in ["sag", "cor", "ax"]:
        fig = plot_mask_pred(
            images,
            pred,
            visible_mask,
            pred_mask=pred_mask,
            img_mask=masks,
            view=view,
            nrow=args.plot_nrow,
            ncol=args.plot_ncol,
        )
        path = plot_dir / f"mae_{view}_{step:07d}.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"mae_{view}"] = path
    return plot_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
