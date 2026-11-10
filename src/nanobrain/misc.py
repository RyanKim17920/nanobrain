# References:
# deit: https://github.com/facebookresearch/deit/blob/main/utils.py
# beit3: https://github.com/microsoft/unilm/blob/master/beit3/utils.py
# capi: https://github.com/facebookresearch/capi/blob/main/utils.py
# dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/param_groups.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/cuda.py
# dino: https://github.com/facebookresearch/dino/blob/main/utils.py

import datetime
import itertools
import logging
import os
import queue
import random
import subprocess
import sys
import tarfile
import threading
import time
from collections import defaultdict, deque
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler
from torch.optim import Optimizer

# optional, only used for gpu power in ThroughputMeter
try:
    import pynvml
except ImportError:
    pynvml = None

logger = logging.getLogger(__name__)


# logging utils without syncs copied from capi


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.deque: deque[Tensor | float | int] = deque(maxlen=window_size)
        self.total: Tensor | float | int = 0.0
        self.count: int = 0
        self.fmt = fmt

    def update(self, value: Tensor | float | int):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def synchronize_between_processes(self):
        """Distributed synchronization of the metric"""
        if not torch.distributed.is_initialized():
            return
        count = to_tensor(self.count).to(dtype=torch.float64, device="cuda").reshape(1)
        total = to_tensor(self.total).to(dtype=torch.float64, device="cuda").reshape(1)
        tensor_deque = torch.tensor(list(self.deque), dtype=torch.float64, device="cuda")
        t = torch.cat([count, total, tensor_deque], dim=0)
        torch.distributed.barrier()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
        self.count = int(t[0].cpu().item())
        self.total = t[1]
        self.deque = deque(list(t[2:]), maxlen=self.window_size)

    @property
    def median(self) -> float | int:
        if not self.count:
            return float("nan")
        d = torch.tensor(list(self.deque))
        return d.median().cpu().item()

    @property
    def avg(self) -> float | int:
        if not self.count:
            return float("nan")
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().cpu().item()

    @property
    def global_avg(self) -> float | int:
        if not self.count:
            return float("nan")
        return to_tensor(self.total).cpu().item() / self.count

    @property
    def max(self) -> float | int:
        if not self.count:
            return float("nan")
        return torch.tensor(self.deque).max().cpu().item()

    @property
    def value(self) -> float | int:
        if not self.count:
            return float("nan")
        v = self.deque[-1]
        return to_tensor(v).cpu().item()

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter: str = "  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter!s}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, total_steps=None, start_step=0):
        i = start_step
        total_steps = total_steps or len(iterable)
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(total_steps))) + "d"
        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list.append("max mem: {memory:.0f}MB")

        log_msg = self.delimiter.join(log_list)
        for obj in iterable:
            if i >= total_steps:
                break
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_steps - 1:
                self.synchronize_between_processes()
                eta_seconds = iter_time.global_avg * (total_steps - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            total_steps,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        ),
                    )

                else:
                    logger.info(
                        log_msg.format(
                            i,
                            total_steps,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        num_steps = max(i - start_step, 1)
        logger.info(f"{header} Total time: {total_time_str} ({total_time / num_steps:.6f} s / it)")


def to_tensor(x: Tensor | float | int) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.tensor(x)


def setup_for_distributed(log_path=None):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    is_master = is_main_process()

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
            # tee to log file
            if log_path and "file" not in kwargs:
                with open(log_path, "a") as f:
                    builtin_print(*args, file=f, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(ckpt, path):
    if is_main_process():
        # atomic save in case we are interrupted
        tmp_path = Path(path)
        tmp_path = tmp_path.parent / f".tmp-{tmp_path.name}"
        try:
            torch.save(ckpt, tmp_path)
            tmp_path.rename(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def init_distributed_mode(args):
    # removed slurm block, can add if we use slurm
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank})")
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=args.rank,
        device_id=args.gpu,
    )
    torch.distributed.barrier()


# checkpoint saving utils adapted from beit3


def save_model(args, step, model_without_ddp, optimizer, loss_scaler, wall_time=0.0):
    output_dir = Path(args.output_dir)
    last_checkpoint_path = output_dir / "checkpoint-last.pth"

    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "wall_time": wall_time,
        "scaler": None if loss_scaler is None else loss_scaler.state_dict(),
        "args": OmegaConf.to_container(args),
        "model_class": f"{type(model_without_ddp).__module__}:{type(model_without_ddp).__name__}",
        "git": git_info(),
    }

    logger.info(f"saving checkpoint {last_checkpoint_path}")
    save_on_master(to_save, last_checkpoint_path)

    if args.checkpoint_steps and step in args.checkpoint_steps:
        checkpoint_path = output_dir / f"checkpoint-{step:07d}.pth"
        logger.info(f"saving checkpoint {checkpoint_path}")
        save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler) -> tuple[int, float]:
    """Load args.ckpt, or the last checkpoint when auto resuming. Returns (start step, wall time)."""
    auto_resume = getattr(args, "auto_resume", True)
    output_dir = Path(args.output_dir)

    last_checkpoint_path = output_dir / "checkpoint-last.pth"
    if auto_resume and last_checkpoint_path.exists():
        args.ckpt = str(last_checkpoint_path)
        args.resume = True

    start_step = 0
    wall_time = 0.0
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(ckpt["model"])
        logger.info(f"loaded model state from checkpoint {args.ckpt}")

        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            if loss_scaler is not None:
                loss_scaler.load_state_dict(ckpt["scaler"])
            start_step = ckpt["step"]
            wall_time = ckpt["wall_time"]
            logger.info(f"loaded optimizer state, resuming training from step {start_step}")
    return start_step, wall_time


# optimization utils


# adapted from timm backward logic
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/cuda.py
def backward_step(
    loss: Tensor,
    optimizer: Optimizer,
    scaler: GradScaler = None,
    need_update: bool = True,
    max_norm: float | None = None,
) -> Tensor | None:
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if need_update:
        if scaler is not None:
            scaler.unscale_(optimizer)

        total_norm = clip_grad(optimizer, max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    else:
        total_norm = None
    return total_norm


def clip_grad(optimizer: Optimizer, max_norm: float | None = None) -> Tensor:
    params = [p for group in optimizer.param_groups for p in group["params"]]
    # optimizers that defer the clip (fused muon) take the clip coefficient as grad_scale and
    # apply it inside their step, saving a pass over their grads. same math as clip_grad_norm_
    deferred = [
        opt
        for opt in getattr(optimizer, "optimizers", [optimizer])
        if getattr(opt, "defers_grad_clip", False)
    ]
    if max_norm and deferred:
        skip = {p for opt in deferred for group in opt.param_groups for p in group["params"]}
        grads = [p.grad for p in params if p.grad is not None]
        total_norm = nn.utils.get_total_norm(grads, error_if_nonfinite=False)
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        rest = [p.grad for p in params if p.grad is not None and p not in skip]
        torch._foreach_mul_(rest, clip_coef)
        for opt in deferred:
            opt.grad_scale = clip_coef
    elif max_norm:
        total_norm = nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=False)
    else:
        grads = [p.grad for p in params if p.grad is not None]
        total_norm = nn.utils.get_total_norm(grads, error_if_nonfinite=False)
    torch._assert_async(torch.isfinite(total_norm), "non-finite gradient norm")
    return total_norm


# from dinov2 with some minor changes
def get_param_groups(model, patch_embed_lr_mult=1.0, skip_names=(), lr_mult_fn=None, tag_fn=None):
    # no lr decay, we could add this later if needed
    all_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad or name in skip_names:
            continue
        d = {"param": param, "lr_multiplier": 1.0, "wd_multiplier": 1.0, "name": name}

        if name.endswith(".bias") or "norm" in name or "gamma" in name or "qk_scale" in name:
            d["wd_multiplier"] = 0.0

        if "patch_embed" in name:
            d["lr_multiplier"] = d["lr_multiplier"] * patch_embed_lr_mult
        if lr_mult_fn is not None:
            d["lr_multiplier"] = d["lr_multiplier"] * lr_mult_fn(name)
        if tag_fn is not None:
            # a group key (backbone True / False): groups split by it, the base fusing is unchanged
            d["backbone"] = bool(tag_fn(name))

        all_params.append(d)

    param_groups = _fuse_param_groups(all_params)
    return param_groups


def _fuse_param_groups(all_param_groups):
    fused_param_groups = defaultdict(lambda: {"params": []})
    for d in all_param_groups:
        keys = sorted(set(d.keys()) - {"param", "name"})
        identifier = "_".join(f"{k}{d[k]}" for k in keys)
        for k in keys:
            fused_param_groups[identifier][k] = d[k]
        fused_param_groups[identifier]["params"].append(d["param"])

    param_groups = list(fused_param_groups.values())
    return param_groups


def update_lr(param_groups, lr: float):
    for group in param_groups:
        group["lr"] = lr * group["lr_multiplier"]


def update_wd(param_groups, weight_decay: float | None = None):
    for group in param_groups:
        group["weight_decay"] = weight_decay * group["wd_multiplier"]


# moving data to cuda utils copied from capi
# added device and transform


def send_data(x, device=None):
    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if isinstance(x, torch.Tensor):
        x = x.to(device=device, non_blocking=True)
        return x
    if isinstance(x, dict):
        return {k: send_data(v, device=device) for k, v in x.items()}
    if isinstance(x, list):
        return [send_data(v, device=device) for v in x]
    return x


def record_stream(x, stream):
    if isinstance(x, torch.Tensor) and x.is_cuda:
        x.record_stream(stream)
    if isinstance(x, dict):
        for v in x.values():
            record_stream(v, stream)
    if isinstance(x, list):
        for v in x:
            record_stream(v, stream)


def pre_send_to_cuda_wrapper(generator, batch_transform=None, device=None):
    data = None
    stream = torch.cuda.Stream(device)
    for next_data in generator:
        with torch.cuda.stream(stream):
            next_data = send_data(next_data, device=device)
            if batch_transform is not None:
                next_data = batch_transform(next_data)
        if data is not None:
            yield data
        main_stream = torch.cuda.current_stream(device)
        main_stream.wait_stream(stream)
        record_stream(next_data, main_stream)
        data = next_data
    if data is not None:
        yield data


def prep_thread(loader, batch_transform, device, num_threads: int = 1, depth: int = 1):
    # send + transform the next batches in background threads, each on its own side stream, so the
    # gpu prep's ~3.5k small launches per batch overlap the model's cpu work instead of draining the
    # gpu queue. thread k preps batches k, k + n, ..., consumed in loader order
    it = iter(loader)
    turn = threading.Condition()
    state = {"next": 0}
    queues = [queue.Queue(maxsize=depth) for _ in range(num_threads)]
    stop = threading.Event()

    def put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=1)
                return
            except queue.Full:
                pass

    def run(k):
        stream = torch.cuda.Stream(device)
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                for index in itertools.count(k, num_threads):
                    with turn:
                        turn.wait_for(lambda: state["next"] == index or stop.is_set())
                        if stop.is_set():
                            return
                        batch = next(it, None)
                        state["next"] += 1
                        turn.notify_all()
                    if batch is None:
                        put(queues[k], None)
                        return
                    batch = batch_transform(send_data(batch, device=device))
                    event = torch.cuda.Event()
                    event.record(stream)
                    put(queues[k], (batch, event))
        except BaseException as e:  # noqa: BLE001
            put(queues[k], e)

    threads = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(num_threads)]
    for thread in threads:
        thread.start()
    try:
        for index in itertools.count():
            item = queues[index % num_threads].get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            batch, event = item
            main_stream = torch.cuda.current_stream(device)
            main_stream.wait_event(event)
            record_stream(batch, main_stream)
            yield batch
    finally:
        stop.set()
        with turn:
            turn.notify_all()
        for thread in threads:
            thread.join(timeout=60)
        # the loader iterator lives in the threads' closure, so it may outlive this generator. its
        # workers ignore SIGTERM (ignore_sigterm), so multiprocessing's exit handler would hang on them
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()



# other misc utils


class ThroughputMeter:
    """Rates per second over a window of steps, plus average gpu power over the window."""

    def __init__(self, units_per_step: dict[str, float], device: torch.device | None = None):
        self.units_per_step = units_per_step
        self.device = device
        self.gpu_handle = None
        if pynvml is not None and device is not None and device.type == "cuda":
            # match by uuid since nvml device indices ignore CUDA_VISIBLE_DEVICES
            pynvml.nvmlInit()
            uuid = torch.cuda.get_device_properties(device).uuid
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        self.reset()

    def reset(self):
        self.start_time = time.monotonic()
        self.num_steps = 0
        if self.gpu_handle is not None:
            self.start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)

    def step(self):
        self.num_steps += 1

    def compute(self) -> dict[str, float]:
        # sync so queued gpu work is counted in this window
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - self.start_time

        stats = {}
        for name, units in self.units_per_step.items():
            stats[f"{name}/s"] = self.num_steps * units / elapsed
        if self.gpu_handle is not None:
            # energy counter is in mJ
            energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
            stats["watts"] = (energy - self.start_energy) / 1000 / elapsed

        self.reset()
        return stats


def random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# brainmarks-smri git_info, plus the origin remote
def git_info(path: Path = Path(__file__).parent) -> dict[str, Any] | None:
    """Commit, branch, dirty state and origin of the git repo containing `path`; None outside one."""
    kwargs = dict(cwd=path, capture_output=True, text=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], **kwargs)
    if sha.returncode != 0:
        return None
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], **kwargs).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "-uno"], **kwargs).stdout.strip()
    remote = subprocess.run(["git", "remote", "get-url", "origin"], **kwargs).stdout.strip()
    return {
        "sha": sha.stdout.strip(),
        "branch": branch,
        "dirty": bool(status),
        "remote": remote or None,
    }


def snapshot_source(src_dir: Path, path: Path) -> None:
    """Save `src_dir` as a .tar.gz archive at `path`, without python caches."""

    def exclude_caches(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = Path(info.name).name
        if name == "__pycache__" or name.endswith((".pyc", ".egg-info")):
            return None
        return info

    with tarfile.open(path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name, filter=exclude_caches)


def setup_logging(logger: logging.Logger, log_path: Path | None = None) -> None:
    # non-main ranks only log warnings, to stdout. call after init_distributed_mode
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main_process():
        level = logging.INFO
        if log_path is not None:
            handlers.append(logging.FileHandler(log_path))
    else:
        level = logging.WARNING
    logger.setLevel(level)
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
