"""Throughput bench / profile of nanobrain.train (ms-speed). Runs the real training main() with the
same config, stops after BENCH_STEPS optimizer steps and writes OUT/bench.json:
  steady-state samples/s over [BENCH_T0, BENCH_STEPS] (synced at the window edges only),
  optional sync-timed phase breakdown (BENCH_SYNC=a:b: forward / backward / optimizer, cuda synced),
  optional torch.profiler window (BENCH_PROF=a:b): kernel table + gpu busy fraction.

  python scripts/speed_bench.py --config CFG --overrides k=v ...
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

import nanobrain.misc as misc
import nanobrain.train as train_mod
from nanobrain.model import ViTMAE3D

STEPS = int(os.environ.get("BENCH_STEPS", "600"))
T0 = int(os.environ.get("BENCH_T0", "200"))
SYNC = tuple(int(x) for x in os.environ["BENCH_SYNC"].split(":")) if os.environ.get("BENCH_SYNC") else None
PROF = tuple(int(x) for x in os.environ["BENCH_PROF"].split(":")) if os.environ.get("BENCH_PROF") else None

def cpu_snapshot():
    # cpu seconds: main thread, every thread of this process (autograd / prep / pin threads), loader workers
    tick = os.sysconf("SC_CLK_TCK")
    pid = os.getpid()

    def secs(path):
        try:
            f = open(path).read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / tick
        except OSError:
            return 0.0

    tasks = {int(t): secs(f"/proc/self/task/{t}/stat") for t in os.listdir("/proc/self/task")}
    kids = 0.0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == pid:
                kids += secs(f"/proc/{d}/stat")
    return {"main": tasks.get(pid, 0.0), "threads": tasks, "workers": kids, "wall": time.perf_counter()}


state = {"cpu": {}, "step": 0, "t": {}, "phase": defaultdict(float), "prof": None, "out": None}


class Stop(Exception):
    pass


def in_window(win):
    return win is not None and win[0] <= state["step"] < win[1]


def sync_time(name, fn, *a, **k):
    if not in_window(SYNC):
        return fn(*a, **k)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.profiler.record_function(name):
        out = fn(*a, **k)
    torch.cuda.synchronize()
    state["phase"][name] += time.perf_counter() - t
    return out


_forward = ViTMAE3D.forward


def forward(self, *a, **k):
    with torch.profiler.record_function("fwd"):
        return sync_time("forward", _forward, self, *a, **k)


ViTMAE3D.forward = forward

_clip = misc.clip_grad


def backward_step(loss, optimizer, scaler=None, need_update=True, max_norm=None):
    with torch.profiler.record_function("bwd"):
        sync_time("backward", loss.backward)

    def opt():
        norm = _clip(optimizer, max_norm)
        optimizer.step()
        optimizer.zero_grad()
        return norm

    with torch.profiler.record_function("opt"):
        return sync_time("optimizer", opt)


misc.backward_step = backward_step

_step = misc.ThroughputMeter.step


def meter_step(self):
    _step(self)
    state["step"] += 1
    s = state["step"]
    if PROF and s == PROF[0]:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=bool(os.environ.get("BENCH_STACK")),
        )
        prof.__enter__()
        torch.cuda.synchronize()
        state["prof"] = (prof, time.perf_counter())
    if PROF and s == PROF[1] and state["prof"]:
        torch.cuda.synchronize()
        prof, t = state["prof"]
        wall = time.perf_counter() - t
        prof.__exit__(None, None, None)
        dump_prof(prof, wall, PROF[1] - PROF[0])
    if s in (T0, STEPS):
        torch.cuda.synchronize()
        state["t"][s] = time.perf_counter()
        state["cpu"][s] = cpu_snapshot()
    if s == STEPS:
        raise Stop()


misc.ThroughputMeter.step = meter_step


def dump_prof(prof, wall, steps):
    out = state["out"]
    ka = prof.key_averages()
    (out / "prof_cuda.txt").write_text(ka.table(sort_by="self_cuda_time_total", row_limit=60))
    (out / "prof_cpu.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=40))
    if os.environ.get("BENCH_STACK"):
        kg = prof.key_averages(group_by_stack_n=6)
        (out / "prof_cpu_stack.txt").write_text(kg.table(sort_by="cpu_time_total", row_limit=80))
    try:
        prof.export_chrome_trace(str(out / "trace.json"))
    except Exception as e:  # noqa: BLE001
        print("trace export failed", e)
    # gpu busy: union of kernel intervals (all streams); per-region gpu time from annotations
    ivs, regions = [], defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            iv = (e.time_range.start, e.time_range.end)
            if e.name in ("fwd", "bwd", "opt"):
                regions[e.name] += (iv[1] - iv[0]) / 1e6
            else:
                ivs.append(iv)
    ivs.sort()
    busy, cur = 0.0, None
    for s, e in ivs:
        if cur is None or s > cur[1]:
            if cur:
                busy += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        busy += cur[1] - cur[0]
    kern = sum(e - s for s, e in ivs)
    state["prof_summary"] = {
        "wall_s_per_step": wall / steps,
        "gpu_busy_s_per_step": busy / 1e6 / steps,
        "gpu_busy_frac": busy / 1e6 / wall,
        "kernel_sum_s_per_step": kern / 1e6 / steps,
        "kernels_per_step": len(ivs) / steps,
        "annot_gpu_s_per_step": {k: v / steps for k, v in regions.items()},
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--overrides", nargs="+", default=[])
    a = p.parse_args()
    cfg = OmegaConf.load(train_mod.DEFAULT_CONFIG)
    cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(a.config))
    cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(a.overrides))
    out = Path(cfg.output_dir) / cfg.name if cfg.name and not cfg.output_dir.endswith(cfg.name) else Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state["out"] = out
    t_start = time.perf_counter()
    try:
        train_mod.main(cfg)
    except Stop:
        pass
    t = state["t"]
    res = {
        "overrides": a.overrides,
        "steps": STEPS,
        "t0": T0,
        "startup_to_t0_s": t.get(T0, 0) - t_start,
        "steady_s_per_step": (t[STEPS] - t[T0]) / (STEPS - T0),
        "steady_images_per_s": (STEPS - T0) * cfg.batch_size / (t[STEPS] - t[T0]),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    a0, a1 = state["cpu"].get(T0), state["cpu"].get(STEPS)
    if a0 and a1:
        wall = a1["wall"] - a0["wall"]
        per = {t: a1["threads"][t] - a0["threads"].get(t, 0.0) for t in a1["threads"]}
        res["cpu_util"] = {
            "main_thread": (a1["main"] - a0["main"]) / wall,
            "other_threads_top": sorted((round(v / wall, 3) for t, v in per.items() if t != os.getpid()), reverse=True)[:8],
            "loader_workers_total": (a1["workers"] - a0["workers"]) / wall,
            "cpus": len(os.sched_getaffinity(0)),
        }
    if SYNC:
        n = SYNC[1] - SYNC[0]
        res["sync_phase_s_per_step"] = {k: v / n for k, v in state["phase"].items()}
    if "prof_summary" in state:
        res["prof"] = state["prof_summary"]
    (out / "bench.json").write_text(json.dumps(res, indent=2))
    print("BENCH", json.dumps(res), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
