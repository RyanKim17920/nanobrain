"""Checkpoint resume on requeue: a run stopped mid-way (SIGTERM save, or a hard kill after a timed
checkpoint_minutes save) and resumed ends bit-identical to an uninterrupted run (cpu, num_workers 0: the
model, optimizer, step, lr schedule and rng streams all continue), and a resumed process's startup is not
counted as training wall time.

    BRAINMARKS_SMRI_DATA=... uv run pytest tests/test_resume.py
"""

import json
import os
import signal
import time
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(
    not os.environ.get("BRAINMARKS_SMRI_DATA"), reason="needs BRAINMARKS_SMRI_DATA (fomo300k images)"
)

STEPS = 6
TINY = [
    "device=cpu", "compile=false", "amp=false", "fused_adamw=false", "wandb=false",
    "num_workers=0", "prep_threads=0", "plot_period=0", "print_freq=1",
    "grid_size=[64,64,64]", "num_visible=32", "num_predict=64", "num_samples=1", "batch_size=1",
    "model_kwargs.depth=1", "model_kwargs.embed_dim=64", "model_kwargs.num_heads=2",
    "model_kwargs.decoder_depth=1", "model_kwargs.decoder_embed_dim=32", "model_kwargs.decoder_num_heads=2",
    f"total_steps={STEPS}", "warmup_steps=2", "decay_steps=3", "checkpoint_period=100000000", "seed=5",
]


def config(out: Path, *extra: str):
    import nanobrain.train as T

    cfg = OmegaConf.load(T.DEFAULT_CONFIG)
    return OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist([*TINY, f"output_dir={out}", "name=", *extra]))


def run(out: Path, *extra: str, on_backward=None):
    """nanobrain.train.main in process; on_backward(n) is called before the n-th backward (1-based)."""
    import nanobrain.misc as misc
    import nanobrain.train as T

    real = misc.backward_step
    calls = [0]

    def backward_step(*a, **k):
        calls[0] += 1
        if on_backward is not None:
            on_backward(calls[0])
        return real(*a, **k)

    misc.backward_step = backward_step
    old = signal.getsignal(signal.SIGTERM)
    try:
        T.main(config(out, *extra))
    finally:
        misc.backward_step = real
        signal.signal(signal.SIGTERM, old)


def state(out: Path) -> dict:
    return torch.load(out / "checkpoint-last.pth", map_location="cpu", weights_only=True)


def lrs(out: Path) -> dict:
    rows = [json.loads(line) for line in (out / "log.json").read_text().splitlines()]
    return {r["step"]: r["lr"] for r in rows}


def same_weights(a: dict, b: dict) -> bool:
    return a["model"].keys() == b["model"].keys() and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    out = tmp_path_factory.mktemp("ref")
    run(out)
    return out


def test_sigterm_stop_then_resume_matches_an_uninterrupted_run(tmp_path, reference):
    out = tmp_path / "run"

    def term_at_3(n):
        if n == 3:
            os.kill(os.getpid(), signal.SIGTERM)

    run(out, on_backward=term_at_3)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["step"] == 3 and summary["stopped_by_signal"] and not summary["done"]
    assert json.loads((out / "rng_state_rank0.json").read_text())["step"] == 3

    run(out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["start_step"] == 3 and summary["step"] == STEPS and summary["done"]
    assert "restored the rng streams of step 3" in (out / "log.txt").read_text()
    ref, got = state(reference), state(out)
    assert got["step"] == ref["step"] == STEPS
    assert same_weights(ref, got)
    assert lrs(out) == lrs(reference)          # the lr schedule continues (warmup 2, decay over the last 3)


def test_hard_kill_after_a_timed_checkpoint_resumes_from_it(tmp_path, reference):
    # checkpoint_minutes ~0: a timed save after every step; a crash in step 5 (no final save) loses only step 5
    out = tmp_path / "run"

    def crash_at_5(n):
        if n == 5:
            raise RuntimeError("simulated hard kill")

    with pytest.raises(RuntimeError, match="simulated hard kill"):
        run(out, "checkpoint_minutes=1e-9", on_backward=crash_at_5)
    assert state(out)["step"] == 4 and not (out / "summary.json").exists()

    run(out, "checkpoint_minutes=1e-9")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["start_step"] == 4 and summary["done"]
    assert same_weights(state(reference), state(out))


def test_resumed_startup_is_not_training_wall_time(tmp_path):
    import nanobrain.misc as misc

    out = tmp_path / "run"

    def term_at_2(n):
        if n == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    run(out, "total_steps=4", "warmup_steps=1", "decay_steps=1", on_backward=term_at_2)
    wall_at_stop = state(out)["wall_time"]

    real = misc.load_model
    delay = 8.0

    def slow_load(*a, **k):                    # a slow resumed startup (a cold node, compile, loader warmup)
        time.sleep(delay)
        return real(*a, **k)

    misc.load_model = slow_load
    try:
        t0 = time.monotonic()
        run(out, "total_steps=4", "warmup_steps=1", "decay_steps=1")
        elapsed = time.monotonic() - t0
    finally:
        misc.load_model = real
    summary = json.loads((out / "summary.json").read_text())
    assert summary["done"] and summary["start_step"] == 2
    assert elapsed > delay
    assert summary["wall_seconds"] - wall_at_stop < elapsed - delay + 0.5
