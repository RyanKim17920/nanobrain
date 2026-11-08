# SIGReg on the eval-read global embedding (anti-collapse readout regularizer, training only).
#
# The cls / reg probes are StandardScaler + L2 logistic / ridge on the mean last-layer token with
# n ~ 300-1000 subjects and D = 768. Ridge shrinks each eigendirection of the standardized features
# by lam / (lam + alpha), so signal in low-variance directions is paid for by the penalty; mean-pooled
# MAE features are dimensionally collapsed (U-MAE, arXiv 2210.08344), and the FINO age head adds one
# more dominant direction. LeJEPA (arXiv 2511.08544) shows the isotropic Gaussian minimizes the
# worst-case linear-probe risk and gives SIGReg: an Epps-Pulley test of N(0, 1) on random 1-D slices.
#
# Here: the pooled vector of each sequence (mean of its visible encoder tokens, the training version
# of the eval readout) is standardized per dimension like the probe's StandardScaler (stats over the
# current rows and a FIFO queue of recent detached rows, so N ~ 1k instead of 64), then SIGReg pushes
# every random slice towards N(0, 1). Per-dim scale is free (standardized away); what is penalized is
# correlation / low effective rank / heavy clustering. No invariance term (the r12t-pair failure mode:
# same-image positives learn a histogram shortcut). The queue is a non-persistent buffer and the
# module has no parameters, so checkpoints and the eval are unchanged.
import torch
import torch.nn as nn
from torch import Tensor


class PooledSigReg(nn.Module):
    def __init__(self, embed_dim: int, cfg):
        super().__init__()
        self.weight = float(cfg["weight"])
        self.token_weight = float(cfg.get("token_weight") or 0.0)
        self.num_slices = int(cfg["num_slices"])
        self.queue_size = int(cfg["queue"])
        self.warmup = float(cfg.get("warmup") or 0.0)
        self.rank_every = int(cfg.get("rank_every") or 0)
        knots = int(cfg["knots"])
        # LeJEPA reference quadrature: t in [0, 3], trapezoid weights times the N(0, 1) cf window
        t = torch.linspace(0.0, 3.0, knots)
        dt = 3.0 / (knots - 1)
        w = torch.full((knots,), 2.0 * dt)
        w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("qw", w * phi, persistent=False)
        self.register_buffer("queue", torch.zeros(self.queue_size, embed_dim), persistent=False)
        self.ptr = 0
        self.filled = 0
        self.calls = 0
        self.seed = int(cfg.get("seed") or 0)
        self.gen = None
        self.last_rank = {}

    def _directions(self, dim: int, device) -> Tensor:
        if self.gen is None or self.gen.device != torch.device(device):
            self.gen = torch.Generator(device=device).manual_seed(self.seed)
        a = torch.randn(dim, self.num_slices, device=device, generator=self.gen)
        return a / a.norm(dim=0, keepdim=True)

    def statistic(self, z: Tensor, extra: Tensor | None = None) -> Tensor:
        # z [N, D] standardized rows with grad, extra [M, D] standardized rows without; mean over
        # slices of the Epps-Pulley integral (NOT times N: the weight is independent of the queue)
        a = self._directions(z.shape[1], z.device)
        rows = z if extra is None else torch.cat([z, extra])
        x_t = (rows @ a).unsqueeze(-1) * self.t  # [N, S, K]
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()  # [S, K]
        return (err @ self.qw).mean()

    def forward(self, pooled: Tensor, tokens: Tensor | None, progress: float) -> tuple[Tensor, dict]:
        # pooled [R, D] (grad), tokens [R, N, D] visible encoder tokens (grad) or None
        pooled = pooled.float()
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            old = self.queue[: self.filled]
            both = torch.cat([pooled, old])
            mu = both.mean(0)
            sd = both.std(0, correction=0).clamp_min(1e-4)
            z = (pooled - mu) / sd
            stat = self.statistic(z, ((old - mu) / sd).detach() if self.filled else None)
            ramp = 1.0 if self.warmup <= 0 else min(max(progress, 0.0) / self.warmup, 1.0)
            loss = self.weight * ramp * stat
            stats = {"sigreg": stat.detach()}
            if self.token_weight and tokens is not None:
                # token-level variant: per-sequence centred tokens (removes the image mean, which the
                # pooled term handles), batch standardized, 8k random tokens
                tok = tokens.float()
                tok = tok - tok.mean(1, keepdim=True)
                tok = tok.flatten(0, 1)
                idx = torch.randint(0, tok.shape[0], (min(8192, tok.shape[0]),), device=tok.device,
                                    generator=self.gen)
                tok = tok[idx]
                tz = (tok - tok.mean(0)) / tok.std(0, correction=0).clamp_min(1e-4)
                tstat = self.statistic(tz)
                loss = loss + self.token_weight * ramp * tstat
                stats["sigreg_tok"] = tstat.detach()
            self.calls += 1
            if self.rank_every and self.calls % self.rank_every == 1:
                with torch.no_grad():
                    zz = torch.cat([z.detach(), (old - mu) / sd]) if self.filled else z.detach()
                    s = torch.linalg.svdvals(zz - zz.mean(0))
                    p = s.square() / s.square().sum().clamp_min(1e-12)
                    self.last_rank = {"pool_erank": torch.exp(-(p * p.clamp_min(1e-12).log()).sum()),
                                      "pool_top1": p[0]}
            stats.update(self.last_rank)
            # FIFO queue of detached raw pooled rows
            with torch.no_grad():
                n = min(pooled.shape[0], self.queue_size)
                rows = pooled.detach()[:n]
                end = self.ptr + n
                if end <= self.queue_size:
                    self.queue[self.ptr : end] = rows
                else:
                    k = self.queue_size - self.ptr
                    self.queue[self.ptr :] = rows[:k]
                    self.queue[: n - k] = rows[k:]
                self.ptr = end % self.queue_size
                self.filled = min(self.filled + n, self.queue_size)
        return loss, stats
