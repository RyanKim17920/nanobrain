# XVIEW: a light cross-image agreement objective on the pooled embedding (wave 7 moonshot).
#
# Positive pair = two DIFFERENT images of the same subject (FOMO300K paths carry
# derivatives/<collection>[/<ds>]/sub-X/ses-Y/anat/<file>): anatomy fixed, contrast / session /
# acquisition varies. Never two crops of one volume (r12t-pair learned a histogram shortcut there).
# The pair sampler yields anchor, partner, anchor, partner, ... so the partner is an ordinary MAE
# training image in the same batch: no extra volume is loaded or encoded, the throughput is the
# base's. Each image is drawn once per pass as an anchor and (in expectation) about once as a
# partner, so the image mix stays roughly uniform.
#
# Loss (on the image-level pooled embedding = mean over its num_samples sequences of the mean
# visible encoder token), batch-mean centered so the shared mean direction does not dominate:
#   simsiam: symmetric 1 - cos(pred(z_a), sg(z_p)), asymmetric MLP predictor + stop-grad, no
#            negatives (negatives would reward site identity: pairs share a site, negatives mostly
#            do not)
#   infonce: symmetric InfoNCE over the batch's pairs, MLP projector, temperature tau
# weight ramped 0 -> weight with FINO's sigmoid ramp over training progress. Pairs whose partner is
# not a distinct same-subject image (singleton subjects, ~9%) are masked.
import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Sampler

logger = logging.getLogger()

_SUBJECT = re.compile(r"(.*?/sub-[^/]+)/")


def subject_key(path: str) -> str:
    m = _SUBJECT.match(path)
    return m.group(1) if m else path


def view_key(path: str) -> tuple[str, str]:
    # (session, contrast): run-1 / run-2 repeats of one contrast in one session share it
    parts = path.split("/")
    ses = next((p for p in parts if p.startswith("ses-")), "")
    contrast = parts[-1].rsplit(".", 1)[0].rsplit("_", 1)[-1]
    return ses, contrast


def partner_lists(paths: list[str], policy: str = "diff") -> list[list[int]]:
    """Per image, the candidate partner indices. 'diff': other images of the subject with a
    different (session, contrast), else any other image of the subject; 'any': any other image of
    the subject. Empty = no partner (masked)."""
    assert policy in ("diff", "any"), f"unknown xview partner policy {policy}"
    subjects: dict[str, list[int]] = {}
    for ii, path in enumerate(paths):
        subjects.setdefault(subject_key(path), []).append(ii)
    views = [view_key(p) for p in paths]
    out = []
    for ii, path in enumerate(paths):
        others = [jj for jj in subjects[subject_key(path)] if jj != ii]
        if policy == "diff":
            others = [jj for jj in others if views[jj] != views[ii]] or others
        out.append(others)
    return out


class PairSampler(Sampler[int]):
    """Wraps an anchor sampler: yields anchor, partner for each anchor (a random candidate partner,
    else a random image, masked later). The batch size must be even so pairs never straddle."""

    def __init__(self, base: Sampler[int], partners: list[list[int]], num_samples: int, seed: int = 0):
        self.base = base
        self.partners = partners
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 7919)
        for anchor in self.base:
            cands = self.partners[anchor]
            r = int(torch.randint(1 << 30, (1,), generator=generator))
            yield anchor
            yield cands[r % len(cands)] if cands else r % self.num_samples


def pair_valid(paths: list[str]) -> Tensor:
    # [B/2] bool: the partner is a distinct image of the anchor's subject
    return torch.tensor([
        paths[2 * k] != paths[2 * k + 1] and subject_key(paths[2 * k]) == subject_key(paths[2 * k + 1])
        for k in range(len(paths) // 2)
    ])


class XViewHead(nn.Module):
    def __init__(self, embed_dim: int, cfg):
        super().__init__()
        self.weight = float(cfg["weight"])
        self.loss_type = str(cfg["loss"])
        assert self.loss_type in ("simsiam", "infonce"), f"unknown xview loss {self.loss_type}"
        self.tau = float(cfg["tau"])
        hidden = int(cfg["hidden"])
        # simsiam: predictor; infonce: projector
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim),
        )

    def forward(self, z: Tensor, valid: Tensor, ramp: float) -> tuple[Tensor, dict]:
        # z [B, D] image-level pooled embeddings, rows anchor, partner, anchor, partner, ...
        # valid [B/2] bool on z's device
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            z = z - z.mean(dim=0, keepdim=True).detach()
            za, zp = z[0::2], z[1::2]
            n = valid.sum().clamp_min(1)
            vf = valid.float()
            if self.loss_type == "simsiam":
                pa, pp = self.mlp(za), self.mlp(zp)
                cos_a = F.cosine_similarity(pa, zp.detach(), dim=-1)
                cos_p = F.cosine_similarity(pp, za.detach(), dim=-1)
                term = ((2.0 - cos_a - cos_p) * 0.5 * vf).sum() / n
                stats = {"cos": (cos_a * vf).sum().detach() / n}
            else:
                qa = F.normalize(self.mlp(za), dim=-1)
                qp = F.normalize(self.mlp(zp), dim=-1)
                logits = qa @ qp.T / self.tau
                target = torch.arange(len(qa), device=z.device)
                ce = F.cross_entropy(logits, target, reduction="none") + F.cross_entropy(
                    logits.T, target, reduction="none")
                term = (ce * 0.5 * vf).sum() / n
                stats = {"acc": ((logits.argmax(1) == target).float() * vf).sum().detach() / n}
            # raw (unscaled) agreement of the centered pooled embeddings, the eval-side quantity
            stats["zcos"] = (F.cosine_similarity(za, zp, dim=-1).detach() * vf).sum() / n
            stats["term"] = term.detach()
            loss = self.weight * ramp * term
        return loss, stats
