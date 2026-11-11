import io
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data import Dataset, Sampler


class Fomo300kDataset(Dataset):
    def __init__(self, root: str | Path, filelist: str | Path):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().split()
        members = pd.read_parquet(
            self.root / "tables" / "shards.parquet", columns=["path", "shard", "offset", "size"]
        )
        members = members.set_index("path").loc[self.paths]
        self.shards = members["shard"].tolist()
        self.offsets = members["offset"].to_numpy()
        self.sizes = members["size"].to_numpy()
        # fino3d per-image metadata [N, 6] (nanobrain.fino.build_meta), None = off
        self.meta = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        # npz members are stored uncompressed, so read them straight from their offset
        with open(self.root / self.shards[index], "rb") as shard:
            shard.seek(self.offsets[index])
            data = shard.read(self.sizes[index])
        with np.load(io.BytesIO(data)) as npz:
            sample = {
                "path": self.paths[index],
                "values": torch.from_numpy(npz["values"]),
                "scale": float(npz["scale"]),
                "mask_rle": torch.from_numpy(npz["mask_rle"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": npz["affine"],
            }
        if self.meta is not None:
            # numpy, so send_data leaves it on the cpu (no per-step host <-> device sync)
            sample["fino"] = self.meta[index].numpy()
        return sample


def subject_temper_weights(paths: list[str], alpha: float) -> torch.Tensor:
    """Per-image sampling weights n_subject^(-alpha), so a subject with n scans is drawn with total
    mass n^(1-alpha): alpha 0 = uniform over images, 1 = uniform over subjects. The subject key is
    the collection / sub-* part of the path (derivatives/<collection>/<sub>/...)."""
    keys = []
    for path in paths:
        # up to the first sub-* segment (OpenNeuro nests a dsXXXXXX level before it)
        parts = path.split("/")
        cut = next((ii for ii, part in enumerate(parts) if part.startswith("sub")), 2)
        keys.append("/".join(parts[: cut + 1]))
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    n = torch.tensor([counts[key] for key in keys], dtype=torch.float64)
    return n.pow(-alpha)


def subject_counts(paths: list[str]) -> tuple[list[str], torch.Tensor]:
    """Subject key of every path (as in subject_temper_weights) and n_subject per image."""
    keys = []
    for path in paths:
        parts = path.split("/")
        cut = next((ii for ii, part in enumerate(parts) if part.startswith("sub")), 2)
        keys.append("/".join(parts[: cut + 1]))
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return keys, torch.tensor([counts[key] for key in keys], dtype=torch.float64)


def subject_cap_weights(paths: list[str], cap: float, soft: bool = True) -> torch.Tensor:
    """Per-image weights that cap each subject's total draw mass near `cap` images: soft
    (cap * tanh(n / cap) / n, a smooth knee) or hard (min(1, cap / n)). Subjects with n << cap
    keep their full mass n (unlike the n^-alpha power law, which shrinks every subject); only the
    heavily repeated tail (multi-shell DWI, longitudinal) is flattened."""
    _, n = subject_counts(paths)
    if soft:
        return cap * torch.tanh(n / cap) / n
    return (cap / n).clamp(max=1.0)


def effective_subjects(paths: list[str], weights: torch.Tensor) -> float:
    """1 / sum_s p_s^2 of the subject draw distribution under per-image weights."""
    keys, _ = subject_counts(paths)
    prob = (weights / weights.sum()).tolist()
    mass: dict[str, float] = {}
    for key, pr in zip(keys, prob):
        mass[key] = mass.get(key, 0.0) + pr
    return 1.0 / sum(v * v for v in mass.values())


class InfiniteSampler(Sampler[int]):
    """Endless shuffled passes over the dataset, starting from sample index `start`. With ddp
    each rank takes every world_size-th sample of the same stream."""

    def __init__(
        self,
        num_samples: int,
        seed: int = 0,
        start: int = 0,
        rank: int = 0,
        world_size: int = 1,
        late_indices: list[int] | None = None,
        switch_at: int | None = None,
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.start = start
        self.rank = rank
        self.world_size = world_size
        # from sample index switch_at on, endless shuffled passes over late_indices only (anneal mix)
        self.late_indices = late_indices
        self.switch_at = switch_at if late_indices is not None else None
        # optional per-image sampling weights (subject tempering): passes of num_samples draws
        # with replacement instead of permutations (None = uniform permutations, the default)
        self.weights = None

    def _pass(self, generator):
        if self.weights is None:
            return torch.randperm(self.num_samples, generator=generator).tolist()
        return torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        ).tolist()

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        index = 0
        while self.switch_at is None or index < self.switch_at:
            for sample_index in self._pass(generator):
                if self.switch_at is not None and index >= self.switch_at:
                    break
                if index >= self.start and index % self.world_size == self.rank:
                    yield sample_index
                index += 1
        late_generator = torch.Generator().manual_seed(self.seed + 1)
        while True:
            for ii in torch.randperm(len(self.late_indices), generator=late_generator).tolist():
                if index >= self.start and index % self.world_size == self.rank:
                    yield self.late_indices[ii]
                index += 1


def brle_to_dense(mask_rle: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    # mask_rle is a binary run length encoding, alternating false and true runs starting
    # with false
    X, Y, Z = shape
    run_lengths = mask_rle.long()
    run_flags = torch.arange(len(run_lengths), device=mask_rle.device) % 2 == 1
    mask = torch.repeat_interleave(run_flags, run_lengths, output_size=X * Y * Z)
    return mask.reshape(shape)


def crop_to_mask(
    image: Float[Tensor, "X Y Z"], mask: Bool[Tensor, "X Y Z"], affine: np.ndarray
) -> tuple[Tensor, Tensor, np.ndarray]:
    starts = []
    stops = []
    for axis in range(3):
        other_axes = tuple(other for other in range(3) if other != axis)
        ids = mask.any(dim=other_axes).nonzero()[:, 0]
        starts.append(ids.min().item())
        stops.append(ids.max().item() + 1)
    crop = tuple(slice(start, stop) for start, stop in zip(starts, stops))

    shift = np.eye(4)
    shift[:3, 3] = starts
    return image[crop], mask[crop], affine @ shift


def reorient_to_ras(
    image: Float[Tensor, "X Y Z"], mask: Bool[Tensor, "X Y Z"], affine: np.ndarray
) -> tuple[Tensor, Tensor, np.ndarray]:
    # same as nibabel apply_orientation: flip input axes, then permute
    ornt = nib.orientations.io_orientation(affine)
    flip_axes = [axis for axis in range(3) if ornt[axis, 1] == -1]
    order = np.argsort(ornt[:, 0]).tolist()
    new_affine = affine @ nib.orientations.inv_ornt_aff(ornt, image.shape)
    image = image.flip(flip_axes).permute(order)
    mask = mask.flip(flip_axes).permute(order)
    return image, mask, new_affine


def native_spacing(affine: np.ndarray, target_spacing: float = 1.0) -> tuple[float, float, float]:
    # voxel size per axis. header float noise around target_spacing snaps to it, the rest rounds to
    # 1e-4 mm so scans from one protocol share a spacing
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return tuple(
        target_spacing if abs(sp - target_spacing) < 1e-3 else round(float(sp), 4) for sp in spacing
    )


def native_box(
    spacing: tuple[float, float, float], box_mm: tuple[int, int, int]
) -> tuple[int, int, int]:
    # the box_mm box in voxels of this spacing, rounded up unless whole up to float noise
    shape = []
    for mm, sp in zip(box_mm, spacing):
        size = mm / sp
        shape.append(round(size) if abs(size - round(size)) < 1e-2 else math.ceil(size))
    return tuple(shape)


def resample_to_spacing(
    image: Float[Tensor, "X Y Z"],
    mask: Bool[Tensor, "X Y Z"],
    affine: np.ndarray,
    target_spacing: float = 1.0,
) -> tuple[Tensor, Float[Tensor, "X Y Z"], np.ndarray]:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    shape = np.array(image.shape)
    new_shape = np.maximum(np.round(shape * spacing / target_spacing), 1).astype(int)

    # anti-aliasing for downsampled axes, sigma rule from skimage.transform.resize
    sigmas = np.maximum(0, (target_spacing / spacing - 1) / 2)
    image = gaussian_smooth(image, sigmas.tolist())

    new_size = new_shape.tolist()
    image = F.interpolate(image[None, None], size=new_size, mode="trilinear")[0, 0]
    mask = F.interpolate(mask[None, None].float(), size=new_size, mode="nearest-exact")[0, 0]
    image = image * mask

    # interpolate maps output voxel i to input voxel (i + 0.5) * scale - 0.5
    scale = shape / new_shape
    resample = np.eye(4)
    resample[:3, :3] = np.diag(scale)
    resample[:3, 3] = 0.5 * scale - 0.5
    return image, mask, affine @ resample


def gaussian_smooth(image: Float[Tensor, "X Y Z"], sigmas: list[float]) -> Float[Tensor, "X Y Z"]:
    # separable gaussian, sigmas in voxels per axis, zero padded
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = math.ceil(3 * sigma)
        offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        lines = image.movedim(axis, -1)
        lines_shape = lines.shape
        lines = F.conv1d(lines.reshape(-1, 1, lines_shape[-1]), kernel[None, None], padding=radius)
        image = lines.reshape(lines_shape).movedim(-1, axis)
    return image


def fit_to_shape(
    image: Float[Tensor, "X Y Z"],
    mask: Float[Tensor, "X Y Z"],
    affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[Tensor, Tensor, np.ndarray]:
    # center pad or crop. F.pad takes pads last dim first, negative pads crop
    befores = []
    pad = []
    for axis in reversed(range(3)):
        diff = target_shape[axis] - image.shape[axis]
        before = diff // 2
        befores.insert(0, before)
        pad += [before, diff - before]

    shift = np.eye(4)
    shift[:3, 3] = -np.array(befores)
    return F.pad(image, pad), F.pad(mask, pad), affine @ shift


def normalize_values(
    values: Float[Tensor, " N"], clip_quantiles: tuple[float, float] = (0.005, 0.995)
) -> Float[Tensor, " N"]:
    # nearest rank clip thresholds. torch.quantile needs a q tensor, and copying it to the gpu syncs
    # cf https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/
    # nb, another option would a torch module with fixed q tensor buffer. but can't be bothered.
    sorted_values = values.sort().values
    last = len(values) - 1
    low = sorted_values[round(clip_quantiles[0] * last)]
    high = sorted_values[round(clip_quantiles[1] * last)]
    values = values.clamp(low, high)
    values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
    return values
