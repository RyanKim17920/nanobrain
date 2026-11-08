"""Fast pretraining prep: same values as train.sample_transform (resampled path), with the numpy
part planned in the loader workers, cached gaussian kernels, int16 histogram clip thresholds and
in-place writes into the batch tensors. Training only; the eval path (data.py, transform_cuda) is
untouched."""

import functools
import math

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import mni as mni_lib
from .data import brle_to_dense


def ras_orientation(affine: np.ndarray, shape) -> tuple[list[int], list[int], np.ndarray]:
    # data.reorient_to_ras as (flip axes, axis order, new affine)
    ornt = nib.orientations.io_orientation(affine)
    flip_axes = [axis for axis in range(3) if ornt[axis, 1] == -1]
    order = np.argsort(ornt[:, 0]).tolist()
    return flip_axes, order, affine @ nib.orientations.inv_ornt_aff(ornt, shape)


def resample_size(shape, affine: np.ndarray, target_spacing: float = 1.0):
    # data.resample_to_spacing's output size and anti-aliasing sigmas (voxels)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    new_shape = np.maximum(np.round(np.array(shape) * spacing / target_spacing), 1).astype(int)
    sigmas = np.maximum(0, (target_spacing / spacing - 1) / 2)
    return new_shape.tolist(), sigmas.tolist()


def int16_clip(values: np.ndarray, scale: float, clip_quantiles=(0.005, 0.995)):
    # data.normalize_values' nearest rank thresholds of float32(values) * scale, from an int16
    # histogram: x -> x * scale keeps the order for scale > 0
    if values.dtype != np.int16 or not scale > 0:
        return None
    cum = np.cumsum(np.bincount(values.view(np.uint16) ^ np.uint16(0x8000), minlength=65536))
    last = len(values) - 1
    ranks = [round(clip_quantiles[0] * last), round(clip_quantiles[1] * last)]
    ints = np.searchsorted(cum, ranks, side="right") - 32768
    low, high = ints.astype(np.float32) * np.float32(scale)
    return float(low), float(high)


def plan_batch(
    batch: list[dict],
    target_spacing: float = 1.0,
    clip_quantiles=(0.005, 0.995),
    lean: bool = False,
    mni_align: str = "off",
    mni_jitter=None,
    grid_size=None,
):
    # loader collate_fn: the numpy part of the prep, in the workers
    for sample in batch:
        sample["clip"] = int16_clip(sample["values"].numpy(), sample["scale"], clip_quantiles)
        if mni_align != "off":
            # mni.py: one grid_sample from the stored voxels onto the MNI grid; the jitter draws come
            # from the worker's torch rng (seeded per worker by the loader)
            jitter = None
            if mni_jitter is not None and any(float(v) for v in mni_jitter):
                jitter = mni_lib.jitter_matrix(mni_jitter)
            voxel_map, _ = mni_lib.sampling_affine(
                sample["affine"], sample.pop("mni_affine"), mni_align, grid_size, jitter
            )
            sample["mni_theta"] = mni_lib.grid_theta(voxel_map, sample["shape"], grid_size)
            sample["mni_sigmas"] = mni_lib.smoothing_sigmas(sample["affine"], target_spacing)
        else:
            orientation = ras_orientation(sample["affine"], sample["shape"])
            shape = [sample["shape"][axis] for axis in orientation[1]]
            sample["orientation"] = orientation
            sample["size"] = resample_size(shape, orientation[2], target_spacing)
        if lean:
            # the dense mask as bits (brle_to_dense on the cpu), unpacked on the gpu in 2 kernels
            runs = sample.pop("mask_rle").numpy()
            dense = np.repeat(np.arange(len(runs)) % 2 == 1, runs)
            assert len(dense) == math.prod(sample["shape"])
            sample["mask_bits"] = torch.from_numpy(np.packbits(dense))
    if lean:
        return concat_batch(batch)
    return batch


def concat_batch(batch: list[dict]) -> dict:
    # one tensor per key for the whole batch (values, mask bits) and per sample slices, so the loader
    # shares, pins and copies 2 tensors per batch instead of 2 per sample
    out = {"samples": batch}
    for key in ("values", "mask_bits"):
        if len({sample[key].dtype for sample in batch}) > 1:
            continue
        start = 0
        for sample in batch:
            num = len(sample[key])
            sample[key + "_slice"] = (start, start + num)
            start += num
        out[key] = torch.cat([sample.pop(key) for sample in batch])
    return out


def split_batch(batch: dict) -> list[dict]:
    # concat_batch's samples, their tensors as views of the batch tensors
    samples = [dict(sample) for sample in batch["samples"]]
    for key in ("values", "mask_bits"):
        if key in batch:
            for sample in samples:
                start, stop = sample[key + "_slice"]
                sample[key] = batch[key][start:stop]
    return samples


@functools.lru_cache(maxsize=1024)
def gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype, stream: int = 0) -> Tensor:
    # per stream: a kernel made on one stream isn't safe to read on another before it's done
    radius = math.ceil(3 * sigma)
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_smooth(image: Tensor, sigmas: list[float]) -> Tensor:
    # data.gaussian_smooth with cached kernels
    stream = torch.cuda.current_stream(image.device).stream_id if image.is_cuda else 0
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = math.ceil(3 * sigma)
        kernel = gaussian_kernel(sigma, image.device, image.dtype, stream)
        lines = image.movedim(axis, -1)
        lines_shape = lines.shape
        lines = F.conv1d(lines.reshape(-1, 1, lines_shape[-1]), kernel[None, None], padding=radius)
        image = lines.reshape(lines_shape).movedim(-1, axis)
    return image


def normalize_values(values: Tensor, clip_quantiles=(0.005, 0.995), clip=None) -> Tensor:
    # data.normalize_values, optionally with precomputed thresholds
    if clip is None:
        sorted_values = values.sort().values
        last = len(values) - 1
        clip = (
            sorted_values[round(clip_quantiles[0] * last)],
            sorted_values[round(clip_quantiles[1] * last)],
        )
    values = values.clamp(*clip)
    return (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)


def sample_transform(sample: dict, clip_quantiles=(0.005, 0.995), grid_size=None) -> tuple[Tensor, Tensor]:
    # train.sample_transform (resampled path) after plan_batch, without the fit_to_shape copy
    values = sample["values"].float() * sample["scale"]
    values = normalize_values(values, clip_quantiles, clip=sample["clip"])
    mask = brle_to_dense(sample["mask_rle"], shape=sample["shape"])
    image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
    if "mni_theta" in sample:
        image = gaussian_smooth(image, sample["mni_sigmas"])
        image, mask = mni_lib.resample(image, mask, sample["mni_theta"], grid_size)
        return image, mask.to(torch.uint8)
    flip_axes, order, _ = sample["orientation"]
    if flip_axes:  # flip([]) copies
        image, mask = image.flip(flip_axes), mask.flip(flip_axes)
    image, mask = image.permute(order), mask.permute(order)
    new_size, sigmas = sample["size"]
    image = gaussian_smooth(image, sigmas)
    image = F.interpolate(image[None, None], size=new_size, mode="trilinear")[0, 0]
    mask = F.interpolate(mask[None, None].to(torch.uint8), size=new_size, mode="nearest-exact")[0, 0]
    image = image * mask
    return image, mask


_RUN_FLAGS: dict = {}


def run_flags(num: int, device: torch.device) -> Tensor:
    # brle_to_dense's alternating false / true run flags, from a cached tensor per stream
    stream = torch.cuda.current_stream(device).stream_id if device.type == "cuda" else 0
    flags = _RUN_FLAGS.get((device, stream))
    if flags is None or len(flags) < num:
        flags = torch.arange(max(2 * num, 1 << 17), device=device) % 2 == 1
        _RUN_FLAGS[(device, stream)] = flags
    return flags[:num]


_BIT_SHIFTS: dict = {}


def unpack_mask(bits: Tensor, shape) -> Tensor:
    # np.packbits (big endian bit order) -> bool [shape]
    stream = torch.cuda.current_stream(bits.device).stream_id if bits.is_cuda else 0
    shifts = _BIT_SHIFTS.get((bits.device, stream))
    if shifts is None:
        shifts = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
        _BIT_SHIFTS[(bits.device, stream)] = shifts
    dense = torch.bitwise_and(torch.bitwise_right_shift(bits[:, None], shifts), 1)
    return dense.view(-1)[: math.prod(shape)].view(torch.bool).reshape(shape)


def sample_transform_into(sample: dict, image_out: Tensor, mask_out: Tensor, clip_quantiles=(0.005, 0.995)):
    # sample_transform + the fit copy with the same values, in fewer kernels: one cast-and-scale,
    # cached run flags, the bool mask viewed as uint8, and the mask product written into the slot
    values = sample["values"]
    if values.is_floating_point():
        values = values.float() * sample["scale"]
    else:
        values = torch.mul(values, sample["scale"])
    values = normalize_values(values, clip_quantiles, clip=sample["clip"])
    shape = sample["shape"]
    if "mask_bits" in sample:
        mask = unpack_mask(sample["mask_bits"], shape)
    else:
        mask_rle = sample["mask_rle"]
        mask = torch.repeat_interleave(
            run_flags(len(mask_rle), mask_rle.device), mask_rle.long(), output_size=math.prod(shape)
        ).reshape(shape)
    image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
    if "mni_theta" in sample:
        image = gaussian_smooth(image, sample["mni_sigmas"])
        image, mask = mni_lib.resample(image, mask, sample["mni_theta"], image_out.shape)
        image_out.copy_(image)
        mask_out.copy_(mask)
        return
    flip_axes, order, _ = sample["orientation"]
    if flip_axes:  # flip([]) copies
        image, mask = image.flip(flip_axes), mask.flip(flip_axes)
    image, mask = image.permute(order), mask.permute(order)
    new_size, sigmas = sample["size"]
    image = gaussian_smooth(image, sigmas)
    image = F.interpolate(image[None, None], size=new_size, mode="trilinear")[0, 0]
    mask = F.interpolate(mask.view(torch.uint8)[None, None], size=new_size, mode="nearest-exact")[0, 0]
    dst, src = fit_slices(image.shape, image_out.shape)
    torch.mul(image[src], mask[src], out=image_out[dst])
    mask_out[dst] = mask[src]


def fit_slices(shape, target_shape):
    # data.fit_to_shape's center pad / crop as (output, input) slices
    dst, src = [], []
    for size, target in zip(shape, target_shape):
        before = (target - size) // 2
        start = max(before, 0)
        stop = max(min(target, size + before), start)
        dst.append(slice(start, stop))
        src.append(slice(start - before, stop - before))
    return tuple(dst), tuple(src)


def make_batch_transform(
    grid_size,
    target_spacing: float = 1.0,
    clip_quantiles=(0.005, 0.995),
    lean: bool = False,
    mni_align: str = "off",
    mni_jitter=None,
):
    grid_size = tuple(grid_size)

    def fn(batch: list[dict] | dict) -> dict[str, Tensor]:
        # written into zero filled batch tensors; uint8 masks (0/1)
        if isinstance(batch, dict):
            batch = split_batch(batch)
        if "size" not in batch[0] and "mni_theta" not in batch[0]:
            batch = plan_batch(batch, target_spacing, clip_quantiles, False, mni_align, mni_jitter, grid_size)
        device = batch[0]["values"].device
        images = torch.zeros(len(batch), *grid_size, device=device)
        masks = torch.zeros(len(batch), *grid_size, device=device, dtype=torch.uint8)
        for image_out, mask_out, sample in zip(images, masks, batch):
            if lean:
                sample_transform_into(sample, image_out, mask_out, clip_quantiles)
                continue
            image, mask = sample_transform(sample, clip_quantiles, grid_size)
            dst, src = fit_slices(image.shape, grid_size)
            image_out[dst] = image[src]
            mask_out[dst] = mask[src]
        return {"image": images, "mask": masks}

    return fn
