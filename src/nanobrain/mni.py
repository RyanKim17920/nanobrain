"""MNI-aligned model input (mni_align: rigid | affine), the same transform for training and eval.

Every FOMO300K image ships `mni_affine` (image world RAS mm -> MNI152NLin6Asym mm, ANTs 12-dof) and
the brainmarks eval passes the same per sample. The model input grid (grid_size at 1 mm) is a fixed
box in MNI space, centred on the template brain with the x midline (MNI x = 0) exactly between the
two grid halves, so level_mirror's x flip is a flip about the midsagittal plane. Each image is
sampled onto that grid with one trilinear grid_sample from its own voxels (anti-aliased like the
resampled path). rigid keeps head size (rotation of the polar decomposition, fixing the image point
the full affine sends to the brain centre, as brainmarks walnut.py); affine also removes it.
Training adds a small random rigid + isotropic scale jitter about the brain centre (mni_jitter).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# MNI152NLin6Asym 1 mm brain mask: x -70..72, y -106..73, z -72..81 mm, centre of mass (0.6, -21.5,
# 9.8). Grid centre: x = 0 (midline), y / z the brain bounding box centre
MNI_CENTRE = (0.0, -17.0, 5.0)


def normalize_mode(mode) -> str:
    # yaml reads a bare off as False
    if mode in (None, False, "off", "none", "", 0):
        return "off"
    assert mode in ("rigid", "affine"), f"mni_align: off | rigid | affine, got {mode!r}"
    return mode


def input_affine(grid_size) -> np.ndarray:
    # model input voxel -> MNI mm: 1 mm, grid centre at MNI_CENTRE
    affine = np.eye(4)
    affine[:3, 3] = np.array(MNI_CENTRE) - (np.array(grid_size, dtype=np.float64) - 1) / 2
    return affine


def mni_to_image(image_to_mni: np.ndarray, mode: str) -> np.ndarray:
    # MNI mm -> image world mm: the inverse full affine, or its rigid part
    image_to_mni = np.asarray(image_to_mni, dtype=np.float64)
    if mode == "affine":
        return np.linalg.inv(image_to_mni)
    u, _, vt = np.linalg.svd(image_to_mni[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # never for a RAS -> RAS registration; keep it proper
        u[:, -1] *= -1
        rotation = u @ vt
    centre = np.array([*MNI_CENTRE, 1.0])
    image_centre = np.linalg.solve(image_to_mni, centre)[:3]
    out = np.eye(4)
    out[:3, :3] = rotation.T
    out[:3, 3] = image_centre - rotation.T @ centre[:3]
    return out


def jitter_matrix(jitter, rand: Tensor | None = None) -> np.ndarray:
    # MNI mm -> MNI mm: x -> c + s R (x - c) + t about MNI_CENTRE; rotation U(+-deg) per axis
    # (x then y then z), translation U(+-mm) per axis, scale exp(U(+-frac)). rand: 7 U(0, 1) draws
    rot_deg, trans_mm, scale_frac = (float(v) for v in jitter)
    if rand is None:
        rand = torch.rand(7, dtype=torch.float64)
    u = 2 * rand.double().numpy() - 1
    ax, ay, az = np.deg2rad(rot_deg * u[:3])
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    linear = math.exp(scale_frac * u[6]) * (rz @ ry @ rx)
    centre = np.array(MNI_CENTRE)
    out = np.eye(4)
    out[:3, :3] = linear
    out[:3, 3] = centre - linear @ centre + trans_mm * u[3:6]
    return out


def sampling_affine(
    image_affine: np.ndarray, image_to_mni: np.ndarray, mode: str, grid_size, jitter=None
) -> tuple[np.ndarray, np.ndarray]:
    # (model input voxel -> image voxel, model input voxel -> image world mm)
    to_mni = input_affine(grid_size)
    if jitter is not None:
        to_mni = jitter @ to_mni
    to_world = mni_to_image(image_to_mni, mode) @ to_mni
    return np.linalg.inv(np.asarray(image_affine, dtype=np.float64)) @ to_world, to_world


def smoothing_sigmas(image_affine: np.ndarray, target_spacing: float = 1.0) -> list[float]:
    # anti-aliasing per image axis, the resample_to_spacing rule (skimage resize)
    spacing = np.linalg.norm(np.asarray(image_affine)[:3, :3], axis=0)
    return np.maximum(0, (target_spacing / spacing - 1) / 2).tolist()


def grid_theta(voxel_map: np.ndarray, in_shape, out_shape) -> Tensor:
    # F.affine_grid theta (align_corners=True) for output voxel v -> input voxel voxel_map @ v.
    # normalized n = 2 v / (S - 1) - 1, axes reversed (grid x = last dim)
    s_in = np.array(in_shape, dtype=np.float64)
    s_out = np.array(out_shape, dtype=np.float64)
    d_in = 2 / np.maximum(s_in - 1, 1)
    d_out_inv = np.maximum(s_out - 1, 1) / 2
    a, t = voxel_map[:3, :3], voxel_map[:3, 3]
    linear = d_in[:, None] * a * d_out_inv[None, :]
    offset = d_in * (a @ d_out_inv + t) - 1
    theta = np.concatenate([linear, offset[:, None]], axis=1)[::-1]  # rows to (x, y, z) = (k, j, i)
    theta = np.concatenate([theta[:, :3][:, ::-1], theta[:, 3:]], axis=1)  # columns too
    return torch.from_numpy(np.ascontiguousarray(theta, dtype=np.float32))


def resample(image: Tensor, mask: Tensor, theta: Tensor, grid_size) -> tuple[Tensor, Tensor]:
    # image (normalized, 0 outside the mask, smoothed) and mask on the image voxels -> the model
    # grid: one trilinear grid_sample of both (float32 grid: bf16 coordinates are off by voxels);
    # mask = sampled mask > 0.5, image times it
    stacked = torch.stack([image.float(), mask.float()])[None]
    grid = F.affine_grid(
        theta.to(image.device, torch.float32)[None], (1, 2, *grid_size), align_corners=True
    )
    out = F.grid_sample(stacked, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    mask_out = out[1] > 0.5
    return out[0] * mask_out, mask_out
