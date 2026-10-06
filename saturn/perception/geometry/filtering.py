"""Statistical outlier filtering of per-object point clouds (depth-cluster / MAD).

Pure numpy geometry; must not import saturn.vlm/serving.
"""
from typing import Tuple

import numpy as np
import torch
from PIL import Image

import saturn.perception.cv_compat as cv_compat


def normalize_mask(mask) -> np.ndarray:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask_arr = mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.squeeze(mask_arr)
    if mask_arr.dtype != np.bool_:
        if np.issubdtype(mask_arr.dtype, np.floating):
            mask_arr = mask_arr > 0.5
        else:
            mask_arr = mask_arr > 0
    return mask_arr.astype(bool)


def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return None
    if mask.shape == shape:
        return mask.astype(bool)
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    pil_mask = pil_mask.resize((shape[1], shape[0]), Image.NEAREST)
    return (np.asarray(pil_mask) > 128)


def denoise_point_cloud(
    points: np.ndarray,
    depth_percentiles: Tuple[float, float] = (2.0, 98.0),
    mad_k: float = 3.5,
    min_points: int = 20,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    points = np.asarray(points, dtype=float)
    finite_mask = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    clean_points = points[finite_mask]
    if len(clean_points) < min_points:
        return clean_points
    z_low, z_high = np.percentile(clean_points[:, 2], depth_percentiles)
    clean_points = clean_points[(clean_points[:, 2] >= z_low) & (clean_points[:, 2] <= z_high)]
    if len(clean_points) < min_points:
        return clean_points
    med = np.median(clean_points, axis=0)
    mad = np.median(np.abs(clean_points - med), axis=0)
    robust_scale = 1.4826 * mad + 1e-8
    axis_mask = np.all(np.abs(clean_points - med) <= (mad_k * robust_scale), axis=1)
    return clean_points[axis_mask] if np.sum(axis_mask) >= min_points else clean_points


def robust_depth_and_center(points: np.ndarray):
    if points is None or len(points) == 0:
        return 0.0, np.zeros(3, dtype=float)
    z_vals = points[:, 2]
    z_low, z_high = np.percentile(z_vals, [2.0, 98.0])
    inliers = (z_vals >= z_low) & (z_vals <= z_high)
    pos_points = points[inliers] if np.sum(inliers) >= 10 else points
    z_valid = pos_points[:, 2]
    z_valid = z_valid[np.isfinite(z_valid) & (z_valid > 0)]
    if len(z_valid) >= 1:
        depth_value = float(np.percentile(z_valid, 10.0))
    else:
        depth_value = float(np.percentile(z_vals[np.isfinite(z_vals) & (z_vals > 0)], 10.0) if np.any(np.isfinite(z_vals) & (z_vals > 0)) else np.median(z_vals))
    center = np.array(
        [float(np.mean(pos_points[:, 0])), float(np.mean(pos_points[:, 1])), depth_value],
        dtype=float,
    )
    return depth_value, center


def erode_mask(mask: np.ndarray) -> np.ndarray:
    mask_area = int(mask.sum())
    # Cap erosion at a single pass. Two passes trims large masks too much and
    # can undershoot the final 3D box fit.
    if mask_area >= 500:
        iterations = 1
    else:
        iterations = 0
    if iterations == 0:
        return mask
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv_compat.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    if int(eroded.sum()) >= max(5, int(mask_area * 0.1)):
        return eroded.astype(bool)
    return mask
