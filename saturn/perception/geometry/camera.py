"""Pinhole intrinsics and pixel<->ray helpers for depth-based cameras.

Pure numpy geometry; must not import saturn.vlm/serving.
"""
from typing import Optional, Tuple

import numpy as np

from ..types import CameraModel, DepthEstimate


def intrinsics_from_image_size(image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    focal = float(max(width, height))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def camera_from_depth_estimate(image_size: Tuple[int, int], depth: Optional[DepthEstimate]) -> CameraModel:
    if depth is None:
        return CameraModel(
            intrinsics=intrinsics_from_image_size(image_size),
            extrinsics=None,
            image_size=image_size,
            metadata={"intrinsics_source": "approximate"},
        )
    raw = depth.raw
    intrinsics = getattr(raw, "intrinsics", None) if raw is not None else None
    extrinsics = getattr(raw, "extrinsics", None) if raw is not None else None
    metadata = {"intrinsics_source": "backend" if intrinsics is not None else "approximate"}
    return CameraModel(
        intrinsics=intrinsics if intrinsics is not None else intrinsics_from_image_size(image_size),
        extrinsics=extrinsics,
        image_size=image_size,
        metadata=metadata,
    )


def pixel_to_3d(pixel, depth_value: float, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (pixel[0] - cx) * depth_value / fx
    y_cam = (pixel[1] - cy) * depth_value / fy
    return np.array([x_cam, y_cam, depth_value], dtype=float)
