"""Back-projection of depth maps into camera-frame point clouds.

Pure numpy geometry; must not import saturn.vlm/serving.
"""
import numpy as np

from .filtering import erode_mask, normalize_mask, resize_mask_to_shape


def point_cloud_from_depth_and_mask(depth_map, intrinsics, mask=None, bbox=None):
    h, w = depth_map.shape[:2]
    if mask is not None:
        mask_arr = normalize_mask(mask)
        mask_arr = resize_mask_to_shape(mask_arr, (h, w))
        mask_arr = erode_mask(mask_arr)
        ys, xs = np.where(mask_arr)
    elif bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        ys, xs = np.mgrid[y1:y2, x1:x2]
        ys = ys.reshape(-1)
        xs = xs.reshape(-1)
    else:
        return np.zeros((0, 3), dtype=float), None
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=float), None
    zs = depth_map[ys, xs]
    valid = np.isfinite(zs) & (zs > 0)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=float), None
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    points = np.column_stack([x_cam, y_cam, zs]).astype(float)
    return points, valid
