"""Project a fused object into a camera view to get a 2D box in ORIGINAL pixels.

Used when the program's selected object has no native detection in the query
view: the object is located (fused from other views), so its position in the
query view follows from the pipeline's own reconstruction — no ground truth
involved.

VGGT works in a resized+letterboxed space (long side -> 518, short side to a
multiple of 14, centre-padded); intrinsics live in that space, boxes in
original pixels, so the projection inverts that transform.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def _vggt_to_original(u: np.ndarray, v: np.ndarray, view_size, orig_w: int, orig_h: int):
    Hv, Wv = view_size
    if orig_w >= orig_h:
        s = Wv / orig_w
        prep_h = round(orig_h * s / 14) * 14
        pad_top = (Hv - prep_h) / 2.0
        return u / s, (v - pad_top) / (prep_h / orig_h)
    s = Hv / orig_h
    prep_w = round(orig_w * s / 14) * 14
    pad_left = (Wv - prep_w) / 2.0
    return (u - pad_left) / (prep_w / orig_w), v / s


def _cam_project(P, K, E):
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Pc = E[:3, :3] @ P.T + E[:3, 3:4]
    z = Pc[2]
    keep = z > 1e-6
    uv = (K @ Pc)[:2] / np.where(keep, z, 1.0)
    return uv[0][keep], uv[1][keep]


def project_object_bbox(obj, camera, orig_w: int, orig_h: int) -> Optional[list]:
    """Axis-aligned box of ``obj`` in ``camera``'s ORIGINAL image, or None.

    Prefers the fitted 3D box corners; falls back to the point cloud with a
    1/99-percentile trim. The cloud contains only VISIBLE-surface points (it
    comes from per-view depth maps), so this box approximates the object's
    visible extent.
    """
    K = np.asarray(camera.intrinsics, dtype=float)
    E = np.asarray(camera.extrinsics, dtype=float)
    for P, trim in ((getattr(obj, "corners_world", None), False), (getattr(obj, "world_points", None), True)):
        if P is None or len(np.atleast_2d(P)) == 0:
            continue
        u, v = _cam_project(P, K, E)
        if not u.size:
            continue
        x, y = _vggt_to_original(u, v, camera.image_size, orig_w, orig_h)
        if trim:
            def lo(a):
                return float(np.percentile(a, 1))

            def hi(a):
                return float(np.percentile(a, 99))
        else:
            def lo(a):
                return float(a.min())

            def hi(a):
                return float(a.max())
        b = [max(0.0, lo(x)), max(0.0, lo(y)), min(float(orig_w), hi(x)), min(float(orig_h), hi(y))]
        if b[2] > b[0] and b[3] > b[1]:
            return b
    return None
