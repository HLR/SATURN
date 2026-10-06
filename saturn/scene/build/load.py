"""
load_scene — entry point for multi-view scene construction.

Wires together SAM3 segmentation, VGGT multi-view reconstruction, OrientAnything
orientation estimation, cross-view fusion, ground leveling, and returns
a fully-populated Scene object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import saturn.perception.cv_compat as cv_compat
from saturn.perception.geometry.filtering import erode_mask
from saturn.scene.fusion import (
    apply_ground_leveling,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extrinsic_to_4x4(ext: np.ndarray) -> np.ndarray:
    """Promote a (3,4) extrinsic to (4,4)."""
    ext = np.asarray(ext, dtype=float)
    if ext.shape == (4, 4):
        return ext
    m = np.eye(4, dtype=float)
    m[:3, :] = ext[:3, :]
    return m


def _level_merged_dicts(merged: List[Dict[str, Any]], ground_info: Dict[str, Any]) -> None:
    """Move fused object dicts from the reconstruction frame into the
    ground-leveled world, in place. Every world-frame field moves, including
    ``per_view_centers`` (points) and ``per_view_fronts`` (directions);
    ``per_view_fronts_camera`` is camera-frame and stays as is."""
    rot = ground_info["rotation"]
    y_shift = ground_info["y_shift"]
    for obj_dict in merged:
        obj_dict["center_world"] = apply_ground_leveling(
            rot, y_shift, obj_dict["center_world"].reshape(1, 3)
        ).ravel()
        if obj_dict.get("corners_world") is not None:
            obj_dict["corners_world"] = apply_ground_leveling(
                rot, y_shift, obj_dict["corners_world"]
            )
        obj_dict["rotation_world"] = rot @ obj_dict["rotation_world"]
        obj_dict["front_world"] = rot @ obj_dict["front_world"]
        obj_dict["up_world"] = rot @ obj_dict["up_world"]
        obj_dict["right_world"] = rot @ obj_dict["right_world"]
        if obj_dict.get("world_points") is not None:
            obj_dict["world_points"] = apply_ground_leveling(
                rot, y_shift, obj_dict["world_points"]
            )
        if obj_dict.get("per_view_centers"):
            obj_dict["per_view_centers"] = {
                v: apply_ground_leveling(
                    rot, y_shift, np.asarray(c, dtype=float).reshape(1, 3)
                ).ravel()
                for v, c in obj_dict["per_view_centers"].items()
            }
        if obj_dict.get("per_view_fronts"):
            obj_dict["per_view_fronts"] = {
                v: rot @ np.asarray(f, dtype=float)
                for v, f in obj_dict["per_view_fronts"].items()
            }


def _merge_new_observations(observations, scene, unique_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fuse post-build detect()/ground() observations, then level them.

    The observations are back-projected through the stashed, UNLEVELED
    ``scene._world_geometries``, while ``scene.cameras`` were leveled at build
    time. Fusion's depth gates compare points with camera centres, so hand it
    the camera centres mapped back into the reconstruction frame, and level
    the merged result afterwards. ``unique_keyword``: the planner says there is
    one such object (see fusion.split_unique_track).
    """
    from saturn.scene.fusion import _scene_scale_from_cameras, merge_objects_by_keyword

    _uk = {unique_keyword} if unique_keyword else None
    ground_info = getattr(scene, "ground_info", None)
    if ground_info is None:
        return merge_objects_by_keyword(observations, cameras=scene.cameras, unique_keywords=_uk)
    rot = np.asarray(ground_info["rotation"], dtype=float)
    shift = np.array([0.0, float(ground_info["y_shift"]), 0.0])
    cam_positions = {}
    for ci, cam in enumerate(scene.cameras):
        pos = getattr(cam, "position_world", None)
        if pos is None:
            ext = np.asarray(cam.extrinsics, dtype=float)
            pos = -ext[:3, :3].T @ ext[:3, 3]
        cam_positions[int(getattr(cam, "id", ci))] = rot.T @ (np.asarray(pos, dtype=float) - shift)
    merged = merge_objects_by_keyword(
        observations,
        cam_positions=cam_positions,
        # rigid leveling preserves camera-camera distances
        scene_scale=_scene_scale_from_cameras(scene.cameras),
        unique_keywords=_uk,
    )
    _level_merged_dicts(merged, ground_info)
    return merged


def _resize_mask_to_prediction(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a mask to match depth-prediction spatial dimensions."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape == tuple(target_shape):
        return mask.astype(bool)
    resized = cv_compat.resize(
        mask, (target_shape[1], target_shape[0]), interpolation=cv_compat.INTER_NEAREST
    )
    return resized.astype(bool)


def _bbox_to_mask(
    image_size: Tuple[int, int], bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    """Build a binary mask from an XYXY bbox in image coordinates."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1 = int(np.floor(max(0.0, min(x1, width))))
    x2 = int(np.ceil(max(0.0, min(x2, width))))
    y1 = int(np.floor(max(0.0, min(y1, height))))
    y2 = int(np.ceil(max(0.0, min(y2, height))))

    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


# ---------------------------------------------------------------------------
# World geometry from a per-view depth prediction
# ---------------------------------------------------------------------------


def _world_geometry_from_depth(prediction) -> Dict[str, Any]:
    """Build world-space geometry from one view's depth prediction
    (``depth``, ``confidence``, ``intrinsics``, ``extrinsics``) by pinhole
    back-projection through the intrinsics and extrinsics.

    Returns dict with keys: depth, confidence, ray_confidence, intrinsics,
    extrinsics, ray_origins_world, ray_directions_world, world_points.
    """
    depth = np.asarray(prediction.depth, dtype=float)
    conf = np.asarray(prediction.confidence, dtype=float)
    intrinsics = np.asarray(prediction.intrinsics, dtype=float)

    # Standard pinhole back-projection
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    rays_cam = np.stack(
        [
            (xs - intrinsics[0, 2]) / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(depth),
        ],
        axis=-1,
    )
    ext = _extrinsic_to_4x4(prediction.extrinsics)
    c2w = np.linalg.inv(ext)
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    ray_directions_world = rays_cam @ R_c2w.T
    ray_origins_world = np.broadcast_to(t_c2w, ray_directions_world.shape).copy()
    ray_conf = conf
    world_points = ray_origins_world + depth[..., None] * ray_directions_world

    return {
        "depth": depth,
        "confidence": conf,
        "ray_confidence": ray_conf,
        "intrinsics": intrinsics,
        "extrinsics": np.asarray(prediction.extrinsics, dtype=float),
        "ray_origins_world": ray_origins_world,
        "ray_directions_world": ray_directions_world,
        "world_points": world_points,
    }


def _depth_cluster_mask(
    depth_values: np.ndarray,
    k_mad: float = 3.0,
    min_spread_m: float = 0.12,
    min_survivors_frac: float = 0.2,
    bimodal_rel_gap: float = 0.15,
    bimodal_min_cluster_frac: float = 0.15,
) -> np.ndarray:
    """Return a boolean mask over ``depth_values`` keeping the dominant cluster.

    Two-stage filter:

    1. **Bimodality pre-pass.** Sort depths and look for the largest gap.
       If that gap is >= ``bimodal_rel_gap`` of the total depth span AND
       each side holds >= ``bimodal_min_cluster_frac`` of the samples,
       the distribution is bimodal (e.g. a door mask bleeding onto the
       wall ~1m behind it). In that case, keep ONLY the nearer cluster
       (smaller camera-Z depth). The MAD filter fails on this pattern
       because the median falls into the gap and MAD inflates so much
       that the 3-sigma window straddles both clusters, making the
       filter a no-op.

    2. **MAD trim.** Apply the standard median-absolute-deviation filter
       on the (possibly already-halved) set. If fewer than
       ``min_survivors_frac`` of points survive (indicating the median
       fell onto an outlier cluster or the filter is too tight), fall
       back to an IQR trim (p5..p95).

    This rejects depth-bleed through SAM3 masks onto surfaces behind the
    object.
    """
    if depth_values.size == 0:
        return np.zeros(0, dtype=bool)

    # Track the active indices against the original array so the final
    # returned mask aligns with the caller's view.
    active_idx = np.arange(depth_values.size)

    # --- Stage 1: bimodality pre-pass ---
    if depth_values.size >= 20:
        order = np.argsort(depth_values)
        ds = depth_values[order]
        total_span = float(ds[-1] - ds[0])
        if total_span > 2.0 * min_spread_m:
            gaps = np.diff(ds)
            gi = int(np.argmax(gaps))
            gap_val = float(gaps[gi])
            rel_gap = gap_val / total_span if total_span > 1e-9 else 0.0
            left = gi + 1                          # near-cluster size
            right = ds.size - left                 # far-cluster size
            left_frac = left / ds.size
            right_frac = right / ds.size
            if (
                rel_gap >= bimodal_rel_gap
                and min(left_frac, right_frac) >= bimodal_min_cluster_frac
            ):
                # Bimodal. Keep the NEAR cluster (smaller depth -> first
                # ``left`` entries in the sorted order).
                active_idx = order[:left]

    # --- Stage 2: MAD trim on the active set ---
    active_depths = depth_values[active_idx]
    med = float(np.median(active_depths))
    mad = float(np.median(np.abs(active_depths - med)))
    sigma = max(mad * 1.4826, min_spread_m)
    sub_keep = np.abs(active_depths - med) <= k_mad * sigma
    frac = float(sub_keep.mean()) if sub_keep.size else 0.0
    if sub_keep.size and frac < min_survivors_frac:
        lo, hi = np.percentile(active_depths, [5, 95])
        sub_keep = (active_depths >= lo) & (active_depths <= hi)

    keep = np.zeros(depth_values.size, dtype=bool)
    keep[active_idx[sub_keep]] = True
    return keep


def _extract_mask_world_points(
    geometry: Dict[str, Any],
    mask: np.ndarray,
    conf_percentile: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract world points within a mask, filtering by confidence and depth.

    Filtering stages (in order):
      1. Adaptive mask erosion to pull the mask away from object edges.
      2. Confidence filter: keep points with confidence above ``conf_percentile``.
      3. Depth cluster filter: keep points within the dominant depth cluster
         (MAD-based with IQR fallback). Removes back-wall bleed through the
         mask when SAM3 returns a pixel-perfect silhouette whose interior
         pixels still include gaps to far-away surfaces.

    Returns (world_points, point_confidence, resized_mask).
    """
    if geometry.get("source") == "vggt":
        # VGGT path: transform mask through VGGT's preprocessing pipeline
        from saturn.perception.reconstruction.vggt import transform_mask_to_vggt_space

        mask_rs = transform_mask_to_vggt_space(mask, geometry["transform_info"])
        # Ensure mask matches world_points spatial dims
        wp_shape = geometry["world_points"].shape[:2]  # (H, W)
        if mask_rs.shape != wp_shape:
            mask_rs = _resize_mask_to_prediction(mask_rs, wp_shape)
    else:
        mask_rs = _resize_mask_to_prediction(mask, geometry["depth"].shape)

    # Adaptive mask erosion — shrink mask edges just enough to reduce depth
    # bleed without shaving large objects twice.
    mask_rs = erode_mask(mask_rs)

    valid = mask_rs & np.isfinite(geometry["depth"]) & (geometry["depth"] > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), mask_rs

    conf = geometry.get("ray_confidence", geometry["confidence"])
    conf_vals = conf[valid]
    if len(conf_vals) > 0 and conf_percentile is not None:
        conf_cut = np.percentile(conf_vals, conf_percentile)
        valid_filtered = valid & (conf >= conf_cut)
        if np.any(valid_filtered):
            valid = valid_filtered

    # Depth cluster filter: reject back-wall / foreground-occluder bleed.
    depth_vals = geometry["depth"][valid]
    if depth_vals.size >= 20:
        keep = _depth_cluster_mask(depth_vals)
        if keep.any():
            # Rebuild valid mask keeping only the dominant-cluster pixels.
            valid_idx = np.flatnonzero(valid.ravel())
            new_valid = np.zeros_like(valid, dtype=bool).ravel()
            new_valid[valid_idx[keep]] = True
            valid = new_valid.reshape(valid.shape)

    points_world = geometry["world_points"][valid]
    conf_valid = geometry["confidence"][valid]
    return points_world, conf_valid, mask_rs


def _make_static_depth_provider(prediction):
    """A depth provider whose ``predict()`` returns *prediction*'s depth and
    confidence as a DepthEstimate."""
    from saturn.perception.types import DepthEstimate

    depth_est = DepthEstimate(
        depth_map=prediction.depth,
        confidence_map=prediction.confidence,
        is_metric=prediction.is_metric,
        source="vggt",
    )

    class _StaticDepthProvider:
        def __init__(self, de):
            self._de = de

        def predict(self, image):
            return self._de

    return _StaticDepthProvider(depth_est)
