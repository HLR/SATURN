"""Per-object 3D extraction: mask + depth -> object points, centre and extent.

Pure numpy geometry; must not import saturn.vlm/serving.
"""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from saturn.perception.orientation.convention import (
    backend_euler_to_user,
    front_direction_2d_from_user_azimuth,
    front_direction_3d_from_user_azimuth,
    rotation_matrix_from_user_euler,
    user_euler_from_rotation_matrix,
)
from ..types import (
    OrientationEstimate,
    PoseEstimate3D,
    SceneObject3D,
    SizeEstimate3D,
)
from .camera import camera_from_depth_estimate
from .filtering import denoise_point_cloud, erode_mask, normalize_mask, resize_mask_to_shape, robust_depth_and_center
from .pointcloud import point_cloud_from_depth_and_mask
from .pose_fusion import estimate_extent_from_depth, fit_visible_bbox, refine_orientation_with_depth


def _prediction_from_depth_estimate(depth_estimate):
    if depth_estimate is None:
        return None
    raw = depth_estimate.raw
    if raw is not None and hasattr(raw, "depth"):
        return raw
    return SimpleNamespace(
        depth=np.asarray(depth_estimate.depth_map),
        confidence=(
            np.asarray(depth_estimate.confidence_map)
            if depth_estimate.confidence_map is not None
            else np.ones_like(depth_estimate.depth_map, dtype=float)
        ),
        intrinsics=getattr(raw, "intrinsics", None) if raw is not None else None,
        extrinsics=getattr(raw, "extrinsics", None) if raw is not None else None,
        is_metric=bool(depth_estimate.is_metric),
    )


def _orientation_raw_results(orientation_provider, image, detections):
    if orientation_provider is None:
        return []
    extractor = getattr(orientation_provider, "extractor", None)
    estimator = getattr(orientation_provider, "estimator", None)
    bboxes = [det.bbox_xyxy for det in detections]
    masks = [det.mask for det in detections]
    if extractor is not None and estimator is not None:
        object_images = extractor.extract_object_images(image, np.asarray(bboxes), masks=masks)
        if not object_images:
            return []
        estimate_raw = getattr(orientation_provider, "_estimate_raw", None)
        if callable(estimate_raw):
            return estimate_raw(object_images)
        return estimator.estimate_orientations_batch(object_images)
    # Cached/serialized path: provider already produced OrientationEstimates
    # (e.g. async pipeline using a static wrapper around Ray-Serve OriAny).
    # Return them directly, not their `.raw`: Ray's `dataclasses.asdict`
    # flattens `.raw` to a plain dict, where `getattr(dict, key, 0.0)` returns
    # the default and every orientation would collapse to identity.
    return list(orientation_provider.predict(image, bboxes, masks=masks))


def _orientation_from_raw(raw, fallback_orientation=None):
    if raw is None:
        return fallback_orientation
    # Cached/serialized providers may hand us a fully-built OrientationEstimate
    # (rotation_matrix, confidence, etc. already populated). Pass it through
    # rather than rebuilding from `raw` attributes — if `raw` was serialized
    # to a dict, attribute access silently returns defaults.
    if isinstance(raw, OrientationEstimate):
        return raw
    backend_euler = np.array(
        [
            float(getattr(raw, "azimuth", 0.0)),
            float(getattr(raw, "polar", getattr(raw, "elevation", 0.0))),
            float(getattr(raw, "rot", getattr(raw, "rotation", getattr(raw, "roll", 0.0)))),
        ],
        dtype=float,
    )
    euler = backend_euler_to_user(backend_euler)
    rotation_matrix = rotation_matrix_from_user_euler(euler)
    front_3d = front_direction_3d_from_user_azimuth(euler[0])
    front_2d = front_direction_2d_from_user_azimuth(euler[0])
    return OrientationEstimate(
        euler_deg=euler,
        rotation_matrix=rotation_matrix,
        quaternion=R.from_matrix(rotation_matrix).as_quat(),
        front_direction_3d=front_3d,
        front_direction_2d=front_2d,
        symmetry_alpha=int(getattr(raw, "dir_num", getattr(raw, "alpha", 1))),
        confidence=float(getattr(raw, "confidence", 0.0)),
        source="standalone_extraction",
        raw=raw,
    )


def _confidence_from_mask(confidence_map, mask):
    if confidence_map is None or mask is None:
        return None
    mask_arr = normalize_mask(mask)
    if mask_arr is None:
        return None
    mask_arr = resize_mask_to_shape(mask_arr, confidence_map.shape)
    mask_arr = erode_mask(mask_arr)
    if not np.any(mask_arr):
        return None
    return float(np.mean(confidence_map[mask_arr]))


def _crop_reinference_depth(depth_provider, image, bbox, current_depth, current_confidence):
    if depth_provider is None or current_confidence is None or current_confidence >= 0.4 or current_depth <= 0:
        return current_depth, current_confidence
    try:
        x1, y1, x2, y2 = map(int, bbox)
        width, height = image.size
        pad_x = int(max((x2 - x1) * 0.20, 4))
        pad_y = int(max((y2 - y1) * 0.20, 4))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(width, x2 + pad_x)
        cy2 = min(height, y2 + pad_y)
        if (cx2 - cx1) < 16 or (cy2 - cy1) < 16:
            return current_depth, current_confidence
        crop_image = image.crop((cx1, cy1, cx2, cy2))
        crop_depth = depth_provider.predict(crop_image)
        crop_prediction = _prediction_from_depth_estimate(crop_depth)
        conf_map = getattr(crop_prediction, "confidence", None)
        if conf_map is None:
            return current_depth, current_confidence
        cdh, cdw = crop_prediction.depth.shape[:2]
        ph = max(1, int(cdh * 0.05))
        pw = max(1, int(cdw * 0.05))
        mcy, mcx = cdh // 2, cdw // 2
        patch_d = crop_prediction.depth[max(0, mcy - ph): min(cdh, mcy + ph), max(0, mcx - pw): min(cdw, mcx + pw)]
        patch_c = conf_map[max(0, mcy - ph): min(cdh, mcy + ph), max(0, mcx - pw): min(cdw, mcx + pw)]
        valid = patch_d > 0
        if int(valid.sum()) < 5:
            return current_depth, current_confidence
        crop_depth_val = float(np.median(patch_d[valid]))
        crop_conf = float(np.mean(patch_c[valid]))
        alpha = crop_conf / (current_confidence + crop_conf + 1e-6)
        blended_depth = float((1.0 - alpha) * current_depth + alpha * crop_depth_val)
        return blended_depth, float(max(current_confidence, crop_conf))
    except Exception:
        return current_depth, current_confidence


def extract_objects_with_providers(
    image,
    detections,
    *,
    depth_provider=None,
    orientation_provider=None,
    config=None,
):
    depth_estimate = depth_provider.predict(image) if depth_provider is not None else None
    camera = camera_from_depth_estimate(image.size, depth_estimate)
    if camera.metadata is not None and depth_estimate is not None:
        camera.metadata["depth_map"] = depth_estimate.depth_map

    raw_orientations = _orientation_raw_results(orientation_provider, image, detections)

    if depth_estimate is None:
        return _objects_without_depth(detections, raw_orientations), camera, depth_estimate

    prediction = _prediction_from_depth_estimate(depth_estimate)
    depth_map = np.asarray(prediction.depth)
    intrinsics = getattr(prediction, "intrinsics", None)
    if intrinsics is None:
        intrinsics = camera.intrinsics

    objects = []
    for idx, det in enumerate(detections):
        raw_orientation = raw_orientations[idx] if idx < len(raw_orientations) else None
        objects.append(
            _object_from_depth(
                idx,
                det,
                raw_orientation,
                image=image,
                depth_map=depth_map,
                intrinsics=intrinsics,
                depth_estimate=depth_estimate,
                depth_provider=depth_provider,
                config=config,
            )
        )
    return objects, camera, depth_estimate


def _objects_without_depth(detections, raw_orientations):
    """One object per detection at the camera origin, with orientation only."""
    objects = []
    for idx, det in enumerate(detections):
        orientation = _orientation_from_raw(raw_orientations[idx]) if idx < len(raw_orientations) else None
        pose = PoseEstimate3D(
            position_xyz=np.zeros(3, dtype=float),
            orientation=orientation,
            frame="camera",
            confidence=None if orientation is None else orientation.confidence,
        )
        objects.append(
            SceneObject3D(
                id=idx,
                detection=det,
                pose=pose,
                depth_value=None,
                depth_confidence=None,
                size=SizeEstimate3D(dimensions_whl=None, oriented_bbox=None, confidence=None, source="unknown"),
            )
        )
    return objects


def _object_from_depth(
    idx, det, raw_orientation, *, image, depth_map, intrinsics, depth_estimate, depth_provider, config
):
    """Back-project one detection and fit its orientation, position and box."""
    orientation = _orientation_from_raw(raw_orientation)
    points, _ = point_cloud_from_depth_and_mask(
        depth_map=depth_map,
        intrinsics=intrinsics,
        mask=det.mask,
        bbox=det.bbox_xyxy,
    )
    raw_points = points
    geom_points = denoise_point_cloud(points)

    conf, rotation_matrix = _model_rotation(raw_orientation)
    orientation, refined_rotation = _refine_orientation(geom_points, rotation_matrix, orientation, conf, config)

    depth_value, position = robust_depth_and_center(raw_points)
    box = _fit_box(geom_points, refined_rotation, position, depth_value)

    depth_conf = _confidence_from_mask(depth_estimate.confidence_map, det.mask)
    if depth_conf is None:
        depth_conf = 1.0 if depth_estimate.confidence_map is None else 0.0
    depth_value, depth_conf = _crop_reinference_depth(
        depth_provider,
        image,
        det.bbox_xyxy,
        depth_value,
        depth_conf,
    )
    position = np.asarray(box.position, dtype=float)
    position[2] = depth_value

    return _scene_object(
        idx, det, position, orientation, depth_value, depth_conf, refined_rotation, box, raw_points, config
    )


def _model_rotation(raw_orientation):
    """``(confidence, rotation matrix)`` of the orientation model's raw output."""
    if raw_orientation is None:
        conf = 0.0
        rotation_matrix = rotation_matrix_from_user_euler(np.zeros(3, dtype=float))
    elif isinstance(raw_orientation, OrientationEstimate):
        # Already-built estimate (cached/static provider path). Use its
        # rotation_matrix directly — round-tripping through (az, el, rot)
        # would require user→backend euler inversion and lose precision.
        conf = float(raw_orientation.confidence or 0.0)
        rotation_matrix = (
            np.asarray(raw_orientation.rotation_matrix, dtype=float)
            if raw_orientation.rotation_matrix is not None
            else rotation_matrix_from_user_euler(np.zeros(3, dtype=float))
        )
    else:
        az = float(getattr(raw_orientation, "azimuth", 0.0))
        el = float(getattr(raw_orientation, "polar", getattr(raw_orientation, "elevation", 0.0)))
        rot = float(getattr(raw_orientation, "rot", getattr(raw_orientation, "rotation", getattr(raw_orientation, "roll", 0.0))))
        conf = float(getattr(raw_orientation, "confidence", 0.0))
        # Build the rotation matrix using the SAME convention as the rest
        # of the pipeline (_orientation_from_raw, OrientAnythingProvider
        # ._convert, load.py's `-col2 = front` interpretation). The raw
        # backend azimuth must be mapped through backend_euler_to_user
        # (az -> (180-az) % 360) before constructing the yxz rotation
        # matrix; otherwise col2 points INTO the camera instead of away
        # from it (Y,Z sign flip).
        user_euler = backend_euler_to_user(np.array([az, el, rot], dtype=float))
        rotation_matrix = rotation_matrix_from_user_euler(user_euler)
    return conf, rotation_matrix


def _refine_orientation(geom_points, rotation_matrix, orientation, conf, config):
    """``(orientation, rotation)`` refined against the object's points.

    Needs at least 10 points; with fewer, both are returned unchanged.
    """
    if len(geom_points) < 10:
        return orientation, rotation_matrix
    refined_rotation = refine_orientation_with_depth(
        geom_points,
        rotation_matrix,
        trust_model_elevation=getattr(config, "trust_model_elevation", 0.85),
        trust_pca_horizontal=getattr(config, "trust_pca_horizontal", 0.0),
    )
    refined_euler = user_euler_from_rotation_matrix(refined_rotation)
    refined = OrientationEstimate(
        euler_deg=np.asarray(refined_euler, dtype=float),
        rotation_matrix=refined_rotation,
        quaternion=R.from_matrix(refined_rotation).as_quat(),
        front_direction_3d=front_direction_3d_from_user_azimuth(refined_euler[0]),
        front_direction_2d=front_direction_2d_from_user_azimuth(refined_euler[0]),
        symmetry_alpha=None if orientation is None else orientation.symmetry_alpha,
        confidence=conf,
        source=None if orientation is None else orientation.source,
        raw=None if orientation is None else orientation.raw,
    )
    return refined, refined_rotation


def _fit_box(geom_points, refined_rotation, position, depth_value):
    """Oriented box fitted to the points (needs at least 10 points).

    Returns the object position (the completed box centre at depth
    *depth_value*, else *position*), the completed size, and the visible
    box's extent, centre and corners.
    """
    box = SimpleNamespace(
        position=position,
        size=None,
        visible_extent=None,
        center=np.array(position, dtype=float, copy=True),
        corners=None,
        completed_center=np.array(position, dtype=float, copy=True),
    )
    if len(geom_points) >= 10:
        visible_extent, visible_center, visible_corners = fit_visible_bbox(geom_points, refined_rotation)
        extent, refined_center = estimate_extent_from_depth(geom_points, refined_rotation)
        box.size = extent
        box.visible_extent = visible_extent
        box.center = np.array(visible_center, dtype=float, copy=True)
        box.corners = None if visible_corners is None else np.array(visible_corners, dtype=float, copy=True)
        box.completed_center = np.array(refined_center, dtype=float, copy=True)
        box.position = refined_center
        box.position[2] = depth_value
    return box


def _scene_object(idx, det, position, orientation, depth_value, depth_conf, refined_rotation, box, raw_points, config):
    pose = PoseEstimate3D(
        position_xyz=position,
        orientation=orientation,
        frame="camera",
        confidence=min(
            [v for v in [depth_conf, None if orientation is None else orientation.confidence] if v is not None],
            default=depth_conf,
        ),
    )
    size = SizeEstimate3D(
        dimensions_whl=None if box.size is None else np.asarray(box.size, dtype=float),
        oriented_bbox=box.corners,
        confidence=depth_conf,
        source="standalone_extraction",
        raw={
            "rotation_matrix": refined_rotation,
            "bbox_center_xyz": box.center,
            "visible_dimensions_whl": box.visible_extent,
            "completed_bbox_center_xyz": box.completed_center,
        },
    )
    return SceneObject3D(
        id=idx,
        detection=det,
        pose=pose,
        depth_value=float(depth_value),
        depth_confidence=float(depth_conf),
        size=size,
        point_cloud=raw_points if getattr(config, "keep_point_clouds", False) else None,
        artifacts={"raw_points": raw_points if getattr(config, "keep_point_clouds", False) else None},
        metadata={
            "refined_rotation_matrix": refined_rotation,
            "bbox_center_xyz": box.center,
            "completed_bbox_center_xyz": box.completed_center,
        },
    )
