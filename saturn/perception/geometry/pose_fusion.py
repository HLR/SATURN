"""Fusion of per-view object poses (orientation + position) into a single estimate.

Pure numpy geometry; must not import saturn.vlm/serving.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from saturn.perception.orientation.convention import (
    front_direction_2d_from_user_azimuth,
    front_direction_3d_from_user_azimuth,
    rotation_matrix_from_user_euler,
    user_euler_from_rotation_matrix,
)
from ..types import OrientationEstimate, PoseEstimate3D, SceneObject3D, SizeEstimate3D
from .filtering import denoise_point_cloud, robust_depth_and_center
from .pointcloud import point_cloud_from_depth_and_mask


def oriented_bbox_corners(center: np.ndarray, extent: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    extent = np.asarray(extent, dtype=float)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    sw, sh, sl = extent / 2.0
    corners_local = np.array(
        [
            [-sw, -sh, -sl],
            [sw, -sh, -sl],
            [sw, sh, -sl],
            [-sw, sh, -sl],
            [-sw, -sh, sl],
            [sw, -sh, sl],
            [sw, sh, sl],
            [-sw, sh, sl],
        ],
        dtype=float,
    )
    return corners_local @ rotation_matrix.T + center


def fit_visible_bbox(points: np.ndarray, rotation_matrix: np.ndarray):
    points = denoise_point_cloud(points)
    if len(points) == 0:
        center = np.zeros(3, dtype=float)
        extent = np.zeros(3, dtype=float)
        return extent, center, None
    center_mean = np.mean(points, axis=0)
    pts_local = (points - center_mean) @ rotation_matrix
    min_b = np.percentile(pts_local, 2, axis=0)
    max_b = np.percentile(pts_local, 98, axis=0)
    visible_extent = max_b - min_b
    local_center = (min_b + max_b) / 2.0
    visible_center = center_mean + local_center @ rotation_matrix.T
    corners = oriented_bbox_corners(visible_center, visible_extent, rotation_matrix)
    return visible_extent.astype(float), np.asarray(visible_center, dtype=float), corners


def orientation_to_rotation_matrix(orientation: OrientationEstimate) -> np.ndarray:
    if orientation is None:
        return np.eye(3)
    if orientation.rotation_matrix is not None:
        return np.asarray(orientation.rotation_matrix, dtype=float)
    if orientation.euler_deg is None:
        return np.eye(3)
    return rotation_matrix_from_user_euler(orientation.euler_deg)


def rotation_matrix_to_orientation(rotation_matrix, template: OrientationEstimate = None) -> OrientationEstimate:
    euler = user_euler_from_rotation_matrix(rotation_matrix)
    az = float(euler[0])
    front_3d = front_direction_3d_from_user_azimuth(az)
    front_2d = front_direction_2d_from_user_azimuth(az)
    return OrientationEstimate(
        euler_deg=np.asarray(euler, dtype=float),
        rotation_matrix=np.asarray(rotation_matrix, dtype=float),
        quaternion=R.from_matrix(rotation_matrix).as_quat(),
        front_direction_3d=front_3d,
        front_direction_2d=front_2d,
        symmetry_alpha=None if template is None else template.symmetry_alpha,
        confidence=None if template is None else template.confidence,
        source=None if template is None else template.source,
        raw=None if template is None else template.raw,
    )


def refine_orientation_with_depth(points: np.ndarray, rotation_matrix: np.ndarray, trust_model_elevation=0.85, trust_pca_horizontal=0.5):
    points = denoise_point_cloud(points)
    if len(points) < 10:
        return rotation_matrix
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    model_up = rotation_matrix[:, 1]
    # PCA axis closest to the model's up, sign-aligned before blending
    # (eigenvector signs are arbitrary). The smallest-variance axis is the
    # view-facing normal of a depth-visible surface, not up.
    pca_up = eigvecs[:, int(np.argmax(np.abs(eigvecs.T @ model_up)))]
    pca_up = pca_up * np.sign(np.dot(pca_up, model_up) or 1.0)
    blended_up = model_up * trust_model_elevation + pca_up * (1 - trust_model_elevation)
    blended_up = blended_up / (np.linalg.norm(blended_up) + 1e-8)
    if np.dot(blended_up, model_up) < 0:
        blended_up = -blended_up
    model_forward = rotation_matrix[:, 2]
    pca_forward = eigvecs[:, 0]
    blended_forward = model_forward * (1 - trust_pca_horizontal) + pca_forward * trust_pca_horizontal
    blended_forward = blended_forward / (np.linalg.norm(blended_forward) + 1e-8)
    if np.dot(blended_forward, model_forward) < 0:
        blended_forward = -blended_forward
    blended_right = np.cross(blended_up, blended_forward)
    blended_right = blended_right / (np.linalg.norm(blended_right) + 1e-8)
    blended_forward = np.cross(blended_right, blended_up)
    blended_forward = blended_forward / (np.linalg.norm(blended_forward) + 1e-8)
    return np.column_stack([blended_right, blended_up, blended_forward])


def estimate_extent_from_depth(points: np.ndarray, rotation_matrix: np.ndarray, depth_buffer_factor: float = 1.2):
    points = denoise_point_cloud(points)
    if len(points) < 10:
        center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        return np.array([1.0, 1.0, 1.0]), center
    visible_extent, visible_center, _ = fit_visible_bbox(points, rotation_matrix)
    max_visible_dim = max(visible_extent[0], visible_extent[1])
    if visible_extent[2] < 0.3 * max_visible_dim:
        extent_z = max_visible_dim * 0.6
    else:
        extent_z = visible_extent[2] * depth_buffer_factor
    extent = np.array([visible_extent[0], visible_extent[1], extent_z], dtype=float)
    world_center = np.asarray(visible_center, dtype=float).copy()
    # Nudge center along R[:,2] by 0.1 * extent_z to account for the
    # unobserved back face.
    world_center = world_center + rotation_matrix[:, 2] * (extent_z * 0.1)
    return extent, world_center


def fuse_object_geometry(
    object_id: int,
    detection,
    camera,
    depth_estimate,
    orientation: OrientationEstimate = None,
    keep_point_clouds: bool = False,
):
    points, _ = point_cloud_from_depth_and_mask(
        depth_map=depth_estimate.depth_map,
        intrinsics=camera.intrinsics,
        mask=detection.mask,
        bbox=detection.bbox_xyxy,
    )
    raw_points = points
    geom_points = denoise_point_cloud(points)
    depth_value, position = robust_depth_and_center(raw_points)
    rotation_matrix = orientation_to_rotation_matrix(orientation)
    refined_rotation = refine_orientation_with_depth(geom_points, rotation_matrix)
    if orientation is not None:
        orientation = rotation_matrix_to_orientation(refined_rotation, template=orientation)
    visible_extent, visible_center, visible_corners = fit_visible_bbox(geom_points, refined_rotation)
    extent, refined_center = estimate_extent_from_depth(geom_points, refined_rotation)
    bbox_center = np.array(visible_center, dtype=float, copy=True)
    bbox_corners = None if visible_corners is None else np.array(visible_corners, dtype=float, copy=True)
    if len(geom_points) >= 10:
        position = refined_center
        position[2] = depth_value
    depth_conf = None
    if depth_estimate.confidence_map is not None:
        from .filtering import normalize_mask, resize_mask_to_shape

        mask = normalize_mask(detection.mask)
        if mask is not None:
            mask = resize_mask_to_shape(mask, depth_estimate.confidence_map.shape)
            if np.any(mask):
                depth_conf = float(np.mean(depth_estimate.confidence_map[mask]))
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
        dimensions_whl=extent,
        oriented_bbox=bbox_corners,
        confidence=depth_conf,
        source="depth_estimate",
        raw={
            "rotation_matrix": refined_rotation,
            "bbox_center_xyz": bbox_center,
            "visible_dimensions_whl": visible_extent,
            "completed_bbox_center_xyz": np.array(refined_center, dtype=float, copy=True),
            "completed_oriented_bbox": oriented_bbox_corners(np.array(refined_center, dtype=float, copy=True), extent, refined_rotation),
        },
    )
    return SceneObject3D(
        id=object_id,
        detection=detection,
        pose=pose,
        depth_value=depth_value,
        depth_confidence=depth_conf,
        size=size,
        point_cloud=raw_points if keep_point_clouds else None,
        artifacts={"raw_points": raw_points if keep_point_clouds else None},
        metadata={
            "refined_rotation_matrix": refined_rotation,
            "bbox_center_xyz": bbox_center,
            "completed_bbox_center_xyz": np.array(refined_center, dtype=float, copy=True),
        },
    )
