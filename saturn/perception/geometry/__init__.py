"""Camera / point-cloud geometry shared by the perception providers.

Pure numpy geometry; must not import saturn.vlm/serving.
"""
from .camera import camera_from_depth_estimate, intrinsics_from_image_size, pixel_to_3d
from .filtering import denoise_point_cloud, normalize_mask, resize_mask_to_shape, robust_depth_and_center
from .pose_fusion import fuse_object_geometry

__all__ = [
    "camera_from_depth_estimate",
    "intrinsics_from_image_size",
    "pixel_to_3d",
    "normalize_mask",
    "resize_mask_to_shape",
    "denoise_point_cloud",
    "robust_depth_and_center",
    "fuse_object_geometry",
]
