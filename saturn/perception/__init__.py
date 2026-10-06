"""Perception backbones (detection, reconstruction, orientation, depth) and their shared types.

Model-side; imported only by saturn.pipeline.models, saturn.scene.build and saturn.serving.
"""
from .types import (
    CameraModel,
    DepthEstimate,
    Detection2D,
    OrientationEstimate,
    PoseEstimate3D,
    SceneObject3D,
    SizeEstimate3D,
    SpatialRelationRecord,
    SpatialScene,
)

__all__ = [
    "Detection2D",
    "OrientationEstimate",
    "DepthEstimate",
    "CameraModel",
    "PoseEstimate3D",
    "SizeEstimate3D",
    "SceneObject3D",
    "SpatialRelationRecord",
    "SpatialScene",
]
