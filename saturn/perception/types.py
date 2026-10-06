"""Dataclasses exchanged between perception providers (detections, depth, orientation, cameras).

Pure data types; must not import saturn.vlm/serving.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection2D:
    bbox_xyxy: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    label: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None


@dataclass
class OrientationEstimate:
    euler_deg: Optional[np.ndarray]
    rotation_matrix: Optional[np.ndarray]
    quaternion: Optional[np.ndarray]
    front_direction_3d: Optional[np.ndarray]
    front_direction_2d: Optional[np.ndarray]
    symmetry_alpha: Optional[int]
    confidence: Optional[float]
    source: Optional[str] = None
    raw: Any = None


@dataclass
class DepthEstimate:
    depth_map: np.ndarray
    confidence_map: Optional[np.ndarray]
    is_metric: bool
    source: Optional[str] = None
    raw: Any = None


@dataclass
class CameraModel:
    intrinsics: Optional[np.ndarray]
    extrinsics: Optional[np.ndarray]
    image_size: Tuple[int, int]
    frame_name: str = "camera"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SizeEstimate3D:
    dimensions_whl: Optional[np.ndarray]
    oriented_bbox: Optional[np.ndarray]
    confidence: Optional[float]
    source: Optional[str] = None
    raw: Any = None


@dataclass
class PoseEstimate3D:
    position_xyz: Optional[np.ndarray]
    orientation: Optional[OrientationEstimate]
    frame: str = "camera"
    confidence: Optional[float] = None


@dataclass
class SceneObject3D:
    id: int
    detection: Detection2D
    pose: PoseEstimate3D
    depth_value: Optional[float]
    depth_confidence: Optional[float]
    size: SizeEstimate3D
    point_cloud: Optional[np.ndarray] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialRelationRecord:
    subject_id: int
    object_id: int
    relation_type: str
    frame: str
    confidence: float
    magnitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialScene:
    objects: List[SceneObject3D]
    camera: CameraModel
    depth: Optional[DepthEstimate] = None
    relations: List[SpatialRelationRecord] = field(default_factory=list)
    graph: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
