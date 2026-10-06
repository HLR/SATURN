"""OrientAnythingProvider: in-process Orient-Anything wrapper behind the orientation-provider API.

Model-side; imported only by pipeline.models and serving.
"""
from typing import Optional

import numpy as np

from scipy.spatial.transform import Rotation as R

from saturn.perception.types import OrientationEstimate
import saturn.perception.orientation.orient_anything as orient_anything
from saturn.perception.orientation.convention import (
    backend_euler_to_user,
    front_direction_2d_from_user_azimuth,
    front_direction_3d_from_user_azimuth,
    rotation_matrix_from_user_euler,
)


class OrientAnythingProvider:
    """Orient-Anything V2 with test-time augmentation (``TTA_CROPS`` random crops
    plus the original per object)."""

    def __init__(
        self,
        device: str = None,
        tta_num_crops: Optional[int] = None,   # None -> orient_anything.TTA_CROPS (3)
        tta_crop_scale=(0.8, 0.95),
        tta_include_original: bool = True,
        tta_outlier_threshold: float = 1.5,
        **kwargs,
    ):
        OrientAnythingEstimator = orient_anything.OrientAnythingEstimator
        ObjectPerspectiveSpatialRelations = orient_anything.ObjectPerspectiveSpatialRelations
        self.tta_num_crops = orient_anything._tta_crops(tta_num_crops)
        self.tta_crop_scale = tta_crop_scale
        self.tta_include_original = tta_include_original
        self.tta_outlier_threshold = tta_outlier_threshold
        self.estimator = OrientAnythingEstimator(device=device, **kwargs)
        self.extractor = ObjectPerspectiveSpatialRelations(
            orientation_estimator=self.estimator,
            device=device,
        )

    def _estimate_raw(self, object_images):
        return self.estimator.estimate_orientations_batch_with_tta(
            object_images,
            num_crops=self.tta_num_crops,
            crop_scale=self.tta_crop_scale,
            include_original=self.tta_include_original,
            outlier_threshold=self.tta_outlier_threshold,
        )

    def predict(self, image, bboxes, masks=None):
        object_images = self.extractor.extract_object_images(image, np.asarray(bboxes), masks=masks)
        if not object_images:
            return []
        results = self._estimate_raw(object_images)
        return [self._convert(result) for result in results]

    def _convert(self, result):
        backend_euler = np.array(
            [
                float(getattr(result, "azimuth", 0.0)),
                float(getattr(result, "polar", getattr(result, "elevation", 0.0))),
                float(getattr(result, "rotation", getattr(result, "roll", 0.0))),
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
            front_direction_2d=np.asarray(front_2d, dtype=float),
            symmetry_alpha=int(getattr(result, "dir_num", getattr(result, "alpha", 1))),
            confidence=float(getattr(result, "confidence", 0.0)),
            source="orientanything_v2",
            raw=result,
        )
