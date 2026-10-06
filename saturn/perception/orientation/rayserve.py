"""Async Ray Serve adapter implementing ``OrientationProviderAsync``."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image

from saturn.perception.codec import encode_image

from saturn.perception.types import OrientationEstimate


class RayServeOriAnyOrientationProvider:
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        # Callers may read ``orientation_provider.estimator``; behind a Ray
        # handle there is no in-process estimator.
        self.estimator = None

    async def predict(
        self,
        image: Image.Image,
        bboxes: Sequence,
        masks: Sequence | None = None,
    ):
        img_bytes = encode_image(image)
        # Masks are not forwarded (payload size), so the deployment crops the
        # raw bbox. The in-process OrientAnythingProvider.predict(masks=...)
        # masks out the background, so the two paths can disagree on
        # cluttered crops.
        out = await self._handle.predict.remote(img_bytes, list(bboxes), None)
        estimates = []
        for item in out.get("estimates", []):
            if isinstance(item, OrientationEstimate):
                estimates.append(item)
                continue
            estimates.append(
                OrientationEstimate(
                    euler_deg=None
                    if item.get("euler_deg") is None
                    else np.asarray(item["euler_deg"], dtype=float),
                    rotation_matrix=None
                    if item.get("rotation_matrix") is None
                    else np.asarray(item["rotation_matrix"], dtype=float),
                    quaternion=None
                    if item.get("quaternion") is None
                    else np.asarray(item["quaternion"], dtype=float),
                    front_direction_3d=None
                    if item.get("front_direction_3d") is None
                    else np.asarray(item["front_direction_3d"], dtype=float),
                    front_direction_2d=None
                    if item.get("front_direction_2d") is None
                    else np.asarray(item["front_direction_2d"], dtype=float),
                    symmetry_alpha=item.get("symmetry_alpha"),
                    confidence=item.get("confidence"),
                    source=item.get("source"),
                    raw=item.get("raw"),
                )
            )
        return estimates
