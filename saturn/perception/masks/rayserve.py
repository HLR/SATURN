"""Async Ray Serve adapter implementing ``MaskProviderAsync`` + ``DetectionProviderAsync``."""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
from PIL import Image

from saturn.perception.codec import encode_image


def _decode_masks(serialized: List[bytes]) -> List[np.ndarray]:
    """Inverse of ``serving.deployments.sam3_deployment._serialize_masks``."""
    out: List[np.ndarray] = []
    for item in serialized:
        payload, _, tail = item.rpartition(b"|shape=")
        shape = eval(tail.decode())  # noqa: S307 — produced by our own serializer
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
        out.append(arr)
    return out


class RayServeSAM3MaskProvider:
    """Async provider that delegates mask/detection calls to Ray Serve.

    Both ``predict_with_masks`` (text-grounded detection) and
    ``mask_from_boxes`` (bbox→mask) go through the SAM3 handle.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    async def predict(self, image: Image.Image) -> Sequence:
        """Detection entrypoint — matches the sync ``DetectionProvider``."""
        img_bytes = encode_image(image)
        result = await self._handle.predict_with_masks.remote(img_bytes, "", 0.0)
        return result

    async def predict_with_masks(
        self, image: Image.Image, text: str, threshold: float = 0.0
    ) -> dict:
        img_bytes = encode_image(image)
        out = await self._handle.predict_with_masks.remote(img_bytes, text, threshold)
        out["masks"] = _decode_masks(out.get("masks", []))
        return out

    async def predict_masks(
        self, image: Image.Image, bboxes: Sequence
    ) -> Sequence:
        img_bytes = encode_image(image)
        out = await self._handle.mask_from_boxes.remote(img_bytes, list(bboxes))
        return _decode_masks(out.get("masks", []))

    # Sync shim used by the cache-bridging path in generated programs.
    def mask_from_boxes(self, image: Image.Image, bboxes: Sequence):
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.predict_masks(image, bboxes)
        )
