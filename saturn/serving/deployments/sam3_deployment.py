"""Ray Serve deployment wrapping ``object_detector("sam3")``.

The deployment class + a ``build()`` factory that the launcher
(``saturn.serving.deploy``) binds. Heavy imports (the detector, torch) happen
only inside ``__init__`` to keep the module import-safe when Ray is absent.
"""

from __future__ import annotations

from saturn.settings import env
from typing import Any, Dict, List


from saturn.perception.codec import decode_image

try:
    from ray import serve  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover - Ray optional at import time
    serve = None  # type: ignore
    _HAS_RAY = False


NAME = "sam3"


class _SAM3DeploymentImpl:
    """Implementation body, agnostic of Ray Serve decoration."""

    def __init__(self) -> None:
        # Heavy imports kept inside __init__ so this module can be imported
        # in environments without torch / transformers / sam3.
        from saturn.perception.detection.sam3 import object_detector

        self._detector = object_detector("sam3")

    async def predict_with_masks(
        self,
        image_bytes: bytes,
        text: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        image = decode_image(image_bytes)
        out = self._detector.predict_with_masks(image, text=text, threshold=threshold)
        return _serialize_detection(out)

    async def mask_from_boxes(
        self,
        image_bytes: bytes,
        boxes: List[List[float]],
    ) -> Dict[str, Any]:
        image = decode_image(image_bytes)
        masks = self._detector.mask_from_boxes(image, boxes)
        return {"masks": _serialize_masks(masks)}


def _serialize_masks(masks: Any) -> List[bytes]:
    import numpy as np
    import torch

    out: List[bytes] = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            arr = m.detach().cpu().numpy().astype(np.uint8)
        else:
            arr = np.asarray(m).astype(np.uint8)
        out.append(arr.tobytes() + b"|shape=" + str(arr.shape).encode())
    return out


def _serialize_detection(out: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import torch

    def _conv(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    return {
        "boxes": _conv(out.get("boxes", [])),
        "scores": _conv(out.get("scores", [])),
        "masks": _serialize_masks(out.get("masks", [])),
    }


if _HAS_RAY:

    _SAM3_NUM_GPUS = float(env("SAPY_SAM3_NUM_GPUS"))
    # Autoscaled replica count. At num_gpus=0.5 two replicas share one GPU,
    # so the default 2-4 replicas occupy 1-2 GPUs.
    _SAM3_MIN_REPLICAS = int(env("SAPY_SAM3_MIN_REPLICAS"))
    _SAM3_MAX_REPLICAS = int(env("SAPY_SAM3_MAX_REPLICAS"))
    _SAM3_TARGET_INFLIGHT = int(env("SAPY_SAM3_TARGET_INFLIGHT"))
    _SAM3_MAX_ONGOING = int(env("SAPY_SAM3_MAX_ONGOING"))

    @serve.deployment(
        name=NAME,
        ray_actor_options={"num_gpus": _SAM3_NUM_GPUS},
        max_ongoing_requests=_SAM3_MAX_ONGOING,
        autoscaling_config={
            "min_replicas": _SAM3_MIN_REPLICAS,
            "max_replicas": _SAM3_MAX_REPLICAS,
            "target_ongoing_requests": _SAM3_TARGET_INFLIGHT,
            "upscale_delay_s": 5.0,
            "downscale_delay_s": 60.0,
        },
    )
    class SAM3Deployment(_SAM3DeploymentImpl):
        pass

    def build() -> Any:
        return SAM3Deployment.bind()

else:  # pragma: no cover

    SAM3Deployment = _SAM3DeploymentImpl  # type: ignore

    def build() -> Any:
        raise RuntimeError(
            "Ray is not installed; cannot build the SAM3 Serve deployment."
        )
