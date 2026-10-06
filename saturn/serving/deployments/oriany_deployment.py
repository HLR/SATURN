"""Ray Serve deployment wrapping ``OrientAnythingProvider`` (Orient-Anything V2, with TTA)."""

from __future__ import annotations

from saturn.settings import env
import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from saturn.perception.codec import decode_image

try:
    from ray import serve  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover
    serve = None  # type: ignore
    _HAS_RAY = False


NAME = "oriany"


class _OriAnyDeploymentImpl:
    def __init__(self) -> None:
        from saturn.perception.orientation.orientanything_serve import (
            OrientAnythingProvider,
        )

        self._provider = OrientAnythingProvider()

    def _crop_for_request(
        self,
        image_bytes: bytes,
        bboxes: List[List[float]],
        masks: Optional[List[bytes]],
    ) -> List[Image.Image]:
        if not bboxes:
            return []
        image = decode_image(image_bytes)
        return list(
            self._provider.extractor.extract_object_images(
                image, np.asarray(bboxes), masks=masks
            )
        )

    def _build_response(self, raw_results: List[Any]) -> Dict[str, Any]:
        estimates = [self._provider._convert(r) for r in raw_results]
        return {"estimates": [_serialize_orientation(e) for e in estimates]}

    async def predict(
        self,
        image_bytes: bytes,
        bboxes: List[List[float]],
        masks: Optional[List[bytes]] = None,
    ) -> Dict[str, Any]:
        # No-Ray fallback path. The deployment subclass overrides this with
        # the @serve.batch'd version that coalesces concurrent calls into
        # one GPU forward pass.
        crops = self._crop_for_request(image_bytes, bboxes, masks)
        if not crops:
            return {"estimates": []}
        raw = self._provider._estimate_raw(crops)
        return self._build_response(raw)


def _serialize_orientation(est: Any) -> Dict[str, Any]:
    import dataclasses
    import numpy as np
    import torch

    def _conv(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return {k: _conv(v) for k, v in dataclasses.asdict(x).items()}
        return x

    if dataclasses.is_dataclass(est) and not isinstance(est, type):
        return {k: _conv(v) for k, v in dataclasses.asdict(est).items()}
    if hasattr(est, "__dict__"):
        return {k: _conv(v) for k, v in vars(est).items()}
    return {"_raw": _conv(est)}


if _HAS_RAY:

    # A replica needs ~7 GB, so num_gpus=0.25 packs up to 4 replicas onto
    # one GPU.
    _ORIANY_NUM_GPUS = float(env("SAPY_ORIANY_NUM_GPUS"))
    _ORIANY_MIN_REPLICAS = int(env("SAPY_ORIANY_MIN_REPLICAS"))
    # max_replicas * num_gpus bounds how many GPUs oriany can scale into
    # under burst load (8 * 0.25 = 2).
    _ORIANY_MAX_REPLICAS = int(env("SAPY_ORIANY_MAX_REPLICAS"))
    _ORIANY_TARGET_INFLIGHT = int(
        env("SAPY_ORIANY_TARGET_INFLIGHT")
    )
    # Server-side coalescing window: concurrent oriany calls are bundled into
    # one forward pass to keep the GPU saturated. The wait is well below
    # per-call latency, so it adds no tail when the queue is busy.
    _ORIANY_BATCH_SIZE = int(env("SAPY_ORIANY_BATCH_SIZE"))
    _ORIANY_BATCH_WAIT_S = float(env("SAPY_ORIANY_BATCH_WAIT_S"))
    # max_ongoing_requests must be >= max_batch_size or batches won't fill.
    _ORIANY_MAX_ONGOING = max(_ORIANY_BATCH_SIZE * 2, 32)

    @serve.deployment(
        name=NAME,
        ray_actor_options={"num_gpus": _ORIANY_NUM_GPUS},
        max_ongoing_requests=_ORIANY_MAX_ONGOING,
        autoscaling_config={
            "min_replicas": _ORIANY_MIN_REPLICAS,
            "max_replicas": _ORIANY_MAX_REPLICAS,
            "target_ongoing_requests": _ORIANY_TARGET_INFLIGHT,
            "upscale_delay_s": 5.0,
            "downscale_delay_s": 60.0,
        },
    )
    class OriAnyDeployment(_OriAnyDeploymentImpl):
        @serve.batch(
            max_batch_size=_ORIANY_BATCH_SIZE,
            batch_wait_timeout_s=_ORIANY_BATCH_WAIT_S,
        )
        async def _predict_batched(
            self,
            image_bytes_list: List[bytes],
            bboxes_list: List[List[List[float]]],
            masks_list: List[Optional[List[bytes]]],
        ) -> List[Dict[str, Any]]:
            # Per-request cropping is CPU work (decode + TTA crops); run the
            # requests' crops in threads so they overlap each other and the
            # previous batch's GPU pass instead of serialising on this loop.
            per_req_crops: List[List[Image.Image]] = list(await asyncio.gather(*(
                asyncio.to_thread(self._crop_for_request, image_bytes, bboxes, masks)
                for image_bytes, bboxes, masks in zip(image_bytes_list, bboxes_list, masks_list)
            )))

            # One super-batch through the model (in a thread: the forward pass
            # must not block the event loop that is collecting the next batch),
            # then split the flat result back per request by crop count.
            flat_crops: List[Image.Image] = [c for cs in per_req_crops for c in cs]
            flat_raw: List[Any] = (
                await asyncio.to_thread(self._provider._estimate_raw, flat_crops) if flat_crops else []
            )
            if len(flat_raw) != len(flat_crops):
                # a dropped/reordered result would shift every later request by one object
                raise RuntimeError(f"orientation batch returned {len(flat_raw)} results for {len(flat_crops)} crops")

            outputs: List[Dict[str, Any]] = []
            offset = 0
            for crops in per_req_crops:
                n = len(crops)
                outputs.append(self._build_response(flat_raw[offset : offset + n]))
                offset += n
            return outputs

        async def predict(
            self,
            image_bytes: bytes,
            bboxes: List[List[float]],
            masks: Optional[List[bytes]] = None,
        ) -> Dict[str, Any]:
            return await self._predict_batched(image_bytes, bboxes, masks)

    def build() -> Any:
        # Autoscaling is configured in the decorator above: set
        # SAPY_ORIANY_MIN_REPLICAS / MAX_REPLICAS / TARGET_INFLIGHT to tune it.
        return OriAnyDeployment.bind()

else:  # pragma: no cover

    OriAnyDeployment = _OriAnyDeploymentImpl  # type: ignore

    def build() -> Any:
        raise RuntimeError("Ray is not installed.")
