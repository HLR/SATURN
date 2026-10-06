"""Ray Serve deployment wrapping ``VGGTReconstructor``."""

from __future__ import annotations

from saturn.settings import env
import io
from typing import Any, Dict, List

from PIL import Image

try:
    from ray import serve  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover
    serve = None  # type: ignore
    _HAS_RAY = False


NAME = "vggt"


def _decode_images(images_bytes: List[bytes]) -> List[Image.Image]:
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]


class _VGGTDeploymentImpl:
    def __init__(self) -> None:
        from saturn.perception.reconstruction.vggt import VGGTReconstructor

        # VGGTReconstructor loads facebook/VGGT-1B and takes ``device`` as its
        # first positional arg, so pass keyword args here.
        self._reconstructor = VGGTReconstructor(device="cuda", dtype="bfloat16")

    async def reconstruct(self, images_bytes: List[bytes]) -> Dict[str, Any]:
        images = _decode_images(images_bytes)
        result = self._reconstructor.reconstruct(images)
        return _serialize_reconstruction(result)


def _serialize_reconstruction(result: Any) -> Dict[str, Any]:
    """Convert the reconstruction output to picklable primitives.

    VGGTReconstructor returns a provider-specific structure; torch tensors
    become numpy arrays. The async provider wrapper rehydrates.
    """
    import numpy as np
    import torch

    def _conv(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, dict):
            return {k: _conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_conv(v) for v in x]
        return x

    if hasattr(result, "__dict__"):
        out = {k: _conv(v) for k, v in vars(result).items()}
        out["_type"] = type(result).__name__
        return out
    return {"_raw": _conv(result)}


if _HAS_RAY:

    _VGGT_NUM_GPUS = float(env("SAPY_VGGT_NUM_GPUS"))

    @serve.deployment(
        name=NAME,
        ray_actor_options={"num_gpus": _VGGT_NUM_GPUS},
        max_ongoing_requests=1,
    )
    class VGGTDeployment(_VGGTDeploymentImpl):
        pass

    def build() -> Any:
        return VGGTDeployment.bind()

else:  # pragma: no cover

    VGGTDeployment = _VGGTDeploymentImpl  # type: ignore

    def build() -> Any:
        raise RuntimeError("Ray is not installed.")
