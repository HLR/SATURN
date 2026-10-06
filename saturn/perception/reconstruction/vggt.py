"""
VGGT-1B reconstruction wrapper for the multi-view pipeline.

Provides a synchronous VGGTReconstructor that runs a single forward pass on
all input images and returns world-space 3D point clouds, extrinsics, and
intrinsics in the format the scene loader (``load.py``) consumes.

Imports VGGT from the vendored copy at tools/Orient-Anything-V2/vggt/, which
is added to sys.path on first use when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from saturn.log import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import (avoid import-time GPU allocation)
# ---------------------------------------------------------------------------
_torch = None
_F = None


def _ensure_torch():
    global _torch, _F
    if _torch is None:
        import torch
        import torch.nn.functional as F

        _torch = torch
        _F = F


# ---------------------------------------------------------------------------
# Preprocessing (adapted from GCA's VGGT preprocessing)
# ---------------------------------------------------------------------------

TARGET_SIZE = 518


def _preprocess_images(
    image_list: List[Image.Image],
) -> Tuple:
    """Resize, pad and batch images for VGGT inference.

    Returns
    -------
    images : torch.Tensor, shape (S, 3, H, W), dtype float32, range [0,1]
    transform_info_list : per-image metadata for inverse-mapping masks
    """
    _ensure_torch()
    from torchvision import transforms as TF

    to_tensor = TF.ToTensor()
    target_size = TARGET_SIZE

    tensors = []
    shapes: set = set()
    transform_info_list: List[Dict[str, Any]] = []

    for img in image_list:
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")

        width, height = img.size
        transform_info: Dict[str, Any] = {
            "original_shape": (width, height),
            "preprocessed_shape": None,
            "crop_box": None,
            "pad_box": None,
            "multi_image_padding": None,
        }

        # Pad mode: largest dim -> 518, smaller -> nearest multiple of 14
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14

        resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        transform_info["preprocessed_shape"] = (new_width, new_height)
        t = to_tensor(resized)  # (3, new_height, new_width), [0,1]

        # Center-pad to 518x518 with white (1.0)
        h_pad = target_size - t.shape[1]
        w_pad = target_size - t.shape[2]
        if h_pad > 0 or w_pad > 0:
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            pad_box = (pad_left, pad_right, pad_top, pad_bottom)
            t = _torch.nn.functional.pad(t, pad_box, mode="constant", value=1.0)
            transform_info["pad_box"] = pad_box

        shapes.add((t.shape[1], t.shape[2]))
        tensors.append(t)
        transform_info_list.append(transform_info)

    # If images ended up with different shapes, pad to max
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for i, t in enumerate(tensors):
            h_pad = max_h - t.shape[1]
            w_pad = max_w - t.shape[2]
            if h_pad > 0 or w_pad > 0:
                pt = h_pad // 2
                pb = h_pad - pt
                pl = w_pad // 2
                pr = w_pad - pl
                pad_info = (pl, pr, pt, pb)
                transform_info_list[i]["multi_image_padding"] = pad_info
                t = _torch.nn.functional.pad(t, pad_info, mode="constant", value=1.0)
            padded.append(t)
        tensors = padded

    images = _torch.stack(tensors)  # (S, 3, H, W)
    return images, transform_info_list


def transform_mask_to_vggt_space(
    mask: np.ndarray,
    transform_info: Dict[str, Any],
) -> np.ndarray:
    """Transform a full-resolution boolean mask to VGGT's preprocessed space.

    Mirrors GCA's _tensor_transform with nearest interpolation for masks.

    Parameters
    ----------
    mask : (H_orig, W_orig) bool array
    transform_info : dict from _preprocess_images

    Returns
    -------
    transformed : bool array matching VGGT spatial dims
    """
    _ensure_torch()

    # (H, W) -> (1, 1, H, W) float
    t = _torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Step 1: resize to preprocessed shape  (note: preprocessed_shape is (W, H))
    prep_w, prep_h = transform_info["preprocessed_shape"]
    t = _F.interpolate(t, size=(prep_h, prep_w), mode="nearest")

    # Step 2: center-pad (same as image preprocessing)
    pad_box = transform_info.get("pad_box")
    if pad_box:
        t = _torch.nn.functional.pad(t, pad_box, mode="constant", value=0)

    # Step 3: multi-image padding (if images had different sizes)
    multi_pad = transform_info.get("multi_image_padding")
    if multi_pad:
        t = _torch.nn.functional.pad(t, multi_pad, mode="constant", value=0)

    return (t.squeeze().numpy() > 0.5).astype(bool)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VGGTResult:
    """Result from a VGGT reconstruction pass."""

    world_points: np.ndarray  # (S, H, W, 3)
    world_points_conf: np.ndarray  # (S, H, W)
    depth: np.ndarray  # (S, H, W)
    depth_conf: np.ndarray  # (S, H, W)
    extrinsics: np.ndarray  # (S, 4, 4) -- world-to-camera
    intrinsics: np.ndarray  # (S, 3, 3)
    transform_info: List[Dict[str, Any]] = field(default_factory=list)
    num_views: int = 0

    def get_world_geometry(self, view_idx: int) -> Dict[str, Any]:
        """Return a per-view geometry dict compatible with load.py's pipeline.

        The dict has the keys of ``_world_geometry_from_depth()`` in load.py,
        plus ``source`` and ``transform_info`` for the mask transform.
        """
        return {
            "depth": self.depth[view_idx],
            "confidence": self.depth_conf[view_idx],
            "ray_confidence": self.world_points_conf[view_idx],
            "intrinsics": self.intrinsics[view_idx],
            "extrinsics": self.extrinsics[view_idx],
            "world_points": self.world_points[view_idx],
            # Not needed for VGGT (world_points are direct), but kept for
            # interface compat -- set to None so callers don't use them.
            "ray_origins_world": None,
            "ray_directions_world": None,
            # Flag for mask-transform logic in load.py
            "source": "vggt",
            "transform_info": self.transform_info[view_idx],
        }


# ---------------------------------------------------------------------------
# Pseudo DepthPrediction for compatibility with code that reads .extrinsics
# ---------------------------------------------------------------------------


class _PseudoDepthPrediction:
    """Lightweight shim so code that reads pred.extrinsics / pred.intrinsics
    / pred.depth still works when the actual depth came from VGGT."""

    def __init__(self, vggt_result: VGGTResult, view_idx: int):
        self.depth = vggt_result.depth[view_idx]
        self.confidence = vggt_result.depth_conf[view_idx]
        self.intrinsics = vggt_result.intrinsics[view_idx]
        self.extrinsics = vggt_result.extrinsics[view_idx, :3, :]  # (3, 4) world-to-camera
        self.is_metric = False  # VGGT is relative-scale
        self.ray_map = None
        self.ray_confidence = None


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------


class VGGTReconstructor:
    """Synchronous VGGT-1B 3D reconstruction provider.

    Usage::

        rec = VGGTReconstructor(device="cuda")
        result = rec.reconstruct(images)  # list of PIL Images
        geom = result.get_world_geometry(0)  # per-view geometry dict
    """

    MODEL_ID = "facebook/VGGT-1B"

    def __init__(
        self,
        device: str = "cuda",
        dtype: Optional[str] = "bfloat16",
    ):
        _ensure_torch()
        self._device = device
        self._dtype = getattr(_torch, dtype) if dtype else _torch.float32
        self._model = None  # lazy load

    # -- lazy model loading --------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        import os
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        vendored_vggt_root = os.path.join(repo_root, "tools", "Orient-Anything-V2")
        if os.path.isdir(vendored_vggt_root) and vendored_vggt_root not in sys.path:
            sys.path.insert(0, vendored_vggt_root)
        from vggt.models.vggt import VGGT

        log.info(f"[VGGTReconstructor] Loading {self.MODEL_ID} ...")
        self._model = VGGT.from_pretrained(self.MODEL_ID)
        self._model.to(self._device)
        self._model.eval()
        log.info("[VGGTReconstructor] Model loaded.")

    # -- public API ----------------------------------------------------------

    def reconstruct(self, images: List[Image.Image]) -> VGGTResult:
        """Run VGGT on all images in a single forward pass.

        Parameters
        ----------
        images : list of PIL Images (one per view)

        Returns
        -------
        VGGTResult with unified world-space point clouds and cameras.
        """
        self._ensure_model()
        _ensure_torch()
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        image_tensor, transform_info = _preprocess_images(images)
        image_tensor = image_tensor.to(self._device)

        with _torch.no_grad(), _torch.amp.autocast(self._device, dtype=self._dtype):
            predictions = self._model(image_tensor)

        # -- Extract outputs (squeeze batch dim, move to CPU) ----------------
        # predictions shapes: world_points [B,S,H,W,3], pose_enc [B,S,9], etc.
        world_points = predictions["world_points"][0].cpu().float().numpy()
        world_points_conf = predictions["world_points_conf"][0].cpu().float().numpy()
        depth = predictions["depth"][0].cpu().float().numpy()
        if depth.ndim == 4:  # (S, H, W, 1) -> (S, H, W)
            depth = depth[..., 0]
        depth_conf = predictions["depth_conf"][0].cpu().float().numpy()

        # -- Camera poses ----------------------------------------------------
        extrinsic_34, intrinsic_33 = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_tensor.shape[-2:]
        )
        # extrinsic_34: (B, S, 3, 4) -> (S, 4, 4)
        ext = extrinsic_34[0].cpu().float().numpy()  # (S, 3, 4)
        S = ext.shape[0]
        extrinsics = np.zeros((S, 4, 4), dtype=np.float64)
        extrinsics[:, :3, :] = ext
        extrinsics[:, 3, 3] = 1.0

        intrinsics = intrinsic_33[0].cpu().float().numpy()  # (S, 3, 3)

        # Canonicalize to the Y-up world (see conventions.py and
        # adapters.canonicalize_y_up). VGGT emits OpenCV-convention
        # extrinsics in a Y-down world; flip poses + world_points to true
        # Y-up at the model boundary so downstream never sees Y-down data.
        # VGGT shares one world across views — decide once on view 0 and
        # apply uniformly.
        from saturn.scene.adapters import _is_y_down_extrinsics, _FLIP_Y_4X4, _FLIP_Y_3
        if S > 0 and _is_y_down_extrinsics(extrinsics[0]):
            extrinsics = np.einsum("sij,jk->sik", extrinsics, _FLIP_Y_4X4)
            world_points = world_points * _FLIP_Y_3

        return VGGTResult(
            world_points=world_points,
            world_points_conf=world_points_conf,
            depth=depth,
            depth_conf=depth_conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics.astype(np.float64),
            transform_info=transform_info,
            num_views=S,
        )
