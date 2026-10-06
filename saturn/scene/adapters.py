"""Neural-model adapters → canonical SATURN conventions.

Single entry point for converting each external neural model's native output
into SATURN's canonical (right, up, forward) frames defined in
``conventions.py``. ALL neural-model ingestion MUST go through one of
these adapters.

Supported sources:

* VGGT (``from_vggt``): world points, extrinsics, intrinsics.
* Orient-Anything V2 (``from_orient_anything``): (azimuth, polar, roll)
  camera-frame Euler triplet → world-frame (right, up, forward) triad.

Conventions recap:

* World: X-right, **Y-up**, **Z-forward-into-scene**, right-handed.
* Camera: OpenCV (X-right, Y-down, Z-forward). Extrinsics are world-to-camera.
* OA azimuth (user-az): 0 = object faces TOWARD the camera (front = -Z_cam).

See ``conventions.py`` for authoritative reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np



__all__ = [
    "OrientationTriad",
    "DepthAdapterResult",
    "from_vggt",
    "from_orient_anything",
    "from_camera_rotation_matrix",
    "canonicalize_y_up",
]


# ---------------------------------------------------------------------------
# Y-up canonicalization helper (used by from_vggt)
# ---------------------------------------------------------------------------

_FLIP_Y_4X4 = np.diag([1.0, -1.0, 1.0, 1.0])
_FLIP_Y_3 = np.array([1.0, -1.0, 1.0])


def _is_y_down_extrinsics(extrinsics_w2c: np.ndarray) -> bool:
    """True if extrinsics indicate a Y-down world (OpenCV camera in raw frame).

    Heuristic: an upright OpenCV camera has its image-down axis aligned with
    gravity. ``cam_down_world = R_c2w @ [0,1,0]`` measures where the camera's
    +Y_cam (image-down) axis points in world. Y-up world → cam_down_world ≈
    -Y_world (negative Y). Y-down world → cam_down_world ≈ +Y_world (positive
    Y). Threshold 0.3 leaves a wide margin for non-upright shots.
    """
    R_w2c = np.asarray(extrinsics_w2c, dtype=float)[:3, :3]
    cam_down_world = R_w2c.T @ np.array([0.0, 1.0, 0.0])
    return float(cam_down_world[1]) > 0.3


def canonicalize_y_up(
    extrinsics_w2c: np.ndarray,
    world_points: Optional[np.ndarray] = None,
) -> tuple:
    """Flip Y-down → canonical Y-up if needed; otherwise pass through.

    Returns ``(extrinsics_canonical, world_points_canonical_or_None)``.
    Idempotent: a Y-up input is returned unchanged.
    """
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=float)
    if not _is_y_down_extrinsics(extrinsics_w2c):
        return extrinsics_w2c, world_points
    extrinsics_canonical = extrinsics_w2c @ _FLIP_Y_4X4
    if world_points is None:
        return extrinsics_canonical, None
    return extrinsics_canonical, np.asarray(world_points, dtype=float) * _FLIP_Y_3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OrientationTriad:
    """Canonical orientation triad in SATURN world frame.

    Columns of the equivalent rotation matrix: [right, up, forward].
    All three vectors are unit-length and mutually orthogonal
    (right-handed).
    """

    right: np.ndarray  # (3,)
    up: np.ndarray  # (3,)
    forward: np.ndarray  # (3,)
    confidence: float = 0.0
    source: str = ""  # e.g. "orient_anything_v2"


@dataclass
class DepthAdapterResult:
    """Canonical per-view depth/geometry result.

    All arrays are in the canonical conventions: camera extrinsics are
    world-to-camera 4x4 with OpenCV camera axes; world_points live in
    SATURN world space (X-right, Y-up, Z-forward).
    """

    world_points: np.ndarray  # (H, W, 3) or (N, 3)
    depth: np.ndarray  # (H, W)
    intrinsics: np.ndarray  # (3, 3)
    extrinsics_w2c: np.ndarray  # (4, 4)
    confidence: Optional[np.ndarray] = None  # (H, W) or None
    source: str = ""
    raw: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _as_4x4(ext: np.ndarray) -> np.ndarray:
    ext = np.asarray(ext, dtype=float)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        m = np.eye(4, dtype=float)
        m[:3, :] = ext
        return m
    raise ValueError(f"Extrinsic must be (3,4) or (4,4), got {ext.shape}")


def _invert_w2c(ext_w2c: np.ndarray) -> np.ndarray:
    """Return camera-to-world 4x4 given world-to-camera 4x4."""
    return np.linalg.inv(_as_4x4(ext_w2c))


def _rotation_from_user_euler_oa(az_deg: float, polar_deg: float, roll_deg: float) -> np.ndarray:
    """OA-V2 user-Euler → camera-frame rotation matrix (cols = right, up, forward).

    Delegates to ``saturn.perception.orientation.convention.rotation_matrix_from_user_euler``
    (single source of truth, derived from OA's reference code).

    Convention (OpenCV camera):
        At (az, polar, roll) = (0, 0, 0) the object's forward axis points
        along +Z_cv (into the scene / away from the camera) — this matches
        OA's internal convention where az=0 has the camera seeing the
        object's BACK.
    """
    from saturn.perception.orientation.convention import (
        rotation_matrix_from_user_euler,
    )

    return rotation_matrix_from_user_euler(
        [float(az_deg), float(polar_deg), float(roll_deg)]
    )


# ---------------------------------------------------------------------------
# VGGT depth adapter
# ---------------------------------------------------------------------------


def _validate_depth_geometry(
    world_points: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    source: str,
) -> None:
    """Light sanity checks -- does the camera look in a sensible direction?

    VGGT produces extrinsics in OpenCV convention already. If the
    Y-up assumption of the canonical world is violated we'd expect the
    camera to be upside-down in the scene, which would show up as
    ``R_c2w @ CAM_DOWN`` having a strong +Y component (i.e. camera's "down"
    axis points up in the world). Warn (don't fail) if that happens.
    """
    ext4 = _as_4x4(extrinsics_w2c)
    R_w2c = ext4[:3, :3]
    # Camera down axis in world frame: cam_down_world = R_c2w @ CAM_DOWN.
    R_c2w = R_w2c.T
    cam_down_world = R_c2w @ np.array([0.0, 1.0, 0.0])
    # If cam_down_world has strongly +Y component the world is Y-DOWN, not Y-UP.
    if cam_down_world[1] > 0.9:
        import warnings

        warnings.warn(
            f"[{source}] Camera down-axis points UP in world frame "
            f"(cam_down_world={cam_down_world}). Extrinsics may not match "
            f"SATURN's Y-up world convention. See multiview/conventions.py.",
            stacklevel=3,
        )


def from_vggt(view_geometry: Dict[str, Any]) -> DepthAdapterResult:
    """Convert a VGGT per-view geometry dict into ``DepthAdapterResult``.

    Accepts the dict produced by ``VGGTResult.get_world_geometry(view_idx)``.
    VGGT's native convention matches SATURN canonical.
    """
    extrinsics = _as_4x4(view_geometry["extrinsics"])
    intrinsics = np.asarray(view_geometry["intrinsics"], dtype=float)
    depth = np.asarray(view_geometry["depth"], dtype=float)
    world_points = np.asarray(view_geometry["world_points"], dtype=float)
    confidence = view_geometry.get("confidence")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)

    extrinsics, world_points = canonicalize_y_up(extrinsics, world_points)

    _validate_depth_geometry(world_points, extrinsics, intrinsics, source="VGGT")

    return DepthAdapterResult(
        world_points=world_points,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics,
        confidence=confidence,
        source="vggt",
        raw=view_geometry,
    )


# ---------------------------------------------------------------------------
# Orient-Anything adapter
# ---------------------------------------------------------------------------


def from_orient_anything(
    *,
    azimuth_deg: float,
    polar_deg: float,
    roll_deg: float,
    extrinsics_w2c: np.ndarray,
    confidence: float = 0.0,
) -> OrientationTriad:
    """Convert an Orient-Anything V2 (az, polar, roll) triplet to world-frame.

    OA V2 outputs (azimuth_deg, polar_deg, rotation_deg) where the angles
    describe the CAMERA's spherical coordinates around the object:
      - az ∈ [0, 360)   camera azimuth around the object (OA convention)
      - polar ∈ [-90, 90]   camera elevation
      - rotation ∈ [-180, 180]   in-plane roll

    At az = 0 the object faces the camera: its front is -Z_cam (step 1 below).

    After world-frame fusion:
      - az_world = 0   → object faces +Z_world (into scene, away from initial camera)
      - az_world = 90  → object faces -X_world (world-left)
      - az_world = 180 → object faces -Z_world (toward initial camera)
      - az_world = 270 → object faces +X_world (world-right)

    The returned triad is in SATURN canonical world frame (X-right, Y-up,
    Z-forward, right-handed). Columns of the equivalent matrix are
    ``[right, up, forward]``.

    Parameters
    ----------
    azimuth_deg, polar_deg, roll_deg
        Raw OA outputs, in degrees.
    extrinsics_w2c
        World-to-camera extrinsic 4x4 (or 3x4) for the view the OA
        detection came from. Used to transform the camera-frame triad
        into world frame.
    confidence
        OA's reported confidence (0..1). Passed through to the triad.
    """
    # 1. Build camera-frame rotation from user-Euler.
    #    Columns of R_cam = [right_cam, up_cam, +Z_cam_local]. Under OA V2's
    #    formula D (Ry(az) @ Rx(pol) @ Rz(rot)) the training semantic is
    #    "az=0 ⇔ object faces the camera", which in OpenCV camera frame
    #    (+Z pointing INTO the scene, away from viewer) means the object's
    #    FRONT direction = −col2. Columns 0 (right) and 1 (up) of R_cam
    #    carry the correct sign already.
    #    (For az ≈ 0, −col2 in world points back at the camera: "faces camera".)
    R_cam = _rotation_from_user_euler_oa(azimuth_deg, polar_deg, roll_deg)

    # 2. Transform each column to world frame via c2w rotation.
    c2w = _invert_w2c(extrinsics_w2c)
    R_c2w = c2w[:3, :3]
    R_world = R_c2w @ R_cam

    right_world = _safe_normalize(R_world[:, 0])
    up_world = _safe_normalize(R_world[:, 1])
    forward_world = _safe_normalize(-R_world[:, 2])

    # 3. The triad is not re-orthonormalized against WORLD_UP; downstream
    #    code flattens to horizontal where needed.

    return OrientationTriad(
        right=right_world,
        up=up_world,
        forward=forward_world,
        confidence=float(confidence),
        source="orient_anything_v2",
    )


def from_camera_rotation_matrix(
    rotation_matrix_cam: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    confidence: float = 0.0,
    source: str = "camera_rotation_matrix",
) -> OrientationTriad:
    """Convert a camera-frame rotation matrix to a world-frame triad.

    Use this when a provider has already produced a 3x3 camera-frame
    rotation matrix with columns ``[right, up, forward]`` (e.g. via
    :func:`saturn.perception.orientation.convention.rotation_matrix_from_user_euler`).
    The adapter applies the camera-to-world rotation derived from
    ``extrinsics_w2c`` and returns the canonical triad.

    This is the preferred ingestion path for objects whose orientation
    has already been fused by the pose_fusion pipeline (which stores a
    ``rotation_matrix`` on the ``OrientationEstimate``).

    Parameters
    ----------
    rotation_matrix_cam
        3x3 rotation matrix in the CAMERA frame, columns = [right, up, forward].
    extrinsics_w2c
        World-to-camera extrinsic (3x4 or 4x4) for the view the rotation
        was expressed in.
    confidence
        Optional confidence score to attach to the triad.
    source
        Tag forwarded onto the resulting :class:`OrientationTriad`.
    """
    R_cam = np.asarray(rotation_matrix_cam, dtype=float)
    if R_cam.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix_cam must be (3,3); got {R_cam.shape}"
        )

    c2w = _invert_w2c(extrinsics_w2c)
    R_c2w = c2w[:3, :3]
    R_world = R_c2w @ R_cam

    # col2 of R_cam is the object's local +Z axis in camera frame. Under
    # OA V2 formula D this points INTO the scene when az=0; the object's
    # FRONT points TOWARD the viewer, so front = −col2.
    return OrientationTriad(
        right=_safe_normalize(R_world[:, 0]),
        up=_safe_normalize(R_world[:, 1]),
        forward=_safe_normalize(-R_world[:, 2]),
        confidence=float(confidence),
        source=source,
    )
