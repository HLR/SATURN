"""
Multi-view scene types for SATURN.

Core data structures: MergedObject, Camera, CameraHeading, DirectionValue.
"""

from __future__ import annotations

import base64
import copy
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .anchor import Anchor
from .direction_utils import (
    RELATIVE_LABELS_4,
    RELATIVE_LABELS_8,
)


class SceneBuildError(RuntimeError):
    """Raised when a Scene cannot be built — e.g. an object with no
    orientation under ``strict=True``."""


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _ambiguous_axis(name: str, vec: str, predicate: str) -> property:
    """``entity.up`` could mean the up VECTOR or an "above" PREDICATE, and
    ``entity.front`` is already the predicate column; refuse the ambiguous
    name with the two unambiguous ones instead of guessing."""

    def _get(self):
        raise AttributeError(
            f"{type(self).__name__}.{name} is not defined: use .{vec} for the "
            f"unit {name} axis (world frame) or .{predicate} for the predicate column."
        )

    return property(_get)


# ---------------------------------------------------------------------------
# DirectionValue
# ---------------------------------------------------------------------------


class DirectionValue:
    """
    Represents a direction in a reference frame as spherical coordinates.

    - yaw: 0 = straight ahead, increases clockwise (degrees)
    - elevation: 0 = horizontal, positive = up, negative = down (degrees)
    """

    def __init__(self, yaw_deg: float, elevation_deg: float = 0.0):
        self._yaw = float(yaw_deg) % 360.0
        self._elevation = float(elevation_deg)

    # -- accessors --

    def degree(self) -> float:
        """Horizontal angle: 0 = ahead, increases clockwise."""
        return self._yaw

    def yaw_degree(self) -> float:
        """Alias for degree()."""
        return self._yaw

    def elevation_degree(self) -> float:
        """Vertical angle: 0 = horizontal, +up, -down."""
        return self._elevation

    def elevation(self) -> float:
        """Alias for elevation_degree()."""
        return self._elevation

    def to_vec(self) -> np.ndarray:
        """Return a unit 3D vector in the reference frame's (right, up, front)
        basis, consistent with the (yaw, elevation) representation.

        - ``yaw=0, elevation=0`` → ``[0, 0, 1]`` (straight ahead / +front).
        - ``yaw=90, elevation=0`` → ``[1, 0, 0]`` (right).
        - ``elevation=90`` → ``[0, 1, 0]`` (up).
        """
        y = math.radians(self._yaw)
        e = math.radians(self._elevation)
        ce = math.cos(e)
        return np.array([ce * math.sin(y), math.sin(e), ce * math.cos(y)], dtype=float)

    def unit(self) -> np.ndarray:
        """Alias for :py:meth:`to_vec`."""
        return self.to_vec()

    def label(self, freedom: int = 4) -> str:
        """Discretize to a string label.

        freedom=4: front, right, back, left
        freedom=8: front, front-right, right, back-right, back, back-left, left, front-left

        The spelling is canonical ("back", never "behind"); the returned
        :class:`~saturn.scene.direction_utils.DirectionLabel` compares
        alias-aware, so ``label(8) == "behind-left"`` holds for "back-left".
        """
        if freedom == 4:
            labels = RELATIVE_LABELS_4
            step = 360.0 / 4
        elif freedom == 8:
            labels = RELATIVE_LABELS_8
            step = 360.0 / 8
        else:
            raise ValueError(f"freedom must be 4 or 8, got {freedom}")

        half = step / 2.0
        yaw = self._yaw % 360.0
        idx = int((yaw + half) / step) % len(labels)
        return labels[idx]

    def __repr__(self) -> str:
        return f"DirectionValue(yaw={self._yaw:.1f}, elevation={self._elevation:.1f})"


# ---------------------------------------------------------------------------
# CameraHeading
# ---------------------------------------------------------------------------


class CameraHeading:
    """Camera heading expressed as a forward direction vector in world frame."""

    def __init__(self, forward: np.ndarray):
        fwd = np.asarray(forward, dtype=float).ravel()[:3]
        norm = np.linalg.norm(fwd)
        self._forward = fwd / (norm + 1e-12)

    @property
    def forward(self) -> np.ndarray:
        return self._forward.copy()

    @forward.setter
    def forward(self, value: np.ndarray):
        """Set the forward direction (re-normalizes)."""
        fwd = np.asarray(value, dtype=float).ravel()[:3]
        norm = np.linalg.norm(fwd)
        self._forward = fwd / (norm + 1e-12)

    @property
    def right(self) -> np.ndarray:
        """Right direction: cross(forward, world_up), projected to horizontal."""
        world_up = np.array([0.0, 1.0, 0.0])
        r = np.cross(self._forward, world_up)
        n = np.linalg.norm(r)
        if n < 1e-6:
            # forward is nearly vertical — use world +X as right
            return np.array([1.0, 0.0, 0.0])
        return r / n

    @property
    def up(self) -> np.ndarray:
        """Up direction: cross(right, forward)."""
        r = self.right
        u = np.cross(r, self._forward)
        n = np.linalg.norm(u)
        if n < 1e-6:
            return np.array([0.0, 1.0, 0.0])
        return u / n


    def __repr__(self) -> str:
        return f"CameraHeading(forward={self._forward})"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass
class Camera(Anchor):
    """A camera in the multi-view scene."""

    id: int
    entity_id: int  # index usable in distance tensors
    intrinsics: np.ndarray  # (3,3)
    extrinsics: np.ndarray  # (4,4) world-to-camera
    image_size: Tuple[int, int]  # (H, W)
    heading: CameraHeading = field(default=None)
    position_world: np.ndarray = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # ----- Anchor protocol fields -----
    # ``anchor_index`` is set by ``Scene.__init__`` (None = free-floating).
    anchor_index: Optional[int] = field(default=None)
    # Cameras have orientation by construction (extrinsics-derived).
    orientation_confidence: float = field(default=1.0)

    def __post_init__(self):
        # Derive position and heading from extrinsics if not provided
        if self.position_world is None:
            self.position_world = self._extract_position()
        if self.heading is None:
            self.heading = self._extract_heading()

    def _extract_position(self) -> np.ndarray:
        """Camera center in world frame: -R^T @ t."""
        ext = np.asarray(self.extrinsics, dtype=float)
        R_mat = ext[:3, :3]
        t = ext[:3, 3]
        return -R_mat.T @ t

    def _extract_heading(self) -> CameraHeading:
        """Forward direction = +Z axis of camera, in world frame (OpenCV convention).

        Extrinsics use OpenCV convention where +Z points into the scene.
        In world frame the camera forward is R_w2c^T @ [0,0,1] = third row of R_w2c.
        """
        ext = np.asarray(self.extrinsics, dtype=float)
        R_mat = ext[:3, :3]
        # OpenCV: camera looks along +Z in camera frame.
        # World-frame forward = R_w2c^T @ [0,0,1] (no negation).
        forward_world = R_mat.T @ np.array([0.0, 0.0, 1.0])
        return CameraHeading(forward_world)

    def clone(self) -> "Camera":
        """Deep copy of this camera; the ``_scene`` back-reference is shared,
        not copied (deep-copying it would clone the whole Scene)."""
        scene = getattr(self, "_scene", None)
        memo = {id(scene): scene} if scene is not None else None
        return copy.deepcopy(self, memo)

    # ------------------------------------------------------------------
    # Convenience: cam.first_person / cam.third_person
    # ------------------------------------------------------------------
    # Scene-builder sets ``cam._scene = scene`` so generated DSL code can write
    # ``scene.cameras[k].third_person.left("x1","x2")`` directly without first
    # building a frame. Equivalent to:
    #     view = scene.frame(position=cam.position, orientation=cam.orientation)
    #     view.third_person.left("x1","x2")

    def _build_view(self):
        scene = getattr(self, "_scene", None)
        if scene is None:
            raise RuntimeError(
                "Camera.first_person/third_person require a scene back-reference. "
                "Use scene.frame(position=cam.position, orientation=cam.orientation) "
                "explicitly, or build the camera through Scene().__init__."
            )
        return scene.frame(position=self.position, orientation=self.orientation)

    @property
    def first_person(self):
        return self._build_view().first_person

    @property
    def third_person(self):
        return self._build_view().third_person

    # ------------------------------------------------------------------
    # Frame-first entity surface: the names used by ``scene.frame(...)``
    # and the ``Frame`` API.
    # ------------------------------------------------------------------

    @property
    def pos(self) -> np.ndarray:
        """Camera position in world frame (alias for ``position_world``)."""
        return np.asarray(self.position_world, dtype=float)

    @property
    def position(self) -> np.ndarray:
        """Camera position in world frame (3-vector)."""
        return np.asarray(self.position_world, dtype=float)

    @property
    def hfov_deg(self) -> Optional[float]:
        """Horizontal field-of-view in degrees, computed from intrinsics + image_size.

        Returns ``None`` if intrinsics are unavailable or degenerate. Pass this
        to ``scene.frame(position=..., orientation=..., hfov_deg=cam.hfov_deg)``
        to enable FOV-aware MCQ matching in ``view.match()``.
        """
        try:
            import math as _math
            K = np.asarray(self.intrinsics, dtype=float)
            fx = float(K[0, 0])
            if fx < 1e-6:
                return None
            W = float(self.image_size[1])  # (H, W)
            if W <= 0:  # e.g. MergedObject.clone() cameras: image_size=(0, 0)
                return None
            hfov = float(2.0 * _math.degrees(_math.atan(W / (2.0 * fx))))
            return hfov if _math.isfinite(hfov) and hfov > 0 else None
        except Exception:
            return None

    @property
    def orientation(self) -> np.ndarray:
        """Camera orientation in world frame as a 3x3 matrix.

        Columns are [right, up, front] in the right-hand-rule convention:
        ``right = cross(up, front)``. Using this matrix in
        ``scene.frame(position=..., orientation=...)`` yields the same
        FrameNamespace as ``scene.frame(position=camera(k + 1).position, orientation=camera(k + 1).orientation)``.

        Column 0 is image-right in the canonical world (see
        ``saturn.scene.adapters.canonicalize_y_up``: the Y flip is a
        reflection, so image-right = ``cross(up, front)``).
        ``self.heading.right`` (``cross(forward, up)``) is its exact
        negation, i.e. image-LEFT.
        """
        f = np.asarray(self.heading.forward, dtype=float)
        u = np.asarray(self.heading.up, dtype=float)
        nf = float(np.linalg.norm(f))
        if nf < 1e-9:
            raise ValueError("Camera.orientation: forward has zero magnitude.")
        f = f / nf
        nu = float(np.linalg.norm(u))
        if nu < 1e-9:
            u = np.array([0.0, 1.0, 0.0])
        else:
            u = u / nu
        if abs(float(np.dot(f, u))) > 0.999:
            u = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(f, u))) > 0.999:
                u = np.array([0.0, 0.0, 1.0])
        r = np.cross(u, f)
        r = r / (float(np.linalg.norm(r)) + 1e-12)
        u_ortho = np.cross(f, r)
        u_ortho = u_ortho / (float(np.linalg.norm(u_ortho)) + 1e-12)
        return np.column_stack([r, u_ortho, f])

    # Axis accessors (``front_vec``/``right_vec``/``up_vec``, unit, world
    # frame) live on Anchor — it derives them from ``heading.forward``:
    # ``right_vec = cross(world_up, front_vec) = orientation[:, 0]``, which is
    # IMAGE-right in the canonical (Y-flipped) world. ``heading.right`` is
    # its exact negation (image-LEFT); do not use it as "right".
    up = _ambiguous_axis("up", "up_vec", "above")
    down = _ambiguous_axis("down", "up_vec", "below")

    def rotate(
        self,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        *,
        yaw_deg: Optional[float] = None,
        pitch_deg: Optional[float] = None,
    ) -> "Camera":
        """Return a new ``Camera`` with rotated orientation (non-mutating).

        yaw: rotation around world Y axis (positive = clockwise from above,
            i.e. turn right; -90 = turn left)
        pitch: rotation around camera right axis (positive = tilt up)
        roll: rotation around camera forward axis

        Camera position is preserved.  The original camera is unchanged;
        the returned camera is a deep copy with rebuilt extrinsics so it
        can be chained or passed to ``scene.add_camera``.

        ``yaw_deg`` / ``pitch_deg`` are accepted as aliases for ``yaw`` /
        ``pitch`` — the canonical Anchor protocol names.
        """
        if yaw_deg is not None:
            yaw = yaw_deg
        if pitch_deg is not None:
            pitch = pitch_deg

        # Yaw about the WORLD Y axis, then pitch about the (yawed) body right
        # axis and roll about the body front axis -- a world-frame euler
        # "yxz" would pitch about world X, i.e. the wrong axis for any camera
        # not facing +/-Z, and tilt DOWN for a camera facing +Z.
        # ``_sync_pose`` then writes the rotated axes through to
        # position_world, heading, AND extrinsics atomically.
        yaw_rot = R.from_euler("y", yaw, degrees=True).as_matrix()
        new_front = yaw_rot @ np.asarray(self.front_vec, dtype=float)
        new_right = yaw_rot @ np.asarray(self.right_vec, dtype=float)
        new_up    = yaw_rot @ np.asarray(self.up_vec,    dtype=float)
        if pitch:
            cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
            new_front, new_up = new_front * cp + new_up * sp, new_up * cp - new_front * sp
        if roll:
            cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
            new_right, new_up = new_right * cr + new_up * sr, new_up * cr - new_right * sr

        new_cam = self._shallow_copy_for_transform()
        new_cam._sync_pose(self.position_world, new_front, new_right, new_up)
        new_cam.anchor_index = None
        return new_cam

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "intrinsics": _arr_to_list(self.intrinsics),
            "extrinsics": _arr_to_list(self.extrinsics),
            "image_size": list(self.image_size),
            "position_world": _arr_to_list(self.position_world),
            "heading_forward": _arr_to_list(self.heading.forward)
            if self.heading
            else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Camera":
        """Reconstruct from a dict produced by ``to_dict``."""
        cam = cls(
            id=d["id"],
            entity_id=d["entity_id"],
            intrinsics=np.array(d["intrinsics"], dtype=float),
            extrinsics=np.array(d["extrinsics"], dtype=float),
            image_size=tuple(d["image_size"]),
            metadata=d.get("metadata", {}),
        )
        # position_world and heading are derived from extrinsics in __post_init__
        return cam


# ---------------------------------------------------------------------------
# MergedObject
# ---------------------------------------------------------------------------


class MissingViewError(KeyError):
    """``obj.per_view_centers[v]`` for an image the object was not detected in.

    A ``KeyError`` (callers that catch KeyError keep working) whose message says
    which images the object WAS detected in; the executor adds which objects
    were detected in the requested image. ``mapping`` is the dict that raised."""

    def __init__(self, field_name: str, key, available, mapping=None):
        self.field_name, self.key, self.available, self.mapping = field_name, key, list(available), mapping
        super().__init__(key)

    def __str__(self) -> str:
        imgs = [k + 1 for k in self.available if isinstance(k, int)]
        if isinstance(self.key, (int, np.integer)) and not isinstance(self.key, bool):
            k = int(self.key)
            head = (f"{self.field_name}[{k}]: this object was not detected in image {k + 1} "
                    f"(keys are 0-based image indices), so its data for image {k + 1} is unknown.")
        else:
            head = f"{self.field_name}[{self.key!r}]: keys are 0-based image indices (int)."
        return f"{head} It was detected in images {imgs} (keys {self.available})."


class PerViewDict(dict):
    """Per-image data of an object, keyed by 0-based image index. Only the images
    where the object was detected have an entry; a missing key raises
    ``MissingViewError`` (a KeyError) that names the images that do."""

    _field_name = "per_view"

    def __missing__(self, key):
        raise MissingViewError(self._field_name, key, sorted(self.keys(), key=str), mapping=self)

    def __reduce__(self):
        return (_make_per_view_dict, (self._field_name, dict(self)))


def _make_per_view_dict(field_name: str, items: dict) -> "PerViewDict":
    d = PerViewDict(items)
    d._field_name = field_name
    return d


@dataclass
class MergedObject(Anchor):
    """A merged object across views."""

    id: int
    label: str  # keyword/prompt string used for detection
    views: List[int]  # view indices where detected

    # 3D world-frame geometry
    center_world: np.ndarray  # (3,)
    rotation_world: np.ndarray  # (3,3) columns = right, up, front
    front_world: np.ndarray  # (3,) unit front direction
    up_world: np.ndarray  # (3,) unit up direction
    right_world: np.ndarray  # (3,) unit right direction
    euler_world_deg: np.ndarray  # (3,) azimuth, elevation, roll
    dims: np.ndarray  # (3,) width, height, depth
    corners_world: np.ndarray  # (8,3) OBB corners
    height: float  # from dims[1]
    support_y: float  # bottom-most Y
    world_points: Optional[np.ndarray] = None  # (N,3) concatenated point cloud

    # Per-view data
    per_view_bboxes: Dict[int, List[float]] = field(default_factory=dict)
    per_view_masks: Dict[int, np.ndarray] = field(default_factory=dict)
    per_view_scores: Dict[int, float] = field(default_factory=dict)
    per_view_centers: Dict[int, np.ndarray] = field(default_factory=dict)
    # Per-view object front directions (unit vector in world frame) from Oriany.
    # Used by scene.object_rotation(...) to detect per-object rotation between
    # frames when the camera is stationary.
    per_view_fronts: Dict[int, np.ndarray] = field(default_factory=dict)
    # Per-view object front directions in the CAMERA frame (unit vector,
    # OpenCV convention: +X right, +Y down, +Z forward) directly from
    # Oriany's `front_direction_3d`. Used by
    # `scene.object_rotation_camera_frame(...)` so orientation-delta
    # primitives can sidestep VGGT extrinsic noise (see
    # Scene.object_rotation_camera_frame).
    per_view_fronts_camera: Dict[int, np.ndarray] = field(default_factory=dict)
    per_view_orientation_confidence: Dict[int, float] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----- Anchor protocol fields -----
    # ``anchor_index`` is set by ``Scene.__init__`` (None = free-floating).
    anchor_index: Optional[int] = field(default=None)
    # Scalar orientation confidence (max across views) — derived in
    # ``__post_init__`` if ``per_view_orientation_confidence`` is non-empty.
    orientation_confidence: float = field(default=1.0)

    def __setattr__(self, name, value):
        # per_view_* dicts are PerViewDicts, so a missing image is an actionable error.
        if name.startswith("per_view_") and type(value) is dict:
            value = _make_per_view_dict(name, value)
        object.__setattr__(self, name, value)

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k, v in list(state.items()):
            if k.startswith("per_view_") and type(v) is dict:
                self.__dict__[k] = _make_per_view_dict(k, v)

    def __post_init__(self):
        # Derive scalar orientation_confidence from per-view dict, if any.
        if self.per_view_orientation_confidence:
            try:
                self.orientation_confidence = float(
                    max(self.per_view_orientation_confidence.values())
                )
            except (ValueError, TypeError):
                pass

    def clone(self) -> "Camera":
        """Create a Camera from this object's pose (for hypothetical observer)."""
        # Build extrinsics from object position + orientation
        # Object's rotation_world has columns [right, up, front]
        right = np.asarray(self.right_world, dtype=float)
        up = np.asarray(self.up_world, dtype=float)
        front = np.asarray(self.front_world, dtype=float)
        center = np.asarray(self.center_world, dtype=float)

        # Camera convention: X=right, Y=down, Z=forward
        # Object convention: right, up, front
        # Camera R_c2w columns: [right, -up, front] (camera Y is down)
        R_c2w = np.column_stack([right, -up, front])
        R_w2c = R_c2w.T
        t = -R_w2c @ center

        ext = np.eye(4, dtype=float)
        ext[:3, :3] = R_w2c
        ext[:3, 3] = t

        # Use dummy intrinsics and image size
        K = np.eye(3, dtype=float)
        return Camera(
            id=-1,  # will be reassigned
            entity_id=-1,  # will be reassigned
            intrinsics=K,
            extrinsics=ext,
            image_size=(0, 0),
            position_world=center.copy(),
        )

    # ------------------------------------------------------------------
    # Frame-first entity surface: the names used by ``scene.frame(...)``
    # and the ``Frame`` API.
    # ------------------------------------------------------------------

    @property
    def pos(self) -> np.ndarray:
        """Object center in world frame (alias for ``center_world``)."""
        return np.asarray(self.center_world, dtype=float)

    @property
    def position(self) -> np.ndarray:
        """Object center in world frame (3-vector)."""
        return np.asarray(self.center_world, dtype=float)

    @property
    def orientation(self) -> np.ndarray:
        """Object orientation in world frame as a 3x3 matrix.

        Columns are [right, up, front] in the right-hand-rule convention:
        ``right = cross(up, front)``. Using this matrix in
        ``scene.frame(position=..., orientation=...)`` yields the same
        FrameNamespace as the engine's ``scene._frame(at=obj_idx)`` (programs: ``scene.frame(at=<description>)``).

        Column 0 (``cross(up, front)``) is the object's body-right: for an
        object facing the same way as a camera it equals that camera's
        image-right.  The MergedObject build paths (fusion,
        ``scene.constraint.face``, VLM grounding) store ``right_world`` with
        this same sign; :py:attr:`right_vec` is derived from this matrix so
        it is right-handed by construction whatever ``right_world`` holds.
        """
        f = np.asarray(self.front_world, dtype=float)
        u = np.asarray(self.up_world, dtype=float)
        nf = float(np.linalg.norm(f))
        if nf < 1e-9:
            raise ValueError("MergedObject.orientation: front has zero magnitude.")
        f = f / nf
        nu = float(np.linalg.norm(u))
        if nu < 1e-9:
            u = np.array([0.0, 1.0, 0.0])
        else:
            u = u / nu
        if abs(float(np.dot(f, u))) > 0.999:
            u = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(f, u))) > 0.999:
                u = np.array([0.0, 0.0, 1.0])
        r = np.cross(u, f)
        r = r / (float(np.linalg.norm(r)) + 1e-12)
        u_ortho = np.cross(f, r)
        u_ortho = u_ortho / (float(np.linalg.norm(u_ortho)) + 1e-12)
        return np.column_stack([r, u_ortho, f])

    # Axis vectors: unit, world frame, orthonormal, right-handed, and the
    # same names as on Camera. ``front_world`` / ``up_world`` /
    # ``right_world`` remain the stored fields; ``.front`` / ``.right`` are
    # predicate columns (Anchor), not vectors.
    @property
    def front_vec(self) -> np.ndarray:
        """Unit front axis in world frame (``orientation[:, 2]``)."""
        if float(np.linalg.norm(self.front_world)) < 1e-9:
            return _unit(self.front_world)
        return self.orientation[:, 2]

    @property
    def right_vec(self) -> np.ndarray:
        """Unit body-right axis in world frame: ``cross(up, front)`` =
        ``orientation[:, 0]`` (same side as a co-facing camera's image-right)."""
        if float(np.linalg.norm(self.front_world)) < 1e-9:
            return _unit(self.right_world)
        return self.orientation[:, 0]

    @property
    def up_vec(self) -> np.ndarray:
        """Unit up axis in world frame, orthogonal to front (``orientation[:, 1]``)."""
        if float(np.linalg.norm(self.front_world)) < 1e-9:
            return _unit(self.up_world)
        return self.orientation[:, 1]

    up = _ambiguous_axis("up", "up_vec", "above")
    down = _ambiguous_axis("down", "up_vec", "below")

    @property
    def extent(self) -> Dict[str, float]:
        """3D bounding-box extents.

        Returns a dict with ``width`` (x), ``height`` (y), ``depth`` (z) and
        ``length`` = ``max(width, depth)`` (the larger horizontal extent).
        """
        d = np.asarray(self.dims, dtype=float).ravel()
        if d.size < 3:
            d = np.pad(d, (0, 3 - d.size))
        w, h, dp = float(d[0]), float(d[1]), float(d[2])
        return {
            "width": w,
            "height": h,
            "depth": dp,
            "length": max(w, dp),
        }

    # ``rotate(yaw, pitch)`` and ``translate(delta)`` are inherited from
    # ``Anchor``. ``Anchor.rotate`` returns a new ``MergedObject`` with
    # rotated axes; ``_sync_pose`` updates ``front_world`` / ``right_world``
    # / ``up_world`` / ``rotation_world`` / ``corners_world`` /
    # ``euler_world_deg`` atomically. Semantic fields (``label``, ``dims``,
    # ``per_view_*``) are pose-invariant or observation snapshots and
    # carry through unchanged.

    def to_dict(self, *, include_points: bool = False) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Parameters
        ----------
        include_points : bool
            If True, include subsampled ``world_points`` (max 2000 per object).
            Default False to keep JSON small.
        """
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "views": list(self.views),
            "center_world": _arr_to_list(self.center_world),
            "rotation_world": _arr_to_list(self.rotation_world),
            "front_world": _arr_to_list(self.front_world),
            "up_world": _arr_to_list(self.up_world),
            "right_world": _arr_to_list(self.right_world),
            "euler_world_deg": _arr_to_list(self.euler_world_deg),
            "dims": _arr_to_list(self.dims),
            "corners_world": _arr_to_list(self.corners_world),
            "height": float(self.height),
            "support_y": float(self.support_y),
            "per_view_bboxes": {
                str(k): list(v) for k, v in self.per_view_bboxes.items()
            },
            "per_view_scores": {
                str(k): float(v) for k, v in self.per_view_scores.items()
            },
            "per_view_centers": {
                str(k): _arr_to_list(v) for k, v in self.per_view_centers.items()
            },
            "per_view_fronts": {
                str(k): _arr_to_list(v) for k, v in self.per_view_fronts.items()
            },
            "per_view_fronts_camera": {
                str(k): _arr_to_list(v)
                for k, v in self.per_view_fronts_camera.items()
            },
            "per_view_orientation_confidence": {
                str(k): float(v) for k, v in self.per_view_orientation_confidence.items()
            },
            "metadata": self.metadata,
        }
        # Serialize masks as zlib-compressed base64 with shape info
        if self.per_view_masks:
            encoded_masks = {}
            for k, mask in self.per_view_masks.items():
                if mask is not None and isinstance(mask, np.ndarray):
                    flat = mask.astype(np.uint8).tobytes()
                    compressed = zlib.compress(flat, level=6)
                    encoded_masks[str(k)] = {
                        "data": base64.b64encode(compressed).decode("ascii"),
                        "shape": list(mask.shape),
                    }
            if encoded_masks:
                d["per_view_masks"] = encoded_masks
        if include_points and self.world_points is not None:
            pts = self.world_points
            max_pts = 2000
            if len(pts) > max_pts:
                idx = np.linspace(0, len(pts) - 1, max_pts, dtype=int)
                pts = pts[idx]
            d["world_points"] = _arr_to_list(pts)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MergedObject":
        """Reconstruct from a dict produced by ``to_dict``."""
        world_points = None
        if "world_points" in d and d["world_points"] is not None:
            world_points = np.array(d["world_points"], dtype=float)
        return cls(
            id=d["id"],
            label=d["label"],
            views=d["views"],
            center_world=np.array(d["center_world"], dtype=float),
            rotation_world=np.array(d["rotation_world"], dtype=float),
            front_world=np.array(d["front_world"], dtype=float),
            up_world=np.array(d["up_world"], dtype=float),
            right_world=np.array(d["right_world"], dtype=float),
            euler_world_deg=np.array(d["euler_world_deg"], dtype=float),
            dims=np.array(d["dims"], dtype=float),
            corners_world=np.array(d["corners_world"], dtype=float),
            height=float(d["height"]),
            support_y=float(d["support_y"]),
            world_points=world_points,
            per_view_bboxes={
                int(k): list(v) for k, v in d.get("per_view_bboxes", {}).items()
            },
            per_view_scores={
                int(k): float(v) for k, v in d.get("per_view_scores", {}).items()
            },
            per_view_masks=_decode_masks(d.get("per_view_masks", {})),
            per_view_centers={
                int(k): np.array(v, dtype=float)
                for k, v in d.get("per_view_centers", {}).items()
            },
            per_view_fronts={
                int(k): np.array(v, dtype=float)
                for k, v in d.get("per_view_fronts", {}).items()
            },
            per_view_fronts_camera={
                int(k): np.array(v, dtype=float)
                for k, v in d.get("per_view_fronts_camera", {}).items()
            },
            per_view_orientation_confidence={
                int(k): float(v)
                for k, v in d.get("per_view_orientation_confidence", {}).items()
            },
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _arr_to_list(arr) -> Any:
    """Convert numpy array (or None) to nested Python list."""
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def _decode_masks(raw: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """Decode per_view_masks from zlib+base64 encoded dict."""
    masks: Dict[int, np.ndarray] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "data" in v and "shape" in v:
            compressed = base64.b64decode(v["data"])
            flat = zlib.decompress(compressed)
            masks[int(k)] = np.frombuffer(flat, dtype=np.uint8).reshape(v["shape"])
    return masks
