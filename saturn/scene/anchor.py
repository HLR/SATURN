"""Anchor — body-frame predicate backend.

Provides the shared backend for "from anchor X's perspective, which
direction is Y in?". ``Camera`` and ``MergedObject`` both inherit
``Anchor``. The 10 body-frame predicates — ``left``, ``right``, ``front``,
``back``, four diagonals, ``above``, ``below`` — are cosine-scored against
the anchor's ``orientation_front`` / ``orientation_right`` /
``orientation_up`` axes.

Index convention: subject-first, as in ``obj_left[i, j]`` ("i is to the
left of j, from j's perspective"); ``compute_anchor_predicates`` returns
matrices keyed the same way.

Horizontal predicates use the anchor's heading flattened to the ground
plane, so they score the horizontal yaw of the target regardless of the
anchor's pitch. ``view.first_person.<dir>[k]`` and
``predicates[<dir>][k, cam.anchor_index]`` agree for level anchors; for a
pitched camera the frame path projects onto the pitched axes instead.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import numpy as np
from saturn.log import get_logger

log = get_logger(__name__)


# 8 horizontal directions and their target yaws in radians.
# yaw is computed as atan2(disp · anchor_right, disp · anchor_front).
#   yaw =  0     → target is straight in front  (+front axis)
#   yaw = +pi/2  → target is to the right
#   yaw = -pi/2  → target is to the left
_TARGET_YAW_RAD = {
    "front":       0.0,
    "front_right": math.pi / 4,
    "right":       math.pi / 2,
    "back_right":  3 * math.pi / 4,
    "back":        math.pi,
    "back_left":  -3 * math.pi / 4,
    "left":       -math.pi / 2,
    "front_left": -math.pi / 4,
}

HORIZONTAL_PREDICATES = list(_TARGET_YAW_RAD.keys())
VERTICAL_PREDICATES = ["above", "below"]
PREDICATE_NAMES = HORIZONTAL_PREDICATES + VERTICAL_PREDICATES


def _flatten_axis(v: np.ndarray) -> np.ndarray:
    """Unit horizontal (y=0) part of axis ``v``; ``v`` itself if near-vertical.

    Horizontal predicates compare a horizontal displacement with the
    anchor's heading, so a pitched axis must be flattened AND renormalised:
    dotting with the raw pitched front shrinks the front component by
    cos(pitch) and biases the yaw toward +/-90 degrees.
    """
    h = np.array([v[0], 0.0, v[2]], dtype=float)
    n = float(np.linalg.norm(h))
    return h / n if n >= 1e-6 else v


def _score_single(
    anchor_pos: np.ndarray,
    anchor_front: np.ndarray,
    anchor_right: np.ndarray,
    target_pos: np.ndarray,
    predicate: str,
) -> float:
    """Cosine score for a single (anchor, target, predicate) tuple."""
    disp = target_pos - anchor_pos

    if predicate in ("above", "below"):
        norm = float(np.linalg.norm(disp))
        if norm < 1e-8:
            return float("nan")
        y_norm = float(disp[1] / norm)
        if predicate == "above":
            return float((1.0 + y_norm) / 2.0)
        return float((1.0 - y_norm) / 2.0)

    # Horizontal: project onto the anchor's front/right plane.
    disp_h = np.array([disp[0], 0.0, disp[2]], dtype=float)
    norm_h = float(np.linalg.norm(disp_h))
    if norm_h < 1e-8:
        return float("nan")
    fc = float(np.dot(disp_h, _flatten_axis(anchor_front))) / norm_h
    rc = float(np.dot(disp_h, _flatten_axis(anchor_right))) / norm_h
    yaw = math.atan2(rc, fc)
    target_yaw = _TARGET_YAW_RAD[predicate]
    return float((1.0 + math.cos(yaw - target_yaw)) / 2.0)


def _has_valid_orientation(anchor) -> bool:
    """True iff anchor exposes finite, non-zero front / right axes."""
    try:
        f = np.asarray(anchor.orientation_front, dtype=float)
        r = np.asarray(anchor.orientation_right, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return False
    if f.shape != (3,) or r.shape != (3,):
        return False
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(r))):
        return False
    if float(np.linalg.norm(f)) < 1e-6 or float(np.linalg.norm(r)) < 1e-6:
        return False
    return True


def compute_anchor_predicates(
    anchors: List["Anchor"],
    *,
    strict: bool = True,
) -> dict:
    """Precompute the (N, N) predicate matrices for ``len(anchors) == N``.

    Returns a dict keyed by predicate name (10 entries), each value an
    ``(N, N)`` float64 matrix in subject-first convention:
    ``predicates[name][i, j]`` = "is anchor i in <name> of anchor j, from
    j's perspective?".

    The diagonal (i == j) is NaN. When ``strict=False``, any anchor that
    lacks a valid orientation fills its column with NaN rather than
    raising.
    """
    N = len(anchors)
    out = {name: np.full((N, N), np.nan, dtype=float) for name in PREDICATE_NAMES}
    if N == 0:
        return out

    positions = np.array([np.asarray(a.position, dtype=float) for a in anchors])
    valid = [_has_valid_orientation(a) for a in anchors]

    fronts = np.zeros((N, 3), dtype=float)
    rights = np.zeros((N, 3), dtype=float)
    for j, a in enumerate(anchors):
        if valid[j]:
            f = np.asarray(a.orientation_front, dtype=float)
            r = np.asarray(a.orientation_right, dtype=float)
            fronts[j] = _flatten_axis(f / (float(np.linalg.norm(f)) + 1e-12))
            rights[j] = _flatten_axis(r / (float(np.linalg.norm(r)) + 1e-12))

    for j in range(N):
        if not valid[j]:
            # Whole column j stays NaN — j is unusable as a reference frame.
            continue
        fj = fronts[j]
        rj = rights[j]
        pj = positions[j]
        for i in range(N):
            if i == j:
                continue
            disp = positions[i] - pj
            norm = float(np.linalg.norm(disp))
            if norm < 1e-8:
                continue  # leave NaN — coincident anchors

            # Vertical predicates use elevation.
            y_norm = float(disp[1] / norm)
            out["above"][i, j] = (1.0 + y_norm) / 2.0
            out["below"][i, j] = (1.0 - y_norm) / 2.0

            # Horizontal predicates share a single yaw computation.
            disp_h = np.array([disp[0], 0.0, disp[2]], dtype=float)
            norm_h = float(np.linalg.norm(disp_h))
            if norm_h < 1e-8:
                continue  # target overhead; horizontal predicates stay NaN
            fc = float(np.dot(disp_h, fj)) / norm_h
            rc = float(np.dot(disp_h, rj)) / norm_h
            yaw = math.atan2(rc, fc)
            for name, target_yaw in _TARGET_YAW_RAD.items():
                out[name][i, j] = (1.0 + math.cos(yaw - target_yaw)) / 2.0
    return out


class Anchor:
    """Mixin providing 10 body-frame predicates and immutable transforms.

    Subclasses MUST expose:
      - ``position``               : (3,) world-frame coordinates
      - ``front_vec`` / ``right_vec`` / ``up_vec`` : (3,) body-frame axes
      - ``orientation_confidence`` : float in [0, 1]
      - ``anchor_index``           : int (slot in scene.anchors) or None
      - ``_scene``                 : Scene back-reference, or None

    Anchor supplies:
      - ``orientation_front`` / ``orientation_right`` / ``orientation_up``
        as canonical aliases over ``front_vec`` / ``right_vec`` / ``up_vec``.
      - 10 predicate @properties: ``left``, ``right``, ``front``, ``back``,
        ``front_left``, ``front_right``, ``back_left``, ``back_right``,
        ``above``, ``below``. Predicates win at top-level; axis access
        goes through ``*_vec``.

    Predicates read from ``self._scene._anchor_predicates`` when the anchor
    is in the scene's anchor list. Otherwise they compute lazily against
    the scene's existing entity positions (free-floating anchors returned
    by ``translate`` / ``rotate``).
    """

    # No class-level type annotations on required fields: @dataclass
    # subclasses would treat them as dataclass fields and break __init__.

    # ------------------------------------------------------------------
    # Axis accessors. Anchor reads axes from whichever storage shape the
    # concrete subclass uses: ``front_world`` for MergedObject, ``heading``
    # for Camera. Subclasses don't declare their own ``*_vec`` properties.
    #
    # ``orientation_*`` and ``*_vec`` are aliases.
    # ------------------------------------------------------------------

    @property
    def front_vec(self) -> np.ndarray:
        v = getattr(self, "front_world", None)
        if v is not None:
            return np.asarray(v, dtype=float)
        heading = getattr(self, "heading", None)
        if heading is not None:
            return np.asarray(heading.forward, dtype=float)
        raise AttributeError(f"{type(self).__name__}: no front-axis storage")

    @property
    def right_vec(self) -> np.ndarray:
        v = getattr(self, "right_world", None)
        if v is not None:
            return np.asarray(v, dtype=float)
        if getattr(self, "heading", None) is not None:
            # Right-hand-rule: cross(world_up, forward). ``heading.right``
            # is OpenCV screen-right (opposite sign).
            f = self.front_vec
            world_up = np.array([0.0, 1.0, 0.0])
            r = np.cross(world_up, f)
            n = float(np.linalg.norm(r))
            if n < 1e-6:
                return np.array([1.0, 0.0, 0.0])
            return r / n
        raise AttributeError(f"{type(self).__name__}: no right-axis storage")

    @property
    def up_vec(self) -> np.ndarray:
        v = getattr(self, "up_world", None)
        if v is not None:
            return np.asarray(v, dtype=float)
        if getattr(self, "heading", None) is not None:
            # cross(front, right) — RH up axis consistent with right_vec.
            f = self.front_vec
            r = self.right_vec
            u = np.cross(f, r)
            n = float(np.linalg.norm(u))
            if n < 1e-6:
                return np.array([0.0, 1.0, 0.0])
            return u / n
        raise AttributeError(f"{type(self).__name__}: no up-axis storage")

    @property
    def orientation_front(self) -> np.ndarray:
        return self.front_vec

    @property
    def orientation_right(self) -> np.ndarray:
        return self.right_vec

    @property
    def orientation_up(self) -> np.ndarray:
        return self.up_vec

    # ------------------------------------------------------------------
    # Predicate readers
    # ------------------------------------------------------------------

    def _read_predicate(self, name: str) -> np.ndarray:
        scene = getattr(self, "_scene", None)
        idx = getattr(self, "anchor_index", None)
        if scene is not None and idx is not None:
            preds = getattr(scene, "_anchor_predicates", None)
            if preds is not None:
                mat = preds.get(name)
                if mat is not None and 0 <= idx < mat.shape[1]:
                    return mat[:, idx]
        return self._compute_lazy(name)

    def _compute_lazy(self, name: str) -> np.ndarray:
        """Compute predicate column for this anchor against scene entities."""
        scene = getattr(self, "_scene", None)
        if scene is None:
            raise RuntimeError(
                f"Anchor.{name}: this anchor has no scene back-reference; "
                "free-floating predicate access requires a scene."
            )
        N = len(scene.objects) + len(scene.cameras)
        out = np.full(N, np.nan, dtype=float)
        f = np.asarray(self.orientation_front, dtype=float)
        r = np.asarray(self.orientation_right, dtype=float)
        nf = float(np.linalg.norm(f))
        nr = float(np.linalg.norm(r))
        if nf < 1e-6 or nr < 1e-6:
            return out  # unusable orientation → NaN array
        f = f / nf
        r = r / nr
        anchor_pos = np.asarray(self.position, dtype=float)

        idx = 0
        for obj in scene.objects:
            out[idx] = _score_single(anchor_pos, f, r, np.asarray(obj.position, dtype=float), name)
            idx += 1
        for cam in scene.cameras:
            out[idx] = _score_single(anchor_pos, f, r, np.asarray(cam.position, dtype=float), name)
            idx += 1
        return out

    @property
    def left(self):        return self._read_predicate("left")

    @property
    def right(self):       return self._read_predicate("right")

    @property
    def front(self):       return self._read_predicate("front")

    @property
    def back(self):        return self._read_predicate("back")

    @property
    def front_left(self):  return self._read_predicate("front_left")

    @property
    def front_right(self): return self._read_predicate("front_right")

    @property
    def back_left(self):   return self._read_predicate("back_left")

    @property
    def back_right(self):  return self._read_predicate("back_right")

    @property
    def above(self):       return self._read_predicate("above")

    @property
    def below(self):       return self._read_predicate("below")

    # ------------------------------------------------------------------
    # Immutable transforms — return new free-floating Anchor instances.
    # Both flow through ``_sync_pose`` so every stored representation
    # stays consistent. Anchor doesn't know which storage fields the
    # concrete subclass uses, so ``_sync_pose`` duck-types on attribute
    # presence:
    #
    #   - Camera       : ``position_world`` + ``heading`` + ``extrinsics``
    #   - MergedObject : ``center_world`` + ``front_world`` + ``right_world``
    #                    + ``up_world`` + ``rotation_world`` +
    #                    ``corners_world`` + ``euler_world_deg``
    # ------------------------------------------------------------------

    def translate(self, delta_xyz) -> "Anchor":
        """Return a new anchor with ``position`` shifted by ``delta_xyz``.
        Orientation unchanged; result is free-floating (anchor_index=None).
        """
        delta = np.asarray(delta_xyz, dtype=float).ravel()[:3]
        new = self._shallow_copy_for_transform()
        new._sync_pose(
            np.asarray(self.position, dtype=float) + delta,
            self.orientation_front,
            self.orientation_right,
            self.orientation_up,
        )
        new.anchor_index = None
        return new

    def rotate(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        *,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> "Anchor":
        """Return a new anchor with rotated orientation; position unchanged.

        Right-hand-rule yaw: ``yaw_deg > 0`` rotates body axes clockwise
        from above (front rotates toward right). Accepts both ``yaw_deg=``
        (canonical) and ``yaw=`` (likewise ``pitch_deg=`` / ``pitch=``).
        """
        if yaw is not None:
            yaw_deg = yaw
        if pitch is not None:
            pitch_deg = pitch

        ya = math.radians(float(yaw_deg))
        cy, sy = math.cos(ya), math.sin(ya)
        f = np.asarray(self.orientation_front, dtype=float)
        r = np.asarray(self.orientation_right, dtype=float)
        u = np.asarray(self.orientation_up, dtype=float)
        new_front = f * cy + r * sy
        new_right = -f * sy + r * cy
        new_up = u.copy()

        if pitch_deg != 0.0:
            pa = math.radians(float(pitch_deg))
            cp, sp = math.cos(pa), math.sin(pa)
            f2 = new_front * cp + new_up * sp
            u2 = -new_front * sp + new_up * cp
            new_front = f2
            new_up = u2

        new = self._shallow_copy_for_transform()
        new._sync_pose(self.position, new_front, new_right, new_up)
        new.anchor_index = None
        return new

    def _shallow_copy_for_transform(self) -> "Anchor":
        """Shallow copy that preserves the ``_scene`` back-reference."""
        new = copy.copy(self)
        new._scene = getattr(self, "_scene", None)
        return new

    def _sync_pose(self, position, front, right, up) -> None:
        """Atomically write every stored representation of the new pose."""
        p = np.asarray(position, dtype=float).ravel()[:3]
        f = np.asarray(front, dtype=float).ravel()[:3]
        r = np.asarray(right, dtype=float).ravel()[:3]
        u = np.asarray(up, dtype=float).ravel()[:3]

        # Normalize axes to keep extrinsics/rotation_world valid.
        nf = float(np.linalg.norm(f))
        nr = float(np.linalg.norm(r))
        nu = float(np.linalg.norm(u))
        if nf >= 1e-12:
            f = f / nf
        if nr >= 1e-12:
            r = r / nr
        if nu >= 1e-12:
            u = u / nu

        # ---- position storage (subclass-specific field name) ----
        if hasattr(self, "position_world"):
            self.position_world = p.copy()
        if hasattr(self, "center_world"):
            self.center_world = p.copy()

        # ---- orientation storage (subclass-specific field names) ----
        if hasattr(self, "front_world"):
            self.front_world = f.copy()
        if hasattr(self, "right_world"):
            self.right_world = r.copy()
        if hasattr(self, "up_world"):
            self.up_world = u.copy()

        # ---- rotation_world (3x3 matrix, columns [right, up, front]) ----
        if hasattr(self, "rotation_world"):
            self.rotation_world = np.column_stack([r, u, f])

        # ---- CameraHeading (camera-specific; derived from forward) ----
        if hasattr(self, "heading") and getattr(self, "heading", None) is not None:
            # Late import — types.py imports Anchor, so avoid the cycle.
            from .types import CameraHeading
            self.heading = CameraHeading(f)

        # ---- extrinsics (camera-native 4x4; rebuilt from pose) ----
        if hasattr(self, "extrinsics") and isinstance(
            getattr(self, "extrinsics", None), np.ndarray
        ):
            # Canonical camera convention (OpenCV-style, as produced by
            # ``adapters.canonicalize_y_up`` and read by projection /
            # ``_camera_frame_axes`` / ``_is_y_down_extrinsics``): rows of
            # R_w2c are (image-right, image-DOWN, forward), with image-right
            # = ``right_vec`` and image-down = -up. Translation: t = -R_w2c @ p.
            R_w2c = np.stack([r, -u, f], axis=0)
            ext = np.eye(4, dtype=float)
            ext[:3, :3] = R_w2c
            ext[:3, 3] = -R_w2c @ p
            self.extrinsics = ext

        # ---- corners_world (OBB corners; rebuilt from dims + new pose) ----
        if (
            hasattr(self, "corners_world")
            and isinstance(getattr(self, "corners_world", None), np.ndarray)
            and self.corners_world.size > 0
            and hasattr(self, "dims")
        ):
            dims = np.asarray(self.dims, dtype=float).ravel()
            if dims.size >= 3:
                w, h, d = float(dims[0]), float(dims[1]), float(dims[2])
                # 8 corners in object-local frame, then transform to world:
                # corner_world = center + (right * x + up * y + front * z)
                # for (x, y, z) in (±w/2, ±h/2, ±d/2).
                signs = np.array([
                    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
                    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
                ], dtype=float)
                local = signs * np.array([w / 2.0, h / 2.0, d / 2.0])
                # World axes columns: [right_vec, up_vec, front_vec]
                axes = np.column_stack([r, u, f])
                self.corners_world = p[None, :] + local @ axes.T

        # ---- euler_world_deg (recompute from rotation_world) ----
        if hasattr(self, "euler_world_deg") and isinstance(
            getattr(self, "euler_world_deg", None), np.ndarray
        ):
            # rotation_world has just been set above; use it.
            try:
                from scipy.spatial.transform import Rotation as _R
                euler = _R.from_matrix(np.column_stack([r, u, f])).as_euler(
                    "yxz", degrees=True
                )
                # rotation_world columns are [right, up, front] = body axes.
                # Same (azimuth, elevation, roll) layout as the loader's.
                self.euler_world_deg = np.array(euler, dtype=float)
            except Exception:
                # Degenerate rotation matrix — leave euler unchanged.
                log.debug("suppressed: degenerate rotation matrix; euler unchanged", exc_info=True)
                pass
