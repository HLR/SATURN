"""Scene cardinal / direction / rotation queries — mixin for Scene.

Imports: may import saturn.soft_logic / saturn.scene.types / saturn.scene.direction_utils; must not import saturn.perception, saturn.vlm, saturn.serving."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:  # annotations only
    from saturn.scene.types import DirectionValue

import numpy as np

from saturn.soft_logic.tensor import ProbabilisticTensor
from .direction_utils import (
    as_vector3,
    canonical_direction,
    classify_direction,
    resolve_north,
)
from saturn.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------


def _relative_direction(diff, src_fwd, src_right) -> "DirectionValue":
    """Yaw and elevation of the displacement ``diff`` in the source's frame."""
    from .types import DirectionValue

    if src_fwd is None or src_right is None:
        raise ValueError(
            "Relative direction requires a source with orientation "
            "(camera or object).  For bare 3D points, provide a "
            "north_vector or anchor pair to get cardinal directions, "
            "or use observer= or facing= to define the reference frame."
        )

    r = np.dot(diff, src_right)
    u_axis = np.cross(src_fwd, src_right)  # up = front × right (right-handed)
    u = np.dot(diff, u_axis)
    f = np.dot(diff, src_fwd)

    yaw_rad = math.atan2(r, f)
    yaw_deg = math.degrees(yaw_rad) % 360.0

    horiz = math.sqrt(r * r + f * f)
    el_rad = math.atan2(u, horiz)
    el_deg = math.degrees(el_rad)

    return DirectionValue(yaw_deg, el_deg)


def _matching_facing_override(overrides, label: str) -> Optional[Dict[str, Any]]:
    """First facing override whose phrase and *label* contain one another.

    A leading "the" is removed from the phrase first, so "the door" matches the
    label "door" and "the entrance" matches "entrance_door". An empty label, or
    a phrase that is empty without its article, matches nothing.
    """
    label = (label or "").strip()
    if not label:
        return None
    for ov in overrides:
        phrase = (ov.get("phrase") or "").lower().strip()
        phrase_words = re.sub(r"^the\b\s*", "", phrase)
        if not phrase_words:
            continue
        if phrase_words in label or label in phrase_words:
            return ov
    return None


# Option labels (canonical spelling) that need 8-way direction labels.
_COMPOUND_DIRECTIONS = frozenset({
    "front-right",
    "front-left",
    "back-right",
    "back-left",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
})

# Relative labels of the yaw buckets, clockwise from straight ahead.
_YAW_BUCKET_LABELS = {
    8: (
        "front",
        "front right",
        "right",
        "back right",
        "back",
        "back left",
        "left",
        "front left",
    ),
    4: ("front", "right", "back", "left"),
}


def _yaw_bucket_label(yaw: float, freedom: int) -> str:
    """Relative label of the quadrant (freedom 4) or octant (8) holding *yaw*."""
    step = 360.0 / freedom
    half = step / 2.0
    bucket_labels = _YAW_BUCKET_LABELS[8 if freedom == 8 else 4]
    idx = int((yaw + half) / step) % len(bucket_labels)
    return bucket_labels[idx]


# ---------------------------------------------------------------------------
# Object-rotation helpers
# ---------------------------------------------------------------------------


def _rotation_sense(label: str) -> Optional[str]:
    """'clockwise' / 'counterclockwise' named by a rotation option, or None.

    "counter-clockwise" / "anticlockwise" count as counterclockwise (plain
    substring matching finds "clockwise" inside them); "right" / "left" are
    top-down clockwise / counterclockwise.
    """
    s = str(label).lower()
    squashed = re.sub(r"[\s_\-]+", "", s)
    if "counterclockwise" in squashed or "anticlockwise" in squashed:
        return "counterclockwise"
    if "clockwise" in squashed:
        return "clockwise"
    words = set(re.split(r"[^a-z]+", s))
    if "right" in words:
        return "clockwise"
    if "left" in words:
        return "counterclockwise"
    return None


def _match_rotation_option(direction: str, options: Dict[str, str], best_option) -> str:
    """Letter of the option whose rotation sense is *direction*."""
    for k, v in options.items():
        if _rotation_sense(v) == direction:
            return k
    return best_option(direction, options)


def _horizontal_yaw_terms(f0, f1) -> Optional[Tuple[float, float]]:
    """``(cross_y, dot)`` of two fronts projected onto the XZ plane.

    Both projections are normalised first, ``cross_y = f0_x * f1_z - f0_z * f1_x``
    and ``dot`` is clipped to [-1, 1]. None when either projection is
    shorter than 1e-6.
    """
    f0_h = np.array([f0[0], 0.0, f0[2]], dtype=float)
    f1_h = np.array([f1[0], 0.0, f1[2]], dtype=float)
    n0 = float(np.linalg.norm(f0_h))
    n1 = float(np.linalg.norm(f1_h))
    if n0 < 1e-6 or n1 < 1e-6:
        return None
    f0_h /= n0
    f1_h /= n1
    cross_y = float(f0_h[0] * f1_h[2] - f0_h[2] * f1_h[0])
    dot = float(np.clip(np.dot(f0_h, f1_h), -1.0, 1.0))
    return cross_y, dot


def _no_rotation(f0, f1) -> Dict[str, Any]:
    """Rotation result for fronts with no horizontal part."""
    return {
        "signed_angle_deg": 0.0,
        "top_down_direction": "none",
        "confidence": 0.0,
        "from_front": f0,
        "to_front": f1,
    }


def _rotation_result(obj, from_cam, to_cam, f0, f1, signed_angle_deg: float) -> Dict[str, Any]:
    """Rotation result dict: the angle, its top-down sense and the confidence.

    Angles under 5 degrees count as no rotation; the confidence is the
    lower of the two views' orientation confidences.
    """
    if abs(signed_angle_deg) < 5.0:
        top_down = "none"
    elif signed_angle_deg > 0:
        top_down = "clockwise"
    else:
        top_down = "counterclockwise"

    pvc = obj.per_view_orientation_confidence
    conf = min(
        float(pvc.get(from_cam, 0.0)),
        float(pvc.get(to_cam, 0.0)),
    )
    return {
        "signed_angle_deg": signed_angle_deg,
        "top_down_direction": top_down,
        "confidence": conf,
        "from_front": f0,
        "to_front": f1,
    }


class _CardinalMixin:
    def direction(
        self,
        source,
        target,
        *,
        # Cardinal frame options (pick at most one group)
        anchor: Optional[Tuple] = None,
        anchor_cardinal: Optional[str] = None,
        north_vector: Optional[np.ndarray] = None,
        # Landmark-based cardinal frame
        north_landmark=None,
        landmark_cardinal: Optional[str] = None,
        landmark_heading: Optional[str] = None,
        # Relative-mode frame overrides
        observer: Optional[int] = None,
        facing=None,
        freedom: int = 8,
        use_scene_cardinal: bool = True,
    ) -> Union["DirectionValue", str]:
        """Unified direction query: direction FROM source TO target.

        Parameters
        ----------
        source, target : entity specifiers
            Each can be:
              - a ``Camera`` or ``MergedObject`` instance
              - ``("camera", cam_id)`` or ``("cam", cam_id)``
              - ``("object", obj_id)`` or ``("obj", obj_id)``
              - ``int`` — object index
              - ``np.ndarray`` or list — 3D world point

        anchor : tuple, optional
            ``(anchor_entity, reference_entity)`` pair that defines a
            cardinal frame.  The vector reference → anchor is declared
            to correspond to ``anchor_cardinal``.
            Each element follows the same entity conventions as source/target.

        anchor_cardinal : str, optional
            Cardinal label of the anchor relative to the reference
            (e.g. "northeast", "north").  Required when ``anchor`` is given.

        north_vector : np.ndarray, optional
            A 3D vector in world coordinates pointing "north".
            Alternative to anchor pair for establishing a cardinal frame.

        north_landmark : entity specifier, optional
            A landmark entity used to derive the north vector.  Must be
            combined with either ``landmark_cardinal=`` or
            ``landmark_heading=``.

        landmark_cardinal : str, optional
            The cardinal label of the landmark's **position** relative to
            the room center (e.g. "southeast").  The vector from room
            center to landmark IS that direction.

        landmark_heading : str, optional
            The cardinal direction the landmark entity is **facing**.
            Use when the question states "camera N is facing east".
            Mutually exclusive with ``landmark_cardinal``.

        observer : int, optional
            Camera index whose frame defines left/right/front/behind for the
            direction query.  Use this for object-to-object direction when
            "left/right" should be interpreted from a camera's viewpoint.
            Mutually exclusive with ``facing``.

        facing : entity specifier, optional
            Entity that the source is "looking at" / facing toward.  Defines
            the forward direction as ``vector(source, facing)``.  Use this
            for "sitting on chair facing TV, where is X?" questions.
            Accepts same entity forms as source/target.
            Mutually exclusive with ``observer``.

        freedom : int
            4 or 8 — used for label discretization (both cardinal and
            relative modes).

        use_scene_cardinal : bool
            When False, the scene's persistent north (``set_cardinal_vector``)
            is ignored and the call stays in relative mode unless it gives
            its own cardinal frame.

        Returns
        -------
        DirectionValue
            When no cardinal frame is specified.  Has ``.yaw_degree()``,
            ``.label(freedom=4|8)``, ``.elevation_degree()``.
        str
            When a cardinal frame is provided (anchor pair, north_vector,
            or landmark).  Returns a cardinal label like "north",
            "southeast", etc.
        """
        if observer is not None and facing is not None:
            raise ValueError("'observer' and 'facing' are mutually exclusive.")
        has_anchor_pair = anchor is not None and anchor_cardinal is not None
        north_vector = self._direction_north_vector(
            north_vector=north_vector,
            north_landmark=north_landmark,
            landmark_cardinal=landmark_cardinal,
            landmark_heading=landmark_heading,
            allow_scene_north=(
                use_scene_cardinal
                and not has_anchor_pair
                and observer is None
                and facing is None
            ),
        )

        src_pos, src_fwd, src_right = self._resolve_entity(source, label="source")
        tgt_pos, _, _ = self._resolve_entity(target, label="target")
        diff = tgt_pos - src_pos

        if has_anchor_pair or north_vector is not None:
            north_vec = self._resolve_cardinal_north(
                anchor=anchor,
                anchor_cardinal=anchor_cardinal,
                north_vector=north_vector,
                fallback_forward=src_fwd,
            )
            return self._cardinal_label(diff, north_vec, freedom)

        if observer is not None:
            src_right, _, src_fwd, _ = self._camera_frame_axes(self.cameras[observer])
        elif facing is not None:
            src_fwd, src_right = self._facing_axes(src_pos, src_fwd, src_right, facing)
        return _relative_direction(diff, src_fwd, src_right)

    def _direction_north_vector(
        self,
        *,
        north_vector,
        north_landmark,
        landmark_cardinal,
        landmark_heading,
        allow_scene_north: bool,
    ) -> Optional[np.ndarray]:
        """North vector for ``direction``, or None for relative mode.

        An explicit ``north_vector`` wins; otherwise the landmark defines
        north. The scene's persistent north applies only when no landmark is
        given and ``allow_scene_north`` is set (the call neither defines its
        own frame nor asks for relative reasoning).
        """
        if north_landmark is not None and north_vector is None:
            resolved = self._resolve_landmark_north(
                north_landmark=north_landmark,
                landmark_cardinal=landmark_cardinal,
                landmark_heading=landmark_heading,
            )
            if resolved is not None:
                north_vector = resolved
        if (
            allow_scene_north
            and north_vector is None
            and north_landmark is None
            and self._scene_north_vector is not None
        ):
            north_vector = self._scene_north_vector.copy()
        return north_vector

    def _facing_axes(self, src_pos, src_fwd, src_right, facing):
        """(forward, right) of a source that looks toward the ``facing`` entity.

        Forward is the horizontal direction from the source to ``facing``.
        When the two share a position, the source's own axes are kept.
        """
        facing_pos, _, _ = self._resolve_entity(facing, label="facing")
        fwd = facing_pos - src_pos
        fwd[1] = 0.0  # project to ground plane
        n = np.linalg.norm(fwd)
        if n < 1e-9:
            if src_fwd is None:
                raise ValueError(
                    "Cannot determine facing direction: source and facing "
                    "entity are at the same position, and source has no "
                    "intrinsic orientation."
                )
            return src_fwd, src_right
        src_fwd = fwd / n
        # right = up × forward, as Camera/MergedObject.orientation
        # (forward × up is the LEFT side in the canonical world)
        world_up = np.array([0.0, 1.0, 0.0])
        src_right = np.cross(world_up, src_fwd)
        rn = np.linalg.norm(src_right)
        if rn < 1e-9:
            src_right = np.array([1.0, 0.0, 0.0])
        else:
            src_right = src_right / rn
        return src_fwd, src_right

    def angle(self, entity1, entity2, *, anchor=None) -> float:
        """Signed top-down yaw between two entities' forward axes, in degrees.

        Returns the angle entity1 needs to turn to face the same direction as
        entity2, signed so that positive = clockwise from above ("turn right"
        in human terms) and negative = counterclockwise ("turn left").

        This is the scalar readout of the orientation primitive
        q_i^a = R_a^T @ R_i @ e_front: without ``anchor`` the comparison is in
        world frame; with ``anchor`` the forwards are first re-expressed in the
        anchor's local FoR.

        Examples
        --------
        >>> # How much did the camera turn from image 1 to image 2?
        >>> scene.angle(scene.cameras[0], scene.cameras[1])
        +88.5   # ~90° right turn (clockwise from above)

        >>> # Is chair facing the table?  (small |angle| → yes)
        >>> scene.angle(scene.objects[chair_idx], scene.objects[table_idx])
        12.3

        >>> # Parallel predicate: both directions count.
        >>> a = abs(scene.angle(e1, e2))
        >>> parallel = min(a, 180 - a) < 15

        Parameters
        ----------
        entity1, entity2 : accepted forms — ``scene.objects[i]``,
            ``scene.cameras[k]``, an int (treated as object index), or a
            ``("camera", k)`` / ``("object", i)`` tuple. Both must have an
            orientation (raw 3D points are rejected).
        anchor : optional. An anchor / frame whose ``.orientation`` (3x3)
            re-expresses both forwards in its local frame before measuring.
            Default: world frame.

        Returns
        -------
        float — signed degrees in (-180, 180]. Positive = clockwise from above
        (entity1 turns right to face same way as entity2).
        """
        import math as _math
        _, f1, _ = self._resolve_entity(entity1, label="entity1")
        _, f2, _ = self._resolve_entity(entity2, label="entity2")
        if f1 is None or f2 is None:
            raise ValueError(
                "scene.angle: both entities must have an orientation "
                "(bare 3D points have no forward axis)."
            )
        f1 = np.asarray(f1, dtype=float).ravel()
        f2 = np.asarray(f2, dtype=float).ravel()

        # Optional anchor frame: re-express forwards in anchor's local frame,
        # q_i^a = R_a^T @ R_i @ e_front.
        if anchor is not None:
            R_a = np.asarray(getattr(anchor, "orientation", anchor), dtype=float)
            if R_a.shape != (3, 3):
                raise ValueError(
                    f"scene.angle: anchor.orientation must be 3x3, got {R_a.shape}"
                )
            f1 = R_a.T @ f1
            f2 = R_a.T @ f2

        # Top-down signed yaw around the up-axis (+Y in world frame; +Y in
        # anchor frame after rotation). Project onto the horizontal plane,
        # compute signed angle via atan2. Sign flipped so positive = CW from
        # above (matches the "turn right is positive" human convention).
        cross_up = f1[2] * f2[0] - f1[0] * f2[2]
        dot_xz = f1[0] * f2[0] + f1[2] * f2[2]
        return _math.degrees(_math.atan2(cross_up, dot_xz))


    def angular_predicate(
        self,
        target_deg: float,
        *,
        tol: float = 15.0,
        anchor=None,
        temp: float = 5.0,
    ) -> ProbabilisticTensor:
        """Soft 2D (K+C, K+C) predicate matrix for the fixed-angle relation.

        ``S[i, j] in [0, 1]`` = soft score of "the signed top-down yaw
        ``scene.angle(entity_i, entity_j)`` lies within ``tol`` degrees of
        ``target_deg``". This is the fixed-angle predicate ``S^a_{θ*}[i, j]``
        for any target angle, with a smooth sigmoid edge.

        Use this when the question asks about a SPECIFIC orientation relation
        across all entity pairs at once (good with iota/argmax). For one-pair
        queries, prefer ``scene.angle(...)`` and compare directly.

        Common targets
        --------------
        - ``target_deg=0``   — "j faces the same direction as i"
        - ``target_deg=180`` — "j faces the opposite direction"
        - ``target_deg=+90`` — "j is rotated 90° clockwise from i"
        - ``target_deg=-90`` (or ``270``) — "j is rotated 90° counter-clockwise"
        - parallel := angular_predicate(0) | angular_predicate(180)
        - perpendicular := angular_predicate(90) | angular_predicate(-90)

        Parameters
        ----------
        target_deg : float
            Target signed angle in degrees. Any real value; wraps mod 360.
        tol : float, default 15
            Half-width of the high-score band, in degrees. Within ±tol of
            target gets score ≈ 1; beyond ±tol the score falls off sigmoidally.
        anchor : optional
            If provided, both entity forwards are first re-expressed in this
            anchor's local frame (q_i^a = R_a^T @ R_i @ e_front).
            Default: world frame.
        temp : float, default 5
            Sigmoid temperature for the soft edge. Lower = sharper boundary.

        Returns
        -------
        ProbabilisticTensor of shape (K+C, K+C). Diagonal is zeroed. Indexing
        follows the same convention as ``scene.parallel`` / ``scene.facing``:
        first K rows/cols are objects, last C are cameras.
        """
        n_obj = len(self.objects)
        n_cam = len(self.cameras)
        n = n_obj + n_cam
        if n == 0:
            return self._wrap_tensor(np.zeros((0, 0), dtype=np.float32), ndim=2)

        # Gather forward axes: objects then cameras.
        fronts = np.zeros((n, 3), dtype=float)
        for i, obj in enumerate(self.objects):
            fronts[i] = np.asarray(obj.orientation[:, 2], dtype=float)
        for k, cam in enumerate(self.cameras):
            fronts[n_obj + k] = np.asarray(cam.orientation[:, 2], dtype=float)

        # Optional anchor: re-express forwards in anchor's local frame.
        # (R_a^T @ f).T = f.T @ R_a, so for row-vector storage: F_local = F @ R_a.
        if anchor is not None:
            R_a = np.asarray(getattr(anchor, "orientation", anchor), dtype=float)
            if R_a.shape != (3, 3):
                raise ValueError(
                    f"angular_predicate: anchor.orientation must be 3x3, "
                    f"got {R_a.shape}"
                )
            fronts = fronts @ R_a

        # Pairwise signed top-down yaw: angle[i, j] = how much i must turn CW
        # to face the same direction as j.
        fx = fronts[:, 0]
        fz = fronts[:, 2]
        cross_up = fz[:, None] * fx[None, :] - fx[:, None] * fz[None, :]
        dot_xz = fx[:, None] * fx[None, :] + fz[:, None] * fz[None, :]
        angles_deg = np.degrees(np.arctan2(cross_up, dot_xz))  # (n, n)

        # Circular distance to target, wrapped to (-180, 180].
        diff = (angles_deg - float(target_deg) + 180.0) % 360.0 - 180.0
        abs_diff = np.abs(diff)

        # Soft score: ≈1 inside the ±tol band, falls off sigmoidally outside.
        score = 1.0 / (1.0 + np.exp((abs_diff - float(tol)) / max(float(temp), 1e-6)))
        np.fill_diagonal(score, 0.0)
        return self._wrap_tensor(score.astype(np.float32), ndim=2)


    def _resolve_cardinal_north(
        self,
        *,
        anchor: Optional[Tuple],
        anchor_cardinal: Optional[str],
        north_vector: Optional[np.ndarray],
        fallback_forward: Optional[np.ndarray],
    ) -> np.ndarray:
        """Determine the world-space north vector from user-supplied cardinal frame.

        Delegates to ``direction_utils.resolve_north`` after resolving entity
        positions from the anchor tuple.
        """
        anchor_pos = None
        reference_pos = None
        if anchor is not None and anchor_cardinal is not None:
            anchor_pos, _, _ = self._resolve_entity(anchor[0], label="anchor")
            reference_pos, _, _ = self._resolve_entity(anchor[1], label="reference")

        return resolve_north(
            anchor_pos=anchor_pos,
            reference_pos=reference_pos,
            anchor_cardinal=anchor_cardinal,
            north_vector=north_vector,
            fallback_forward=fallback_forward,
        )


    def _cardinal_label(
        self, diff: np.ndarray, north_vec: np.ndarray, freedom: int
    ) -> str:
        """Classify a horizontal displacement vector into a cardinal label.

        Delegates to ``direction_utils.classify_direction``.
        """
        return classify_direction(diff, north_vec, freedom, labels="cardinal")


    def _resolve_landmark_north(
        self,
        *,
        north_landmark=None,
        landmark_cardinal: Optional[str] = None,
        landmark_heading: Optional[str] = None,
        relative_to=None,
    ) -> Optional[np.ndarray]:
        """Derive a north vector from a landmark entity.

        Parameters
        ----------
        north_landmark : entity specifier, optional
            A landmark entity known to be at a specific position in the scene
            or facing a specific direction.
        landmark_cardinal : str, optional
            The cardinal label of the landmark's **position** relative to the
            room center (e.g. "north", "southeast").  Use when the question
            states something like "the cushion is in the southeast corner".
            When ``relative_to`` is given, the position is relative to that
            entity instead of the room center.
        landmark_heading : str, optional
            The cardinal direction the landmark entity is **facing** / heading.
            Use when the question states "camera 3 is facing east" — the
            entity's forward vector IS that direction.
        relative_to : entity specifier, optional
            When using ``landmark_cardinal`` (position mode), the reference
            point from which the landmark is at the given cardinal position.
            Defaults to ``scene.room_center()`` when omitted.

        Returns
        -------
        np.ndarray or None
            A unit north vector in world space, or None if no landmark
            information was provided / resolution failed.
        """
        if north_landmark is None:
            return None
        if landmark_cardinal is None and landmark_heading is None:
            raise ValueError(
                "north_landmark requires either landmark_cardinal= or "
                "landmark_heading= to be specified."
            )
        if landmark_cardinal is not None and landmark_heading is not None:
            raise ValueError(
                "landmark_cardinal= and landmark_heading= are mutually "
                "exclusive.  Provide only one."
            )

        if landmark_heading is not None:
            # The landmark's forward vector points toward landmark_heading.
            _, lm_fwd, _ = self._resolve_entity(north_landmark, label="landmark")
            if lm_fwd is None:
                return None
            lm_fwd = self._landmark_front_override(north_landmark, lm_fwd)
            return self._north_from_known_direction(lm_fwd, landmark_heading)

        # The landmark lies toward landmark_cardinal from the reference point.
        lm_pos, _, _ = self._resolve_entity(north_landmark, label="landmark")
        if relative_to is not None:
            rc, _, _ = self._resolve_entity(relative_to, label="relative_to")
        else:
            rc = self.room_center()
        return self._north_from_known_direction(lm_pos - rc, landmark_cardinal)

    def _landmark_front_override(self, north_landmark, lm_fwd):
        """Landmark front from the clarifier's facing override, else ``lm_fwd``.

        An override (e.g. "front visible in Image 3") names the camera that
        sees the landmark's front; the front then points horizontally from
        the object toward that camera, replacing OrientAnything's estimate.
        """
        lm_obj = self._resolve_entity_object(north_landmark)
        if lm_obj is None or not self._facing_overrides:
            return lm_fwd
        lm_label = (getattr(lm_obj, "label", "") or "").lower()
        override = _matching_facing_override(self._facing_overrides, lm_label)
        if override is None:
            return lm_fwd
        front_cam_id = override.get("front_cam_id")
        if front_cam_id is None or not 0 <= front_cam_id < len(self.cameras):
            return lm_fwd
        cam_pos = np.asarray(self.cameras[front_cam_id].position_world, dtype=float)
        obj_pos = np.asarray(lm_obj.center_world, dtype=float)
        fwd = cam_pos - obj_pos
        fwd[1] = 0.0
        fn = np.linalg.norm(fwd)
        if fn > 1e-9:
            log.info(
                f"Facing override applied for '{lm_label}': "
                f"front toward camera {front_cam_id}"
            )
            return fwd / fn
        return lm_fwd

    def _north_from_known_direction(self, direction, cardinal: str) -> Optional[np.ndarray]:
        """North vector given that ``direction`` points toward ``cardinal``.

        Only the horizontal part of ``direction`` counts; None when it has
        none.
        """
        horizontal = direction.copy()
        horizontal[1] = 0.0
        norm = np.linalg.norm(horizontal)
        if norm > 1e-9:
            horizontal = horizontal / norm
            known_angle = self._CARDINAL_TO_ANGLE.get(cardinal.lower().strip())
            if known_angle is None:
                raise ValueError(f"Unknown cardinal label: '{cardinal}'")
            world_angle = math.degrees(math.atan2(horizontal[0], horizontal[2]))
            north_angle = world_angle - known_angle
            nr = math.radians(north_angle)
            return np.array([math.sin(nr), 0.0, math.cos(nr)], dtype=float)
        return None


    def match_direction(
        self,
        source,
        target,
        options: Dict[str, str],
        *,
        freedom: Optional[int] = None,
        observer: Optional[int] = None,
        facing=None,
        # Cardinal frame options (auto-detected from option labels)
        anchor=None,
        anchor_cardinal: Optional[str] = None,
        north_vector: Optional[np.ndarray] = None,
        north_landmark=None,
        landmark_cardinal: Optional[str] = None,
        landmark_heading: Optional[str] = None,
    ) -> str:
        """Compute direction from *source* to *target* and match to MCQ options.

        This is the **single unified direction-matching method**.  It handles
        both relative directions (front/back/left/right) and cardinal
        directions (north/south/east/west) automatically based on the option
        labels.

        **Auto-detect logic**: if option labels contain cardinal words (north,
        south, east, west, NE, SW, ...) the method uses cardinal mode.  If
        option labels contain relative words (front, back, left, right, ...)
        the method uses relative mode.  You can also force cardinal mode by
        providing any of the cardinal frame parameters.  Cardinal mode does
        not use ``observer`` or ``facing``: the compass direction from source
        to target does not depend on which way anyone faces.

        Parameters
        ----------
        source, target : entity specifiers (same as ``scene.direction``).
        options : dict  ``{"A": "northwest", "B": "front left", ...}``
        freedom : int, optional
            Ignored; accepted so existing programs still run.  The options
            decide the number of directions: 8 when any option is a compound
            direction (e.g. "front-left", "northeast"), else 4, so the
            computed label is always one the options can name.
        observer : int, optional
            Relative options only: camera index whose frame defines
            left/right/front/behind.  Use for object-to-object direction
            from a camera's viewpoint.
        facing : entity specifier, optional
            Relative options only: entity the source is facing toward.
            Defines forward as ``vector(source, facing)``.

        anchor : tuple, optional
            ``(anchor_entity, reference_entity)`` defining a cardinal frame.
        anchor_cardinal : str, optional
            Cardinal label for the anchor (e.g. "north").
        north_vector : np.ndarray, optional
            Explicit north direction vector.
        north_landmark : entity specifier, optional
            Landmark entity for deriving north.
        landmark_cardinal : str, optional
            Cardinal label of landmark's position relative to room center.
        landmark_heading : str, optional
            Cardinal direction the landmark entity is facing.

        Returns
        -------
        str — the matching option letter.
        """
        # Canonical spelling first, so "north-east" / "south east" count as
        # compounds and "behind-right" matches the engine's "back right".
        options = {k: canonical_direction(self._normalize_label(v))
                   for k, v in options.items()}
        # Any compound option label means 8 directions, else 4.
        any_compound = any(v in _COMPOUND_DIRECTIONS for v in options.values())
        effective_freedom = 8 if any_compound else 4

        has_explicit_cardinal_frame = (
            (anchor is not None and anchor_cardinal is not None)
            or north_vector is not None
            or north_landmark is not None
        )
        if has_explicit_cardinal_frame or self._options_name_cardinals(options):
            label = self.direction(
                source,
                target,
                freedom=effective_freedom,
                anchor=anchor,
                anchor_cardinal=anchor_cardinal,
                north_vector=north_vector,
                north_landmark=north_landmark,
                landmark_cardinal=landmark_cardinal,
                landmark_heading=landmark_heading,
            )
            # direction() returns a DirectionValue when no north resolves.
            if not isinstance(label, str):
                label = label.label(effective_freedom).replace("-", " ")
            return self._best_option(label, options)

        return self._match_relative_direction(
            source, target, options, effective_freedom, observer=observer, facing=facing
        )

    def _options_name_cardinals(self, options: Dict[str, str]) -> bool:
        """True when an option label, or a word in one, is a cardinal word."""
        for v in options.values():
            norm_v = self._normalize_label(v)
            if norm_v in self._CARDINAL_WORDS:
                return True
            if any(word in self._CARDINAL_WORDS for word in norm_v.split()):
                return True
        return False

    def _match_relative_direction(
        self, source, target, options: Dict[str, str], freedom: int, *, observer, facing
    ) -> str:
        """Option letter for the relative direction from *source* to *target*.

        The direction's label is matched first; when the best option does
        not contain that label (or vice versa), the option that best matches
        the label of the yaw's quadrant / octant is chosen instead.
        """
        d = self.direction(
            source,
            target,
            freedom=freedom,
            observer=observer,
            facing=facing,
            use_scene_cardinal=False,
        )
        label = d.label(freedom).replace("-", " ")

        result = self._best_option(label, options)
        norm_label = self._normalize_label(label)
        norm_result = self._normalize_label(options[result])
        if (
            norm_label == norm_result
            or norm_label in norm_result
            or norm_result in norm_label
        ):
            return result
        return self._best_option(_yaw_bucket_label(d.yaw_degree(), freedom), options)


    def cardinalize(self, vector, known: str = None) -> np.ndarray:
        """Convert a known-direction vector into the implied world-space north.

        Given a horizontal direction ``vector`` that the caller knows points
        toward cardinal label ``known`` (e.g. ``"east"``), return the
        world-space unit vector that points toward **north** in the same
        cardinal frame.  This is the planner's primitive for *deriving* a
        north vector from any landmark direction.

        Examples
        --------
        >>> # Camera 2 is known to face east; derive scene north.
        >>> n = scene.cardinalize(scene.cameras[2].front, known="east")
        >>> scene.set_cardinal_vector(n)

        Parameters
        ----------
        vector : np.ndarray or list
            World-space direction.  Y component is dropped.
        known : str
            Cardinal label that ``vector`` points toward
            (one of ``north / east / south / west`` and 8-way variants).

        Returns
        -------
        np.ndarray
            Unit-norm world-space vector pointing toward north.
        """
        if known is None:
            raise TypeError(
                "cardinalize(vector, known=<compass word>) turns a vector KNOWN to point <known> into "
                "the north vector for scene.set_cardinal_vector(...); it does not name a vector's "
                "direction. To read which compass direction a vector points, use "
                "scene.score_cardinals(vector) (a {label: score} dict) after set_cardinal_vector; for "
                "the way an object faces, anchor.facing.<compass>(\"x1\").")
        cardinal_key = str(known).lower().strip()
        cardinal_angle = self._CARDINAL_TO_ANGLE.get(cardinal_key)
        if cardinal_angle is None:
            cardinal_angle = self._CARDINAL_TO_ANGLE.get(cardinal_key.replace("-", "").replace(" ", ""))
        if cardinal_angle is None:
            raise ValueError(f"cardinalize: unknown cardinal label '{known}'")
        v = as_vector3(vector, "cardinalize(vector)")
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            raise ValueError(
                "cardinalize: vector has zero horizontal magnitude (it points straight up or "
                "down); a compass direction needs a vector with a horizontal (x, z) part."
            )
        v /= n
        # ``v`` points along ``known``; rotate by ``-cardinal_angle`` around +Y
        # to recover the north direction.  Cardinal angles are north=0,
        # east=90 (clockwise from above viewing -Y down), so::
        #     world_angle(v) = north_angle + cardinal_angle
        #     north_angle    = world_angle(v) - cardinal_angle
        world_angle = math.degrees(math.atan2(v[0], v[2]))
        north_angle = world_angle - cardinal_angle
        north_rad = math.radians(north_angle)
        return np.array(
            [math.sin(north_rad), 0.0, math.cos(north_rad)], dtype=float
        )


    def score_cardinals(self, vector) -> dict:
        """Score similarity of ``vector`` to each of the 8 cardinal directions.

        Returns a dict mapping each hyphenated 8-way cardinal label
        (``"north"``, ``"north-east"``, ``"east"``, ``"south-east"``,
        ``"south"``, ``"south-west"``, ``"west"``, ``"north-west"``) to a
        similarity score in ``[0, 1]``.  The score is the **clamped cosine
        similarity** between ``vector`` (horizontal-projected and unit-
        normalised) and the world-space unit vector representing each
        cardinal label, derived from the scene's stored north vector
        (``set_cardinal_vector`` / ``cardinalize``).

        Use this as the primitive for cardinal-direction questions.
        Pattern::

            scores = scene.score_cardinals(target_pos - origin_pos)
            best = max(scores, key=scores.get)        # argmax label
            # Or compose with identity scores in MCQ-style code.

        Parameters
        ----------
        vector : np.ndarray or list
            World-space 3D vector.  Y component is dropped (cardinal
            scoring is horizontal-only).

        Returns
        -------
        dict[str, float]
            ``{"north": s_n, "north-east": s_ne, ..., "north-west": s_nw}``
            with each ``s`` in ``[0, 1]``.  All zeros if ``vector`` has
            negligible horizontal magnitude.

        Raises
        ------
        RuntimeError
            If the scene has no cardinal frame established.  Call
            ``scene.set_cardinal_vector(...)`` first.
        """
        if self._scene_north_vector is None:
            raise RuntimeError(
                "score_cardinals(): scene has no cardinal frame.  Call "
                "scene.set_cardinal_vector(...) first."
            )

        v = as_vector3(vector, "score_cardinals(vector)")
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return {label: 0.0 for label in self._CARDINAL_LABELS_8_HYPHENATED}
        v /= n

        # World-space cardinal unit vectors: the stored north vector rotated
        # clockwise (viewed from above, looking -Y) by k * 45 degrees for
        # k = 0..7. With +Y up, angles are measured via atan2(x, z), as in
        # ``cardinalize``.
        north = self._scene_north_vector  # already horizontal + unit
        nx, nz = float(north[0]), float(north[2])

        scores: dict = {}
        for k, label in enumerate(self._CARDINAL_LABELS_8_HYPHENATED):
            angle_rad = math.radians(k * 45.0)
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            # Rotate north clockwise from above by ``angle_rad``.
            #   x' = c*nx + s*nz
            #   z' = -s*nx + c*nz
            cx = c * nx + s * nz
            cz = -s * nx + c * nz
            cos_sim = v[0] * cx + v[2] * cz
            scores[label] = max(0.0, float(cos_sim))
        return scores


    def cardinal_vector(self, direction: str = "north") -> np.ndarray:
        """World-space unit vector pointing toward ``direction`` ("east",
        "south-west", ...) in the frame fixed by set_cardinal_vector. The
        inverse of cardinalize(): cardinalize(cardinal_vector(d), known=d) is
        north."""
        n = getattr(self, "_scene_north_vector", None)
        if n is None:
            raise ValueError("cardinal_vector(): call scene.set_cardinal_vector(north) first")
        key = str(direction).lower().strip()
        angle = self._CARDINAL_TO_ANGLE.get(key)
        if angle is None:
            angle = self._CARDINAL_TO_ANGLE.get(key.replace("-", "").replace(" ", ""))
        if angle is None:
            raise ValueError(f"cardinal_vector: unknown cardinal label {direction!r}")
        n = np.asarray(n, dtype=float)
        a = math.radians(math.degrees(math.atan2(n[0], n[2])) + angle)
        return np.array([math.sin(a), 0.0, math.cos(a)], dtype=float)

    def set_cardinal_vector(self, north_vector) -> np.ndarray:
        """Set the scene-level cardinal frame directly from a world-space north vector.

        Accepts the vector verbatim (after horizontal projection + normalization).
        Use this when the caller already knows which world-space direction is
        "north" — for example, from ``scene.cardinalize(vec, known="east")``::

            n = scene.cardinalize(camera.front, known="east")
            scene.set_cardinal_vector(n)

        Parameters
        ----------
        north_vector : np.ndarray or list
            3D world-space direction.  The Y component is dropped (cardinal
            frames are horizontal).  The vector is normalized.

        Returns
        -------
        np.ndarray
            The normalized world-space north vector that was stored.
        """
        v = as_vector3(north_vector, "set_cardinal_vector(north_vector)")
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            raise ValueError(
                "set_cardinal_vector(): vector has zero horizontal magnitude."
            )
        v /= n
        self._scene_north_vector = v
        return v.copy()


    def clear_cardinal(self):
        """Clear any persistent scene-level cardinal frame."""
        self._scene_north_vector = None


    def object_rotation(
        self,
        obj_idx: int,
        from_cam: int,
        to_cam: int,
    ) -> Dict[str, Any]:
        """Per-object rotation about the world-up axis between two frames.

        For "did the OBJECT rotate clockwise or counterclockwise" questions
        where the camera is stationary and a single object rotates in place.
        Uses Oriany per-view fronts captured during fusion.

        The angle is the signed yaw rotation (in degrees) of the object's front
        direction projected onto the horizontal plane, measured between
        ``from_cam`` and ``to_cam``. Sign convention is "top-down":

            +angle → clockwise        when viewed from above
            -angle → counterclockwise when viewed from above

        Parameters
        ----------
        obj_idx : int
            Index into ``scene.objects``.
        from_cam, to_cam : int
            Frame indices. Must both be present in
            ``scene.objects[obj_idx].per_view_fronts``.

        Returns
        -------
        dict with keys:
            ``signed_angle_deg`` : float
                Right-hand-rule yaw about world +Y, in degrees, in (-180, 180].
            ``top_down_direction`` : str
                One of ``"clockwise"``, ``"counterclockwise"``, or ``"none"``
                (when |angle| < 5°).
            ``confidence`` : float
                Min of the two per-view orientation confidences (in [0, 1]).
            ``from_front`` : np.ndarray (3,)
            ``to_front``   : np.ndarray (3,)

        Raises
        ------
        ValueError
            If per-view fronts are missing for either frame.
        """
        if obj_idx < 0 or obj_idx >= len(self.objects):
            raise ValueError(f"object_rotation: obj_idx {obj_idx} out of range")
        obj = self.objects[obj_idx]
        pvf = obj.per_view_fronts
        if from_cam not in pvf or to_cam not in pvf:
            raise ValueError(
                f"object_rotation: object {obj_idx} ({obj.label!r}) is missing "
                f"per-view front for cam {from_cam} or {to_cam}. "
                f"Available views: {sorted(pvf.keys())}"
            )

        f0 = np.asarray(pvf[from_cam], dtype=float)
        f1 = np.asarray(pvf[to_cam], dtype=float)

        # Yaw only: the fronts are compared in the horizontal (world XZ) plane.
        terms = _horizontal_yaw_terms(f0, f1)
        if terms is None:
            return _no_rotation(f0, f1)
        cross_y, dot = terms
        # Seen from above the scene (looking along -Y), cross_y < 0 is
        # clockwise; cross_y is negated so that a positive angle is clockwise.
        angle_rad = math.atan2(-cross_y, dot)
        return _rotation_result(obj, from_cam, to_cam, f0, f1, math.degrees(angle_rad))


    def match_object_rotation_direction(
        self,
        obj_idx: int,
        from_cam: int,
        to_cam: int,
        options: Dict[str, str],
    ) -> str:
        """Match per-object top-down rotation direction to MCQ options.

        For rotation-direction questions with options like
        ``{"A": "clockwise", "B": "counterclockwise"}``.

        Parameters
        ----------
        obj_idx : int
        from_cam, to_cam : int
        options : dict mapping letters to vocabulary like
            ``"clockwise" / "counterclockwise"`` or
            ``"left" / "right"`` (interpreted as
            top-down ``counterclockwise`` / ``clockwise`` respectively, which
            matches a head-rotation OWN-perspective when the person initially
            faces the camera).

        Returns
        -------
        str — matching option letter.
        """
        rot = self.object_rotation(obj_idx, from_cam, to_cam)
        direction = rot["top_down_direction"]
        if direction == "none":
            # Tiny / noisy rotation: use the sign of the raw signed angle.
            direction = (
                "clockwise" if rot["signed_angle_deg"] >= 0 else "counterclockwise"
            )
        return _match_rotation_option(direction, options, self._best_option)


    def object_rotation_camera_frame(
        self,
        obj_idx: int,
        from_cam: int,
        to_cam: int,
    ) -> Dict[str, Any]:
        """Per-object rotation as observed in each view's camera frame.

        Use this primitive **when the camera is stationary between views**,
        or more generally when the camera's relative pose between
        ``from_cam`` and ``to_cam`` is unreliable.

        Where ``object_rotation(...)`` rotates each per-view front into the
        world frame using the reconstruction backbone's ``R_c2w`` (VGGT,
        DepthAnythingV3, etc.), this method **does not**: it reads each
        view's camera-frame front (``MergedObject.per_view_fronts_camera``,
        passed straight through from Oriany), computes a yaw bearing in
        each camera independently, and returns the difference.

        On a 2-image input where only a rigid object moved, the appearance
        change is *gauge-ambiguous*: it can be explained as object motion or
        as camera motion. VGGT, trained predominantly on moving-camera
        data, tend to attribute the rotation to the camera, and the
        world-frame transform then cancels most of the object's rotation.

        Camera-frame comparison never passes through the inter-view
        rotation, so with a stationary camera the signed delta of
        camera-frame yaws equals the object's apparent rotation between
        views. If the camera also moved, no two-view method can recover the
        true object rotation without an external anchor.

        Sign convention is the same as ``object_rotation``:

            +angle → clockwise        (top-down)
            −angle → counterclockwise (top-down)

        The camera frame is OpenCV: +X right, +Y down, +Z forward.

        Parameters
        ----------
        obj_idx : int
            Index into ``scene.objects``.
        from_cam, to_cam : int
            Frame indices. Must both be present in
            ``scene.objects[obj_idx].per_view_fronts_camera``.

        Returns
        -------
        dict with the same keys as ``object_rotation``:
            ``signed_angle_deg``, ``top_down_direction``, ``confidence``,
            ``from_front`` (camera frame), ``to_front`` (camera frame).

        Raises
        ------
        ValueError
            If camera-frame fronts are missing for either frame; use
            ``object_rotation`` instead.
        """
        obj, f0, f1 = self._camera_frame_fronts(obj_idx, from_cam, to_cam)

        # Dropping cam-Y ("down") leaves the yaw bearing the camera sees
        # (x = right, z = forward).
        terms = _horizontal_yaw_terms(f0, f1)
        if terms is None:
            return _no_rotation(f0, f1)
        cross_y, dot = terms
        # Negated so that a positive angle is clockwise from above.
        angle_rad = -math.atan2(cross_y, dot)
        return _rotation_result(obj, from_cam, to_cam, f0, f1, math.degrees(angle_rad))

    def _camera_frame_fronts(self, obj_idx: int, from_cam: int, to_cam: int):
        """(object, from-front, to-front) for ``object_rotation_camera_frame``.

        The fronts are the object's camera-frame per-view fronts; a missing
        object or view raises ValueError.
        """
        if obj_idx < 0 or obj_idx >= len(self.objects):
            raise ValueError(
                f"object_rotation_camera_frame: obj_idx {obj_idx} out of range"
            )
        obj = self.objects[obj_idx]
        pvfc = obj.per_view_fronts_camera
        if from_cam not in pvfc or to_cam not in pvfc:
            raise ValueError(
                f"object_rotation_camera_frame: object {obj_idx} ({obj.label!r}) "
                f"is missing camera-frame per-view front for cam {from_cam} "
                f"or {to_cam}. Available views: {sorted(pvfc.keys())}. "
                f"Re-run scene construction with the latest fusion code, "
                f"or use object_rotation(...) (world-frame) instead."
            )
        f0 = np.asarray(pvfc[from_cam], dtype=float)
        f1 = np.asarray(pvfc[to_cam], dtype=float)
        return obj, f0, f1


    def match_object_rotation_direction_camera_frame(
        self,
        obj_idx: int,
        from_cam: int,
        to_cam: int,
        options: Dict[str, str],
    ) -> str:
        """Camera-frame analogue of ``match_object_rotation_direction``.

        Use this when the camera is stationary between views. See
        ``object_rotation_camera_frame`` for rationale.
        """
        rot = self.object_rotation_camera_frame(obj_idx, from_cam, to_cam)
        direction = rot["top_down_direction"]
        if direction == "none":
            direction = (
                "clockwise" if rot["signed_angle_deg"] >= 0 else "counterclockwise"
            )
        return _match_rotation_option(direction, options, self._best_option)
