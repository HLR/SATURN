"""
FrameNamespace — frame-scoped spatial relations.

Returned by scene.frame(...).
Wraps relation tensors as ProbabilisticTensor attributes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, TYPE_CHECKING, Optional

import numpy as np

from saturn.scene.direction_utils import (
    CARDINAL_TO_ANGLE,
    CARDINAL_LABELS_4,
    CARDINAL_LABELS_8,
    canonical_direction,
)
from saturn.soft_logic.predicate_array import PredicateArray


from .registry import _RegisteredKaryWrapper, _RegisteredPairwiseWrapper
from .scoring import _DirectionScoringMixin, _compute_pairwise_score
from .transforms import _TransformMixin


from .relations import compute_frame_obj_facing, compute_frame_relations

if TYPE_CHECKING:
    from saturn.scene.scene import Scene


class FrameNamespace(_DirectionScoringMixin, _TransformMixin):
    """Provides frame-scoped spatial relations.

    Attributes (all ProbabilisticTensor-wrapped):
        left, right, front, behind, above, below          — (N, N)
        left_normalized, right_normalized, ...             — (N, N)
        obj_facing_left, obj_facing_right, ...             — (N,)
    """

    def __init__(
        self,
        scene: "Scene",
        frame_right: np.ndarray,
        frame_up: np.ndarray,
        frame_front: np.ndarray,
        frame_origin: np.ndarray,
        *,
        hfov_deg: Optional[float] = None,
    ):
        self._scene = scene
        self._frame_right = np.asarray(frame_right, dtype=float)
        self._frame_up = np.asarray(frame_up, dtype=float)
        self._frame_front = np.asarray(frame_front, dtype=float)
        self._frame_origin = np.asarray(frame_origin, dtype=float)
        self._hfov_deg = hfov_deg  # horizontal FOV in degrees, if known

        # Lazily computed
        self._directional: Optional[dict] = None
        self._facing: Optional[dict] = None

    # -- public frame axes (read-only) -----------------------------------

    @property
    def frame_right(self) -> np.ndarray:
        """Right axis of this frame in world coordinates."""
        return self._frame_right

    @property
    def frame_up(self) -> np.ndarray:
        """Up axis of this frame in world coordinates."""
        return self._frame_up

    @property
    def frame_front(self) -> np.ndarray:
        """Front (forward) axis of this frame in world coordinates."""
        return self._frame_front

    @property
    def frame_origin(self) -> np.ndarray:
        """Origin of this frame in world coordinates."""
        return self._frame_origin

    @property
    def position(self) -> np.ndarray:
        """Origin of this frame in world coordinates.

        Alias for ``frame_origin`` so views share the API of ``Camera`` and
        ``MergedObject`` (which both expose ``.position``). Lets code like
        ``scene.frame(position=other_view.position, orientation=...)`` work
        without poking at internals.
        """
        return self._frame_origin

    @property
    def orientation(self) -> np.ndarray:
        """Frame orientation as a 3x3 matrix; columns are [right, up, front].

        Mirrors ``Camera.orientation`` and ``MergedObject.orientation`` so a
        view can be plugged back into ``scene.frame(position=..., orientation=...)``
        to rebuild a frame at a new origin while preserving the rotation::

            view2 = scene.frame(position=other_pos, orientation=view.orientation)

        Equivalent to ``np.column_stack([view.frame_right, view.frame_up,
        view.frame_front])``; ``orientation[:, 2]`` is ``frame_front``.
        """
        return np.column_stack([self._frame_right, self._frame_up, self._frame_front])

    # Axis aliases shared with Camera / MergedObject (``camera(N).front_vec``).
    @property
    def front_vec(self) -> np.ndarray:
        """Unit front axis in world coordinates (alias of ``frame_front``)."""
        return self._frame_front

    @property
    def right_vec(self) -> np.ndarray:
        """Unit right axis in world coordinates (alias of ``frame_right``)."""
        return self._frame_right

    @property
    def up_vec(self) -> np.ndarray:
        """Unit up axis in world coordinates (alias of ``frame_up``)."""
        return self._frame_up

    # -- lazy computation ------------------------------------------------

    def _ensure_directional(self):
        if self._directional is not None:
            return
        # Use the mixed entity space (objects + cameras) so that left/right/
        # front/behind tensors are (N+C, N+C) and the LLM can write queries
        # like  ``camera("x1") & view.behind("x1", "x2") & car("x2")``  to
        # find which camera is behind the car.
        positions = self._scene._entity_positions()
        # Directional predicates use delta / s_scene (0.9-quantile of
        # pairwise entity distances).  Scenes without the helper fall back
        # to raw units.
        scale_fn = getattr(self._scene, "_compute_scene_scale", None)
        scene_scale = float(scale_fn()) if callable(scale_fn) else 1.0
        self._directional = compute_frame_relations(
            positions,
            self._frame_right,
            self._frame_up,
            self._frame_front,
            scene_scale=scene_scale,
        )

    def _ensure_facing(self):
        if self._facing is not None:
            return
        fronts = self._scene._entity_front_directions()
        elevations = np.array(self._scene._entity_elevations(), dtype=float)
        # A camera's front is its full 3D view axis, so its up/down is the pitch of
        # that axis relative to THIS frame: in camera 1's frame, facing.up[camera(2)]
        # reads "camera 2 is tilted up from camera 1" (objects keep their own elevation).
        n_obj = len(self._scene.objects)
        if len(fronts) > n_obj:
            cam = np.asarray(fronts, dtype=float)[n_obj:]
            norm = np.linalg.norm(cam, axis=1)
            ok = np.isfinite(norm) & (norm > 1e-8)
            s = np.zeros(len(cam))
            s[ok] = (cam[ok] @ np.asarray(self._frame_up, dtype=float)) / norm[ok]
            elevations[n_obj:] = np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
        self._facing = compute_frame_obj_facing(
            fronts,
            elevations,
            self._frame_right,
            self._frame_up,
            self._frame_front,
        )

    # -- perspective shift -----------------------------------------------

    def at(self, target) -> "FrameNamespace":
        """Return a new view with origin shifted to ``target``, axes copied from this view.

        The canonical way to ask "from entity j's position, with this view's
        orientation, where is k?" without rebuilding the frame manually.
        Equivalent to the engine's ``scene._frame(at=target, same_as=self)``.

        Parameters
        ----------
        target : int | tuple | MergedObject | Camera
            Anchor to shift the frame origin to: a named object's description,
            camera(N), an object index or a 3D point (``Scene._resolve_entity``).

        Returns
        -------
        FrameNamespace — new view at target's position, this view's axes.

        Examples
        --------
        >>> cam_view = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
        >>> # "From the bin's position (camera's axes), what's at the bin's back?"
        >>> bin_back_scores = cam_view.at(bin_idx).first_person.back

        Notes
        -----
        The returned view shares this view's axes, NOT the target's intrinsic
        axes.  For target's intrinsic body frame, use ``scene.frame(at=<description>)``
        directly (no ``same_as=``).
        """
        return self._scene._frame(at=target, same_as=self)

    # -- wrapping helpers ------------------------------------------------

    def _wrap2d(self, arr: np.ndarray):
        """Wrap a (K,K) numpy array as a ProbabilisticTensor with vars (x1, x2)."""
        return self._scene._wrap_tensor(arr, ndim=2)

    def _wrap1d(self, arr: np.ndarray):
        """Wrap a (K,) numpy array as a ProbabilisticTensor with var x1."""
        return self._scene._wrap_tensor(arr, ndim=1)

    # -- 2D directional relations ----------------------------------------

    @property
    def left(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["left"])

    @property
    def right(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["right"])

    @property
    def front(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["front"])

    @property
    def behind(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["behind"])

    @property
    def above(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["above"])

    @property
    def below(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["below"])

    # -- normalized variants ---------------------------------------------

    @property
    def left_normalized(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["left_normalized"])

    @property
    def right_normalized(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["right_normalized"])

    @property
    def front_normalized(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["front_normalized"])

    @property
    def behind_normalized(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["behind_normalized"])

    @property
    def above_normalized(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["above_normalized"])

    @property
    def below_normalized(self):
        self._ensure_directional()
        return self._wrap2d(self._directional["below_normalized"])

    # -- per-object facing -----------------------------------------------

    @property
    def obj_facing_left(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_left"])

    @property
    def obj_facing_right(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_right"])

    @property
    def obj_facing_front(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_front"])

    @property
    def obj_facing_back(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_back"])

    @property
    def obj_facing_up(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_up"])

    @property
    def obj_facing_down(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_down"])

    @property
    def obj_facing_front_right(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_front_right"])

    @property
    def obj_facing_front_left(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_front_left"])

    @property
    def obj_facing_back_right(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_back_right"])

    @property
    def obj_facing_back_left(self):
        self._ensure_facing()
        return self._wrap1d(self._facing["obj_facing_back_left"])

    # -- direction queries -----------------------------------------------
    # Entry point is :meth:`direction` (``direction(target=...)`` or
    # ``direction(source=..., target=...)``).

    # -- cardinal direction helper ----------------------------------------
    # Constants imported from direction_utils (single source of truth).
    _CARDINAL_TO_ANGLE = CARDINAL_TO_ANGLE
    _CARDINAL_LABELS_8 = CARDINAL_LABELS_8
    _CARDINAL_LABELS_4 = CARDINAL_LABELS_4


    # -- transforms (additive) ------------------------------------------

    def rotate(self, yaw: float = 0.0, pitch: float = 0.0) -> "FrameNamespace":
        """Return a new frame rotated by ``yaw`` and ``pitch`` degrees.

        Sign convention:
          * ``yaw > 0``  → turn RIGHT (clockwise when viewed from above)
          * ``yaw < 0``  → turn LEFT
          * ``pitch > 0`` → tilt UP
          * ``pitch < 0`` → tilt DOWN

        Rotation is applied in the frame's own axes: yaw around the frame's
        up axis, pitch around the (rotated) right axis.  The origin is
        preserved; chain with ``scene._frame(at=p, same_as=rotated)`` to
        also relocate.

        Examples:
          * ``view.rotate(yaw=180)``   — turn around
          * ``view.rotate(pitch=-90)`` — look at the floor
          * ``view.rotate(yaw=90)``    — turn right 90 degrees
        """
        if yaw == 0.0 and pitch == 0.0:
            return self

        right = self._frame_right.copy()
        up = self._frame_up.copy()
        front = self._frame_front.copy()

        # --- Yaw around the frame's up axis ---
        if yaw != 0.0:
            # yaw > 0 = turn RIGHT = front rotates toward +right axis.
            # In a right-handed frame with cross(right, up) == front, the
            # rotation that sends front -> +right around the up axis is
            # Rodrigues with angle = +yaw using axis = up and the identity
            # cross(up, front) == +right.
            theta = math.radians(yaw)
            c, s = math.cos(theta), math.sin(theta)
            front_new = (
                front * c
                + np.cross(up, front) * s
                + up * float(np.dot(up, front)) * (1 - c)
            )
            right_new = (
                right * c
                + np.cross(up, right) * s
                + up * float(np.dot(up, right)) * (1 - c)
            )
            front = front_new / (np.linalg.norm(front_new) + 1e-12)
            right = right_new / (np.linalg.norm(right_new) + 1e-12)

        # --- Pitch around the (new) right axis ---
        if pitch != 0.0:
            # pitch > 0 = tilt UP = front gains +up component.
            # In a right-handed frame, cross(right, front) == -up, so a
            # Rodrigues rotation around +right with angle = +pitch sends
            # front toward -up (i.e. tilts down). Negate the angle so that
            # +pitch tilts up, as documented.
            theta = math.radians(-pitch)
            c, s = math.cos(theta), math.sin(theta)
            front_new = (
                front * c
                + np.cross(right, front) * s
                + right * float(np.dot(right, front)) * (1 - c)
            )
            up_new = (
                up * c
                + np.cross(right, up) * s
                + right * float(np.dot(right, up)) * (1 - c)
            )
            front = front_new / (np.linalg.norm(front_new) + 1e-12)
            up = up_new / (np.linalg.norm(up_new) + 1e-12)

        return FrameNamespace(
            self._scene,
            right,
            up,
            front,
            self._frame_origin.copy(),
            hfov_deg=self._hfov_deg,
        )

    def translate(self, vector) -> "FrameNamespace":
        """Return a new frame with origin shifted by world-space ``vector``.

        Axes are preserved.  Use this to model "step forward 2 meters"
        with ``view.translate(view.frame_front * 2.0)``.
        """
        v = np.asarray(vector, dtype=float).ravel()[:3]
        if v.size != 3:
            raise ValueError(
                f"translate(): vector must have 3 components, got {v.size}"
            )
        return FrameNamespace(
            self._scene,
            self._frame_right.copy(),
            self._frame_up.copy(),
            self._frame_front.copy(),
            self._frame_origin + v,
            hfov_deg=self._hfov_deg,
        )

    # -- two-frame queries --------------------------------------------

    def local(self, target) -> tuple:
        """Where ``target`` (an entity, camera(N) or a 3D point) is as seen from
        this frame: ``(right, up, forward)`` along the frame's own axes, like a
        game engine's InverseTransformPoint. Ignores set_axis_convention."""
        v = np.asarray(self.displacement(target), dtype=float)
        return (float(np.dot(v, self._frame_right)), float(np.dot(v, self._frame_up)),
                float(np.dot(v, self._frame_front)))

    def look_at(self, target) -> "FrameNamespace":
        """A new frame at this frame's position, turned to face ``target`` (a named
        object's description, an entity, camera(N) or a 3D point), kept upright and
        level: only the horizontal direction to the target counts, so a high target
        (a window, a camera) does not tilt the frame and skew its left/right."""
        from saturn.soft_logic.tensor import ProbabilisticTensor
        if isinstance(target, ProbabilisticTensor):
            target = self._scene._described_object(target)
        v = np.asarray(self.displacement(target), dtype=float)
        if float(np.linalg.norm(v)) < 1e-9:
            raise ValueError("look_at(): the target is at the frame's own position")
        up = np.asarray(self._scene.up, dtype=float)
        h = v - float(np.dot(v, up)) * up  # keep only the horizontal direction
        if float(np.linalg.norm(h)) < 1e-6 * float(np.linalg.norm(v)):
            raise ValueError("look_at(): the target is straight above or below; there is no horizontal direction to face")
        return self._scene.frame(position=self._frame_origin,
                                 orientation=self._scene.orientation_from_forward(h))

    def displacement(self, to) -> np.ndarray:
        """World-space vector from this frame's origin to ``to``.

        ``to`` accepts any form understood by ``scene._resolve_entity``:
        a ``Camera`` / ``MergedObject`` instance, a ``("camera"/"object", i)``
        tuple, an int (object index), a string label, a ``FrameNamespace``,
        or a bare 3D point.
        """
        if isinstance(to, FrameNamespace):
            target_pos = to._frame_origin
        else:
            target_pos, _, _ = self._scene._resolve_entity(to, label="to")
        return np.asarray(target_pos, dtype=float) - self._frame_origin


    def rotation_to(self, other: "FrameNamespace") -> tuple:
        """Yaw/pitch (degrees) needed to align self's front with ``other``'s.

        Returns ``(yaw_deg, pitch_deg)`` such that
        ``self.rotate(yaw=yaw_deg, pitch=pitch_deg).front`` is parallel to
        ``other.front``.  Sign convention matches :meth:`rotate` (yaw>0 =
        turn right; pitch>0 = tilt up).

        Useful for "is camera B turned right or left compared to camera A?"
        questions: yaw>0 = right, yaw<0 = left.
        """
        if not isinstance(other, FrameNamespace):
            raise TypeError(
                "rotation_to() expects a Frame; got " f"{type(other).__name__}"
            )
        # Express other.front in self's basis (right, up, front).
        of = np.asarray(other._frame_front, dtype=float)
        r = float(np.dot(of, self._frame_right))
        u = float(np.dot(of, self._frame_up))
        f = float(np.dot(of, self._frame_front))
        # yaw: in horizontal plane (right, front).  yaw>0 = turn right,
        # which rotates self.front toward self.right, so a target with
        # positive r-component lies to the right -> positive yaw.
        yaw_deg = math.degrees(math.atan2(r, f))
        horiz = math.hypot(r, f)
        pitch_deg = math.degrees(math.atan2(u, horiz))
        return (yaw_deg, pitch_deg)


    def rotation_about_axes(self, other: "FrameNamespace") -> dict:
        """The turn from this frame to ``other`` as signed angles (degrees)
        about the question's labeled axes: ``{"X": ax, "Y": ay, "Z": az}``.

        Axes are the ones declared with ``scene.set_axis_convention`` (the
        default labels are right=+X, up=+Y, forward=+Z). Signs follow the
        right-hand rule in that labeled system, so with ``up="+Y",
        forward="-Z"`` a turn to the right is a NEGATIVE angle about Y and
        tilting up is a POSITIVE angle about X. Unlike ``rotation_to`` it also
        reports roll, and the caller never converts signs by hand.
        """
        from scipy.spatial.transform import Rotation

        if not isinstance(other, FrameNamespace):
            raise TypeError("rotation_about_axes() expects a Frame; got " f"{type(other).__name__}")
        basis = [self._frame_right, self._frame_up, self._frame_front]
        # Columns: other's (right, up, front) expressed in self's (right, up, front).
        R_local = np.array([[float(np.dot(o, b)) for o in (other._frame_right, other._frame_up, other._frame_front)]
                            for b in basis])
        M = getattr(self._scene, "_axis_convention_M", None)
        if M is None:
            M = np.eye(3)
        R_label = M @ R_local @ M.T
        # (right, up, front) is a left-handed triple; the labeled frame the
        # question declares is taken as right-handed. A left-handed declared
        # basis would flip every sign, so it is reported, not silently used.
        if np.linalg.det(M) > 0:
            raise ValueError(
                "rotation_about_axes(): the declared axes are left-handed (e.g. right=+X, up=+Y, "
                "forward=+Z); declare the question's right-handed axes with scene.set_axis_convention(...).")
        rv = np.degrees(Rotation.from_matrix(R_label).as_rotvec())
        return {"X": float(rv[0]), "Y": float(rv[1]), "Z": float(rv[2])}

    def translation_to(self, other: "FrameNamespace") -> tuple:
        """Yaw/pitch (degrees) describing where ``other``'s origin sits
        relative to self, projected as virtual rotation angles.

        Mirrors :meth:`rotation_to` in units and sign convention but operates
        on the camera-to-camera *displacement* instead of the front vector.
        Useful for "did the camera step up or to the right?" questions where
        the operator translated rather than rotated.

        Returns ``(yaw_deg, pitch_deg)`` where positive yaw means ``other`` is
        on self's right and positive pitch means ``other`` is above self.
        """
        if not isinstance(other, FrameNamespace):
            raise TypeError(
                "translation_to() expects a Frame; got "
                f"{type(other).__name__}"
            )
        disp = np.asarray(other._frame_origin, dtype=float) - self._frame_origin
        r = float(np.dot(disp, self._frame_right))
        u = float(np.dot(disp, self._frame_up))
        f = float(np.dot(disp, self._frame_front))
        # yaw: in horizontal plane (right, front).  Same sign convention as
        # rotation_to: positive r-component → other is to self's right → yaw>0.
        yaw_deg = math.degrees(math.atan2(r, f))
        horiz = math.hypot(r, f)
        pitch_deg = math.degrees(math.atan2(u, horiz))
        return (yaw_deg, pitch_deg)


    # ------------------------------------------------------------------
    # First-person namespace: view.first_person
    # ------------------------------------------------------------------
    @property
    def first_person(self) -> "_FirstPersonNamespace":
        """First-person perspective predicate: "I AM the anchor; where is k relative to ME?"

        1D scores for each entity from the anchor's body-frame perspective.
        The anchor is the view's origin; the directions are the view's
        body axes.  The anchor IS one of the parties in the relation.

        Examples::

            view.first_person.back[k]              # k is at MY back side
            view.first_person.west[lamp_idx]       # cardinal: lamp is west of MY position
            view.at(chair).first_person.left[k]    # if I were AT the chair, k is at MY left

        Vocabulary
        ----------
        - **Relative**: ``front``, ``front-right``, ``right``, ``back-right``,
          ``back``, ``back-left``, ``left``, ``front-left``.  Aliases (see
          :func:`~saturn.scene.direction_utils.canonical_direction`) mean
          exactly the same: ``behind`` / ``rear`` == ``back``,
          ``forward`` == ``front``, ``behind-left`` == ``back-left``,
          ``forward-right`` == ``front-right``, ...  Here ``behind`` is
          "at MY back side"; the pairwise occlusion predicate
          ``view.third_person.behind[i, j]`` is a separate namespace.
        - **Vertical**: ``above``, ``below``.
        - **Cardinal** (requires :py:meth:`Scene.set_cardinal_vector` first):
          ``north``, ``north-east``, ``east``, ``south-east``, ``south``,
          ``south-west``, ``west``, ``north-west`` (hyphen optional, and
          abbreviations ``n / ne / e / se / s / sw / w / nw``; underscores
          accepted in attribute access, e.g. ``view.first_person.north_east``).

        Indexing
        --------
        Returned arrays have length ``K + C`` where ``K = len(scene.objects)``
        and ``C = len(scene.cameras)``.  Camera ``c`` is at index ``K + c``::

            cam0_score = view.first_person.front[len(scene.objects) + 0]

        The returned :class:`~saturn.soft_logic.predicate_array.PredicateArray`
        behaves as a plain numpy array AND composes in the soft-logic algebra::

            target = (rect("x1") & view.first_person.left("x1")).iota("x1")

        Scoring
        -------
        For each entity ``e`` at world position ``p_e``::

            disp = p_e - view.frame_origin
            yaw  = atan2(disp · frame_right, disp · frame_front)
            score = (1 + cos(yaw - target_yaw(label))) / 2

        Vertical labels (``above`` / ``below``) use elevation:
        ``score = (1 ± disp_y / |disp|) / 2``.
        """
        return _FirstPersonNamespace(self)

    # ------------------------------------------------------------------
    # Third-person namespace: view.third_person
    # ------------------------------------------------------------------
    @property
    def third_person(self) -> "_ThirdPersonNamespace":
        """Third-person observer predicate: "I WATCH i and j; is i <dir>-of j?"

        2D pairwise scores between any two entities, judged from the view's
        observer position.  Neither i nor j is "you" — you are an outside
        observer watching the relation.

        Examples::

            view.third_person.behind[i, j]                # i is depth-behind j
            view.third_person.behind[:, sofa].argmax()    # what's behind the sofa
            view.third_person.behind("x1", "x2")          # variable form (quantifier algebra)

        Vocabulary
        ----------
        - **Axial**: ``front``, ``behind``, ``left``, ``right``
        - **Vertical**: ``above``, ``below``

        Note ``back`` is rejected here — use ``view.first_person.back`` for
        body-frame queries.

        Semantics
        ---------
        ``view.third_person.behind[i, j]`` is high when ``i`` is at greater
        depth than ``j`` along the observer's view axis (the view's
        frame_front).  Colloquially: "from the observer's perspective, j is
        between the observer and i — i.e., i is occluded by j."

        Why this differs from ``view.first_person.back``
        ------------------------------------------------
        ``view.first_person.back[k]`` answers "is k at MY back side (MY =
        the anchor)?" — k is in the −front direction from the anchor.
        ``view.third_person.behind[k, j]`` answers "is k depth-behind j in
        the observer's gaze?" — k is in the +front direction from j.  These
        are OPPOSITE for the front/back axis (left/right coincide).  Match
        the namespace to the question's pronoun:

        - "What's behind ME (the photographer)?"      → ``first_person.back``
        - "Is the cat behind the sofa from camera?"   → ``third_person.behind[cat, sofa]``
        """
        return _ThirdPersonNamespace(self)

    # ------------------------------------------------------------------
    # Intrinsic-orientation namespace: view.facing
    # ------------------------------------------------------------------
    @property
    def facing(self) -> "_FacingNamespace":
        """Intrinsic-orientation predicates: ``view.facing.<dir>[k]``.

        Asks "does entity k face <dir> in the view's frame?"  Each surface
        proxies onto the existing ``view.obj_facing_<dir>`` 1D tensor.

        Vocabulary
        ----------
        ``front``, ``back``, ``left``, ``right``, ``front-left``,
        ``front-right``, ``back-left``, ``back-right``, ``up``, ``down``
        (mirrors :py:attr:`first_person` axial+diagonal labels plus
        ``up``/``down`` for vertical orientation).  The same aliases as
        :py:attr:`first_person` apply (``behind`` == ``back``,
        ``forward-left`` == ``front-left``, ...).

        Indexing
        --------
        ::

            view.facing.front[k]               # k faces +view.front
            view.facing.back_left[k]           # k faces back-and-left
        """
        # up/down: a camera's is the pitch of its view axis relative to THIS frame
        # (tilted up from the asking camera); an object's is its own elevation.
        return _FacingNamespace(self)

    @property
    def turned(self) -> "_TurnedNamespace":
        """How each entity's front has swung away from this frame's front:
        ``view.turned.<dir>[k]``. For a camera, how it turned since the frame's
        own view: left/right (panned), above/below (tilted), back (turned around).

        Unlike ``facing`` (where k points now), ``turned`` scores the turn: a
        20-degree left pan still *faces* front, but it *turned* left.
        """
        return _TurnedNamespace(self)


def _eager_first_person_array(
    view: "FrameNamespace", scene, spec: Dict[str, Any]
) -> np.ndarray:
    """Build the (K+C,) ndarray for ``anchor.first_person.<NAME>`` access.

    Mirrors the shape and indexing of built-in directional labels so existing
    codegen patterns (``.argmax()``, ``[:scene.objects_count]``) work
    unchanged.  For each entity i, computes ``S^{anchor}_r[i, anchor]`` —
    i.e., reference j=anchor (the first-person collapse).
    """
    K = len(scene.objects)
    C = len(scene.cameras)
    N = K + C
    out = np.zeros(N, dtype=float)
    for idx in range(N):
        out[idx] = _compute_pairwise_score(view, scene, spec, idx, None)
    return out


def _entity_centers(scene) -> list:
    """World positions of every entity: objects, then cameras."""
    return ([np.asarray(o.center_world, dtype=float) for o in scene.objects]
            + [np.asarray(c.pos, dtype=float) for c in scene.cameras])


class CompassNotSetError(AttributeError, ValueError):
    """A compass word was used before the scene's north was set."""


def _require_cardinal(scene, where: str) -> None:
    if getattr(scene, "_scene_north_vector", None) is None:
        raise CompassNotSetError(
            f"{where}: compass words need scene.set_cardinal_vector(...) first (Vocabulary without it: "
            "front, behind/back, left, right, above, below and their diagonals).")


class _FirstPersonNamespace:
    """Lazy 1D first-person perspective accessor for :py:meth:`FrameNamespace.first_person`."""

    __slots__ = ("_view",)

    def __init__(self, view: "FrameNamespace"):
        self._view = view

    # ----- accessors -------------------------------------------------------
    # Public accessors return a PredicateArray: numpy-identical for existing
    # programs, and composable in the soft-logic algebra (``left("x1") & ...``)
    # like every other predicate family.
    def __getattr__(self, name: str) -> PredicateArray:
        # ``__slots__`` blocks attribute creation; only labels reach here.
        return self._array(name)

    def __call__(self, label: str) -> PredicateArray:
        return self._array(label)

    def __getitem__(self, label: str) -> PredicateArray:
        return self._array(label)

    def _array(self, label: str) -> PredicateArray:
        arr = PredicateArray(self._compute(label))
        # ``arr[p]`` for a 3D point p: the same score for that location.
        arr._point_scorer = lambda P, _label=label: self._compute(_label, positions=P)
        # An entity standing exactly where the anchor stands (the camera an
        # anchor was built from) has no direction from it: reading it is an
        # error instead of a silent 0 for every label.
        arr._undefined = self._at_origin(label)
        return arr

    def _at_origin(self, label: str) -> dict:
        view = self._view
        centers = _entity_centers(view._scene)
        if not centers:
            return {}
        P = np.asarray(centers, dtype=float).reshape(-1, 3)
        dist = np.linalg.norm(P - view._frame_origin[None, :], axis=1)
        K = len(view._scene.objects)
        out = {}
        for i in np.nonzero(dist <= 1e-9)[0]:
            i = int(i)
            what = f"camera({i - K + 1})" if i >= K else f"object {i}"
            out[i] = (f"first_person.{label}: {what} stands exactly where the anchor stands, so it has no "
                      f"direction from the anchor. Read the entity the question asks about (bind it with "
                      f"score(...)); to ask from where image N was taken, anchor at camera(N).")
        return out

    # ----- internals -------------------------------------------------------
    @staticmethod
    def _canonicalize(label: str) -> str:
        """Lower-case, replace underscores with hyphens, strip whitespace."""
        return str(label).strip().lower().replace("_", "-")

    def _compute(self, label: str, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Scores for every entity, or for the given (M, 3) world ``positions``."""
        view = self._view
        scene = view._scene
        canon = self._canonicalize(label)

        # ----- user-registered predicates take precedence ---------------
        user = getattr(scene, "_user_predicates", None)
        if user and canon in user:
            if positions is not None:
                raise TypeError(f"view.first_person.{label}: a registered predicate scores entities, "
                                "not 3D points; index it with an entity index.")
            spec = user[canon]
            if spec["kind"] in ("h_r", "angle"):
                # Eagerly materialize a (K+C,) ndarray so existing codegen
                # patterns like .argmax() / [:scene.objects_count] keep
                # working unchanged.  Each entity i is scored against
                # j = anchor (first-person collapse).
                return _eager_first_person_array(view, scene, spec)
            # kind == "fn"  (arity >= 2):  redirect to third_person
            raise AttributeError(
                f"view.first_person.{label}: registered predicate {canon!r} "
                f"has arity={spec['arity']}; access it via "
                f"view.third_person.{canon}[i_1, ..., i_{spec['arity']}]."
            )

        # Built-in labels: aliases fold to one spelling ("behind-left" ->
        # "back-left", "north-east" -> "northeast").
        canon = canonical_direction(label)

        # ----- vertical short-circuit ------------------------------------
        canon = {"up": "above", "down": "below"}.get(canon, canon)
        if canon in ("above", "below"):
            return self._vertical_scores(canon, positions=positions)

        # ----- cardinal guard --------------------------------------------
        is_cardinal = canon in CARDINAL_TO_ANGLE
        if is_cardinal and getattr(scene, "_scene_north_vector", None) is None:
            raise ValueError(
                f"view.first_person.{label}: scene.set_cardinal_vector(...) "
                "must be called first to ground cardinal directions."
            )

        # Resolve the requested label to a target yaw in degrees.
        # ``_resolve_direction_angle`` uses world-cardinal convention
        # (north=0, east=90, south=180, west=270).  For cardinal labels
        # that target_yaw is in the *world* frame; for relative labels
        # it's already in the *view's* frame.
        try:
            target_deg = FrameNamespace._resolve_direction_angle(canon)
        except ValueError as e:
            raise ValueError(
                f"view.first_person.{label}: {e}.  Vocabulary: front (or forward), "
                "front-right, right, back-right (or behind-right), back (or behind/rear), "
                "back-left (or behind-left), left, front-left, "
                "above, below, north, north-east, east, south-east, south, "
                "south-west, west, north-west (and abbreviations)."
            ) from None

        target_rad = math.radians(target_deg)

        # For cardinal labels, the "view-frame" yaw of the target direction
        # is offset by the angle between the scene's north and the view's
        # frame_front.  Compute that offset once.
        if is_cardinal:
            north = scene._scene_north_vector  # unit, horizontal
            # Yaw of north in the view's frame: 0 = view's +front, +CW.
            n_r = float(np.dot(north, view._frame_right))
            n_f = float(np.dot(north, view._frame_front))
            view_yaw_of_north = math.atan2(n_r, n_f)
            # World "north" sits at view-yaw = view_yaw_of_north.  A target
            # at world-cardinal-yaw target_rad sits at view-yaw
            # = view_yaw_of_north + target_rad.
            target_rad = view_yaw_of_north + target_rad

        return self._horizontal_scores(target_rad, positions=positions)

    def _horizontal_scores(self, target_rad: float, positions=None) -> np.ndarray:
        """Return per-entity cosine scores for a horizontal target yaw.

        Each entity scores ``(1 + cos(yaw - target_rad)) / 2``, where ``yaw`` is its
        view-frame bearing (0 = view +front, clockwise positive). The score depends
        on direction only; the camera field of view does not enter it.
        """
        view = self._view
        scene = view._scene
        origin = view._frame_origin
        right = view._frame_right
        front = view._frame_front

        if positions is None:
            positions = _entity_centers(scene)
        if len(positions) == 0:
            return np.zeros(0, dtype=float)

        P = np.asarray(positions, dtype=float).reshape(-1, 3)  # (K+C, 3) or given points
        disp = P - origin[None, :]
        r = disp @ right
        f = disp @ front
        horiz = np.sqrt(r * r + f * f)

        scores = np.zeros(len(positions), dtype=float)
        valid = horiz > 1e-12
        if not np.any(valid):
            return scores

        yaw = np.arctan2(r[valid], f[valid])  # 0 = view +front, CW positive
        scores[valid] = (1.0 + np.cos(yaw - target_rad)) / 2.0
        return scores

    def _vertical_scores(self, label: str, positions=None) -> np.ndarray:
        """Return per-entity (or per-point) scores for ``above`` / ``below`` via elevation."""
        view = self._view
        scene = view._scene
        origin = view._frame_origin
        up = view._frame_up

        if positions is None:
            positions = _entity_centers(scene)
        if len(positions) == 0:
            return np.zeros(0, dtype=float)

        P = np.asarray(positions, dtype=float).reshape(-1, 3)
        disp = P - origin[None, :]
        norm = np.linalg.norm(disp, axis=1)

        scores = np.zeros(len(positions), dtype=float)
        valid = norm > 1e-12
        if not np.any(valid):
            return scores

        elev_sin = (disp[valid] @ up) / norm[valid]  # ∈ [-1, 1]
        if label == "above":
            scores[valid] = (1.0 + elev_sin) / 2.0
        else:  # below
            scores[valid] = (1.0 - elev_sin) / 2.0
        return scores


class _ThirdPersonNamespace:
    """Lazy pairwise third-person observer accessor for :py:meth:`FrameNamespace.third_person`."""

    __slots__ = ("_view",)

    _AXIAL = ("front", "behind", "left", "right", "above", "below")
    # Directional combinations: sigmoid(min(h_c1, h_c2) / tau_comb), computed
    # in ``compute_frame_relations``.
    _DIAGONAL = {
        "front-left": "front_left",
        "front-right": "front_right",
        "behind-left": "behind_left",
        "behind-right": "behind_right",
    }

    def __init__(self, view: "FrameNamespace"):
        self._view = view

    def _resolve(self, name: str):
        canon = name.lower().strip().replace("_", "-")
        # ----- user-registered predicates take precedence ---------------
        scene = self._view._scene
        user = getattr(scene, "_user_predicates", None)
        if user and canon in user:
            spec = user[canon]
            if spec["kind"] in ("h_r", "angle"):
                return _RegisteredPairwiseWrapper(scene, spec, self._view, canon)
            # kind == "fn"  (arity >= 2):  K-ary tuple-indexed wrapper.
            return _RegisteredKaryWrapper(
                scene, spec["fn"], spec["arity"], canon,
            )
        if canon in self._AXIAL:
            return getattr(self._view, canon)
        if canon in self._DIAGONAL:
            self._view._ensure_directional()
            return self._view._wrap2d(self._view._directional[self._DIAGONAL[canon]])
        if canonical_direction(canon) in CARDINAL_TO_ANGLE:
            return self._compass(canonical_direction(canon), name)
        if canon in ("back-left", "back-right"):
            raise AttributeError(
                f"view.third_person.{name}: 'back-*' is first-person vocabulary; "
                f"use view.third_person.{canon.replace('back', 'behind')}[i, j] "
                f"or view.first_person.{canon}[k]."
            )
        if canon == "back":
            raise AttributeError(
                "view.third_person.back: 'back' is first-person vocabulary, not "
                "third-person observer.  For 'depth-behind j in observer's "
                "gaze' use view.third_person.behind[i, j].  For 'at the "
                "anchor's body-back side' use view.first_person.back[k]."
            )
        raise AttributeError(
            f"view.third_person.{name}: unknown direction.  Vocabulary: "
            f"{', '.join(self._AXIAL + tuple(self._DIAGONAL))}.  "
            f"Note: 'back' is NOT accepted here — "
            f"use 'behind' for the third-person family.  For the first-person "
            f"family use view.first_person.back.  Compass words (north, south-east, ...) "
            f"work after scene.set_cardinal_vector(...)."
        )

    def _compass(self, label: str, name: str):
        """``third_person.<compass>("x1", "x2")``: x1 lies <compass> of x2. A compass
        relation does not depend on the observer: it is the same score as
        ``first_person.<compass>`` of x1 for a level anchor standing at x2
        ((1 + cos) / 2 of the horizontal angle between x2->x1 and the compass
        direction; 0 on the diagonal)."""
        scene = self._view._scene
        _require_cardinal(scene, f"view.third_person.{name}")
        c = np.asarray(scene.cardinal_vector(label), dtype=float)
        P = np.asarray(_entity_centers(scene), dtype=float).reshape(-1, 3)
        D = P[:, None, :] - P[None, :, :]          # D[i, j] = p_i - p_j
        D[..., 1] = 0.0                            # compass is horizontal (world +Y up)
        norm = np.linalg.norm(D, axis=-1)
        cos = np.where(norm > 1e-12, (D @ c) / np.maximum(norm, 1e-12), 0.0)
        M = np.where(norm > 1e-12, (1.0 + cos) / 2.0, 0.0)
        return self._view._wrap2d(M)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._resolve(name)

    def __call__(self, label: str):
        return self._resolve(label)

    def __getitem__(self, label: str):
        return self._resolve(label)


class _TurnedNamespace:
    """``view.turned.<dir>[k]``: how entity k's front has turned away from the view's
    front. With f = k's unit front in the view's (right, up, front) axes, each label
    scores its component, (1 + f . axis) / 2: right (1 + f_x)/2, above (1 + f_y)/2,
    back (1 - f_z)/2, diagonals on the normalized sum. "+x", "-y", ...: a turn about the
    question's axis (scene.set_axis_convention), read as one of these labels. The score carries the turn's
    size (a 2-degree pan is right 0.52, a 90-degree pan right 1.0 / back 0.5; right
    and back cross at 135 degrees). A "front" part adds nothing to a turn and is
    dropped from diagonals; "front" alone means "did not turn": 1 - the strongest
    of right/left/above/below/back, so any real turn beats it. Unknown front: 0.5.
    """

    __slots__ = ("_view",)

    _AXES = {"right": (1.0, 0.0, 0.0), "left": (-1.0, 0.0, 0.0), "above": (0.0, 1.0, 0.0),
             "below": (0.0, -1.0, 0.0), "front": (0.0, 0.0, 1.0), "back": (0.0, 0.0, -1.0)}

    def __init__(self, view: "FrameNamespace"):
        self._view = view

    # A turn about one of the question's axes (scene.set_axis_convention), right-hand rule in its
    # right-handed system: + about an axis pointing up is a left turn, + about an axis pointing
    # right is a tilt up; an axis pointing down / left swaps them; about the forward axis is a roll,
    # which turned does not measure (0.5).
    _AXIS_TURN = {(1, +1): "left", (1, -1): "right", (0, +1): "above", (0, -1): "below"}   # (role, sign)

    def _axis_label(self, name: str):
        """'+x' / '-Y' / ...: the turned label it names under the declared axes, or None for a roll."""
        sign = +1 if name[0] == "+" else -1
        q = "xyz".index(name[1].lower())
        M = getattr(self._view._scene, "_axis_convention_M", None)
        M = np.eye(3) if M is None else np.asarray(M, dtype=float)
        role = int(np.argmax(np.abs(M[q])))                  # 0 right, 1 up, 2 front
        points = int(np.sign(M[q, role]))
        if role == 2:
            return None
        return self._AXIS_TURN[(role, sign * points)]

    def _resolve(self, name: str):
        if isinstance(name, str) and len(name) == 2 and name[0] in "+-" and name[1].lower() in "xyz":
            label = self._axis_label(name)
            if label is None:
                return self._view._wrap1d(np.full(len(self._view._scene._entity_front_directions()), 0.5))
            return self._resolve(label)
        canon = canonical_direction(name)
        parts = [{"up": "above", "down": "below"}.get(p, p) for p in canon.split("-")]
        if not parts or not all(p in self._AXES for p in parts):
            raise AttributeError(
                f"view.turned.{name}: unknown direction.  Vocabulary: left, right, above/up, "
                f"below/down, back (turned around), pairs such as above-left, and front (did not turn)."
            )
        view = self._view
        if parts == ["front"]:
            turns = [self._component(("right",)), self._component(("left",)), self._component(("above",)),
                     self._component(("below",)), self._component(("back",))]
            return view._wrap1d(1.0 - np.max(np.stack(turns), axis=0))
        return view._wrap1d(self._component(tuple(p for p in parts if p != "front") or ("front",)))

    def _component(self, parts) -> np.ndarray:
        axis = np.sum([self._AXES[p] for p in parts], axis=0)
        axis = axis / np.linalg.norm(axis)
        view = self._view
        fronts = np.asarray(view._scene._entity_front_directions(), dtype=float).reshape(-1, 3)
        basis = np.stack([np.asarray(view._frame_right, dtype=float), np.asarray(view._frame_up, dtype=float),
                          np.asarray(view._frame_front, dtype=float)])
        local = fronts @ basis.T  # (K+C, 3) components along (right, up, front)
        norm = np.linalg.norm(local, axis=1)
        ok = np.isfinite(norm) & (norm > 1e-8)
        scores = np.full(len(local), 0.5)
        scores[ok] = (1.0 + (local[ok] / norm[ok, None]) @ axis) / 2.0
        return scores

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._resolve(name)

    def __call__(self, label: str):
        return self._resolve(label)

    def __getitem__(self, label: str):
        return self._resolve(label)


class _FacingNamespace:
    """Lazy 1D self-orientation accessor for :py:meth:`FrameNamespace.facing`."""

    __slots__ = ("_view",)

    # Maps namespace label → existing FrameNamespace property name.
    _LABEL_MAP = {
        "front": "obj_facing_front",
        "back": "obj_facing_back",
        "left": "obj_facing_left",
        "right": "obj_facing_right",
        "front-left": "obj_facing_front_left",
        "front-right": "obj_facing_front_right",
        "back-left": "obj_facing_back_left",
        "back-right": "obj_facing_back_right",
        "up": "obj_facing_up",
        "down": "obj_facing_down",
        # first_person's vertical words, so one vocabulary serves both namespaces
        "above": "obj_facing_up",
        "below": "obj_facing_down",
    }

    def __init__(self, view: "FrameNamespace"):
        self._view = view

    def _resolve(self, name: str):
        canon = canonical_direction(name)  # "behind" -> "back", ...
        if canon in self._LABEL_MAP:
            return getattr(self._view, self._LABEL_MAP[canon])
        if canon in CARDINAL_TO_ANGLE:
            # "x1 faces <compass>": x1's front points that way, scored by the same
            # facing kernel as facing.front of a level frame looking <compass>.
            scene = self._view._scene
            _require_cardinal(scene, f"view.facing.{name}")
            front = np.asarray(scene.cardinal_vector(canon), dtype=float)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(up, front)
            return FrameNamespace(scene, right / np.linalg.norm(right), up, front,
                                  self._view._frame_origin).obj_facing_front
        raise AttributeError(
            f"view.facing.{name}: unknown direction.  Vocabulary: "
            f"{', '.join(self._LABEL_MAP.keys())} (aliases: behind/rear = back, "
            f"forward = front), and compass words (north, south-east, ...) after "
            f"scene.set_cardinal_vector(...)."
        )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._resolve(name)

    def __call__(self, label: str):
        return self._resolve(label)

    def __getitem__(self, label: str):
        return self._resolve(label)


# ---------------------------------------------------------------------------
# ``Frame`` is the public name.
# ---------------------------------------------------------------------------
Frame = FrameNamespace
