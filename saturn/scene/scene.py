"""
Scene — the central multi-view scene object.

Holds merged objects, cameras, images, and all pre-computed relation tensors.
Provides the public API: frame(), object(), metric_distance(), detect(),
add_camera().
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from saturn.soft_logic.tensor import BINDABLE_ENTITIES, ProbabilisticTensor, is_entity_var
from saturn.scene.direction_utils import CARDINAL_TO_ANGLE, canonical_direction
from saturn.predicates.frame import FrameNamespace
from saturn.predicates.relations import (
    compute_distance_matrices,
    compute_frame_independent_relations,
)
from saturn.predicates.metrics import (
    compute_obj_relative_from_axes,
)
from .direction_utils import (
    CARDINAL_TO_ANGLE,
    CARDINAL_LABELS_4,
    CARDINAL_LABELS_8,
    RELATIVE_LABELS_4,
    RELATIVE_LABELS_8,
)
from .types import Camera, MergedObject
from .cardinal import _CardinalMixin
from .constraints import _ConstraintNamespace
from .entities import _EntityAccessMixin
from .serialization import scene_dump, scene_from_dict, scene_load, scene_to_dict
from saturn.log import get_logger

log = get_logger(__name__)


def _is_entity_tuple(x) -> bool:
    """True for an entity spec like ``("camera", k)`` / ``("object", i)``."""
    return isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], str)


def _orthonormal_frame_axes(f_vec, u_vec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit (right, up, front) axes of a frame from a forward and an up vector.

    ``u_vec`` None means world up. When front and up are nearly parallel, up
    falls back to world up, then to +Z.
    """
    f = np.asarray(f_vec, dtype=float).ravel()[:3]
    if u_vec is None:
        u_vec = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.asarray(u_vec, dtype=float).ravel()[:3]

    nf = float(np.linalg.norm(f))
    if nf < 1e-9:
        raise ValueError("scene.frame(): front vector has zero magnitude.")
    f = f / nf

    if abs(float(np.dot(f, u))) > 0.999:
        u = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(f, u))) > 0.999:
            u = np.array([0.0, 0.0, 1.0], dtype=float)

    right = np.cross(u, f)
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        raise ValueError("scene.frame(): degenerate frame (front || up).")
    right = right / rn
    # Re-orthogonalise up so the basis is exactly right-handed orthonormal.
    up_ortho = np.cross(f, right)
    up_ortho = up_ortho / (float(np.linalg.norm(up_ortho)) + 1e-12)
    return right, up_ortho, f


_FRAME_FORMS = (
    "A frame is scene.frame(position=<3-vector>, orientation=<3x3>) at a point (camera(N).position, "
    "scene.room_center(), any 3D point; without orientation it answers only compass words until "
    ".look_at(target) gives it a facing), or "
    "scene.frame(at=<description>) at a named object: x = score(\"is the object in the red bounding box "
    "a <name>?\").iota(\"x2\"); scene.frame(at=x).")


def _as_point(x) -> Optional[np.ndarray]:
    """``x`` as a 3D point (any 3 numbers: list, tuple, ndarray, tensor), or None if it is not one."""
    if isinstance(x, (ProbabilisticTensor, str, int, np.integer)) or type(x).__name__ == "PredicateArray" \
            or hasattr(x, "cam"):
        return None
    try:
        a = np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return a if a.size == 3 else None


class UnfacedFrame:
    """``scene.frame(position=p)`` without an orientation: a position with no facing yet.

    Compass words (north, south-east, ...) do not depend on a facing, so ``first_person``
    answers them; ``.look_at(target)`` gives the frame a facing for everything else.
    """

    def __init__(self, scene: "Scene", position, hfov_deg=None):
        self._scene = scene
        self._position = np.asarray(position, dtype=float).ravel()[:3]
        self._unfaced_position = self._position      # read by Scene._resolve_entity (look_at / displacement)
        self._hfov_deg = hfov_deg

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    def _any_facing(self) -> FrameNamespace:
        # an upright frame at the position; look_at and compass words do not depend on its facing
        return self._scene._frame_from_pose(self._position, np.eye(3), self._hfov_deg)

    def look_at(self, target) -> FrameNamespace:
        return self._any_facing().look_at(target)

    @property
    def first_person(self) -> "_CompassWords":
        return _CompassWords(self)

    def translate(self, vector) -> "UnfacedFrame":
        return UnfacedFrame(self._scene, self._position + np.asarray(vector, dtype=float).ravel()[:3], self._hfov_deg)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"scene.frame(position=...) has no facing yet, so it has no .{name}: pass orientation=<3x3> "
            "or turn it with .look_at(<target>) first.")


class _CompassWords:
    """``first_person`` of a frame with no facing: compass words only (they need no facing)."""

    __slots__ = ("_frame",)

    def __init__(self, frame: UnfacedFrame):
        self._frame = frame

    def _ns(self, label):
        if canonical_direction(str(label)) not in CARDINAL_TO_ANGLE:
            raise AttributeError(
                f"scene.frame(position=...) has no facing yet, so first_person.{label} is undefined. "
                "Compass words (north, south-east, ...) need no facing; for this one pass "
                "orientation=<3x3> or turn the frame with .look_at(<target>) first.")
        return self._frame._any_facing().first_person

    def __getattr__(self, label: str):
        if label.startswith("_"):
            raise AttributeError(label)
        return getattr(self._ns(label), label)

    def __call__(self, label):
        return self._ns(label)(label)

    def __getitem__(self, label):
        return self._ns(label)[label]


class Scene(_EntityAccessMixin, _CardinalMixin):
    """Multi-view 3D scene.

    Attributes
    ----------
    objects : list[MergedObject]
        Merged objects across all views.
    cameras : list[Camera]
        One camera per input image, plus any added virtual cameras.
    images : list
        Input images (PIL or similar).
    """

    def __init__(
        self,
        objects: List[MergedObject],
        cameras: List[Camera],
        images: List[Any],
        *,
        ground_info: Optional[Dict[str, Any]] = None,
        detect_fn: Optional[Callable[..., List[int]]] = None,
        strict: bool = False,
    ):
        self.objects = list(objects)
        self.cameras = list(cameras)
        self.images = list(images)
        self.ground_info = ground_info
        self._detect_fn = detect_fn

        # Back-reference so cam.first_person / cam.third_person can lazily
        # build a frame at the camera's pose. Lets generated DSL write
        # ``scene.cameras[k].third_person.left(...)`` directly.
        for _cam in self.cameras:
            _cam._scene = self

        # Anchor indices: objects 0..K-1, cameras K..K+C-1. Objects also get
        # a scene back-reference so rotated views can resolve targets lazily.
        K = len(self.objects)
        for i, obj in enumerate(self.objects):
            obj.anchor_index = i
            obj._scene = self
        for c, cam in enumerate(self.cameras):
            cam.anchor_index = K + c

        # Under strict mode every object must have a valid orientation.
        if strict:
            from .anchor import _has_valid_orientation
            for obj in self.objects:
                if not _has_valid_orientation(obj):
                    from .types import SceneBuildError
                    raise SceneBuildError(
                        f"Object {getattr(obj, 'label', '?')!r} "
                        f"(index {getattr(obj, 'id', '?')}) has no valid "
                        f"orientation. OrientAnything must produce a "
                        f"finite, non-zero front/right axis for every "
                        f"object under strict=True."
                    )

        # Precompute (K+C, K+C) predicate matrices.
        from .anchor import compute_anchor_predicates
        self._anchor_predicates = compute_anchor_predicates(
            self.objects + self.cameras, strict=strict,
        )

        # Lazily-computed caches
        self._frame_cache: Dict[str, FrameNamespace] = {}
        self._distance_cache: Optional[Dict[str, np.ndarray]] = None
        self._frame_independent_cache: Optional[Dict[str, np.ndarray]] = None
        self._obj_relative_cache: Optional[Dict[str, np.ndarray]] = None
        self._scene_north_vector: Optional[np.ndarray] = None
        self._facing_overrides: List[Dict[str, Any]] = []  # [{"phrase": str, "front_cam_id": int}]
        # User-registered predicates wired into both anchor.first_person and
        # anchor.third_person via scene.register_predicate(...).  Value shape:
        #   {"kind":"h_r", "h_r":fn, "margin":float, "temperature":float,
        #     "normalize_distance":bool}
        #   {"kind":"angle", "angle_deg":float, "margin":float, "temperature":float,
        #     "normalize_distance":bool}
        #   {"kind":"fn", "fn":callable, "arity":int}   # third_person only, K>=2
        self._user_predicates: Dict[str, Dict[str, Any]] = {}
        # Lazily-computed scene scale (s_scene = 0.9-quantile of pairwise
        # entity distances) used for delta_local normalization in registered
        # predicates with normalize_distance=True.
        self._scene_scale_cache: Optional[float] = None
        # Per-question axis convention (declared via set_axis_convention).
        # Default = identity: anchor's (right, up, front) = question's (+X, +Y, +Z).
        # Stored as a 3x3 signed permutation matrix M with M[q_axis_idx, anchor_role_idx] = sign,
        # where roles are ordered (right=0, up=1, front=2) and q_axes are ('X'=0, 'Y'=1, 'Z'=2).
        self._axis_convention_M: Optional[np.ndarray] = None  # None → identity (no flip)

        # Planner-emitted scene caption surfaced into dump_facts_str so codegen
        # sees scene narrative alongside scene facts. Populated by
        # ``set_planner_context`` after the planner runs. (Object groundings
        # are surfaced via a separate codegen placeholder, not this field.)
        self._setup_caption: Optional[str] = None

    # ------------------------------------------------------------------
    # Per-question axis convention (declared via set_axis_convention)
    # ------------------------------------------------------------------

    def set_axis_convention(
        self,
        *,
        right: Optional[str] = None,
        up: Optional[str] = None,
        forward: Optional[str] = None,
    ) -> None:
        """Declare the question's axis labels for the anchor's (right, up, front).

        Each kwarg is a signed axis token: ``"+X"``, ``"-Y"``, ``"+Z"``, etc.
        The string is parsed as ``(sign, axis_letter)``; the framework builds a
        signed permutation matrix mapping anchor-local (right, up, front)
        scalars into the question's labeled (X, Y, Z) axes. Only
        ``anchor.project()`` output is affected — predicates and underlying
        geometry stay in the internal frame.

        Omitted kwargs default to the internal convention:
            right="+X",  up="+Y",  forward="+Z".

        Examples
        --------
        >>> # "+Y up, -Z forward, right-handed" (OpenGL labels):
        >>> scene.set_axis_convention(up="+Y", forward="-Z")
        >>> # Z-up world, Y points toward viewer:
        >>> scene.set_axis_convention(up="+Z", forward="-Y")
        >>> # Reset to default (the internal +X right, +Y up, +Z front):
        >>> scene.set_axis_convention()

        Notes
        -----
        The convention does NOT affect ``first_person`` / ``third_person`` /
        ``facing`` predicates, nor ``rotation_to`` / ``scene.angle`` outputs.
        It only changes the sign/permutation applied to ``anchor.project(vec)``.
        """
        defaults = {"right": "+X", "up": "+Y", "forward": "+Z"}
        raw = {
            "right": right if right is not None else defaults["right"],
            "up": up if up is not None else defaults["up"],
            "forward": forward if forward is not None else defaults["forward"],
        }

        def _parse(tok: str, role: str) -> tuple:
            if not isinstance(tok, str):
                raise TypeError(
                    f"set_axis_convention({role}={tok!r}): each role takes a string naming the "
                    f"question's axis with a sign, e.g. right='+X', up='+Y', forward='-Z'.")
            tok = tok.strip()
            if len(tok) != 2 or tok[0] not in ("+", "-") or tok[1].upper() not in ("X", "Y", "Z"):
                raise ValueError(
                    f"set_axis_convention({role}={tok!r}): expected '+X', '-Y', "
                    f"'+Z', etc. — a sign followed by a single axis letter."
                )
            sign = +1 if tok[0] == "+" else -1
            axis_idx = {"X": 0, "Y": 1, "Z": 2}[tok[1].upper()]
            return sign, axis_idx

        parsed = {role: _parse(tok, role) for role, tok in raw.items()}

        # Validate that the three roles map to three distinct question axes.
        question_axes = {parsed[role][1] for role in ("right", "up", "forward")}
        if len(question_axes) != 3:
            seen = {role: ("XYZ"[parsed[role][1]]) for role in ("right", "up", "forward")}
            raise ValueError(
                f"set_axis_convention: the three roles must map to distinct question "
                f"axes (X, Y, Z); got {seen}."
            )

        # If all defaults, no need to store a matrix — keep None for fast path.
        if (parsed["right"] == (+1, 0) and parsed["up"] == (+1, 1)
                and parsed["forward"] == (+1, 2)):
            self._axis_convention_M = None
            return

        # Build M[q_axis_idx, role_idx] = sign so that
        # (x_q, y_q, z_q) = M @ (sx, sy, sz).
        # Reasoning: if the user wrote forward="-Z", that means anchor's +front
        # direction equals the question's -Z direction. So projecting a world-vec
        # onto question's +Z axis = vec · (-anchor_front) = -sz. The matrix
        # entry M[Z, front] = -1 captures that. Same logic for any kwarg.
        M = np.zeros((3, 3), dtype=float)
        for role_idx, role in enumerate(("right", "up", "forward")):
            sign, q_axis_idx = parsed[role]
            M[q_axis_idx, role_idx] = sign
        self._axis_convention_M = M

    def _axis_project(self, local_scalars: tuple) -> tuple:
        """Apply the current axis convention to anchor-local scalars.

        ``local_scalars`` is ``(sx, sy, sz)`` along anchor's (right, up, front).
        Returns ``(x_q, y_q, z_q)`` along the question's (+X, +Y, +Z) axes.

        Note: we avoid ``np.matmul`` here because M is a signed permutation
        matrix, and matmul's ``0 * inf = NaN`` rule would contaminate clean
        output axes when one input component is inf/NaN. With this explicit
        loop, pathological inputs stay isolated to the axis they enter on.
        """
        sx, sy, sz = float(local_scalars[0]), float(local_scalars[1]), float(local_scalars[2])
        if self._axis_convention_M is None:
            return (sx, sy, sz)
        M = self._axis_convention_M
        locals_arr = (sx, sy, sz)
        out = [0.0, 0.0, 0.0]
        for row in range(3):
            # Signed permutation: exactly one non-zero entry per row.
            for col in range(3):
                if M[row, col] != 0.0:
                    out[row] = float(M[row, col]) * locals_arr[col]
                    break
        return (out[0], out[1], out[2])

    # ------------------------------------------------------------------
    # Scene state dump for codegen LLM
    # ------------------------------------------------------------------

    def set_planner_context(
        self,
        setup_caption: Optional[str] = None,
    ) -> None:
        """Attach the planner's setup_caption to the scene for dump_facts_str().

        The caption surfaces inside the SCENE FACTS block so codegen sees it
        as scene-context narrative rather than a task-framing imperative.
        Object groundings live in their own codegen placeholder (built and
        injected by the caller), not on the Scene.
        """
        cap = (setup_caption or "").strip()
        self._setup_caption = cap or None

    def dump_facts_str(self, objects_table: bool = False) -> str:
        """Compact text dump of post-detection scene state for the codegen LLM.

        Always includes the planner's setup_caption (attached via
        ``set_planner_context``), the object/camera counts and the axis
        convention in force.

        ``objects_table=True`` also lists every detected object as
        ``[i] 'label' views=[...] merged=...``. Off by default: with the
        table in the prompt, generated programs tend to hard-code
        ``scene.objects[i]`` indices instead of grounding objects by name.
        Used for debugging and reports.
        """
        if not self.objects and not self.cameras:
            return "SCENE FACTS: (empty scene — no objects or cameras)"

        lines = ["SCENE FACTS:"]
        if self._setup_caption:
            lines.append(f"- setup_caption: {self._setup_caption}")
        lines.append(f"- objects_count: {len(self.objects)}")
        lines.append(f"- cameras_count: {len(self.cameras)}")

        # Surface the actual axis labels currently in force. The default
        # (identity) is right=+X, up=+Y, forward=+Z — print them explicitly so
        # codegen never has to assume what "not set" means. (right, up, forward)
        # is a physically LEFT-handed triple (right x up points backward in the real
        # world; numerically np.cross(right, up) == front only because the canonical
        # engine world is mirrored), so a labelling with det(M) > 0 is left-handed;
        # same test as FrameNamespace.rotation_about_axes.
        if self._axis_convention_M is None:
            r_tok, u_tok, f_tok, status, handed = "+X", "+Y", "+Z", "default", "left-handed"
        else:
            M = self._axis_convention_M
            tokens = []
            for role_idx in range(3):
                col = M[:, role_idx]
                q_axis_idx = int(np.argmax(np.abs(col)))
                sign = "+" if col[q_axis_idx] > 0 else "-"
                tokens.append(f"{sign}{'XYZ'[q_axis_idx]}")
            r_tok, u_tok, f_tok = tokens
            status = "set"
            handed = "left-handed" if int(round(float(np.linalg.det(M)))) > 0 else "right-handed"
        lines.append(
            f"- axis_convention: right={r_tok}, up={u_tok}, forward={f_tok} ({status}, {handed})"
        )

        if objects_table and self.objects:
            lines.append("- objects (detected):")
            for i, obj in enumerate(self.objects):
                label = getattr(obj, "label", "") or "?"
                views = list(getattr(obj, "views", []) or [])
                merged = len(views) > 1
                # Truncate over-long labels (planner cue phrases can be ~80 chars)
                label_disp = label if len(label) <= 50 else label[:49] + "…"
                lines.append(
                    f"  [{i}] {label_disp!r:54s}  views={views}  merged={merged}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pose-constraint namespace (opt-in; inert unless `scene.constraint`
    # is explicitly accessed by caller code or by an extractor pipeline)
    # ------------------------------------------------------------------

    @property
    def constraint(self) -> "_ConstraintNamespace":
        """Lazy namespace for adding pose constraints stated in question text.

        Usage::

            scene.constraint.rotation(scene.cameras[0], scene.cameras[2], yaw=180)
            scene.constraint.same_position(scene.cameras[0], scene.cameras[1])
            scene.constraint.face(scene.objects[5], toward=scene.cameras[0])

        Camera-pose calls (``rotation``, ``same_position``) accumulate a
        constraint and re-solve immediately, mutating
        ``scene.cameras[i].extrinsics`` (and the derived ``position_world`` /
        ``heading`` fields). Object-orientation calls (``face``) mutate the
        target object's world-frame axes directly so that any anchor built
        from ``obj.orientation`` reflects the constraint; they do not re-solve
        camera poses. ``scene.constraint.clear()`` restores VGGT's original
        extrinsics AND original object orientations.

        The namespace has no effect unless caller code invokes it.
        """
        ns = getattr(self, "_constraint_ns", None)
        if ns is None:
            ns = _ConstraintNamespace(self)
            self._constraint_ns = ns
        return ns

    # ------------------------------------------------------------------
    # Frame factory — single unified entry point.
    # ------------------------------------------------------------------

    def frame(self, at=None, *, position=None, orientation=None, hfov_deg: Optional[float] = None, **other):
        """A frame of reference for a program: where I stand and which way I face. Two forms.

        ``scene.frame(position=<3-vector>, orientation=<3x3>)``
            At a point (a camera's position, ``scene.room_center()``, any 3D point; ``at=<point>``
            is the same). Without ``orientation`` the frame has no facing: it answers compass
            words (north, south-east, ...), which do not depend on one, and ``.look_at(target)``
            gives it a facing for the rest: ``scene.frame(position=p).look_at(camera(2))``.
        ``scene.frame(at=<description>, orientation=None)``
            At the object a description names, ``score("... a cat?").iota("x2")``: the frame stands
            at ``description.assign()``, the index program's pick, facing where that object faces
            unless ``orientation`` is given. The program never handles the index.

        Everything else (an index, a camera or a variable name as ``at=``; ``front=``,
        ``up=``, ``same_as=``) is an error that names these two forms. The engine's own frame
        construction uses ``Scene._frame``.
        """
        if other:
            raise TypeError(
                f"scene.frame() does not take {', '.join(f'{k}=' for k in other)}. "
                + _FRAME_FORMS)
        if at is not None:
            if position is not None:
                raise TypeError("scene.frame(): at= and position= are two different forms; pass one. "
                                + _FRAME_FORMS)
            point = _as_point(at)
            if point is not None:                  # a place: the same as position=
                position, at = point, None
            elif not isinstance(at, ProbabilisticTensor):
                hint = (" score(...) needs a variable: score(...).iota(\"x2\")."
                        if type(at).__name__ == "PredicateArray" else "")
                raise TypeError(
                    f"scene.frame(at=...) takes a named object's description or a 3D point, not "
                    f"{type(at).__name__}.{hint} " + _FRAME_FORMS)
            else:
                return self._frame_at_description(at, orientation, hfov_deg)
        if position is None:
            raise TypeError("scene.frame(): pass at=<description> or position=. " + _FRAME_FORMS)
        if orientation is None:
            return UnfacedFrame(self, position, hfov_deg)
        return self._frame_from_pose(position, orientation, hfov_deg)

    def _described_object(self, desc: ProbabilisticTensor) -> int:
        """The object a description names: ``desc.assign()``, exactly the index program's pick."""
        if len(desc.vars) != 1 or not is_entity_var(desc.vars[0]):
            raise ValueError(
                f"a frame takes the description of one object (a score over one variable, "
                f"e.g. .iota(\"x2\")); got one over {list(desc.vars)}.")
        if not self.objects:
            raise ValueError("no objects were grounded in this scene, so a description names none.")
        token = BINDABLE_ENTITIES.set(len(self.objects))
        try:
            return int(desc.assign()[desc.vars[0]])
        finally:
            BINDABLE_ENTITIES.reset(token)

    def _frame_at_description(self, desc: ProbabilisticTensor, orientation, hfov_deg) -> FrameNamespace:
        """The frame at ``desc.assign()`` (in that object's own orientation unless one is given)."""
        c = self._described_object(desc)
        R = self.objects[c].orientation if orientation is None else orientation
        return self._frame_from_pose(self.objects[c].position, R, hfov_deg)

    def _frame(
        self,
        at=None,
        *,
        position=None,
        orientation=None,
        front=None,
        up=None,
        same_as=None,
        hfov_deg: Optional[float] = None,
    ) -> FrameNamespace:
        """Every frame form the engine uses internally (programs call ``frame``).

        Build a Frame anchored at a 3D pose.

        Two usage forms:

        1. **Explicit pose (preferred)** — ``scene._frame(position=..., orientation=...)``.
           Pass any 3-vector and any 3×3 orientation matrix:

               view = scene._frame(position=scene.objects[i].position,
                                  orientation=scene.cameras[k].orientation)

           Both arguments accept arbitrary numpy values, so centroids, midpoints,
           or computed poses flow through naturally:

               midpoint = (scene.objects[a].position + scene.objects[b].position) / 2
               view = scene._frame(position=midpoint,
                                  orientation=scene.cameras[0].orientation)

           When both ``position`` and ``orientation`` are supplied, the
           ``at`` / ``front`` / ``up`` / ``same_as`` arguments are not allowed.

        2. **Entity anchor** — ``scene._frame(at=entity, ...)``, in one of three ways:

           a. **Intrinsic axes** — ``scene._frame(at=entity)``.
              ``front`` is the entity's ``.front``, ``up`` is the entity's
              ``.up`` (or world up for raw points).

           b. **Custom forward** — ``scene._frame(at=entity_or_point, front=vec_or_entity)``.
              Builds a horizontal forward from ``front``.  If ``front`` is an
              entity, its ``.front`` is used.  If it's a 3D vector, it's used
              directly.  If it's a "facing target" (e.g. another entity whose
              **position** you want to face, not its forward), use
              ``front=scene._frame(at=entity).displacement(to=target)``.
              ``up=`` defaults to world-up; pass an explicit vector for tilted
              frames.

           c. **Copy axes** — ``scene._frame(at=entity, same_as=other_frame)``.
              Origin comes from ``at``; axes are copied from ``other_frame``
              (which may be a ``Frame`` or anything with ``.front`` / ``.up``).
              Useful for "sitting at the table, oriented the same way as
              camera 2" queries.

        Parameters
        ----------
        at : entity specifier or 3D point
            Origin of the frame.  Accepts the same forms as
            ``_resolve_entity``, and ``camera(N)``, which anchors on that
            camera exactly as ``scene.cameras[N - 1]`` does.
        position : 3-vector, optional
            Origin of the frame (explicit-pose form).
        orientation : 3x3 matrix, optional
            Columns ``[right, up, front]`` (explicit-pose form); each column
            is normalised.
        front : np.ndarray, entity, or None
            Horizontal forward direction.  Mutually exclusive with
            ``same_as=``.  If None and ``same_as`` is None, uses ``at``'s
            intrinsic forward (requires ``at`` to be an entity).
        up : np.ndarray or None
            World-up override.  Defaults to ``[0, 1, 0]``.
        same_as : Frame, entity, or None
            Copy ``front`` and ``up`` from this source.  Mutually exclusive
            with ``front=``.
        hfov_deg : float, optional
            Horizontal field of view for FOV-aware matching, in both forms.
            When omitted, the explicit-pose form uses the FOV of the camera
            at ``position`` (if any), and the entity-anchor form uses the
            FOV of ``at`` when it is a camera.

        Returns
        -------
        FrameNamespace
            The new frame.  All existing relation tensors and methods work
            unchanged.
        """
        self._check_frame_arguments(at, position, orientation, front, up, same_as)
        if position is not None:
            return self._frame_from_pose(position, orientation, hfov_deg)
        return self._frame_at_entity(
            at, front=front, up=up, same_as=same_as, hfov_deg=hfov_deg
        )

    @staticmethod
    def _check_frame_arguments(at, position, orientation, front, up, same_as) -> None:
        """Reject argument combinations ``scene.frame`` does not accept.

        On return either both ``position`` and ``orientation`` are given
        (explicit-pose form) or neither is and ``at`` is given.
        """
        if position is not None or orientation is not None:
            if position is None or orientation is None:
                raise ValueError(
                    "scene.frame(): pass BOTH ``position=`` and ``orientation=``, "
                    "or neither (use entity-anchor form)."
                )
            if at is not None or front is not None or same_as is not None or up is not None:
                raise ValueError(
                    "scene.frame(): explicit-pose form (position=, orientation=) is "
                    "mutually exclusive with at=, front=, same_as=, up=."
                )
            return
        if at is None:
            raise TypeError(
                "scene.frame(): missing required argument. Pass either "
                "``position=`` and ``orientation=`` (preferred), or ``at=`` "
                "(entity-anchor form)."
            )
        if front is not None and same_as is not None:
            raise ValueError(
                "scene.frame(): pass either ``front=`` or ``same_as=``, not both."
            )

    def _frame_from_pose(self, position, orientation, hfov_deg) -> FrameNamespace:
        """Frame at ``position`` whose axes are the normalised orientation columns."""
        origin = np.asarray(position, dtype=float).ravel()[:3]
        if origin.size != 3:
            raise ValueError(
                f"scene.frame(): position must be a 3-vector; got shape "
                f"{np.asarray(position).shape}."
            )
        R_mat = np.asarray(orientation, dtype=float)
        if R_mat.shape != (3, 3):
            raise ValueError(
                f"scene.frame(): orientation must be a 3x3 matrix with "
                f"columns [right, up, front]; got shape {R_mat.shape}."
            )
        r_col, u_col, f_col = R_mat[:, 0], R_mat[:, 1], R_mat[:, 2]
        nf = float(np.linalg.norm(f_col))
        nr = float(np.linalg.norm(r_col))
        nu = float(np.linalg.norm(u_col))
        if min(nf, nr, nu) < 1e-9:
            raise ValueError(
                "scene.frame(): orientation has a zero-magnitude column."
            )
        # An explicit hfov_deg wins; otherwise a frame placed at a camera
        # (the common ``position=cameras[k].position`` pattern) gets that
        # camera's FOV, so generated programs need not pass it.
        if hfov_deg is None:
            hfov_deg = self._hfov_of_camera_at(origin)
        return FrameNamespace(
            self, r_col / nr, u_col / nu, f_col / nf, origin, hfov_deg=hfov_deg
        )

    def _hfov_of_camera_at(self, origin: np.ndarray) -> Optional[float]:
        """FOV of the first camera within 1e-4 of ``origin``, else None."""
        for cam in self.cameras:
            cam_pos = np.asarray(cam.position_world, dtype=float).ravel()[:3]
            if cam_pos.size == 3 and np.linalg.norm(origin - cam_pos) < 1e-4:
                return self._camera_hfov_deg(cam)
        return None

    def _frame_at_entity(self, at, *, front, up, same_as, hfov_deg=None) -> FrameNamespace:
        """Entity-anchor form of ``scene.frame``.

        The origin is ``at``; the axes come from ``same_as``, ``front`` or
        ``at`` itself, and ``up`` replaces their up vector. An explicit
        ``hfov_deg`` wins over the FOV of a camera ``at``.
        """
        # camera(N) is an int index that carries its Camera; a bare int is an
        # object index, so anchor on the carried camera instead.
        if isinstance(getattr(at, "cam", None), Camera):
            at = at.cam
        origin_pos, _, _ = self._resolve_entity(at, label="at")
        origin_pos = np.asarray(origin_pos, dtype=float)

        if same_as is not None:
            f_vec, u_vec = self._copied_frame_axes(same_as)
        elif front is not None:
            f_vec, u_vec = self._frame_front_vector(front), None
        else:
            f_vec, u_vec = self._frame_intrinsic_axes(at)
        if up is not None:
            u_vec = np.asarray(up, dtype=float).ravel()[:3]

        right, up_ortho, f = _orthonormal_frame_axes(f_vec, u_vec)
        if hfov_deg is None:
            hfov_deg = self._frame_hfov_of_entity(at)
        return FrameNamespace(self, right, up_ortho, f, origin_pos, hfov_deg=hfov_deg)

    @staticmethod
    def _copied_frame_axes(same_as) -> Tuple[np.ndarray, np.ndarray]:
        """(front, up) copied from a Frame or an entity (``same_as=``)."""
        if isinstance(same_as, FrameNamespace):
            return (np.asarray(same_as._frame_front, dtype=float),
                    np.asarray(same_as._frame_up, dtype=float))
        if hasattr(same_as, "front_vec") and hasattr(same_as, "up_vec"):
            return (np.asarray(same_as.front_vec, dtype=float),
                    np.asarray(same_as.up_vec, dtype=float))
        raise TypeError(
            "scene.frame(same_as=...): expected Frame or entity "
            "with .front_vec/.up_vec attributes; got "
            f"{type(same_as).__name__}."
        )

    def _frame_front_vector(self, front) -> np.ndarray:
        """Forward vector from ``front=``: a vector, an entity or an entity spec."""
        if isinstance(front, np.ndarray) or isinstance(front, list):
            return np.asarray(front, dtype=float).ravel()[:3]
        if hasattr(front, "front_vec"):
            return np.asarray(front.front_vec, dtype=float).ravel()[:3]
        # An entity specifier (tuple / int / label) contributes its forward.
        _, fwd, _ = self._resolve_entity(front, label="front")
        if fwd is None:
            raise TypeError(
                "scene.frame(front=...): expected np.ndarray, entity, "
                f"or entity-spec; got {type(front).__name__}."
            )
        return np.asarray(fwd, dtype=float).ravel()[:3]

    def _frame_intrinsic_axes(self, at) -> Tuple[np.ndarray, np.ndarray]:
        """(front, up) of the entity ``at``; a raw point has none."""
        if isinstance(at, (Camera, MergedObject)):
            ent = at
        elif _is_entity_tuple(at):
            kind, idx = at
            if kind.lower() in ("camera", "cam"):
                ent = self.cameras[int(idx)]
            else:
                ent = self.objects[int(idx)]
        elif isinstance(at, (int, np.integer)):
            ent = self.objects[int(at)]
        else:
            raise ValueError(
                "scene.frame(): when ``at`` is a raw point, you must pass "
                "``front=`` or ``same_as=``."
            )
        return (np.asarray(ent.front_vec, dtype=float),
                np.asarray(ent.up_vec, dtype=float))

    def _frame_hfov_of_entity(self, at) -> Optional[float]:
        """FOV of ``at`` when it is a camera (instance or camera tuple), else None."""
        if isinstance(at, Camera):
            return self._camera_hfov_deg(at)
        if _is_entity_tuple(at) and at[0].lower() in ("camera", "cam"):
            return self._camera_hfov_deg(self.cameras[int(at[1])])
        return None

    # ------------------------------------------------------------------
    # Orientation helper — build a 3x3 from a forward vector + world-up.
    # ------------------------------------------------------------------

    def orientation_from_forward(
        self,
        forward: np.ndarray,
        up: np.ndarray = None,
    ) -> np.ndarray:
        """Build a 3x3 orientation matrix from a forward direction.

        Use when you have a forward *direction vector* (e.g. "from A toward B")
        and want to feed it to ``scene.frame(orientation=...)``.  The matrix's
        columns are ``[right, up, forward]`` — the convention scene.frame uses.

        Parameters
        ----------
        forward : np.ndarray
            World-frame forward direction, shape (3,).  Need not be unit.
        up : np.ndarray, optional
            World-frame up direction, shape (3,).  Defaults to ``[0, 1, 0]``.
            Falls back to ``[0, 0, 1]`` if forward is parallel to the supplied up.

        Returns
        -------
        np.ndarray
            3x3 orthonormal matrix with columns [right, up, forward].

        Examples
        --------
        # "Sitting at A, facing B"
        forward = scene.objects[B].position - scene.objects[A].position
        view = scene.frame(
            position=scene.objects[A].position,
            orientation=scene.orientation_from_forward(forward),
        )
        """
        from saturn.scene.direction_utils import as_vector3
        f = as_vector3(forward, "scene.orientation_from_forward(forward)")
        nf = float(np.linalg.norm(f))
        if nf < 1e-9:
            raise ValueError(
                "scene.orientation_from_forward(): forward has zero magnitude."
            )
        f = f / nf

        if up is None:
            u = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            u = np.asarray(up, dtype=float).ravel()[:3]
            nu = float(np.linalg.norm(u))
            if nu < 1e-9:
                raise ValueError(
                    "scene.orientation_from_forward(): up has zero magnitude."
                )
            u = u / nu

        # If forward and up are parallel, fall back to a non-degenerate up.
        if abs(float(np.dot(f, u))) > 0.999:
            u = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(f, u))) > 0.999:
                u = np.array([0.0, 0.0, 1.0])

        right = np.cross(u, f)
        right = right / float(np.linalg.norm(right))
        up_ortho = np.cross(f, right)
        up_ortho = up_ortho / float(np.linalg.norm(up_ortho))
        return np.column_stack([right, up_ortho, f])


    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _ensure_distances(self):
        if self._distance_cache is not None:
            return
        positions = self._entity_positions()
        point_clouds = self._entity_point_clouds()
        self._distance_cache = compute_distance_matrices(positions, point_clouds)

    def _entity_positions(self) -> np.ndarray:
        """Positions for all entities (objects + camera entity_ids)."""
        obj_pos = self._object_positions()
        cam_pos = (
            np.array([cam.position_world for cam in self.cameras], dtype=float)
            if self.cameras
            else np.zeros((0, 3), dtype=float)
        )
        if len(obj_pos) == 0 and len(cam_pos) == 0:
            return np.zeros((0, 3), dtype=float)
        return np.concatenate([obj_pos, cam_pos], axis=0)

    def _entity_point_clouds(self) -> List[Optional[np.ndarray]]:
        """Point clouds for all entities (None for cameras)."""
        clouds: List[Optional[np.ndarray]] = [obj.world_points for obj in self.objects]
        clouds.extend([None] * len(self.cameras))
        return clouds

    @property
    def distance(self) -> ProbabilisticTensor:
        """NxN center-to-center distance divided by the largest one (N = objects + cameras).

        Unclipped, so it ranks every pair (use it for "farthest"); closeness is the
        saturating degree of "near"."""
        self._ensure_distances()
        return self._wrap_tensor(self._distance_cache["distance"], ndim=2)

    @property
    def distance_edge(self) -> ProbabilisticTensor:
        """NxN normalized edge-to-edge distance tensor."""
        self._ensure_distances()
        return self._wrap_tensor(self._distance_cache["distance_edge"], ndim=2)

    @property
    def closeness(self) -> ProbabilisticTensor:
        """1 - distance. Diagonal zeroed."""
        self._ensure_distances()
        return self._wrap_tensor(self._distance_cache["closeness"], ndim=2)

    def metric_distance(
        self,
        a_idx: int,
        b_idx: int,
        mode: str = "center",
        return_uncertainty: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """Metric (unnormalized) distance between two entities.

        Parameters
        ----------
        a_idx, b_idx : entity indices (object idx or camera entity_id)
        mode : "center" or "edge"
        return_uncertainty : if True, return (distance, uncertainty)
        """
        self._ensure_distances()
        key = "distance_center_raw" if mode == "center" else "distance_edge_raw"
        raw = self._distance_cache[key]
        dist = float(raw[a_idx, b_idx])
        if return_uncertainty:
            # Simple uncertainty heuristic: larger distance → larger uncertainty
            unc = dist * 0.1  # 10% relative uncertainty
            return dist, unc
        return dist


    def room_center(self) -> np.ndarray:
        """Estimate the center of the room / scene.

        Computed as the centroid of all object centers and camera positions,
        which is more robust than using only camera positions (especially
        with only 2 cameras).

        Returns
        -------
        np.ndarray — (3,) world-space position.
        """
        positions = []
        for obj in self.objects:
            positions.append(np.asarray(obj.center_world, dtype=float))
        for cam in self.cameras:
            positions.append(np.asarray(cam.position_world, dtype=float))
        if not positions:
            return np.zeros(3, dtype=float)
        return np.mean(positions, axis=0)

    @property
    def step(self) -> float:
        """A small reference distance — roughly one human step (~30 cm).

        Use this whenever code needs a "small move" that is meaningful at
        the scale of the actual scene, instead of hardcoding a magic number.

        VGGT world units are arbitrary (relative scale), so the step is ~5%
        of the bounding-box diagonal of all object + camera positions. A
        typical room reconstructed by VGGT has a diagonal of a few units, so
        this lands at fractions of a unit — small enough to be a step, large
        enough to be detectable above reconstruction noise.

        Returns
        -------
        float — a positive distance in the same units as object/camera
            positions.
        """
        positions = []
        for obj in self.objects:
            positions.append(np.asarray(obj.center_world, dtype=float))
        for cam in self.cameras:
            positions.append(np.asarray(cam.position_world, dtype=float))
        if not positions:
            return 0.05  # tiny fallback for empty scenes

        arr = np.asarray(positions, dtype=float)
        diameter = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
        if not np.isfinite(diameter) or diameter <= 0:
            return 0.05
        return 0.05 * diameter


    # ------------------------------------------------------------------
    # User-registered predicates (anchor.first_person + anchor.third_person)
    # ------------------------------------------------------------------

    def register_predicate(
        self,
        name: str,
        *,
        h_r: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
        margin: float = 0.05,
        temperature: float = 0.03,
        normalize_distance: bool = True,
        angle_deg: Optional[float] = None,
        fn: Optional[Callable[..., float]] = None,
        arity: Optional[int] = None,
    ) -> None:
        """Register a custom predicate on this scene.

        Three registration forms, matching the paper's :math:`S^a_r[i,j]` model:

        1) ``h_r=<callable>`` — pairwise evidence function with
           signature ``h_r(delta_local, R_i_local, R_j_local) -> float``.
           ``delta_local`` is the (3,) displacement from j to i expressed in
           anchor a's local frame; ``R_i_local``, ``R_j_local`` are the
           entity rotations in that frame.  The harness applies
           ``sigma((h_r - margin) / temperature)`` and wires BOTH
           ``anchor.first_person.<name>[k]`` (j=anchor) AND
           ``anchor.third_person.<name>[i, j]`` (general).
        2) ``angle_deg=<float>`` — sugar for a yaw-target direction predicate.
           ``angle_deg=0`` is the anchor's front, ``90`` is right, ``180`` is
           back, ``270`` is left.  Same dual wiring as ``h_r``.
        3) ``fn=<callable>, arity=<K>`` — arbitrary K-ary callable with
           signature ``fn(scene, e_1, ..., e_K) -> float`` (already in [0,1]).
           Only ``anchor.third_person.<name>[i_1, ..., i_K]`` is wired
           (arity must be ``>= 2``).  No sigmoid wrapping is applied.

        Exactly one of ``h_r`` / ``angle_deg`` / ``fn`` must be provided.

        Each bracket index resolves to ``scene.objects[i]`` if ``i < K`` else
        ``scene.cameras[i - K]``, where ``K = len(scene.objects)``.

        Raises
        ------
        ValueError
            If ``name`` is empty / collides with a built-in directional or
            vertical label, multiple forms are provided, ``fn`` is given
            without ``arity >= 2``, or the form arguments are inconsistent.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("register_predicate: name must be a non-empty string")
        canon = name.strip().lower().replace("_", "-")

        forms = [k for k, v in (("h_r", h_r), ("angle_deg", angle_deg), ("fn", fn))
                 if v is not None]
        if len(forms) != 1:
            raise ValueError(
                f"register_predicate: exactly one of h_r=, angle_deg=, fn= must "
                f"be provided (got {len(forms)}: {forms})"
            )
        form = forms[0]

        if form == "h_r":
            if not callable(h_r):
                raise ValueError("register_predicate: h_r must be callable")
        elif form == "angle_deg":
            try:
                angle_val = float(angle_deg)
            except (TypeError, ValueError):
                raise ValueError(
                    f"register_predicate: angle_deg must be a number, got "
                    f"{angle_deg!r}"
                )
        else:  # form == "fn"
            if not callable(fn):
                raise ValueError("register_predicate: fn must be callable")
            if not isinstance(arity, int) or arity < 2:
                raise ValueError(
                    f"register_predicate: fn form requires arity int >= 2 "
                    f"(arity=1 cases go via h_r/angle_deg + first_person), "
                    f"got arity={arity!r}"
                )

        # Collision check vs built-in horizontal vocabulary
        # (front/back/left/right/diagonals/cardinals).
        try:
            FrameNamespace._resolve_direction_angle(canon)
            collides = True
        except ValueError:
            collides = False
        if collides:
            raise ValueError(
                f"register_predicate: name {name!r} (canonical {canon!r}) "
                f"collides with a built-in direction label; choose another name."
            )
        # Vertical labels live outside _resolve_direction_angle.
        if canon in ("above", "below"):
            raise ValueError(
                f"register_predicate: {canon!r} is a built-in vertical label; "
                f"choose another name."
            )

        if form == "h_r":
            self._user_predicates[canon] = {
                "kind": "h_r",
                "h_r": h_r,
                "margin": float(margin),
                "temperature": float(temperature),
                "normalize_distance": bool(normalize_distance),
            }
        elif form == "angle_deg":
            self._user_predicates[canon] = {
                "kind": "angle",
                "angle_deg": angle_val,
                "margin": float(margin),
                "temperature": float(temperature),
                "normalize_distance": bool(normalize_distance),
            }
        else:
            self._user_predicates[canon] = {
                "kind": "fn", "fn": fn, "arity": int(arity),
            }


    def __getattr__(self, name: str):
        """Trap ``scene.obj_<name>`` for registered h_r / angle_deg predicates.

        Implements the paper's S^{obj}_r[i, j] = S^{a_j}_r[i, j] access path
        for user-registered predicates, mirroring built-in
        ``scene.obj_left/right/front/behind``.  Only fires for names that
        (a) start with ``obj_``, (b) have a non-underscore suffix that maps
        to a registered ``h_r`` or ``angle`` predicate.  All other lookups
        fall through to the default ``AttributeError`` — preserving normal
        introspection behavior for typos and dunder names.
        """
        # Python invokes __getattr__ only after normal lookup fails, so the
        # built-in @property obj_left / obj_right / obj_front / obj_behind
        # never trigger this path.  We guard against recursion by reading
        # ``_user_predicates`` via ``object.__getattribute__``.
        if name.startswith("_"):
            raise AttributeError(name)
        if not name.startswith("obj_"):
            import difflib
            public = [n for n in dir(type(self)) if not n.startswith("_")]
            close = difflib.get_close_matches(name, public, n=3, cutoff=0.6)
            hint = (" Did you mean: " + ", ".join(f"scene.{c}" for c in close) + "?") if close else ""
            raise AttributeError(f"scene.{name} does not exist.{hint}")
        suffix = name[len("obj_"):]
        if not suffix:
            raise AttributeError(name)
        canon = suffix.strip().lower().replace("_", "-")
        try:
            user = object.__getattribute__(self, "_user_predicates")
        except AttributeError:
            raise AttributeError(name)
        if canon not in user:
            raise AttributeError(name)
        spec = user[canon]
        if spec["kind"] not in ("h_r", "angle"):
            raise AttributeError(
                f"scene.{name}: registered predicate {canon!r} is fn-form "
                f"(arity={spec.get('arity')!r}); obj-centric access is only "
                f"defined for h_r / angle_deg forms.  Use "
                f"view.third_person.{canon}[...] instead."
            )
        # Local import to avoid a circular import at module load time.
        from saturn.predicates.registry import _RegisteredObjCentricWrapper
        return _RegisteredObjCentricWrapper(self, spec, canon)


    def _compute_scene_scale(self) -> float:
        """Return the paper's ``s_scene`` (quantile_0.9 of pairwise distances).

        Lazily computed and cached.  Falls back to ``1.0`` for trivial scenes
        (fewer than 2 entities) to avoid division-by-zero downstream.
        """
        if self._scene_scale_cache is not None:
            return self._scene_scale_cache

        positions: List[np.ndarray] = []
        for obj in self.objects:
            if hasattr(obj, "center_world") and obj.center_world is not None:
                positions.append(np.asarray(obj.center_world, dtype=float))
        for cam in self.cameras:
            pos = getattr(cam, "pos", None)
            if pos is None:
                pos = getattr(cam, "position", None)
            if pos is not None:
                positions.append(np.asarray(pos, dtype=float))

        if len(positions) < 2:
            self._scene_scale_cache = 1.0
            return 1.0

        P = np.stack(positions, axis=0)  # (N, 3)
        diffs = P[:, None, :] - P[None, :, :]  # (N, N, 3)
        dists = np.linalg.norm(diffs, axis=-1)  # (N, N)
        iu = np.triu_indices_from(dists, k=1)
        flat = dists[iu]
        if flat.size == 0:
            self._scene_scale_cache = 1.0
        else:
            s = float(np.quantile(flat, 0.9))
            self._scene_scale_cache = s if s > 1e-9 else 1.0
        return self._scene_scale_cache


    # 8-way cardinal labels in hyphenated form, in clockwise order starting at
    # north.  Used as keys for :meth:`score_cardinals`.
    _CARDINAL_LABELS_8_HYPHENATED = (
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    )


    def centroid(self, anchors) -> np.ndarray:
        """Mean world-space position of one or more anchor entities.

        Used for "abstract location" semantics: callers pass the anchor list
        directly to ``centroid`` to obtain a single point.  The point can
        then feed ``scene.frame(at=...)`` or relational queries.

        Parameters
        ----------
        anchors : iterable
            Each element may be a ``Camera`` / ``MergedObject`` instance,
            an int (object index), a ``("camera", i)`` / ``("object", i)``
            tuple, a string label (resolved via ``self.objects_by_label``),
            or a 3D vector (used verbatim).

        Returns
        -------
        np.ndarray
            ``[x, y, z]`` mean position.  Single-element input returns that
            entity's position.
        """
        if anchors is None:
            raise ValueError("centroid(): anchors is None.")
        # Wrap a bare entity / vector / label as a 1-element list.
        if isinstance(anchors, (Camera, MergedObject, str, int, np.integer)):
            anchors = [anchors]
        elif isinstance(anchors, np.ndarray):
            if anchors.ndim == 1:
                anchors = [anchors]
        elif isinstance(anchors, tuple) and len(anchors) == 2 and isinstance(anchors[0], str):
            anchors = [anchors]

        pts: List[np.ndarray] = []
        for a in anchors:
            if isinstance(a, (np.ndarray, list)):
                arr = np.asarray(a, dtype=float).ravel()[:3]
                if arr.size != 3:
                    raise ValueError(
                        f"centroid(): bare vector must be length 3, got {arr.size}"
                    )
                pts.append(arr)
                continue
            pos, _, _ = self._resolve_entity(a, label="anchor")
            if pos is None:
                raise ValueError(
                    f"centroid(): anchor {a!r} has no resolvable position."
                )
            pts.append(np.asarray(pos, dtype=float).ravel()[:3])

        if not pts:
            raise ValueError("centroid(): empty anchors list.")
        return np.mean(np.stack(pts, axis=0), axis=0)

    # ------------------------------------------------------------------
    # Frame-independent relations
    # ------------------------------------------------------------------

    def _ensure_frame_independent(self):
        if self._frame_independent_cache is not None:
            return
        positions = self._entity_positions()
        fronts = self._entity_front_directions()
        self._frame_independent_cache = compute_frame_independent_relations(
            positions, fronts
        )

    @property
    def facing(self) -> ProbabilisticTensor:
        self._ensure_frame_independent()
        return self._wrap_tensor(self._frame_independent_cache["facing"], ndim=2)

    @property
    def parallel(self) -> ProbabilisticTensor:
        self._ensure_frame_independent()
        return self._wrap_tensor(self._frame_independent_cache["parallel"], ndim=2)

    @property
    def perpendicular(self) -> ProbabilisticTensor:
        self._ensure_frame_independent()
        return self._wrap_tensor(self._frame_independent_cache["perpendicular"], ndim=2)

    @property
    def orientation_distance(self) -> ProbabilisticTensor:
        self._ensure_frame_independent()
        return self._wrap_tensor(
            self._frame_independent_cache["orientation_distance"], ndim=2
        )

    @property
    def between(self) -> ProbabilisticTensor:
        self._ensure_frame_independent()
        return self._wrap_tensor(self._frame_independent_cache["between"], ndim=3)

    # ------------------------------------------------------------------
    # Object-centric (frame-independent) directional relations
    # ------------------------------------------------------------------
    #
    # ``obj_left/right/front/behind`` are entity-space (N+C, N+C) tensors
    # whose semantics use **each entity's own intrinsic axes**:
    #
    #     obj_left[i, j]   = "i is to the left of j, from j's perspective"
    #     obj_right[i, j]  = "i is to the right of j, from j's perspective"
    #     obj_front[i, j]  = "i is in front of j, from j's perspective"
    #     obj_behind[i, j] = "i is behind j, from j's perspective"
    #
    # No external reference frame (camera, world) is needed — that is what
    # makes them "frame independent".  Cameras are first-class anchors:
    # their forward axis is the view direction, so e.g. ``obj_behind[i,
    # cam]`` answers "is i behind the camera?".

    def _ensure_obj_relative(self):
        if self._obj_relative_cache is not None:
            return
        positions = self._entity_positions()
        fronts = self._entity_front_directions()
        rights = self._entity_right_directions()
        self._obj_relative_cache = compute_obj_relative_from_axes(
            positions, fronts, rights
        )

    @property
    def obj_left(self) -> ProbabilisticTensor:
        """``obj_left[i, j]`` = "i is to the left of j, from j's perspective"."""
        self._ensure_obj_relative()
        return self._wrap_tensor(self._obj_relative_cache["obj_left"], ndim=2)

    @property
    def obj_right(self) -> ProbabilisticTensor:
        """``obj_right[i, j]`` = "i is to the right of j, from j's perspective"."""
        self._ensure_obj_relative()
        return self._wrap_tensor(self._obj_relative_cache["obj_right"], ndim=2)

    @property
    def obj_front(self) -> ProbabilisticTensor:
        """``obj_front[i, j]`` = "i is in front of j, from j's perspective"."""
        self._ensure_obj_relative()
        return self._wrap_tensor(self._obj_relative_cache["obj_front"], ndim=2)

    @property
    def obj_behind(self) -> ProbabilisticTensor:
        """``obj_behind[i, j]`` = "i is behind j, from j's perspective"."""
        self._ensure_obj_relative()
        return self._wrap_tensor(self._obj_relative_cache["obj_behind"], ndim=2)

    # ------------------------------------------------------------------
    # Camera-0 frame convenience shortcuts
    # ------------------------------------------------------------------
    # These delegate to self._frame(at=cameras[0]) so that generated programs
    # can write ``left[i, j]`` directly without building an explicit frame.

    @property
    def _cam0_frame(self) -> "FrameNamespace":
        if not self.cameras:
            raise AttributeError(
                "scene.left/right/front/behind/above/below require at least one camera"
            )
        # One FrameNamespace per scene state, so the shortcuts below share its
        # lazily computed relation tensors. Cleared by _invalidate_caches.
        fr = self._frame_cache.get("cam0")
        if fr is None:
            fr = self._frame_cache["cam0"] = self._frame(at=self.cameras[0])
        return fr

    @property
    def left(self): return self._cam0_frame.left
    @property
    def right(self): return self._cam0_frame.right
    @property
    def front(self): return self._cam0_frame.front
    @property
    def behind(self): return self._cam0_frame.behind
    @property
    def above(self): return self._cam0_frame.above
    @property
    def below(self): return self._cam0_frame.below
    @property
    def left_normalized(self): return self._cam0_frame.left_normalized
    @property
    def right_normalized(self): return self._cam0_frame.right_normalized
    @property
    def front_normalized(self): return self._cam0_frame.front_normalized
    @property
    def behind_normalized(self): return self._cam0_frame.behind_normalized
    @property
    def above_normalized(self): return self._cam0_frame.above_normalized
    @property
    def below_normalized(self): return self._cam0_frame.below_normalized
    @property
    def obj_facing_left(self): return self._cam0_frame.obj_facing_left
    @property
    def obj_facing_right(self): return self._cam0_frame.obj_facing_right
    @property
    def obj_facing_front(self): return self._cam0_frame.obj_facing_front
    @property
    def obj_facing_back(self): return self._cam0_frame.obj_facing_back
    @property
    def obj_facing_up(self): return self._cam0_frame.obj_facing_up
    @property
    def obj_facing_down(self): return self._cam0_frame.obj_facing_down

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def add_camera(self, cam: Camera) -> int:
        """Add a camera (or virtual camera from object.clone()) to the scene.

        Returns the new camera index. Assigns entity_id and camera id.
        Invalidates cached relations so they are recomputed on next access.
        """
        new_cam_idx = len(self.cameras)
        cam.id = new_cam_idx
        cam.entity_id = len(self.objects) + new_cam_idx
        # Ensure position/heading are derived if not set
        if cam.position_world is None:
            cam.position_world = cam._extract_position()
        if cam.heading is None:
            cam.heading = cam._extract_heading()

        cam._scene = self
        self.cameras.append(cam)
        self._invalidate_caches()
        return new_cam_idx

    # ------------------------------------------------------------------
    # Dynamic detection
    # ------------------------------------------------------------------

    def detect(self, description: str, camera: Optional[int] = None, unique: bool = False) -> List[int]:
        """Detect new objects by description across all views or one view.

        Runs SAM3 on all views by default, or only the specified input camera
        when ``camera`` is provided. Back-projects, merges, and appends the
        detected objects to ``self.objects``.
        ``unique``: the question names ONE such object, so the most confident
        box of each view is fused into one track (fusion.split_unique_track).
        Returns list of new object indices.
        """
        if self._detect_fn is None:
            raise RuntimeError(
                "scene.detect() is not available. "
                "the scene loader must provide a detect_fn callback."
            )
        if camera is not None:
            if not isinstance(camera, int):
                raise ValueError("camera must be an int or None.")
            if camera < 0 or camera >= len(self.images):
                raise ValueError(
                    f"camera={camera} out of range for {len(self.images)} input images."
                )
        new_indices = self._detect_fn(self, description, camera=camera, unique=unique)
        self._invalidate_caches()
        return new_indices

    def ground(
        self, description: str, vlm=None, camera: Optional[int] = None, unique: bool = False,
    ) -> List[int]:
        """Locate objects by VLM bounding-box grounding (candidate generation).

        Unlike ``detect()`` which runs SAM3, this calls the VLM's ``ground()``
        method on each view to produce bounding boxes, then back-projects them
        to 3D.  Use this when the object may not be in the SAM3 detection
        results but is visible in the images.

        Parameters
        ----------
        description : str
            Object description to locate (e.g. "door", "balcony").
        vlm : Agent, optional
            VLM with ``ground(image, phrase)`` method.  If not provided,
            uses ``self._vlm`` (set by the scene loader).
        camera : int, optional
            If provided, only run VLM grounding on the specified view.
            ``None`` (default) runs across all views.
        unique : bool
            The question names ONE such object, as in ``detect()``: its box in
            each view is fused into one track (fusion.split_unique_track).

        Returns
        -------
        list[int]
            Indices of newly appended objects.
        """
        vlm = vlm or getattr(self, "_vlm", None)
        if not hasattr(self, "ground_fn") or self.ground_fn is None:
            raise RuntimeError(
                "scene.ground() is not available. "
                "the scene loader must provide a ground_fn callback."
            )
        if camera is not None:
            if not isinstance(camera, int):
                raise ValueError("camera must be an int or None.")
            if camera < 0 or camera >= len(self.images):
                raise ValueError(
                    f"camera={camera} out of range for {len(self.images)} input images."
                )
        new_indices = self.ground_fn(self, description, vlm, camera=camera, unique=unique)
        return new_indices

    # ------------------------------------------------------------------
    # Unified direction API
    # ------------------------------------------------------------------

    @property
    def up(self) -> np.ndarray:
        """World up direction. The reconstructed world is Y-up: ``(0, 1, 0)``."""
        return np.array([0.0, 1.0, 0.0])

    def vector(self, from_entity, to_entity=None, *rest) -> np.ndarray:
        """Return the normalized 3D direction vector from one entity to another.

        ``scene.vector(x, y, z)`` with three numbers is the plain 3-vector
        ``np.array([x, y, z])`` (not normalized).

        Parameters
        ----------
        from_entity, to_entity : entity specifiers
            Each can be:
              - a ``Camera`` or ``MergedObject`` instance
              - ``("camera", cam_id)`` or ``("cam", cam_id)``
              - ``("object", obj_id)`` or ``("obj", obj_id)``
              - ``int`` — object index
              - ``np.ndarray`` or list — 3D world point

        Returns
        -------
        np.ndarray — unit-length 3D vector (or zero vector if positions coincide).
        """
        if rest or to_entity is None:
            comps = (from_entity, to_entity) + rest
            if len(comps) == 3 and all(isinstance(c, (int, float, np.number)) and not isinstance(c, bool)
                                       for c in comps):
                return np.array([float(c) for c in comps])
            raise TypeError("scene.vector(from, to) is the unit direction between two entities; "
                            "scene.vector(x, y, z) builds a 3-vector from three numbers.")
        src_pos, _, _ = self._resolve_entity(from_entity, label="from")
        tgt_pos, _, _ = self._resolve_entity(to_entity, label="to")
        diff = tgt_pos - src_pos
        n = np.linalg.norm(diff)
        if n < 1e-12:
            return np.zeros(3, dtype=float)
        return diff / n


    # -- cardinal helpers (internal) ------------------------------------
    # Constants imported from direction_utils (single source of truth).
    _CARDINAL_TO_ANGLE = CARDINAL_TO_ANGLE
    _CARDINAL_LABELS_8 = CARDINAL_LABELS_8
    _CARDINAL_LABELS_4 = CARDINAL_LABELS_4
    _RELATIVE_LABELS_8 = RELATIVE_LABELS_8
    _RELATIVE_LABELS_4 = RELATIVE_LABELS_4


    # ------------------------------------------------------------------
    # match_* convenience methods  (MCQ option matching)
    # ------------------------------------------------------------------

    # Synonym table for normalizing direction / option labels
    _DIRECTION_SYNONYMS = {
        "behind": "back",
        "rear": "back",
        "backward": "back",
        "backwards": "back",
        "forward": "front",
        "forwards": "front",
        "ahead": "front",
        "directly ahead": "front",
        "directly in front": "front",
        "straight ahead": "front",
        "directly to the right": "right",
        "directly to the left": "left",
        "immediate left": "left",
        "immediate right": "right",
        "to my immediate left": "left",
        "to my immediate right": "right",
        "to my left": "left",
        "to my right": "right",
        "left front": "front left",
        "right front": "front right",
        "left rear": "back left",
        "right rear": "back right",
        "left back": "back left",
        "right back": "back right",
        "my left front": "front left",
        "my right front": "front right",
        "my left rear": "back left",
        "my right rear": "back right",
    }

    @staticmethod
    def _normalize_label(s: str) -> str:
        """Lowercase, strip, replace hyphens with spaces, apply synonyms."""
        s = s.lower().strip().replace("-", " ").replace("_", " ")
        # Remove leading articles / filler
        for prefix in ("to my ", "to the ", "on the ", "on my "):
            if s.startswith(prefix):
                s = s[len(prefix) :]
        s = s.strip()
        # Apply synonym table
        if s in Scene._DIRECTION_SYNONYMS:
            return Scene._DIRECTION_SYNONYMS[s]
        return s

    @staticmethod
    def _best_option(computed: str, options: Dict[str, str]) -> str:
        """Match *computed* label against option dict, returning the best letter.

        Matching rules (tried in order):
        1. Exact match after normalization.
        2. Substring containment (computed in option OR option in computed).
        3. Shared-tokens: option with the most overlapping words wins.
        4. First option as fallback.
        """
        norm = Scene._normalize_label(computed)
        norm_opts = {k: Scene._normalize_label(v) for k, v in options.items()}

        # 1. Exact
        for k, nv in norm_opts.items():
            if norm == nv:
                return k

        # 2. Substring containment
        for k, nv in norm_opts.items():
            if norm in nv or nv in norm:
                return k

        # 3. Token overlap
        norm_tokens = set(norm.split())
        best_k, best_overlap = None, 0
        for k, nv in norm_opts.items():
            overlap = len(norm_tokens & set(nv.split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_k = k
        if best_k is not None and best_overlap > 0:
            return best_k

        # 4. Fallback: first option
        return next(iter(options))

    # Sets used by match_direction's auto-detect to classify option labels as
    # cardinal vs relative.
    _CARDINAL_WORDS = frozenset(
        {
            "north",
            "south",
            "east",
            "west",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
            "ne",
            "nw",
            "se",
            "sw",
            "n",
            "s",
            "e",
            "w",
        }
    )
    _RELATIVE_WORDS = frozenset(
        {
            "front",
            "back",
            "behind",
            "left",
            "right",
            "front right",
            "front left",
            "back right",
            "back left",
            "front-right",
            "front-left",
            "back-right",
            "back-left",
        }
    )



    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _rebuild_anchor_predicates(self) -> None:
        """Re-stamp anchor indices and recompute the (K+C, K+C) predicate matrix.

        The matrix is built once in __init__; objects appended later by
        detect()/ground() (``_append_merged_objects``) need ``anchor_index``
        and ``_scene`` stamped and a matrix of the new size.
        Called from every cache invalidation so the matrix tracks the entity
        list and per-object pose edits (constraint.face etc.).
        """
        K = len(self.objects)
        for i, obj in enumerate(self.objects):
            obj.anchor_index = i
            obj._scene = self
        for c, cam in enumerate(self.cameras):
            cam.anchor_index = K + c
            cam._scene = self
        try:
            from .anchor import compute_anchor_predicates
            self._anchor_predicates = compute_anchor_predicates(
                self.objects + self.cameras, strict=False,
            )
        except Exception as e:  # never leave a stale matrix behind
            log.warning(f"[scene] anchor predicate rebuild failed ({e}); falling back to lazy compute")
            self._anchor_predicates = None

    def _invalidate_caches(self):
        """Invalidate all cached relation tensors and rebuild anchor predicates.

        Called after add_camera(), detect(), or any structural mutation.
        """
        self._frame_cache.clear()
        self._scene_scale_cache = None
        self._distance_cache = None
        self._frame_independent_cache = None
        self._obj_relative_cache = None
        self._rebuild_anchor_predicates()

    # ------------------------------------------------------------------
    # Non-Maximum Suppression for duplicate objects
    # ------------------------------------------------------------------

    def nms_objects(
        self,
        iou_threshold: float = 0.5,
        label_similarity_threshold: float = 0.4,
        verbose: bool = False,
    ) -> int:
        """Remove duplicate objects with similar labels and overlapping bboxes.

        For each pair of objects, if their labels are similar (by token overlap)
        AND their per-view bboxes overlap above ``iou_threshold`` on at least
        one shared view, keep the one with higher average detection score and
        remove the other.

        Returns the number of objects removed.
        """
        import re as _re

        def _tokenize(label: str) -> set:
            """Extract meaningful tokens from a label."""
            label = _re.sub(r'_\d+$', '', label)  # strip "_0", "_1" suffixes
            tokens = set(_re.findall(r'[a-z]{3,}', label.lower()))
            # Remove common stopwords
            tokens -= {
                'the', 'and', 'with', 'for', 'from', 'that', 'this',
                'area', 'containing', 'located', 'visible', 'shown',
            }
            return tokens

        def _label_similar(a: str, b: str) -> bool:
            """Check if two labels refer to the same object."""
            ta, tb = _tokenize(a), _tokenize(b)
            if not ta or not tb:
                return False
            # Jaccard similarity on tokens
            intersection = ta & tb
            union = ta | tb
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= label_similarity_threshold:
                return True
            # Also check substring: one label contained in the other
            a_clean = _re.sub(r'_\d+$', '', a.lower().strip())
            b_clean = _re.sub(r'_\d+$', '', b.lower().strip())
            if a_clean in b_clean or b_clean in a_clean:
                return True
            return False

        def _bbox_iou(a, b):
            """Compute IoU between two [x1,y1,x2,y2] bboxes."""
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0

        def _avg_score(obj):
            scores = getattr(obj, 'per_view_scores', {})
            vals = [float(v) for v in scores.values() if v is not None]
            return sum(vals) / len(vals) if vals else 0

        n = len(self.objects)
        to_remove = set()

        # Greedy NMS: visit objects best-score first (stable, so ties keep the
        # lower index) and let only surviving objects suppress later ones --
        # an already-removed box must not delete a box the keeper never
        # overlapped.
        order = sorted(range(n), key=lambda k: -_avg_score(self.objects[k]))
        for a, i in enumerate(order):
            if i in to_remove:
                continue
            for j in order[a + 1:]:
                if j in to_remove:
                    continue
                obj_i = self.objects[i]
                obj_j = self.objects[j]
                label_i = getattr(obj_i, 'label', '')
                label_j = getattr(obj_j, 'label', '')

                if not _label_similar(label_i, label_j):
                    continue

                # Check bbox IoU on shared views
                bboxes_i = getattr(obj_i, 'per_view_bboxes', {})
                bboxes_j = getattr(obj_j, 'per_view_bboxes', {})
                shared_views = set(bboxes_i.keys()) & set(bboxes_j.keys())

                max_iou = 0
                for v in shared_views:
                    iou = _bbox_iou(bboxes_i[v], bboxes_j[v])
                    max_iou = max(max_iou, iou)

                if max_iou >= iou_threshold:
                    # j scores no higher than i (visit order): remove j.
                    to_remove.add(j)
                    if verbose:
                        log.debug(
                            f"  NMS: removing obj[{j}] '{label_j}' "
                            f"(score={_avg_score(obj_j):.3f}) — "
                            f"duplicate of obj[{i}] '{label_i}' "
                            f"(score={_avg_score(obj_i):.3f}, IoU={max_iou:.2f})"
                        )

        if to_remove:
            self.objects = [
                obj for idx, obj in enumerate(self.objects)
                if idx not in to_remove
            ]
            # Re-assign IDs
            for idx, obj in enumerate(self.objects):
                obj.id = idx

        if to_remove:
            self._invalidate_caches()  # ids changed: re-stamp anchors + rebuild predicates
        return len(to_remove)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def objects_count(self) -> int:
        return len(self.objects)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    def __repr__(self) -> str:
        return (
            f"Scene(objects={len(self.objects)}, "
            f"cameras={len(self.cameras)}, "
            f"images={len(self.images)})"
        )

    # ------------------------------------------------------------------
    # Serialization (JSON dump / load)
    # ------------------------------------------------------------------

    def to_dict(
        self, *, include_metadata: bool = True, include_points: bool = False
    ) -> Dict[str, Any]:
        return scene_to_dict(self, include_metadata=include_metadata, include_points=include_points)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        images: Optional[List[Any]] = None,
        detect_fn: Optional[Callable] = None,
    ) -> "Scene":
        return scene_from_dict(cls, data, images=images, detect_fn=detect_fn)

    def dump(
        self,
        filepath: str,
        *,
        include_metadata: bool = True,
        include_points: bool = False,
    ) -> None:
        return scene_dump(self, filepath, include_metadata=include_metadata, include_points=include_points)

    @classmethod
    def load(
        cls,
        filepath: str,
        *,
        images: Optional[List[Any]] = None,
        detect_fn: Optional[Callable] = None,
    ) -> "Scene":
        return scene_load(cls, filepath, images=images, detect_fn=detect_fn)

