"""Soft scoring functions turning metrics into [0, 1] predicate scores (sigmoids, cones, ranks).

Pure geometry; must not import saturn.perception/vlm/serving.
"""
import math
from typing import Any, Dict, TYPE_CHECKING

import numpy as np

from saturn.log import get_logger
from saturn.scene.direction_utils import (
    CARDINAL_TO_ANGLE,
    RELATIVE_TO_CARDINAL,
    canonical_direction,
)
from saturn.scene.types import DirectionValue, MergedObject

if TYPE_CHECKING:
    from .frame import FrameNamespace

log = get_logger(__name__)


def steep_sigmoid_signed(values, midpoint: float = 0.0, steepness: float = 14.0):
    z = steepness * (values - midpoint)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def label_yaw(label) -> "float | None":
    """Yaw in degrees (0 = front/north, clockwise) of a direction label, or None.

    Alias-aware via :func:`canonical_direction`: ``"forward"``, ``"rear left"``,
    ``"behind-left"``, ``"north-east"``, ``"NE"`` all resolve.  Shared by the
    MCQ label matchers (``match`` / ``match_rotation`` / ``match_translation``).
    """
    canon = canonical_direction(str(label))
    if canon in CARDINAL_TO_ANGLE:
        return CARDINAL_TO_ANGLE[canon]
    if canon in RELATIVE_TO_CARDINAL:
        return CARDINAL_TO_ANGLE[RELATIVE_TO_CARDINAL[canon]]
    return None


def warn_unresolved_labels(caller: str, options: dict, yaws: dict) -> None:
    """Log option labels that :func:`label_yaw` could not resolve (scored worst)."""
    bad = [str(options[k]) for k, y in yaws.items() if y is None]
    if bad:
        log.warning("%s: unrecognised direction label(s) %s; scored as worst match", caller, bad)


# Relation-matrix key (``compute_frame_relations``) for each horizontal yaw
# of ``obj - ref`` in the frame, in 45 deg steps from +front, clockwise.
# The matrices use the occluder convention: ``front`` = closer to the
# observer (-front), ``behind`` = farther (+front).
_RELATION_KEYS_BY_FRAME_YAW = (
    "behind", "behind_right", "right", "front_right",
    "front", "front_left", "left", "behind_left",
)


def _relation_frame_yaw(direction: str) -> "float | None":
    """Frame yaw (0 = +front, clockwise, degrees) that ``direction`` points to.

    Relative labels follow the occluder convention of the relation matrices
    ("A in front of B" = A is closer to the observer, i.e. at -front), so
    their front/back component is mirrored.  Compass labels are directions
    in the frame (north = +front, e.g. a frame built facing north).  Returns
    None for labels that are neither (including above / below).
    """
    canon = canonical_direction(str(direction))
    if canon in RELATIVE_TO_CARDINAL:
        return (180.0 - CARDINAL_TO_ANGLE[RELATIVE_TO_CARDINAL[canon]]) % 360.0
    if canon in CARDINAL_TO_ANGLE:
        return float(CARDINAL_TO_ANGLE[canon])
    return None


def _relation_key(direction: str) -> str:
    """Relation-matrix key for ``direction`` (``"front_left"``, ``"above"``, ...).

    Axial and diagonal, relative and compass labels all use ONE convention
    (see :func:`_relation_frame_yaw`).  Unknown labels come back unchanged
    so the caller's lookup fails into its angle fallback / error path.
    """
    canon = canonical_direction(str(direction))
    if canon in ("above", "below"):
        return canon
    yaw = _relation_frame_yaw(direction)
    if yaw is None:
        return direction
    return _RELATION_KEYS_BY_FRAME_YAW[int(round(yaw / 45.0)) % 8]


def _entity_pose(entity) -> "tuple[np.ndarray, np.ndarray]":
    """Extract (position, rotation_world) from an entity for registered predicates.

    Handles MergedObject (``center_world`` / ``rotation_world``) and Camera
    (``position`` / ``orientation``).  Falls back to zeros / identity when an
    attribute is missing — keeps user-defined ``h_r`` callbacks robust against
    unoriented entities.
    """
    # Position: prefer .position (works for both MergedObject and Camera as
    # they both expose this alias), else .center_world, else .pos, else zeros.
    pos = (
        getattr(entity, "position", None)
        if getattr(entity, "position", None) is not None
        else getattr(entity, "center_world", None)
    )
    if pos is None:
        pos = getattr(entity, "pos", None)
    if pos is None:
        pos = np.zeros(3, dtype=float)
    pos = np.asarray(pos, dtype=float).reshape(3)

    R = getattr(entity, "rotation_world", None)
    if R is None:
        R = getattr(entity, "orientation", None)
    if R is None:
        R = np.eye(3, dtype=float)
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        R = np.eye(3, dtype=float)
    return pos, R


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for the paper's predicate score."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _compute_pairwise_score(
    view: "FrameNamespace",
    scene,
    spec: Dict[str, Any],
    i_idx: int,
    j_idx: int,
) -> float:
    """Compute S^a_r[i, j] for a registered ``h_r`` or ``angle`` predicate.

    Implements the paper's $S^a_r[i,j] = \\sigma((h_r - m_r)/\\tau_r)$ pipeline:

    1. Take the anchor's frame axes (``view._frame_right/up/front``) as the
       columns of $R_a$; pull positions and rotations from entities i, j.
    2. Build $\\Delta^a_{ij} = R_a^\\top (x_i - x_j)$ and
       $R^a_i = R_a^\\top R_i$, $R^a_j = R_a^\\top R_j$.
    3. Optionally normalize by ``s_scene`` (paper's scene-scale norm).
    4. Call ``h_r`` (or the directional sugar built from ``angle_deg``).
    5. Apply the sigmoid with the spec's ``margin`` and ``temperature``.
    """
    # R_a columns = anchor's body axes.
    R_a = np.column_stack(
        [view._frame_right, view._frame_up, view._frame_front]
    )  # (3, 3)
    R_a_T = R_a.T  # rows = local-frame axes expressed in world

    if i_idx is None:
        # first-person collapses i = target idx for the array build site;
        # the caller passes the target idx as i_idx.
        raise ValueError("_compute_pairwise_score: i_idx required")
    e_i = scene._entity_at(i_idx)

    if j_idx is None:
        # first_person convention: j = the anchor itself.  Reference position
        # is the anchor's frame_origin and R_j = R_a (so R_j_local = identity).
        p_j = np.asarray(view._frame_origin, dtype=float).reshape(3)
        R_j_world = R_a  # so R_j_local = R_a_T @ R_a = I (up to numerical noise)
    else:
        e_j = scene._entity_at(j_idx)
        p_j, R_j_world = _entity_pose(e_j)

    p_i, R_i_world = _entity_pose(e_i)

    delta_world = p_i - p_j  # (3,) world frame
    delta_local = R_a_T @ delta_world  # (3,) local frame
    R_i_local = R_a_T @ R_i_world  # (3, 3)
    R_j_local = R_a_T @ R_j_world  # (3, 3)

    if spec.get("normalize_distance", True):
        s_scene = scene._compute_scene_scale()
        if s_scene > 1e-9:
            delta_local = delta_local / s_scene

    if spec["kind"] == "h_r":
        evidence = float(spec["h_r"](delta_local, R_i_local, R_j_local))
    else:  # "angle"
        angle_rad = math.radians(spec["angle_deg"])
        target_x = math.sin(angle_rad)
        target_z = math.cos(angle_rad)
        # Paper's directional h_r: dot of delta_local with the target unit
        # direction in the (right, front) horizontal plane.
        evidence = float(delta_local[0] * target_x + delta_local[2] * target_z)

    m_r = float(spec.get("margin", 0.05))
    tau_r = float(spec.get("temperature", 0.03))
    if tau_r <= 1e-12:
        tau_r = 1e-12  # avoid division-by-zero; effectively a step function
    return _sigmoid((evidence - m_r) / tau_r)


class _DirectionScoringMixin:
    @staticmethod
    def _to_float(val) -> float:
        """Extract a plain Python float from a ProbabilisticTensor, torch.Tensor, or number."""
        # Import here to avoid circular imports
        from saturn.soft_logic.tensor import ProbabilisticTensor as PT

        if isinstance(val, PT):
            val = val.tensor
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)


    def _direction_score_from_origin(self, obj_idx: int, direction: str) -> float:
        """Compute a [0, 1] score for how much object obj_idx is in the given
        direction from the frame origin.

        Supports all 4-way and 8-way directions:
          - Relative: front, behind, left, right, front-left, front-right,
            back-left, back-right
          - Cardinal: north, south, east, west, northeast, northwest,
            southeast, southwest (and abbreviations)

        Uses generalised cosine scoring:
          score = (1 + cos(yaw - target_angle)) / 2
        which peaks at 1.0 when the object is exactly in the stated
        direction and drops to 0.0 when it's 180° away.
        """
        target_pos = self._scene.objects[obj_idx].center_world
        vec = np.asarray(target_pos, dtype=float) - self._frame_origin
        r = np.dot(vec, self._frame_right)
        f = np.dot(vec, self._frame_front)
        horiz = math.sqrt(r * r + f * f)
        if horiz < 1e-12:
            return 0.0

        yaw_rad = math.atan2(r, f)  # 0 = front, CW positive

        # Resolve the direction label to a target angle in degrees.
        # Convention: 0 = front/north, 90 = right/east, 180 = behind/south, 270 = left/west
        target_deg = self._resolve_direction_angle(direction)

        target_rad = math.radians(target_deg)
        return (1.0 + math.cos(yaw_rad - target_rad)) / 2.0


    @staticmethod
    def _resolve_direction_angle(direction: str) -> float:
        """Convert any direction label to a yaw angle in degrees.

        Convention: 0 = front/north, 90 = right/east, 180 = behind/south,
        270 = left/west.  Works for both relative and cardinal labels,
        4-way and 8-way.
        """
        d = direction.strip().lower().replace("_", "-")

        # Direct cardinal lookup (north, northeast, n, ne, ...)
        if d in CARDINAL_TO_ANGLE:
            return CARDINAL_TO_ANGLE[d]
        # Hyphenated cardinal form ("north-east" → "northeast").  The public
        # 8-way label set used by Scene.score_cardinals is hyphenated, but
        # CARDINAL_TO_ANGLE keys are not.  Reconcile here so callers can use
        # either form interchangeably.
        d_nohyphen_card = d.replace("-", "")
        if d_nohyphen_card in CARDINAL_TO_ANGLE:
            return CARDINAL_TO_ANGLE[d_nohyphen_card]

        # Relative label → cardinal → angle.
        # Aliases: "forward"/"forwards" → "front", "behind"/"behinds" → "back",
        # plus diagonal forms "behind-left"/"behind-right" → "back-left"/"back-right".
        # The "behind" alias is accepted on ``first_person`` (this resolver's
        # caller) where the semantic is unambiguous: ``first_person.behind[k]``
        # means "object k is in the back direction from the frame origin" — same
        # as ``first_person.back[k]``. The 2D pairwise predicate
        # ``view.behind[i, j]`` ("i is occluded by j") lives on a separate
        # namespace and does not use this resolver.  "back" is the canonical name.
        if d in ("forward", "forwards"):
            d = "front"
        if d in ("behind", "behinds"):
            d = "back"
        if d == "behind-left":
            d = "back-left"
        if d == "behind-right":
            d = "back-right"
        if d in RELATIVE_TO_CARDINAL:
            return CARDINAL_TO_ANGLE[RELATIVE_TO_CARDINAL[d]]

        # Try without hyphens (e.g. "frontleft" → "front-left", "behindleft" → "back-left")
        d_nohyphen = d.replace("-", "").replace(" ", "")
        if d_nohyphen == "behindleft":
            d_nohyphen = "backleft"
        elif d_nohyphen == "behindright":
            d_nohyphen = "backright"
        for rel_key, card_key in RELATIVE_TO_CARDINAL.items():
            if rel_key.replace("-", "").replace(" ", "") == d_nohyphen:
                return CARDINAL_TO_ANGLE[card_key]

        raise ValueError(
            f"Unknown direction: {direction}. "
            f"Use front/back (or behind)/left/right (+ diagonals) or "
            f"north/south/east/west (+ diagonals)."
        )


    def _to_score_vector(self, entity_or_score) -> list:
        """Convert an entity reference to a list of per-object identity scores.

        Accepted forms:
          - ProbabilisticTensor / tensor  — subscript with [i]
          - list/tuple of ints            — 1.0 for each member index, 0.0 elsewhere
          - int                           — 1.0 at that index, 0.0 elsewhere
          - MergedObject                  — 1.0 at that object's index, 0.0 elsewhere
          - ("camera", N) / ("cam", N)    — 1.0 at the nearest object to the camera
        """
        N = len(self._scene.objects)

        # List/tuple of indices — group of objects.
        if isinstance(entity_or_score, (list, tuple)) and (
            len(entity_or_score) == 0
            or all(
                isinstance(e, (int, np.integer))
                for e in entity_or_score
            )
        ):
            scores = [0.0] * N
            for idx in entity_or_score:
                idx = int(idx)
                if 0 <= idx < N:
                    scores[idx] = 1.0
            return scores

        if isinstance(entity_or_score, (int, np.integer)):
            scores = [0.0] * N
            idx = int(entity_or_score)
            if 0 <= idx < N:
                scores[idx] = 1.0
            return scores

        if isinstance(entity_or_score, MergedObject):
            scores = [0.0] * N
            for i, obj in enumerate(self._scene.objects):
                if obj is entity_or_score:
                    scores[i] = 1.0
                    break
            return scores

        # Camera tuple: ("camera", idx) or ("cam", idx) — resolve to
        # the camera's world position and assign 1.0 to the nearest object.
        if (
            isinstance(entity_or_score, (tuple, list))
            and len(entity_or_score) == 2
            and isinstance(entity_or_score[0], str)
            and entity_or_score[0].lower() in ("camera", "cam")
        ):
            cam_pos, _, _ = self._scene._resolve_entity(entity_or_score, label="camera")
            cam_pos = np.asarray(cam_pos, dtype=float)
            best_i, best_dist = 0, float("inf")
            for i, obj in enumerate(self._scene.objects):
                d = float(
                    np.linalg.norm(np.asarray(obj.center_world, dtype=float) - cam_pos)
                )
                if d < best_dist:
                    best_dist = d
                    best_i = i
            scores = [0.0] * N
            if N > 0:
                scores[best_i] = 1.0
            return scores

        # Default: ProbabilisticTensor or array-like — extract per-element
        return [self._to_float(entity_or_score[i]) for i in range(N)]


    def _direction_score_relative(
        self, obj_idx: int, ref_idx: int, direction: str
    ) -> float:
        """Score how much object obj_idx is in the given direction relative to
        ref_idx, using the generalised cosine formula.

        This works for any direction label (4-way or 8-way, cardinal or relative)
        and uses the same front/back convention as the relation matrices (see
        :func:`_relation_frame_yaw`).  Used as a fallback when the pre-computed
        relation matrices don't cover the direction (e.g. normalised diagonals).
        """
        obj_pos = np.asarray(self._scene.objects[obj_idx].center_world, dtype=float)
        ref_pos = np.asarray(self._scene.objects[ref_idx].center_world, dtype=float)
        vec = obj_pos - ref_pos
        r = np.dot(vec, self._frame_right)
        f = np.dot(vec, self._frame_front)
        horiz = math.sqrt(r * r + f * f)
        if horiz < 1e-12:
            return 0.0
        yaw_rad = math.atan2(r, f)
        target_deg = _relation_frame_yaw(direction)
        if target_deg is None:
            target_deg = self._resolve_direction_angle(direction)  # raises for unknown labels
        target_rad = math.radians(target_deg)
        return (1.0 + math.cos(yaw_rad - target_rad)) / 2.0


    def _best_match_origin(self, direction: str, options: dict) -> str:
        """Internal: best MCQ option for ``direction`` from frame origin.

        Public callers should use :meth:`best_match` (no ``reference=``).
        """
        N = len(self._scene.objects)
        best_letter = None
        best_combined = -1.0

        for letter, score_tensor in options.items():
            for i in range(N):
                identity = self._to_float(score_tensor[i])
                dir_score = self._direction_score_from_origin(i, direction)

                combined = identity * dir_score
                if combined > best_combined:
                    best_combined = combined
                    best_letter = letter

        return best_letter if best_letter is not None else list(options.keys())[0]


    def _best_match_relative(
        self,
        direction: str,
        reference_score,
        options: dict,
        nearest: bool = False,
    ) -> str:
        """Internal: best MCQ option for ``direction`` relative to a reference.

        Public callers should use :meth:`best_match` with ``reference=``.
        """
        self._ensure_directional()

        # Resolve reference index — supports ProbabilisticTensor, list[int], int
        N = len(self._scene.objects)
        ref_scores = self._to_score_vector(reference_score)
        ref_idx = None
        best_ref = -1.0
        for i in range(N):
            if ref_scores[i] > best_ref:
                best_ref = ref_scores[i]
                ref_idx = i

        if ref_idx is None:
            return list(options.keys())[0]

        # Get the appropriate normalized direction matrix (e.g. "west" -> "left",
        # "back" -> "behind"); diagonals have none and use the angle fallback.
        dir_key = f"{_relation_key(direction)}_normalized"
        use_angle_fallback = dir_key not in self._directional
        dir_matrix = self._directional.get(dir_key)

        # Helper to get directional score for object i relative to ref_idx.
        # Uses pre-computed relation matrix when available, else angle-based.
        def _dir_score(i: int) -> float:
            if use_angle_fallback:
                return self._direction_score_relative(i, ref_idx, direction)
            return self._to_float(dir_matrix[i, ref_idx])

        # --- nearest mode: two-phase approach ---
        # Phase 1: For each option, find the best object (by identity) and
        #          collect its direction score and distance to the reference.
        # Phase 2: Among options whose object IS in the specified direction
        #          (dir_score > 0.5), pick the nearest one. If none qualifies,
        #          fall back to the multiplicative score.
        if nearest:
            positions = self._scene._object_positions()
            ref_pos = positions[ref_idx] if len(positions) > 0 else None

            # Per-option: best object, its direction score, distance, identity
            option_info = {}  # letter -> (best_obj_idx, identity, dir_score, dist)
            for letter, score_tensor in options.items():
                best_i = None
                best_identity = -1.0
                for i in range(N):
                    if i == ref_idx:
                        continue
                    identity = self._to_float(score_tensor[i])
                    if identity > best_identity:
                        best_identity = identity
                        best_i = i
                if best_i is not None and best_identity > 0.05:
                    d_score = _dir_score(best_i)
                    dist = (
                        float(np.linalg.norm(positions[best_i] - ref_pos))
                        if ref_pos is not None
                        else 0.0
                    )
                    option_info[letter] = (best_i, best_identity, d_score, dist)

            # Phase 2: pick nearest among those in the correct direction
            DIR_THRESH = 0.5  # normalized score > 0.5 means on the correct side
            qualified = {
                letter: info
                for letter, info in option_info.items()
                if info[2] > DIR_THRESH  # dir_score > threshold
            }
            if qualified:
                best_letter = min(qualified, key=lambda letter: qualified[letter][3])
                return best_letter

            # Fallback: no option lies in the direction, so use the multiplicative
            # score identity * dir_score and ignore closeness.

        # --- default (non-nearest) mode or nearest fallback ---
        best_letter = None
        best_combined = -1.0

        for letter, score_tensor in options.items():
            for i in range(N):
                if i == ref_idx:
                    continue
                identity = self._to_float(score_tensor[i])
                # how much object i is in {direction} of ref_idx
                dir_score = _dir_score(i)

                combined = identity * dir_score
                if combined > best_combined:
                    best_combined = combined
                    best_letter = letter

        return best_letter if best_letter is not None else list(options.keys())[0]


    def check_relation(
        self,
        direction: str,
        subject_score,
        reference_score,
        min_identity: float = 0.05,
    ) -> float:
        """Check whether the subject is in the given direction relative to the
        reference, returning a [0, 1] score.

        This is designed for binary "Is X behind Y?" questions.  It finds the
        best object for *subject* and the best (distinct) object for *reference*
        using their identity scores, then returns the geometric relation score
        between them.

        Unlike the ProbabilisticTensor pattern
        ``(view.behind & is_A & is_B).exists()``, this avoids the min-operator
        that causes systematic false negatives when detection confidence is low.

        Parameters
        ----------
        direction : str
            "left", "right", "front", "behind", "above", "below", a diagonal
            ("front-left", "behind-right", ...) or a compass label.  Relative
            labels use the occluder convention ("front" = closer to the
            observer); compass labels are frame directions (north = +front).
        subject_score : ProbabilisticTensor or similar
            1D identity score tensor for the subject (the one that should be
            in the given direction).
        reference_score : ProbabilisticTensor or similar
            1D identity score tensor for the reference object.
        min_identity : float
            Minimum identity score for an object to be considered a valid
            match.  If both best scores are below this, returns 0.0.

        Returns
        -------
        float
            Score in [0, 1].  Values > 0.5 indicate the subject IS in the
            given direction relative to the reference.
        """
        self._ensure_directional()

        N = len(self._scene.objects)
        if N < 2:
            return 0.0

        # Build arrays of identity scores — supports ProbabilisticTensor, list[int], int
        subj_scores = self._to_score_vector(subject_score)
        ref_scores = self._to_score_vector(reference_score)

        # Find best subject and best reference (must be different objects)
        # Strategy: jointly pick (s, r) pair maximizing subj_scores[s] + ref_scores[r]
        best_s, best_r = None, None
        best_pair_score = -1.0
        for s in range(N):
            for r in range(N):
                if s == r:
                    continue
                pair = subj_scores[s] + ref_scores[r]
                if pair > best_pair_score:
                    best_pair_score = pair
                    best_s, best_r = s, r

        if best_s is None or best_r is None:
            return 0.0

        # Check minimum identity
        if subj_scores[best_s] < min_identity and ref_scores[best_r] < min_identity:
            return 0.0

        # Get the binary (sigmoid) relation score (diagonals included).
        dir_key = _relation_key(direction)  # use the binary version, not normalized
        if dir_key not in self._directional:
            # Angle-based fallback (raises a helpful error for unknown labels)
            score = self._direction_score_relative(best_s, best_r, direction)
        else:
            rel_matrix = self._directional[dir_key]
            score = float(rel_matrix[best_s, best_r])

        return score


    def best_match(
        self,
        direction: str,
        options: dict,
        *,
        reference=None,
        nearest: bool = False,
    ):
        """Pick the MCQ option that best matches ``direction`` in this view.

        Two modes, dispatched by whether ``reference=`` is provided:

        - **Origin-relative** (``reference=None``): "Which option lies
          ``direction`` of the frame origin?"  Combines identity scores
          with origin→object direction scores.
        - **Object-relative** (``reference`` is a 1D identity score
          tensor): "What is to the ``direction`` of the reference object?"
          Uses the K×K relation tensors.  Pass ``nearest=True`` to
          additionally weight by proximity to the reference.

        Parameters
        ----------
        direction : str
            One of ``"front" / "behind" / "left" / "right" / "above" /
            "below"``, a diagonal, or a compass label (same conventions as
            :meth:`check_relation`).
        options : dict
            Mapping from option letter (``"A"``, ``"B"``, ...) to a 1D
            identity score tensor over scene objects.
        reference : tensor or None
            Optional 1D identity score tensor for the reference object.
        nearest : bool, default False
            Object-relative mode only: prefer options whose object is
            on the correct side AND closest to the reference.

        Returns
        -------
        str
            The option key with the highest combined score.
        """
        if reference is not None:
            return self._best_match_relative(
                direction, reference, options, nearest=nearest
            )
        return self._best_match_origin(direction, options)


    def direction(self, source=None, target=None, *, as_label: bool = False):
        """Return the direction from ``source`` to ``target`` in this view.

        - If ``source`` is None, uses the frame origin.
        - ``source`` and ``target`` may be object indices (int),
          ``Camera`` / ``MergedObject`` instances, or 3D points; instances
          are resolved via the scene's ``_resolve_entity``.

        Parameters
        ----------
        source, target : entity, int, or 3D point
            Required: ``target``.  Optional: ``source`` (defaults to frame
            origin).
        as_label : bool, default False
            If True, return the 8-way relative-direction string label
            (``"front"`` / ``"front-right"`` / ``"right"`` / ``"back-right"``
            / ``"back"`` / ``"back-left"`` / ``"left"`` / ``"front-left"``).
            If False, return a numeric :class:`DirectionValue` with
            ``yaw_deg`` (0=ahead, increasing clockwise) and
            ``elevation_deg``.
        """
        if target is None:
            raise ValueError("view.direction() requires target=")

        def _to_xyz(e):
            if hasattr(e, "cam"):  # camera(N): an int index that carries its Camera
                e = e.cam
            if isinstance(e, (int, np.integer)):
                return np.asarray(
                    self._scene.objects[int(e)].center_world, dtype=float
                )
            # Anything else (a 3D point, a description, ...) goes through _resolve_entity, which
            # accepts exactly 3 numbers as a point and never reads a predicate's scores as one.
            pos, _, _ = self._scene._resolve_entity(e, label="entity")
            return np.asarray(pos, dtype=float)

        tgt_xyz = _to_xyz(target)
        src_xyz = self._frame_origin if source is None else _to_xyz(source)
        vec = tgt_xyz - src_xyz

        if as_label:
            from saturn.scene.direction_utils import RELATIVE_LABELS_8

            r = float(np.dot(vec, self._frame_right))
            f = float(np.dot(vec, self._frame_front))
            yaw = math.degrees(math.atan2(r, f)) % 360.0
            bin_idx = int(((yaw + 22.5) % 360.0) // 45.0)
            return RELATIVE_LABELS_8[bin_idx]

        return self._direction_from_vec(vec)


    def _direction_from_vec(self, vec: np.ndarray) -> DirectionValue:
        """Project a world-space vector onto frame axes, return DirectionValue."""
        vec = np.asarray(vec, dtype=float)
        r = float(np.dot(vec, self._frame_right))
        u = float(np.dot(vec, self._frame_up))
        f = float(np.dot(vec, self._frame_front))
        # Convert (right, up, front) → (yaw_deg, elevation_deg)
        # yaw: 0 = ahead (+front), increases clockwise (toward +right)
        yaw_deg = float(np.degrees(np.arctan2(r, f))) % 360.0
        horiz = float(np.hypot(r, f))
        el_deg = float(np.degrees(np.arctan2(u, horiz)))
        return DirectionValue(yaw_deg, el_deg)


    def match(self, source, target, options: dict):
        """Which MCQ option best describes ``source`` relative to ``target``?

        Computes the direction from ``target`` to ``source`` (in this view)
        and picks the option whose cardinal/relative label is angularly
        closest to the computed yaw.

        ``options`` maps option key (e.g. ``"A"``) to a label string
        (e.g. ``"front-left"``, ``"left"``, ``"behind"``, ``"south-east"``,
        ``"southeast"``, ``"north"``).
        """
        direction_val = self.direction(source=target, target=source)
        yaw_computed = float(direction_val.yaw_degree()) % 360.0

        def _ang_dist(a: float, b: float) -> float:
            d = abs(a - b) % 360.0
            return d if d <= 180.0 else 360.0 - d

        # Score each option by angular proximity of its label's canonical yaw
        # to the computed yaw. Options whose labels can't be resolved get a
        # distance of 180 (worst).
        #
        # Camera-only asymmetry: people use a narrower notion of "front" than
        # "back" when reasoning from a camera/photo viewpoint. Keep the shared
        # direction algebra symmetric, but in MCQ matching add a small penalty
        # to relative front-ish labels (front / front-left / front-right) when
        # the frame is camera-derived.
        _CAMERA_FRONTISH_PENALTY = 20.0  # degrees, applied only to camera views
        #
        # FOV-aware tie-breaking: when the frame has a known horizontal FOV
        # (camera-based views) and the computed yaw places the target outside
        # the FOV, penalise "front" (yaw=0) so it cannot beat lateral options.
        # This prevents borderline front-left cases (e.g. yaw=317°) from
        # being classified as "front" when the target is off-screen to the left.
        _FRONT_PENALTY = 45.0  # degrees added to front's angular distance
        apply_fov_penalty = False
        if self._hfov_deg is not None:
            half_fov = self._hfov_deg / 2.0
            # Signed yaw: convert [0,360) to [-180,180)
            signed_yaw = yaw_computed if yaw_computed <= 180.0 else yaw_computed - 360.0
            if abs(signed_yaw) > half_fov:
                apply_fov_penalty = True

        yaws = {key: label_yaw(lab) for key, lab in options.items()}
        warn_unresolved_labels("match", options, yaws)
        best_key = None
        best_dist = float("inf")
        for key, lab in options.items():
            y = yaws[key]
            canon = canonical_direction(str(lab))
            rel = canon if canon in RELATIVE_TO_CARDINAL else None
            if y is None:
                d = 180.0
            else:
                d = _ang_dist(yaw_computed, y)
                if self._hfov_deg is not None and rel is not None and rel.startswith("front"):
                    d += _CAMERA_FRONTISH_PENALTY
                # Penalise "front" (y==0) when target is outside the FOV
                if apply_fov_penalty and rel == "front":
                    d += _FRONT_PENALTY
            if d < best_dist:
                best_dist = d
                best_key = key

        if best_key is not None:
            return best_key
        # Absolute last resort: return the first option key.
        return next(iter(options.keys()))
