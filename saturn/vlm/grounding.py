"""ObjectGrounder — agentic VLM + SAM3 loop for grounding objects in a scene.

Given object groundings from the QueryPlanner (phrase, description, cam_id,
is_region), runs a detect → verify → refine loop for each grounding until
every mentioned object is found in the scene with high confidence.

The loop for each grounding:
  1. **Detect** on the preferred camera (cam_id) via SAM3, fallback to VLM grounding.
  2. **Verify** each candidate by rendering its bbox and asking the VLM
     "Is this the {phrase}?"
  3. **Refine** if verification fails — ask the VLM to produce a better
     description and re-detect (up to ``max_refine`` iterations).
  4. **Disambiguate** when multiple candidates survive — keep the one with
     highest VLM confidence.

Two-phase architecture
----------------------
``ground_all`` runs in two ordered phases so region grounding benefits from
the fully-grounded object set:

  * **Phase 1 — objects.** All ``is_region=False`` groundings are processed
    with the detect → verify → refine loop.  NMS runs after phase 1.
  * **Phase 2 — regions.** Each ``is_region=True`` grounding is expanded
    into its comma-separated constituents.  A constituent that already
    matches an existing object's label is reused (no re-detection);
    missing constituents go through the same object loop.  A region with no
    member falls back to grounding its phrase as one object.  After NMS each
    grounded region is also appended as ONE merged scene entity labelled with
    the region phrase (see ``_add_region_entity``); members stay in the scene.

This avoids redundant SAM3/VLM calls when an object is both mentioned
directly (e.g. ``"the stove"``) and listed as part of a region
(e.g. ``"kitchen area with stove, sink, counter"``) — and makes region
decomposition deterministic w.r.t. the order the planner emits groundings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from saturn.log import get_logger

log = get_logger(__name__)

# A verify call whose answer cannot be read is asked once more before the
# candidate is scored 0.0 (the "unparsed" outcome, see ``_unparsed``).
VERIFY_ATTEMPTS = 2
UNPARSED_FEEDBACK = "verification unparsed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_bbox(img, bbox, color=(255, 0, 0), width=4) -> Image.Image:
    """Draw a bounding box on a copy of *img* and return it."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


# Corner offsets of a unit box, in the order ``fusion.fit_bbox_to_rotation`` uses.
_BOX_OFFSETS = np.array(
    [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
     [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
    dtype=float,
)


def _extent_points(obj) -> np.ndarray:
    """(K, 3) points spanning *obj*: its fitted box corners, else its center."""
    corners = getattr(obj, "corners_world", None)
    if corners is not None:
        corners = np.asarray(corners, dtype=float).reshape(-1, 3)
        if corners.size and np.ptp(corners, axis=0).any():
            return corners
    return np.asarray(obj.center_world, dtype=float).reshape(1, 3)


def _is_region_entity(obj) -> bool:
    return bool((getattr(obj, "metadata", None) or {}).get("is_region"))


def _swap_image(scene_images, view_idx, annotated):
    """Return a copy of *scene_images* with *view_idx* replaced."""
    out = []
    for vi, img in enumerate(scene_images):
        if vi == view_idx:
            out.append(annotated)
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            out.append(img)
    return out


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GroundingResult:
    """Result of grounding a single phrase."""
    phrase: str
    description: str
    cam_id: Optional[int]
    is_region: bool
    obj_indices: List[int] = field(default_factory=list)
    attempts: int = 0
    verified: bool = False
    final_description: str = ""
    verify_score: float = 0.0
    multi_view: bool = False  # True for motion/ordering targets — keep all per-frame instances
    # Failure diagnostic for the planner-retry loop:
    #   ""              — success, no diagnostic
    #   "no_candidates" — SAM3+VLM both returned 0 boxes (description likely off)
    #   "no_verify"     — SAM3 returned candidates but none cleared verify_threshold
    #                     (cam_id and/or description likely off)
    last_diagnostic: str = ""
    n_candidates: int = 0  # how many SAM3 boxes the last attempt actually saw
    verify_feedback: str = ""  # free-form VLM critique from the verify step;
    # surfaces signals like "this is the wrong desk" or "the bed is in image 2"
    # that the planner can act on in refine_groundings
    # Region groundings only: scene indices of the constituent objects. After
    # ``ground_all`` merges them into one region entity, ``obj_indices`` points
    # at that entity and this list keeps the members.
    region_members: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ObjectGrounder
# ---------------------------------------------------------------------------

class ObjectGrounder:
    """Agentic object grounding loop using VLM + SAM3.

    Parameters
    ----------
    vlm : object
        VLM with ``_query(images, prompt, max_new_tokens)`` and
        ``_score_simple(images, prompt)`` methods.
    scene : Scene
        The scene to populate with detected objects.
    question : str
        The spatial reasoning question (used in verification prompts).
    max_refine : int
        Maximum refinement iterations per grounding.
    verify_threshold : float
        Minimum VLM yes-probability to accept a detection.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        vlm: Any,
        scene: Any,
        question: str,
        max_refine: int = 2,
        verify_threshold: float = 0.4,
        verbose: bool = True,
    ):
        self.vlm = vlm
        self.scene = scene
        self.question = question
        self.max_refine = max_refine
        self.verify_threshold = verify_threshold
        self.verbose = verbose
        # Per-grounder session: maps scene-object index → (verified, score)
        # for objects the grounder itself grounded in this session. Used
        # by region grounding to decide whether reused members are actually
        # trustworthy anchors, not just labelled shapes.
        self._obj_verification: Dict[int, Tuple[bool, float]] = {}
        # Every GroundingResult this grounder returned, across ``ground_all``
        # calls (the planner-retry pass calls it again). NMS re-indexes the
        # scene, so ``_nms`` re-points all of their indices, not only the
        # current call's.
        self._session_results: List[GroundingResult] = []

    # -- public API --------------------------------------------------------

    def ground_all(
        self,
        groundings: List[Dict[str, Any]],
        item_id: str = "",
    ) -> List[GroundingResult]:
        """Run the detect → verify → refine loop for every grounding.

        Modifies ``self.scene`` in place (adds detected objects).
        Returns a list of ``GroundingResult`` for diagnostics, in the same
        order as the input ``groundings`` list.

        Two-phase:
          1. All ``is_region=False`` groundings are processed first, so that
             at the start of phase 2 the scene contains every concrete object
             the planner mentioned.
          2. ``is_region=True`` groundings are then expanded into
             constituents. A constituent whose label already matches a scene
             object is reused directly; otherwise it goes through the object
             loop. This eliminates duplicate detection when a concrete
             object (e.g. ``"the stove"``) is ALSO listed as part of a
             region's description (e.g. ``"kitchen area with stove, ..."``).
        """
        pre_count = self.scene.objects_count

        # Partition while preserving original positions for result ordering.
        object_items: List[Tuple[int, Dict[str, Any]]] = []
        region_items: List[Tuple[int, Dict[str, Any]]] = []
        for i, g in enumerate(groundings):
            if bool(g.get("is_region", False)):
                region_items.append((i, g))
            else:
                object_items.append((i, g))

        # Pre-allocate results slots so we can insert in-order later.
        results: List[Optional[GroundingResult]] = [None] * len(groundings)

        # ------------------------------------------------------------------
        # Phase 1: concrete objects
        # ------------------------------------------------------------------
        if self.verbose and region_items:
            log.info(
                f"[{item_id}] GroundObj phase 1/2: "
                f"{len(object_items)} object(s), {len(region_items)} region(s) queued"
            )

        for i, g in object_items:
            r = self._process_one(g, item_id)
            if r is not None:
                results[i] = r
                self._session_results.append(r)

        # NMS between phases so region constituent-matching sees a clean
        # label set (no near-duplicate detections shadowing each other).
        added_phase1 = self.scene.objects_count - pre_count
        if added_phase1 > 0:
            self._nms(item_id, "phase 1")

        # ------------------------------------------------------------------
        # Phase 2: regions (expand constituents, reuse already-grounded objects)
        # ------------------------------------------------------------------
        pre_phase2_count = self.scene.objects_count
        if region_items and self.verbose:
            log.info(
                f"[{item_id}] GroundObj phase 2/2: expanding "
                f"{len(region_items)} region(s)"
            )

        for i, g in region_items:
            r = self._process_one(g, item_id)
            if r is not None:
                results[i] = r
                self._session_results.append(r)

        # Final NMS if any new objects were added in phase 2 (counted from
        # after the phase-1 NMS, which may have removed objects).
        added_phase2 = self.scene.objects_count - pre_phase2_count
        if added_phase2 > 0:
            self._nms(item_id, "phase 2")

        # Each grounded region also becomes ONE scene entity (after NMS, so
        # member indices are final). Its members stay in the scene.
        for r in results:
            if r is not None and r.region_members:
                self._add_region_entity(r, item_id)

        if self.verbose:
            log.info(
                f"[{item_id}] GroundObj done: {pre_count} → "
                f"{self.scene.objects_count} objects "
                f"({sum(1 for r in results if r is not None)} groundings processed)"
            )
        # Drop None slots (skipped groundings) while preserving order.
        return [r for r in results if r is not None]

    # -- per-grounding dispatch -------------------------------------------

    def _process_one(
        self, g: Dict[str, Any], item_id: str,
    ) -> Optional[GroundingResult]:
        """Dispatch a single planner grounding to the object or region path.

        Returns ``None`` if the grounding is malformed or already covered.
        """
        phrase = str(g.get("phrase", "")).strip()
        desc = str(g.get("description", "")).strip()
        # cam_id may be an int or a list of ints; the first valid view is the
        # primary cam_id. Out-of-range or empty list → None (scan all views).
        cam_id = self._normalize_cam_id(g.get("cam_id"))
        is_region = bool(g.get("is_region", False))
        multi_view = bool(g.get("multi_view", False))
        unique = bool(g.get("unique", True))
        if not phrase or not desc:
            return None

        if self._already_in_scene(phrase):
            if self.verbose:
                log.info(f"[{item_id}] GroundObj: '{phrase}' already in scene, skip")
            return None

        if is_region:
            return self._ground_region(phrase, desc, cam_id, item_id)
        return self._ground_object(
            phrase, desc, cam_id, item_id,
            multi_view=multi_view, unique=unique,
        )

    # -- region handling ---------------------------------------------------

    def _ground_region(
        self, phrase: str, desc: str, cam_id: Optional[int], item_id: str,
    ) -> GroundingResult:
        """Ground a region by resolving its constituent objects.

        For each comma-separated constituent in *desc*:
          1. If an existing scene object's label **exactly** matches, reuse it.
          2. Otherwise run the detect → verify → refine object loop for it.

        Verification semantics (strict):
          A region is ``verified=True`` only if it has at least one member
          whose OWN grounding passed the verify threshold. Region membership
          alone is NOT verification. Downstream frame-building code should
          treat ``verified=False`` regions as weak anchors.

        The region's ``verify_score`` is set to the maximum member score,
        and ``attempts`` is set to the number of constituents processed.

        Reusing already-grounded objects via exact-label match matters
        because the planner often lists concrete objects (e.g.
        ``"stove"``) both as first-class groundings AND as part of a
        region's constituent list (e.g. ``"kitchen area with stove,
        sink, counter"``).
        """
        result = GroundingResult(
            phrase=phrase, description=desc, cam_id=cam_id,
            is_region=True, final_description=desc,
        )
        parts = [p.strip() for p in desc.split(",") if p.strip()]

        member_indices: List[int] = []
        member_verified: List[bool] = []
        member_scores: List[float] = []
        sub_results: List[GroundingResult] = []
        reused = 0
        newly_detected = 0
        for part in parts:
            existing = self._find_object_by_label(part)
            if existing is not None:
                member_indices.append(existing)
                # Reused: trust whatever this grounder already verified.
                was_verified, was_score = self._obj_verification.get(
                    existing, (False, 0.0),
                )
                member_verified.append(was_verified)
                member_scores.append(was_score)
                reused += 1
                if self.verbose:
                    log.info(
                        f"[{item_id}] GroundObj region '{phrase}' reuse "
                        f"obj {existing} for constituent '{part}' "
                        f"(verified={was_verified}, score={was_score:.3f})"
                    )
                continue
            sub = self._ground_object(part, part, cam_id, item_id)
            sub_results.append(sub)
            if sub.obj_indices:
                for mi in sub.obj_indices:
                    member_indices.append(mi)
                    # Propagate the sub-grounding's own verified flag.
                    member_verified.append(bool(sub.verified))
                    member_scores.append(float(sub.verify_score))
                newly_detected += 1

        # Dedupe while preserving order + verification provenance, and drop
        # any index that does not point into scene.objects.
        n_scene = len(self.scene.objects)
        seen: set = set()
        deduped: List[int] = []
        deduped_verified: List[bool] = []
        deduped_scores: List[float] = []
        for mi, v, s in zip(member_indices, member_verified, member_scores):
            if mi < 0 or mi >= n_scene:
                continue
            if mi not in seen:
                seen.add(mi)
                deduped.append(mi)
                deduped_verified.append(v)
                deduped_scores.append(s)

        result.obj_indices = deduped
        result.region_members = list(deduped)
        # Strict: region is verified iff at least one member is verified.
        result.verified = any(deduped_verified)
        result.verify_score = max(deduped_scores) if deduped_scores else 0.0
        result.attempts = len(parts)

        if not deduped:
            # No constituent was found: ground the region phrase itself as a
            # single object before giving up (e.g. "kitchen area" is often
            # detectable as a whole even when "counter, sink" are not).
            self._ground_region_as_object(result, sub_results, item_id)

        if self.verbose:
            n_verified = sum(1 for v in deduped_verified if v)
            log.info(
                f"[{item_id}] GroundObj region '{phrase}': "
                f"{len(deduped)} member(s) ({n_verified} verified, "
                f"reused={reused}, new={newly_detected}) "
                f"→ verified={result.verified} "
                f"score={result.verify_score:.3f} from [{desc}]"
            )
        return result

    def _ground_region_as_object(
        self,
        result: GroundingResult,
        sub_results: List[GroundingResult],
        item_id: str,
    ) -> None:
        """Fallback for a region with no members: ground its phrase as ONE object.

        Updates *result* in place. On failure it carries the same diagnostic
        ``_ground_object`` sets -- ``"no_verify"`` when any attempt (a
        constituent or the whole phrase) saw detector candidates, else
        ``"no_candidates"`` -- so the planner-retry and salvage passes pick
        the region up instead of silently skipping it.
        """
        whole = self._ground_object(
            result.phrase, result.phrase, result.cam_id, item_id,
        )
        result.attempts += whole.attempts
        if whole.obj_indices:
            result.obj_indices = list(whole.obj_indices)
            result.verified = whole.verified
            result.verify_score = whole.verify_score
            result.final_description = whole.final_description
            if self.verbose:
                log.info(
                    f"[{item_id}] GroundObj region '{result.phrase}': no member "
                    f"found, grounded the phrase as one object "
                    f"obj_idx={result.obj_indices}"
                )
            return

        tried = sub_results + [whole]
        n_candidates = max(r.n_candidates for r in tried)
        result.last_diagnostic = "no_verify" if n_candidates > 0 else "no_candidates"
        result.n_candidates = n_candidates
        result.verify_score = max(r.verify_score for r in tried)
        result.verify_feedback = next(
            (r.verify_feedback for r in reversed(tried) if r.verify_feedback), "",
        )
        if self.verbose:
            log.warning(
                f"[{item_id}] GroundObj FAILED region '{result.phrase}': "
                f"{result.last_diagnostic} (n_candidates={n_candidates})"
            )

    def _add_region_entity(self, result: GroundingResult, item_id: str) -> None:
        """Append ONE scene entity standing for a grounded region.

        Programs score a region phrase ("walkway between table and trash
        bins", "kitchen area") like any object, so the region must exist as a
        single entity, not only as scattered constituents. The entity merges
        the members of ``result.region_members`` (which stay in the scene):

          * label        -- the region phrase (leading "the" stripped)
          * position     -- centroid of the member positions
          * extent       -- axis-aligned box around the member boxes
          * per-view bbox / mask -- union over members; views -- union
          * world points -- union
          * orientation  -- identity, ``orientation_confidence=0`` (unknown:
            a region has no front)

        ``result.obj_indices`` is re-pointed at the new entity.
        """
        from saturn.scene.types import MergedObject

        members = [self.scene.objects[i] for i in result.region_members]
        if not members:
            return

        centers = np.array(
            [np.asarray(m.center_world, dtype=float).ravel()[:3] for m in members]
        )
        extent_pts = np.concatenate([_extent_points(m) for m in members], axis=0)
        lo, hi = extent_pts.min(axis=0), extent_pts.max(axis=0)
        dims = hi - lo
        box_center = (lo + hi) / 2.0
        corners = box_center + 0.5 * dims * _BOX_OFFSETS

        per_view_bboxes: Dict[int, List[float]] = {}
        per_view_scores: Dict[int, float] = {}
        for m in members:
            for v, b in getattr(m, "per_view_bboxes", {}).items():
                prev = per_view_bboxes.get(v)
                per_view_bboxes[v] = [float(c) for c in b[:4]] if prev is None else [
                    min(prev[0], b[0]), min(prev[1], b[1]),
                    max(prev[2], b[2]), max(prev[3], b[3]),
                ]
                s = float(getattr(m, "per_view_scores", {}).get(v, 0.0))
                per_view_scores[v] = max(per_view_scores.get(v, s), s)

        # A per-view mask is the union of the member masks, kept only when
        # every member boxed in that view has one (otherwise the union would
        # silently drop a member); without a mask, scoring uses the bbox.
        per_view_masks: Dict[int, np.ndarray] = {}
        for v in per_view_bboxes:
            masks = [
                getattr(m, "per_view_masks", {}).get(v) for m in members
                if v in getattr(m, "per_view_bboxes", {})
            ]
            if all(mk is not None for mk in masks) and len(
                {np.shape(mk) for mk in masks}
            ) == 1:
                per_view_masks[v] = np.logical_or.reduce(
                    [np.asarray(mk, dtype=bool) for mk in masks]
                )

        points = [
            np.asarray(m.world_points, dtype=float) for m in members
            if getattr(m, "world_points", None) is not None and len(m.world_points)
        ]
        views = sorted({int(v) for m in members for v in getattr(m, "views", [])}
                       | set(per_view_bboxes))

        idx = len(self.scene.objects)
        entity = MergedObject(
            id=idx,
            label=re.sub(r"^the\s+", "", result.phrase.lower()).strip(),
            views=views,
            center_world=centers.mean(axis=0),
            rotation_world=np.eye(3),
            front_world=np.array([0.0, 0.0, 1.0]),
            up_world=np.array([0.0, 1.0, 0.0]),
            right_world=np.array([1.0, 0.0, 0.0]),
            euler_world_deg=np.zeros(3),
            dims=dims,
            corners_world=corners,
            height=float(dims[1]),
            support_y=float(lo[1]),
            world_points=np.concatenate(points, axis=0) if points else None,
            per_view_bboxes=per_view_bboxes,
            per_view_masks=per_view_masks,
            per_view_scores=per_view_scores,
            metadata={
                "is_region": True,
                "member_indices": list(result.region_members),
                "source": "region",
            },
            orientation_confidence=0.0,
        )
        self.scene.objects.append(entity)
        invalidate = getattr(self.scene, "_invalidate_caches", None)
        if callable(invalidate):
            invalidate()
        result.obj_indices = [idx]
        self._obj_verification[idx] = (result.verified, result.verify_score)
        if self.verbose:
            log.info(
                f"[{item_id}] GroundObj region '{result.phrase}' → entity "
                f"obj_idx={idx} merging members {result.region_members}"
            )

    # -- single object loop ------------------------------------------------

    def _ground_object(
        self, phrase: str, desc: str, cam_id: Optional[int], item_id: str,
        multi_view: bool = False, unique: bool = True,
    ) -> GroundingResult:
        """Detect → verify → refine loop for a single object.

        ``multi_view=True`` switches behaviour for motion / ordering
        questions where the SAME logical object should be tracked across
        every camera frame: every candidate that passes the (view-blind)
        verify threshold is kept (instead of disambiguating to a single best),
        and rejected candidates are NOT removed from the scene.  Programs
        consume the per-frame instances via ``score()`` returning a tensor
        with several high values, or via the merged object's
        ``per_view_centers`` / ``per_view_bboxes`` if fusion clusters
        them together.
        """
        result = GroundingResult(
            phrase=phrase, description=desc, cam_id=cam_id,
            is_region=False, final_description=desc,
            multi_view=multi_view,
        )
        clean_label = re.sub(r"^the\s+", "", phrase.lower()).strip()

        effective_cam = cam_id

        for attempt in range(1 + self.max_refine):
            result.attempts = attempt + 1
            current_desc = result.final_description

            # 1. Detect on the current camera (cam_id is authoritative: no
            # all-camera fallback).
            obj_ids = self._detect_candidates(
                clean_label, current_desc, effective_cam, item_id, unique,
            )

            if not obj_ids:
                # Nothing detected: with refine attempts left, ask the VLM
                # where the object actually is (no rejected boxes to show).
                if attempt < self.max_refine:
                    effective_cam, retry = self._revise_grounding(
                        result, phrase, current_desc, effective_cam, item_id, attempt,
                    )
                    if retry:
                        continue
                return self._fail_no_candidates(result, phrase, attempt, item_id)

            for oi in obj_ids:
                self.scene.objects[oi].label = clean_label

            # 2a. Multi-view / non-unique: keep every candidate that passes.
            scored = None
            if multi_view or not unique:
                passed, scored = self._keep_verified_candidates(
                    result, obj_ids, phrase, current_desc, attempt, item_id,
                )
                if passed:
                    return result

            # 2b. Verify + disambiguate (anchor on planner's cam_id when set).
            # After 2a the candidates are not asked again: detection on a
            # camera yields candidates seen only in that camera, so 2a's
            # view-blind scores are the anchored ones.
            best_idx, best_score, best_feedback = self._verify_and_pick(
                obj_ids, phrase, current_desc, item_id, cam_id=effective_cam,
                scored=scored,
            )
            if best_idx is not None and best_score >= self.verify_threshold:
                self._accept_best(
                    result, obj_ids, best_idx, best_score, phrase, attempt, item_id,
                )
                return result

            # 3. Verification failed: remove the candidates, keeping their
            # boxes so the refine probe can show the VLM what was rejected.
            rejected_boxes = self._rejected_boxes(obj_ids, effective_cam)
            n_cands = len(obj_ids)
            self._remove_objects(obj_ids)

            if attempt < self.max_refine:
                effective_cam, retry = self._revise_grounding(
                    result, phrase, current_desc, effective_cam, item_id, attempt,
                    rejected_boxes=rejected_boxes, best_score=best_score,
                )
                if retry:
                    continue

            self._fail_no_verify(
                result, n_cands, best_score, best_feedback, phrase, attempt, item_id,
            )
            return result

        return result

    def _detect_candidates(
        self, clean_label: str, desc: str, cam_id: Optional[int], item_id: str,
        unique: bool,
    ) -> List[int]:
        """Detect the short label first, then the description if that finds nothing.

        SAM3 grounds nouns like "TV" cleanly but often returns 0 boxes for
        verbose descriptions like "television screen displaying text".
        """
        obj_ids = self._detect(clean_label, cam_id, item_id, unique=unique)
        if not obj_ids and desc and desc != clean_label:
            obj_ids = self._detect(desc, cam_id, item_id, unique=unique)
        return obj_ids

    def _keep_verified_candidates(
        self,
        result: GroundingResult,
        obj_ids: List[int],
        phrase: str,
        desc: str,
        attempt: int,
        item_id: str,
    ) -> Tuple[bool, List[Tuple[float, str]]]:
        """Multi-view / non-unique path: keep every candidate that passes verify.

        Rejected candidates stay in the scene: each surviving instance is
        either the same logical object in a different camera frame
        (multi_view) or a separate instance of a category whose every member
        is wanted (unique=False). Each instance can live in any camera, so
        verification is view-blind. Returns ``(passed, scored)``: ``passed``
        is True when some candidate passed and *result* is final; otherwise
        the first rejection feedback is stored and the caller falls through
        to the single-best path, which picks from ``scored`` (the
        ``[(score, feedback)]`` of every candidate) instead of asking again.
        """
        kept: List[int] = []
        kept_scores: List[float] = []
        rejected_feedback: List[str] = []
        scored = self._score_candidates_ordered(
            obj_ids, phrase, desc, item_id, cam_id=None,
        )
        for oi, (sc, fb) in zip(obj_ids, scored):
            if sc >= self.verify_threshold:
                kept.append(oi)
                kept_scores.append(sc)
            elif fb:
                rejected_feedback.append(fb)
        if kept:
            result.obj_indices = list(kept)
            result.verified = True
            result.verify_score = max(kept_scores)
            for oi, sc in zip(kept, kept_scores):
                self._obj_verification[oi] = (True, sc)
            if self.verbose:
                log.info(
                    f"[{item_id}] GroundObj OK (multi_view) '{phrase}': "
                    f"kept {len(kept)}/{len(obj_ids)} candidates "
                    f"max_score={max(kept_scores):.3f} attempt={attempt+1}"
                )
            return True, scored
        # The single-best path overwrites this feedback if it produces its own.
        if rejected_feedback:
            result.verify_feedback = rejected_feedback[0]
        return False, scored

    def _accept_best(
        self,
        result: GroundingResult,
        obj_ids: List[int],
        best_idx: int,
        best_score: float,
        phrase: str,
        attempt: int,
        item_id: str,
    ) -> None:
        """Keep the verified winner, remove the other candidates, record it."""
        result.obj_indices = [best_idx]
        result.verified = True
        result.verify_score = best_score
        rejected = [oi for oi in obj_ids if oi != best_idx]
        if rejected:
            self._remove_objects(rejected)
            shift = sum(1 for r in rejected if r < best_idx)
            result.obj_indices = [best_idx - shift]
        # Record for region reuse: the kept object is verified.
        self._obj_verification[result.obj_indices[0]] = (True, best_score)
        if self.verbose:
            log.info(
                f"[{item_id}] GroundObj OK '{phrase}': "
                f"obj_idx={result.obj_indices[0]} "
                f"score={best_score:.3f} attempt={attempt+1}"
            )

    def _rejected_boxes(
        self, obj_ids: List[int], cam_id: Optional[int],
    ) -> Dict[int, Any]:
        """``{view: bbox}`` of the rejected candidates, for the refine probe.

        A candidate boxed in *cam_id* contributes that box; any other
        candidate contributes its first view's box unless that view already
        has one.
        """
        rejected_boxes: Dict[int, Any] = {}
        for oi in obj_ids:
            per_view = getattr(self.scene.objects[oi], "per_view_bboxes", {})
            if cam_id is not None and cam_id in per_view:
                rejected_boxes[cam_id] = per_view[cam_id]
            elif per_view:
                v = next(iter(per_view))
                rejected_boxes.setdefault(v, per_view[v])
        return rejected_boxes

    def _revise_grounding(
        self,
        result: GroundingResult,
        phrase: str,
        current_desc: str,
        cam_id: Optional[int],
        item_id: str,
        attempt: int,
        rejected_boxes: Optional[Dict[int, Any]] = None,
        best_score: Optional[float] = None,
    ) -> Tuple[Optional[int], bool]:
        """Run the fail-recovery probe and apply its revision to *result*.

        ``rejected_boxes`` is None after a detection that found nothing;
        otherwise it holds the boxes of the candidates verification rejected
        (best score ``best_score``). The probe's reason is stored for the
        planner-retry layer whether or not the revision is applied here; a
        new description replaces ``result.final_description``.

        Returns ``(cam_id, retry)``: the camera to detect on next (the
        probe's first suggested view) and whether the probe changed anything
        worth another attempt.
        """
        revised = self._refine_grounding(
            phrase, current_desc,
            prior_cam_ids=[cam_id] if cam_id is not None else [],
            item_id=item_id,
            rejected_boxes=rejected_boxes,
        )
        if revised is None:
            return cam_id, False
        if revised.get("reason"):
            result.verify_feedback = revised["reason"]
        new_desc = revised.get("description") or ""
        new_cams = revised.get("cam_ids") or []
        switched_cam = False
        if new_cams and new_cams[0] != cam_id:
            cam_id = new_cams[0]
            switched_cam = True
        if new_desc and new_desc != current_desc:
            result.final_description = new_desc
        if self.verbose:
            cam_msg = f" → cam_id={cam_id}" if switched_cam else ""
            if rejected_boxes is None:
                log.info(
                    f"[{item_id}] GroundObj refine '{phrase}' "
                    f"attempt {attempt+1}{cam_msg}: "
                    f"'{(new_desc or current_desc)[:80]}'"
                )
            else:
                log.info(
                    f"[{item_id}] GroundObj verify-fail '{phrase}' "
                    f"(score={best_score:.3f}){cam_msg}, refine → "
                    f"'{(new_desc or current_desc)[:80]}'"
                )
        # Retrying only helps if the probe changed something; otherwise the
        # same detect query would just run again.
        return cam_id, bool(new_cams or (new_desc and new_desc != current_desc))

    def _fail_no_candidates(
        self, result: GroundingResult, phrase: str, attempt: int, item_id: str,
    ) -> GroundingResult:
        result.last_diagnostic = "no_candidates"
        result.n_candidates = 0
        if self.verbose:
            log.warning(
                f"[{item_id}] GroundObj FAILED '{phrase}' "
                f"after {attempt+1} attempts"
            )
        return result

    def _fail_no_verify(
        self,
        result: GroundingResult,
        n_cands: int,
        best_score: Optional[float],
        best_feedback: str,
        phrase: str,
        attempt: int,
        item_id: str,
    ) -> None:
        result.last_diagnostic = "no_verify"
        result.n_candidates = n_cands
        result.verify_score = float(best_score) if best_score is not None else 0.0
        # Prefer the verify-step feedback (per-candidate critique) over any
        # feedback already stamped by _refine_grounding; verify is closer to
        # the candidates that were ultimately rejected.
        if best_feedback:
            result.verify_feedback = best_feedback
        if self.verbose:
            log.warning(
                f"[{item_id}] GroundObj FAILED verify '{phrase}' "
                f"score={best_score:.3f} after {attempt+1} attempts"
            )

    # -- detect (SAM3 then VLM ground) ------------------------------------

    def _detect(
        self, desc: str, cam_id: Optional[int], item_id: str, unique: bool = False,
    ) -> List[int]:
        """Try SAM3 detect, fallback to VLM grounding."""
        # Try SAM3 on specific camera first
        try:
            ids = self.scene.detect(desc, camera=cam_id, unique=unique)
            if ids:
                return ids
        except Exception as e:
            if self.verbose:
                log.error(f"[{item_id}] SAM3 detect failed: {e}")

        # Fallback: VLM grounding (anchored at planner's cam_id)
        if hasattr(self.scene, "ground_fn") and self.scene.ground_fn:
            try:
                ids = self.scene.ground(desc, camera=cam_id, unique=unique)
                if ids:
                    return ids
            except Exception as e:
                if self.verbose:
                    log.error(f"[{item_id}] VLM ground failed: {e}")

        return []

    # -- salvage (last-resort: SAM3 best candidate, no verify) ----------

    def salvage(
        self,
        phrase: str,
        description: str,
        cam_id: Optional[int],
        item_id: str,
    ) -> Optional[int]:
        """Last-resort grounding: accept SAM3's best candidate without verify.

        Used by the orchestrator when the verify-loop AND the planner-retry
        have both failed and the alternative is leaving the scene empty.
        Detects with the original description (then the bare-noun fallback);
        keeps the candidate with the highest detection score (its best
        per-view SAM3 score; the first of tied candidates) and removes the
        rest, then labels it from ``phrase``. Fusion returns candidates in
        spatial order, not score order, so the first one is not the best.

        Returns the (possibly-shifted-after-removal) scene-object index of
        the kept candidate, or ``None`` if even SAM3 returned no boxes.

        Trades precision for recall — caller is expected to mark the result
        as low-confidence (``verified=False``) so downstream consumers can
        weight it appropriately if they choose.
        """
        clean_label = re.sub(r"^the\s+", "", phrase.lower()).strip()
        # Planner cam_id may be a list; detect() takes one int.
        cam_id = self._normalize_cam_id(cam_id)
        # Try the description first (richer signal); fall back to the bare noun.
        obj_ids = self._detect(description, cam_id, item_id) if description else []
        if not obj_ids and clean_label and clean_label != (description or "").lower():
            obj_ids = self._detect(clean_label, cam_id, item_id)
        if not obj_ids:
            return None

        keep_idx = max(obj_ids, key=self._detection_score)
        rejected = [oi for oi in obj_ids if oi != keep_idx]
        self.scene.objects[keep_idx].label = clean_label or phrase.strip().lower()
        if rejected:
            self._remove_objects(rejected)
            shift = sum(1 for r in rejected if r < keep_idx)
            keep_idx -= shift
        if self.verbose:
            log.info(
                f"[{item_id}] GroundObj SALVAGED '{phrase}': "
                f"obj_idx={keep_idx} (no verify, kept best of {len(obj_ids)} SAM3 boxes)"
            )
        return keep_idx

    def _detection_score(self, obj_idx: int) -> float:
        """Best per-view detection score of a scene object (0.0 without one)."""
        scores = getattr(self.scene.objects[obj_idx], "per_view_scores", None) or {}
        return max((float(s) for s in scores.values()), default=0.0)

    # -- verify + pick best candidate -------------------------------------

    def _score_candidates_ordered(
        self,
        obj_ids: List[int],
        phrase: str,
        desc: str,
        item_id: str,
        cam_id: Optional[int] = None,
    ) -> List[Tuple[float, str]]:
        """``[(score, feedback)]`` for *obj_ids*, in obj_ids order (serial).

        Detection and verification are deliberately NOT parallelised:
        detection hits the Ray Serve stack, which a single stream already
        saturates.
        """
        return [
            self._score_candidate(oi, phrase, desc, item_id, cam_id=cam_id)
            for oi in obj_ids
        ]

    def _verify_and_pick(
        self,
        obj_ids: List[int],
        phrase: str,
        desc: str,
        item_id: str,
        cam_id: Optional[int] = None,
        scored: Optional[List[Tuple[float, str]]] = None,
    ) -> Tuple[Optional[int], float, str]:
        """Score each candidate with VLM and return ``(best_idx, best_score, feedback)``.

        ``scored`` (``[(score, feedback)]`` in *obj_ids* order) are scores the
        caller already has; the candidates are then not asked again.

        ``feedback`` is the VLM's free-form observation for the WINNING
        candidate — non-empty when the winner is still below threshold,
        empty when the winner clearly cleared (and downstream doesn't
        need a critique). This is what gets surfaced to the planner's
        ``refine_groundings`` call when verification ultimately fails.

        If VLM scoring is unavailable, returns the first candidate with
        score 1.0 (trust the detector).

        ``cam_id`` (planner-supplied reference view) anchors verification:
        candidates without a bbox in that view are excluded, and remaining
        candidates are judged by the VLM in the cam_id image (not in their
        own best-detected view). Pass ``None`` for view-blind scoring (each
        candidate judged in its own best view).
        """
        if not obj_ids:
            return None, 0.0, ""

        if scored is None:
            scored = self._score_candidates_ordered(
                obj_ids, phrase, desc, item_id, cam_id=cam_id,
            )
        if len(obj_ids) == 1:
            score, fb = scored[0]
            return obj_ids[0], score, fb

        best_idx = None
        best_score = -1.0
        best_fb = ""
        per_cand_scores: List[Tuple[int, float]] = []
        for oi, (score, fb) in zip(obj_ids, scored):
            per_cand_scores.append((oi, score))
            if score > best_score:
                best_score, best_idx, best_fb = score, oi, fb

        if self.verbose and len(obj_ids) > 1:
            anchor = f"cam {cam_id}" if cam_id is not None else "candidate-best-view"
            log.info(
                f"[{item_id}] GroundObj disambiguate '{phrase}': "
                f"picked obj {best_idx} (score={best_score:.3f}) "
                f"from {len(obj_ids)} candidates [anchor={anchor}, "
                f"scores={[(o, round(s, 3)) for o, s in per_cand_scores]}]"
            )
        return best_idx, best_score, best_fb

    def _score_candidate(
        self, obj_idx: int, phrase: str, desc: str, item_id: str,
        cam_id: Optional[int] = None,
    ) -> Tuple[float, str]:
        """Ask VLM whether the boxed region is the {phrase}.

        Returns ``(score, feedback)``:
          - ``score`` in [0,1] — the verify confidence.
          - ``feedback`` — a diagnostic for the planner's
            ``refine_groundings`` retry; empty when the VLM answered.

        When ``cam_id`` is provided (the planner's intended reference view),
        the candidate is judged in *that* view rather than its own
        highest-confidence detection view. Candidates with no bbox in
        ``cam_id`` are rejected outright (score 0.0).

        Verification is the calibrated P(Yes) logprob of a yes/no question
        (graded, thresholded at ``verify_threshold``): the soft-predicate
        principle applied to grounding. The feedback string is empty except
        for the rejections below.
        """
        if not hasattr(self.vlm, "_score_simple"):
            return 1.0, ""  # no VLM scoring → trust detector

        obj = self.scene.objects[obj_idx]
        per_view_bboxes = getattr(obj, "per_view_bboxes", {})

        if cam_id is not None:
            # View-aware: planner's cam_id is the reference view. A candidate
            # the detector did not box in that view is rejected; the message
            # reaches the planner's retry prompt.
            if cam_id not in per_view_bboxes:
                return 0.0, (
                    f"candidate does not project into image {cam_id} "
                    f"(outside that camera's frustum)"
                )
            best_view = cam_id
        else:
            # View-blind path: pick the candidate's own best view.
            best_view = None
            best_view_score = -1.0
            for v, s in getattr(obj, "per_view_scores", {}).items():
                if s > best_view_score:
                    best_view_score = s
                    best_view = v
            if best_view is None and hasattr(obj, "views") and obj.views:
                best_view = obj.views[0]
            if best_view is None or best_view not in per_view_bboxes:
                # Nothing to show the VLM: an unverifiable candidate is not
                # a verified one.
                return 0.0, "no bbox to verify the candidate against"

        bbox = per_view_bboxes[best_view]
        annotated = _render_bbox(self.scene.images[best_view], bbox)
        all_images = _swap_image(self.scene.images, best_view, annotated)

        prompt = (
            f'Question: "{self.question}"\n\n'
            f'I need to find "{phrase}" ({desc}) to answer the question above. '
            f'Is the object highlighted by the red bounding box in the marked '
            f'image the correct "{phrase}"? Answer Yes or No.\n'
            f'Answer the question using a single word or phrase.'
        )
        yes_prob = None
        for _ in range(VERIFY_ATTEMPTS):
            try:
                p = self.vlm._score_simple(all_images, prompt)
                p = float(p.item() if hasattr(p, "item") else p)
            except Exception as e:
                if self.verbose:
                    log.error(f"[{item_id}] VLM score error obj {obj_idx}: {e}")
                continue
            if np.isfinite(p):
                yes_prob = p
                break
        if yes_prob is None:
            return self._unparsed(item_id, obj_idx, "score call failed")
        return yes_prob, ""

    def _unparsed(self, item_id: str, obj_idx: int, detail: str) -> Tuple[float, str]:
        """The explicit "unparsed" verify outcome: score 0.0 plus a diagnostic.

        A verification the VLM never answered is NOT a verification.
        """
        if self.verbose:
            log.warning(
                f"[{item_id}] GroundObj verify UNPARSED obj {obj_idx} after "
                f"{VERIFY_ATTEMPTS} attempt(s): {detail[:120]!r}"
            )
        return 0.0, f"{UNPARSED_FEEDBACK}: {detail[:200]}"

    # -- refine grounding via VLM (fail-recovery probe) -------------------

    def _refine_grounding(
        self,
        phrase: str,
        old_desc: str,
        prior_cam_ids: List[int],
        item_id: str,
        rejected_boxes: Optional[Dict[int, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single fail-recovery probe — used by both no_candidates and no_verify.

        Shows the VLM all N scene images (with rejected SAM3 boxes annotated
        in red when available) and asks for a revised grounding: a new
        description AND a list of cam_ids where the object is actually
        visible. Returns ``None`` if VLM is unavailable or the response is
        unparseable.

        On success, returns ``{"description": str, "cam_ids": List[int],
        "reason": str}``. ``cam_ids`` is empty when the VLM cannot locate
        the object in any view — caller should not retry the same phrase.
        ``reason`` is short free-form text suitable for surfacing as
        ``GroundingResult.verify_feedback`` to the planner-retry layer.
        """
        if not hasattr(self.vlm, "_query"):
            return None

        n = len(self.scene.images)
        imgs = self._refine_images(rejected_boxes)
        prompt = self._refine_prompt(phrase, old_desc, prior_cam_ids, rejected_boxes, n)

        try:
            raw = self.vlm._query(imgs, prompt, max_new_tokens=256)
        except Exception as e:
            if self.verbose:
                log.error(f"[{item_id}] _refine_grounding VLM error: {e}")
            return None

        return self._parse_revision(raw, n)

    def _refine_images(
        self, rejected_boxes: Optional[Dict[int, Any]],
    ) -> List[Image.Image]:
        """All scene images, with the rejected candidates' boxes drawn in red."""
        imgs: List[Image.Image] = []
        for vi, img in enumerate(self.scene.images):
            if rejected_boxes and vi in rejected_boxes:
                imgs.append(_render_bbox(img, rejected_boxes[vi]))
            else:
                imgs.append(
                    Image.fromarray(img) if isinstance(img, np.ndarray) else img
                )
        return imgs

    @staticmethod
    def _refine_prompt(
        phrase: str,
        old_desc: str,
        prior_cam_ids: List[int],
        rejected_boxes: Optional[Dict[int, Any]],
        n: int,
    ) -> str:
        """The fail-recovery prompt over *n* images.

        Its context lines say what was tried, so the VLM produces a useful
        revision rather than re-suggesting the same description.
        """
        context_lines: List[str] = []
        if rejected_boxes:
            view_list = ", ".join(str(v) for v in sorted(rejected_boxes.keys()))
            context_lines.append(
                f'A detector flagged the red-boxed regions in image(s) '
                f'{view_list} as candidates for "{phrase}", but verification '
                f'rejected all of them.'
            )
        else:
            context_lines.append(
                f'A detector could not find any candidates for "{phrase}".'
            )
        if prior_cam_ids:
            cam_str = ", ".join(str(c) for c in prior_cam_ids)
            context_lines.append(
                f'Planner-suggested cam_id(s): {cam_str}. '
                f'Prior description: "{old_desc}".'
            )

        return (
            "\n".join(context_lines) + "\n\n"
            f'Look at all {n} images (indexed 0..{n-1}). Where IS "{phrase}" '
            f'actually visible?\n\n'
            f'Respond in JSON only — no markdown fences, no extra text:\n'
            f'{{\n'
            f'  "description": "<short concrete description (10-20 words); '
            f'must DIFFER from the prior one above>",\n'
            f'  "cam_ids": [<list of 0..{n-1} image indices where the object '
            f'is visible; empty list [] if not visible in any view>],\n'
            f'  "reason": "<one short sentence: visual cue that justifies '
            f'your choice, OR why the object isn\'t visible>"\n'
            f'}}'
        )

    @classmethod
    def _parse_revision(cls, raw: Any, n: int) -> Optional[Dict[str, Any]]:
        """The probe's JSON answer as ``{"description", "cam_ids", "reason"}``, or None."""
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if not isinstance(raw, str) or not raw.strip():
            return None

        obj = cls._load_json_object(raw.strip())
        if not isinstance(obj, dict):
            return None

        desc = str(obj.get("description") or "").strip().strip('"').strip("'")
        if len(desc) > 200:
            desc = desc[:200]
        cam_ids = cls._valid_cam_ids(obj.get("cam_ids"), n)
        reason = str(obj.get("reason") or "").strip()
        if len(reason) > 240:
            reason = reason[:240]
        return {"description": desc, "cam_ids": cam_ids, "reason": reason}

    @staticmethod
    def _load_json_object(text: str) -> Any:
        """JSON from *text*: a fenced block's body, else the text, else from its first ``{``."""
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        obj = None
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            idx = text.find("{")
            if idx >= 0:
                try:
                    obj = json.loads(text[idx:])
                except (json.JSONDecodeError, ValueError):
                    pass
        return obj

    @staticmethod
    def _valid_cam_ids(raw_cams: Any, n: int) -> List[int]:
        """Distinct view indices in ``[0, n)`` from a list of ints or a single int."""
        cam_ids: List[int] = []
        if isinstance(raw_cams, list):
            for c in raw_cams:
                try:
                    ci = int(c)
                except (TypeError, ValueError):
                    continue
                if 0 <= ci < n and ci not in cam_ids:
                    cam_ids.append(ci)
        elif isinstance(raw_cams, int) and 0 <= raw_cams < n:
            cam_ids = [raw_cams]
        return cam_ids

    # -- helpers -----------------------------------------------------------

    def _already_in_scene(self, phrase: str) -> bool:
        """Check if an object matching *phrase* is already detected."""
        return self._find_object_by_label(phrase) is not None

    def _find_object_by_label(self, phrase: str) -> Optional[int]:
        """Return the first scene-object index whose label EXACTLY matches *phrase*.

        Exact-key reuse only: case-insensitive equality after stripping a
        leading ``"the "`` on both sides. Substring matching would let a
        phrase like ``"room"`` match any label containing ``"room"`` and
        suppress a needed detection; two *different* phrases go through the
        detect → verify → refine loop independently.

        Returns ``None`` if no exact match exists.
        """
        clean = re.sub(r"^the\s+", "", phrase.lower()).strip()
        if not clean:
            return None
        for idx, obj in enumerate(self.scene.objects):
            lab = str(getattr(obj, "label", "") or "").lower().strip()
            if not lab:
                continue
            lab_clean = re.sub(r"^the\s+", "", lab).strip()
            if lab_clean == clean:
                return idx
        return None

    def _normalize_cam_id(self, raw: Any) -> Optional[int]:
        """Planner cam_id → a valid view index, or None (scan all views).

        Accepts an int, a numeric string / numpy int, or a list of those (its
        first entry is used). Anything unparseable or outside
        ``[0, len(scene.images))`` becomes None: ``scene.detect`` raises on it.
        """
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if raw is None:
            return None
        try:
            cam = int(raw)
        except (TypeError, ValueError):
            return None
        return cam if 0 <= cam < len(self.scene.images) else None

    def _nms(self, item_id: str, tag: str) -> int:
        """Scene NMS that keeps every index this grounder handed out valid.

        ``scene.nms_objects`` removes duplicates and re-indexes the rest, so
        ``GroundingResult.obj_indices`` (and region member lists) are
        re-resolved here by object identity; an index whose object NMS
        removed moves to the surviving object with the same label (NMS only
        merges same-label duplicates).

        Region entities are excluded from NMS: their union box overlaps their
        own members by construction and their label contains a member's, so
        NMS would otherwise delete a region as a "duplicate" of its member.
        Blanking the label is enough -- NMS never pairs an empty label.
        """
        objs = list(self.scene.objects)

        def refs(indices):
            return [objs[i] for i in indices if 0 <= i < len(objs)]

        tracked = [
            (r, refs(r.obj_indices), refs(r.region_members))
            for r in self._session_results
        ]
        verified = [
            (objs[i], v) for i, v in self._obj_verification.items()
            if 0 <= i < len(objs)
        ]
        regions = [o for o in objs if _is_region_entity(o)]
        region_members = [(o, refs(o.metadata.get("member_indices", []))) for o in regions]
        labels = [(o, o.label) for o in regions]
        for o in regions:
            o.label = ""
        try:
            removed = self.scene.nms_objects(
                iou_threshold=0.5,
                label_similarity_threshold=0.4,
                verbose=self.verbose,
            )
        finally:
            for o, lab in labels:
                o.label = lab
        if not removed:
            return 0

        pos = {id(o): i for i, o in enumerate(self.scene.objects)}

        def reindex(objects):
            out: List[int] = []
            for o in objects:
                i = pos.get(id(o))
                if i is None:
                    i = self._find_object_by_label(str(getattr(o, "label", "")))
                if i is not None and i not in out:
                    out.append(i)
            return out

        for r, obj_refs, member_refs in tracked:
            r.obj_indices = reindex(obj_refs)
            r.region_members = reindex(member_refs)
        for o, member_refs in region_members:
            o.metadata["member_indices"] = reindex(member_refs)
        self._obj_verification = {
            pos[id(o)]: v for o, v in verified if id(o) in pos
        }
        if self.verbose:
            log.info(
                f"[{item_id}] GroundObj {tag} NMS: removed {removed} "
                f"→ {self.scene.objects_count} objects"
            )
        return removed

    def _remove_objects(self, indices: List[int]) -> None:
        """Remove objects at *indices* from the scene and reindex remaining.

        Also rebuilds ``self._obj_verification`` so its keys stay aligned
        with the new scene-object indices. Removed entries are dropped.
        """
        if not indices:
            return
        to_remove = set(indices)
        self.scene.objects = [
            obj for idx, obj in enumerate(self.scene.objects)
            if idx not in to_remove
        ]
        for idx, obj in enumerate(self.scene.objects):
            obj.id = idx

        # Rebuild verification map with shifted indices.
        if self._obj_verification:
            new_map: Dict[int, Tuple[bool, float]] = {}
            for old_idx, val in self._obj_verification.items():
                if old_idx in to_remove:
                    continue
                shift = sum(1 for r in to_remove if r < old_idx)
                new_map[old_idx - shift] = val
            self._obj_verification = new_map


__all__ = ["ObjectGrounder", "GroundingResult"]
