"""Pipeline step 3: Pre-execution object detection from groundings."""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
from saturn.log import get_logger

log = get_logger(__name__)

# Rounds in which the planner may revise the groundings the grounder could not verify.
_MAX_PLANNER_RETRIES = 3


def pre_detect_objects(
    scene,
    parsed: Optional[Dict],
    item_id: str,
    sample_result: Dict,
    question: str = "",
    planner: Any = None,
    images: Optional[List[Any]] = None,
) -> None:
    """Ensure objects mentioned in planner groundings exist in the scene.

    Delegates to :class:`ObjectGrounder` for a VLM + SAM3 detect → verify →
    refine loop.  Falls back to a simple SAM3/VLM detect when the VLM is not
    available on the scene.

    If ``planner`` and ``images`` are provided AND the first grounding pass
    leaves any phrase unverified, the planner is invoked via
    :meth:`QueryPlanner.refine_groundings` to revise the failed entries
    (description and/or cam_id) and the grounder is re-run for those phrases.
    """
    if not parsed or not parsed.get("object_groundings"):
        return

    from saturn.vlm.grounding import ObjectGrounder

    groundings = parsed["object_groundings"]
    vlm = getattr(scene, "_vlm", None)

    if vlm is not None:
        grounder = ObjectGrounder(
            vlm=vlm,
            scene=scene,
            question=question,
            max_refine=2,
            verify_threshold=0.4,
            verbose=True,
        )
        results = grounder.ground_all(groundings, item_id=item_id)
        plan = _plan_entries(groundings, results)
        failed_idx = [i for i, r in enumerate(results) if r.last_diagnostic]
        if failed_idx and planner is not None and images is not None:
            _recover_failed_groundings(grounder, planner, question, images, results, plan,
                                       failed_idx, item_id, sample_result)
        sample_result["grounding_results"] = _grounding_records(results)
        sample_result["zero_verified_region_count"] = _count_zero_score_verified(
            sample_result.get("grounding_results", []))
    else:
        _detect_without_verification(scene, groundings, item_id)

    sample_result["objects_count_after_predetect"] = scene.objects_count
    sample_result["objects_count_after_nms"] = scene.objects_count


def _plan_entries(groundings: List[Dict], results: List) -> List[Dict]:
    """The planner grounding behind each grounder result.

    ``ground_all`` drops skipped groundings ("already in scene"), so
    results[i] is NOT groundings[i]. Pair each result with its own planner
    grounding by phrase.
    """
    by_phrase = {
        str(g.get("phrase", "")).strip().lower(): g for g in groundings
    }
    return [by_phrase.get(r.phrase.strip().lower(), {}) for r in results]


def _recover_failed_groundings(grounder, planner, question: str, images: List, results: List,
                               plan: List[Dict], failed_idx: List[int], item_id: str,
                               sample_result: Dict) -> None:
    """Let the planner revise the unverified groundings, then salvage what is still unresolved.

    Updates ``results`` and ``plan`` in place. Salvage only runs after all
    retry rounds, so a low-confidence SAM3 box is not accepted while the
    planner can still find a better phrasing. Any error is logged and leaves
    the groundings as they are.
    """
    try:
        failed_idx, retry_round, total_retry_time = _retry_with_planner(
            grounder, planner, question, images, results, plan, failed_idx, item_id)
        sample_result["planner_retry_time_s"] = round(total_retry_time, 2)
        sample_result["planner_retry_rounds"] = retry_round

        n_salvaged = _salvage_unresolved(grounder, results, plan, failed_idx, item_id)
        if n_salvaged:
            sample_result["salvaged_count"] = n_salvaged
            log.info(
                f"[{item_id}] salvage: {n_salvaged} phrase(s) "
                f"recovered SAM3-best-candidate (no verify) "
                f"after {retry_round} retry round(s)"
            )
    except Exception as e:
        log.error(f"[{item_id}] planner-retry error: {e}")


def _retry_with_planner(grounder, planner, question: str, images: List, results: List,
                        plan: List[Dict], failed_idx: List[int],
                        item_id: str) -> Tuple[List[int], int, float]:
    """Feed the unverified phrases back to the planner and re-ground its revisions.

    Each of up to _MAX_PLANNER_RETRIES rounds gives the planner another chance
    to revise descriptions / cam_ids based on the latest verify scores.
    Returns (still-failed indices, rounds run, seconds spent in the planner).
    """
    total_retry_time = 0.0
    retry_round = 0
    while failed_idx and retry_round < _MAX_PLANNER_RETRIES:
        retry_round += 1
        failures, prior_with_status = _retry_request(results, plan, failed_idx)
        t_retry = time.time()
        revised = planner.refine_groundings(
            question,
            images,
            prior_groundings=prior_with_status,
            failed_groundings=failures,
        )
        total_retry_time += time.time() - t_retry
        # Keep only revisions for phrases that failed: a revision of a verified
        # entry would be re-grounded by `ground_all(revised)` and could disturb
        # a good grounding.
        failed_phrases_lc = {
            results[i].phrase.strip().lower() for i in failed_idx
        }
        revised = [
            r for r in revised
            if r.get("phrase", "").strip().lower() in failed_phrases_lc
        ]
        if not revised:
            log.warning(
                f"[{item_id}] planner-retry {retry_round}/"
                f"{_MAX_PLANNER_RETRIES}: {len(failed_idx)} failed → "
                "0 revised (planner produced no usable revision)"
            )
            break
        log.info(
            f"[{item_id}] planner-retry {retry_round}/"
            f"{_MAX_PLANNER_RETRIES}: {len(failed_idx)} failed → "
            f"{len(revised)} revised ({round(time.time() - t_retry, 2)}s)"
        )
        _reground_revisions(grounder, revised, results, plan, failed_idx, item_id)
        failed_idx = [
            i for i in failed_idx if results[i].last_diagnostic
        ]
    return failed_idx, retry_round, total_retry_time


def _retry_request(results: List, plan: List[Dict],
                   failed_idx: List[int]) -> Tuple[List[Dict], List[Dict]]:
    """What the planner sees in a retry round: (failed groundings, all groundings with status).

    The failed entries carry the grounder's diagnostics; every entry carries its
    verify status, so the planner sees which already succeeded (and at what
    confidence). Both are indexed over ``results``, which omits groundings the
    grounder skipped as duplicates ("already in scene").
    """
    failures = [
        {
            **plan[i],
            "diagnostic": results[i].last_diagnostic,
            "n_candidates": results[i].n_candidates,
            "verify_score": results[i].verify_score,
            "verify_feedback": results[i].verify_feedback,
        }
        for i in failed_idx
    ]
    prior_with_status = [
        {
            **plan[i],
            "verified": bool(results[i].verified),
            "verify_score": float(results[i].verify_score or 0.0),
        }
        for i in range(len(results))
    ]
    return failures, prior_with_status


def _reground_revisions(grounder, revised: List[Dict], results: List, plan: List[Dict],
                        failed_idx: List[int], item_id: str) -> None:
    """Ground the planner's revisions and put each in place of the failed result it revises."""
    retry_results = grounder.ground_all(revised, item_id=item_id)
    retry_by_phrase = {
        r.phrase.strip().lower(): r for r in retry_results
    }
    revised_by_phrase = {
        g.get("phrase", "").strip().lower(): g for g in revised
    }
    for idx in failed_idx:
        orig_phrase = results[idx].phrase.strip().lower()
        new_r = retry_by_phrase.get(orig_phrase)
        if new_r is not None:
            results[idx] = new_r
            # Keep plan[idx] in step with results[idx], so the
            # next round and salvage use the revised grounding.
            plan[idx] = revised_by_phrase.get(orig_phrase, plan[idx])


def _salvage_unresolved(grounder, results: List, plan: List[Dict], failed_idx: List[int],
                        item_id: str) -> int:
    """Give each still-unresolved phrase SAM3's best candidate without verification.

    Trades precision for recall so that every phrase has an object and
    codegen can run. Returns the number of phrases salvaged.
    """
    n_salvaged = 0
    for idx in failed_idx:
        r = results[idx]
        if not r.last_diagnostic:
            continue
        g = plan[idx]
        # A region's description is a comma list of constituents;
        # salvaging it would box one constituent under the region's
        # label. Salvage the region phrase as one object instead.
        description = (
            r.phrase if r.is_region
            else g.get("description", "") or r.final_description
        )
        salvaged_idx = grounder.salvage(
            phrase=r.phrase,
            description=description,
            cam_id=g.get("cam_id", r.cam_id),
            item_id=item_id,
        )
        if salvaged_idx is not None:
            r.obj_indices = [salvaged_idx]
            r.verified = False
            r.verify_score = 0.0
            r.last_diagnostic = "salvaged"
            n_salvaged += 1
    return n_salvaged


def _grounding_records(results: List) -> List[Dict]:
    """The grounding results as stored in the sample record."""
    return [
        {
            "phrase": r.phrase,
            "description": r.final_description or r.description,
            "is_region": r.is_region,
            "verified": r.verified,
            "attempts": r.attempts,
            "score": r.verify_score,
            "obj_indices": r.obj_indices,
            "region_members": r.region_members,
            "diagnostic": r.last_diagnostic,
            "verify_feedback": r.verify_feedback,
        }
        for r in results
    ]


def _count_zero_score_verified(grounding_records: List[Dict]) -> int:
    """Region groundings marked verified with a zero score (suspect regions)."""
    return sum(
        1 for g in grounding_records
        if g.get("is_region") and g.get("verified") and float(g.get("score") or 0.0) <= 0.0
    )


def _detect_without_verification(scene, groundings: List[Dict], item_id: str) -> None:
    """Minimal fallback without a VLM: detect and label each grounding, no verify loop."""
    pre_count = scene.objects_count
    for g in groundings:
        phrase = str(g.get("phrase", "")).strip()
        desc = str(g.get("description", "")).strip()
        if not phrase or not desc:
            continue
        clean_label = re.sub(r"^the\s+", "", phrase.lower()).strip()
        try:
            new_ids = scene.detect(desc)
            if new_ids:
                for ni in new_ids:
                    scene.objects[ni].label = clean_label
                log.info(f"[{item_id}] Pre-detect '{phrase}': {len(new_ids)} obj(s)")
        except Exception as e:
            log.error(f"[{item_id}] Pre-detect failed '{phrase}': {e}")
    if scene.objects_count > pre_count:
        scene.nms_objects(iou_threshold=0.5, label_similarity_threshold=0.4, verbose=True)
