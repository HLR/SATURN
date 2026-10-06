"""Per-sample pipeline: pre_execute (phase A) -> execute (phase B).

Phase A plans the question, builds the scene, grounds the planner's objects and
generates the program; phase B runs the program and records the answer.
"""

from saturn.settings import env
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from saturn.pipeline.evaluate import vlm_fallback_answer
from saturn.pipeline.execute import execute_with_retry
from saturn.pipeline.generate import generate_code
from saturn.pipeline.ground import pre_detect_objects
from saturn.pipeline.models import Models
from saturn.pipeline.plan import plan_question
from saturn.pipeline.state import _PreExecState
from saturn.pipeline.template import CODE_TEMPLATE
from saturn.log import get_logger

log = get_logger(__name__)

# The class-agnostic SAM3 prompt that proposes every object of the scene, for questions the
# unified planner marks "search" (planner_prompt_unified.py): the program lets every object take
# every role and checks each description itself with score().
PROPOSAL_PROMPT = "object"

# Item fields that may hold the image paths, in the order they are tried.
_IMAGE_PATH_FIELDS = ("image_file_name", "image_paths", "image_files", "image_filenames",
                      "images_path")


def pre_execute(
    item: Dict,
    item_id: str,
    models: Models,
    args,
    *,
    build_scene_fn,
) -> _PreExecState:
    """Phase A: setup + plan + scene + detect + code-gen.

    Uses planner-VLM, VGGT, oriany, SAM3, and the code-gen LLM — but NOT the
    main VLM scoring path. Phase B (``execute``) does the scoring.
    Splitting the two phases lets the async runner pipeline a previous
    sample's scoring against this phase for the next sample.
    """
    question = item.get("query", "")
    gt_answer = item.get("answer", "")
    images = item["images"]
    _tag_image_paths(item, images, item_id)
    result = _new_result(item, item_id, args, images)

    if not question or not gt_answer:
        result["error"] = "Missing question or GT answer"
        return _PreExecState(early_return=True, result=result)

    # --- Step 1: Plan question (reference frame + groundings + detection flag) ---
    clarification = plan_question(
        question, images, models, args, item_id, result
    )
    parsed = clarification.get("parsed")
    objects = _object_source(parsed, item_id, result)

    # --- Step 2: Build scene ---
    scene = _build_scene(images, models, build_scene_fn, objects, parsed, result)
    if getattr(args, "use_pose_constraints", False):
        _apply_pose_constraints(scene, question, models, item_id, result)
    if not objects.needs_detection and objects.source != "search":
        log.info(f"[{item_id}] Detection skipped (no planner groundings)")
    log.info(
        f"[{item_id}] Scene built: {scene.objects_count} objects, "
        f"{scene.num_cameras} cameras ({result['scene_build_time_s']}s)"
    )

    # --- Step 3: Detect missing grounded objects (GROUNDING stage) ---
    if objects.needs_detection:
        _ground_objects(scene, parsed, question, models, images, item_id, result)
    _dump_scene(scene, args, item_id)

    # --- Step 4: Generate code ---
    code_snippet = _generate_program(question, models, clarification, scene, item_id, result)
    if not code_snippet:
        result["error"] = "Code generation failed."
        log.error(f"[{item_id}] Code generation failed.")
        return _PreExecState(early_return=True, result=result)

    return _PreExecState(
        early_return=False,
        result=result,
        clarification=clarification,
        scene=scene,
        code_snippet=code_snippet,
    )


def _tag_image_paths(item: Dict, images: List, item_id: str) -> None:
    """Tag each image with its source path (as ``image._sapy_path``).

    PIL drops ``.filename`` on convert(), and later stages need to know which
    files an image came from: the scene cache (build_scene.py) keys on these
    paths, and the fusion-calibration dump records them.
    """
    paths = None
    for key in _IMAGE_PATH_FIELDS:
        value = item.get(key)
        if (isinstance(value, (list, tuple)) and len(value) == len(images)
                and all(isinstance(x, str) for x in value)):
            paths = value
            break
    if paths is None:  # last resort: any list-of-strings field with the right length
        for value in item.values():
            if isinstance(value, (list, tuple)) and len(value) == len(images) and all(
                isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg"))
                for x in value
            ):
                paths = value
                break
    for image, path in zip(images, paths or []):
        try:
            image._sapy_path = str(path)
        except Exception:
            log.debug(f"[{item_id}] suppressed: could not tag image with its path", exc_info=True)


def item_fields(item: Dict, item_id: str) -> Dict:
    """The fields every result record copies from its dataset item, error records included."""
    return {
        "id": str(item_id),
        "query": item.get("query", ""),
        "subset": item.get("subset", ""),
        "image_file_name": item.get("image_file_name", []),
        "ground_truth_answer": item.get("answer", ""),
        "question_type": item.get("question_type", ""),
    }


def _new_result(item: Dict, item_id: str, args, images: List) -> Dict:
    """The sample's result record, with every field the reports read at its default."""
    return {
        **item_fields(item, item_id),
        "vlm_model_name": args.vlm_model_name,
        "backend": "vggt",
        "num_images": len(images),
        "error": "",
        "program_code": None,
        "code_generated_answer": None,
        "final_answer_text": None,
        "correct_final_answer": None,
        "execution_successful": False,
        "objects_count": 0,
        "num_cameras": 0,
        "scene_build_time_s": None,
        "exec_time_s": None,
        "disambiguation": [],
        # --- Accounting fields for split-path reporting ---
        "execution_failed": False,
        "vlm_fallback_used": False,
        "vlm_fallback_produced_letter": False,
        "symbolic_answer_text": None,
        "objects_count_before_predetect": 0,
        "objects_count_after_predetect": 0,
        "objects_count_after_nms": 0,
        "scene_region_auto_detect_count": 0,
        "zero_verified_region_count": 0,
        "anchor_invalid_events": [],
    }


@dataclass
class _ObjectSource:
    """Where the scene's objects come from, as decided from the plan."""
    source: str                            # the planner's "objects" field
    keywords: Optional[List[str]]          # descriptions of the named groundings
    unique_keywords: Set[str]              # the subset marked unique
    build_keywords: Optional[List[str]]    # SAM3 prompts used while the scene is built
    needs_detection: bool                  # run the grounder after the scene is built


def _object_source(parsed: Optional[Dict], item_id: str, result: Dict) -> _ObjectSource:
    """Decide where the scene's objects come from and record it in ``result``.

    The planner's "objects" field ("named" when the plan is missing or does not write it):
        named      each grounding is detected and checked by the VLM (Step 3)
        search     SAM3 proposes every object for PROPOSAL_PROMPT while the scene is built
        no_object  nothing is detected; the program reads camera poses only
    """
    source = (parsed or {}).get("objects", "named")
    groundings = ((parsed or {}).get("object_groundings") or []) if source == "named" else []
    keywords = []
    unique_keywords = set()
    needs_detection = True if parsed is None else bool(groundings)
    for g in groundings:
        desc = str(g.get("description", "")).strip()
        if desc:
            keywords.append(desc)
            if g.get("unique", True) is not False:
                unique_keywords.add(desc)
    if not keywords:
        keywords = None
    else:
        log.info(f"[{item_id}] Scene keywords from planner: {keywords}")
    build_keywords = [PROPOSAL_PROMPT] if source == "search" else None
    if build_keywords:
        log.info(f"[{item_id}] Scene keywords from the planner (search): {build_keywords}")
    result["planner_objects"] = source
    result["scene_keywords"] = keywords or build_keywords or []
    result["needs_detection"] = needs_detection
    return _ObjectSource(source, keywords, unique_keywords, build_keywords, needs_detection)


def _build_scene(images: List, models: Models, build_scene_fn, objects: _ObjectSource,
                 parsed: Optional[Dict], result: Dict):
    """Reconstruct the scene (VGGT + SAM3 + orientation) and attach the VLM and the plan."""
    t_scene = time.time()
    scene = build_scene_fn(
        images, models,
        keywords=objects.build_keywords,
        unique_keywords=objects.unique_keywords or None,
    )
    result["scene_build_time_s"] = round(time.time() - t_scene, 2)
    result["objects_count"] = scene.objects_count
    result["objects_count_before_predetect"] = scene.objects_count
    result["num_cameras"] = scene.num_cameras
    scene._vlm = models.vl_model

    # The planner's setup_caption reaches codegen through scene.dump_facts_str();
    # the object groundings reach it through their own prompt block.
    if parsed is not None:
        scene.set_planner_context(
            setup_caption=parsed.get("setup_caption"),
        )
    return scene


def _apply_pose_constraints(scene, question: str, models: Models, item_id: str,
                            result: Dict) -> None:
    """Opt-in (--use_pose_constraints): read camera-pose constraints from the question.

    A separate VLM pass extracts statements such as "rotated 90° clockwise" or
    "same spot" and applies them through ``scene.constraint`` before codegen.
    """
    try:
        from saturn.planning.constraint_extractor import (
            question_has_pose_keywords,
        )

        if question_has_pose_keywords(question):
            t_pose = time.time()
            extracted = _extract_pose_constraints(scene, question, models, item_id)
        else:
            extracted = None
        if extracted is None:
            result["pose_constraints_extracted"] = []
            return
        result["pose_constraints_extracted"] = list(extracted)
        result["pose_constraints_extract_time_s"] = round(time.time() - t_pose, 2)
        # Each call re-solves the camera poses inside scene.constraint.
        n_applied = 0
        for rec in extracted:
            n_applied += _apply_pose_record(scene, rec, item_id)
        if extracted:
            n_skipped = len(extracted) - n_applied
            log.info(
                f"[{item_id}] pose-constraints: applied {n_applied} record(s)"
                + (f", skipped {n_skipped}" if n_skipped else "")
                + f" ({result['pose_constraints_extract_time_s']}s)"
            )
    except Exception as _pe:  # noqa: BLE001
        # Best-effort, but a failure must stay distinguishable from
        # "the question had no pose keywords" in the result record.
        log.error(f"[{item_id}] pose-constraints: extraction failed ({_pe}); continuing")
        result["pose_constraints_extracted"] = []
        result["pose_constraints_error"] = f"{type(_pe).__name__}: {_pe}"[:300]


def _extract_pose_constraints(scene, question: str, models: Models, item_id: str):
    """The VLM's pose-constraint records for this question ([] without the sync bridge)."""
    vlm_agent = models.vl_model
    # Sync wrapper installed at startup by
    # saturn.pipeline.models._install_vlm_sync_bridge; without it,
    # no constraints are extracted.
    extract_fn = getattr(vlm_agent, "extract_pose_constraints", None)
    if extract_fn is None:
        log.warning(f"[{item_id}] pose-constraints: bridge not installed; skip")
        return []
    return extract_fn(
        question=question,
        images=scene.images,
        num_cameras=scene.num_cameras,
    )


def _apply_pose_record(scene, rec: Dict, item_id: str) -> bool:
    """Apply one extracted record; returns whether it was applied.

    A malformed record or one of an unknown type is logged and skipped.
    """
    try:
        if rec["type"] == "rotation":
            scene.constraint.rotation(
                scene.cameras[rec["from_cam"]],
                scene.cameras[rec["to_cam"]],
                yaw=float(rec["yaw"]),
                axis=rec.get("axis", "up"),
            )
            return True
        if rec["type"] == "same_position":
            scene.constraint.same_position(
                *[scene.cameras[i] for i in rec["cams"]]
            )
            return True
        log.warning(f"[{item_id}] pose-constraints: skipping {rec} (unknown type)")
    except Exception as _ce:  # noqa: BLE001
        log.warning(f"[{item_id}] pose-constraints: skipping {rec} ({_ce})")
    return False


def _ground_objects(scene, parsed: Optional[Dict], question: str, models: Models,
                    images: List, item_id: str, result: Dict) -> None:
    """Detect and verify the planner's object groundings in the built scene.

    Timed separately so the runtime breakdown can report grounding distinctly
    from planner and code-gen (all three are part of "program generation").
    """
    t_ground = time.time()
    pre_detect_objects(
        scene, parsed, item_id, result, question=question,
        planner=getattr(models, "planner", None),
        images=images,
    )
    result["grounding_time_s"] = round(time.time() - t_ground, 2)
    result["objects_count"] = scene.objects_count

    if scene.objects_count == 0:
        # No objects detected, but the scene still has cameras populated by
        # VGGT. Camera-only questions (e.g. "did I move from view 1 to
        # view 2") can be answered purely from scene.cameras[k] without
        # any scene.objects[i], so the program runs against the camera-only
        # scene. If it reads a missing object, the execution error handler
        # surfaces a normal failure rather than a silent None.
        log.info(f"[{item_id}] No objects detected; proceeding with camera-only scene.")
        result["objects_count_after_predetect"] = 0
        result["empty_objects_scene"] = True


def _dump_scene(scene, args, item_id: str) -> None:
    """Write the grounded scene (with point clouds) for the debug reports.

    Writing a dump holds the GIL for seconds, so only this post-detection dump
    is written, and only when reports (or SAPY_SCENE_DUMPS=1) are requested.
    """
    results_dir = os.path.join("experiments", args.dataset, args.vlm_model_name)
    scene_dump_dir = os.path.join(results_dir, "scenes")
    if not bool(getattr(args, "generate_reports", False) or env("SAPY_SCENE_DUMPS") == "1"):
        return
    os.makedirs(scene_dump_dir, exist_ok=True)
    try:
        scene.dump(os.path.join(scene_dump_dir, f"{item_id}.json"), include_points=True)
    except Exception as e:
        log.warning(f"[{item_id}] WARNING: scene.dump() failed: {e}")


def _generate_program(question: str, models: Models, clarification: Dict, scene,
                      item_id: str, result: Dict) -> Optional[str]:
    """The program body for this question: a pinned program, else the code LLM's.

    Timed as its own stage so the runtime breakdown can report perception
    (scene_build) / program-generation (codegen) / symbolic-execution (exec)
    separately.
    """
    t_code = time.time()
    pinned = getattr(models, "pinned_programs", None)
    if pinned is not None and str(item_id) in pinned:
        code_snippet = raw_code = pinned[str(item_id)]
        result["program_source"] = "pinned"
    else:
        code_snippet, raw_code = generate_code(
            question,
            models.code_generator,
            clarification,
            scene=scene,
        )
    result["codegen_time_s"] = round(time.time() - t_code, 2)
    if raw_code:
        result["raw_llm_code"] = raw_code
    return code_snippet


def execute(
    item: Dict,
    item_id: str,
    models: Models,
    args,
    state: _PreExecState,
) -> Dict:
    """Phase B: Step 5 — execute the generated program with retry.

    All main-VLM scoring happens here. The async runner gates this with its own
    semaphore so vLLM contention stays bounded while phase A of the next
    sample runs concurrently on a different subsystem (VGGT/oriany/SAM3).
    """
    if state.early_return:
        return state.result

    # Reset the VLM agent's score-cache before scoring this sample. Resetting
    # here (not in pre_execute) keeps pipelined execution from clearing the
    # cache while a previous sample's execute phase is still reading it.
    if hasattr(models.vl_model, "clean_cache"):
        models.vl_model.clean_cache()

    result = state.result
    question = item.get("query", "")
    images = item["images"]

    # --- Step 5: Execute code (with retry, max 3) ---
    answer_str, cache, exec_error = _run_program(state, question, images, models, item_id)
    _record_predicted_bbox(result, state.scene, answer_str)
    if env("SAPY_DUMP_SCORES") and cache:
        result["unary_scores"] = _unary_scores(cache)
    _record_outcome(result, exec_error, answer_str, question, images, models, args, item_id)
    return result


def _run_program(state: _PreExecState, question: str, images: List, models: Models,
                 item_id: str):
    """Run the program (repairing it with the code LLM on failure) and record it.

    Returns (answer_str, score_trace, error_msg).
    """
    result = state.result
    t_exec = time.time()
    final_snippet, answer_str, cache, exec_error, retry_count = execute_with_retry(
        state.code_snippet, question,
        models.vl_model, state.scene, images,
        models.code_generator, state.clarification,
        item_id,
        max_retries=3,
    )
    result["exec_time_s"] = round(time.time() - t_exec, 2)
    result["retry_count"] = retry_count
    result["program_code"] = CODE_TEMPLATE.format(
        code=final_snippet.replace("\n", "\n    ")
    )
    result["code_generated_answer"] = answer_str
    return answer_str, cache, exec_error


def _record_predicted_bbox(result: Dict, scene, answer_str) -> None:
    """Record the box of the object the program selected.

    REF is graded on the prediction: the scene objects are SATURN's own
    detections and the answer index carries no relation to item['bboxes'].
    """
    try:
        _idx = int(float(str(answer_str).strip()))
        _obj = scene.objects[_idx] if 0 <= _idx < len(scene.objects) else None
        _bb = _obj.per_view_bboxes.get(0) if _obj is not None else None
        result["predicted_bbox_source"] = "detected" if _bb is not None else None
        if _bb is None and _obj is not None and scene.cameras:
            # No native detection in the query view: project the fused object
            # through camera 0 instead of scoring a correct answer as wrong.
            from saturn.scene.projection import project_object_bbox
            _img = (scene.images or [None])[0]
            if _img is not None and hasattr(_img, "size"):
                _bb = project_object_bbox(_obj, scene.cameras[0], _img.size[0], _img.size[1])
                if _bb is not None:
                    result["predicted_bbox_source"] = "projected"
        result["predicted_bbox"] = [float(v) for v in _bb] if _bb is not None else None
        result["predicted_obj_label"] = getattr(_obj, "label", None) if _obj is not None else None
    except (TypeError, ValueError, AttributeError, IndexError):
        result["predicted_bbox"] = None
        result["predicted_obj_label"] = None


def _unary_scores(cache) -> Dict:
    """The unary semantic scores of an execution trace (for SAPY_DUMP_SCORES).

    The full score trace is not serialized: each __and__ entry expands the
    joint-distribution tensor (objects^chain-length), which can reach tens of
    MB per entry, and the debug report shows a "No score cache" notice when the
    field is absent. The unary entries -- one (N,) vector per semantic
    predicate, a few hundred bytes -- are what attributes a wrong answer to the
    semantic channel rather than to geometry.

    ``cache`` is a list of records like
        {"action": "__init__",
         "inputs": {"args": "<question>", "self.tensor": [per-object scores]}}
    The unary predicates arrive as __init__ records whose tensor holds one
    score per object; everything else (the __and__ joints) is skipped.
    """
    unary = {}
    try:
        for rec in (cache if isinstance(cache, list) else []):
            if not isinstance(rec, dict):
                continue
            inp = rec.get("inputs") or {}
            q = inp.get("args")
            vec = inp.get("self.tensor")
            if not isinstance(q, str) or not isinstance(vec, list):
                continue
            if "__and__" in q or not vec or isinstance(vec[0], list):
                continue
            if len(vec) > 64:
                continue
            try:
                unary[q[:200]] = [round(float(v), 6) for v in vec]
            except (TypeError, ValueError):
                continue
    except Exception as _e:
        unary["_error"] = f"{type(_e).__name__}: {_e}"
    return unary


def _record_outcome(result: Dict, exec_error: Optional[str], answer_str, question: str,
                    images: List, models: Models, args, item_id: str) -> None:
    """Record the final answer, or the failure (with the VLM fallback on MMSI)."""
    if exec_error:
        result["error"] = exec_error
        result["execution_successful"] = False
        result["execution_failed"] = True
        # VLM fallback for MMSI
        if args.dataset.startswith("mmsi"):
            result["vlm_fallback_used"] = True
            fallback = vlm_fallback_answer(models.vl_model, images, question, item_id)
            if fallback:
                result["final_answer_text"] = fallback
                result["vlm_fallback"] = True
                result["vlm_fallback_produced_letter"] = True
    elif answer_str is None:
        result["error"] = "Code returned None."
        result["execution_successful"] = True
        result["symbolic_answer_text"] = None
    else:
        result["execution_successful"] = True
        result["final_answer_text"] = answer_str
        result["symbolic_answer_text"] = answer_str
