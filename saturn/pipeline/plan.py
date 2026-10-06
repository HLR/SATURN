"""Pipeline step 1: Plan question (reference frame + groundings)."""

import time
from typing import Dict, List

from PIL import Image

from saturn.pipeline.models import Models
from saturn.log import get_logger

log = get_logger(__name__)


def plan_question(
    question: str,
    scene_images: List[Image.Image],
    models: Models,
    args,
    item_id: str,
    sample_result: Dict,
) -> Dict:
    """Run the planner to get object groundings + setup caption.

    Returns a dict with keys:
        clarified_query_block, parsed

    The caller passes the planner output (setup_caption + object_groundings)
    to ``scene.set_planner_context(...)`` after build_scene runs; codegen
    reads it via ``scene.dump_facts_str()``.
    """
    result = {
        "clarified_query_block": "",
        "parsed": None,
    }

    # --- Planner (absent only in Models built without build_models_async) ---
    if models.planner is not None:
        t_cl = time.time()
        parsed = models.planner.clarify(question, scene_images)
        result["parsed"] = parsed

        if parsed:
            sample_result["planner"] = parsed

        t_key = "planner_time_s"
        sample_result[t_key] = round(time.time() - t_cl, 2)
        n_g = len(parsed.get("object_groundings", [])) if parsed else 0
        source = (parsed or {}).get("objects", "named")
        has_caption = bool((parsed or {}).get("setup_caption"))
        log.info(
            f"[{item_id}] planner: "
            f"{'hit' if parsed else 'miss'} "
            f"(objects={source}, groundings={n_g}, caption={has_caption}, {sample_result[t_key]}s)"
        )

    return result
