"""Pipeline step 5: Generate code."""

import re
from typing import Dict, Optional

from saturn.codegen import CodeGenerator
from saturn.log import get_logger

log = get_logger(__name__)


def generate_code(
    question: str,
    code_generator: CodeGenerator,
    clarification: Dict,
    *,
    scene=None,
) -> Optional[str]:
    """Ask the code LLM to generate a spatial reasoning program.

    Returns (code_snippet, raw_code) tuple. code_snippet has XML tags stripped;
    raw_code preserves the original output (may contain frame_analysis etc.).
    """
    scene_facts_block = ""
    # The facts block (setup caption, counts, cameras, axis convention) is part
    # of the rendered prompt and therefore of the content-addressed program-cache
    # key. It omits the per-object index table so programs ground objects by
    # name rather than by scene.objects[i] index.
    if scene is not None:
        try:
            scene_facts_block = scene.dump_facts_str()
        except Exception as e:
            log.warning(f"[generate_code] scene.dump_facts_str() failed: {e}")

    # Object groundings get their own codegen placeholder outside SCENE FACTS,
    # so the LLM reads them as task framing rather than scene context.
    object_groundings_block = ""
    parsed = clarification.get("parsed")
    if parsed:
        from saturn.planning.query_planner import QueryPlanner
        groundings = parsed.get("object_groundings") or []
        if groundings:
            object_groundings_block = QueryPlanner.format_groundings_block(groundings) or ""

    kwargs = dict(
        scene_facts_block=scene_facts_block,
        object_groundings_block=object_groundings_block,
        clarified_query_block=clarification.get("clarified_query_block", ""),
    )

    # Retry codegen up to 3 times on empty or unparseable output, so a transient
    # API failure does not drop the question. Retries set force_generate=True so
    # the cache does not replay the empty result.
    MAX_CODEGEN_ATTEMPTS = 3
    code_snippet = None
    _objects_str = None
    for attempt in range(1, MAX_CODEGEN_ATTEMPTS + 1):
        attempt_kwargs = dict(kwargs)
        if attempt > 1:
            attempt_kwargs["force_generate"] = True
            log.warning(f"[codegen-retry] empty result; attempt {attempt}/{MAX_CODEGEN_ATTEMPTS}")
        try:
            code_snippet, _objects_str = code_generator.generate_code(
                question, **attempt_kwargs
            )
        except Exception as e:
            log.error(f"[codegen-retry] attempt {attempt} raised: {e}")
            code_snippet = None
        if code_snippet:
            break

    raw_code = code_snippet  # Before stripping XML tags (may contain frame_analysis)
    if code_snippet:
        # Strip XML-like tags the LLM sometimes injects
        code_snippet = re.sub(
            r"</?(?:type_of_reasoning|steps|text|reasoning|code|frame_analysis)>.*",
            "",
            code_snippet,
        ).strip()

    return code_snippet or None, raw_code
