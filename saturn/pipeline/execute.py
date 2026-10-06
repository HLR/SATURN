"""Pipeline step 6: Execute code (with retry)."""

from saturn.settings import env
import ast
import os
import re
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from saturn.codegen import CodeGenerator
from saturn.soft_logic import ProbabilisticTensor, and_op, or_op
from saturn.pipeline.template import CODE_TEMPLATE
from saturn.pipeline.formula import make_formula_helpers
from saturn.log import get_logger

log = get_logger(__name__)

# Names bound in the scaffold (template.py) before the program body runs.
_SCAFFOLD_NAMES = frozenset({
    "query", "score_fn", "query_fn", "scene", "images", "history", "score", "objects_count",
    "camera",
})
# Line of the scaffold on which the program body starts (1-based).
_BODY_FIRST_LINE = CODE_TEMPLATE[: CODE_TEMPLATE.index("{code}")].count("\n") + 1


def _statements_outside_defs(nodes):
    """Walk statements without entering nested def / lambda / class bodies."""
    scoped = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    stack = [n for n in nodes if not isinstance(n, scoped)]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(c for c in ast.iter_child_nodes(n) if not isinstance(c, scoped))


def ensure_entry_called(snippet: str) -> str:
    """A program with no top-level ``return`` whose logic sits in a ``def`` that
    is never called (``def logic_executor(...)`` or ``def answer():``) returns
    None. Call that function: the last top-level def that nothing references and
    whose required parameters the scaffold can supply. Otherwise unchanged."""
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return snippet
    if any(isinstance(n, ast.Return) for n in _statements_outside_defs(tree.body)):
        return snippet
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for d in reversed(defs):
        others = [n for n in tree.body if n is not d]
        used = any(isinstance(x, ast.Name) and x.id == d.name
                   for o in others for x in ast.walk(o))
        if used:
            continue
        a = d.args
        pos = a.posonlyargs + a.args
        required = pos[: len(pos) - len(a.defaults)]
        kw_required = [k for k, dflt in zip(a.kwonlyargs, a.kw_defaults) if dflt is None]
        if any(p.arg not in _SCAFFOLD_NAMES for p in required + kw_required):
            continue
        call_args = [p.arg for p in required] + [f"{k.arg}={k.arg}" for k in kw_required]
        log.warning(f"generated program defines {d.name}() but never calls it; calling it")
        return snippet.rstrip() + f"\nreturn {d.name}({', '.join(call_args)})\n"
    return snippet


def _hint(e: BaseException) -> str:
    """One generic line on how to repair common API/Python mistakes."""
    msg = str(e)
    if isinstance(e, TypeError) and ("bitwise_" in msg or "ufunc 'invert'" in msg
                                     or "unsupported operand type(s) for &" in msg
                                     or "unsupported operand type(s) for |" in msg
                                     or "bad operand type for unary ~" in msg):
        return ("& | ~ combine formulas and truth values (pred[i], pred(camera(N))); for plain "
                "Python numbers use min(a, b), max(a, b), 1 - a.")
    if isinstance(e, AttributeError) and "'numpy.ndarray' object has no attribute" in msg:
        return ("positions and directions are plain numpy 3-vectors: use np.linalg.norm(v), "
                "v / np.linalg.norm(v), np.dot(a, b), np.cross(a, b), v[0], v[1], v[2].")
    if isinstance(e, TypeError) and "indices must be integers or slices, not str" in msg:
        return ("lists such as scene.objects take an integer index; a variable name such as \"x1\" "
                "only appears inside formulas. Bind it first: i = J.assign()[\"x1\"].")
    if isinstance(e, TypeError) and "Cannot convert multi-element ProbabilisticTensor" in msg:
        return ("a formula over a variable has one value per entity: read one entity with J[i] "
                "(or pred(i)), or its best binding with float(J.exists()).")
    if isinstance(e, IndexError) and "arrays used as indices" in msg:
        return ("predicates are indexed by an entity (an object index or camera(N)); only "
                "anchor.first_person.<dir>[p] reads a 3D point p.")
    return ""


def _missing_view_context(e, scene) -> str:
    """Name the object whose per-view data is missing and what that image shows."""
    from saturn.scene.types import MissingViewError
    if not isinstance(e, MissingViewError) or scene is None:
        return ""
    objs = list(getattr(scene, "objects", []) or [])
    owner = next((i for i, o in enumerate(objs) if getattr(o, e.field_name, None) is e.mapping), None)
    parts = []
    if owner is not None:
        parts.append(f"The object is scene.objects[{owner}] ({getattr(objs[owner], 'label', '?')!r}).")
    if isinstance(e.key, int) and not isinstance(e.key, bool):
        present = [f"{i} ({getattr(o, 'label', '?')!r})" for i, o in enumerate(objs)
                   if e.key in (getattr(o, e.field_name, None) or {})]
        parts.append(f"Objects detected in image {e.key + 1}: " + (", ".join(present) if present else "none") + ".")
    return " ".join(parts)


def format_execution_error(e: BaseException, snippet: str, scene=None) -> str:
    """The error as the program's author needs it: the exception, a repair hint,
    and the failing line numbered in the PROGRAM (not in the scaffold around it,
    which is never shown, so a retry does not re-declare logic_executor)."""
    lines = snippet.split("\n")
    out = [f"Execution Error: {type(e).__name__}: {e}"]
    extra = _missing_view_context(e, scene)
    if extra:
        out.append(extra)
    hint = _hint(e)
    if hint:
        out.append(f"Hint: {hint}")
    frames = traceback.extract_tb(e.__traceback__)
    started = False
    trace = []
    for f in frames:
        if f.filename == "<string>":
            started = True
            n = f.lineno - _BODY_FIRST_LINE + 1
            if 1 <= n <= len(lines):
                trace.append(f"  program line {n}: {lines[n - 1].strip()}")
        elif started:
            trace.append(f"  in {f.name}() [{os.path.basename(f.filename)}]")
    gen = [t for t in trace if t.startswith("  program line")]
    if gen:
        out.append("Failing " + gen[-1].strip()[len("program "):] + "   (line numbers count your program's lines)")
    if trace:
        out.append("Call chain:\n" + "\n".join(trace[-6:]))
    return "\n".join(out)


NO_ANSWER_ERROR = (
    "Execution Error: NoAnswer: the program finished without returning an answer (it returned None). "
    "The program is the body of one function: its top-level statements must end with "
    "`return max(options, ...)`; a def that is never called does not run.")


# Scene relations bound as globals of the program, in the order they are read.
_SPATIAL_RELATION_NAMES = (
    "left", "right", "above", "below", "front", "behind",
    "left_normalized", "right_normalized", "above_normalized", "below_normalized",
    "front_normalized", "behind_normalized",
    "obj_facing_left", "obj_facing_right", "obj_facing_front", "obj_facing_back",
    "obj_facing_up", "obj_facing_down",
    "facing", "parallel", "perpendicular", "orientation_distance",
    "between", "distance", "closeness", "distance_edge",
)


def execute_code(
    code_snippet: str,
    question: str,
    vl_model,
    scene,
    images: List[Image.Image],
) -> Tuple[Optional[str], Optional[list], Optional[str]]:
    """Execute a generated spatial program.

    Returns (answer_str, score_cache, error_msg); score_cache is the per-op score
    trace, recorded only under SAPY_DUMP_SCORES or SAPY_TRACE_SCORES=1.
    """
    spatial_relations = _spatial_relations(scene)
    code_snippet = ensure_entry_called(code_snippet)
    code_to_execute = CODE_TEMPLATE.format(
        code=code_snippet.replace("\n", "\n    ")
    )
    exec_context = _exec_context(scene, spatial_relations)
    syntax_error = _syntax_error(code_to_execute)
    if syntax_error is not None:
        return None, None, syntax_error

    _bind_token = None
    _tracing = False
    _held = {}
    error_msg = None
    cache = None
    result_str = None
    try:
        # All detection happens before execution: scene.detect / scene.ground are
        # no-ops while the program runs.
        _install_stand_ins(scene, _held)
        exec(code_to_execute, exec_context)
        logic_executor_func = exec_context["logic_executor"]
        # Record the per-op score trace ONLY when something will read it.
        # The trace serialises every operand and result with .tolist(), which
        # grows as N^k for a k-variable joint over N entities. Consumers: the
        # SAPY_DUMP_SCORES unary dump; SAPY_TRACE_SCORES=1 forces it on.
        _want_trace = bool(env("SAPY_DUMP_SCORES") or env("SAPY_TRACE_SCORES") == "1")
        if _want_trace:
            ProbabilisticTensor.start_cache()
            _tracing = True
        _score_fn = _score_function(vl_model)
        from saturn.soft_logic.tensor import BINDABLE_ENTITIES
        _bind_token = BINDABLE_ENTITIES.set(len(scene.objects))
        result_str = logic_executor_func(
            query=question,
            score_fn=_score_fn,
            query_fn=vl_model.query_multiview,
            scene=scene,
            images=images,
            history=None,
        )
        if _tracing:
            _tracing = False
            cache = ProbabilisticTensor.end_cache()

        if not isinstance(result_str, str) and result_str is not None:
            result_str = str(result_str)
        elif result_str is None:
            # Retryable: a None answer goes through the retry and the fallback.
            log.warning("Warning: generated code returned None.")
            error_msg = NO_ANSWER_ERROR
    except Exception as e:
        error_msg = format_execution_error(e, code_snippet, scene)
        log.error(f"Code execution failed: {e}")
        traceback.print_exc()
    finally:
        if _tracing:
            ProbabilisticTensor.end_cache()
        if _bind_token is not None:
            BINDABLE_ENTITIES.reset(_bind_token)
        _remove_stand_ins(scene, _held)

    return result_str, cache, error_msg


def _spatial_relations(scene) -> Dict:
    """The scene's relation predicates by name; stops (with a warning) at the first missing one."""
    spatial_relations = {}
    try:
        for name in _SPATIAL_RELATION_NAMES:
            spatial_relations[name] = getattr(scene, name)
    except Exception as e:
        log.warning(f"Warning: could not build some spatial relations: {e}")
    return spatial_relations


def _exec_context(scene, spatial_relations: Dict) -> Dict:
    """The globals the program runs with."""
    return {
        "ProbabilisticTensor": ProbabilisticTensor,
        "and_op": and_op,
        "or_op": or_op,
        "scene": scene,
        "np": np,
        "torch": torch,
        "math": __import__("math"),
        "formula_helpers": make_formula_helpers,
        **spatial_relations,
    }


def _syntax_error(code_to_execute: str) -> Optional[str]:
    """The error message when the scaffolded program does not compile, else None."""
    try:
        compile(code_to_execute, "<generated>", "exec")
    except SyntaxError as e:
        return (
            f"SyntaxError: {e}\n"
            f"Line {e.lineno}: {e.text.strip() if e.text else '(unknown)'}\n"
            f"Code:\n{code_to_execute[:800]}..."
        )
    return None


_ABSENT = object()


def _install_stand_ins(scene, held: Dict) -> None:
    """Replace scene.detect / scene.ground with no-ops; ``held`` records what each name held."""
    for name, stand_in in (("detect", _detect_disabled), ("ground", _ground_disabled)):
        previous = getattr(scene, name, _ABSENT)
        setattr(scene, name, stand_in)
        held[name] = previous


def _remove_stand_ins(scene, held: Dict) -> None:
    """Undo ``_install_stand_ins``: restore each name, or remove it if the scene had none."""
    for name, previous in held.items():
        if previous is _ABSENT:
            delattr(scene, name)
        else:
            setattr(scene, name, previous)


def _detect_disabled(desc, camera=None):
    """Stands in for scene.detect while a program runs."""
    log.warning(
        f"[WARN] scene.detect('{desc}'"
        + (f", camera={camera}" if camera is not None else "")
        + ") called from codegen — no-op"
    )
    return []


def _ground_disabled(desc, vlm=None):
    """Stands in for scene.ground while a program runs."""
    log.warning(f"[WARN] scene.ground('{desc}') called from codegen — no-op")
    return []


def _score_function(vl_model):
    """The program's score(): the VLM's, or a constant stub under SAPY_SCORE_STUB.

    SAPY_SCORE_STUB=<float> replaces VLM score() with a constant (object slots
    = value, camera slots and self-pairs hard 0) so cached programs can be
    replayed without a VLM (used for scorer A/B replays).
    """
    _stub_val = env("SAPY_SCORE_STUB")
    if not _stub_val:
        return vl_model.score_multiview
    _sv = float(_stub_val)

    def _score_stub(question, num_objects=1, type=None, scene=None,
                    images=None, cam_id=None, **kw):
        _N = len(scene.objects)
        _C = len(scene.cameras)
        _T = _N + _C
        if num_objects == 1:
            _t = torch.zeros(_T)
            _t[:_N] = _sv
        else:
            _t = torch.zeros(_T, _T)
            _t[:_N, :_N] = _sv
            _t.fill_diagonal_(0.0)
        _w = getattr(vl_model, "wrapper", None)
        return _w(_t, extra_info=question) if _w is not None else _t
    return _score_stub


def _program_state(scene) -> Dict:
    """The scene state a program may set: north and the question's axis labels."""
    north = getattr(scene, "_scene_north_vector", None)
    return {"north": None if north is None else np.array(north, dtype=float, copy=True),
            "axes": getattr(scene, "_axis_convention_M", None)}


def _restore_program_state(scene, state: Dict) -> None:
    scene._scene_north_vector = None if state["north"] is None else state["north"].copy()
    scene._axis_convention_M = state["axes"]


def execute_with_retry(
    code_snippet: str,
    question: str,
    vl_model,
    scene,
    images: List[Image.Image],
    code_generator: CodeGenerator,
    clarification: Dict,
    item_id: str,
    max_retries: int = 3,
) -> Tuple[str, Optional[str], Optional[list], Optional[str], int]:
    """Execute code, retrying with LLM-generated fixes on failure.

    Returns (final_code_snippet, answer_str, score_cache, error_msg, retry_count).
    """
    # A failed attempt's north / axis labels must neither leak into the retry nor show in its
    # scene facts: snapshot what the scene held before the first attempt (the pipeline's pose
    # constraints are applied earlier and left alone; programs never call scene.constraint).
    pre_program = _program_state(scene) if scene is not None else None
    answer_str, cache, error = execute_code(
        code_snippet, question, vl_model, scene, images
    )

    retry_count = 0
    current_snippet = code_snippet
    while error and retry_count < max_retries:
        retry_count += 1
        if pre_program is not None:
            _restore_program_state(scene, pre_program)
        log.warning(
            f"[{item_id}] Retry {retry_count}/{max_retries}: "
            f"code failed with {error[:120]}..."
        )
        # Pass the same scene-facts block that the original generate_code call used,
        # so retries see post-detection scene state (objects_count, merged, orient_conf, ...).
        retry_scene_facts_block = ""
        if scene is not None and hasattr(scene, "dump_facts_str"):
            try:
                retry_scene_facts_block = scene.dump_facts_str()
            except Exception as e:
                log.warning(f"[retry_generate_code] scene.dump_facts_str() failed: {e}")
        retry_object_groundings_block = ""
        _retry_parsed = clarification.get("parsed")
        if _retry_parsed:
            from saturn.planning.query_planner import QueryPlanner as _QP
            _grd = _retry_parsed.get("object_groundings") or []
            if _grd:
                retry_object_groundings_block = _QP.format_groundings_block(_grd) or ""
        retry_snippet, _ = code_generator.retry_generate_code(
            query=question,
            failed_code=current_snippet,
            error_message=error,
            clarified_query_block=clarification.get("clarified_query_block", ""),
            scene_facts_block=retry_scene_facts_block,
            object_groundings_block=retry_object_groundings_block,
        )
        if not retry_snippet:
            log.error(f"[{item_id}] Retry {retry_count} failed to generate code.")
            break
        retry_snippet = re.sub(
            r"</?(?:type_of_reasoning|steps|text|reasoning|code|frame_analysis)>.*",
            "",
            retry_snippet,
        ).strip()
        current_snippet = retry_snippet
        answer_str, cache, error = execute_code(
            current_snippet, question, vl_model, scene, images
        )
        if not error:
            log.info(f"[{item_id}] Retry {retry_count} succeeded!")

    return current_snippet, answer_str, cache, error, retry_count
