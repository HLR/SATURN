"""VLM-based extractor that turns question text + scene images into
camera-pose constraint records consumable by ``pose_solver.solve_camera_poses``.

Pipeline per sample:
    1. Regex pre-filter on question text — skip extraction entirely when no
       pose-related keywords are present (saves VLM calls + eliminates the
       hallucinated-constraint failure mode).
    2. Render the constraint-extraction prompt with the question text appended.
    3. Call the VLM with the scene images + prompt; parse the returned JSON.
    4. Validate each record against the supported schema; drop malformed ones
       and log a warning rather than failing the whole sample.

The module does NOT apply the constraints — that's the caller's job (typically
via ``scene.constraint.rotation(...)`` and ``scene.constraint.same_position(...)``).
This separation keeps the extractor pure and easy to mock for tests.

This module is import-side-effect-free.
"""
from __future__ import annotations


import json
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

CONSTRAINT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "constraint_extractor.txt"
)

# Words/phrases in question text that suggest the question STATES camera
# pose relationships. We only call the VLM extractor when at least one of
# these matches — otherwise the extractor returns [] without an LLM call.
_POSE_KEYWORDS = re.compile(
    r"\b("
    r"turn(ed|ing)?|rotat(ed|ing|ion)?|"
    r"opposite\s+direction|"
    r"clockwise|counter[-\s]?clockwise|"
    r"\d+\s*degrees?|\d+\s*°|"
    r"same\s+(spot|place|position|location|vantage)|"
    r"same\s+direction|"
    r"facing\s+the\s+same|"
    r"to\s+the\s+(left|right)\s+(from|of)|"
    r"to\s+the\s+(left|right)\s+from\s+image"
    r")\b",
    re.IGNORECASE,
)


def question_has_pose_keywords(question: str) -> bool:
    """Cheap regex pre-filter — True if the question text contains any phrase
    suggesting a camera pose relationship is being stated. When False, the
    extractor short-circuits to ``[]`` without calling the VLM.
    """
    return bool(_POSE_KEYWORDS.search(question or ""))


# Type alias for an async VLM-call function with the same shape as
# QwenVLvLLM.client.generate(images, text, max_tokens=...).
VLMGenerateFn = Callable[..., Awaitable[str]]


async def extract_camera_constraints_async(
    question: str,
    images: Sequence[Any],
    vlm_generate: VLMGenerateFn,
    *,
    num_cameras: Optional[int] = None,
    prompt_path: Optional[Path] = None,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    """Extract camera pose constraints from a question via a VLM.

    Args:
        question: the multi-view question text (may include MCQ options).
        images: scene images (PIL.Image or whatever the VLM client accepts).
        vlm_generate: an async callable matching the VLM client's
            ``generate(images, text, max_tokens=...) -> str`` signature.
            Passing the function (rather than the client object) keeps this
            module decoupled from any specific client implementation.
        num_cameras: optional sanity-check; constraints referencing camera
            indices >= num_cameras are dropped with a warning.
        prompt_path: override the default prompt file path (for testing).
        max_tokens: budget for the VLM response.

    Returns:
        List of constraint records consumable by
        ``pose_solver.solve_camera_poses``. Empty list if no constraints
        were stated, the regex pre-filter rejected the question, or all
        extracted records failed validation.
    """
    if not question_has_pose_keywords(question):
        return []

    prompt_text = _load_prompt(prompt_path)
    full_prompt = f"{prompt_text}\n\nQUESTION:\n  {question.strip()}\n\nOUTPUT:\n"

    try:
        raw = await vlm_generate(images, full_prompt, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001
        logger.warning("constraint_extractor: VLM call failed (%s); returning []", e)
        return []

    parsed = _parse_constraints_json(raw)
    return _validate_constraints(parsed, num_cameras=num_cameras)



# ---------------------------------------------------------------------------
# Helpers (parsing + validation)
# ---------------------------------------------------------------------------

_PROMPT_CACHE: Dict[str, str] = {}


def _load_prompt(path: Optional[Path]) -> str:
    p = str(path) if path else str(CONSTRAINT_PROMPT_PATH)
    if p not in _PROMPT_CACHE:
        _PROMPT_CACHE[p] = Path(p).read_text()
    return _PROMPT_CACHE[p]


def _parse_constraints_json(raw: str) -> List[Dict[str, Any]]:
    """Parse the VLM's JSON response into a list of raw constraint dicts.

    Handles common output deviations:
      - markdown ```json fences
      - leading/trailing prose
      - JSON wrapped in {"constraints": [...]} (the schema we documented)
        OR returned as a bare [...] list.
    """
    if not raw or not raw.strip():
        return []

    text = raw.strip()

    # Strip markdown fences if present.
    if text.startswith("```"):
        # Drop opening fence + optional language tag
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Drop closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Find a JSON object or array. Try object first (matches schema), then array.
    obj_match = _find_balanced(text, "{", "}")
    arr_match = _find_balanced(text, "[", "]")

    candidates = []
    if obj_match is not None:
        candidates.append(obj_match)
    if arr_match is not None:
        candidates.append(arr_match)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "constraints" in data:
            cs = data["constraints"]
            if isinstance(cs, list):
                return cs
        elif isinstance(data, list):
            return data
    return []


def _find_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the substring of `text` containing the first balanced
    open/close pair (handles nested pairs). Returns None if no match.
    """
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _validate_constraints(
    records: List[Dict[str, Any]],
    num_cameras: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter to well-formed constraint records; drop + warn on bad entries."""
    valid: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            logger.debug("constraint_extractor: dropping non-dict record %r", rec)
            continue
        ctype = rec.get("type")
        if ctype == "rotation":
            try:
                f = int(rec["from_cam"])
                t = int(rec["to_cam"])
                yaw = float(rec["yaw"])
            except (KeyError, ValueError, TypeError):
                logger.debug("constraint_extractor: malformed rotation %r", rec)
                continue
            if num_cameras is not None and (
                f < 0 or t < 0 or f >= num_cameras or t >= num_cameras
            ):
                logger.debug(
                    "constraint_extractor: rotation cam idx out of range (have %d cams) %r",
                    num_cameras, rec,
                )
                continue
            if f == t:
                continue  # self-constraint is meaningless
            valid.append({
                "type": "rotation",
                "from_cam": f,
                "to_cam": t,
                "yaw": yaw,
                "axis": rec.get("axis", "up"),
            })
        elif ctype == "same_position":
            # A stated "same spot": the cameras in the set snap to the
            # lowest-indexed member (anchor gauge; see pose_solver).
            cams = rec.get("cams") or rec.get("entities") or []
            try:
                cam_ids = [int(c) for c in cams]
            except (ValueError, TypeError):
                logger.debug("constraint_extractor: malformed same_position %r", rec)
                continue
            if num_cameras is not None:
                cam_ids = [c for c in cam_ids if 0 <= c < num_cameras]
            cam_ids = sorted(set(cam_ids))
            if len(cam_ids) >= 2:
                valid.append({"type": "same_position", "cams": cam_ids})
        else:
            logger.debug("constraint_extractor: unknown constraint type %r", rec)
    return valid
