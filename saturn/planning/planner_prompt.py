"""Planner output checks shared by the query planner (planning/query_planner.py).

The prompt itself is UNIFIED_PLANNER_PROMPT (planner_prompt_unified.py). This module holds:

Index invariant:
  Every per-grounding view reference carries BOTH a 1-indexed ``image`` and a
  0-indexed ``cam_id``, with the contract ``cam_id == image - 1``.
  ``_validate_planner_output`` enforces the invariant; if it is violated the
  planner is re-prompted once (``_build_refine_prompt``) with the specific
  error message.

Option coverage:
  ``ensure_option_groundings`` adds a grounding for every physical MCQ option
  the planner left out.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from saturn.planning.planner_prompt_unified import UNIFIED_PLANNER_PROMPT


def _check_image_cam_pair(
    image: Any, cam_id: Any, *, where: str,
) -> Optional[str]:
    """Return error message if (image, cam_id) violates the invariant, else None.

    The invariant: both null, OR both non-null integers with cam_id == image - 1.
    """
    if image is None and cam_id is None:
        return None
    if image is None or cam_id is None:
        return (
            f"{where}: image and cam_id must both be set or both null "
            f"(got image={image!r}, cam_id={cam_id!r})"
        )
    if not isinstance(image, int) or not isinstance(cam_id, int):
        return (
            f"{where}: image and cam_id must be integers when present "
            f"(got image={image!r}, cam_id={cam_id!r})"
        )
    if image < 1:
        return f"{where}: image must be >= 1 (got image={image})"
    if cam_id != image - 1:
        return (
            f"{where}: cam_id must equal image - 1 "
            f"(got image={image}, cam_id={cam_id}, expected cam_id={image - 1})"
        )
    return None


def _validate_planner_output(parsed: Dict[str, Any]) -> List[str]:
    """Validate (image, cam_id) pairing on each object grounding.

    Returns a list of human-readable error messages — empty when every
    grounding satisfies the invariant. Designed to be passed back to the
    planner verbatim in a refinement prompt.
    """
    errors: List[str] = []
    for i, g in enumerate(parsed.get("object_groundings") or []):
        if not isinstance(g, dict):
            errors.append(f"object_groundings[{i}]: entry must be an object")
            continue
        phrase = g.get("phrase", "?")
        err = _check_image_cam_pair(
            g.get("image"), g.get("cam_id"),
            where=f"object_groundings[{i}] '{phrase}'",
        )
        if err:
            errors.append(err)
    return errors


def _build_refine_prompt(question: str, errors: List[str]) -> str:
    """Compose the refinement prompt: original prompt + error list + ask."""
    base = UNIFIED_PLANNER_PROMPT.replace("{question}", question)
    bullets = "\n".join(f"  - {e}" for e in errors)
    suffix = (
        "\n\n──── REFINEMENT REQUEST ────\n"
        "Your previous response violated the (image, cam_id) invariant:\n"
        f"{bullets}\n\n"
        "Re-emit the JSON block with these corrected. Reminder: cam_id == "
        "image - 1 in every (image, cam_id) pair, and both must be null "
        "together or both non-null together. Output the corrected JSON only.\n"
    )
    return base + suffix


# ---------------------------------------------------------------------------
# Option coverage
# ---------------------------------------------------------------------------
# Weak planners can skip physical MCQ options as "option labels" and ground
# only the question subject, leaving the program a 1-object scene. An option
# is an ANSWER STATE (not an object) only if every word is in this closed
# vocabulary; any other option names a thing in the scene and must be grounded.
_NON_OBJECT_WORDS = frozenset("""
yes no true false none neither both all not cannot can be determined unknown same different
directly diagonally forward forwards backward backwards front back behind ahead left right
up down above below over under top bottom upward downward upwards downwards clockwise
counterclockwise counter anticlockwise degrees degree turn turned turning rotate rotated
north south east west northeast northwest southeast southwest n s e w ne nw se sw
closer farther further nearer near far bigger smaller larger taller shorter higher lower
more less equal the a an and or to of my me i it is on in at from toward towards side
away camera viewer
rear upper lower you your yourself due immediate approximately basically equally then while
height width size length speed position direction time angle obtuse acute movement
positive negative axis x y z shape t f l circle rectangle triangle square small large big
one two three four five six seven eight nine ten unable determine reach sometimes former latter upside
rearward stationary see partially amount almost about close letter step
""".split())
_OPTION_RE = re.compile(r"(?:^|\s)\(?([A-H])[\.\):]\s+(.+?)(?=,?\s+\(?[A-H][\.\):]\s|$)", re.S)


def parse_mcq_options(question: str) -> List[tuple]:
    """``[(letter, text)]`` for 'A. x B. y' and 'A: x, B: y' styles."""
    q = question.replace("\n", " ")
    for anchor in ("Options:", "options:"):
        if anchor in q:
            q = q.split(anchor, 1)[1]
            break
    else:
        m = re.search(r"(?:^|\s)\(?A[\.\):]\s", q)
        if not m:
            return []
        q = q[m.start():]
    return [(k.upper(), v.strip().rstrip(".").strip()) for k, v in _OPTION_RE.findall(q)]


def _norm_phrase(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


# An option names a scene object only if it reads as a short noun phrase.
# Sentences ("The chair is left of the table"), measurements ("2 meters"),
# image references ("Figure 1", "First image") and motions ("Go straight")
# are answer states, not things to ground.
_NOT_A_THING = re.compile(
    r"\d|\b(is|are|was|were|has|have|had|will|would|does|did|do|go|goes|went|move|moves|moved|"
    r"walk|walked|walking|moving|rotate|rotates|rotated|rotating|rotation|turns|turning|look|flipped|taken|"
    r"stay|stays|facing|straight|meters?|metres?|cm|centimeters?|feet|foot|inches?|"
    r"figure|image|images|photo|photos|picture \d|view|views|frame|first|second|third|fourth|"
    r"last|degrees?|percent|times)\b", re.I)


def is_object_option(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if not words or len(words) > 8 or _NOT_A_THING.search(text):
        return False
    return not all(w in _NON_OBJECT_WORDS for w in words)


def ensure_option_groundings(question: str, parsed: Dict[str, Any]) -> List[str]:
    """Append a grounding for every physical MCQ option the planner left out.

    Appended groundings search all views (image/cam_id null) and carry
    ``role="option"`` + ``auto_added=True``. Returns the added phrases.
    """
    groundings = parsed.setdefault("object_groundings", []) or []
    parsed["object_groundings"] = groundings
    have = {_norm_phrase(g.get("phrase", "")) for g in groundings if isinstance(g, dict)}
    added: List[str] = []
    for _, text in parse_mcq_options(question):
        if not is_object_option(text) or _norm_phrase(text) in have:
            continue
        groundings.append({
            "phrase": text, "description": text, "image": None, "cam_id": None,
            "is_region": False, "multi_view": False,
            "unique": not re.search(r"\b(several|two|three|four|many|some|pictures|chairs)\b", text.lower()),
            "role": "option", "auto_added": True,
        })
        have.add(_norm_phrase(text))
        added.append(text)
    return added

