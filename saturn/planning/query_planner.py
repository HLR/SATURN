"""QueryPlanner — VLM pre-processing step for grounding and reference-frame hints.

Given multi-view images and a raw spatial-reasoning question, identifies every
object that should be grounded, which camera shows it best, and a compact
reference frame with only position and orientation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from saturn.planning.planner_prompt import (
    _build_refine_prompt,
    _validate_planner_output,
    ensure_option_groundings,
)
from saturn.planning.planner_prompt_unified import UNIFIED_PLANNER_PROMPT
from saturn.log import get_logger

log = get_logger(__name__)

# Bump whenever the planner's prompt or output format changes. Cache entries
# without this version (or with a lower one) are treated as misses.
_PLANNER_CACHE_VERSION = 38


# The planner's first field, "objects": where the scene's objects come from
# (planner_prompt_unified.py; pipeline/sample.py acts on it). Missing or unknown -> "named".
OBJECT_SOURCES = ("named", "search", "no_object")


def _object_source(value: Any) -> str:
    return value if value in OBJECT_SOURCES else "named"


def _int_cam_id(value: Any) -> Optional[Any]:
    """Coerce a raw cam_id to int, keeping the list form ([1], [0, 2]) that
    the planner prompts ask for. Range clamping is
    left to ``QueryPlanner._coerce_cam_ids``."""
    if value is None:
        return None
    if isinstance(value, list):
        out = []
        for c in value:
            try:
                out.append(int(c))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _images_fingerprint(images: Any) -> str:
    """Content hash of the sample's images, part of the planner cache key.

    The plan (setup_caption, groundings, cam_ids) describes one scene, so two
    samples that share a question string must not share a plan.
    """
    h = hashlib.sha1()
    for im in images or []:
        if hasattr(im, "tobytes"):
            meta = (getattr(im, "size", None), getattr(im, "mode", None), getattr(im, "shape", None))
            h.update(repr(meta).encode())
            h.update(im.tobytes())
        else:
            h.update(repr(im).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

REFINE_GROUNDINGS_PROMPT = """\
You are a scene preprocessor for multi-view spatial reasoning. Your prior plan
already grounded most objects, but the downstream detector failed on the entries
listed below. Re-evaluate ONLY the failed phrases — keep all other prior
groundings exactly as they were.

Return exactly one JSON object with no markdown fences and no extra text.

OUTPUT FORMAT:
{
  "reasoning": "<≤~300 chars: for each failed phrase, briefly justify the new description and/or cam_id you picked.>",
  "revised_groundings": [
    {
      "phrase": "<EXACTLY the same phrase string as in the failed list>",
      "description": "<a NEW concrete description; do NOT repeat the prior one>",
      "cam_id": <list of 0..N-1 image indices, e.g. [1] or [0, 2]; [] = scan every view. Bare integer also accepted and treated as a single-element list. When verify_feedback names specific images, list ALL of them.>,
      "is_region": <bool, same as prior>,
      "multi_view": <bool, same as prior>
    }
  ]
}

DIAGNOSTIC LEGEND (in the failure list below):
  - "no_candidates" — SAM3 + VLM both returned 0 boxes for the prior description
                      on the prior cam_id. Most likely the description names
                      something not visually present in that view (wrong word,
                      wrong cam_id, or too-verbose phrasing). Try: a SHORTER
                      noun phrase, a different attribute (color, shape, material),
                      OR a different cam_id where the object is more obvious.
  - "no_verify"     — SAM3 returned N candidate boxes but none passed the VLM
                      verification ("Is this the {phrase}?"). The boxes likely
                      surrounded look-alikes, not the actual object. Try: a more
                      DISCRIMINATING description (mention a unique attribute that
                      separates the target from look-alikes), and/or a different
                      cam_id where the target is unambiguous.

  Some failures additionally include a "verify_feedback" line — this is the
  VLM's free-form observation of what it actually saw inside the best
  candidate box (e.g. "this is the study desk, not the bed; the bed is in
  image 2"). When present, treat it as ground truth about that view:
    * If the feedback names a different object in the box, your revised
      description must EITHER target that object's defining attributes
      OR move to a cam_id where the original target is actually visible.
    * If the feedback names a different image index where the target
      lives, set cam_id to THAT index in the revised entry.
    * Do not contradict verify_feedback — it is derived from a vision
      pass over the actual image, not from your prior plan.

REVISION RULES — STRICT:
1. The revised `description` MUST differ from the prior one. Do not just rephrase.
2. Prefer SHORT, concrete noun phrases over long verbose descriptions — SAM3
   grounds nouns ("TV", "blue chair") much more reliably than full sentences
   ("the rectangular black television screen displaying text").
3. If the failure was "no_verify" with several candidates, try a DIFFERENT
   cam_id (look at the images and pick one where the target is clearly visible).
4. If you cannot find a better description or cam_id, you may emit the same
   phrase with `"cam_id": null` to ask SAM3 to scan every view.
5. DO NOT include phrases that already verified successfully — only the failed ones.
6. DO NOT change the `phrase` text. Match it character-for-character.
7. PRESERVE OBJECT IDENTITY — the revised `description` MUST describe the SAME
   physical object/region as the original `phrase`. The phrase names the target;
   the description gives SAM3 visual attributes (color, material, shape, position,
   orientation, distinguishing features) to find THAT target. You may change ANY
   attributes, but you may NOT change the head noun or the object class implied by
   the phrase.
   - WRONG: phrase="the two single sofas" → description="leather loveseat with three
     seat cushions"  (a loveseat is not "two single sofas" — count and class changed)
   - WRONG: phrase="the chair" → description="the wooden desk near the window"
     (changed object class entirely from chair to desk)
   - WRONG: phrase="the red bottle" → description="a tall transparent vase"
     (color and class both changed)
   - RIGHT: phrase="the two single sofas" → description="two matching armchairs side
     by side along the back wall" (kept seating-for-one count of two; changed
     attributes to make them more findable)
   - RIGHT: phrase="the red bottle" → description="small red plastic water bottle on
     the wooden table near the window" (kept "red" and "bottle"; added position)
   - RIGHT: phrase="the chair" → description="black office chair with chrome wheels
     in the corner" (kept chair; added attributes/location)
   The phrase is the ground-truth identity; the description is just SAM3's hint.

Question: {question}

Prior plan (object_groundings already emitted; verified entries shown for context):
{prior_groundings_block}

Failed groundings (these need revision):
{failures_block}
"""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _atomic_write_json(path: str, data: dict) -> None:
    dir_name = os.path.dirname(path) or "."
    base = os.path.basename(path)
    tmp_name = f".{base}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    tmp_path = os.path.join(dir_name, tmp_name)
    os.makedirs(dir_name, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# QueryPlanner
# ---------------------------------------------------------------------------


class QueryPlanner:
    """Runs a single VLM call per question to produce object groundings.

    Parameters
    ----------
    vl_model : object
        A VLM wrapper exposing ``_query(image, text, max_new_tokens)``.
        Must accept ``image`` as either a PIL image or a list of PIL images.
    cache_path : str
        Where to persist the JSON cache.  Keyed by question string.
    write_cache : bool
        If ``False``, reads are served but no writes are made.
    max_new_tokens : int
        Cap on decoded tokens for the response.
    verbose : bool
        Print activity to stdout.
    """

    def __init__(
        self,
        vl_model: Any,
        cache_path: str = "programs/query_clarification_cache.json",
        write_cache: bool = True,
        max_new_tokens: int = 10240,  # must fit under the VLM's max-model-len (16384) together with the image and prompt tokens
        verbose: bool = False,
    ):
        self.vl_model = vl_model
        self.cache_path = cache_path
        self.write_cache = write_cache
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self._cache: dict = self._load_cache()

    # -- cache ------------------------------------------------------------

    def _load_cache(self) -> dict:
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _persist_atomic(self, query: str, record: dict) -> None:
        """Atomic merge-update of ``self._cache`` under a file lock."""
        if not self.write_cache:
            return
        lock_path = f"{self.cache_path}.lock"
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                disk: dict = {}
                if os.path.exists(self.cache_path):
                    try:
                        with open(self.cache_path, "r") as f:
                            disk = json.load(f)
                            if not isinstance(disk, dict):
                                disk = {}
                    except Exception:
                        disk = {}
                disk[query] = record
                _atomic_write_json(self.cache_path, disk)
                self._cache[query] = record
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    # -- inference --------------------------------------------------------

    def _invoke_vlm(
        self,
        question: str,
        images: List[Any],
        *,
        override_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Call the VLM with question + images; return raw text or None.

        Uses ``_query_thinking`` when available (Qwen3.5+) to get better
        structured JSON output; falls back to ``_query`` otherwise.

        ``override_prompt``: if provided, use it verbatim instead of the
        default ``UNIFIED_PLANNER_PROMPT``. Used by
        ``clarify()`` to send a refinement prompt after an (image, cam_id)
        invariant violation.
        """
        if self.vl_model is None:
            return None
        if override_prompt is not None:
            prompt = override_prompt
        else:
            prompt = UNIFIED_PLANNER_PROMPT.replace("{question}", question)
        if self.verbose:
            tag = " (refine)" if override_prompt is not None else ""
            log.info(f"[QueryPlanner] planner_mode=program{tag}")
        imgs = images if images else None

        # Prefer _query_thinking (returns thinking, answer tuple).
        if hasattr(self.vl_model, "_query_thinking") and callable(
            self.vl_model._query_thinking
        ):
            try:
                _, answer = self.vl_model._query_thinking(
                    imgs,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                )
                if isinstance(answer, str):
                    return answer.strip()
            except Exception as e:
                if self.verbose:
                    log.warning(f"[QueryPlanner] _query_thinking failed, falling back: {e}")

        # Fallback to standard _query.
        try:
            out = self.vl_model._query(
                imgs,
                prompt,
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as e:
            if self.verbose:
                log.error(f"[QueryPlanner] VLM call failed: {e}")
            return None
        if isinstance(out, list):
            out = out[0] if out else ""
        if not isinstance(out, str):
            return None
        return out.strip()

    # -- parse ------------------------------------------------------------

    @staticmethod
    def _extract_groundings_from_truncated(text: str) -> List[Dict[str, Any]]:
        """Recover ``object_groundings`` from a response cut off mid-JSON.

        Only the ``object_groundings`` array is scanned (up to the next
        top-level key, ``skipped_phrases``), so phrases quoted elsewhere in
        the response are never mistaken for groundings.
        """
        rg_idx = text.find('"object_groundings"')
        if rg_idx < 0:
            return []
        sk = text.find('"skipped_phrases"', rg_idx + 1)
        scoped = text[rg_idx:sk if sk >= 0 else len(text)]

        groundings: List[Dict[str, Any]] = []
        # Order-independent match — the LLM may emit phrase/description/cam_id
        # in any order inside each grounding object, so look up each key by name.
        # Each `{...}` chunk between top-level commas is one candidate grounding.
        chunk_pattern = r'\{[^{}]*\}'
        for chunk in re.findall(chunk_pattern, scoped):
            phrase_m = re.search(r'"phrase"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            desc_m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
            cam_m = re.search(r'"cam_id"\s*:\s*(\d+|null|\[[\d,\s]*\])', chunk)
            if not (phrase_m and desc_m and cam_m):
                continue
            phrase = phrase_m.group(1)
            if not phrase.strip():
                continue
            cam_id = _int_cam_id(json.loads(cam_m.group(1)))
            mv_m = re.search(r'"multi_view"\s*:\s*(true|false)', chunk)
            multi_view = bool(mv_m and mv_m.group(1) == "true")
            entry = {
                "phrase": phrase,
                "description": desc_m.group(1),
                "cam_id": cam_id,
                "is_region": False,
                "multi_view": multi_view,
            }
            # Groundings may pair cam_id with `image`; plural/category
            # groundings carry unique=false. Keep both when present.
            img_m = re.search(r'"image"\s*:\s*(\d+|null)', chunk)
            if img_m:
                entry["image"] = _int_cam_id(json.loads(img_m.group(1)))
            uq_m = re.search(r'"unique"\s*:\s*(true|false)', chunk)
            if uq_m:
                entry["unique"] = uq_m.group(1) == "true"
            groundings.append(entry)

        return groundings

    @staticmethod
    def parse(raw: str) -> Dict[str, Any]:
        """Parse the planner's JSON response.

        The planner answers ``{objects, setup_caption, program_sketch,
        object_groundings[], skipped_phrases[]}`` (planner_prompt_unified.py).
        Returns a dict with keys ``objects`` (named | search | no_object),
        ``object_groundings``, ``reasoning`` (the program sketch),
        ``setup_caption`` and ``skipped_phrases``.
        """
        stripped = raw.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
        text = fence.group(1).strip() if fence else stripped

        def _empty(reasoning: str = "", recovered: Optional[bool] = None) -> Dict[str, Any]:
            out: Dict[str, Any] = {
                "objects": "named",
                "object_groundings": [],
                "reasoning": reasoning,
                "setup_caption": "",
                "skipped_phrases": [],
            }
            if recovered is not None:
                out["_recovered_from_truncation"] = recovered
            return out

        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            idx = text.find("{")
            if idx < 0:
                return _empty()
            try:
                # raw_decode ignores trailing prose after a complete object.
                obj, _ = json.JSONDecoder().raw_decode(text, idx)
            except (json.JSONDecodeError, ValueError):
                recovered_groundings = QueryPlanner._extract_groundings_from_truncated(text[idx:])
                rec_reasoning = ""
                rm = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text[idx:])
                if rm:
                    rec_reasoning = rm.group(1)
                om = re.search(r'"objects"\s*:\s*"(\w+)"', text[idx:])
                rec_objects = _object_source(om.group(1) if om else None)
                if recovered_groundings or rec_objects != "named":
                    out = _empty(rec_reasoning, recovered=True)
                    out["objects"] = rec_objects
                    out["object_groundings"] = recovered_groundings
                    return out
                # Recovery attempted but extracted nothing — flag it as False
                # (NOT None) so downstream code can distinguish "we tried and
                # failed" from "we never had to try".
                return _empty(rec_reasoning, recovered=False)

        if not isinstance(obj, dict):
            return _empty()

        reasoning = obj.get("program_sketch")
        if not isinstance(reasoning, str):
            reasoning = ""

        groundings = obj.get("object_groundings")
        if not isinstance(groundings, list):
            groundings = []
        # Non-dict entries (e.g. bare strings) would crash _normalize_groundings.
        groundings = [g for g in groundings if isinstance(g, dict)]

        # Normalise grounding entries — backfill missing flags so downstream
        # grounder code can rely on them.
        for g in groundings:
            if isinstance(g, dict):
                g.setdefault("is_region", False)
                g.setdefault("multi_view", False)
                g.setdefault("unique", True)
                g["cam_id"] = _int_cam_id(g.get("cam_id"))
                img = g.get("image")
                if img is not None:
                    try:
                        g["image"] = int(img)
                    except (TypeError, ValueError):
                        g["image"] = None

        # An observational description of the scene/observer setup; codegen
        # receives it in the SCENE FACTS block.
        setup_caption = obj.get("setup_caption")
        if not isinstance(setup_caption, str):
            setup_caption = ""

        # {phrase, reason} for nouns the planner deliberately did not ground
        # (e.g. "the camera", self-referential); kept for the reports.
        skipped = obj.get("skipped_phrases")
        if not isinstance(skipped, list):
            skipped = []

        return {
            "objects": _object_source(obj.get("objects")),
            "object_groundings": groundings,
            "reasoning": reasoning,
            "setup_caption": setup_caption,
            "skipped_phrases": skipped,
        }

    # -- main entry point -------------------------------------------------

    @staticmethod
    def format_groundings_block(groundings: List[Dict[str, str]]) -> str:
        """Format parsed groundings into a text block for prompt injection.

        Reference table for the codegen LLM: maps the planner-extracted noun
        (normalized: leading "the/a/an" stripped, lowercased) to a canonical
        description and the planner-assigned role. The codegen prompt may use
        either the noun (default for SAM3 scoring) or the description (richer
        text grounding) when calling ``score(...)``. Both are surfaced so the
        LLM can choose; the role tag distinguishes the anchor from targets and
        context objects.
        """
        import re as _re

        def _normalize_noun(p: str) -> str:
            return _re.sub(r"^(the|a|an)\s+", "", p.strip(), flags=_re.IGNORECASE).lower()

        if not groundings:
            return ""
        lines = ["OBJECT GROUNDINGS:"]
        for g in groundings:
            phrase = g.get("phrase", "").strip()
            description = g.get("description", "").strip()
            role = g.get("role", "").strip()
            if phrase and description:
                noun = _normalize_noun(phrase)
                role_tag = f"  [{role}]" if role else ""
                planner_tag = (
                    f' (planner: "{phrase}")' if noun != phrase.strip().lower() else ""
                )
                lines.append(
                    f'- "{noun}"{planner_tag}  —  {description}{role_tag}'
                )
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _maybe_ensure_options(self, question: str, parsed: Dict[str, Any]) -> None:
        """Ground every physical MCQ option the planner skipped (cached AND
        fresh plans; see planner_prompt.ensure_option_groundings)."""
        if not isinstance(parsed, dict):
            return
        added = ensure_option_groundings(question, parsed)
        parsed["_auto_option_groundings"] = added
        if added and self.verbose:
            log.info(f"[QueryPlanner] added {len(added)} skipped option grounding(s): {added}")

    def _cache_key(self, question: str, images: List[Any]) -> str:
        # Key format: the question, then "[planner_mode=program prompt=unified images=<fingerprint>]".
        return f"{question}\n[planner_mode=program prompt=unified images={_images_fingerprint(images)}]"

    def clarify(
        self,
        question: str,
        images: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Produce the object groundings (and setup caption) for the question.

        Returns the parsed planner output (see :meth:`parse`).

        Results are cached per (question, image content): the
        plan is scene-specific, so samples sharing a question string must not
        share it. Failed calls are recorded but retried on the next call.
        """
        if not question or not isinstance(question, str):
            return None

        cache_key = self._cache_key(question, images)
        cached = self._cache.get(cache_key)
        if cached is not None:
            if (
                cached.get("status") == "ok"
                and cached.get("_planner_version", 0) >= _PLANNER_CACHE_VERSION
                and cached.get("raw")
            ):
                # `raw` is the single source of truth: re-derive content fields
                # via parse() so any schema field added to parse() automatically
                # flows through to cache hits without a separate cache-IO edit.
                parsed = self.parse(cached["raw"])
                self._maybe_ensure_options(question, parsed)
                parsed["object_groundings"] = self._normalize_groundings(
                    parsed.get("object_groundings", []),
                    len(images) if images else 0,
                )
                return parsed

        raw = self._invoke_vlm(question, images)
        if not raw:
            record = {
                "question": question,
                "raw": "",
                "status": "failed",
                "_planner_version": _PLANNER_CACHE_VERSION,
                "timestamp": time.time(),
            }
            self._persist_atomic(cache_key, record)
            return None

        parsed = self.parse(raw)

        # (image, cam_id) invariant check + refine on violation, with at most
        # one refine attempt per question.
        refine_attempts = 0
        validation_errors: List[str] = _validate_planner_output(parsed)
        if validation_errors:
            if self.verbose:
                log.warning(
                    f"[QueryPlanner] (image, cam_id) invariant violated; "
                    f"refining: {validation_errors[:2]}{' …' if len(validation_errors) > 2 else ''}"
                )
            refine_prompt = _build_refine_prompt(question, validation_errors)
            refined_raw = self._invoke_vlm(
                question, images, override_prompt=refine_prompt,
            )
            refine_attempts = 1
            if refined_raw:
                refined_parsed = self.parse(refined_raw)
                # Keep the refined output only if it actually reduces
                # the error count — otherwise fall back to the first
                # response (avoids replacing a near-correct plan with
                # a worse one when the model gets confused by the
                # refinement framing).
                refined_errors = _validate_planner_output(refined_parsed)
                if len(refined_errors) < len(validation_errors):
                    parsed = refined_parsed
                    validation_errors = refined_errors
                    raw = refined_raw

        self._maybe_ensure_options(question, parsed)
        parsed["object_groundings"] = self._normalize_groundings(
            parsed.get("object_groundings", []),
            len(images) if images else 0,
        )
        parsed["_planner_refine_attempts"] = refine_attempts
        parsed["_planner_validation_errors"] = validation_errors

        # Cache record = full parsed dict + metadata. Spreading `**parsed`
        # means any field parse() emits is persisted automatically; new
        # planner fields don't require a cache-record edit.
        record = {
            **parsed,
            "question": question,
            "raw": raw,
            "status": "ok",
            "_planner_version": _PLANNER_CACHE_VERSION,
            "timestamp": time.time(),
        }
        self._persist_atomic(cache_key, record)

        if self.verbose:
            reasoning = parsed.get("reasoning", "")
            if reasoning:
                # One-line truncation; full text lives in the cache for inspection.
                snippet = reasoning.replace("\n", " ").strip()
                if len(snippet) > 220:
                    snippet = snippet[:217] + "..."
                log.debug(f"[QueryPlanner] reasoning: {snippet}")
            n_g = len(parsed["object_groundings"])
            log.info(f"[QueryPlanner] groundings: {n_g}")
            for g in parsed["object_groundings"]:
                region_tag = " [REGION]" if g.get("is_region") else ""
                cam_tag = f" [cam={g.get('cam_id', '?')}]" if "cam_id" in g else ""
                role_tag = f" [role={g.get('role', '?')}]" if "role" in g else ""
                log.info(f"    - {g['phrase']}: {g['description'][:80]}{cam_tag}{region_tag}{role_tag}")

        return parsed

    # -- refine (planner-retry on grounder failure) -----------------------

    def refine_groundings(
        self,
        question: str,
        images: List[Any],
        prior_groundings: List[Dict[str, Any]],
        failed_groundings: List[Dict[str, Any]],
        retry_round: int = 1,
    ) -> List[Dict[str, Any]]:
        """Re-evaluate failing groundings; return revised entries (not a full plan).

        Parameters
        ----------
        question : str
            The original spatial-reasoning question.
        images : list
            Multi-view images (same set passed to ``clarify``).
        prior_groundings : list of dict
            The full original plan (used as context — verified entries are
            shown so the planner doesn't re-suggest them).
        failed_groundings : list of dict
            Subset of ``prior_groundings`` that failed grounding. Each entry
            must include the original ``phrase``/``description``/``cam_id``
            plus a ``diagnostic`` field ("no_candidates" or "no_verify ...").

        Returns
        -------
        list of dict
            Revised grounding entries, one per failed phrase, in the same
            schema as the planner's ``object_groundings``. May be empty if
            the planner produced no usable revision.
        """
        if not failed_groundings or self.vl_model is None:
            return []

        # Not cached: each retry round needs a fresh planner call, and a cache
        # would replay the first round's revision. The clarify() cache (the
        # planner's first plan) is separate.
        prior_block = self._format_prior_groundings_block(prior_groundings)
        failures_block = self._format_failures_block(failed_groundings)

        prompt = (
            REFINE_GROUNDINGS_PROMPT
            .replace("{question}", question)
            .replace("{prior_groundings_block}", prior_block)
            .replace("{failures_block}", failures_block)
        )

        if self.verbose:
            log.info(
                f"[QueryPlanner] refine_groundings: "
                f"{len(failed_groundings)} failed phrase(s)"
            )

        raw = ""
        # Reuse _query_thinking if available (gives cleaner JSON on Qwen3.5+).
        if hasattr(self.vl_model, "_query_thinking") and callable(
            self.vl_model._query_thinking
        ):
            try:
                _, answer = self.vl_model._query_thinking(
                    images if images else None,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                )
                if isinstance(answer, str):
                    raw = answer.strip()
            except Exception as e:
                if self.verbose:
                    log.warning(f"[QueryPlanner] refine _query_thinking failed: {e}")
        if not raw:
            try:
                out = self.vl_model._query(
                    images if images else None,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                )
                if isinstance(out, list):
                    out = out[0] if out else ""
                if isinstance(out, str):
                    raw = out.strip()
            except Exception as e:
                if self.verbose:
                    log.error(f"[QueryPlanner] refine _query failed: {e}")

        revised = self._parse_revised(raw)
        # The refine schema has no `unique`, and the model may drop `role`;
        # inherit them from the failed grounding so a retried plural or
        # category phrase keeps all its instances.
        by_phrase = {
            (f.get("phrase") or "").strip().lower(): f for f in failed_groundings
        }
        for r in revised:
            orig = by_phrase.get(r["phrase"].lower())
            if orig is None:
                continue
            if "unique" in orig:
                r.setdefault("unique", orig["unique"])
            if not r.get("role") and orig.get("role"):
                r["role"] = orig["role"]
            if "auto_added" in orig:
                r["auto_added"] = orig["auto_added"]
        revised = self._normalize_groundings(
            revised, len(images) if images else 0,
        )

        if self.verbose:
            log.info(
                f"[QueryPlanner] refine_groundings: returned {len(revised)} "
                f"revised entries"
            )
            for r in revised:
                log.info(
                    f"    → {r.get('phrase')!r} cam={r.get('cam_id')} "
                    f"desc={(r.get('description') or '')[:80]!r}"
                )

        return revised

    # -- plan-normalization gate ------------------------------------------

    @staticmethod
    def _coerce_cam_ids(value: Any, n_images: int) -> Optional[Any]:
        """Coerce a cam_id field to the schema-valid form.

        Accepts any of: int, list[int], None. Returns ``List[int]`` (clamped
        to ``[0, n_images)`` with duplicates removed) when the input is a
        list, the preferred schema form. Returns a single int, or ``None``
        (out-of-range or null), when the input is a scalar. Both forms are
        valid; the grounder normalizes again on read.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if 0 <= value < n_images else None
        if isinstance(value, list):
            out: List[int] = []
            for c in value:
                try:
                    ci = int(c)
                except (TypeError, ValueError):
                    continue
                if 0 <= ci < n_images and ci not in out:
                    out.append(ci)
            return out
        return None

    @staticmethod
    def _normalize_groundings(
        groundings: List[Dict[str, Any]],
        n_images: int,
        *,
        item_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Single planner-output normalization gate.

        Two invariants enforced here so downstream code (grounder, retry,
        salvage, codegen) can trust the plan:

        1. **Dedupe by canonical phrase.** The planner sometimes enumerates
           the same logical object twice (typically question text + MCQ
           option text colliding on the same noun, e.g. ``'the window'`` and
           ``'Window'``). Without dedup, both pass through the grounder
           independently; if the first attempt fails, the "already in scene,
           skip" path on the duplicate doesn't fire, and after retry/salvage
           the scene contains phantom anchors for one logical target.

        2. **Clamp out-of-range cam_id.** The planner sometimes emits a
           ``cam_id`` outside ``[0, n_images)`` (e.g. cam_id=3 on a 3-image
           sample). Without clamping, the grounder's view-aware verify path
           rejects every candidate with no useful diagnostic; clamping to
           ``None`` lets SAM3 scan every view as a recovery.

        Returns a new list; does not mutate input dicts.
        """
        if not groundings:
            return groundings
        seen: set = set()
        out: List[Dict[str, Any]] = []
        n_dropped = 0
        n_clamped = 0
        for g in groundings:
            phrase = (g.get("phrase") or "").strip()
            key = re.sub(r"^the\s+", "", phrase.lower()).strip()
            if not key or key in seen:
                n_dropped += 1
                continue
            seen.add(key)
            # Coerce cam_id (scalar or list form). Out-of-range values are
            # dropped silently inside the helper; track whether the value changed.
            raw_cam = g.get("cam_id")
            new_cam = QueryPlanner._coerce_cam_ids(raw_cam, n_images)
            if new_cam != raw_cam:
                n_clamped += 1
                g = {**g, "cam_id": new_cam}
            out.append(g)
        if n_dropped or n_clamped:
            tag = f"[{item_id}] " if item_id else ""
            log.info(
                f"{tag}[QueryPlanner] normalize: "
                f"{len(groundings)} → {len(out)} groundings "
                f"(dropped {n_dropped} duplicate, clamped {n_clamped} cam_id)"
            )
        return out

    # -- formatting helpers for refine_groundings -------------------------

    @staticmethod
    def _format_prior_groundings_block(
        prior: List[Dict[str, Any]],
    ) -> str:
        if not prior:
            return "(no prior groundings)"
        lines = []
        for g in prior:
            phrase = g.get("phrase", "")
            desc = g.get("description", "")
            cam = g.get("cam_id")
            mv = " multi_view" if g.get("multi_view") else ""
            region = " is_region" if g.get("is_region") else ""
            # cam_id renders as scalar or list depending on storage form;
            # both are valid schema, both should display readably.
            if cam is None:
                cam_str = "null"
            elif isinstance(cam, list):
                cam_str = "[]" if not cam else "[" + ", ".join(str(c) for c in cam) + "]"
            else:
                cam_str = str(cam)
            # Surface verify status when present — so the retry planner knows
            # which prior entries already succeeded (and at what confidence)
            # vs which are in the failures list below.
            status = ""
            if "verified" in g:
                v = g.get("verified")
                s = g.get("verify_score")
                if v:
                    status = (
                        f" [VERIFIED score={s:.2f}]" if isinstance(s, (int, float))
                        else " [VERIFIED]"
                    )
                else:
                    status = (
                        f" [UNVERIFIED score={s:.2f} — see failures]"
                        if isinstance(s, (int, float))
                        else " [UNVERIFIED — see failures]"
                    )
            lines.append(
                f"- {phrase!r}: {desc!r} [cam_id={cam_str}{mv}{region}]{status}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_failures_block(
        failures: List[Dict[str, Any]],
    ) -> str:
        if not failures:
            return "(none)"
        lines = []
        for g in failures:
            phrase = g.get("phrase", "")
            desc = g.get("description", "")
            cam = g.get("cam_id")
            if cam is None:
                cam_str = "null"
            elif isinstance(cam, list):
                cam_str = "[]" if not cam else "[" + ", ".join(str(c) for c in cam) + "]"
            else:
                cam_str = str(cam)
            diag = g.get("diagnostic", "unknown")
            n_cands = g.get("n_candidates", 0)
            score = g.get("verify_score")
            extra = ""
            if diag == "no_verify":
                extra = (
                    f" (saw {n_cands} candidate boxes; max verify score "
                    f"{score:.2f})" if isinstance(score, (int, float)) else
                    f" (saw {n_cands} candidate boxes)"
                )
            lines.append(
                f"- {phrase!r}: prior_description={desc!r}, "
                f"prior_cam_id={cam_str}, diagnostic={diag}{extra}"
            )
            # Free-form VLM critique of the winning rejected candidate,
            # produced by the verify step. Surfaces signals like "this is
            # the study desk, not the bed" or "the bed is in image 2" that
            # the planner can act on directly.
            vfb = (g.get("verify_feedback") or "").strip()
            if vfb:
                lines.append(f"    verify_feedback: {vfb}")
        return "\n".join(lines)

    @staticmethod
    def _parse_revised(raw: str) -> List[Dict[str, Any]]:
        """Parse the refine-prompt response. Returns a list of revised entries.

        Falls back to an empty list if parsing fails.
        """
        if not raw:
            return []
        stripped = raw.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
        text = fence.group(1).strip() if fence else stripped
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            idx = text.find("{")
            if idx < 0:
                return []
            try:
                obj = json.loads(text[idx:])
            except (json.JSONDecodeError, ValueError):
                return []
        if not isinstance(obj, dict):
            return []
        revised = obj.get("revised_groundings")
        if not isinstance(revised, list):
            return []
        out: List[Dict[str, Any]] = []
        for g in revised:
            if not isinstance(g, dict):
                continue
            phrase = g.get("phrase")
            description = g.get("description")
            if not (isinstance(phrase, str) and isinstance(description, str)):
                continue
            entry = {
                "phrase": phrase.strip(),
                "description": description.strip(),
                "cam_id": _int_cam_id(g.get("cam_id")),
                "is_region": bool(g.get("is_region", False)),
                "multi_view": bool(g.get("multi_view", False)),
                "role": g.get("role", ""),
            }
            if isinstance(g.get("unique"), bool):
                entry["unique"] = g["unique"]
            out.append(entry)
        return out


__all__ = [
    "QueryPlanner",
    "REFINE_GROUNDINGS_PROMPT",
]
