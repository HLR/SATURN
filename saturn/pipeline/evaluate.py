"""Pipeline steps 7-8: Evaluate MCQ answer; VLM fallback for MMSI."""

import re
from typing import Dict, List, Optional

from PIL import Image

from saturn.codegen.postprocess import unified_postprocess
from saturn.log import get_logger

log = get_logger(__name__)


def _parse_mcq_options(question: str) -> Dict[str, str]:
    """Parse MCQ options from question text."""
    # MMSI colon format: "A: text, B: text"
    colon_pattern = r"(?<![A-Za-z])([A-Ea-e]):\s*(.+?)(?=,\s*[A-Ea-e]:|$)"
    colon_matches = re.findall(colon_pattern, question, re.DOTALL)
    if colon_matches:
        options: Dict[str, str] = {}
        for letter, text in colon_matches:
            cleaned = re.sub(r"\s+", " ", text).strip(" .,")
            if cleaned:
                options[letter.upper()] = cleaned.lower()
        if options:
            return options

    # Fallback: "A." or "A)" format
    pattern = r"(?<![A-Za-z])([A-Ea-e])[.\)]\s*([^A-E][^.)\n]*?)(?=\s+[A-Ea-e][.\)]|\s*$)"
    matches = re.findall(pattern, question)
    return {letter.upper(): text.strip().lower() for letter, text in matches}


def evaluate_answer(
    predicted: Optional[str],
    gt_answer,
    question: str = "",
    item: Optional[Dict] = None,
    result_record: Optional[Dict] = None,
) -> bool:
    """Format-aware scorer.

    Dispatches on the *shape* of ``gt_answer``:

    1. **bbox IoU** — when GT is a list of 4 numbers (Force3DRef): the box of
       the object the program selected (``result_record['predicted_bbox']``) is
       compared with the GT bbox; correct iff IoU ≥ 0.5.
    2. **boolean** — when GT is "true"/"false" (Force3DPuzzle): case-insensitive
       string match.
    3. **MCQ letter** — fallback for everything else.
    """
    if predicted is None:
        return False
    pred_str = str(predicted).strip()

    # --- (1) Force3DRef: GT is bbox list, pred is object-index string ---
    if isinstance(gt_answer, (list, tuple)) and len(gt_answer) == 4 and all(
        isinstance(v, (int, float)) for v in gt_answer
    ):
        # Grade the box SATURN actually predicted. The scene objects are
        # SATURN's own detections, so the answer index has no relationship to
        # item['bboxes'].
        pred_box = (result_record or {}).get("predicted_bbox")
        if pred_box is None:
            return False
        gx1, gy1, gx2, gy2 = (float(v) for v in gt_answer)
        px1, py1, px2, py2 = (float(v) for v in pred_box)
        ix1, iy1 = max(gx1, px1), max(gy1, py1)
        ix2, iy2 = min(gx2, px2), min(gy2, py2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = ((gx2 - gx1) * (gy2 - gy1)
                 + (px2 - px1) * (py2 - py1)
                 - inter)
        iou = inter / union if union > 0 else 0.0
        return iou >= 0.5

    gt_str = str(gt_answer).strip()

    # --- (2) Force3DPuzzle: boolean string match ---
    if gt_str.lower() in ("true", "false"):
        return pred_str.lower().strip(".") == gt_str.lower()

    # --- (3) MCQ fallback ---
    if not pred_str:
        return False  # '' is a substring of every option text below
    # Ground-truth letter
    gt_match = re.match(r"^([A-Ea-e])", gt_str.upper())
    gt_letter = gt_match.group(1) if gt_match else gt_str.upper()

    # Predicted letter (direct): the letter must stand alone or be followed by
    # a delimiter, so option text like "chair" / "bed" is not read as C / B.
    pred_match = re.match(r"^\(?([A-Ea-e])(?:[.:\s)\-]|$)", pred_str)
    pred_letter = pred_match.group(1).upper() if pred_match else ""

    if pred_letter and pred_letter == gt_letter:
        return True

    # Text-to-letter fallback
    if not pred_letter and question:
        options = _parse_mcq_options(question)
        pred_lower = pred_str.strip().lower()
        for letter, opt_text in options.items():
            if pred_lower == opt_text or pred_lower in opt_text or opt_text in pred_lower:
                pred_letter = letter
                break
        if pred_letter and pred_letter == gt_letter:
            return True

    return unified_postprocess(pred_str) == unified_postprocess(gt_str)


def vlm_fallback_answer(
    vl_model,
    images: List[Image.Image],
    question: str,
    item_id: str,
) -> Optional[str]:
    """When code execution fails, ask the VLM directly for an MCQ answer.

    Two-stage strategy:
      1. Normal ask with a letter-only instruction, max_new_tokens=8.
      2. If stage 1 produces no A-E letter anywhere in the output (model
         ignored the instruction and started reasoning), retry with a
         terse pre-filled prompt that forces the first token to be a letter.
      3. If both stages fail, return None (caller leaves final_answer_text
         unset; scorer marks WRONG but does not pollute the result with
         prose). Never return non-letter prose as the answer.
    """
    def _extract_letter(text: str) -> Optional[str]:
        if not text:
            return None
        text = text.strip()
        # Prefer leading letter; otherwise accept any standalone A-E.
        m = re.match(r"^\s*([A-Ea-e])\b", text)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-Ea-e])\b", text)
        if m:
            return m.group(1).upper()
        return None

    # Stage 1: normal ask
    try:
        fallback_q = question + "\nAnswer with ONLY the option letter (A, B, C, D, or E)."
        vlm_answer = vl_model._query(images, fallback_q, max_new_tokens=8)
        if isinstance(vlm_answer, list):
            vlm_answer = vlm_answer[0] if vlm_answer else ""
        letter = _extract_letter(vlm_answer or "")
        if letter:
            log.info(f"[{item_id}] VLM fallback answer (stage1): {letter}")
            return letter
        log.warning(f"[{item_id}] VLM fallback stage1 produced no letter: {repr((vlm_answer or '')[:60])}")
    except Exception as e:
        log.error(f"[{item_id}] VLM fallback stage1 failed: {e}")

    # Stage 2: terse re-prompt — force letter-only
    try:
        terse_q = (
            "Multiple choice question. Output exactly one character: the letter "
            "of the correct option (A, B, C, D, or E). No other text.\n\n"
            + question
            + "\n\nAnswer: "
        )
        vlm_answer2 = vl_model._query(images, terse_q, max_new_tokens=4)
        if isinstance(vlm_answer2, list):
            vlm_answer2 = vlm_answer2[0] if vlm_answer2 else ""
        letter = _extract_letter(vlm_answer2 or "")
        if letter:
            log.info(f"[{item_id}] VLM fallback answer (stage2): {letter}")
            return letter
        log.warning(f"[{item_id}] VLM fallback stage2 produced no letter: {repr((vlm_answer2 or '')[:60])}")
    except Exception as e:
        log.error(f"[{item_id}] VLM fallback stage2 failed: {e}")

    return None
