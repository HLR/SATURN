"""
MindCube multi-view dataset loader.

Loads samples from the MindCube-tinybench JSONL file.  Each sample has
multiple images (typically 4 views: front/left/back/right), a question
with multiple-choice answers, and object keyword metadata.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from saturn.log import progress


# Default paths — override via constructor arguments
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "mindcube"
_DEFAULT_JSONL = _DEFAULT_DATA_ROOT / "raw" / "MindCube_tinybench.jsonl"


def _infer_question_type(sample_id: str) -> str:
    """Infer question type from sample ID."""
    sid = str(sample_id).lower()
    if "among" in sid:
        return "among"
    if "around" in sid:
        return "around"
    if "rotation" in sid:
        return "rotation"
    return "other"


def _option_map_from_question(question: str) -> List[tuple]:
    """Extract (letter, text) answer options from the question string."""
    # Use lookahead for next option or end-of-string to delimit option text;
    # option text may itself contain uppercase A-E (e.g. "Bronze statue").
    matches = re.findall(r"([A-E])\.\s*(.+?)(?=\s+[A-E]\.\s|$)", question)
    out = []
    for key, value in matches:
        value = re.sub(r"\s+", " ", value).strip(" .")
        if value:
            out.append((key, value))
    return out


def _canonicalize_prompt(prompt: str) -> str:
    prompt = str(prompt).strip()
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt


def _extract_focus_prompts(sample: Dict, max_prompts: int = 6) -> List[str]:
    """Extract keyword prompts for SAM3 from sample metadata and answer options.

    Handles three meta_info formats:
      - among:    meta_info[0] = ["obj1", "obj2", ...]  (list of object name strings)
      - around:   meta_info[0] = [count, occ1, ...], meta_info[1][1] = ["obj1", "obj2", ...]
      - rotation: meta_info = ["obj1", "obj2", ...]  (flat list of object name strings)
    """
    prompts = []
    meta_info = sample.get("meta_info", [])

    if meta_info:
        if isinstance(meta_info[0], list):
            # Could be "among" format (list of strings) or "around" format (list starting with int)
            if meta_info[0] and isinstance(meta_info[0][0], int):
                # "around" format: meta_info[0] = [count, occ1, ...],
                #                  meta_info[1][1] = [obj_name1, obj_name2, ...]
                if (
                    len(meta_info) > 1
                    and isinstance(meta_info[1], list)
                    and len(meta_info[1]) > 1
                    and isinstance(meta_info[1][1], list)
                ):
                    prompts.extend(meta_info[1][1])
            else:
                # "among" format: meta_info[0] = ["obj1", "obj2", ...]
                prompts.extend(meta_info[0])
        elif isinstance(meta_info[0], str):
            # "rotation" format: meta_info = ["obj1", "obj2", ...]
            prompts.extend(meta_info)

    # Also extract object nouns mentioned in the question text itself.
    # Pattern: "is there a <subject> behind the <reference>?"
    question = sample.get("question", "")
    relation_match = re.search(
        r"is there (?:a |an )?(.+?)\s+(?:behind|left of|right of|in front of|above|below)\s+(?:the |a |an )?(.+?)(?:\?|$)",
        question,
        re.IGNORECASE,
    )
    if relation_match:
        for g in (relation_match.group(1), relation_match.group(2)):
            cleaned = g.strip().rstrip("?. ")
            if cleaned and len(cleaned) <= 60:
                prompts.append(cleaned)

    for _, option in _option_map_from_question(sample["question"]):
        # Filter out non-object answer options (Yes, No, directions, etc.)
        low = option.lower().strip()
        if low in ("yes", "no", "true", "false", "none", "n/a"):
            continue
        if len(option) <= 40:
            prompts.append(option)

    prompts = [_canonicalize_prompt(p) for p in prompts if p]

    deduped = []
    seen = set()
    for prompt in prompts:
        key = prompt.lower()
        if key not in seen:
            deduped.append(prompt)
            seen.add(key)

    return deduped[:max_prompts]


class MindCubeDataset(Dataset):
    """PyTorch Dataset for MindCube multi-view VQA.

    Each item returns:
        id              : str  — unique sample ID
        query           : str  — question text (with answer options)
        answer          : str  — ground-truth answer letter (e.g. "A")
        images          : list[PIL.Image]  — loaded view images
        image_paths     : list[str]  — absolute paths to view images
        focus_prompts   : list[str]  — keyword prompts for SAM3
        options         : list[tuple(str, str)]  — (letter, text) answer options
        question_type   : str  — "among", "around", "rotation", "other"
        meta_info       : list  — raw meta_info from JSONL

    Args:
        jsonl_path: Path to MindCube_tinybench.jsonl.
        data_root: Root directory for images (image paths are relative to this).
        question_type: If given, filter to this type only (e.g. "among").
        num_samples: If given, limit to first N samples after filtering.
        load_image: If True (default), load images as PIL Image objects.
        max_prompts: Max number of focus prompts to extract per sample.
    """

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        data_root: Optional[str] = None,
        question_type: Optional[str] = None,
        num_samples: Optional[int] = None,
        load_image: bool = True,
        max_prompts: int = 6,
    ) -> None:
        self.data_root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
        jsonl_path = Path(jsonl_path) if jsonl_path else _DEFAULT_JSONL
        self.load_image = load_image
        self.max_prompts = max_prompts

        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        if question_type is not None:
            rows = [r for r in rows if _infer_question_type(r["id"]) == question_type]

        if num_samples is not None:
            rows = rows[:num_samples]

        self.rows = rows
        progress(
            f"MindCubeDataset: loaded {len(self.rows)} samples"
            f" (type={question_type or 'all'}, data_root={self.data_root})"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.rows[idx]

        image_paths = []
        images = []
        for rel_path in sample.get("images", []):
            abs_path = str(self.data_root / rel_path)
            image_paths.append(abs_path)
            if self.load_image:
                images.append(Image.open(abs_path).convert("RGB"))

        focus_prompts = _extract_focus_prompts(sample, max_prompts=self.max_prompts)

        options = _option_map_from_question(sample["question"])

        return {
            "id": str(sample["id"]),
            "query": sample["question"],
            "answer": str(sample.get("gt_answer", "")),
            "images": images,
            "image_paths": image_paths,
            "focus_prompts": focus_prompts,
            "options": options,
            "question_type": _infer_question_type(sample["id"]),
            "meta_info": sample.get("meta_info", []),
        }
