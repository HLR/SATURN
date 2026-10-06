"""
3D-FORCE dataset loaders.

Two task types:
  - Force3DPuzzleDataset  (3DForcePuzzle.json)  existential puzzles, answer="true"/"false"
  - Force3DRefDataset     (3DForceRef.json)      grounding, answer=target_bbox [x1,y1,x2,y2]

Both JSONs share the same top-level schema:
    {"tasks": [{"subset": "<name>", "data": {"questions": [...]}}]}

Image paths embedded in the JSONs are absolute paths into the Spatial457 multi-view renders
(``.../scene_NNNNNN/HASH/image.png``). The loaders rebase them onto ``image_root`` (default: the
JPEG image pack that setup/30_datasets.sh unpacks to data/3d-force/multiview) with ``image_ext``.
"""

from saturn.settings import env
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from saturn.log import progress


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Question files and images (Hugging Face dataset iamdanialkamali/3D-FORCE-Zip, fetched by setup/30_datasets.sh).
_DEFAULT_DATA_ROOT = Path(env("FORCE3D_DATA_ROOT") or _REPO_ROOT / "data" / "3d-force")
_DEFAULT_IMAGE_ROOT = _DEFAULT_DATA_ROOT / "multiview"
_DEFAULT_IMAGE_EXT = ".jpg"   # the image pack holds JPEG copies of the renders
_DEFAULT_PUZZLE_JSON = _DEFAULT_DATA_ROOT / "3DForcePuzzle.json"
_DEFAULT_REF_JSON = _DEFAULT_DATA_ROOT / "3DForceRef.json"


def _remap_image_path(original: str, image_root: Optional[str], image_ext: Optional[str]) -> str:
    """Optionally remap an absolute image path to a different root / extension."""
    p = Path(original)
    if image_root is not None:
        # Keep last 3 components: scene_NNNNNN / HASH / image.ext
        relative = Path(*p.parts[-3:])
        p = Path(image_root) / relative
    if image_ext is not None:
        p = p.with_suffix(image_ext)
    return str(p)


def _flatten_questions(json_path: Path) -> List[Dict[str, Any]]:
    """Load and flatten all questions from a 3D-FORCE JSON, tagging each with its subset name."""
    with open(json_path, "r") as f:
        raw = json.load(f)
    rows = []
    for task in raw.get("tasks", []):
        subset = task.get("subset", "unknown")
        for q in task.get("data", {}).get("questions", []):
            # _qidx = GLOBAL position in the unfiltered file, so ids are stable
            # across --category / --num_samples slices.
            rows.append({"_subset": subset, "_qidx": len(rows), **q})
    return rows


class Force3DPuzzleDataset(Dataset):
    """Existential multi-view spatial puzzles (3DForcePuzzle.json).

    Each item:
        id              str   — "{subset}_q{idx}"
        query           str   — natural-language question
        answer          str   — "true" or "false"
        images          list[PIL.Image]  — loaded view images (or str paths)
        image_paths     list[str]
        bboxes          list[[x1,y1,x2,y2]]  — all objects in canonical view
        masks           list  — COCO-RLE masks (may be empty list)
        solution_indexes list[int]  — object indices that satisfy the existential
        slot_dict       dict
        program         str
        question_type   str   — dataset subset name (for per-category stats)
        original_item   dict
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        image_root: Optional[str] = None,
        image_ext: Optional[str] = None,
        category: Optional[str] = None,
        num_samples: Optional[int] = None,
        load_image: bool = True,
    ) -> None:
        self.json_path = Path(json_path) if json_path else _DEFAULT_PUZZLE_JSON
        self.image_root = image_root or env("FORCE3D_IMAGE_ROOT") or str(_DEFAULT_IMAGE_ROOT)
        self.image_ext = image_ext or env("FORCE3D_IMAGE_EXT") or _DEFAULT_IMAGE_EXT
        self.load_image = load_image

        if not self.json_path.exists():
            raise FileNotFoundError(f"3DForcePuzzle JSON not found: {self.json_path}")

        rows = _flatten_questions(self.json_path)

        if category is not None:
            rows = [r for r in rows if category in r["_subset"]]

        if num_samples is not None:
            rows = rows[:num_samples]

        self.rows = rows
        progress(
            f"Force3DPuzzleDataset: {len(self.rows)} samples"
            f" (category={category or 'all'})"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.rows[idx]
        subset = item["_subset"]
        item_id = f"{Path(subset).stem}_q{item['_qidx']}"

        raw_paths: List[str] = item.get("image_filename", [])
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]

        image_paths = [
            _remap_image_path(p, self.image_root, self.image_ext) for p in raw_paths
        ]
        images = []
        if self.load_image:
            for p in image_paths:
                images.append(Image.open(p).convert("RGB"))

        answer_bool: bool = item.get("answer", False)

        return {
            "id": item_id,
            "subset": subset,
            "image_file_name": image_paths,
            "query": item.get("question", ""),
            "answer": "true" if answer_bool else "false",
            "images": images if self.load_image else image_paths,
            "image_paths": image_paths,
            "bboxes": item.get("bboxes", []),
            "masks": item.get("maskes", []),
            "solution_indexes": item.get("solution_indexes", []),
            "solution": item.get("solution", {}),
            "slot_dict": item.get("slot_dict", {}),
            "program": item.get("program", ""),
            "question_type": Path(subset).stem,
            "original_item": item,
        }


class Force3DRefDataset(Dataset):
    """Multi-view spatial reference-resolution (3DForceRef.json).

    Each item:
        id              str   — "{subset}_q{idx}"
        query           str   — natural-language question
        description     str   — just the target description (without "Find …")
        answer          list  — [x1,y1,x2,y2] target bbox (for IoU scoring)
        answer_idx      int   — original integer index into bboxes
        images          list[PIL.Image]
        image_paths     list[str]
        bboxes          list[[x1,y1,x2,y2]]  — all candidate objects
        masks           list  — COCO-RLE masks aligned to bboxes
        anchor_indices  dict  — {OBJ_KEY: bbox_index} for anchor objects
        slot_dict       dict
        program         str
        category        str   — "chain" / "hybrid" / "star"
        question_type   str   — same as category (for per-category stats)
        original_item   dict
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        image_root: Optional[str] = None,
        image_ext: Optional[str] = None,
        category: Optional[str] = None,
        num_samples: Optional[int] = None,
        load_image: bool = True,
    ) -> None:
        self.json_path = Path(json_path) if json_path else _DEFAULT_REF_JSON
        self.image_root = image_root or env("FORCE3D_IMAGE_ROOT") or str(_DEFAULT_IMAGE_ROOT)
        self.image_ext = image_ext or env("FORCE3D_IMAGE_EXT") or _DEFAULT_IMAGE_EXT
        self.load_image = load_image

        if not self.json_path.exists():
            raise FileNotFoundError(f"3DForceRef JSON not found: {self.json_path}")

        rows = _flatten_questions(self.json_path)

        if category is not None:
            rows = [r for r in rows if r.get("category", "").startswith(category)]

        if num_samples is not None:
            rows = rows[:num_samples]

        self.rows = rows
        progress(
            f"Force3DRefDataset: {len(self.rows)} samples"
            f" (category={category or 'all'})"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.rows[idx]
        subset = item["_subset"]
        item_id = f"{Path(subset).stem}_q{item['_qidx']}"
        raw_paths: List[str] = item.get("image_filename", [])
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]

        image_paths = [
            _remap_image_path(p, self.image_root, self.image_ext) for p in raw_paths
        ]
        images = []
        if self.load_image:
            for p in image_paths:
                images.append(Image.open(p).convert("RGB"))

        all_bboxes: List[List[float]] = item.get("bboxes", [])
        answer_idx: Optional[int] = item.get("answer")
        answer_bbox: Optional[List[float]] = None
        if isinstance(answer_idx, int) and 0 <= answer_idx < len(all_bboxes):
            answer_bbox = all_bboxes[answer_idx]

        raw_category: str = item.get("category", "")
        top_category = raw_category.split("_")[0] if raw_category else "unknown"

        return {
            "id": item_id,
            "image_file_name": image_paths,
            "subset": subset,
            "query": item.get("question", ""),
            "description": item.get("description", ""),
            "answer": answer_bbox,
            "answer_idx": answer_idx,
            "images": images if self.load_image else image_paths,
            "image_paths": image_paths,
            "bboxes": all_bboxes,
            "masks": item.get("maskes", []),
            "anchor_indices": item.get("anchor_indices", {}),
            "slot_dict": item.get("slot_dict", {}),
            "program": item.get("program", ""),
            "category": raw_category,
            "question_type": top_category,
            "original_item": item,
        }
