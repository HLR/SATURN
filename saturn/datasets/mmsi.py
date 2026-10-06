"""
MMSI-Bench multi-view dataset loader.

Loads samples from the MMSI_Bench.parquet file. Each sample has
multiple images (2-10 views), an MCQ question, and metadata including
human-written reasoning traces.

Reference: MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
           https://arxiv.org/abs/2505.23764
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from saturn.log import progress


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "mmsi"
_DEFAULT_PARQUET = _DEFAULT_DATA_ROOT / "MMSI_Bench.parquet"


def _option_map_from_question(question: str) -> List[tuple]:
    """Extract (letter, text) answer options from the question string.

    MMSI uses "Options: A: text, B: text" format (colon-separated).
    """
    # Try MMSI format: "A: text, B: text" or "A: text\n B: text"
    matches = re.findall(r"([A-D]):\s*(.+?)(?=,\s*[A-D]:|$)", question, re.DOTALL)
    if matches:
        out = []
        for key, value in matches:
            value = re.sub(r"\s+", " ", value).strip(" .")
            if value:
                out.append((key, value))
        return out

    # Fallback: "A. text B. text" format
    matches = re.findall(r"([A-D])\.\s*(.+?)(?=\s+[A-D]\.\s|$)", question)
    out = []
    for key, value in matches:
        value = re.sub(r"\s+", " ", value).strip(" .")
        if value:
            out.append((key, value))
    return out


_STOP_WORDS = frozenset(
    [
        # generic image/scene words
        "image",
        "images",
        "photo",
        "photos",
        "picture",
        "pictures",
        "figure",
        "figures",
        "frame",
        "frames",
        "view",
        "views",
        "perspective",
        "scene",
        # pronouns / generic spatial refs
        "you",
        "me",
        "myself",
        "i",
        "we",
        "them",
        "it",
        "this",
        "that",
        "spot",
        "place",
        "position",
        "location",
        "direction",
        "way",
        "side",
        "area",
        # ordinals / numbers (stand-alone)
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "one",
        "two",
        "three",
        "four",
        "five",
        # common non-object preposition heads
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
        "center",
        "middle",
        "edge",
        "corner",
    ]
)

# Directional option patterns — these appear as MCQ option text in MMSI
_DIRECTIONAL_OPTION_RE = re.compile(
    r"^("
    # 8-way compound: front/back + left/right (with optional "to the")
    r"(front|back|forward)\s+(right|left|to the right|to the left)"
    r"|(right|left)\s+(front|back|rear)"
    r"|back\s*-?\s*right|back\s*-?\s*left"
    r"|front\s*-?\s*right|front\s*-?\s*left"
    r"|left\s+rear|right\s+rear|left\s+front|right\s+front"
    # Simple directional words
    r"|left|right|forward|backward|up|down"
    # Cardinal + compound cardinal
    r"|north(east|west)?|south(east|west)?|east|west"
    r"|northeast\s+corner|northwest\s+corner|southeast\s+corner|southwest\s+corner"
    # "directly to the X"
    r"|directly\s+(to\s+the\s+)?(right|left|front|back|above|below|forward|backward)"
    # Motion descriptions
    r"|walking|standing\s+still|running|moving"
    r"|left\s+while|right\s+while|forward\s+to\s+the|clockwise|counterclockwise"
    r")",
    re.IGNORECASE,
)

_COMPARISON_OPTION_RE = re.compile(
    r"^(the\s+same\s+(height|size|distance|level|width)|"
    r"sometimes.*(taller|shorter|wider|narrower|bigger|smaller|closer|farther)|"
    r"the\s+.*\s+is\s+(taller|shorter|wider|narrower|bigger|smaller|closer|farther))",
    re.IGNORECASE,
)


def _extract_focus_prompts(question: str, max_prompts: int = 8) -> List[str]:
    """Extract keyword prompts for SAM3 from the question text.

    Uses heuristics to find object-like nouns mentioned in the question.
    Filters out directional options, cardinal directions, generic words, and
    motion descriptions that are not useful as SAM3 detection queries.
    """
    prompts = []

    # Extract object nouns from common spatial patterns
    # The stop-word lookahead ensures we don't over-capture into verb phrases.
    _STOP_AHEAD = (
        r"is|are|was|were|has|have|in|on|at|to|from|"
        r"behind|left|right|front|above|below|near|far|closer|farther|between|"
        r"located|relative|shown|visible|when|while|that|which|who|where|"
        r"and|or|but|with|without|of|for|by"
    )
    patterns = [
        # "the X" / "a X" / "an X" patterns
        rf"(?:the|a|an)\s+([a-z][a-z\s\-]{{1,30}}?)(?=\s+(?:{_STOP_AHEAD}))",
        # "of the X" pattern
        rf"of\s+(?:the|a|an)\s+([a-z][a-z\s\-]{{1,30}}?)(?=[\?\.,;]|\s+(?:{_STOP_AHEAD}))",
        # "where is/was the X" pattern
        rf"where\s+(?:is|was|were)\s+(?:the|a|an)\s+([a-z][a-z\s\-]{{1,30}}?)(?=[\?\.,;]|\s+(?:{_STOP_AHEAD}))",
    ]
    for pat in patterns:
        for m in re.finditer(pat, question, re.IGNORECASE):
            noun = m.group(1).strip().lower()
            if len(noun) > 2 and len(noun) < 50 and noun not in _STOP_WORDS:
                # Also skip if noun IS a directional phrase
                if not _DIRECTIONAL_OPTION_RE.match(noun):
                    prompts.append(m.group(1).strip())

    # Extract answer options that look like concrete objects (not directions/numbers/yes/no)
    for _, text in _option_map_from_question(question):
        low = text.lower().strip()

        # Skip yes/no/boolean/n/a
        if low in ("yes", "no", "true", "false", "none", "n/a"):
            continue

        # Skip pure numbers / measurements
        if re.match(r"^[\d.]+\s*(m|cm|mm|meters?|feet|ft|degrees?|°)?$", low):
            continue

        # Skip directional options
        if _DIRECTIONAL_OPTION_RE.match(low):
            continue

        # Skip comparison options ("The same height", "Sometimes the former...")
        if _COMPARISON_OPTION_RE.match(low):
            continue

        # Skip pure stop words
        if low in _STOP_WORDS:
            continue

        # Skip head-motion / object-motion descriptors
        if re.match(r".*head\s+(facing|turned|looking)", low):
            continue

        # Keep object-like options (reasonable length)
        if 2 < len(low) <= 60:
            prompts.append(text)

    # Deduplicate preserving order
    deduped = []
    seen = set()
    for p in prompts:
        key = p.lower().strip()
        if key not in seen and key not in _STOP_WORDS:
            deduped.append(p)
            seen.add(key)

    return deduped[:max_prompts]


class MMSIDataset(Dataset):
    """PyTorch Dataset for MMSI-Bench multi-view spatial intelligence VQA.

    Each item returns:
        id              : int or str — unique sample ID
        query           : str — question text (with answer options)
        answer          : str — ground-truth answer letter (e.g. "A")
        images          : list[PIL.Image] — loaded view images
        image_paths     : list[str] — absolute paths to view images
        focus_prompts   : list[str] — keyword prompts for SAM3
        options         : list[tuple(str, str)] — (letter, text) answer options
        question_type   : str — MMSI question type
        meta_info       : dict — extra metadata (thought, difficulty)

    Args:
        parquet_path: Path to MMSI_Bench.parquet.
        data_root: Root directory for images.
        question_type: If given, filter to this type(s) only.
        num_samples: If given, limit to first N samples after filtering.
        load_image: If True (default), load images as PIL Image objects.
        max_prompts: Max number of focus prompts to extract per sample.
    """

    VALID_TYPES = [
        "Attribute (Appr.)",
        "Attribute (Meas.)",
        "Motion (Cam.)",
        "Motion (Obj.)",
        "Positional Relationship (Cam.–Cam.)",
        "Positional Relationship (Cam.–Obj.)",
        "Positional Relationship (Cam.–Reg.)",
        "Positional Relationship (Obj.–Obj.)",
        "Positional Relationship (Obj.–Reg.)",
        "Positional Relationship (Reg.–Reg.)",
        "MSR",
    ]

    def __init__(
        self,
        parquet_path: Optional[str] = None,
        data_root: Optional[str] = None,
        question_type: Optional[str] = None,
        num_samples: Optional[int] = None,
        load_image: bool = True,
        max_prompts: int = 8,
    ) -> None:
        self.data_root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
        parquet_path = Path(parquet_path) if parquet_path else _DEFAULT_PARQUET
        self.load_image = load_image
        self.max_prompts = max_prompts
        self.image_dir = self.data_root / "images"

        df = pd.read_parquet(parquet_path)

        # Extract the images once
        if True:  # per-file existence checks below make this idempotent (resumable after an interrupted extraction)
            progress("Checking/extracting MMSI images from parquet...")
            self.image_dir.mkdir(parents=True, exist_ok=True)
            for _, row in df.iterrows():
                if row["images"] is not None:
                    for n, img_data in enumerate(row["images"]):
                        img_path = self.image_dir / f"{row['id']}_{n}.jpg"
                        if not img_path.exists():
                            with open(img_path, "wb") as f:
                                f.write(img_data)

        # Filter by question type
        if question_type is not None:
            # Support shorthand names like "motion-cam" -> "Motion (Cam.)"
            qt_filter = self._resolve_question_type(question_type)
            if qt_filter:
                df = df[df["question_type"].isin(qt_filter)]
            else:
                raise ValueError(
                    f"question_type '{question_type}' not recognized. Valid types: {self.VALID_TYPES}"
                )

        # Limit samples
        if num_samples is not None:
            df = df.head(num_samples)

        self.df = df.reset_index(drop=True)
        progress(
            f"MMSIDataset: loaded {len(self.df)} samples"
            f" (type={question_type or 'all'}, data_root={self.data_root})"
        )

    # Shorthand aliases for question types.
    # Keys are lowercase; values are exact VALID_TYPES strings.
    _SHORTHAND_MAP = {
        "cam cam": "Positional Relationship (Cam.–Cam.)",
        "cam obj": "Positional Relationship (Cam.–Obj.)",
        "cam reg": "Positional Relationship (Cam.–Reg.)",
        "obj obj": "Positional Relationship (Obj.–Obj.)",
        "obj reg": "Positional Relationship (Obj.–Reg.)",
        "reg reg": "Positional Relationship (Reg.–Reg.)",
        "motion cam": "Motion (Cam.)",
        "motion obj": "Motion (Obj.)",
        "meas": "Attribute (Meas.)",
        "appr": "Attribute (Appr.)",
        "msr": "MSR",
    }

    def _resolve_question_type(self, qt: str) -> Optional[List[str]]:
        """Resolve a question type string to a list of valid MMSI types.

        Supports exact match, shorthand aliases (e.g. "cam-cam", "obj_reg",
        "motion cam"), and substring matching.  Hyphens, underscores, and
        Unicode en-dashes are all normalized to spaces before matching.
        """
        # Exact match
        if qt in self.VALID_TYPES:
            return [qt]

        # Normalize: lowercase, replace hyphens/underscores/en-dashes with spaces
        qt_lower = qt.lower().replace("_", " ").replace("-", " ").replace("\u2013", " ")

        # Check shorthand map first
        if qt_lower in self._SHORTHAND_MAP:
            return [self._SHORTHAND_MAP[qt_lower]]

        # Substring match (also normalize valid types for comparison)
        matched = []
        for valid in self.VALID_TYPES:
            valid_norm = valid.lower().replace("\u2013", " ").replace("-", " ")
            if qt_lower in valid_norm:
                matched.append(valid)
        return matched if matched else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Build image paths
        image_paths = []
        images = []
        num_imgs = len(row["images"]) if row["images"] is not None else 0
        for n in range(num_imgs):
            img_path = str(self.image_dir / f"{row['id']}_{n}.jpg")
            image_paths.append(img_path)
            if self.load_image:
                images.append(Image.open(img_path).convert("RGB"))

        question = row["question"]
        options = _option_map_from_question(question)
        focus_prompts = _extract_focus_prompts(question, max_prompts=self.max_prompts)

        return {
            "id": str(row["id"]),
            "query": question,
            "answer": str(row.get("answer", "")),
            "images": images,
            "image_paths": image_paths,
            "focus_prompts": focus_prompts,
            "options": options,
            "question_type": row.get("question_type", ""),
            "meta_info": {
                "thought": row.get("thought", ""),
                "difficulty": row.get("difficulty", ""),
            },
        }
