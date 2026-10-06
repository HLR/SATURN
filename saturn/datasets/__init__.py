"""Benchmark loaders (3D-FORCE REF/SAG, MindCube, MMSI) behind ``get_dataset(name)``.

Data-side only; must not import saturn.scene/perception/vlm/serving.
"""
from saturn.settings import env
import os
from .mindcube import MindCubeDataset
from .mmsi import MMSIDataset
from .force3d import Force3DPuzzleDataset, Force3DRefDataset


def get_dataset(name, num_samples=None, load_image=True):
    """
    Get a dataset by name.
    """

    if name.startswith("mindcube"):
        # e.g. "mindcube", "mindcube-among", "mindcube-around", "mindcube-rotation"
        MINDCUBE_DATA_ROOT = env("MINDCUBE_DATA_ROOT") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mindcube"
        )
        # "mindcubedev-<type>": a fixed dev split drawn from MindCube_train (disjoint
        # scenes), for developing prompts/engine without touching the test set.
        MINDCUBE_JSONL = os.path.join(
            MINDCUBE_DATA_ROOT, "raw",
            "MindCube_dev.jsonl" if name.startswith("mindcubedev") else "MindCube_tinybench.jsonl",
        )
        parts = name.split("-", 1)
        question_type = parts[1] if len(parts) > 1 else None
        return MindCubeDataset(
            jsonl_path=MINDCUBE_JSONL,
            data_root=MINDCUBE_DATA_ROOT,
            question_type=question_type,
            num_samples=num_samples,
            load_image=load_image,
        )
    elif name.startswith("mmsi"):
        # e.g. "mmsi", "mmsi-motion", "mmsi-MSR", "mmsi-Positional Relationship"
        MMSI_DATA_ROOT = env("MMSI_DATA_ROOT") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mmsi"
        )
        MMSI_PARQUET = os.path.join(MMSI_DATA_ROOT, "MMSI_Bench.parquet")
        parts = name.split("-", 1)
        question_type = parts[1] if len(parts) > 1 else None
        return MMSIDataset(
            parquet_path=MMSI_PARQUET,
            data_root=MMSI_DATA_ROOT,
            question_type=question_type,
            num_samples=num_samples,
            load_image=load_image,
        )
    elif name.startswith("force3d-puzzle"):
        # e.g. "force3d-puzzle", "force3d-puzzle-chain"
        suffix = name[len("force3d-puzzle"):].lstrip("-")
        category = suffix or None
        return Force3DPuzzleDataset(
            category=category,
            num_samples=num_samples,
            load_image=load_image,
        )
    elif name.startswith("force3d-ref"):
        # e.g. "force3d-ref", "force3d-ref-chain", "force3d-ref-star", "force3d-ref-hybrid"
        suffix = name[len("force3d-ref"):].lstrip("-")
        category = suffix or None
        return Force3DRefDataset(
            category=category,
            num_samples=num_samples,
            load_image=load_image,
        )
    else:
        raise ValueError(f"Dataset '{name}' not recognized.")
