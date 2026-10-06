"""Benchmark CLI: argument parser and the per-dataset defaults of the released runs."""

from __future__ import annotations

from saturn.settings import env
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# The per-benchmark setup lives in configs/benchmarks.json: the code-generator prompt
# ("code_prompt") and whether camera-pose constraints are read from the question
# ("pose_constraints", MindCube only). "default" applies to a dataset with no entry.
# Everything else (planner, verification, coder, VLM, ...) is one code path or a
# plain default below; a flag given on the command line overrides the file.
BENCHMARKS_FILE = Path(__file__).resolve().parents[2] / "configs" / "benchmarks.json"
_ARG_NAMES = {"code_prompt": "code_prompt", "program_cache": "program_cache",
              "pose_constraints": "use_pose_constraints"}


def _load_benchmarks(path: Path = BENCHMARKS_FILE):
    """(per-dataset defaults, fallback defaults), as CLI argument names."""
    cfg = json.loads(path.read_text())
    def as_args(entry: dict) -> dict:
        unknown = set(entry) - set(_ARG_NAMES)
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; expected {sorted(_ARG_NAMES)}")
        return {_ARG_NAMES[k]: v for k, v in entry.items()}
    per_dataset = {name: {"args": as_args(entry)} for name, entry in cfg["benchmarks"].items()}
    return per_dataset, as_args(cfg["default"])


DATASET_DEFAULTS, _FALLBACK_ARGS = _load_benchmarks()


def dataset_defaults(dataset: str) -> Optional[Dict[str, Dict]]:
    """The DATASET_DEFAULTS entry for a dataset name ("mindcube-among", "mmsi-MSR", ...)."""
    for family in sorted(DATASET_DEFAULTS, key=len, reverse=True):
        if dataset.startswith(family):
            return DATASET_DEFAULTS[family]
    return None


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the flags left unset with the defaults of ``args.dataset``.

    Flags given on the command line win."""
    entry = dataset_defaults(args.dataset) or {}
    for name, fallback in _FALLBACK_ARGS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, entry.get("args", {}).get(name, fallback))
    return args


def _add_path_args(parser: argparse.ArgumentParser) -> None:
    """Dataset, prompt, cache and results paths."""
    parser.add_argument(
        "--dataset", type=str, default="mindcube",
        help="Benchmark to run: mindcube[-<type>], mindcubedev[-<type>], mmsi[-<category>], force3d-ref[-<topology>] or "
             "force3d-puzzle[-<topology>]; also selects the defaults in configs/benchmarks.json.",
    )
    parser.add_argument(
        "--code_prompt",
        type=str,
        default=None,
        help="Prompt file given to the code generator; unset = per dataset (configs/benchmarks.json).",
    )
    parser.add_argument(
        "--results_save_file", type=str, default="results.json",
        help="Results file name, written under experiments/<dataset>/<vlm_model_name>/.",
    )
    parser.add_argument(
        "--program_cache",
        type=str,
        default=None,
        help="Program cache file (JSON; the objects blocks go to <name>_objects.json beside it). "
             "Unset = per dataset (configs/benchmarks.json).",
    )
    parser.add_argument(
        "--programs_by_id",
        type=str,
        default=None,
        help="JSON {sample id: program} to replay instead of generating (scripts/export_programs.py "
             "builds it from a results file). Samples not in it are generated as usual.",
    )
    parser.add_argument(
        "--desc", type=str, default="Benchmark run",
        help="Text stored in the \"description\" field of the results file.",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """VLM and codegen LLM."""
    parser.add_argument("--vlm_model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Served VLM id; must match vLLM's --served-model-name.")
    parser.add_argument(
        "--code_gen_model_name", type=str, default="deepseek-chat",
        help="Model id sent to the code-generation LLM (see --code_gen_provider).",
    )
    parser.add_argument(
        "--code_gen_provider",
        type=str,
        choices=["deepseek", "openrouter"],
        default="deepseek",
        help=(
            "Code-generation LLM endpoint. 'deepseek': https://api.deepseek.com (DEEPSEEK_API_KEY). "
            "'openrouter': OpenRouter (OPENROUTER_API_KEY), pinned to the upstream and quantization in "
            "SAPY_OPENROUTER_PROVIDER / SAPY_OPENROUTER_QUANT; use it with --code_gen_model_name "
            "deepseek/deepseek-v4-flash to run the pinned model behind the reported results."
        ),
    )


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    """Which samples run: splits, sample count, sample ids."""
    parser.add_argument(
        "--num_splits", type=int, default=1,
        help="Cut the selected samples into this many contiguous chunks; this process runs one.",
    )
    parser.add_argument(
        "--split_index", type=int, default=0,
        help="Which chunk (0-based) of --num_splits this process runs.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Load only the first N samples of the dataset; unset = all.",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        default=None,
        help="Only process samples with these IDs (filter applied after dataset load).",
    )


def _add_saving_args(parser: argparse.ArgumentParser) -> None:
    """When results and caches are written."""
    parser.add_argument(
        "--save_periodically",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write the results file every --save_interval finished samples "
             "(it is always written at the end).",
    )
    parser.add_argument(
        "--write_program_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write newly generated programs to --program_cache (and its _objects.json file). "
             "Off: the cache is read but never written.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5,
        help="Number of finished samples between periodic saves (see --save_periodically).",
    )
    parser.add_argument(
        "--remove_older_results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete an existing results file before the run. Off: its samples are kept and skipped.",
    )


def _add_planner_args(parser: argparse.ArgumentParser) -> None:
    """Query planner cache."""
    parser.add_argument(
        "--planner_cache", type=str, default="cache/planner.json",
        help="Query-planner cache file: plans are read from it and new plans written to it.",
    )


def _add_pose_constraint_args(parser: argparse.ArgumentParser) -> None:
    """Camera pose constraints.

    A separate VLM call extracts the camera-relative-pose constraints stated in the question
    text and applies them via scene.constraint.* before the planner runs.
    """
    parser.add_argument(
        "--use_pose_constraints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable VLM-based camera pose constraint extraction: call "
            "constraint_extractor on each sample's question text + images, then "
            "apply the returned constraints to scene.cameras via "
            "scene.constraint.rotation/same_position. Default: per dataset "
            "(on for MindCube only; see configs/benchmarks.json)."
        ),
    )


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    """Per-sample HTML debug reports."""
    parser.add_argument(
        "--generate_reports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write a per-sample HTML debug report and the grounded-scene dumps it shows.",
    )
    parser.add_argument(
        "--report_dir", type=str, default=None,
        help="Directory of the debug reports; unset = reports/<results_save_file stem>.",
    )


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    """Concurrency, timeouts, worker processes and service endpoints."""
    parser.add_argument(
        "--max_concurrent_samples",
        type=int,
        default=1,
        help="Concurrency for the PRE-EXECUTE phase (plan + scene-build + "
             "detect + code-gen). Default 1 because VGGT/oriany contend "
             "between samples; set higher only if those services have headroom.",
    )
    parser.add_argument(
        "--exec_concurrency",
        type=int,
        default=1,
        help="Concurrency for the EXECUTE phase (program execution + main-VLM "
             "scoring). Default 1 because vLLM scoring throughput drops "
             "under multi-sample contention. With 1+1, "
             "sample N's execute pipelines naturally against sample N+1's "
             "pre-execute (different services).",
    )
    parser.add_argument(
        "--exec_timeout", type=float, default=900.0,
        help="Per-sample wall-clock cap (s) on the execute phase (VLM scoring + program). "
             "A pathological program otherwise blocks its thread forever and, via the GIL, "
             "starves every other in-flight sample. "
             "On timeout the sample is recorded as an error and the run continues; the "
             "stuck thread cannot be killed and keeps burning one core. 0 disables.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of OS-level worker processes for sample-level "
             "parallelism. >1 spawns N subprocess copies of this script, "
             "each owning its own Python interpreter / asyncio loop / "
             "vLLM client, which avoids the single-process GIL limit. "
             "Each worker handles a round-robin "
             "subset of samples; the parent combines per-worker JSONs at the "
             "end. Reports go to the same dir; filenames are per-sample so "
             "no collision.",
    )
    parser.add_argument(
        "--vlm_max_concurrency",
        type=int,
        default=32,
        help="Max concurrent in-flight requests from the VLM client to the vLLM server.",
    )
    parser.add_argument(
        "--vlm_base_url",
        type=str,
        default=env("SAPY_VLM_BASE_URL"),
        help="OpenAI-compatible endpoint of the vLLM server that serves the VLM. "
             "Its default is $SAPY_VLM_BASE_URL.",
    )
    parser.add_argument(
        "--skip_ray_handles",
        action="store_true",
        default=False,
        help="Skip Ray Serve handle resolution (useful for CLI dry-runs).",
    )


def build_parser() -> argparse.ArgumentParser:
    """The benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run multi-view spatial reasoning benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # The order of the calls is the order of the flags in --help.
    _add_path_args(parser)
    _add_model_args(parser)
    _add_split_args(parser)
    _add_saving_args(parser)
    _add_planner_args(parser)
    _add_pose_constraint_args(parser)
    _add_report_args(parser)
    _add_execution_args(parser)
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    return apply_dataset_defaults(build_parser().parse_args(argv))
