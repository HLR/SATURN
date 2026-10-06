"""Results persistence, report generation, and multiproc shard merge helpers."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from saturn.log import get_logger, progress

log = get_logger(__name__)


def load_previous_results(results_file: str) -> Tuple[List[Dict], set]:
    results = []
    processed_ids: set = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
                results = loaded["results"]
                for r in results:
                    rid = r.get("id")
                    if rid:
                        processed_ids.add(str(rid))
                progress(
                    f"Loaded {len(results)} previous results "
                    f"({len(processed_ids)} unique IDs)."
                )
            else:
                log.warning(f"WARNING: {results_file} has an unexpected shape; it will be overwritten.")
        except Exception as e:
            # Never overwrite a file we could not parse: keep it for salvage.
            import time as _t
            keep = f"{results_file}.corrupt-{int(_t.time())}"
            os.replace(results_file, keep)
            log.error(f"Error loading results: {e}. Moved to {keep}; starting fresh.")
            return [], set()
    else:
        progress("No previous results file found. Starting fresh.")
    return results, processed_ids


class _SafeEncoder(json.JSONEncoder):
    """Handles numpy/torch scalar types that stdlib json can't serialize."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
        except ImportError:
            pass
        return super().default(obj)


def save_results(results: List[Dict], path: str, desc: str, args=None):
    desc_full = desc
    if args and args.num_splits > 1:
        desc_full += f" (Split {args.split_index + 1}/{args.num_splits})"
    progress(f"Saving {len(results)} results to: {path}")
    try:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(
                {
                    "description": desc_full,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": results,
                },
                f,
                cls=_SafeEncoder,   # compact: this file is rewritten every save_interval samples
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic: reader always sees a complete file
    except Exception as e:
        log.error(f"Error saving results: {e}")


def generate_report(
    sample_result: Dict,
    result_index: int,
    total_samples: int,
    results_path: str,
    report_dir: str,
    prompt_template_text: Optional[str],
    results_dir: str,
    item_id: str,
):
    """Generate an HTML debug report for one sample."""
    try:
        from saturn.reports.debug_report import render_sample_report
        from saturn.reports.helpers import default_mmsi_image_dir

        scene_dir = os.path.join(results_dir, "scenes")
        html = render_sample_report(
            sample_result,
            result_index,
            total_samples,
            result_json=results_path,
            prompt_template=prompt_template_text,
            program_cache=None,
            planner_cache=None,
            image_dir=default_mmsi_image_dir(),
            scene_dir=scene_dir,
        )
        # Samples finish out of order, so neighbour ids are unknown here and
        # the Prev/Next placeholders would 404; the Index link stays.
        html = re.sub(r"<a href='sample_[^']*_(?:prev|next)\.html'>(?:Prev|Next)</a>", "", html)
        rpath = os.path.join(report_dir, f"sample_{item_id}.html")
        with open(rpath, "w") as f:
            f.write(html)
        tag = "ok" if sample_result.get("correct_final_answer") else "WRONG"
        progress(f"[{item_id}] Report: {rpath} [{tag}]")
    except Exception as e:
        log.error(f"[{item_id}] Report generation failed: {e}")


def _iter_result_rows(path: str):
    """Yield result dicts from a results file in either supported shape."""
    import json
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:                      # a shard mid-write, or corrupt
        log.warning(f"[multiproc] WARNING: could not read {path}: {type(e).__name__}")
        return
    rows = data if isinstance(data, list) else data.get("results")
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                yield r


def _shard_paths(results_dir: str, stem: str, ext: str) -> List[str]:
    """Every per-worker shard for this run, oldest first.

    Ordering matters for the merge: later files win on duplicate ids, so a
    fresher shard supersedes a stale one left by an earlier run.
    """
    import glob
    ext = ext or ".json"
    # Only real worker shards ``{stem}_w<N>{ext}``: a sibling run such as
    # ``{stem}_wo_caption{ext}`` also matches the glob and must not be merged.
    shard_re = re.compile(re.escape(stem) + r"_w\d+" + re.escape(ext))
    paths = [
        p for p in glob.glob(os.path.join(results_dir, f"{stem}_w*{ext}"))
        if shard_re.fullmatch(os.path.basename(p))
    ]
    return sorted(paths, key=lambda p: os.path.getmtime(p))


def _collect_completed_ids(results_dir: str, stem: str, ext: str) -> set:
    """Ids already finished, across ALL shards plus any combined file.

    This is what makes resume independent of the worker count: completion is a
    property of the run, not of whichever shard happened to own an id last time.
    """
    done: set = set()
    candidates = _shard_paths(results_dir, stem, ext)
    combined = os.path.join(results_dir, f"{stem}{ext}")
    if os.path.exists(combined):
        candidates.append(combined)
    for path in candidates:
        for r in _iter_result_rows(path):
            rid = r.get("id")
            if rid is not None:
                done.add(str(rid))
    return done


def _merge_worker_results(
    results_dir: str, stem: str, ext: str, base_results: str,
    args: argparse.Namespace, workers: int, rc: int = 0,
) -> int:
    """Merge shards into the requested results path, de-duplicating by id.

    An id can legitimately appear in two shards when a previous run used a
    different worker count; keeping both would inflate the row count and make
    an id-paired comparison ambiguous. Later shards win.
    """
    import datetime
    import json

    merged: "dict[str, dict]" = {}
    unkeyed: List[dict] = []
    # Seed from any PRE-EXISTING combined file first, so rows written by an
    # earlier run (e.g. a single-worker run being resumed by a multi-worker
    # one) survive the merge. Shards are applied after and therefore win on
    # conflicts.
    _existing = os.path.join(results_dir, f"{stem}{ext}")
    if os.path.exists(_existing):
        for r in _iter_result_rows(_existing):
            rid = r.get("id")
            if rid is None:
                unkeyed.append(r)
            else:
                merged[str(rid)] = r
    for path in _shard_paths(results_dir, stem, ext):
        for r in _iter_result_rows(path):
            rid = r.get("id")
            if rid is None:
                unkeyed.append(r)
            else:
                merged[str(rid)] = r

    combined = list(merged.values()) + unkeyed
    final_path = os.path.join(results_dir, base_results)
    os.makedirs(results_dir, exist_ok=True)
    with open(final_path, "w") as f:
        json.dump(
            {
                "description": getattr(args, "desc", "") or "",
                "timestamp": datetime.datetime.now().isoformat(),
                "results": combined,
                "multiproc_workers": workers,
            },
            f,
        )
    progress(
        f"[multiproc] combined {len(combined)} unique results "
        f"({len(unkeyed)} without ids) -> {final_path}"
    )
    return rc
