"""Multi-process dispatch: spawn N worker copies of the runner and merge shards."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Tuple

from saturn.datasets import get_dataset
from saturn.pipeline.results import _collect_completed_ids, _merge_worker_results
from saturn.log import progress


def _dispatch_multiproc(args: argparse.Namespace, argv: List[str] | None) -> int:
    """Spawn ``args.num_workers`` subprocess copies of this script.

    Each worker owns its own Python interpreter (avoiding the GIL ceiling),
    its own asyncio event loop, its own AsyncOpenAI client to the shared
    vLLM server, and its own Ray Serve handles (the Ray cluster is shared).
    Samples are partitioned round-robin across workers, results land in
    per-worker JSON files, and the parent combines them at the end.
    """
    all_ids = _enumerate_sample_ids(args)
    results_dir = os.path.join("experiments", args.dataset, args.vlm_model_name)
    base_results = args.results_save_file
    results_stem, results_ext = _split_results_name(base_results)

    remaining = _remaining_ids(all_ids, results_dir, results_stem, results_ext)
    if not remaining:
        progress("[multiproc] nothing left to run; merging existing shards.")
        return _merge_worker_results(
            results_dir, results_stem, results_ext, base_results, args, workers=0
        )

    n = max(1, min(args.num_workers, len(remaining)))
    if n < args.num_workers:
        progress(
            f"[multiproc] only {len(remaining)} samples to run; reducing "
            f"num_workers from {args.num_workers} to {n}"
        )
    # Round-robin partition of the REMAINING ids into n chunks. Round-robin
    # interleaves long and short samples; a pull-based queue would balance
    # better but costs a Ray/model startup per extra spawn.
    chunks = [remaining[i::n] for i in range(n)]
    progress(f"[multiproc] partition: {[len(c) for c in chunks]} samples per worker")

    procs = _spawn_workers(chunks, _worker_base_argv(argv), results_stem, results_ext)
    try:
        rc_max = _wait_for_workers(procs)
    except KeyboardInterrupt:
        progress("[multiproc] interrupted; terminating workers…")
        _terminate_workers(procs)
        return 130

    # Combine per-worker JSONs, de-duplicating by id.
    return _merge_worker_results(
        results_dir, results_stem, results_ext, base_results, args,
        workers=n, rc=rc_max,
    )


def _enumerate_sample_ids(args: argparse.Namespace) -> List[str]:
    """The ids to run: the dataset (loaded without images), restricted to --sample_ids."""
    progress("[multiproc] pre-loading dataset to enumerate sample IDs…")
    full_dataset = get_dataset(args.dataset, num_samples=args.num_samples, load_image=False)
    all_ids = [str(it.get("id", f"item_{i}")) for i, it in enumerate(full_dataset)]
    if args.sample_ids:
        wanted = {str(s) for s in args.sample_ids}
        all_ids = [s for s in all_ids if s in wanted]
    return all_ids


def _split_results_name(base_results: str) -> Tuple[str, str]:
    """(stem, extension) of the results file; worker shards are ``<stem>_w<k><ext>``."""
    if base_results.endswith(".json"):
        return base_results[:-5], ".json"
    return base_results, ""


def _remaining_ids(all_ids: List[str], results_dir: str, results_stem: str,
                   results_ext: str) -> List[str]:
    """The ids that no shard (nor an existing combined file) has completed.

    Resume is global, not per worker. The partition depends on the worker
    count, so a resume with a different --num_workers (or --num_samples, or
    after a dataset change) reshuffles ownership: a worker that only consulted
    its own _wN file would redo another worker's finished samples and skip ids
    whose results live in a file it never reads.
    """
    done_ids = _collect_completed_ids(results_dir, results_stem, results_ext)
    if not done_ids:
        return list(all_ids)
    remaining = [s for s in all_ids if s not in done_ids]
    progress(
        f"[multiproc] resume: {len(done_ids)} ids already complete across "
        f"existing shards; {len(remaining)} remaining of {len(all_ids)}"
    )
    return remaining


def _strip_flag(argv_list: List[str], flag_aliases: List[str], nargs: str) -> List[str]:
    """``argv_list`` without the flag (any alias, ``--f v`` or ``--f=v``) and its value(s).

    ``nargs`` is "1" for one value or "+" for every following non-flag token.
    """
    out: List[str] = []
    i = 0
    while i < len(argv_list):
        tok = argv_list[i]
        base = tok.split("=", 1)[0]
        if base in flag_aliases:
            if "=" in tok:
                i += 1
                continue
            if nargs == "1":
                i += 2
                continue
            # nargs='+': consume all following non-flag tokens
            i += 1
            while i < len(argv_list) and not argv_list[i].startswith("-"):
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _worker_base_argv(argv: List[str] | None) -> List[str]:
    """The parent's command line without the flags each worker sets for itself."""
    base_argv = list(argv) if argv is not None else list(sys.argv[1:])
    base_argv = _strip_flag(base_argv, ["--num_workers"], "1")
    base_argv = _strip_flag(base_argv, ["--sample_ids"], "+")
    base_argv = _strip_flag(base_argv, ["--results_save_file"], "1")
    return base_argv


def _spawn_workers(chunks: List[List[str]], base_argv: List[str], results_stem: str,
                   results_ext: str) -> List[subprocess.Popen]:
    """Start one single-worker run per chunk, each writing its own results shard."""
    procs: List[subprocess.Popen] = []
    for wid, chunk in enumerate(chunks):
        per_worker_results = f"{results_stem}_w{wid}{results_ext}"
        worker_argv = list(base_argv) + [
            "--num_workers", "1",
            "--sample_ids", *chunk,
            "--results_save_file", per_worker_results,
        ]
        cmd = [sys.executable, "-u", "-m", "saturn.cli.main", *worker_argv]
        progress(f"[multiproc] worker {wid}: {len(chunk)} samples → {per_worker_results}")
        procs.append(subprocess.Popen(cmd, env=os.environ.copy()))
    return procs


def _wait_for_workers(procs: List[subprocess.Popen]) -> int:
    """Wait for every worker; returns the highest exit code."""
    rc_max = 0
    for wid, p in enumerate(procs):
        rc = p.wait()
        progress(f"[multiproc] worker {wid} exited rc={rc}")
        rc_max = max(rc_max, rc)
    return rc_max


def _terminate_workers(procs: List[subprocess.Popen]) -> None:
    """Terminate every worker, killing any that has not exited after 10 s."""
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
