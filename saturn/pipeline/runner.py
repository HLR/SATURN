"""Async benchmark orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import functools
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set

from saturn.datasets import get_dataset
from saturn.pipeline.build_scene import build_scene
from saturn.pipeline.evaluate import evaluate_answer
from saturn.pipeline.models import Models, bring_up_services, build_models_async
from saturn.pipeline.results import generate_report, load_previous_results, save_results
from saturn.pipeline.sample import execute, item_fields, pre_execute
from saturn.pipeline.state import _PreExecState
from saturn.log import get_logger, progress

log = get_logger(__name__)


@dataclass
class _Outputs:
    """Where a run writes its results and (optionally) its debug reports."""
    results_dir: str
    results_path: str
    report_dir: Optional[str] = None
    prompt_template_text: Optional[str] = None


@dataclass
class _SampleRunner:
    """What every sample's two phases share: models, scene builder, and concurrency limits."""
    args: argparse.Namespace
    models: Models
    build_scene_fn: Callable
    pre_exec_sem: asyncio.Semaphore
    exec_sem: asyncio.Semaphore
    processed_ids: Set[str]


@dataclass
class _Tally:
    n_correct: int = 0
    n_total: int = 0


async def run_benchmark(args: argparse.Namespace) -> List[dict]:
    """Run each sample of this split through the pipeline; grade, save, and return the results.

    Phase A (``pre_execute``) of one sample overlaps phase B (``execute``) of
    another, each phase under its own concurrency limit.
    """
    assert 0 <= args.split_index < args.num_splits, (
        f"split_index={args.split_index} out of range [0, {args.num_splits})"
    )

    full_dataset = await _load_dataset(args)
    dataset = _select_split(full_dataset, args)
    if not dataset:
        return []

    out = _prepare_results_file(args)
    results, processed_ids = load_previous_results(out.results_path)
    _prepare_reports(args, out)

    services = await bring_up_services(args)
    loop = asyncio.get_running_loop()
    models = build_models_async(args, services, loop, dataset=full_dataset)
    runner = _SampleRunner(
        args=args,
        models=models,
        build_scene_fn=_scene_builder(args, loop),
        pre_exec_sem=asyncio.Semaphore(args.max_concurrent_samples),
        exec_sem=asyncio.Semaphore(args.exec_concurrency),
        processed_ids=processed_ids,
    )
    progress(
        f"Pipeline concurrency: pre_exec={args.max_concurrent_samples}, "
        f"execute={args.exec_concurrency} "
        f"(sample N's execute pipelines against sample N+1's pre_exec)"
    )
    tally = _Tally()
    t_start = time.time()

    tasks = [asyncio.create_task(_run_sample(runner, idx, item))
             for idx, item in enumerate(dataset)]
    try:
        for fut in asyncio.as_completed(tasks):
            payload = await fut
            if payload is None:
                continue
            await _record_sample(payload, tally, results, processed_ids, out, args, len(dataset))
    finally:
        await services["vlm_client"].aclose()

    _print_summary(full_dataset, dataset, tally, time.time() - t_start)
    _print_category_accuracy(results)

    await asyncio.to_thread(save_results, results, out.results_path, args.desc, args)
    progress(f"\nResults saved to: {out.results_path}")
    if out.report_dir:
        await _write_report_index(results, out)
    return results


async def _load_dataset(args: argparse.Namespace) -> list:
    """Load the dataset (with images), restricted to --sample_ids when given."""
    progress(f"Loading dataset: {args.dataset}...")
    full_dataset = await asyncio.to_thread(
        get_dataset, args.dataset, num_samples=args.num_samples, load_image=True
    )
    progress(f"Dataset loaded: {len(full_dataset)} samples.")

    if args.sample_ids:
        wanted = {str(s) for s in args.sample_ids}
        full_dataset = [it for it in full_dataset if str(it.get("id")) in wanted]
        progress(f"Filtered by --sample_ids {sorted(wanted)}: {len(full_dataset)} samples.")
    return full_dataset


def _select_split(full_dataset: list, args: argparse.Namespace) -> list:
    """This process's contiguous chunk of the dataset (--split_index of --num_splits)."""
    if args.num_splits == 1:
        dataset = full_dataset
    else:
        chunk = (len(full_dataset) + args.num_splits - 1) // args.num_splits
        start = args.split_index * chunk
        end = min(start + chunk, len(full_dataset))
        if start >= len(full_dataset):
            progress("Split out of range. Nothing to process.")
            return []
        dataset = [full_dataset[i] for i in range(start, end)]
        progress(f"Split {args.split_index + 1}/{args.num_splits}: indices {start}..{end - 1}")

    if not dataset:
        progress("No samples to process.")
    return dataset


def _prepare_results_file(args: argparse.Namespace) -> _Outputs:
    """Create the results directory and remove an older results file on request."""
    results_dir = os.path.join("experiments", args.dataset, args.vlm_model_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, args.results_save_file)
    progress(f"Results will be saved to: {results_path}")

    if args.remove_older_results and os.path.exists(results_path):
        os.remove(results_path)
        progress("Removed previous results file.")
    return _Outputs(results_dir=results_dir, results_path=results_path)


def _prepare_reports(args: argparse.Namespace, out: _Outputs) -> None:
    """With --generate_reports: create the report directory and read the prompt it shows."""
    if not args.generate_reports:
        return
    out.report_dir = args.report_dir or os.path.join(
        "reports", Path(args.results_save_file).stem
    )
    os.makedirs(out.report_dir, exist_ok=True)
    if os.path.exists(args.code_prompt):
        with open(args.code_prompt) as f:
            out.prompt_template_text = f.read()
    progress(f"Reports will be generated in: {out.report_dir}/")


def _scene_builder(args: argparse.Namespace, loop: asyncio.AbstractEventLoop) -> Callable:
    """``build_scene`` bound to this run's event loop, args, and in-process scene cache."""
    from saturn.scene.build.cache import SceneCache
    _scene_cache = SceneCache(
        max_entries=max(8, 2 * int(getattr(args, 'max_concurrent_samples', 4) or 4)))
    return functools.partial(
        build_scene,
        loop=loop,
        args=args,
        scene_cache=_scene_cache,
    )


async def _run_sample(runner: _SampleRunner, idx: int, item: dict):
    """Run one sample's two phases; any error becomes the sample's result.

    Returns (idx, item, item_id, sample_result), or None for a sample that a
    previous run already completed.
    """
    item_id = item.get("id", f"item_{idx}")
    if str(item_id) in runner.processed_ids:
        return None
    try:
        async with runner.pre_exec_sem:
            state = await asyncio.to_thread(
                pre_execute, item, item_id, runner.models, runner.args,
                build_scene_fn=runner.build_scene_fn,
            )
        if state.early_return:
            sample_result = state.result
        else:
            async with runner.exec_sem:
                sample_result = await _execute_with_timeout(runner, item, item_id, state)
    except Exception as e:
        tb = traceback.format_exc()
        sample_result = {
            **item_fields(item, item_id),
            "error": f"MainLoop Error: {type(e).__name__}: {e}",
            "error_type": type(e).__name__,
            "correct_final_answer": False,
        }
        log.error(f"[{item_id}] Error: {e}\n{tb}")
    return idx, item, item_id, sample_result


async def _execute_with_timeout(runner: _SampleRunner, item: dict, item_id,
                                state: _PreExecState) -> dict:
    """Phase B under --exec_timeout; a timed-out sample is recorded as an error."""
    _to = float(getattr(runner.args, "exec_timeout", 0) or 0)
    _coro = asyncio.to_thread(execute, item, item_id, runner.models, runner.args, state)
    try:
        return await (asyncio.wait_for(_coro, timeout=_to) if _to > 0 else _coro)
    except asyncio.TimeoutError:
        log.error(f"[{item_id}] EXEC TIMEOUT after {_to:.0f}s -- recorded as error; "
                  f"its thread is abandoned (still running).")
        return {
            **item_fields(item, item_id),
            "error": f"exec_timeout: execute phase exceeded {_to:.0f}s",
            "error_type": "TimeoutError",
            "correct_final_answer": False,
            "objects_count": (getattr(state, "result", None) or {}).get("objects_count"),
        }


async def _record_sample(payload, tally: _Tally, results: List[dict], processed_ids: Set[str],
                         out: _Outputs, args: argparse.Namespace, n_samples: int) -> None:
    """Grade a finished sample, log the running accuracy, and store its result."""
    idx, item, item_id, sample_result = payload
    tally.n_total += 1
    question = item.get("query", "")
    gt_answer = item.get("answer", "")
    is_correct = evaluate_answer(
        sample_result.get("final_answer_text"), gt_answer, question, item=item,
        # result_record is required: REF is graded on its
        # `predicted_bbox`.
        result_record=sample_result,
    )
    sample_result["correct_final_answer"] = is_correct
    if is_correct:
        tally.n_correct += 1

    n_correct, n_total = tally.n_correct, tally.n_total
    acc = n_correct / n_total * 100 if n_total > 0 else 0.0
    progress(
        f"[{item_id}] GT='{gt_answer}' | "
        f"Pred='{sample_result.get('final_answer_text', 'None')}' | "
        f"Correct={is_correct} | Acc={acc:.1f}% ({n_correct}/{n_total})"
    )

    results.append(sample_result)
    processed_ids.add(str(item_id))

    if out.report_dir:
        await asyncio.to_thread(
            generate_report,
            sample_result,
            len(results) - 1,
            n_samples,
            out.results_path,
            out.report_dir,
            out.prompt_template_text,
            out.results_dir,
            item_id,
        )

    if args.save_periodically and n_total % args.save_interval == 0:
        await asyncio.to_thread(save_results, results, out.results_path, args.desc, args)


def _print_summary(full_dataset: list, dataset: list, tally: _Tally, total_time: float) -> None:
    n_correct, n_total = tally.n_correct, tally.n_total
    progress("\n" + "=" * 40)
    progress("--- Processing Summary ---")
    progress(
        f"Total samples: {len(full_dataset)} | Split: {len(dataset)} | Processed: {n_total}"
    )
    if n_total > 0:
        progress(f"Accuracy: {n_correct / n_total * 100:.2f}% ({n_correct}/{n_total})")
    progress(f"Total time: {total_time:.1f}s | Avg: {total_time / max(n_total, 1):.1f}s/sample")


def _print_category_accuracy(results: List[dict]) -> None:
    """Accuracy per question_type over every graded result (including resumed ones)."""
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        qtype = r.get("question_type", "unknown")
        if r.get("correct_final_answer") is not None:
            cat_stats[qtype]["total"] += 1
            if r["correct_final_answer"]:
                cat_stats[qtype]["correct"] += 1
    if cat_stats:
        progress("\nPer-category accuracy:")
        for cat, stats in sorted(cat_stats.items()):
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                progress(f"  {cat}: {acc:.1f}% ({stats['correct']}/{stats['total']})")


async def _write_report_index(results: List[dict], out: _Outputs) -> None:
    """Write index.html for the per-sample debug reports; a failure is logged."""
    try:
        from saturn.reports.debug_report import render_index

        idx_html = await asyncio.to_thread(
            render_index, results, out.results_path, out.report_dir
        )
        idx_path = os.path.join(out.report_dir, "index.html")
        with open(idx_path, "w") as f:
            f.write(idx_html)
        progress(f"Report index: {idx_path}")
    except Exception as e:
        log.error(f"Report index generation failed: {e}")
