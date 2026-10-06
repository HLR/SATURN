"""Multi-worker partitioning, resume, and merge.

Samples are partitioned round-robin by position, so the partition depends on
the worker count. Completed ids are collected across every ``_wN`` shard and
the combined results file, so resuming with a different ``--num_workers``
neither repeats finished work nor skips an id, and the merge reads every shard
on disk while keeping the rows already in the combined file. A hole here would
be silent: the merged set would simply miss questions, which would drop them
from an id-paired comparison.

These tests exercise the result-file helpers directly rather than through the
runner, which needs GPUs.
"""

import json
import os

import pytest

from saturn.pipeline.results import (
    _collect_completed_ids,
    _merge_worker_results,
    _shard_paths,
    load_previous_results,
)


class _Args:
    desc = "test"


def _write_shard(d, name, ids, extra=None):
    """Write a results shard in the runner's dict shape."""
    path = os.path.join(d, name)
    rows = [{"id": i, "correct_final_answer": True, **(extra or {})} for i in ids]
    with open(path, "w") as f:
        json.dump({"results": rows}, f)
    return path


def _partition(all_ids, done, n):
    """The parent's partitioning logic, mirrored for testability."""
    remaining = [s for s in all_ids if s not in done]
    n = max(1, min(n, len(remaining))) if remaining else 0
    return [remaining[i::n] for i in range(n)] if n else []


# --------------------------------------------------------------- completion
def test_completion_is_collected_across_all_shards(tmp_path):
    d = str(tmp_path)
    _write_shard(d, "run_w0.json", ["a", "c"])
    _write_shard(d, "run_w1.json", ["b"])
    assert _collect_completed_ids(d, "run", ".json") == {"a", "b", "c"}


def test_completion_includes_an_existing_combined_file(tmp_path):
    """A finished run leaves only the combined file; resuming must see it."""
    d = str(tmp_path)
    _write_shard(d, "run.json", ["a", "b"])
    assert _collect_completed_ids(d, "run", ".json") == {"a", "b"}


def test_unreadable_shard_does_not_abort_the_scan(tmp_path):
    """A shard caught mid-write must not take the whole resume down."""
    d = str(tmp_path)
    _write_shard(d, "run_w0.json", ["a"])
    with open(os.path.join(d, "run_w1.json"), "w") as f:
        f.write("{ truncated")
    assert _collect_completed_ids(d, "run", ".json") == {"a"}


# ------------------------------------------------------- resume correctness
def test_resume_with_a_different_worker_count_loses_nothing(tmp_path):
    """4 workers stop partway, resume with 2.

    Worker 0 owns [0,4,8,...] on the first run and [0,2,4,...] on the second,
    so it must skip samples worker 1 finished and see ids sitting in shards it
    does not own.
    """
    d = str(tmp_path)
    all_ids = [f"s{i}" for i in range(20)]

    # first run: 4 workers, each completes only its first two samples
    first = _partition(all_ids, set(), 4)
    completed = []
    for wid, chunk in enumerate(first):
        _write_shard(d, f"run_w{wid}.json", chunk[:2])
        completed += chunk[:2]

    # resume with a DIFFERENT worker count
    done = _collect_completed_ids(d, "run", ".json")
    assert done == set(completed)
    second = _partition(all_ids, done, 2)

    scheduled = [i for chunk in second for i in chunk]
    assert not (set(scheduled) & done), "already-finished ids were rescheduled"
    assert set(scheduled) | done == set(all_ids), "some ids would never run"
    assert len(scheduled) == len(set(scheduled)), "an id was scheduled twice"


@pytest.mark.parametrize("first_n,second_n", [(4, 2), (2, 4), (8, 3), (1, 4), (4, 1)])
def test_resume_is_worker_count_independent(tmp_path, first_n, second_n):
    d = str(tmp_path)
    all_ids = [f"s{i}" for i in range(31)]           # deliberately not divisible
    for wid, chunk in enumerate(_partition(all_ids, set(), first_n)):
        _write_shard(d, f"run_w{wid}.json", chunk[: len(chunk) // 2])
    done = _collect_completed_ids(d, "run", ".json")
    scheduled = [i for c in _partition(all_ids, done, second_n) for i in c]
    assert set(scheduled).isdisjoint(done)
    assert set(scheduled) | done == set(all_ids)
    assert len(scheduled) == len(set(scheduled))


def test_resume_after_a_crash_mid_shard(tmp_path):
    """A worker that died leaves a partial shard; the rest must still be picked up."""
    d = str(tmp_path)
    all_ids = [f"s{i}" for i in range(10)]
    _write_shard(d, "run_w0.json", ["s0", "s2"])     # crashed before s4, s6, s8
    done = _collect_completed_ids(d, "run", ".json")
    scheduled = [i for c in _partition(all_ids, done, 3) for i in c]
    assert set(scheduled) == set(all_ids) - {"s0", "s2"}


def test_nothing_remaining_when_every_id_is_done(tmp_path):
    d = str(tmp_path)
    all_ids = ["a", "b", "c"]
    _write_shard(d, "run_w0.json", all_ids)
    done = _collect_completed_ids(d, "run", ".json")
    assert _partition(all_ids, done, 4) == []


# -------------------------------------------------------------------- merge
def test_merge_deduplicates_by_id(tmp_path):
    """A duplicate id across shards must not inflate the merged row count."""
    d = str(tmp_path)
    _write_shard(d, "run_w0.json", ["a", "b"])
    _write_shard(d, "run_w1.json", ["b", "c"])       # 'b' repeated
    _merge_worker_results(d, "run", ".json", "run.json", _Args(), workers=2)
    rows = json.load(open(os.path.join(d, "run.json")))["results"]
    assert len(rows) == 3
    assert sorted(r["id"] for r in rows) == ["a", "b", "c"]


def test_merge_prefers_the_newer_shard_on_duplicate_ids(tmp_path):
    d = str(tmp_path)
    _write_shard(d, "run_w0.json", ["a"], extra={"tag": "old"})
    os.utime(os.path.join(d, "run_w0.json"), (1, 1))          # force older mtime
    _write_shard(d, "run_w1.json", ["a"], extra={"tag": "new"})
    _merge_worker_results(d, "run", ".json", "run.json", _Args(), workers=2)
    rows = json.load(open(os.path.join(d, "run.json")))["results"]
    assert len(rows) == 1 and rows[0]["tag"] == "new"


def test_merge_keeps_rows_that_have_no_id(tmp_path):
    """Un-keyed rows cannot be de-duplicated, but must not be dropped either."""
    d = str(tmp_path)
    with open(os.path.join(d, "run_w0.json"), "w") as f:
        json.dump({"results": [{"id": "a"}, {"no_id": 1}]}, f)
    _merge_worker_results(d, "run", ".json", "run.json", _Args(), workers=1)
    rows = json.load(open(os.path.join(d, "run.json")))["results"]
    assert len(rows) == 2


def test_merge_round_trips_through_the_completion_scan(tmp_path):
    """After merging, a further resume sees every id as done."""
    d = str(tmp_path)
    _write_shard(d, "run_w0.json", ["a", "b"])
    _write_shard(d, "run_w1.json", ["c"])
    _merge_worker_results(d, "run", ".json", "run.json", _Args(), workers=2)
    assert _collect_completed_ids(d, "run", ".json") == {"a", "b", "c"}


def test_shard_paths_ignores_unrelated_files(tmp_path):
    d = str(tmp_path)
    _write_shard(d, "run_w0.json", ["a"])
    _write_shard(d, "other_w0.json", ["z"])
    _write_shard(d, "run.json", ["a"])               # combined, not a shard
    assert [os.path.basename(p) for p in _shard_paths(d, "run", ".json")] == [
        "run_w0.json"
    ]


def test_merge_reads_shards_from_a_larger_previous_worker_count(tmp_path):
    """The merge reads every shard on disk, not only ``_w0`` .. ``_w{n-1}`` of
    the current worker count: a run of 20 samples completed with 4 workers and
    re-merged with 2 keeps all 20 rows.
    """
    d = str(tmp_path)
    all_ids = [f"s{i}" for i in range(20)]
    for wid, chunk in enumerate(_partition(all_ids, set(), 4)):
        _write_shard(d, f"run_w{wid}.json", chunk)

    # merge as if the current run used only 2 workers
    _merge_worker_results(d, "run", ".json", "run.json", _Args(), workers=2)
    rows = json.load(open(os.path.join(d, "run.json")))["results"]
    assert len(rows) == 20, f"merge dropped rows: only {len(rows)}/20 survived"
    assert {r["id"] for r in rows} == set(all_ids)


def test_merge_preserves_pre_existing_combined_rows(tmp_path):
    """A merge must never destroy rows already in the combined results file.

    Example: a single-worker run wrote 185 rows to `<stem>.json`; resuming with
    3 workers runs only the 15 missing samples (`_collect_completed_ids` reads
    shards AND the combined file), and the merge must then keep the 185 rows
    alongside the 15 new ones. The combined file seeds the merge; shards win
    only on conflicting ids.
    """
    import json
    import saturn.pipeline.results as R

    d = tmp_path
    stem, ext = "run", ".json"
    # 185-row style prior run (use 5 for speed), written as the COMBINED file
    prior = [{"id": f"old{i}", "correct_final_answer": True} for i in range(5)]
    (d / f"{stem}{ext}").write_text(json.dumps({"results": prior}))
    # a later multi-worker resume produces two shards with the remaining ids
    (d / f"{stem}_w0{ext}").write_text(
        json.dumps({"results": [{"id": "new0", "correct_final_answer": False}]})
    )
    (d / f"{stem}_w1{ext}").write_text(
        json.dumps({"results": [{"id": "new1", "correct_final_answer": True}]})
    )

    class _Args:
        desc = ""

    R._merge_worker_results(str(d), stem, ext, f"{stem}{ext}", _Args(), workers=2)

    out = json.loads((d / f"{stem}{ext}").read_text())
    rows = out if isinstance(out, list) else out["results"]
    ids = {r["id"] for r in rows}
    assert ids == {"old0", "old1", "old2", "old3", "old4", "new0", "new1"}, ids
    assert len(rows) == 7, f"expected 7 rows, got {len(rows)} — prior rows were dropped"


def test_merge_shard_wins_over_stale_combined_row(tmp_path):
    """When an id appears in both, the fresher shard row must win."""
    import json
    import saturn.pipeline.results as R

    d = tmp_path
    stem, ext = "run", ".json"
    (d / f"{stem}{ext}").write_text(
        json.dumps({"results": [{"id": "x", "correct_final_answer": False, "src": "old"}]})
    )
    (d / f"{stem}_w0{ext}").write_text(
        json.dumps({"results": [{"id": "x", "correct_final_answer": True, "src": "new"}]})
    )

    class _Args:
        desc = ""

    R._merge_worker_results(str(d), stem, ext, f"{stem}{ext}", _Args(), workers=1)
    out = json.loads((d / f"{stem}{ext}").read_text())
    rows = out if isinstance(out, list) else out["results"]
    assert len(rows) == 1 and rows[0]["src"] == "new"


# ---------------------------------------------------------- sibling runs
def _write_rows(path, rows):
    with open(path, "w") as f:
        json.dump({"results": rows}, f)


def test_sibling_run_is_not_a_shard(tmp_path):
    d = str(tmp_path)
    _write_rows(os.path.join(d, "run_mmsi_wo_caption.json"), [{"id": "q1", "note": "OTHER"}])
    _write_rows(os.path.join(d, "run_mmsi_warm_summary.json"), [{"id": "q2", "note": "OTHER"}])
    _write_rows(os.path.join(d, "run_mmsi_w0.json"), [{"id": "q3"}])
    _write_rows(os.path.join(d, "run_mmsi_w12.json"), [{"id": "q4"}])

    shards = sorted(os.path.basename(p) for p in _shard_paths(d, "run_mmsi", ".json"))
    assert shards == ["run_mmsi_w0.json", "run_mmsi_w12.json"]
    assert _collect_completed_ids(d, "run_mmsi", ".json") == {"q3", "q4"}

    _merge_worker_results(d, "run_mmsi", ".json", "run_mmsi.json",
                          _Args(), workers=0)
    merged = json.load(open(os.path.join(d, "run_mmsi.json")))["results"]
    assert sorted(r["id"] for r in merged) == ["q3", "q4"]


# ------------------------------------------------------- results file
def test_corrupt_results_file_is_kept_not_overwritten(tmp_path):
    """An unreadable results file is moved aside as <name>.corrupt-*, never overwritten; resume starts empty."""
    f = tmp_path / "r.json"; f.write_text('{"results": [{"id": "a"}')   # truncated
    results, ids = load_previous_results(str(f))
    assert results == [] and ids == set()
    assert not f.exists() and any(p.name.startswith("r.json.corrupt-") for p in tmp_path.iterdir())
    g = tmp_path / "ok.json"; g.write_text(json.dumps({"results": [{"id": "a", "correct_final_answer": True}]}))
    results, ids = load_previous_results(str(g)); assert ids == {"a"} and g.exists()
