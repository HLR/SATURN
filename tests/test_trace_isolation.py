"""The ProbabilisticTensor execution trace must be per-task, not global.

With --exec_concurrency > 1 several samples run at once; each sample's record
must carry only its own predicate scores, indexed against its own scene's
objects, so the trace lives in a per-task context rather than on the class.
"""
import asyncio

import torch

from saturn.soft_logic.tensor import ProbabilisticTensor as PT


def _trace_one(tag, n):
    PT.start_cache()
    for i in range(n):
        PT(torch.tensor([0.1 * i, 0.2, 0.3]), extra_info=f"{tag}-{i}")
    return PT.end_cache()


def _tags(trace):
    return {r["inputs"]["args"] for r in trace if r.get("action") == "__init__"}


def test_sequential_traces_do_not_leak():
    a = _trace_one("A", 3)
    b = _trace_one("B", 2)
    assert _tags(a) == {"A-0", "A-1", "A-2"}
    assert _tags(b) == {"B-0", "B-1"}


def test_concurrent_tasks_get_isolated_traces():
    async def worker(tag, n, delay):
        PT.start_cache()
        for i in range(n):
            PT(torch.tensor([0.5, 0.5]), extra_info=f"{tag}-{i}")
            await asyncio.sleep(delay)      # force interleaving
        return _tags(PT.end_cache())

    async def main():
        return await asyncio.gather(
            worker("A", 4, 0.001),
            worker("B", 4, 0.001),
            worker("C", 4, 0.001),
        )

    a, b, c = asyncio.run(main())
    assert a == {f"A-{i}" for i in range(4)}, a
    assert b == {f"B-{i}" for i in range(4)}, b
    assert c == {f"C-{i}" for i in range(4)}, c
    assert not (a & b) and not (b & c) and not (a & c)


def test_threads_get_isolated_traces():
    async def main():
        def body(tag):
            PT.start_cache()
            for i in range(3):
                PT(torch.tensor([0.5, 0.5]), extra_info=f"{tag}-{i}")
            return _tags(PT.end_cache())

        return await asyncio.gather(
            asyncio.to_thread(body, "T1"), asyncio.to_thread(body, "T2")
        )

    t1, t2 = asyncio.run(main())
    assert t1 == {f"T1-{i}" for i in range(3)}
    assert t2 == {f"T2-{i}" for i in range(3)}


def test_records_dropped_when_tracing_not_started():
    PT.end_cache()                       # ensure off
    PT(torch.tensor([0.5, 0.5]), extra_info="orphan")
    PT.start_cache()
    PT(torch.tensor([0.5, 0.5]), extra_info="kept")
    assert _tags(PT.end_cache()) == {"kept"}
