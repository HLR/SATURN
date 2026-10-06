"""The per-question pipeline: record fields, logs, grounding and execution teardown.

Covers ground.py (zero_verified_region_count, salvage with the planner's
revised grounding), sample.py (pose-constraint log), runner.py (error records
carry the item's fields), execute.py (execute_code restores the scene and stops
the score trace on every exit path), logger calls, and the per-sample HTML
report written by results.py.
"""

import ast
import asyncio
import contextvars
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from saturn.pipeline import runner as runner_mod
from saturn.pipeline import sample as sample_mod
from saturn.pipeline.execute import execute_code
from saturn.pipeline.ground import pre_detect_objects
from saturn.pipeline.results import generate_report
from saturn.pipeline.state import _PreExecState
from saturn.soft_logic import ProbabilisticTensor
from saturn.vlm import grounding as grounding_mod
from saturn.vlm.grounding import GroundingResult


# --- ground.py: zero_verified_region_count counts regions only ---------------

def test_zero_verified_region_count_counts_only_regions_verified_at_zero(monkeypatch):
    results = [
        # objects: verified with zero score, or verified without attempts (reused)
        GroundingResult(phrase="cup", description="cup", cam_id=0, is_region=False,
                        obj_indices=[0], attempts=1, verified=True, verify_score=0.0),
        GroundingResult(phrase="mug", description="mug", cam_id=0, is_region=False,
                        obj_indices=[1], attempts=0, verified=True, verify_score=0.9),
        # the one suspect region: verified, zero score
        GroundingResult(phrase="kitchen", description="stove, sink", cam_id=0, is_region=True,
                        obj_indices=[2], attempts=2, verified=True, verify_score=0.0,
                        region_members=[0, 1]),
        # regions that are fine or unverified
        GroundingResult(phrase="desk area", description="desk, chair", cam_id=0, is_region=True,
                        obj_indices=[3], attempts=2, verified=True, verify_score=0.7,
                        region_members=[3]),
        GroundingResult(phrase="corner", description="lamp", cam_id=0, is_region=True,
                        attempts=1, verified=False, verify_score=0.0),
    ]

    class FakeGrounder:
        def __init__(self, **kwargs):
            pass

        def ground_all(self, groundings, item_id=None):
            return results

    monkeypatch.setattr(grounding_mod, "ObjectGrounder", FakeGrounder)
    scene = SimpleNamespace(_vlm=object(), objects_count=4)
    parsed = {"object_groundings": [{"phrase": r.phrase, "description": r.description}
                                    for r in results]}
    record = {}
    pre_detect_objects(scene, parsed, "s1", record)
    assert record["zero_verified_region_count"] == 1
    assert [g["is_region"] for g in record["grounding_results"]] == [False, False, True, True, True]


# --- sample.py: the pose-constraint log counts applied records ---------------

def test_pose_constraint_log_counts_only_applied_records(caplog):
    calls = []

    class Constraint:
        def rotation(self, a, b, yaw, axis="up"):
            calls.append(("rotation", a, b, yaw))

        def same_position(self, *cams):
            calls.append(("same_position", cams))

    records = [
        {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90},
        {"type": "rotation", "from_cam": 0, "to_cam": 7, "yaw": 90},  # no camera 7
        {"type": "same_position"},                                    # no "cams"
    ]
    scene = SimpleNamespace(cameras=["c0", "c1"], constraint=Constraint(),
                            images=[None, None], num_cameras=2)
    models = SimpleNamespace(vl_model=SimpleNamespace(extract_pose_constraints=lambda **kw: records))
    result = {}
    with caplog.at_level(logging.INFO, logger="saturn.pipeline.sample"):
        sample_mod._apply_pose_constraints(
            scene, "Image 2 is rotated 90 degrees clockwise from image 1.", models, "s1", result)

    assert calls == [("rotation", "c0", "c1", 90.0)]
    summary = [r.getMessage() for r in caplog.records if "pose-constraints: applied" in r.getMessage()]
    assert len(summary) == 1
    assert "applied 1 record(s), skipped 2" in summary[0]


# --- runner.py: timeout / main-loop error records carry the item's fields ---

ITEM = {
    "id": "q7", "query": "Which one?", "answer": "B", "question_type": "among",
    "subset": "mc_among", "image_file_name": ["a.png", "b.png"], "images": [],
}


def _expect_item_fields(record):
    assert record["id"] == "q7"
    assert record["question_type"] == "among"
    assert record["subset"] == "mc_among"
    assert record["query"] == "Which one?"
    assert record["ground_truth_answer"] == "B"
    assert record["image_file_name"] == ["a.png", "b.png"]


def _run_one(args, item=ITEM):
    async def go():
        runner = runner_mod._SampleRunner(
            args=args, models=None, build_scene_fn=None,
            pre_exec_sem=asyncio.Semaphore(1), exec_sem=asyncio.Semaphore(1),
            processed_ids=set(),
        )
        return await runner_mod._run_sample(runner, 0, item)
    return asyncio.run(go())


def test_main_loop_error_record_carries_item_fields(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("planner down")

    monkeypatch.setattr(runner_mod, "pre_execute", boom)
    _, _, _, record = _run_one(SimpleNamespace(exec_timeout=0))
    assert record["error"].startswith("MainLoop Error: RuntimeError")
    assert record["correct_final_answer"] is False
    _expect_item_fields(record)


def test_timeout_record_carries_item_fields(monkeypatch):
    state = _PreExecState(early_return=False, result={"objects_count": 3})
    monkeypatch.setattr(runner_mod, "pre_execute", lambda *a, **k: state)
    monkeypatch.setattr(runner_mod, "execute", lambda *a, **k: time.sleep(0.3))
    _, _, _, record = _run_one(SimpleNamespace(exec_timeout=0.05))
    assert record["error_type"] == "TimeoutError"
    assert record["objects_count"] == 3
    _expect_item_fields(record)


# --- execute.py: execute_code restores the scene and stops the trace --------

class _StubVL:
    def score_multiview(self, question, **kwargs):
        raise AssertionError("not called")

    def query_multiview(self, question, **kwargs):
        return ""


class _BareScene:
    """A scene with no detect / ground of its own."""

    def __init__(self):
        self.objects = []
        self.cameras = []
        self.images = []

    @property
    def objects_count(self):
        return len(self.objects)


def _trace_after(program, monkeypatch, stub):
    monkeypatch.setenv("SAPY_TRACE_SCORES", "1")
    if stub is not None:
        monkeypatch.setenv("SAPY_SCORE_STUB", stub)
    else:
        monkeypatch.delenv("SAPY_SCORE_STUB", raising=False)

    def run():
        _, _, err = execute_code(program, "q", _StubVL(), _BareScene(), [])
        return err, ProbabilisticTensor._cache_var.get()

    return contextvars.copy_context().run(run)


@pytest.mark.parametrize("program,stub,error", [
    ("return 'A'", "not-a-number", "ValueError"),   # invalid SAPY_SCORE_STUB
    ("raise KeyError('x')\nreturn 'A'", None, "KeyError"),  # the program fails
])
def test_score_trace_is_stopped_when_execution_fails(monkeypatch, program, stub, error):
    err, trace = _trace_after(program, monkeypatch, stub)
    assert err is not None and error in err
    assert trace is None


def test_score_trace_is_returned_when_execution_succeeds(monkeypatch):
    monkeypatch.setenv("SAPY_TRACE_SCORES", "1")
    monkeypatch.delenv("SAPY_SCORE_STUB", raising=False)

    def run():
        result = execute_code("return 'A'", "q", _StubVL(), _BareScene(), [])
        return result, ProbabilisticTensor._cache_var.get()

    (ans, cache, err), trace = contextvars.copy_context().run(run)
    assert (ans, cache, err) == ("A", [], None)
    assert trace is None


@pytest.mark.parametrize("program", [
    "return 'A'",
    "raise KeyError('x')\nreturn 'A'",
])
def test_stand_ins_are_removed_from_a_scene_without_detect(program):
    scene = _BareScene()
    execute_code(program, "q", _StubVL(), scene, [])
    assert not hasattr(scene, "detect") and not hasattr(scene, "ground")


def test_scene_detect_and_ground_are_restored():
    scene = _BareScene()
    detect, ground = object(), object()
    scene.detect, scene.ground = detect, ground
    _, _, err = execute_code("assert scene.detect('x') == []\nreturn 'A'", "q", _StubVL(), scene, [])
    assert err is None
    assert scene.detect is detect and scene.ground is ground


# --- ground.py: salvage uses the planner's revised grounding --------------

def test_salvage_uses_revised_grounding(monkeypatch):

    calls = []

    class FakeGrounder:
        def __init__(self, **kw):
            pass

        def ground_all(self, groundings, item_id=None):
            return [GroundingResult(phrase=g["phrase"], description=g["description"],
                                    cam_id=g["cam_id"], is_region=False,
                                    final_description=g["description"],
                                    last_diagnostic="no_candidates") for g in groundings]

        def salvage(self, phrase, description, cam_id, item_id):
            calls.append((description, cam_id))
            return None

    class FakePlanner:
        def refine_groundings(self, q, imgs, prior_groundings, failed_groundings):
            return [{"phrase": "the mug", "description": "white coffee mug on the shelf",
                     "cam_id": 2}]

    monkeypatch.setattr(grounding_mod, "ObjectGrounder", FakeGrounder)
    scene = SimpleNamespace(_vlm=object(), objects_count=0)
    parsed = {"object_groundings": [{"phrase": "the mug", "description": "cup", "cam_id": 0}]}
    pre_detect_objects(scene, parsed, "q0", {}, question="?", planner=FakePlanner(),
                       images=[1, 2, 3])
    assert calls == [("white coffee mug on the shelf", 2)]


# --- logger calls pass no print-only kwargs ---------------------------------

_ROOT = Path(__file__).resolve().parents[1] / "saturn"


@pytest.mark.parametrize("sub", ["pipeline", "cli", "datasets"])
def test_no_flush_kwarg_on_logger_calls(sub):
    bad = []
    for path in (_ROOT / sub).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "log"
                    and any(k.arg == "flush" for k in node.keywords)):
                bad.append(f"{path}:{node.lineno}")
    assert not bad, bad


# --- results.py: per-sample reports have no dangling Prev/Next links -------

def test_sample_report_has_no_placeholder_nav_links(tmp_path):
    d = str(tmp_path)
    sample = {"id": "s7", "question": "q?", "final_answer_text": "A", "gt_answer": "A",
              "correct_final_answer": True}
    generate_report(sample, 3, 10, os.path.join(d, "r.json"), d, None, d, "s7")
    html = (tmp_path / "sample_s7.html").read_text()
    links = [l for l in re.findall(r"href=['\"]([^'\"]+\.html)['\"]", html)]
    assert "index.html" in links
    assert not [l for l in links if l.endswith(("_prev.html", "_next.html"))]
