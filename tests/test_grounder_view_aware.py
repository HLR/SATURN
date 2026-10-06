"""View-aware disambiguation tests for ObjectGrounder._verify_and_pick.

Scenario: SAM3 returns several "wall" candidates across different
cameras and the planner specifies cam_id=1. Candidates without a bbox in
the planner's cam_id are excluded outright (score 0.0); the remaining
candidates are judged by the VLM in the cam_id image rather than in each
candidate's own best-detected view.
"""
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image


from saturn.vlm.grounding import ObjectGrounder


def _make_obj(per_view_bboxes, per_view_scores=None):
    obj = types.SimpleNamespace()
    obj.per_view_bboxes = dict(per_view_bboxes)
    obj.per_view_scores = (
        dict(per_view_scores)
        if per_view_scores is not None
        else {v: 1.0 for v in per_view_bboxes}
    )
    obj.views = list(per_view_bboxes.keys())
    obj.label = ""
    return obj


def _make_grounder(objects, vlm_yes_prob_per_obj):
    """Build a minimal ObjectGrounder with a stub scene + VLM.

    ``vlm_yes_prob_per_obj`` is a dict ``{obj_idx_in_image: P(yes)}`` so
    we can simulate "VLM thinks every wall is a wall" — the failure mode.
    """
    scene = types.SimpleNamespace()
    scene.objects = list(objects)
    scene.images = [Image.new("RGB", (480, 480), color=(50, 50, 50)) for _ in range(4)]

    captured_calls = []

    class StubVLM:
        def _score_simple(self, images, prompt):
            # Identify which obj_idx is being scored by inspecting the
            # currently-annotated view (first image whose pixels differ).
            # Simpler: track via call order matched to obj_ids list.
            captured_calls.append(prompt)
            # Return whichever score the test wired up; default to 1.0.
            return float(stub_vlm_state.get("next_score", 1.0))

    stub_vlm_state = {"next_score": 1.0}
    vlm = StubVLM()

    grounder = ObjectGrounder.__new__(ObjectGrounder)
    grounder.scene = scene
    grounder.vlm = vlm
    grounder.verbose = False
    grounder.verify_threshold = 0.5
    grounder.question = "If facing image 2 and I turn right, am I closer to the wall?"
    grounder._obj_verification = {}
    grounder._captured_calls = captured_calls
    grounder._stub_vlm_state = stub_vlm_state
    grounder._scores_by_obj = vlm_yes_prob_per_obj
    return grounder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_view_aware_rejects_candidate_without_bbox_in_cam_id():
    """A wall candidate detected only in cam 3 scores 0 when the planner
    asks for cam_id=1."""
    obj = _make_obj({3: [308, 0, 480, 147]}, per_view_scores={3: 1.0})
    grounder = _make_grounder([obj], vlm_yes_prob_per_obj={0: 1.0})
    score, _fb = grounder._score_candidate(0, "the wall", "wall in image 2", "test", cam_id=1)
    assert score == 0.0, (
        "Candidate has no bbox in cam 1; view-aware verifier must reject it. "
        f"Got {score}."
    )


def test_view_blind_path_without_cam_id():
    """Without a planner cam_id, the candidate is scored view-blind by the VLM."""
    obj = _make_obj({3: [308, 0, 480, 147]}, per_view_scores={3: 1.0})
    grounder = _make_grounder([obj], vlm_yes_prob_per_obj={0: 1.0})
    score, _fb = grounder._score_candidate(0, "the wall", "any wall", "test", cam_id=None)
    assert score == 1.0, f"View-blind path should call VLM. Got {score}."


def test_view_aware_accepts_candidate_present_in_cam_id():
    """Candidate visible in cam_id must use that view as the reference."""
    obj = _make_obj(
        {1: [10, 10, 40, 40], 3: [200, 100, 300, 200]},
        per_view_scores={1: 0.6, 3: 1.0},
    )
    grounder = _make_grounder([obj], vlm_yes_prob_per_obj={0: 1.0})
    score, _fb = grounder._score_candidate(0, "the wall", "wall in image 2", "test", cam_id=1)
    assert score == 1.0, "Candidate present in cam 1; should be VLM-scored."
    last_prompt = grounder._captured_calls[-1]
    assert "the wall" in last_prompt


def test_disambiguation_picks_cam_id_visible_candidate():
    """End-to-end: 4 candidates, planner asks cam_id=1; verifier picks
    only the candidate with a bbox in cam 1, even if others also score 1.0
    in the VLM."""
    objs = [
        _make_obj({3: [10, 10, 40, 40]}),  # obj 0 — wrong (only cam 3)
        _make_obj({0: [10, 10, 40, 40]}),  # obj 1 — wrong (only cam 0)
        _make_obj({1: [50, 50, 80, 80]}),  # obj 2 — RIGHT (cam 1)
        _make_obj({2: [10, 10, 40, 40]}),  # obj 3 — wrong (only cam 2)
    ]
    grounder = _make_grounder(objs, vlm_yes_prob_per_obj={i: 1.0 for i in range(4)})
    best_idx, best_score, _fb = grounder._verify_and_pick(
        [0, 1, 2, 3], "the wall", "wall in image 2", "test", cam_id=1,
    )
    assert best_idx == 2, (
        f"Only obj 2 has a bbox in cam 1; verifier must pick it. "
        f"Got obj {best_idx} (score={best_score})."
    )
    assert best_score == 1.0


def test_disambiguation_view_blind_picks_argmax():
    """Without cam_id: VLM-scored argmax (here all tied at 1.0, so the first
    candidate wins). This view-blind path serves multi_view mode and the
    all-cam recovery retry."""
    objs = [
        _make_obj({3: [10, 10, 40, 40]}),
        _make_obj({0: [10, 10, 40, 40]}),
        _make_obj({1: [50, 50, 80, 80]}),
    ]
    grounder = _make_grounder(objs, vlm_yes_prob_per_obj={i: 1.0 for i in range(3)})
    best_idx, best_score, _fb = grounder._verify_and_pick(
        [0, 1, 2], "the wall", "wall", "test", cam_id=None,
    )
    assert best_idx == 0  # first candidate wins on ties
    assert best_score == 1.0


def test_view_aware_rejects_all_when_no_candidate_in_cam_id():
    """If the planner's cam_id has no candidate at all, every score is 0
    and the verifier returns best_score=0 → triggers downstream all-cam
    recovery (verified by inspecting the score, not by running the
    recovery here)."""
    objs = [
        _make_obj({3: [10, 10, 40, 40]}),
        _make_obj({2: [10, 10, 40, 40]}),
    ]
    grounder = _make_grounder(objs, vlm_yes_prob_per_obj={i: 1.0 for i in range(2)})
    best_idx, best_score, _fb = grounder._verify_and_pick(
        [0, 1], "the wall", "wall in image 2", "test", cam_id=1,
    )
    assert best_score == 0.0, (
        "No candidate is visible in cam 1; all should score 0. "
        f"Got best_score={best_score}."
    )


def test_view_aware_with_no_vlm_returns_one():
    """If VLM scoring is unavailable (no _score_simple), fall back to
    trust-the-detector with score 1.0 — this is the existing safety
    behavior, NOT changed by the view-aware patch (it short-circuits
    before the cam_id check). Ensures the patch doesn't accidentally
    block scoring when VLM is missing."""
    obj = _make_obj({3: [10, 10, 40, 40]})
    grounder = _make_grounder([obj], vlm_yes_prob_per_obj={0: 1.0})
    grounder.vlm = types.SimpleNamespace()  # no _score_simple
    score, _fb = grounder._score_candidate(0, "the wall", "wall in image 2", "test", cam_id=1)
    assert score == 1.0, (
        "VLM-unavailable path should bypass view-awareness and return 1.0. "
        f"Got {score}."
    )
