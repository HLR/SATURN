"""Option grounding: every physical MCQ option the planner skipped gets grounded."""

import json

import pytest

from saturn.planning.planner_prompt import (
    ensure_option_groundings,
    is_object_option,
    parse_mcq_options,
)


def test_parse_mindcube_format():
    q = "Which object is to the left of the sofa? A. Chair B. Potted plant C. Lamp D. Table"
    assert parse_mcq_options(q) == [("A", "Chair"), ("B", "Potted plant"), ("C", "Lamp"), ("D", "Table")]


def test_parse_mmsi_format():
    q = "Which is closer to camera 1?\nOptions: A: the red mug, B: the laptop, C: the window, D: the door"
    assert parse_mcq_options(q) == [("A", "the red mug"), ("B", "the laptop"),
                                    ("C", "the window"), ("D", "the door")]


def test_lowercase_letters_in_stem_are_not_options():
    assert parse_mcq_options("From image (a) to (b), what moved? (a) cup, (b) lamp") == []


def test_no_options():
    assert parse_mcq_options("How many chairs are there?") == []


@pytest.mark.parametrize("text", [
    "Left", "Directly behind", "Away", "Toward the camera", "Yes", "No", "90 degrees clockwise", "Turn left",
    "Cannot be determined", "north-east", "3", "",
])
def test_non_object_options(text):
    assert not is_object_option(text)


@pytest.mark.parametrize("text", ["Chair", "the red mug", "Potted plant", "water bottle"])
def test_object_options(text):
    assert is_object_option(text)


def test_ensure_adds_only_missing_object_options():
    q = "Which is left of the sofa? A. Chair B. The sofa C. Left D. lamp"
    parsed = {"object_groundings": [{"phrase": "sofa", "cam_id": 0}]}
    added = ensure_option_groundings(q, parsed)
    assert added == ["Chair", "lamp"]
    phrases = [g["phrase"] for g in parsed["object_groundings"]]
    assert phrases == ["sofa", "Chair", "lamp"]
    new = parsed["object_groundings"][1]
    assert new["cam_id"] is None and new["image"] is None
    assert new["role"] == "option" and new["auto_added"] is True and new["unique"] is True


def test_ensure_directional_question_adds_nothing():
    q = "Which way is the chair facing? A. Left B. Right C. Toward the camera D. Away"
    parsed = {"object_groundings": [{"phrase": "chair"}]}
    assert ensure_option_groundings(q, parsed) == []
    assert _phrases(parsed) == ["chair"]


def test_ensure_handles_missing_groundings_key():
    parsed = {}
    assert ensure_option_groundings("A. chair B. lamp", parsed) == ["chair", "lamp"]
    assert len(parsed["object_groundings"]) == 2


# ---- QueryPlanner hook (both clarify paths) ---------------------------------

QUESTION = "Which is left of the sofa? A. Chair B. Lamp C. Left D. sofa"
RAW = json.dumps({"object_groundings": [{"phrase": "sofa", "image": 1, "cam_id": 0}]})


@pytest.fixture
def planner(tmp_path, monkeypatch):
    from saturn.planning.query_planner import QueryPlanner

    qp = QueryPlanner(vl_model=None, cache_path=str(tmp_path / "plan.json"),
                      write_cache=False)
    monkeypatch.setattr(qp, "_invoke_vlm", lambda *a, **k: RAW)
    return qp


def _phrases(parsed):
    return [g["phrase"] for g in parsed["object_groundings"]]


def test_fresh_plan_gets_options(planner, monkeypatch):
    parsed = planner.clarify(QUESTION, [None])
    assert _phrases(parsed) == ["sofa", "Chair", "Lamp"]
    assert parsed["_auto_option_groundings"] == ["Chair", "Lamp"]


def test_cached_plan_gets_options(planner, monkeypatch):
    from saturn.planning.query_planner import _PLANNER_CACHE_VERSION

    planner._cache[planner._cache_key(QUESTION, [None])] = {"status": "ok", "raw": RAW, "_planner_version": _PLANNER_CACHE_VERSION}
    monkeypatch.setattr(planner, "_invoke_vlm", lambda *a, **k: pytest.fail("cache miss"))
    assert _phrases(planner.clarify(QUESTION, [None])) == ["sofa", "Chair", "Lamp"]


@pytest.mark.parametrize("text", [
    "The chair is left of the table", "2 meters", "Figure 1", "image 2", "First image", "Go straight",
    "Southwest-facing", "Diagonally forward and left",
])
def test_mmsi_answer_states_are_not_objects(text):
    assert not is_object_option(text)


@pytest.mark.parametrize("text", ["Bed and red blanket", "Leather loveseat with three seat cushions",
                                  "Grey-green decorative wall", "black phone with keyboard", "Two chairs"])
def test_object_options_still_ground(text):
    assert is_object_option(text)
