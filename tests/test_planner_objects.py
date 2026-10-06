"""The planner (planner_prompt_unified.py): its prompt, its "objects" field, and where
pipeline/sample.py takes a question's objects from.

    named      each grounding is detected by the grounder (pre_detect_objects)
    search     SAM3 proposes every object for PROPOSAL_PROMPT while the scene is built
    no_object  nothing is detected
"""
import json
from types import SimpleNamespace

import pytest

from saturn.pipeline import sample
from saturn.pipeline.models import Models
from saturn.planning.planner_prompt import _build_refine_prompt
from saturn.planning.planner_prompt_unified import UNIFIED_PLANNER_PROMPT
from saturn.planning.query_planner import QueryPlanner

GROUNDING = {"phrase": "red chair", "description": "red plastic chair by the door",
             "image": 1, "cam_id": 0, "is_region": False, "unique": True}


# ---- prompt -------------------------------------------------------------------------------------

def test_prompt_has_one_question_slot():
    assert UNIFIED_PLANNER_PROMPT.count("{question}") == 1


def test_refine_prompt_extends_the_unified_prompt():
    out = _build_refine_prompt("Which one?", ["bad pair"])
    assert out.startswith(UNIFIED_PLANNER_PROMPT.replace("{question}", "Which one?"))
    assert "bad pair" in out


# ---- the "objects" field --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["named", "search", "no_object"])
def test_parse_reads_objects(value):
    raw = json.dumps({"objects": value, "program_sketch": "x", "object_groundings": []})
    assert QueryPlanner.parse(raw)["objects"] == value


@pytest.mark.parametrize("value", [None, "", "everything", 3])
def test_parse_missing_or_unknown_objects_is_named(value):
    # a plan without a valid field is treated as "named"
    obj = {"program_sketch": "x", "object_groundings": [GROUNDING]}
    if value is not None:
        obj["objects"] = value
    assert QueryPlanner.parse(json.dumps(obj))["objects"] == "named"


def test_parse_truncated_answer_keeps_objects():
    # "objects" is the first field, so a reply cut off mid-caption still carries it.
    out = QueryPlanner.parse('{"objects": "search", "setup_caption": "Several views of a tab')
    assert out["objects"] == "search"
    assert out["_recovered_from_truncation"] is True


# ---- where pipeline/sample.py takes the objects from ----------------------------------------------

class _Scene:
    objects_count = 0
    num_cameras = 2

    def set_planner_context(self, **_):
        pass


def _pre_execute(monkeypatch, parsed):
    """Run phase A up to (pinned) code generation; record what the scene was built from."""
    seen = {"grounded": False}

    def build_scene(images, models, keywords=None, unique_keywords=None):
        seen["keywords"], seen["unique"] = keywords, unique_keywords
        return _Scene()

    def ground(*_args, **_kwargs):
        seen["grounded"] = True

    monkeypatch.setattr(sample, "pre_detect_objects", ground)
    models = Models()
    models.planner = SimpleNamespace(clarify=lambda question, images: parsed)
    models.pinned_programs = {"q1": "return 'A'"}
    args = SimpleNamespace(vlm_model_name="vlm", dataset="test",
                           use_pose_constraints=False, generate_reports=False)
    item = {"query": "Which one?", "answer": "A", "images": [object()]}
    state = sample.pre_execute(item, "q1", models, args, build_scene_fn=build_scene)
    return seen, state.result


def test_planner_miss_grounds_by_name(monkeypatch):
    # no plan (cache miss and failed call): "named", detection on, nothing proposed at scene build
    seen, result = _pre_execute(monkeypatch, None)
    assert seen["keywords"] is None and seen["grounded"]
    assert result["planner_objects"] == "named" and result["needs_detection"]


def test_named_grounds_each_object(monkeypatch):
    seen, result = _pre_execute(monkeypatch, {"objects": "named", "object_groundings": [GROUNDING]})
    assert seen["keywords"] is None and seen["grounded"]
    assert seen["unique"] == {GROUNDING["description"]}
    assert result["planner_objects"] == "named" and result["needs_detection"]


def test_plans_without_objects_field_are_named(monkeypatch):
    # no "objects" field: each grounding is detected by name
    seen, result = _pre_execute(monkeypatch, {"object_groundings": [GROUNDING]})
    assert seen["keywords"] is None and seen["grounded"]
    assert result["planner_objects"] == "named"


def test_search_proposes_every_object_at_scene_build(monkeypatch):
    parsed = {"objects": "search", "object_groundings": [{**GROUNDING, "phrase": "object", "description": "object"}]}
    seen, result = _pre_execute(monkeypatch, parsed)
    assert seen["keywords"] == [sample.PROPOSAL_PROMPT] and not seen["grounded"]
    assert seen["unique"] is None
    assert result["planner_objects"] == "search" and result["scene_keywords"] == [sample.PROPOSAL_PROMPT]


def test_no_object_detects_nothing(monkeypatch):
    # Groundings the planner wrote anyway are ignored: the decision is the flag.
    seen, result = _pre_execute(monkeypatch, {"objects": "no_object", "object_groundings": [GROUNDING]})
    assert seen["keywords"] is None and not seen["grounded"]
    assert result["scene_keywords"] == [] and not result["needs_detection"]
