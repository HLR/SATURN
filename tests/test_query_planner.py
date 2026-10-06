"""QueryPlanner: answer-option groundings, the clarify() cache, program-mode
parsing, cam_id normalisation, refinement and malformed groundings."""

import json

import pytest
from PIL import Image

from saturn.planning.planner_prompt import ensure_option_groundings, is_object_option
from saturn.planning.query_planner import QueryPlanner


# ---- MMSI answer states are not objects ------------------------------------

@pytest.mark.parametrize("text", [
    "Rear right", "Rear left", "Directly to your left", "Behind you", "On your left", "upper left",
    "Moving to the left", "The same height", "Same position", "Due south", "Right angle",
    "Taken at the same time", "Not moving", "Unable to determine", "L-shape", "Four",
    "Rotate a positive angle around the Y-axis", "In front of you to the left",
])
def test_mmsi_relational_answer_states_are_not_objects(text):
    assert not is_object_option(text)


@pytest.mark.parametrize("text", ["Upper cabinet", "Two chairs", "Rear door", "Window", "Coffee table"])
def test_objects_with_answer_state_words_still_ground(text):
    assert is_object_option(text)


def test_refrigerator_question_adds_no_direction_options():
    q = ("When you took the photo in Figure 1, where was the iron refrigerator located relative to you?\n"
         "Options: A: Front left, B: Rear left, C: Rear right, D: Front right")
    assert ensure_option_groundings(q, {"object_groundings": [{"phrase": "iron refrigerator"}]}) == []


# ---- clarify() cache ---------------------------------------------------------

class _StubVLM:
    def __init__(self, reply, fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def _query(self, imgs, prompt, max_new_tokens=None):
        self.calls += 1
        if self.fail:
            raise ConnectionError("server down")
        return self.reply


def _raw(caption):
    return json.dumps({"setup_caption": caption, "program_sketch": "x",
                       "object_groundings": [{"phrase": "chair", "description": "a chair",
                                              "image": None, "cam_id": None}]})


Q = "In which direction did I move from the first view to the second view?"


def _imgs(color):
    return [Image.new("RGB", (4, 4), color), Image.new("RGB", (4, 4), "white")]


def test_same_question_different_images_is_a_cache_miss(tmp_path):
    path = str(tmp_path / "plan.json")
    vlm = _StubVLM(_raw("street"))
    qp = QueryPlanner(vl_model=vlm, cache_path=path)
    assert qp.clarify(Q, _imgs("red"))["setup_caption"] == "street"
    vlm.reply = _raw("kitchen")
    assert qp.clarify(Q, _imgs("blue"))["setup_caption"] == "kitchen"
    assert vlm.calls == 2
    # Same images again: served from cache (in a new planner too).
    qp2 = QueryPlanner(vl_model=vlm, cache_path=path)
    assert qp2.clarify(Q, _imgs("red"))["setup_caption"] == "street"
    assert vlm.calls == 2


def test_cache_key_carries_the_planner_tags(tmp_path):
    # every key carries "planner_mode=program prompt=unified" before the images fingerprint
    qp = QueryPlanner(vl_model=None, cache_path=str(tmp_path / "plan.json"))
    assert qp._cache_key(Q, _imgs("red")).startswith(f"{Q}\n[planner_mode=program prompt=unified images=")


def test_failed_call_is_retried_on_next_run(tmp_path):
    path = str(tmp_path / "plan.json")
    down = _StubVLM("", fail=True)
    assert QueryPlanner(vl_model=down, cache_path=path).clarify(Q, _imgs("red")) is None
    up = _StubVLM(_raw("street"))
    parsed = QueryPlanner(vl_model=up, cache_path=path).clarify(Q, _imgs("red"))
    assert up.calls == 1 and parsed["setup_caption"] == "street"


# ---- program-mode truncation / trailing prose -------------------------------

PROGRAM_RAW = json.dumps({"program_sketch": "x", "object_groundings": [
    {"phrase": "chair", "description": "a chair", "image": 1, "cam_id": 0, "is_region": False},
    {"phrase": "throw pillows", "description": "pillows", "image": None, "cam_id": None,
     "unique": False},
]})


def test_trailing_prose_after_valid_json_is_ignored():
    p = QueryPlanner.parse(PROGRAM_RAW + "\nNote: I grounded only the chair.")
    assert [g["phrase"] for g in p["object_groundings"]] == ["chair", "throw pillows"]
    assert "_recovered_from_truncation" not in p


def test_truncated_program_mode_reply_is_recovered():
    truncated = PROGRAM_RAW[: PROGRAM_RAW.index('"throw pillows"') + 30]
    p = QueryPlanner.parse(truncated)
    assert p["_recovered_from_truncation"] is True
    g = p["object_groundings"][0]
    assert g["phrase"] == "chair" and g["cam_id"] == 0 and g["image"] == 1


def test_truncation_recovery_keeps_unique_false():
    p = QueryPlanner.parse(PROGRAM_RAW[:-2] + ', {"phrase": "b"')
    pillows = [g for g in p["object_groundings"] if g["phrase"] == "throw pillows"]
    assert pillows and pillows[0]["unique"] is False


# ---- list-form cam_id --------------------------------------------------------

def test_parse_keeps_list_cam_ids():
    raw = json.dumps({"program_sketch": "x", "object_groundings": [
        {"phrase": "the table", "description": "d", "cam_id": [1]},
        {"phrase": "b", "description": "d", "cam_id": [0, 2]},
        {"phrase": "c", "description": "d", "cam_id": "2"},
    ]})
    assert [g["cam_id"] for g in QueryPlanner.parse(raw)["object_groundings"]] == [[1], [0, 2], 2]


def test_parse_revised_keeps_list_cam_ids():
    raw = json.dumps({"revised_groundings": [{"phrase": "a", "description": "b", "cam_id": [2, 3]}]})
    assert QueryPlanner._parse_revised(raw)[0]["cam_id"] == [2, 3]


def test_truncation_recovery_accepts_list_cam_id():
    raw = ('{"object_groundings": [{"phrase": "a", "description": "d", "cam_id": [1], "multi_view": false}, '
           '{"phrase": "b"')
    out = QueryPlanner._extract_groundings_from_truncated(raw)
    assert [(g["phrase"], g["cam_id"]) for g in out] == [("a", [1])]


# ---- refine keeps unique / role ---------------------------------------------

def test_refine_groundings_inherits_unique_and_role():
    reply = json.dumps({"revised_groundings": [
        {"phrase": "throw pillows", "description": "cushions on the sofa", "cam_id": None}]})
    qp = QueryPlanner(vl_model=_StubVLM(reply), cache_path="/nonexistent/plan.json",
                      write_cache=False)
    failed = [{"phrase": "throw pillows", "description": "pillows", "cam_id": None, "unique": False,
               "role": "option", "auto_added": True, "diagnostic": "no_verify"}]
    out = qp.refine_groundings("How many pillows?", [None], failed, failed)
    assert out[0]["unique"] is False
    assert out[0]["role"] == "option" and out[0]["auto_added"] is True


def test_parse_revised_copies_unique():
    raw = json.dumps({"revised_groundings": [
        {"phrase": "chairs", "description": "d", "cam_id": None, "unique": False}]})
    assert QueryPlanner._parse_revised(raw)[0]["unique"] is False


# ---- non-dict groundings -----------------------------------------------------

def test_non_dict_groundings_do_not_crash_clarify(tmp_path):
    vlm = _StubVLM(json.dumps({"object_groundings": ["chair", "table", {"phrase": "sofa", "description": "s"}]}))
    qp = QueryPlanner(vl_model=vlm, cache_path=str(tmp_path / "p.json"))
    parsed = qp.clarify("Is the chair left of the sofa?", [None])
    assert [g["phrase"] for g in parsed["object_groundings"]] == ["sofa"]
