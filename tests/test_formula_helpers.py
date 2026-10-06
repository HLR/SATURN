"""The program helper camera(n) (saturn/pipeline/formula.py) and the one-formula program style."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_frame_first_api import _make_camera, _make_object  # noqa: E402

from saturn.pipeline.execute import execute_code  # noqa: E402
from saturn.pipeline.formula import make_formula_helpers  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402
from saturn.soft_logic import ProbabilisticTensor  # noqa: E402

# Camera 0 at the origin looking +z; lamp far left, phone left and near the lamp,
# laptop left but farther from the lamp, tv on the right.
LABELS = ["table lamp", "black phone", "closed laptop", "tv"]
CENTERS = [[-3, 0, 5], [-2.6, 0, 5.2], [-1.5, 0, 7], [3, 0, 5]]


@pytest.fixture
def scene():
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0), _make_camera([0, 0, 10], [0, 0, -1], cam_id=1)]
    objs = [_make_object(i, c) for i, c in enumerate(CENTERS)]
    for o, lab in zip(objs, LABELS):
        o.label = lab
    return Scene(objects=objs, cameras=cams, images=[None, None])


class StubVL:
    """score(): 1.0 on objects whose label appears in the question, rectangular for all but the lamp."""

    def score_multiview(self, question, num_objects=1, type=None, scene=None, images=None, cam_id=None, **kw):
        n = scene.objects_count + len(scene.cameras)
        v = torch.zeros(n)
        q = question.lower()
        for i, o in enumerate(scene.objects):
            if o.label in q or ("rectangular" in q and o.label != "table lamp"):
                v[i] = 1.0
            elif "nothing-matches" in q:
                v[i] = 0.0
        return ProbabilisticTensor(v)

    def query_multiview(self, question, **kw):
        return ""


def _helpers(scene):
    score = lambda x, **kw: StubVL().score_multiview(x, scene=scene)  # noqa: E731
    return make_formula_helpers(score, scene)


def test_camera_is_one_based(scene):
    h = _helpers(scene)
    assert h["camera"](1) == scene.objects_count and h["camera"](2) == scene.objects_count + 1
    with pytest.raises(IndexError):
        h["camera"](0)
    with pytest.raises(IndexError):
        h["camera"](3)


PROGRAM = """
me = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
J = (score("is the object in the red bounding box rectangular?")("x1") & me.first_person.left("x1")
     & scene.closeness("x1", "x2") & score("is the object in the red bounding box a table lamp?")("x2"))
a = J.assign()
assert a["x2"] == 0
OPTIONS = {"A": "tv", "B": "closed laptop", "C": "black phone"}
truth = {k: float((J & score(f"is the object in the red bounding box a {o}?")("x1")).exists().tensor)
         for k, o in OPTIONS.items()}
return max(truth, key=truth.get)
"""


def test_one_formula_program_end_to_end(scene):
    ans, _, err = execute_code(PROGRAM, "which rectangular object on the left is closest to the lamp?",
                               StubVL(), scene, [None, None])
    assert err is None, err
    assert ans == "C"  # the phone: rectangular, left, nearest the lamp


BIND_THEN_NAME = """
lamp = score("is the object in the red bounding box a table lamp?").iota
me = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
J = me.third_person.right("x1", "x2") & lamp("x2")
t = J.assign()["x1"]
OPTIONS = {"A": "black phone", "B": "tv"}
truth = {k: float(score(f"is the object in the red bounding box a {o}?")[t]) for k, o in OPTIONS.items()}
return max(truth, key=truth.get)
"""


def test_bind_then_name_program(scene):
    ans, _, err = execute_code(BIND_THEN_NAME, "what is to the right of the lamp?", StubVL(), scene, [None, None])
    assert err is None, err
    assert ans == "B"  # the tv is furthest right of the lamp; the phone is barely right


GEOMETRY_ONLY = """
me = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
J = me.first_person.right("x1")
t = J.assign()["x1"]
assert t < scene.objects_count, t
OPTIONS = {"A": "table lamp", "B": "tv"}
truth = {k: float(score(f"is the object in the red bounding box a {o}?")[t]) for k, o in OPTIONS.items()}
return max(truth, key=truth.get)
"""


def test_variables_bind_objects_not_cameras():
    # A third camera stands far to the right: the rightmost ENTITY is a camera,
    # but "what is to my right?" must bind the rightmost OBJECT (the tv).
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0), _make_camera([0, 0, 10], [0, 0, -1], cam_id=1),
            _make_camera([12, 0, 5], [0, 0, 1], cam_id=2)]
    objs = [_make_object(i, c) for i, c in enumerate(CENTERS)]
    for o, lab in zip(objs, LABELS):
        o.label = lab
    scene = Scene(objects=objs, cameras=cams, images=[None, None, None])
    ans, _, err = execute_code(GEOMETRY_ONLY, "what is to my right?", StubVL(), scene, [None, None, None])
    assert err is None, err
    assert ans == "B"


def test_binding_domain_is_scoped_to_the_program():
    from saturn.soft_logic.tensor import BINDABLE_ENTITIES, assign_indices
    t = torch.tensor([0.1, 0.2, 0.9])
    assert assign_indices(t, ["x1"])["x1"] == 2  # outside a program every index binds
    tok = BINDABLE_ENTITIES.set(2)
    try:
        assert assign_indices(t, ["x1"])["x1"] == 1
    finally:
        BINDABLE_ENTITIES.reset(tok)


def test_non_entity_axis_gets_no_distinct_binding_diagonal():
    # 6 options == 6 entities, answer at o=3, x1=3: an option axis ("o") is not an
    # entity variable, so it must not get a "distinct binding" diagonal.
    t = torch.zeros(6, 6)
    t[3, 3] = 1.0
    J = ProbabilisticTensor(t, vars=["o", "x1"]) & ProbabilisticTensor(torch.ones(6), vars=["x1"])
    assert J.assign() == {"o": 3, "x1": 3}


def test_camera_ref_is_index_and_camera(scene):
    h = _helpers(scene)
    c = h["camera"](2)
    assert c == scene.objects_count + 1 and np.allclose(c.position, scene.cameras[1].position)


def test_camera_ref_resolves_as_camera_in_engine_calls(scene):
    h = _helpers(scene)
    anchor = scene.frame(position=scene.cameras[0].position, orientation=scene.cameras[0].orientation)
    # a bare int K+1 would be read as an OBJECT index; camera(2) must mean camera 2
    assert np.allclose(anchor.displacement(h["camera"](2)), scene.cameras[1].position - scene.cameras[0].position)
    scene.constraint.face(scene.objects[0], toward=h["camera"](1))
    to_cam = scene.cameras[0].position - scene.objects[0].position
    to_cam[1] = 0
    assert float(scene.objects[0].front_vec @ (to_cam / np.linalg.norm(to_cam))) > 0.99


def test_all_zero_options_fall_back_to_option_order_but_entities_still_guarded():
    import saturn.soft_logic as sl
    from saturn.errors import DegenerateBindingError
    a = sl.ProbabilisticTensor(torch.zeros(2, dtype=torch.float64), vars=["o"]).assign()
    assert a["o"] == 0
    os.environ["SAPY_STRICT_ASSIGN"] = "1"
    try:
        with pytest.raises(DegenerateBindingError):
            sl.ProbabilisticTensor(torch.zeros(3), vars=["x1"]).assign()
    finally:
        os.environ.pop("SAPY_STRICT_ASSIGN", None)
    assert sl.ProbabilisticTensor(torch.zeros(3), vars=["x1"]).assign()["x1"] == 0  # default: argmax, logged


def test_predicate_called_with_index_reads_that_entity(scene):
    # A constant argument binds the slot to that entity (see test_program_robustness.py).
    anchor = scene.frame(position=scene.cameras[0].position, orientation=scene.cameras[0].orientation)
    assert float(anchor.first_person.front(1)) == pytest.approx(float(anchor.first_person.front[1]))
    with pytest.raises(TypeError, match="variable name"):
        anchor.first_person.front(object())
