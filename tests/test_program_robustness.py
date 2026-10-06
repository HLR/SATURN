"""Robustness of generated programs to common code shapes.

Each block covers one shape that a code model naturally writes against the
engine API: the program runs with the shape's obvious meaning, or fails with an
error the retry loop can act on. Programs below are generic API shapes, not
benchmark items.
"""

import copy
import os
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_frame_first_api import _make_camera, _make_object  # noqa: E402

from saturn.pipeline.execute import (  # noqa: E402
    NO_ANSWER_ERROR, ensure_entry_called, execute_code, execute_with_retry,
)
from saturn.pipeline.formula import make_formula_helpers  # noqa: E402
from saturn.predicates.frame import CompassNotSetError  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402
from saturn.scene.types import MissingViewError, PerViewDict  # noqa: E402
from saturn.soft_logic import PredicateArray, ProbabilisticTensor, and_op, or_op  # noqa: E402
from saturn.soft_logic.predicate_array import Degree  # noqa: E402

LABELS = ["lamp", "chair", "table", "plant"]
CENTERS = [[-3, 0, 5], [2, 0, 4], [0, 0, 8], [3, 1, 9]]


@pytest.fixture
def scene():
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0), _make_camera([0, 0, 10], [0, 0, -1], cam_id=1)]
    objs = [_make_object(i, c, front=(0, 0, 1) if i == 1 else (0, 0, -1)) for i, c in enumerate(CENTERS)]
    for o, lab in zip(objs, LABELS):
        o.label = lab
    objs[0].per_view_centers = {0: np.array(CENTERS[0], float)}               # seen in image 1 only
    objs[2].per_view_centers = {0: np.array(CENTERS[2], float), 1: np.array([0, 0, 7.0])}
    objs[3].per_view_centers = {1: np.array(CENTERS[3], float)}               # seen in image 2 only
    return Scene(objects=objs, cameras=cams, images=[None, None])


class StubVL:
    def score_multiview(self, question, num_objects=1, type=None, scene=None, images=None, cam_id=None, **kw):
        n = scene.objects_count + len(scene.cameras)
        v = torch.zeros(n, dtype=torch.float64)
        for i, o in enumerate(scene.objects):
            if o.label in question.lower():
                v[i] = 1.0
        return ProbabilisticTensor(v)

    def query_multiview(self, question, **kw):
        return ""


def _anchor(scene, cam=0):
    c = scene.cameras[cam]
    return scene.frame(position=c.position, orientation=c.orientation)


def _cam(scene, n):
    return make_formula_helpers(lambda q, **kw: None, scene)["camera"](n)


# ---------------------------------------------------------------------------
# 1. pred[i] is a truth degree: & | ~ work on it
# ---------------------------------------------------------------------------

def test_indexed_scores_compose_with_connectives(scene):
    a = _anchor(scene)
    left, below = a.first_person.left, a.first_person.below
    d = left[0] & below[0]
    assert isinstance(d, Degree) and isinstance(d, np.float64)
    assert float(d) == pytest.approx(min(float(left[0]), float(below[0])))
    assert float(left[0] | below[0]) == pytest.approx(max(float(left[0]), float(below[0])))
    assert float(~left[0]) == pytest.approx(1 - float(left[0]))
    assert float(left[0] & True) == pytest.approx(float(left[0]))
    assert float(np.float64(0.2) & left[0]) == pytest.approx(min(0.2, float(left[0])))
    assert left[0].exists() == left[0]
    # numeric use is unchanged
    assert type(left[0] + 1) is np.float64 and float(left[0]) * 2 == pytest.approx(2 * left[0])


def test_degree_lifts_into_formulas(scene):
    a = _anchor(scene)
    J = a.first_person.left("x1")
    joint = a.first_person.front[_cam(scene, 2)] & J
    assert isinstance(joint, ProbabilisticTensor) and list(joint.vars) == ["x1"]
    expected = np.minimum(float(a.first_person.front[_cam(scene, 2)]), np.asarray(a.first_person.left))
    assert np.allclose(joint.tensor.cpu().numpy(), expected)


# ---------------------------------------------------------------------------
# 2. an entity in place of a variable binds that slot
# ---------------------------------------------------------------------------

def test_first_person_called_with_camera_reads_it(scene):
    a = _anchor(scene)
    c2 = _cam(scene, 2)
    t = a.first_person.front(c2)
    assert isinstance(t, ProbabilisticTensor) and list(t.vars) == []
    assert float(t) == pytest.approx(float(a.first_person.front[c2]))
    assert float(t.exists()) == pytest.approx(float(t))
    both = a.first_person.front(c2) & a.first_person.right(c2)
    assert float(both.exists()) == pytest.approx(min(float(a.first_person.front[c2]), float(a.first_person.right[c2])))


def test_pairwise_predicate_with_one_constant(scene):
    a = _anchor(scene)
    rel = a.third_person.left
    c2 = _cam(scene, 2)
    part = rel("x1", c2)
    assert list(part.vars) == ["x1"]
    assert torch.allclose(part.tensor, rel.tensor[:, int(c2)])
    assert float(rel(0, 2)) == pytest.approx(float(rel.tensor[0, 2]))


def test_bad_predicate_arguments_are_actionable(scene):
    a = _anchor(scene)
    with pytest.raises(TypeError, match="relates 2 entities"):
        a.third_person.left("x1", 1, 2)
    with pytest.raises(TypeError, match="3D point is not an entity"):
        a.third_person.left("x1", np.array([0.0, 1.0, 2.0]))
    with pytest.raises(TypeError, match="ONE entity"):
        a.first_person.left("x1", "x2")
    with pytest.raises(IndexError, match="out of range"):
        a.third_person.left("x1", 99)


# ---------------------------------------------------------------------------
# 3. numbers / bools are constant truths inside & and |
# ---------------------------------------------------------------------------

def test_formulas_accept_constant_truths():
    t = ProbabilisticTensor(torch.tensor([0.2, 0.9], dtype=torch.float64))("x1")
    assert np.allclose((t & True).tensor.cpu().numpy(), [0.2, 0.9])
    assert np.allclose((False & t).tensor.cpu().numpy(), [0.0, 0.0])
    assert np.allclose((t | 0.5).tensor.cpu().numpy(), [0.5, 0.9])
    assert np.allclose((0.5 | t).tensor.cpu().numpy(), [0.5, 0.9])
    assert np.allclose(and_op(t, 0.3).tensor.cpu().numpy(), [0.2, 0.3])
    assert np.allclose(or_op(t, np.float64(0.95)).tensor.cpu().numpy(), [0.95, 0.95])


# ---------------------------------------------------------------------------
# 4. first_person reads a 3D point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["front", "left", "back-right", "above", "below", "up", "down"])
def test_first_person_at_a_point_equals_an_object_there(scene, label):
    a = _anchor(scene)
    p = np.array(CENTERS[2], float)            # object 2 sits exactly at p
    arr = a.first_person(label)
    assert float(arr[p]) == pytest.approx(float(arr[2]))
    assert float(arr(p)) == pytest.approx(float(arr[2]))
    assert float(arr[list(p)]) == pytest.approx(float(arr[2]))


def test_point_reading_is_only_for_first_person(scene):
    a = _anchor(scene)
    with pytest.raises(TypeError, match="first_person"):
        PredicateArray(np.asarray(a.first_person.front)[:])[np.array([0.0, 0.0, 1.0])]
    # an integer index array is still numpy fancy indexing
    assert np.asarray(a.first_person.front)[[0, 1, 2]].shape == (3,)
    assert a.first_person.front[[0, 1, 2]].shape == (3,)


def test_up_down_are_above_below(scene):
    a = _anchor(scene)
    assert np.allclose(a.first_person.up, a.first_person.above)
    assert np.allclose(a.first_person("down"), a.first_person.below)


# ---------------------------------------------------------------------------
# 5. compass words in third_person and facing
# ---------------------------------------------------------------------------

def test_compass_needs_a_north(scene):
    a = _anchor(scene)
    with pytest.raises(CompassNotSetError, match="set_cardinal_vector"):
        a.third_person.north
    with pytest.raises(AttributeError):
        a.facing.northwest
    with pytest.raises(ValueError):
        a.facing("south")


def test_third_person_compass_matches_first_person_at_the_reference(scene):
    scene.set_cardinal_vector(np.array([0.0, 0.0, 1.0]))          # north = +Z, east = +X
    a = _anchor(scene)
    for label in ("north", "north-west", "southeast", "west"):
        rel = a.third_person(label)
        K = scene.objects_count
        for j in range(K):
            at_j = scene.frame(position=scene.objects[j].position, orientation=scene.orientation_from_forward([0, 0, 1]))
            fp = np.asarray(at_j.first_person(label))
            for i in range(K):
                if i != j:
                    assert float(rel.tensor[i, j]) == pytest.approx(fp[i], abs=1e-6)
            assert float(rel.tensor[j, j]) == 0.0
    # table (0,0,8) is north of chair (2,0,4) and west-ish of it
    assert float(a.third_person.north("x1", "x2").tensor[2, 1]) > 0.8
    assert float(a.third_person.northwest[2, 1]) > float(a.third_person.northeast[2, 1])
    # compass does not depend on the observer
    b = _anchor(scene, cam=1)
    assert torch.allclose(a.third_person.east.tensor, b.third_person.east.tensor)


def test_facing_compass(scene):
    scene.set_cardinal_vector(np.array([0.0, 0.0, 1.0]))
    a = _anchor(scene)
    north, south = a.facing.north, a.facing("south")
    assert float(north.tensor[1]) > 0.9 and float(south.tensor[1]) < 0.1     # chair faces +Z
    assert float(north.tensor[0]) < 0.1 and float(south.tensor[0]) > 0.9     # lamp faces -Z
    # same scores from a differently oriented anchor
    assert torch.allclose(north.tensor, _anchor(scene, cam=1).facing.north.tensor)


# ---------------------------------------------------------------------------
# 6. per-view data of an object missing in an image: actionable KeyError
# ---------------------------------------------------------------------------

def test_missing_view_is_an_actionable_key_error(scene):
    pvc = scene.objects[0].per_view_centers
    assert isinstance(pvc, PerViewDict) and isinstance(pvc, dict)
    with pytest.raises(KeyError) as ei:
        pvc[1]
    e = ei.value
    assert isinstance(e, MissingViewError)
    msg = str(e)
    assert "not detected in image 2" in msg and "detected in images [1]" in msg
    assert pvc.get(1) is None and 0 in pvc and 1 not in pvc


def test_per_view_dicts_survive_copy_pickle_and_reassignment(scene):
    o = scene.objects[2]
    for clone in (copy.deepcopy(o), pickle.loads(pickle.dumps(o))):
        assert isinstance(clone.per_view_centers, PerViewDict)
        assert set(clone.per_view_centers) == {0, 1}
        with pytest.raises(MissingViewError, match="per_view_centers"):
            clone.per_view_centers[5]
    o.per_view_fronts = {0: np.array([0, 0, 1.0])}      # plain dict, as constraints.py writes it
    assert isinstance(o.per_view_fronts, PerViewDict)
    assert o.to_dict()["per_view_centers"].keys() == {"0", "1"}


# ---------------------------------------------------------------------------
# 7. vector inputs, compass helpers, scene attributes
# ---------------------------------------------------------------------------

def test_vector_arguments_are_checked(scene):
    scene.set_cardinal_vector([0, 0, 1])
    with pytest.raises(TypeError, match="call it"):
        scene.cardinalize(scene.cardinal_vector, known="north")
    with pytest.raises(TypeError, match="score_cardinals"):
        scene.cardinalize(np.array([1.0, 0, 0]))
    with pytest.raises(ValueError, match="horizontal"):
        scene.cardinalize(np.array([0.0, 1.0, 0.0]), known="north")
    assert np.allclose(scene.cardinalize([1, 0, 0], known="east"), [0, 0, 1])
    assert np.allclose(scene.cardinalize([1, 0, 0], known="north-east"),
                       scene.cardinalize([1, 0, 0], known="northeast"))
    with pytest.raises(TypeError, match="3-vector"):
        scene.orientation_from_forward("north")


def test_scene_attribute_errors_suggest_names(scene):
    with pytest.raises(AttributeError, match="Did you mean: scene.cardinal_vector"):
        scene.cardinal_north_vectr
    with pytest.raises(AttributeError, match="does not exist"):
        scene.no_such_api
    assert not hasattr(scene, "_private_probe")


def test_scene_up_vector_and_numeric_vector(scene):
    assert np.allclose(scene.up, [0, 1, 0])
    assert np.allclose(scene.vector(0, -1, 0), [0, -1, 0])
    assert np.allclose(scene.vector(0, 1), (scene.objects[1].position - scene.objects[0].position)
                       / np.linalg.norm(scene.objects[1].position - scene.objects[0].position))
    with pytest.raises(TypeError, match="three numbers"):
        scene.vector(0, 1, "z")


def test_axis_convention_rejects_non_strings(scene):
    with pytest.raises(TypeError, match="right='\\+X'"):
        scene.set_axis_convention(right=-1, up=2, forward=0)


def test_frame_has_entity_axis_names(scene):
    a = _anchor(scene, cam=1)
    assert np.allclose(a.front_vec, a.frame_front) and np.allclose(a.up_vec, a.frame_up)
    assert np.allclose(a.right_vec, a.frame_right)


# ---------------------------------------------------------------------------
# 8. executor: uncalled entry defs, None answers, errors in the program's own terms
# ---------------------------------------------------------------------------

NESTED_ENTRY = '''def logic_executor(query, score_fn, query_fn, scene, images, history):
    a = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
    labels = {"A": "left", "B": "right"}
    return max(labels, key=lambda k: float(a.first_person(labels[k])[camera(2)]))
'''

NESTED_NO_ARGS = '''def answer():
    J = score("is the object in the red bounding box a table?").iota("x1")
    return "A" if float(J.exists()) > 0.5 else "B"
'''

HELPER_WITHOUT_RETURN = '''def helper():
    return "A"
x = 1
'''


def test_uncalled_entry_function_is_called(scene):
    ans, _, err = execute_code(NESTED_ENTRY, "q", StubVL(), scene, [None, None])
    assert err is None and ans in ("A", "B")
    ans, _, err = execute_code(NESTED_NO_ARGS, "q", StubVL(), scene, [None, None])
    assert err is None and ans == "A"


def test_entry_call_rules():
    assert ensure_entry_called("x = 1\nreturn x") == "x = 1\nreturn x"
    used = "def f():\n    return 1\nv = f()\nreturn v"
    assert ensure_entry_called(used) == used
    needs_arg = "def f(a, b):\n    return a\n"
    assert ensure_entry_called(needs_arg) == needs_arg
    assert ensure_entry_called(HELPER_WITHOUT_RETURN).rstrip().endswith("return helper()")
    # a return nested only inside a def does not count as the program's return
    assert "return answer()" in ensure_entry_called(NESTED_NO_ARGS)


def test_none_answer_is_an_error(scene):
    ans, _, err = execute_code("x = 1\n", "q", StubVL(), scene, [None, None])
    assert ans is None and err == NO_ANSWER_ERROR


def test_errors_speak_in_program_lines_and_hide_the_scaffold(scene):
    prog = "a = 1\nb = 2\nv = scene.objects[0].position.normalized()\nreturn 'A'"
    _, _, err = execute_code(prog, "q", StubVL(), scene, [None, None])
    assert err.startswith("Execution Error: AttributeError")
    assert "Failing line 3: v = scene.objects[0].position.normalized()" in err
    assert "np.linalg.norm" in err                       # repair hint
    assert "def logic_executor" not in err and "score_fn(" not in err


def test_missing_view_error_names_the_object_and_the_image(scene):
    prog = "d = scene.objects[0].per_view_centers[1] - scene.objects[0].per_view_centers[0]\nreturn 'A'"
    _, _, err = execute_code(prog, "q", StubVL(), scene, [None, None])
    assert "MissingViewError" in err and "not detected in image 2" in err
    assert "scene.objects[0] ('lamp')" in err
    assert "Objects detected in image 2: 2 ('table'), 3 ('plant')" in err


def test_logic_ops_on_python_floats_get_a_hint(scene):
    prog = "a = 0.3\nb = 0.7\nreturn 'A' if (a & b) else 'B'"
    _, _, err = execute_code(prog, "q", StubVL(), scene, [None, None])
    assert "min(a, b)" in err


class _RetryGen:
    def __init__(self, fixed):
        self.fixed, self.calls = fixed, []

    def retry_generate_code(self, **kw):
        self.calls.append(kw)
        return self.fixed, None


def test_none_answer_is_retried_and_the_retry_sees_no_scaffold(scene):
    gen = _RetryGen("return 'B'")
    snippet, ans, _, err, n = execute_with_retry("x = 1\n", "q", StubVL(), scene, [None, None],
                                                 gen, {}, "t", max_retries=3)
    assert ans == "B" and err is None and n == 1
    assert "NoAnswer" in gen.calls[0]["error_message"]
    assert "def logic_executor" not in gen.calls[0]["error_message"]


# ---------------------------------------------------------------------------
# 9. whole programs combining these shapes, end to end (generic wording)
# ---------------------------------------------------------------------------

SHAPES = {
    "and_on_indexed_scores": '''
t = score("is the object in the red bounding box a table?").iota("x1").assign()["x1"]
anchor = scene.frame(position=camera(2).position, orientation=camera(2).orientation)
scores = {"A": float(anchor.first_person.left[t] & anchor.first_person.below[t]),
          "B": float(anchor.first_person.right[t] & anchor.first_person.below[t])}
return max(scores, key=lambda k: scores[k])''',
    "predicate_called_with_camera": '''
a1 = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
J = {"A": a1.first_person.front_right(camera(2)), "B": a1.first_person.front_left(camera(2))}
return max(J, key=lambda k: float(J[k].exists()))''',
    "point_as_index": '''
p1 = scene.objects[2].per_view_centers[0]
p2 = scene.objects[2].per_view_centers[1]
anchor = scene.frame(position=p1, orientation=camera(1).orientation)
labels = {"A": "front", "B": "back", "C": "left", "D": "right"}
return max(labels, key=lambda k: float(anchor.first_person(labels[k])[p2]))''',
    "third_person_compass": '''
scene.set_cardinal_vector(scene.cardinalize(camera(1).front_vec, known="north"))
anchor = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
J = score("is the object in the red bounding box a table?").iota("x1") & score("is the object in the red bounding box a chair?").iota("x2")
labels = {"A": "northwest", "B": "northeast", "C": "southwest", "D": "southeast"}
return max(labels, key=lambda k: float((J & anchor.third_person(labels[k])("x1", "x2")).exists()))''',
    "facing_compass": '''
scene.set_cardinal_vector([0, 0, 1])
c = score("is the object in the red bounding box a chair?").iota("x1").assign()["x1"]
anchor = scene.frame(position=scene.objects[c].position, orientation=scene.objects[c].orientation)
labels = {"A": "north", "B": "south", "C": "east", "D": "west"}
return max(labels, key=lambda k: float(anchor.facing(labels[k])[c]))''',
    "numeric_vector_and_claim_constant": '''
anchor = scene.frame(position=scene.vector(0, 10, 0), orientation=scene.orientation_from_forward(scene.vector(0, 0, 1)))
yaw, _ = anchor.rotation_to(scene.frame(position=camera(2).position, orientation=camera(2).orientation))
J = score("is the object in the red bounding box a table?").iota("x1")
return max({"A": 1, "B": 2}, key=lambda k: float((J & (yaw > 0 if k == "A" else yaw <= 0)).exists()))''',
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_combined_program_shapes_run(scene, name):
    ans, _, err = execute_code(SHAPES[name].strip(), "q", StubVL(), scene, [None, None])
    assert err is None, err
    assert ans in ("A", "B", "C", "D")


def test_shapes_give_the_geometric_answer(scene):
    ans, _, _ = execute_code(SHAPES["third_person_compass"].strip(), "q", StubVL(), scene, [None, None])
    assert ans == "A"            # table (0,0,8) is north-west of chair (2,0,4), north = camera 1 forward (+Z)
    ans, _, _ = execute_code(SHAPES["facing_compass"].strip(), "q", StubVL(), scene, [None, None])
    assert ans == "A"            # the chair faces +Z = north
    ans, _, _ = execute_code(SHAPES["point_as_index"].strip(), "q", StubVL(), scene, [None, None])
    assert ans == "B"            # the table moved from z=8 to z=7: toward camera 1, i.e. back
