"""scene.frame for programs: at a point (position= or at=, with orientation=, or without it for compass words
and .look_at), or at a named object's description (the frame at description.assign(), the index program's frame)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_frame_first_api import _make_camera, _make_object  # noqa: E402

from saturn.pipeline.formula import make_formula_helpers  # noqa: E402
from saturn.scene.scene import Scene, UnfacedFrame  # noqa: E402
from saturn.soft_logic import ProbabilisticTensor  # noqa: E402
from saturn.soft_logic.tensor import BINDABLE_ENTITIES  # noqa: E402


def _scene(seed=0, n_obj=7, n_cam=3):
    rng = np.random.RandomState(seed)
    objs = [_make_object(i, rng.randn(3) * np.array([3.0, 1.0, 3.0]), front=rng.randn(3) * np.array([1.0, 0.3, 1.0]))
            for i in range(n_obj)]
    cams = [_make_camera(rng.randn(3) * np.array([4.0, 0.5, 4.0]), rng.randn(3) * np.array([1.0, 0.2, 1.0]), cam_id=c)
            for c in range(n_cam)]
    return Scene(objects=objs, cameras=cams, images=[None] * n_cam)


def _desc(scene, values, var="x2"):
    t = torch.zeros(scene._num_entities(), dtype=torch.float64)
    t[: len(values)] = torch.as_tensor(values, dtype=torch.float64)
    return ProbabilisticTensor(t, vars=[var]).iota(var)


def _np(pt):
    return pt.tensor.detach().cpu().numpy()


def _index_frame(scene, desc, orientation=None):
    token = BINDABLE_ENTITIES.set(len(scene.objects))
    try:
        c = desc.assign()[desc.vars[0]]
    finally:
        BINDABLE_ENTITIES.reset(token)
    o = scene.objects[c]
    return scene.frame(position=o.position, orientation=o.orientation if orientation is None else orientation)


@pytest.mark.parametrize("values", [
    [0.1, 0.3, 0.2, 0.95, 0.4, 0.0, 0.5],              # one clear match
    [0.0004, 0.0009, 0.0002, 0.0001, 0.0, 0.0, 0.0],   # every score tiny: .iota flattens them
    [0.9999, 0.9999, 0.2, 0.1, 0.0, 0.0, 0.0],         # an exact tie at the top
    [0.0] * 7,                                         # the description matched nothing
])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frame_at_a_description_is_the_index_programs_frame(values, seed):
    scene = _scene(seed)
    camera = make_formula_helpers(None, scene)["camera"]
    d = _desc(scene, values)
    for orientation in (None, camera(2).orientation):
        a, b = scene.frame(at=d, orientation=orientation), _index_frame(scene, d, orientation)
        np.testing.assert_allclose(_np(a.first_person.left("x1")), _np(b.first_person.left("x1")), atol=1e-12)
        np.testing.assert_allclose(_np(a.rotate(yaw=90).third_person.behind("x1", "x3")),
                                   _np(b.rotate(yaw=90).third_person.behind("x1", "x3")), atol=1e-12)
        assert float(a.look_at(camera(1)).first_person.front[camera(3)]) == pytest.approx(
            float(b.look_at(camera(1)).first_person.front[camera(3)]), abs=1e-12)


def test_a_description_over_two_variables_is_an_error():
    scene = _scene()
    n = scene._num_entities()
    with pytest.raises(ValueError, match="one object"):
        scene.frame(at=ProbabilisticTensor(torch.zeros(n, n, dtype=torch.float64), vars=["x1", "x2"]))


@pytest.mark.parametrize("call", [
    lambda s, cam: s.frame(at=3),                                      # an index
    lambda s, cam: s.frame(at=cam(1)),                                 # a camera
    lambda s, cam: s.frame(at="x2"),                                   # a variable name
    lambda s, cam: s.frame(at=_desc(s, [0.9]), position=s.room_center()),
    lambda s, cam: s.frame(at=_desc(s, [0.9]), front=np.array([1.0, 0, 0])),
    lambda s, cam: s.frame(orientation=cam(1).orientation),
    lambda s, cam: s.frame(),
])
def test_other_forms_are_errors_that_name_the_two_forms(call):
    scene = _scene()
    camera = make_formula_helpers(None, scene)["camera"]
    with pytest.raises(TypeError) as e:
        call(scene, camera)
    assert "position=" in str(e.value) and "at=" in str(e.value)


def test_a_position_takes_its_facing_from_look_at():
    scene = _scene(1)
    camera = make_formula_helpers(None, scene)["camera"]
    p = scene.room_center()
    f = scene.frame(position=p)
    assert isinstance(f, UnfacedFrame)
    ref = scene.frame(position=p, orientation=camera(3).orientation).look_at(camera(2))   # any starting facing
    got = f.look_at(camera(2))
    np.testing.assert_allclose(_np(got.first_person.left("x1")), _np(ref.first_person.left("x1")), atol=1e-12)
    moved = f.translate([1.0, 0.0, 0.0]).look_at(camera(2))
    np.testing.assert_allclose(moved.position if hasattr(moved, "position") else moved._frame_origin, p + [1.0, 0, 0])
    with pytest.raises(AttributeError, match="look_at"):
        f.first_person.left("x1")


def test_look_at_a_description_faces_the_object_it_names():
    scene = _scene(2)
    camera = make_formula_helpers(None, scene)["camera"]
    window = _desc(scene, [0.1, 0.2, 0.9, 0.3, 0.0, 0.1, 0.2])
    a = scene.frame(position=camera(1).position, orientation=camera(1).orientation).look_at(window)
    b = scene.frame(position=camera(1).position, orientation=camera(1).orientation).look_at(2)   # the index form
    np.testing.assert_allclose(_np(a.first_person.left("x1")), _np(b.first_person.left("x1")), atol=1e-12)
    np.testing.assert_allclose(_np(scene.frame(position=scene.room_center()).look_at(window).first_person.front("x1")),
                               _np(scene.frame(position=scene.room_center()).look_at(2).first_person.front("x1")), atol=1e-12)


def test_a_point_in_at_is_the_position_form():
    scene = _scene(0)
    camera = make_formula_helpers(None, scene)["camera"]
    p = scene.room_center()
    assert isinstance(scene.frame(at=p), UnfacedFrame)
    a = scene.frame(at=p, orientation=camera(2).orientation)
    b = scene.frame(position=p, orientation=camera(2).orientation)
    np.testing.assert_allclose(_np(a.first_person.left("x1")), _np(b.first_person.left("x1")), atol=1e-12)


@pytest.mark.parametrize("word", ["north", "south-west", "northeast", "east"])
def test_compass_words_need_no_facing(word):
    scene = _scene(1)
    scene.set_cardinal_vector(np.array([0.3, 0.0, 1.0]))
    camera = make_formula_helpers(None, scene)["camera"]
    p = scene.room_center()
    unfaced = scene.frame(at=p)
    # upright frames with three different headings (a pitched camera tilts the horizontal plane;
    # a frame at a point with no facing is upright, the basis compass words are defined on)
    faced = [scene.frame(position=p, orientation=scene.orientation_from_forward(np.array(v)))
             for v in ([1.0, 0.0, 0.0], [0.3, 0.0, -1.0], [-0.7, 0.0, 0.4])]
    for f in faced:                                    # the engine's own check: compass scores ignore the heading
        np.testing.assert_allclose(_np(f.first_person(word)("x1")), _np(faced[0].first_person(word)("x1")), atol=1e-9)
    np.testing.assert_allclose(_np(unfaced.first_person(word)("x1")), _np(faced[0].first_person(word)("x1")), atol=1e-9)
    assert float(unfaced.first_person[word][camera(2)]) == pytest.approx(float(faced[0].first_person[word][camera(2)]), abs=1e-9)
    attr = word.replace("-", "_")
    np.testing.assert_allclose(_np(getattr(unfaced.first_person, attr)("x1")), _np(faced[0].first_person(word)("x1")), atol=1e-9)


@pytest.mark.parametrize("use", [
    lambda f: f.first_person.left("x1"),
    lambda f: f.first_person("front-right"),
    lambda f: f.facing,
    lambda f: f.third_person,
    lambda f: f.rotate(yaw=90),
])
def test_words_that_need_a_facing_still_ask_for_one(use):
    scene = _scene(1)
    scene.set_cardinal_vector(np.array([0.3, 0.0, 1.0]))
    with pytest.raises(AttributeError, match="look_at"):
        use(scene.frame(at=scene.room_center()))


# ---------------------------------------------------------------- every place a target is read
def test_a_description_as_a_target_is_the_object_it_names():
    """displacement / direction / at / look_at read a description as description.assign(), never as coordinates."""
    scene = _scene(3)
    camera = make_formula_helpers(None, scene)["camera"]
    d = _desc(scene, [0.1, 0.95, 0.2, 0.3, 0.0, 0.1, 0.2])        # names object 1
    a = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
    np.testing.assert_allclose(a.displacement(d), a.displacement(1), atol=1e-12)
    np.testing.assert_allclose(a.at(d).displacement(camera(2)), a.at(1).displacement(camera(2)), atol=1e-12)
    np.testing.assert_allclose(_np(a.look_at(d).first_person.left("x1")), _np(a.look_at(1).first_person.left("x1")), atol=1e-12)


@pytest.mark.parametrize("bad", [
    lambda s: ProbabilisticTensor(torch.ones(3, dtype=torch.float64), vars=["o"]).tensor[:2].numpy(),   # 2 numbers
    lambda s: [1.0, 2.0, 3.0, 4.0],                                                                      # 4 numbers
    lambda s: "lamp",
])
def test_anything_else_as_a_target_is_an_error(bad):
    scene = _scene(3)
    camera = make_formula_helpers(None, scene)["camera"]
    a = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
    with pytest.raises((TypeError, ValueError)):
        a.displacement(bad(scene))


@pytest.mark.parametrize("point", [
    lambda s: [0.0, 1.0, 2.0], lambda s: (0, 1, 2), lambda s: np.array([0, 1, 2]),
    lambda s: torch.tensor([0.0, 1.0, 2.0]), lambda s: np.array([0.0, 1.0, 2.0], dtype=np.float32),
])
def test_any_three_numbers_are_a_point_in_at(point):
    scene = _scene(0)
    camera = make_formula_helpers(None, scene)["camera"]
    a = scene.frame(at=point(scene), orientation=camera(2).orientation)
    b = scene.frame(position=np.array([0.0, 1.0, 2.0]), orientation=camera(2).orientation)
    np.testing.assert_allclose(_np(a.first_person.left("x1")), _np(b.first_person.left("x1")), atol=1e-6)


def test_a_frame_with_no_facing_has_a_position_and_can_be_looked_at():
    scene = _scene(1)
    camera = make_formula_helpers(None, scene)["camera"]
    p = scene.room_center()
    unfaced = scene.frame(at=p)
    np.testing.assert_allclose(unfaced.position, p)
    a = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
    np.testing.assert_allclose(_np(a.look_at(unfaced).first_person.left("x1")), _np(a.look_at(p).first_person.left("x1")), atol=1e-12)


def test_an_empty_scene_and_a_non_entity_variable_say_so():
    scene = _scene(0)
    with pytest.raises(ValueError, match="one object"):
        scene.frame(at=ProbabilisticTensor(torch.ones(scene._num_entities(), dtype=torch.float64), vars=["o"]))
    empty = Scene(objects=[], cameras=scene.cameras, images=[None] * 3)
    with pytest.raises(ValueError, match="no objects"):
        empty.frame(at=ProbabilisticTensor(torch.zeros(3, dtype=torch.float64), vars=["x2"]))


def test_a_bare_score_in_at_gets_a_hint():
    from saturn.soft_logic.predicate_array import PredicateArray
    scene = _scene(0)
    with pytest.raises(TypeError, match="iota"):
        scene.frame(at=PredicateArray(np.zeros(scene._num_entities())))


def test_direction_reads_targets_like_every_other_reader():
    from saturn.soft_logic.predicate_array import PredicateArray
    scene = _scene(3)
    camera = make_formula_helpers(None, scene)["camera"]
    a = scene.frame(position=camera(1).position, orientation=camera(1).orientation)
    d = _desc(scene, [0.1, 0.95, 0.2, 0.3, 0.0, 0.1, 0.2])
    assert float(a.direction(target=d).degree()) == pytest.approx(float(a.direction(target=1).degree()), abs=1e-9)
    p = scene.room_center()
    assert float(a.direction(target=p).degree()) == pytest.approx(float(a.direction(target=list(p)).degree()), abs=1e-9)
    for bad in (PredicateArray(np.zeros(scene._num_entities())), np.eye(3)):
        with pytest.raises(TypeError):
            a.direction(target=bad)


def test_the_prompts_rotation_example_answers_by_the_right_hand_rule():
    from saturn.pipeline.execute import execute_code
    from test_prompt_motion_examples import StubVL
    text = open(os.path.join(os.path.dirname(__file__), "..", "prompts", "vqa.txt")).read().split("\n")
    i = next(k for k, l in enumerate(text) if l.startswith("Q: With axes x to the right, y ahead, z up"))
    body = []
    for l in text[i + 1:]:
        if not l.strip():
            break
        body.append(l)
    for yaw, pitch, want in ((25.0, 0.0, "B"), (-25.0, 0.0, "A"), (0.0, 20.0, "C")):
        f1 = np.array([0.0, 0.0, 1.0])
        th, ph = np.radians(yaw), np.radians(pitch)                      # turn right by yaw, tilt up by pitch
        f2 = np.array([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)])
        cams = [_make_camera([0, 0, 0], f1, cam_id=0), _make_camera([0.5, 0, 0], f2, cam_id=1)]
        scene = Scene(objects=[_make_object(0, [0, 0, 3.0])], cameras=cams, images=[None, None])
        R = make_formula_helpers(None, scene)["camera"]
        y, p = scene.frame(position=R(1).position, orientation=R(1).orientation).rotation_to(
            scene.frame(position=R(2).position, orientation=R(2).orientation))
        if abs(y) > 1e-6:
            assert np.sign(y) == np.sign(yaw)                             # the fixture really turns that way
        ans, _, err = execute_code("\n".join(body), "rotation", StubVL(), scene, [None, None])
        assert err is None, err
        assert ans == want, (yaw, pitch, ans)
