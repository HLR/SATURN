"""Scene frames: scene._frame(at=camera(N)), hfov_deg= in the at= form,
match_direction's freedom/observer/facing contract, and facing-override
phrase matching (saturn/scene/scene.py, saturn/scene/cardinal.py)."""

import inspect
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_fov_propagation import _make_camera, _make_object  # noqa: E402

from saturn.pipeline.formula import make_formula_helpers  # noqa: E402
from saturn.scene.cardinal import _matching_facing_override  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402


@pytest.fixture
def scene():
    # Two cameras with different poses and focal lengths, two objects.
    cams = [
        _make_camera([0, 0, 0], [0, 0, 1], cam_id=0, fx=500.0),
        _make_camera([2, 0, 5], [1, 0, 0], cam_id=1, fx=1000.0),
    ]
    objs = [_make_object([1, 0, 2], obj_id=0), _make_object([-1, 0, 3], obj_id=1)]
    return Scene(objects=objs, cameras=cams, images=[])


def _camera_helper(scene):
    return make_formula_helpers(lambda *a, **k: None, scene)["camera"]


# ---- scene._frame(at=camera(N)) ---------------------------------------------


@pytest.mark.parametrize("n", [1, 2])
def test_frame_at_camera_ref_uses_that_cameras_axes_and_fov(scene, n):
    camera = _camera_helper(scene)
    cam = scene.cameras[n - 1]
    frame = scene._frame(at=camera(n))
    reference = scene._frame(at=cam)
    np.testing.assert_allclose(frame._frame_origin, cam.position_world)
    np.testing.assert_allclose(frame._frame_front, reference._frame_front)
    np.testing.assert_allclose(frame._frame_right, reference._frame_right)
    np.testing.assert_allclose(frame._frame_up, reference._frame_up)
    assert frame._hfov_deg == pytest.approx(cam.hfov_deg, rel=1e-9)


def test_frame_at_camera_ref_matches_camera_tuple_with_front(scene):
    camera = _camera_helper(scene)
    via_ref = scene._frame(at=camera(2), front=np.array([0.0, 0.0, -1.0]))
    via_tuple = scene._frame(at=("camera", 1), front=np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(via_ref._frame_origin, via_tuple._frame_origin)
    np.testing.assert_allclose(via_ref._frame_front, via_tuple._frame_front)
    assert via_ref._hfov_deg == pytest.approx(via_tuple._hfov_deg, rel=1e-9)


def test_frame_at_bare_int_is_still_an_object(scene):
    frame = scene._frame(at=1)
    np.testing.assert_allclose(frame._frame_origin, scene.objects[1].center_world)
    assert frame._hfov_deg is None


# ---- hfov_deg= in the at= form ---------------------------------------------


def test_frame_at_honours_explicit_hfov(scene):
    camera = _camera_helper(scene)
    assert scene._frame(at=("camera", 1), hfov_deg=42.0)._hfov_deg == pytest.approx(42.0)
    assert scene._frame(at=scene.cameras[0], hfov_deg=42.0)._hfov_deg == pytest.approx(42.0)
    assert scene._frame(at=camera(1), hfov_deg=42.0)._hfov_deg == pytest.approx(42.0)
    assert scene._frame(at=0, hfov_deg=42.0)._hfov_deg == pytest.approx(42.0)
    same = scene._frame(at=0, same_as=scene._frame(at=("camera", 0)), hfov_deg=42.0)
    assert same._hfov_deg == pytest.approx(42.0)


def test_frame_at_without_hfov_keeps_camera_fov(scene):
    cam1 = scene.cameras[1]
    assert scene._frame(at=("camera", 1))._hfov_deg == pytest.approx(cam1.hfov_deg, rel=1e-9)
    assert scene._frame(at=("camera", 1), hfov_deg=None)._hfov_deg == pytest.approx(
        cam1.hfov_deg, rel=1e-9
    )
    assert scene._frame(at=0)._hfov_deg is None


# ---- match_direction: freedom / observer / facing --------------------------


def _point_at_yaw(yaw_deg, dist=3.0):
    """Point at *yaw_deg* clockwise from camera 0's front (+Z), right = +X."""
    r = math.radians(yaw_deg)
    return np.array([math.sin(r) * dist, 0.0, math.cos(r) * dist])


def test_match_direction_freedom_is_documented_as_ignored():
    param = inspect.signature(Scene.match_direction).parameters["freedom"]
    assert param.default is None
    assert "Ignored" in Scene.match_direction.__doc__


@pytest.mark.parametrize("freedom", [None, 4, 8])
def test_match_direction_options_decide_the_number_of_directions(scene, freedom):
    kw = {} if freedom is None else {"freedom": freedom}
    four = {"A": "front", "B": "right", "C": "back", "D": "left"}
    eight = {"A": "front left", "B": "front right", "C": "back left", "D": "back right"}
    # 50 deg right of front: nearest of the four is "right".
    assert scene.match_direction(("camera", 0), _point_at_yaw(50), four, **kw) == "B"
    # 40 deg right of front: nearest of the four diagonals is "front right".
    assert scene.match_direction(("camera", 0), _point_at_yaw(40), eight, **kw) == "B"


def test_match_direction_cardinal_ignores_observer_and_facing(scene):
    options = {"A": "north", "B": "east", "C": "south", "D": "west"}
    north = np.array([0.0, 0.0, 1.0])
    src, tgt = np.zeros(3), np.array([0.0, 0.0, 4.0])
    plain = scene.match_direction(src, tgt, options, north_vector=north)
    assert plain == "A"
    assert scene.match_direction(src, tgt, options, north_vector=north, observer=1) == plain
    assert scene.match_direction(src, tgt, options, north_vector=north, facing=1) == plain


# ---- facing-override phrase matching ---------------------------------------


def _ov(phrase):
    return {"phrase": phrase, "front_cam_id": 0}


def test_facing_override_strips_only_a_leading_article():
    assert _matching_facing_override([_ov("the door")], "door") is not None
    assert _matching_facing_override([_ov("The entrance")], "entrance_door") is not None
    # "the " inside a word ("lathe ", "bathe ") is part of the phrase.
    assert _matching_facing_override([_ov("the lathe machine")], "lathe machine") is not None
    assert _matching_facing_override([_ov("the bathe tub")], "batub") is None
    # A word that merely starts with "the" keeps it ("ater" would match "water").
    assert _matching_facing_override([_ov("theater")], "water heater") is None


@pytest.mark.parametrize("phrase", ["the ", "the", "  ", ""])
def test_facing_override_empty_phrase_matches_nothing(phrase):
    assert _matching_facing_override([_ov(phrase)], "sofa") is None


def test_facing_override_empty_label_matches_nothing():
    assert _matching_facing_override([_ov("door")], "") is None
    assert _matching_facing_override([_ov("door")], None) is None


def test_unlabelled_landmark_keeps_its_own_front():
    # Object 0 has no label and faces +Z; an override for "door" points at the
    # camera on +X and must not apply to it.
    cam = _make_camera([5, 0, 0], [-1, 0, 0], cam_id=0)
    obj = _make_object([0, 0, 0], obj_id=0, label="")
    scene = Scene(objects=[obj], cameras=[cam], images=[])
    src, tgt = np.zeros(3), np.array([0.0, 0.0, 3.0])
    without = scene.direction(src, tgt, north_landmark=0, landmark_heading="north")
    scene._facing_overrides = [_ov("door")]
    with_override = scene.direction(src, tgt, north_landmark=0, landmark_heading="north")
    assert without == "north"
    assert with_override == without
