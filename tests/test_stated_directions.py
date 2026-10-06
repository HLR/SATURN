"""Stated directions as frames, the self-read guard, direction() at camera(N),
and camera facing in compass terms.

Scene (world +Y up; camera 1 at the origin looking +Z, so +X is its right):
    camera 1 at (0,0,0) facing +Z; camera 2 at (0,0,10) facing -Z
    object 0 "tv"    at (0,0,5), its front toward camera 1 (-Z)
    object 1 "lamp"  at (3,0,5)
    object 2 "door"  at (0,0,12)  (behind camera 2: the room is between the cameras)
"""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_frame_first_api import _make_camera, _make_object  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402
from saturn.pipeline.formula import make_formula_helpers  # noqa: E402


@pytest.fixture
def scene():
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0), _make_camera([0, 0, 10], [0, 0, -1], cam_id=1)]
    objs = [_make_object(0, [0, 0, 5]), _make_object(1, [3, 0, 5]), _make_object(2, [0, 0, 12])]
    return Scene(objects=objs, cameras=cams, images=[None, None])   # objects face -Z: the tv faces camera 1


def _camera(scene):
    return make_formula_helpers(lambda *a, **k: None, scene)["camera"]


def _frame(scene, e):
    return scene.frame(position=e.position, orientation=e.orientation)


def test_tv_front_points_at_camera1(scene):
    assert np.allclose(scene.objects[0].front_vec, [0, 0, -1], atol=1e-6)


# ---- "using or facing X": X's frame turned around -------------------------------
def test_watching_x_faces_x(scene):
    me = _frame(scene, scene.objects[0]).rotate(yaw=180)      # watching the tv
    assert np.allclose(me.front_vec, [0, 0, 1], atol=1e-6)    # I look toward the tv (away from camera 1)
    # the lamp at +X is then on my right, camera 1 behind me
    assert me.direction(target=1).label(freedom=4) == "right"
    assert me.direction(target=scene.cameras[0].position).label(freedom=4) == "back"


def test_sitting_on_x_is_x_frame(scene):
    me = _frame(scene, scene.objects[0])                       # X's own frame
    assert me.direction(target=1).label(freedom=4) == "left"   # mirror of the watching case


# ---- look_at for "at X facing Y" and "entering through X" ----------------------
def test_at_x_facing_y(scene):
    me = _frame(scene, scene.objects[0]).look_at(scene.objects[1].position)
    r, u, f = me.local(scene.objects[1].position)
    assert f > 0 and abs(r) < 1e-6
    assert np.allclose(me.frame_origin, scene.objects[0].position)


def test_entering_through_door_faces_room(scene):
    me = _frame(scene, scene.objects[2]).look_at(scene.room_center())
    # the room (and camera 1) lies ahead of someone entering through the door
    assert me.direction(target=scene.cameras[0].position).label(freedom=4) == "front"


def test_look_at_camera_and_object_index(scene):
    camera = _camera(scene)
    a = _frame(scene, scene.cameras[0])
    assert a.look_at(camera(2)).direction(target=camera(2)).label(freedom=8) == "front"
    assert a.look_at(1).direction(target=1).label(freedom=8) == "front"


# ---- direction() accepts camera(N) ---------------------------------------------
def test_direction_accepts_camera_ref(scene):
    camera = _camera(scene)
    a = _frame(scene, scene.cameras[0]).rotate(yaw=90)       # facing +X
    d_ref = a.direction(target=camera(2)).degree()
    d_pos = a.direction(target=scene.cameras[1].position).degree()
    assert d_ref == pytest.approx(d_pos) and d_ref == pytest.approx(270.0, abs=1e-6)  # camera 2 on my left


def test_angle_option_counterclockwise(scene):
    a = _frame(scene, scene.cameras[0])
    # lamp at (3,0,5) from the origin facing +Z: ~31 degrees clockwise
    assert a.direction(target=1).degree() == pytest.approx(np.degrees(np.arctan2(3, 5)), abs=1e-6)


# ---- the self-read guard -------------------------------------------------------
def test_reading_the_anchors_own_camera_raises(scene):
    camera = _camera(scene)
    a = _frame(scene, scene.cameras[0])
    with pytest.raises(ValueError, match="stands exactly where the anchor stands"):
        a.first_person.front[camera(1)]
    with pytest.raises(ValueError, match="stands exactly where the anchor stands"):
        a.first_person("left")(camera(1))


def test_reading_other_entities_and_formulas_unaffected(scene):
    camera = _camera(scene)
    a = _frame(scene, scene.cameras[0])
    assert float(a.first_person.front[camera(2)]) == pytest.approx(1.0)
    assert float(a.first_person.front[0]) == pytest.approx(1.0)
    # a formula over x1 still scores every entity (the anchor's own camera just scores 0)
    t = a.first_person.front("x1")
    assert float(t.exists()) == pytest.approx(1.0)
    # reading a 3D point at the origin is the point form, not an entity read
    assert float(a.first_person.front[np.array([0.0, 0.0, 1.0])]) == pytest.approx(1.0)


def test_numpy_uses_of_first_person_arrays_unchanged(scene):
    a = _frame(scene, scene.cameras[0])
    arr = a.first_person.front
    assert arr.argmax() in (0, 2, 4)                # plain numpy reductions still work
    assert np.asarray(arr[:scene.objects_count]).shape == (3,)


# ---- compass: which way a camera faces ------------------------------------------
def test_camera_facing_in_compass_terms(scene):
    camera = _camera(scene)
    scene.set_cardinal_vector(scene.cardinalize(scene.cameras[0].front_vec, known="north"))
    a = _frame(scene, scene.cameras[0])
    labels = ["north", "east", "south", "west"]
    best = max(labels, key=lambda l: float(a.facing(l)[camera(2)]))
    assert best == "south"                                   # camera 2 looks back toward camera 1


def test_door_faces_out_is_away_from_room_center(scene):
    # "the door faces north" (facing out): v = door - room_center
    v = scene.objects[2].position - scene.room_center()
    scene.set_cardinal_vector(scene.cardinalize(v, known="north"))
    a = _frame(scene, scene.cameras[0])
    # camera 2 stands between the room center and the door: north of camera 1
    assert max(["north", "south", "east", "west"], key=lambda l: float(a.first_person(l)[_camera(scene)(2)])) == "north"
