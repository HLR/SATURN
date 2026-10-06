import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_frame_first_api import _make_camera, _make_object  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402


@pytest.fixture
def scene():
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0), _make_camera([0, 0, 10], [0, 0, -1], cam_id=1)]
    objs = [_make_object(0, [0, 0, 5]), _make_object(1, [3, 0, 5])]
    return Scene(objects=objs, cameras=cams, images=[None, None])


def _cam_frame(scene, i):
    c = scene.cameras[i]
    return scene.frame(position=c.position, orientation=c.orientation)


def test_local_matches_first_person_directions(scene):
    a = _cam_frame(scene, 0)
    r, u, f = a.local(scene.objects[0].position)
    assert f == pytest.approx(5, abs=1e-6) and abs(r) < 1e-6 and abs(u) < 1e-6
    # the direction label of a point and the sign of its local coordinates agree
    for p in ([4, 0, 5], [-4, 0, 5], [0, 0, -5], [3, 0, -3]):
        r, u, f = a.local(np.array(p, float))
        lab = a.direction(target=np.array(p, float)).label(freedom=8)
        if f < -1 and abs(r) < 1: assert lab == "back"
        if r > 1 and abs(f) < 1e-6: assert "right" in lab
        if r < -1 and abs(f) < 1e-6: assert "left" in lab


def test_look_at_faces_target_and_keeps_position(scene):
    a = _cam_frame(scene, 0)
    t = scene.objects[1].position
    b = a.look_at(t)
    assert np.allclose(b.frame_origin, a.frame_origin)
    r, u, f = b.local(t)
    assert f > 0 and abs(r) < 1e-6 and abs(u) < 1e-6


def test_side_of_object_seen_by_another_camera(scene):
    # "camera 1 sees the NORTH side of object 0": a frame at the object facing camera 1 is 'north'.
    north = scene.frame(position=scene.objects[0].position, orientation=np.eye(3)).look_at(scene.cameras[0].position)
    # camera 2 stands on the opposite side -> it sees the south side (local 'back')
    assert north.direction(target=scene.cameras[1].position).label(freedom=8) == "back"


def test_camera_motion_in_camera1_axes(scene):
    a = _cam_frame(scene, 0)
    r, u, f = a.local(scene.cameras[1].position)
    assert f == pytest.approx(10, abs=1e-6)   # camera 2 is straight ahead of camera 1
