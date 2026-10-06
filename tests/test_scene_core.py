"""Scene core: cloned and translated cameras, frame specs, NMS and anchor predicates."""
import math

import numpy as np
import pytest

from saturn.scene.anchor import compute_anchor_predicates
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


def _camera(position, forward, cam_id=0):
    # Canonical camera: rows (image-right, image-down, forward).
    p = np.asarray(position, float)
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    R = np.stack([r, -u, f])
    ext = np.eye(4)
    ext[:3, :3] = R
    ext[:3, 3] = -R @ p
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    return Camera(id=cam_id, entity_id=cam_id, intrinsics=K, extrinsics=ext,
                  image_size=(480, 640))


def _obj(i, center, front=(0, 0, -1), label="obj", score=None, bbox=None):
    c = np.asarray(center, float)
    f = np.asarray(front, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    corners = c + np.array([[sx, sy, sz] for sx in (-.25, .25)
                            for sy in (-.25, .25) for sz in (-.25, .25)])
    obj = MergedObject(
        id=i, label=label, views=[0], center_world=c,
        rotation_world=np.column_stack([r, u, f]), front_world=f, up_world=u,
        right_world=r, euler_world_deg=np.zeros(3), dims=np.array([.5, .5, .5]),
        corners_world=corners, height=.5, support_y=c[1] - .25, world_points=None,
    )
    if score is not None:
        obj.per_view_scores = {0: score}
    if bbox is not None:
        obj.per_view_bboxes = {0: bbox}
    return obj


def _scene():
    cams = [_camera([0, 0, 0], [0, 0, 1], 0), _camera([2, 0, 0], [0, 0, 1], 1)]
    objs = [_obj(0, [0, 0, 3], label="chair"), _obj(1, [1.5, 0, 3], label="table"),
            _obj(2, [-1.5, 0, 3], label="lamp")]
    return Scene(objs, cams, [None, None])


def _vals(t):
    t = getattr(t, "tensor", t)
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else t
    return np.asarray(t, dtype=float)


# cloned cameras ---------------------------------------------------------------

def test_cloned_camera_shares_live_scene_and_sizes_predicates():
    scene = _scene()
    v = scene.cameras[0].clone().rotate(yaw=90)
    assert v._scene is scene
    scene.add_camera(v)
    n = len(scene.objects) + len(scene.cameras)
    assert v._scene is scene
    assert len(v.left) == n
    assert _vals(v.first_person.left).shape == (n,)


def test_object_clone_camera_gets_scene_backref_on_add():
    scene = _scene()
    c = scene.objects[0].clone()
    scene.add_camera(c)
    n = len(scene.objects) + len(scene.cameras)
    assert len(c.left) == n
    assert _vals(c.first_person.left).shape == (n,)


# frame front given as an entity ----------------------------------------------

def test_frame_front_entity_spec_uses_forward_not_right():
    scene = _scene()
    cam1 = scene.cameras[1]
    by_instance = scene._frame(at=("object", 0), front=cam1)
    by_tuple = scene._frame(at=("object", 0), front=("camera", 1))
    by_index = scene._frame(at=("object", 1), front=0)
    np.testing.assert_allclose(by_tuple._frame_front, by_instance._frame_front, atol=1e-9)
    np.testing.assert_allclose(by_tuple._frame_front, [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(by_index._frame_front, scene.objects[0].front_vec, atol=1e-9)


# camera cloned from an object ------------------------------------------------

def test_object_clone_camera_has_no_hfov():
    scene = _scene()
    c = scene.objects[0].clone()
    assert c.hfov_deg is None
    assert Scene._camera_hfov_deg(c) is None
    assert scene.cameras[0].hfov_deg == pytest.approx(2 * math.degrees(math.atan(320 / 500)))


# camera pitch ---------------------------------------------------------------

@pytest.mark.parametrize("forward", [[0, 0, 1], [1, 0, 0], [0, 0, -1]])
def test_camera_rotate_pitch_tilts_up_about_body_right(forward):
    cam = _camera([0, 0, 0], forward)
    f = np.asarray(forward, float)
    new = cam.rotate(pitch=30).front_vec
    np.testing.assert_allclose(new, f * math.cos(math.radians(30)) + [0, 0.5, 0], atol=1e-9)
    # Same answer as the Anchor protocol used by MergedObject.
    from saturn.scene.anchor import Anchor
    np.testing.assert_allclose(Anchor.rotate(cam, pitch_deg=30).front_vec, new, atol=1e-9)


def test_camera_rotate_yaw_then_pitch_keeps_pitch():
    cam = _camera([0, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(cam.rotate(yaw=90, pitch=30).front_vec,
                               [math.cos(math.radians(30)), 0.5, 0], atol=1e-9)
    # Yaw only: turn right.
    np.testing.assert_allclose(cam.rotate(yaw=90).front_vec, [1, 0, 0], atol=1e-9)


# translated cameras ---------------------------------------------------------

def test_translated_camera_keeps_canonical_extrinsics():
    cam = _camera([0, 0, 0], [0, 0, 1])
    for moved in (cam.translate([0, 0, 0]), cam.rotate(yaw=0)):
        np.testing.assert_allclose(moved.extrinsics, cam.extrinsics, atol=1e-9)
    moved = cam.translate([1, 2, 3])
    np.testing.assert_allclose(moved.position, [1, 2, 3], atol=1e-9)
    assert np.linalg.det(moved.extrinsics[:3, :3]) == pytest.approx(-1.0)


# non-maximum suppression ----------------------------------------------------

def test_nms_suppressed_box_does_not_suppress_others():
    objs = [
        _obj(0, [0, 0, 3], label="chair", score=0.5, bbox=[0, 0, 10, 10]),
        _obj(1, [0, 0, 3], label="chair", score=0.9, bbox=[-2, 0, 8, 10]),
        _obj(2, [0, 0, 3], label="chair", score=0.4, bbox=[2, 0, 12, 10]),
    ]
    scene = Scene(objs, [_camera([0, 0, 0], [0, 0, 1])], [None])
    assert scene.nms_objects() == 1
    assert sorted(o.per_view_scores[0] for o in scene.objects) == [0.4, 0.9]


# object elevation -----------------------------------------------------------

@pytest.mark.parametrize("front", [[0, 0, 1], [0, 0, -1], [1, 0, 0]])
def test_object_elevation_from_front_vector(front):
    obj = _obj(0, [0, 0, 3], front=front).rotate(pitch_deg=30)
    scene = Scene([obj], [_camera([0, 0, 0], [0, 0, 1])], [None])
    np.testing.assert_allclose(scene._object_elevations(), [30.0], atol=1e-6)


# anchor predicates under camera pitch ----------------------------------------

def test_anchor_horizontal_predicates_ignore_camera_pitch():
    ang = math.radians(40)
    target = [3 * math.sin(ang), 0.0, 3 * math.cos(ang)]
    level = None
    for pitch in (0, 30, 60, 85):
        f = [0, -math.sin(math.radians(pitch)), math.cos(math.radians(pitch))]
        cam = _camera([0, 1.5, 0], f)
        scene = Scene([_obj(0, target)], [cam], [None])
        vals = (float(cam.front_right[0]), float(cam.right[0]))
        lazy = cam.translate([0, 0, 0])  # free-floating: _score_single path
        np.testing.assert_allclose((float(lazy.front_right[0]), float(lazy.right[0])),
                                   vals, atol=1e-9)
        if level is None:
            level = vals
        np.testing.assert_allclose(vals, level, atol=1e-9)
    assert level[0] > level[1]  # 40 deg to the right is front_right, not right


def test_compute_anchor_predicates_matches_single_for_pitched_anchor():
    cam = _camera([0, 1.5, 0], [0, -math.sin(1.2), math.cos(1.2)])
    obj = _obj(0, [2.0, 0.0, 2.5])
    preds = compute_anchor_predicates([obj, cam])
    Scene([obj], [cam], [None])
    lazy = cam.translate([0, 0, 0])
    for name in ("front", "right", "front_right", "left"):
        assert preds[name][0, 1] == pytest.approx(float(getattr(lazy, name)[0]))
