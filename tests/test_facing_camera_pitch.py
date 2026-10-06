"""Camera turns as predicates: frame.facing.<dir>[camera] reads how a camera is turned
(left/right) and tilted (up/down) relative to the frame it is asked from."""
import math

import numpy as np

from saturn.predicates.relations import compute_frame_obj_facing
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


def _fwd(yaw, pitch):
    """Unit view axis: +yaw turns right (toward +X, image-right of a camera looking +Z), +pitch tilts up."""
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])


def _camera(cid, position, forward):
    forward = np.asarray(forward, dtype=float) / np.linalg.norm(forward)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)  # loader form: image-down is world -Y for a level camera
    R = np.stack([right, down, forward], axis=0)  # world-to-camera (x right, y down, z ahead)
    ext = np.eye(4)
    ext[:3, :3] = R
    ext[:3, 3] = -R @ np.asarray(position, dtype=float)
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
    return Camera(id=cid, entity_id=cid, intrinsics=K, extrinsics=ext, image_size=(480, 640))


def _object(i, center, front=(0.0, 0.0, -1.0)):
    front = np.asarray(front, dtype=float) / np.linalg.norm(front)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, front)
    right /= np.linalg.norm(right)
    return MergedObject(
        id=i, label="box", views=[0], center_world=np.asarray(center, dtype=float),
        rotation_world=np.column_stack([right, up, front]), front_world=front, up_world=up,
        right_world=right, euler_world_deg=np.zeros(3), dims=np.full(3, 0.5),
        corners_world=np.zeros((8, 3)), height=0.5, support_y=0.0,
    )


def _scene(cam1_fwd, cam2_fwd):
    cams = [_camera(0, (0, 1.5, 0), cam1_fwd), _camera(1, (0.3, 1.5, 0.2), cam2_fwd)]
    return Scene(objects=[_object(0, (0, 0.5, 5))], cameras=cams, images=[None, None])


def _read(frame, label, entity):
    return float(frame.facing(label)[entity])


def _camera_frame(scene, c):
    cam = scene.cameras[c]
    return scene.frame(position=cam.position, orientation=cam.orientation)


CAM2 = 2  # entity index of camera 2 (cameras follow the one object)


def test_turn_left_right_and_tilt_up_down():
    for yaw, pitch, want in [(30, 0, "right"), (-30, 0, "left"), (0, 20, "up"), (0, -20, "down")]:
        sc = _scene(_fwd(0, 0), _fwd(yaw, pitch))
        a1 = _camera_frame(sc, 0)
        scores = {d: _read(a1, d, CAM2) for d in ("left", "right", "up", "down")}
        assert max(scores, key=scores.get) == want, (yaw, pitch, scores)
        assert scores[want] > 0.95


def test_tilt_is_relative_to_the_asking_camera():
    # camera 1 looks 20 deg down, camera 2 is level: camera 2 is tilted UP from camera 1
    sc = _scene(_fwd(0, -20), _fwd(0, 0))
    a1 = _camera_frame(sc, 0)
    assert _read(a1, "up", CAM2) > 0.95 and _read(a1, "down", CAM2) < 0.05


def test_dominant_component_wins():
    sc = _scene(_fwd(0, 0), _fwd(10, 25))
    a1 = _camera_frame(sc, 0)
    assert _read(a1, "up", CAM2) > _read(a1, "right", CAM2)
    sc = _scene(_fwd(0, 0), _fwd(60, -10))
    a1 = _camera_frame(sc, 0)
    assert _read(a1, "right", CAM2) > _read(a1, "down", CAM2)


def test_level_object_frame_reads_absolute_camera_tilt():
    sc = _scene(_fwd(0, 0), _fwd(0, 15))
    obj = sc.objects[0]
    f = sc.frame(position=obj.center_world, orientation=obj.rotation_world)
    assert _read(f, "up", CAM2) > 0.95


def test_object_facing_unchanged():
    # objects keep their own elevation; only camera slots derive up/down from the view axis
    sc = _scene(_fwd(0, -30), _fwd(0, 0))
    a1 = _camera_frame(sc, 0)
    ref = compute_frame_obj_facing(sc._entity_front_directions(), sc._entity_elevations(),
                                   a1._frame_right, a1._frame_up, a1._frame_front)
    for d, key in (("up", "obj_facing_up"), ("down", "obj_facing_down"), ("left", "obj_facing_left")):
        assert abs(_read(a1, d, 0) - float(ref[key][0])) < 1e-9


def test_facing_accepts_first_person_vertical_words():
    sc = _scene(_fwd(0, 0), _fwd(0, 20))
    a1 = _camera_frame(sc, 0)
    assert _read(a1, "above", CAM2) == _read(a1, "up", CAM2)
    assert _read(a1, "below", CAM2) == _read(a1, "down", CAM2)


def _turned(frame, label, entity):
    return float(frame.turned(label)[entity])


def test_turned_reads_the_swing_not_the_pointing():
    # a 20-degree left pan still FACES front, but it TURNED left
    sc = _scene(_fwd(0, 0), _fwd(-20, 0))
    a1 = _camera_frame(sc, 0)
    assert _read(a1, "front", CAM2) > _read(a1, "left", CAM2) - 0.02   # facing: front ~ left
    scores = {d: _turned(a1, d, CAM2) for d in ("back", "left", "right", "front")}
    assert max(scores, key=scores.get) == "left", scores


def test_turned_directions_and_turn_around():
    for yaw, pitch, want, labels in [(15, 0, "right", ("left", "right", "above", "below")),
                                     (0, 20, "above", ("left", "right", "above", "below")),
                                     (0, -10, "below", ("left", "right", "above", "below")),
                                     (10, 25, "above", ("left", "right", "above", "below")),
                                     (170, 0, "back", ("back", "left", "right", "front"))]:
        sc = _scene(_fwd(0, 0), _fwd(yaw, pitch))
        a1 = _camera_frame(sc, 0)
        scores = {d: _turned(a1, d, CAM2) for d in labels}
        assert max(scores, key=scores.get) == want, (yaw, pitch, scores)


def test_turned_is_relative_to_the_asking_camera():
    # camera 1 already panned 30 right; camera 2 at 70 right has turned further RIGHT from camera 1
    sc = _scene(_fwd(30, 0), _fwd(70, 0))
    a1 = _camera_frame(sc, 0)
    assert _turned(a1, "right", CAM2) > 0.8 and _turned(a1, "left", CAM2) < 0.2


def test_turned_no_swing_is_undecided():
    sc = _scene(_fwd(0, 0), _fwd(0, 0))
    a1 = _camera_frame(sc, 0)
    assert abs(_turned(a1, "left", CAM2) - 0.5) < 1e-9


def test_turned_right_angle_is_right_not_turned_around():
    # a 90-degree turn is fully "right" and undecided (0.5) between front and back
    sc = _scene(_fwd(0, 0), _fwd(90, 0))
    a1 = _camera_frame(sc, 0)
    assert _turned(a1, "right", CAM2) > 0.99 and abs(_turned(a1, "back", CAM2) - 0.5) < 1e-6
    sc = _scene(_fwd(0, 0), _fwd(150, 0))
    assert _turned(_camera_frame(sc, 0), "back", CAM2) > _turned(_camera_frame(sc, 0), "right", CAM2)


def test_turned_carries_the_size_of_the_turn():
    # a 2-degree jitter is not a confident turn
    sc = _scene(_fwd(0, 0), _fwd(2, 0))
    assert 0.5 < _turned(_camera_frame(sc, 0), "right", CAM2) < 0.55


def test_turned_front_means_did_not_turn():
    sc = _scene(_fwd(0, 0), _fwd(-20, 0))
    a1 = _camera_frame(sc, 0)
    assert _turned(a1, "left", CAM2) > _turned(a1, "front", CAM2)
    sc = _scene(_fwd(0, 0), _fwd(0, 0))
    a1 = _camera_frame(sc, 0)
    assert _turned(a1, "front", CAM2) >= max(_turned(a1, d, CAM2) for d in ("left", "right", "above", "below", "back"))


def test_look_at_stays_level_for_a_high_target():
    import pytest
    sc = _scene(_fwd(0, 0), _fwd(0, 0))
    obj = sc.objects[0]
    f = sc.frame(position=obj.center_world, orientation=obj.rotation_world)
    high = obj.center_world + np.array([1.0, 2.5, 0.5])
    g = f.look_at(high)
    assert np.allclose(g._frame_up, [0.0, 1.0, 0.0], atol=1e-9)
    assert abs(float(g._frame_front[1])) < 1e-9
    assert float(g.first_person.above[high]) > 0.5
    with pytest.raises(ValueError):
        f.look_at(obj.center_world + np.array([0.0, 3.0, 0.0]))


def test_retry_restores_program_state():
    from saturn.pipeline.execute import _program_state, _restore_program_state
    sc = _scene(_fwd(0, 0), _fwd(0, 0))
    pre = _program_state(sc)
    sc.set_cardinal_vector([1.0, 0.0, 0.0])
    sc.set_axis_convention(right="+X", up="-Y", forward="+Z")
    _restore_program_state(sc, pre)
    assert sc._scene_north_vector is None and sc._axis_convention_M is None
