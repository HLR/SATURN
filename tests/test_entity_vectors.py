"""Entity vectors: every object and camera exposes unit, world-frame
``front_vec`` / ``right_vec`` / ``up_vec`` and ``position``, with the sign of
``right_vec`` pinned to image-right / body-right.
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from saturn.scene.adapters import canonicalize_y_up
from saturn.scene.fusion import _ensure_proper_rotation, build_object_frame_from_front
from saturn.scene.scene import Scene
from saturn.scene.types import Camera

sys.path.insert(0, os.path.dirname(__file__))
from test_frame_first_api import _make_object  # noqa: E402


def _real_camera(yaw_deg=30.0, t=(0.2, 0.1, 0.5)):
    """A camera built the way the pipeline builds it: OpenCV extrinsics in a
    Y-down world (x = image-right), then ``canonicalize_y_up`` (a Y flip)."""
    ext = np.eye(4)
    ext[:3, :3] = R.from_euler("y", yaw_deg, degrees=True).as_matrix()
    ext[:3, 3] = t
    ext_canon, _ = canonicalize_y_up(ext)
    cam = Camera(id=0, entity_id=0, intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]]),
                 extrinsics=ext_canon, image_size=(480, 640))
    return cam, ext_canon


def _image_right_point(ext_canon, x_cam=1.0, z_cam=5.0):
    """World point whose camera-frame x > 0, i.e. it projects right of centre."""
    R_w2c, t = ext_canon[:3, :3], ext_canon[:3, 3]
    return np.linalg.solve(R_w2c, np.array([x_cam, 0.0, z_cam]) - t)


def _assert_unit_rh(e):
    f, r, u = e.front_vec, e.right_vec, e.up_vec
    for v in (f, r, u):
        assert v.shape == (3,) and np.linalg.norm(v) == pytest.approx(1.0)
    assert np.allclose(np.cross(r, u), f, atol=1e-9)  # [right, up, front] is RH


def test_camera_right_vec_is_image_right():
    cam, ext = _real_camera()
    p = _image_right_point(ext)
    u_pixel = (cam.intrinsics @ (ext[:3, :3] @ p + ext[:3, 3]))
    assert u_pixel[0] / u_pixel[2] > 320  # really on the image's right half
    assert np.dot(p - cam.position, cam.right_vec) > 0
    assert np.allclose(cam.right_vec, cam.orientation[:, 0])
    assert np.allclose(cam.heading.right, -cam.right_vec)  # heading.right is image-LEFT
    _assert_unit_rh(cam)


def test_first_person_right_agrees_with_right_vec():
    cam, ext = _real_camera()
    p = _image_right_point(ext, x_cam=5.0, z_cam=1.0)  # ~79 deg to the right
    objects = [_make_object(0, center=p), _make_object(1, center=2 * cam.position - p)]
    scene = Scene(objects=objects, cameras=[cam], images=[None])
    fp = scene._frame(at=scene.cameras[0]).first_person
    assert fp.right[0] > 0.9 and fp.left[1] > 0.9


def test_co_facing_object_has_camera_right():
    cam, _ = _real_camera()
    obj = _make_object(0, center=[0, 0, 3], front=cam.front_vec)
    assert np.allclose(obj.right_vec, cam.right_vec)
    _assert_unit_rh(obj)


def test_fusion_built_object_right_world_matches_right_vec():
    """The fusion path stores ``right_world = rotation[:, 0]``; pin that it has
    the right_vec sign (it relies on ``_ensure_proper_rotation`` flipping it)."""
    for front in ([0.3, 0.0, 1.0], [-1.0, 0.1, -0.2], [0.5, 0.0, -0.9]):
        rot = _ensure_proper_rotation(build_object_frame_from_front(np.array(front)))
        obj = _make_object(0, center=[0, 0, 0], front=rot[:, 2])
        obj.right_world, obj.up_world = rot[:, 0], rot[:, 1]
        assert np.allclose(obj.right_vec, rot[:, 0], atol=1e-9)


def test_object_vectors_are_unit_even_if_fields_are_not():
    obj = _make_object(0, center=[1, 2, 3], front=(0, 0, 1))
    obj.front_world = np.array([0.0, 0.0, 4.0])
    obj.right_world = np.array([-3.0, 0.0, 0.0])  # mirrored AND non-unit stored field
    assert np.allclose(obj.front_vec, [0, 0, 1])
    assert np.allclose(obj.right_vec, [1, 0, 0])  # derived: cross(up, front)
    assert np.allclose(obj.right_world, [-3, 0, 0])  # stored field untouched
    _assert_unit_rh(obj)


def test_storage_fields_and_aliases_agree():
    cam, _ = _real_camera()
    obj = _make_object(0, center=[1, 2, 3], front=(1, 0, 0))
    assert np.allclose(obj.front_vec, obj.front_world) and np.allclose(obj.up_vec, obj.up_world)
    assert np.allclose(obj.orientation_right, obj.right_vec)
    assert np.allclose(obj.position, obj.center_world) and np.allclose(obj.pos, obj.position)
    assert np.allclose(cam.position, cam.position_world) and np.allclose(cam.front_vec, cam.heading.forward)


def test_front_is_a_predicate_column_and_up_is_refused():
    cam, _ = _real_camera()
    objects = [_make_object(i, center=[i, 0, 3]) for i in range(3)]
    scene = Scene(objects=objects, cameras=[cam], images=[None])
    for e in (scene.objects[0], scene.cameras[0]):
        assert np.asarray(e.front).shape == (4,)  # K + C entities, not an xyz vector
        with pytest.raises(AttributeError, match="up_vec"):
            e.up
        assert not hasattr(e, "down")
