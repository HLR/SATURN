"""Smoke tests for the scene.constraint namespace.

Verifies:
  - lazy property creation (no __init__ change required)
  - rotation() updates scene.cameras[i].extrinsics in place
  - same_position() makes constrained cameras share world position
  - clear() restores VGGT originals
  - object entities are rejected with NotImplementedError
"""
import numpy as np
import pytest

from saturn.scene.pose_solver import R_y, _world_to_cam_translation, _compose_extrinsics
from saturn.scene.scene import Scene
from saturn.scene.constraints import _ConstraintNamespace
from saturn.scene.types import Camera, MergedObject


def _make_object(obj_id: int, center, front=(0.0, 0.0, 1.0)) -> MergedObject:
    """Build a MergedObject with axis-aligned intrinsic orientation."""
    center = np.asarray(center, dtype=float)
    front = np.asarray(front, dtype=float)
    front = front / (np.linalg.norm(front) + 1e-12)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, front)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    rotation = np.column_stack([right, up, front])
    return MergedObject(
        id=obj_id,
        label=f"obj_{obj_id}",
        views=[0],
        center_world=center,
        rotation_world=rotation,
        front_world=front,
        up_world=up,
        right_world=right,
        euler_world_deg=np.array([0.0, 0.0, 0.0]),
        dims=np.array([0.5, 0.5, 0.5]),
        corners_world=np.zeros((8, 3)),
        height=0.5,
        support_y=float(center[1] - 0.25),
    )


def _make_camera(idx: int, yaw_deg: float, position=None) -> Camera:
    """Build a Camera dataclass with a yaw-only orientation at given position."""
    if position is None:
        position = np.zeros(3)
    # Loader form (canonicalize_y_up): camera y points down, so R_c2w = R_y(yaw) @ diag(1,-1,1)
    # and the image-up is world +Y. A bare R_y(yaw) camera is upside down and the solver
    # (correctly) reads gravity from it as -Y.
    R_w2c = (R_y(yaw_deg) @ np.diag([1.0, -1.0, 1.0])).T  # extrinsics is world-to-cam
    t_w2c = _world_to_cam_translation(R_w2c, position)
    ext = _compose_extrinsics(R_w2c, t_w2c)
    return Camera(
        id=idx,
        entity_id=idx,
        intrinsics=np.eye(3),
        extrinsics=ext,
        image_size=(100, 100),
    )


def _make_scene(headings):
    cams = [_make_camera(i, h) for i, h in enumerate(headings)]
    return Scene(objects=[], cameras=cams, images=[None] * len(cams))


def test_constraint_property_is_lazy_and_idempotent():
    """scene.constraint should be created on first access and cached after."""
    scene = _make_scene([0, 90, 180])
    assert not hasattr(scene, "_constraint_ns")
    ns = scene.constraint
    assert isinstance(ns, _ConstraintNamespace)
    assert scene.constraint is ns  # second access returns same instance


def test_rotation_constraint_propagates_anchor():
    """After rotation(0, 1, yaw=90), cam1's front should be exactly cam0's front
    rotated by 90° clockwise — even if VGGT estimated cam1 incorrectly."""
    # cam1 starts noisy at +78° (true would be +90°)
    scene = _make_scene([0, 78, 0])
    scene.constraint.rotation(scene.cameras[0], scene.cameras[1], yaw=90)

    # cam1's R_c2w should now equal R_y(90) @ R_c2w[0]
    R_c2w_1 = scene.cameras[1].extrinsics[:3, :3].T
    expected = R_y(90) @ np.eye(3)
    # average solver may shift the anchor slightly; cam1 should match anchor + 90
    # We check the relative rotation between cam0 and cam1 is exactly 90°.
    R_c2w_0 = scene.cameras[0].extrinsics[:3, :3].T
    relative = R_c2w_0.T @ R_c2w_1   # = R_y(90)
    assert np.allclose(relative, R_y(90), atol=1e-6)


def test_rotation_constraint_chain():
    """Two chained rotations should compose correctly."""
    scene = _make_scene([0, 78, 172])
    scene.constraint.rotation(scene.cameras[0], scene.cameras[1], yaw=90)
    scene.constraint.rotation(scene.cameras[0], scene.cameras[2], yaw=180)
    R_c2w_0 = scene.cameras[0].extrinsics[:3, :3].T
    R_c2w_2 = scene.cameras[2].extrinsics[:3, :3].T
    relative = R_c2w_0.T @ R_c2w_2
    assert np.allclose(relative, R_y(180), atol=1e-6)


def test_same_position_snaps_to_anchor_camera():
    """same_position moves every camera in the group onto the lowest-indexed
    member (anchor gauge), so camera 0 stays at the origin."""
    cams = [
        _make_camera(0, 0,   position=np.array([0.0, 0.0, 0.0])),
        _make_camera(1, 90,  position=np.array([2.0, 0.0, 0.0])),
        _make_camera(2, 180, position=np.array([0.0, 0.0, 2.0])),
    ]
    scene = Scene(objects=[], cameras=cams, images=[None] * 3)
    scene.constraint.same_position(scene.cameras[0], scene.cameras[1], scene.cameras[2])
    # All three now share camera 0's position
    p0 = scene.cameras[0].position_world
    p1 = scene.cameras[1].position_world
    p2 = scene.cameras[2].position_world
    expected = np.array([0.0, 0.0, 0.0])
    assert np.allclose(p0, expected, atol=1e-9)
    assert np.allclose(p1, expected, atol=1e-9)
    assert np.allclose(p2, expected, atol=1e-9)


def test_clear_restores_originals():
    """clear() should restore VGGT's original extrinsics."""
    scene = _make_scene([5, 78, 172])
    original_ext = [c.extrinsics.copy() for c in scene.cameras]
    scene.constraint.rotation(scene.cameras[0], scene.cameras[2], yaw=180)
    # cam2 should differ from original
    assert not np.allclose(scene.cameras[2].extrinsics, original_ext[2], atol=1e-3)
    scene.constraint.clear()
    for cam, orig in zip(scene.cameras, original_ext):
        assert np.allclose(cam.extrinsics, orig, atol=1e-9)


def test_object_entity_rejected_with_clear_error():
    """Passing an object entity (no `extrinsics` attr) must raise NotImplementedError."""
    scene = _make_scene([0, 90])

    class _FakeObject:
        center_world = np.zeros(3)
    fake_obj = _FakeObject()
    with pytest.raises(NotImplementedError, match="object entities"):
        scene.constraint.rotation(fake_obj, scene.cameras[0], yaw=90)


def test_unrelated_entity_rejected_with_value_error():
    """An entity that's neither a scene camera nor an object should raise ValueError."""
    scene = _make_scene([0, 90])
    other_cam = _make_camera(99, 45)  # not in scene.cameras
    with pytest.raises(ValueError, match="not one of this scene's cameras"):
        scene.constraint.rotation(other_cam, scene.cameras[0], yaw=90)


def test_no_constraint_no_mutation():
    """Touching scene.constraint without calling any method should not mutate
    scene.cameras (verifies lazy init is truly side-effect-free)."""
    scene = _make_scene([5, 78, 172])
    original_ext = [c.extrinsics.copy() for c in scene.cameras]
    _ = scene.constraint   # touch the property
    for cam, orig in zip(scene.cameras, original_ext):
        assert np.allclose(cam.extrinsics, orig, atol=1e-12)


def test_records_returns_accumulated_constraints():
    scene = _make_scene([0, 78, 172])
    scene.constraint.rotation(scene.cameras[0], scene.cameras[1], yaw=90)
    scene.constraint.same_position(scene.cameras[0], scene.cameras[2])
    recs = scene.constraint.records
    assert len(recs) == 2
    assert recs[0]["type"] == "rotation"
    assert recs[0]["from_cam"] == 0 and recs[0]["to_cam"] == 1 and recs[0]["yaw"] == 90.0
    assert recs[1]["type"] == "same_position"
    assert recs[1]["cams"] == [0, 2]


# ---------------------------------------------------------------------------
# face(): object-orientation constraint
# ---------------------------------------------------------------------------


def _scene_with_object(obj_front=(0.0, 0.0, 1.0), obj_pos=(0.0, 0.0, 0.0),
                       cam_positions=((5.0, 0.0, 0.0), (0.0, 0.0, 5.0))):
    cams = [
        _make_camera(i, 0.0, position=np.array(p, dtype=float))
        for i, p in enumerate(cam_positions)
    ]
    obj = _make_object(0, obj_pos, front=obj_front)
    return Scene(objects=[obj], cameras=cams, images=[None] * len(cams))


def test_face_rewrites_front_toward_camera():
    """After face(obj, toward=cam0), obj.front_world should point at cam0."""
    # Object at origin, cam0 at (+5, 0, 0). Object initially faces +Z.
    scene = _scene_with_object()
    scene.constraint.face(scene.objects[0], toward=scene.cameras[0])
    obj = scene.objects[0]
    expected_front = np.array([1.0, 0.0, 0.0])  # toward cam0
    assert np.allclose(obj.front_world, expected_front, atol=1e-9)
    # Up should remain world-up; right = cross(up, front) = cross(+Y, +X) = -Z
    assert np.allclose(obj.up_world, np.array([0.0, 1.0, 0.0]), atol=1e-9)
    assert np.allclose(obj.right_world, np.array([0.0, 0.0, -1.0]), atol=1e-9)
    # rotation_world columns = [right, up, front]
    assert np.allclose(obj.rotation_world[:, 2], expected_front, atol=1e-9)


def test_face_orientation_property_reflects_constraint():
    """obj.orientation should be derived from the rewritten front/up."""
    scene = _scene_with_object()
    scene.constraint.face(scene.objects[0], toward=scene.cameras[0])
    R = scene.objects[0].orientation
    # Third column = front, second column = up
    assert np.allclose(R[:, 2], np.array([1.0, 0.0, 0.0]), atol=1e-9)
    assert np.allclose(R[:, 1], np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_face_accepts_tuple_target():
    """face() should accept ('camera', N) and ('object', N) target specs."""
    scene = _scene_with_object()
    scene.constraint.face(scene.objects[0], toward=("camera", 1))
    expected_front = np.array([0.0, 0.0, 1.0])  # cam1 at (0,0,5)
    assert np.allclose(scene.objects[0].front_world, expected_front, atol=1e-9)


def test_face_accepts_point_target():
    """face() should accept a bare 3D point as the target."""
    scene = _scene_with_object()
    scene.constraint.face(scene.objects[0], toward=np.array([0.0, 0.0, -3.0]))
    assert np.allclose(scene.objects[0].front_world,
                       np.array([0.0, 0.0, -1.0]), atol=1e-9)


def test_face_clear_restores_original_orientation():
    """clear() must restore the pre-face front/up/right axes."""
    scene = _scene_with_object(obj_front=(0.0, 0.0, 1.0))
    original_front = scene.objects[0].front_world.copy()
    original_up = scene.objects[0].up_world.copy()
    scene.constraint.face(scene.objects[0], toward=scene.cameras[0])
    assert not np.allclose(scene.objects[0].front_world, original_front, atol=1e-6)
    scene.constraint.clear()
    assert np.allclose(scene.objects[0].front_world, original_front, atol=1e-9)
    assert np.allclose(scene.objects[0].up_world, original_up, atol=1e-9)


def test_face_rejects_camera_as_obj_arg():
    """face(cam, toward=...) must raise — first arg must be an object."""
    scene = _scene_with_object()
    with pytest.raises(ValueError, match="could not resolve"):
        scene.constraint.face(scene.cameras[0], toward=scene.cameras[1])


def test_face_rejects_coincident_obj_and_target():
    """face() must raise when object and target are at the same position."""
    scene = _scene_with_object(obj_pos=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="same position"):
        scene.constraint.face(scene.objects[0], toward=np.zeros(3))


def test_face_vertical_target_uses_fallback_up():
    """When forward is parallel to world +Y, fall back to a non-degenerate up."""
    scene = _scene_with_object()
    # Target directly above the object
    scene.constraint.face(scene.objects[0], toward=np.array([0.0, 5.0, 0.0]))
    obj = scene.objects[0]
    # Front should point straight up
    assert np.allclose(obj.front_world, np.array([0.0, 1.0, 0.0]), atol=1e-9)
    # Basis must remain orthonormal
    assert abs(float(np.dot(obj.front_world, obj.right_world))) < 1e-9
    assert abs(float(np.dot(obj.front_world, obj.up_world))) < 1e-9
    assert abs(float(np.dot(obj.up_world, obj.right_world))) < 1e-9


def test_face_records_constraint_in_records():
    scene = _scene_with_object()
    scene.constraint.face(scene.objects[0], toward=scene.cameras[0])
    recs = scene.constraint.records
    assert len(recs) == 1
    assert recs[0]["type"] == "face"
    assert recs[0]["object"] == 0
    assert recs[0]["toward"]["kind"] == "camera"
