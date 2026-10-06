"""Tests for FOV propagation into anchor frames and for ``first_person`` scoring
on frames that carry a FOV.

Three layers under test:

1. ``Camera.hfov_deg`` property — computed from intrinsics + image_size.
2. ``Scene._frame(...)`` — propagates FOV when the anchor's position matches a
   camera's ``position_world`` (auto-detect), or when ``hfov_deg=`` is passed
   explicitly. Auto-detect only triggers in the explicit-pose path.
3. ``_FirstPersonNamespace._horizontal_scores`` — scores by direction only:
   a frame's FOV leaves every ``first_person`` score unchanged, inside and
   outside the FOV cone.
"""

import math
import os
import sys

import numpy as np
import pytest


from saturn.predicates.frame import FrameNamespace
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, CameraHeading, MergedObject


# ---------------------------------------------------------------------------
# Helpers — minimal Camera and MergedObject construction
# ---------------------------------------------------------------------------


def _make_camera(
    position,
    forward,
    *,
    cam_id=0,
    fx=500.0,
    image_size=(480, 640),  # (H, W)
):
    """Build a Camera with explicit position, forward direction, and intrinsics.

    The intrinsics are a standard pinhole at fx with image_size (H, W); FOV
    follows directly: hfov_deg = 2 * atan(W / (2*fx)).
    """
    position = np.asarray(position, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0)
    t = -R_w2c @ position
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R_w2c
    extrinsics[:3, 3] = t
    H, W = image_size
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=float)
    return Camera(
        id=cam_id,
        entity_id=cam_id,
        intrinsics=K,
        extrinsics=extrinsics,
        image_size=image_size,
        position_world=position,
        heading=CameraHeading(forward),
    )


def _make_object(
    position,
    *,
    obj_id=0,
    label="obj",
    views=(0,),
    orientation_conf=1.0,
):
    """Build a MergedObject at ``position`` with identity world axes."""
    p = np.asarray(position, dtype=float)
    return MergedObject(
        id=obj_id,
        label=label,
        views=list(views),
        center_world=p,
        rotation_world=np.eye(3),
        front_world=np.array([0.0, 0.0, 1.0]),
        up_world=np.array([0.0, 1.0, 0.0]),
        right_world=np.array([1.0, 0.0, 0.0]),
        euler_world_deg=np.zeros(3),
        dims=np.array([0.1, 0.1, 0.1]),
        corners_world=np.zeros((8, 3)),
        height=0.1,
        support_y=p[1] - 0.05,
        per_view_orientation_confidence={v: orientation_conf for v in views},
    )


# ---------------------------------------------------------------------------
# Layer 1: Camera.hfov_deg
# ---------------------------------------------------------------------------


class TestCameraHfovProperty:
    def test_computes_fov_from_standard_intrinsics(self):
        # fx=500, W=640  →  hfov = 2 * atan(640/1000) = 2 * atan(0.64)
        expected = math.degrees(2.0 * math.atan(640.0 / 1000.0))
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0, image_size=(480, 640))
        assert cam.hfov_deg == pytest.approx(expected, rel=1e-6)

    def test_returns_none_when_fx_is_zero(self):
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0)
        cam.intrinsics = np.zeros((3, 3))  # fx = 0
        assert cam.hfov_deg is None

    def test_returns_none_when_intrinsics_are_garbage(self):
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0)
        cam.intrinsics = None
        assert cam.hfov_deg is None

    def test_fov_scales_with_focal_length(self):
        # Wide-angle (small fx) → larger FOV; telephoto (large fx) → smaller.
        wide = _make_camera([0, 0, 0], [0, 0, 1], fx=200.0, image_size=(480, 640))
        tele = _make_camera([0, 0, 0], [0, 0, 1], fx=2000.0, image_size=(480, 640))
        assert wide.hfov_deg > tele.hfov_deg
        assert wide.hfov_deg > 90.0  # wide-angle should exceed 90°
        assert tele.hfov_deg < 30.0  # telephoto should be tight

    def test_matches_internal_helper(self):
        # The property must agree with the internal Scene helper.
        cam = _make_camera([1, 2, 3], [0, 0, 1], fx=750.0, image_size=(720, 1280))
        scene = Scene(objects=[], cameras=[cam], images=[])
        assert scene._camera_hfov_deg(cam) == pytest.approx(cam.hfov_deg, rel=1e-9)


# ---------------------------------------------------------------------------
# Layer 2: Scene.frame FOV propagation
# ---------------------------------------------------------------------------


class TestFovPropagation:
    @pytest.fixture
    def two_cam_scene(self):
        cam0 = _make_camera([0, 0, 0], [0, 0, 1], cam_id=0, fx=500.0)
        cam1 = _make_camera([2, 0, 5], [1, 0, 0], cam_id=1, fx=1000.0)
        return Scene(objects=[], cameras=[cam0, cam1], images=[])

    def test_auto_detect_matches_first_camera(self, two_cam_scene):
        cam0 = two_cam_scene.cameras[0]
        anchor = two_cam_scene.frame(
            position=cam0.position, orientation=cam0.orientation
        )
        assert anchor._hfov_deg == pytest.approx(cam0.hfov_deg, rel=1e-9)

    def test_auto_detect_matches_second_camera(self, two_cam_scene):
        cam1 = two_cam_scene.cameras[1]
        anchor = two_cam_scene.frame(
            position=cam1.position, orientation=cam1.orientation
        )
        assert anchor._hfov_deg == pytest.approx(cam1.hfov_deg, rel=1e-9)
        # Cameras have different FOVs, so this must not match cam 0.
        assert anchor._hfov_deg != pytest.approx(
            two_cam_scene.cameras[0].hfov_deg, rel=1e-3
        )

    def test_no_auto_detect_when_position_does_not_match(self, two_cam_scene):
        # Random position not coincident with any camera → no auto-detect.
        anchor = two_cam_scene.frame(
            position=np.array([10.0, 10.0, 10.0]),
            orientation=np.eye(3),
        )
        assert anchor._hfov_deg is None

    def test_no_auto_detect_for_position_slightly_off_camera(self, two_cam_scene):
        # Beyond the 1e-4 tolerance, no auto-detect.
        cam0 = two_cam_scene.cameras[0]
        anchor = two_cam_scene.frame(
            position=cam0.position + np.array([1e-3, 0, 0]),
            orientation=cam0.orientation,
        )
        assert anchor._hfov_deg is None

    def test_auto_detect_within_tolerance(self, two_cam_scene):
        # Within 1e-4 tolerance, auto-detect still fires.
        cam0 = two_cam_scene.cameras[0]
        anchor = two_cam_scene.frame(
            position=cam0.position + np.array([5e-5, 0, 0]),
            orientation=cam0.orientation,
        )
        assert anchor._hfov_deg == pytest.approx(cam0.hfov_deg, rel=1e-9)

    def test_explicit_kwarg_overrides_auto_detect(self, two_cam_scene):
        cam0 = two_cam_scene.cameras[0]
        anchor = two_cam_scene.frame(
            position=cam0.position,
            orientation=cam0.orientation,
            hfov_deg=42.0,
        )
        assert anchor._hfov_deg == pytest.approx(42.0)

    def test_explicit_kwarg_works_when_no_camera_match(self, two_cam_scene):
        anchor = two_cam_scene.frame(
            position=np.array([10.0, 10.0, 10.0]),
            orientation=np.eye(3),
            hfov_deg=75.0,
        )
        assert anchor._hfov_deg == pytest.approx(75.0)

    def test_explicit_none_falls_back_to_auto_detect(self, two_cam_scene):
        # Explicit hfov_deg=None should still auto-detect (None means "unspecified").
        cam0 = two_cam_scene.cameras[0]
        anchor = two_cam_scene.frame(
            position=cam0.position,
            orientation=cam0.orientation,
            hfov_deg=None,
        )
        assert anchor._hfov_deg == pytest.approx(cam0.hfov_deg, rel=1e-9)

    def test_empty_scene_no_crash(self):
        scene = Scene(objects=[], cameras=[], images=[])
        anchor = scene.frame(position=np.zeros(3), orientation=np.eye(3))
        assert anchor._hfov_deg is None

    def test_at_entity_path_propagates_fov(self, two_cam_scene):
        # The entity-anchor path (at=camera) carries the camera's FOV.
        cam0 = two_cam_scene.cameras[0]
        anchor = two_cam_scene._frame(at=cam0)
        assert anchor._hfov_deg == pytest.approx(cam0.hfov_deg, rel=1e-9)

    def test_object_anchor_has_no_fov(self):
        cam = _make_camera([0, 0, 0], [0, 0, 1])
        obj = _make_object([1, 0, 2])
        scene = Scene(objects=[obj], cameras=[cam], images=[])
        anchor = scene.frame(
            position=scene.objects[0].center_world,
            orientation=np.eye(3),
        )
        assert anchor._hfov_deg is None


# ---------------------------------------------------------------------------
# Layer 3: first_person scores do not depend on the frame's FOV
# ---------------------------------------------------------------------------


def _with_and_without_fov(anchor, label, idx):
    """``first_person(label)[idx]`` on ``anchor`` as built, and with its FOV cleared."""
    s_yes = float(anchor.first_person(label)[idx])
    saved = anchor._hfov_deg
    anchor._hfov_deg = None
    try:
        s_no = float(anchor.first_person(label)[idx])
    finally:
        anchor._hfov_deg = saved
    return s_yes, s_no


def test_front_labels_outside_fov_unchanged():
    """Toilet at yaw ~270 deg from camera 1 (mostly left, tiny forward component),
    outside the FOV cone: front-left and back-left score the same with and
    without the FOV, so the near-tie of the geometry stays a near-tie."""
    cam0 = _make_camera([0, 0, 0], [0, 0, 1], cam_id=0, fx=720.0, image_size=(480, 640))
    cam1 = _make_camera([0.005, -0.048, -0.046], [0.608, -0.775, 0.172], cam_id=1,
                        fx=720.0, image_size=(480, 640))
    toilet = _make_object([-0.275, -0.014, 1.101], obj_id=0, label="toilet")
    scene = Scene(objects=[toilet], cameras=[cam0, cam1], images=[])
    anchor = scene.frame(position=cam1.position, orientation=cam1.orientation)
    assert anchor._hfov_deg is not None
    fl_yes, fl_no = _with_and_without_fov(anchor, "front-left", 0)
    bl_yes, bl_no = _with_and_without_fov(anchor, "back-left", 0)
    assert fl_yes == pytest.approx(fl_no, rel=1e-9)
    assert bl_yes == pytest.approx(bl_no, rel=1e-9)
    assert abs(fl_yes - bl_yes) < 0.01


def test_front_labels_far_outside_fov_unchanged():
    """Targets 10-90 deg past the edge of a 60 deg FOV keep their cosine scores."""
    cam = _make_camera([0, 0, 0], [0, 0, 1],
                       fx=640.0 / (2.0 * math.tan(math.radians(30.0))), image_size=(480, 640))
    yaws_deg = [40, 60, 80, 120]
    objs = [_make_object([5 * math.sin(math.radians(y)), 0, 5 * math.cos(math.radians(y))], obj_id=i)
            for i, y in enumerate(yaws_deg)]
    scene = Scene(objects=objs, cameras=[cam], images=[])
    anchor = scene.frame(position=cam.position, orientation=cam.orientation)
    assert cam.hfov_deg == pytest.approx(60.0, abs=0.5)
    for i, y in enumerate(yaws_deg):
        for label, target in (("front", 0.0), ("front-right", 45.0), ("front-left", -45.0)):
            s_yes, s_no = _with_and_without_fov(anchor, label, i)
            assert s_yes == pytest.approx(s_no, rel=1e-9)
            assert s_yes == pytest.approx((1.0 + math.cos(math.radians(y - target))) / 2.0, abs=1e-9)


class TestFirstPersonInvariants:
    """Scores with the frame's FOV equal the scores without it, and match
    the cosine formula."""

    @pytest.fixture
    def cam_in_front_scene(self):
        """Camera at origin facing +Z; object directly in front at (0,0,5)."""
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0, image_size=(480, 640))
        obj = _make_object([0, 0, 5])
        return Scene(objects=[obj], cameras=[cam], images=[])

    def test_within_fov_target_unchanged(self, cam_in_front_scene):
        """Object dead-center in FOV: front score is identical with and without FOV.
        Toggle ``_hfov_deg`` in place on the same anchor to keep geometry constant."""
        cam = cam_in_front_scene.cameras[0]
        anchor = cam_in_front_scene.frame(
            position=cam.position, orientation=cam.orientation
        )
        assert anchor._hfov_deg is not None

        for label in ("front", "front-left", "front-right"):
            s_yes = float(anchor.first_person(label)[0])
            saved = anchor._hfov_deg
            anchor._hfov_deg = None
            try:
                s_no = float(anchor.first_person(label)[0])
            finally:
                anchor._hfov_deg = saved
            assert s_no == pytest.approx(s_yes, rel=1e-9), (
                f"label={label!r} on in-FOV target should be unchanged "
                f"(no={s_no} yes={s_yes})"
            )

    def test_back_and_side_labels_unchanged(self):
        """Back, back-left, back-right, left and right scores for a target
        behind the camera are the same with and without the FOV."""
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0)
        obj = _make_object([0, 0, -5])  # behind camera
        scene = Scene(objects=[obj], cameras=[cam], images=[])

        anchor = scene.frame(position=cam.position, orientation=cam.orientation)
        assert anchor._hfov_deg is not None

        for label in ("back", "back-left", "back-right", "left", "right"):
            s_yes = float(anchor.first_person(label)[0])
            saved = anchor._hfov_deg
            anchor._hfov_deg = None
            try:
                s_no = float(anchor.first_person(label)[0])
            finally:
                anchor._hfov_deg = saved
            assert s_no == pytest.approx(s_yes, rel=1e-9), (
                f"label={label!r} must not depend on the FOV; got "
                f"no={s_no} yes={s_yes}"
            )

    def test_object_anchored_frame_unaffected(self):
        """Frame anchored at an object (not a camera) has no FOV → scores match
        the hand-computed cosine formula."""
        cam = _make_camera([0, 0, 0], [0, 0, 1])
        obj = _make_object([1, 0, 2])
        target = _make_object([3, 0, 1], obj_id=1)
        scene = Scene(objects=[obj, target], cameras=[cam], images=[])
        # Build frame at object 0 with identity orientation (front=+Z).
        anchor = scene.frame(
            position=scene.objects[0].center_world,
            orientation=np.eye(3),
        )
        assert anchor._hfov_deg is None
        # Manual: vec = (3-1, 0, 1-2) = (2, 0, -1); right=+X, front=+Z
        # r = 2, f = -1, yaw = atan2(2, -1) ≈ 2.034 rad (≈ 116.6°)
        # For "front" (target_rad=0): (1 + cos(2.034))/2 ≈ (1 + (-0.447))/2 ≈ 0.276
        expected_front = (1.0 + math.cos(math.atan2(2.0, -1.0))) / 2.0
        actual_front = float(anchor.first_person("front")[1])
        assert actual_front == pytest.approx(expected_front, rel=1e-9)

    def test_left_outside_fov_unchanged(self):
        """``left`` for a target outside the FOV cone scores the same with and
        without the FOV."""
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0, image_size=(480, 640))
        obj = _make_object([-10, 0, 0.01])  # mostly left, tiny forward
        scene = Scene(objects=[obj], cameras=[cam], images=[])
        anchor = scene.frame(position=cam.position, orientation=cam.orientation)
        assert anchor._hfov_deg is not None
        s_yes = float(anchor.first_person("left")[0])
        saved = anchor._hfov_deg
        anchor._hfov_deg = None
        try:
            s_no = float(anchor.first_person("left")[0])
        finally:
            anchor._hfov_deg = saved
        assert s_yes == pytest.approx(s_no, rel=1e-9), (
            "label 'left' must not depend on the FOV"
        )

    def test_empty_scene_no_crash(self):
        cam = _make_camera([0, 0, 0], [0, 0, 1])
        scene = Scene(objects=[], cameras=[cam], images=[])
        anchor = scene.frame(
            position=cam.position, orientation=cam.orientation
        )
        # No objects, no cameras to score against (well, cam itself is there).
        scores = anchor.first_person("front")
        # Should be a length-1 numpy array (the one camera), no exception.
        assert scores.shape == (1,)

    def test_score_in_valid_range(self):
        """Scores stay in [0, 1] for all labels and targets."""
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0, image_size=(480, 640))
        # 6 objects spread around the camera at various yaws.
        objs = [
            _make_object([0, 0, 5], obj_id=0),     # front
            _make_object([5, 0, 5], obj_id=1),     # front-right
            _make_object([5, 0, 0.01], obj_id=2),  # right
            _make_object([5, 0, -5], obj_id=3),    # back-right
            _make_object([0, 0, -5], obj_id=4),    # back
            _make_object([-5, 0, -5], obj_id=5),   # back-left
        ]
        scene = Scene(objects=objs, cameras=[cam], images=[])
        anchor = scene.frame(
            position=cam.position, orientation=cam.orientation
        )
        for label in (
            "front", "front-left", "front-right", "left", "right",
            "back", "back-left", "back-right",
        ):
            scores = anchor.first_person(label)
            assert np.all(scores >= 0.0), f"label={label!r} produced negative scores"
            assert np.all(scores <= 1.0 + 1e-9), (
                f"label={label!r} produced scores > 1"
            )


class TestFramesWithoutFov:
    """Code that does not pass ``hfov_deg=`` or a camera position gets frames
    without a FOV, scored by the plain cosine formula."""

    def test_object_anchor_has_no_fov(self):
        """Object-anchored frames carry no FOV."""
        cam = _make_camera([0, 0, 0], [0, 0, 1])
        obj0 = _make_object([1, 0, 2])
        obj1 = _make_object([3, 0, 1], obj_id=1)
        scene = Scene(objects=[obj0, obj1], cameras=[cam], images=[])

        anchor = scene.frame(
            position=scene.objects[0].center_world,
            orientation=np.eye(3),
        )
        # Anchor has no FOV.
        assert anchor._hfov_deg is None
        # Pin a few hand-computed values.
        # obj1 from obj0 origin: vec=(2,0,-1); for front (target=0):
        # yaw=atan2(2,-1) → score = (1+cos(yaw))/2
        manual = (1.0 + math.cos(math.atan2(2.0, -1.0))) / 2.0
        assert float(anchor.first_person("front")[1]) == pytest.approx(manual, rel=1e-9)

    def test_frame_rotate_preserves_hfov(self):
        """Derived frames (rotate/translate) must inherit the parent's hfov."""
        cam = _make_camera([0, 0, 0], [0, 0, 1], fx=500.0)
        scene = Scene(objects=[], cameras=[cam], images=[])
        anchor = scene.frame(position=cam.position, orientation=cam.orientation)
        original_fov = anchor._hfov_deg
        assert original_fov is not None
        rotated = anchor.rotate(yaw=30)
        translated = anchor.translate(np.array([0, 0, 1.0]))
        # Both derived frames should preserve the FOV.
        assert rotated._hfov_deg == pytest.approx(original_fov, rel=1e-9)
        assert translated._hfov_deg == pytest.approx(original_fov, rel=1e-9)


# ---- cardinal and relative labels ignore the FOV ----------------------------


def test_first_person_cardinal_ignores_fov():
    from test_match_methods import _make_object as _make_box_object, _scene_with_cameras_and_objects

    def at_yaw(deg, r=3.0):
        a = math.radians(deg)
        return (r * math.sin(a), 0, r * math.cos(a))

    scene = _scene_with_cameras_and_objects(
        [], [_make_box_object(0, at_yaw(85)), _make_box_object(1, at_yaw(25))]
    )
    scene.set_cardinal_vector(np.array(at_yaw(80, 1.0)))  # north 80 deg right of gaze
    axes = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0])
    plain = np.asarray(FrameNamespace(scene, *axes).first_person.north)
    camera = np.asarray(FrameNamespace(scene, *axes, hfov_deg=60.0).first_person.north)
    np.testing.assert_allclose(camera, plain)
    assert int(camera.argmax()) == 0
    # Relative labels score the same with and without the FOV.
    plain_fr = np.asarray(FrameNamespace(scene, *axes).first_person.front_right)
    camera_fr = np.asarray(FrameNamespace(scene, *axes, hfov_deg=60.0).first_person.front_right)
    np.testing.assert_allclose(camera_fr, plain_fr)
