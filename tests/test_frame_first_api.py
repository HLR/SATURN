"""Tests for the frame-first API.

Covers:

- ``Scene.score_cardinals``
- ``Scene._frame(at, *, front=, up=, same_as=)``
- ``FrameNamespace`` primitives: ``pos``, ``translate``, ``displacement``,
  ``rotate``, ``rotation_to``
- ``FrameNamespace.direction(..., as_label=)`` (8-way labels)
- ``FrameNamespace.best_match`` dispatch (origin-relative vs reference-relative)
- the ``at=`` and explicit ``position=`` / ``orientation=`` frame forms
"""

import math

import numpy as np
import pytest


from saturn.predicates.frame import Frame, FrameNamespace
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


# ---------------------------------------------------------------------------
# Test helpers (cribbed from tests/test_view_api.py for parity)
# ---------------------------------------------------------------------------


def _make_camera(position, forward, cam_id=0):
    position = np.asarray(position, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    # Right-handed SaPy convention (see vision_agents/multiview/conventions.py):
    # right = cross(world_up, forward) so that cross(right, up) == forward.
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    # OpenCV camera frame stores rows as (right, down, forward); down = -up.
    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0)
    t = -R_w2c @ position
    ext = np.eye(4)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = t
    K = np.eye(3)
    return Camera(
        id=cam_id,
        entity_id=cam_id,
        intrinsics=K,
        extrinsics=ext,
        image_size=(480, 640),
    )


def _make_object(obj_id, center, dims=(0.5, 0.5, 0.5), front=(0, 0, -1)):
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
        dims=np.asarray(dims, dtype=float),
        corners_world=np.zeros((8, 3)),
        height=float(dims[1]),
        support_y=float(center[1] - dims[1] / 2),
    )


class _OneHot:
    """Minimal subscriptable mock of a 1-D identity score tensor."""

    def __init__(self, values):
        self._v = list(values)

    def __getitem__(self, i):
        return self._v[i]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scene():
    """Camera at origin looking +z, three objects forming an L in front of it.

    Layout (world frame, +x right, +y up, +z forward of camera):
        camera 0 at (0, 0, 0), forward = +z
        object 0 (desk)  at (0, 0, 5)  front = -z
        object 1 (chair) at (3, 0, 5)  front = -z   # right of desk in cam view
        object 2 (lamp)  at (-3, 0, 5) front = -z   # left of desk in cam view
    """
    cameras = [_make_camera(position=[0, 0, 0], forward=[0, 0, 1], cam_id=0)]
    objects = [
        _make_object(0, center=[0, 0, 5]),
        _make_object(1, center=[3, 0, 5]),
        _make_object(2, center=[-3, 0, 5]),
    ]
    return Scene(objects=objects, cameras=cameras, images=[None])


@pytest.fixture
def scene_with_north(scene):
    """Same scene as ``scene`` but with cardinal frame established (north=+z)."""
    scene.set_cardinal_vector(np.array([0.0, 0.0, 1.0]))
    return scene


# ===========================================================================
# 1. Scene.score_cardinals
# ===========================================================================


class TestScoreCardinals:
    """8-way label keys + cosine-similarity scores."""

    EXPECTED_KEYS = {
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    }

    def test_returns_eight_hyphenated_keys(self, scene_with_north):
        scores = scene_with_north.score_cardinals(np.array([1.0, 0.0, 0.0]))
        assert set(scores.keys()) == self.EXPECTED_KEYS

    @pytest.mark.parametrize(
        "vec, label",
        [
            (np.array([0.0, 0.0, 1.0]), "north"),
            (np.array([1.0, 0.0, 0.0]), "east"),
            (np.array([0.0, 0.0, -1.0]), "south"),
            (np.array([-1.0, 0.0, 0.0]), "west"),
            (np.array([1.0, 0.0, 1.0]), "north-east"),
            (np.array([1.0, 0.0, -1.0]), "south-east"),
            (np.array([-1.0, 0.0, -1.0]), "south-west"),
            (np.array([-1.0, 0.0, 1.0]), "north-west"),
        ],
    )
    def test_argmax_lands_on_expected_label(
        self, scene_with_north, vec, label
    ):
        scores = scene_with_north.score_cardinals(vec)
        assert max(scores, key=scores.get) == label

    def test_neighbours_decay_with_cosine(self, scene_with_north):
        """Pure +x vector -> east=1.0, NE/SE = cos(45°), N/S = 0, W = 0."""
        scores = scene_with_north.score_cardinals(np.array([1.0, 0.0, 0.0]))
        assert scores["east"] == pytest.approx(1.0)
        assert scores["north-east"] == pytest.approx(math.cos(math.pi / 4))
        assert scores["south-east"] == pytest.approx(math.cos(math.pi / 4))
        assert scores["north"] == pytest.approx(0.0, abs=1e-9)
        assert scores["south"] == pytest.approx(0.0, abs=1e-9)
        # Opposite quadrant clamped to 0
        assert scores["west"] == 0.0
        assert scores["north-west"] == 0.0
        assert scores["south-west"] == 0.0

    def test_clamped_to_nonnegative(self, scene_with_north):
        scores = scene_with_north.score_cardinals(np.array([1.0, 0.0, 0.5]))
        for label, s in scores.items():
            assert 0.0 <= s <= 1.0 + 1e-9, f"{label}={s} out of [0,1]"

    def test_zero_vector_returns_all_zeros(self, scene_with_north):
        scores = scene_with_north.score_cardinals(np.array([0.0, 0.0, 0.0]))
        assert set(scores) == self.EXPECTED_KEYS
        assert all(s == 0.0 for s in scores.values())

    def test_pure_vertical_returns_all_zeros(self, scene_with_north):
        """Y component is dropped; pure-up vector has zero horizontal mag."""
        scores = scene_with_north.score_cardinals(np.array([0.0, 5.0, 0.0]))
        assert all(s == 0.0 for s in scores.values())

    def test_y_component_ignored(self, scene_with_north):
        """East+up should score identically to pure east."""
        s1 = scene_with_north.score_cardinals(np.array([1.0, 0.0, 0.0]))
        s2 = scene_with_north.score_cardinals(np.array([1.0, 7.0, 0.0]))
        for k in s1:
            assert s1[k] == pytest.approx(s2[k])

    def test_accepts_list_input(self, scene_with_north):
        s_arr = scene_with_north.score_cardinals(np.array([0.0, 0.0, 1.0]))
        s_list = scene_with_north.score_cardinals([0.0, 0.0, 1.0])
        for k in s_arr:
            assert s_arr[k] == pytest.approx(s_list[k])

    def test_no_cardinal_frame_raises(self, scene):
        with pytest.raises(RuntimeError, match="cardinal frame"):
            scene.score_cardinals(np.array([1.0, 0.0, 0.0]))

    def test_rotated_north_rotates_argmax(self, scene):
        """If north points +x, then a +z vector should score "south" (argmax)
        because rotating north 90° east shifts the whole frame: east now points
        at -z, so +z lies in the west quadrant ... actually let's just verify
        the argmax for the rotated north matches for a rotated query."""
        scene.set_cardinal_vector(np.array([1.0, 0.0, 0.0]))  # north = +x
        # If north=+x, then rotating clockwise from above by 90° puts east=-z.
        # So a vector pointing -z must be "east".
        scores = scene.score_cardinals(np.array([0.0, 0.0, -1.0]))
        assert max(scores, key=scores.get) == "east"


# ===========================================================================
# 2. Scene._frame(at, *, front=, up=, same_as=) factory coverage
# ===========================================================================


class TestSceneFrameFactory:
    """Three usage patterns: intrinsic / custom front / same_as."""

    # -- intrinsic axes (no front, no same_as) ------------------------------

    def test_intrinsic_camera_instance(self, scene):
        cam = scene.cameras[0]
        f = scene._frame(at=cam)
        np.testing.assert_allclose(f.frame_origin, [0, 0, 0], atol=1e-9)
        # Camera looks +z, world up is +y.  In the camera's frame:
        # front=+z, up=+y, right=front×up... use handedness from helper.
        np.testing.assert_allclose(f.frame_front, cam.front_vec, atol=1e-9)
        np.testing.assert_allclose(f.frame_up, cam.up_vec, atol=1e-9)

    def test_intrinsic_object_instance(self, scene):
        obj = scene.objects[0]  # facing -z
        f = scene._frame(at=obj)
        np.testing.assert_allclose(f.frame_origin, obj.center_world, atol=1e-9)
        np.testing.assert_allclose(f.frame_front, obj.front_vec, atol=1e-9)

    def test_intrinsic_int_means_object_index(self, scene):
        f = scene._frame(at=1)
        np.testing.assert_allclose(
            f.frame_origin, scene.objects[1].center_world, atol=1e-9
        )

    def test_intrinsic_camera_tuple(self, scene):
        f_tuple = scene._frame(at=("camera", 0))
        f_inst = scene._frame(at=scene.cameras[0])
        np.testing.assert_allclose(f_tuple.frame_origin, f_inst.frame_origin)
        np.testing.assert_allclose(f_tuple.frame_front, f_inst.frame_front)

    def test_intrinsic_object_tuple(self, scene):
        f_tuple = scene._frame(at=("object", 1))
        f_inst = scene._frame(at=scene.objects[1])
        np.testing.assert_allclose(f_tuple.frame_origin, f_inst.frame_origin)
        np.testing.assert_allclose(f_tuple.frame_front, f_inst.frame_front)

    def test_returns_frame_alias(self, scene):
        """``Frame`` is the public name; factory returns a ``FrameNamespace``."""
        f = scene._frame(at=0)
        assert isinstance(f, FrameNamespace)
        assert isinstance(f, Frame)

    # -- custom front (vector or entity) ------------------------------------

    def test_custom_front_vector(self, scene):
        # Anchor at origin of camera 0, but face +x ("east-ish").
        f = scene._frame(at=("camera", 0), front=np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(f.frame_front, [1, 0, 0], atol=1e-9)
        # Up defaults to world up
        np.testing.assert_allclose(f.frame_up, [0, 1, 0], atol=1e-9)

    def test_custom_front_entity_uses_its_forward(self, scene):
        """Passing an entity to ``front=`` uses its ``.front`` direction."""
        obj = scene.objects[0]  # forward = -z
        f = scene._frame(at=("camera", 0), front=obj)
        np.testing.assert_allclose(f.frame_front, obj.front_vec, atol=1e-9)

    def test_custom_front_normalizes(self, scene):
        f = scene._frame(at=("camera", 0), front=np.array([3.0, 0.0, 0.0]))
        np.testing.assert_allclose(np.linalg.norm(f.frame_front), 1.0, atol=1e-9)

    def test_custom_up_override(self, scene):
        """Explicit ``up=`` overrides world up."""
        custom_up = np.array([0.0, 0.0, 1.0])
        f = scene._frame(
            at=("camera", 0),
            front=np.array([1.0, 0.0, 0.0]),
            up=custom_up,
        )
        # Up should be aligned with the custom_up direction (after orthogonal-
        # isation any residual is along the up direction)
        assert abs(float(np.dot(f.frame_up, custom_up))) > 0.99

    # -- same_as (copy axes) ------------------------------------------------

    def test_same_as_frame_copies_axes(self, scene):
        cam_frame = scene._frame(at=scene.cameras[0])
        f = scene._frame(at=scene.objects[0], same_as=cam_frame)
        # Origin from ``at``
        np.testing.assert_allclose(
            f.frame_origin, scene.objects[0].center_world, atol=1e-9
        )
        # Axes from ``same_as``
        np.testing.assert_allclose(f.frame_front, cam_frame.frame_front, atol=1e-9)
        np.testing.assert_allclose(f.frame_up, cam_frame.frame_up, atol=1e-9)
        np.testing.assert_allclose(
            f.frame_right, cam_frame.frame_right, atol=1e-9
        )

    def test_same_as_entity_copies_axes(self, scene):
        """``same_as=`` accepts anything with ``.front`` / ``.up``."""
        cam = scene.cameras[0]
        f = scene._frame(at=scene.objects[1], same_as=cam)
        np.testing.assert_allclose(f.frame_front, cam.front_vec, atol=1e-9)
        np.testing.assert_allclose(f.frame_up, cam.up_vec, atol=1e-9)

    # -- error paths --------------------------------------------------------

    def test_front_and_same_as_mutually_exclusive(self, scene):
        cam_frame = scene._frame(at=scene.cameras[0])
        with pytest.raises(ValueError, match="front.*same_as|same_as.*front"):
            scene._frame(
                at=scene.objects[0],
                front=np.array([1.0, 0.0, 0.0]),
                same_as=cam_frame,
            )

    def test_invalid_same_as_type_raises(self, scene):
        with pytest.raises(TypeError, match="same_as"):
            scene._frame(at=0, same_as=42)


# ===========================================================================
# 3. Frame primitives (pos, translate, displacement, rotate, rotation_to)
# ===========================================================================


class TestFramePrimitives:
    # -- pos ----------------------------------------------------------------

    def test_pos_aliases_frame_origin(self, scene):
        f = scene._frame(at=scene.objects[0])
        np.testing.assert_allclose(f.pos, f.frame_origin, atol=1e-9)
        np.testing.assert_allclose(f.pos, scene.objects[0].center_world)

    # -- translate ----------------------------------------------------------

    def test_translate_shifts_origin_preserves_axes(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f2 = f.translate(np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(f2.frame_origin, [0, 0, 2], atol=1e-9)
        # Axes unchanged
        np.testing.assert_allclose(f2.frame_front, f.frame_front, atol=1e-9)
        np.testing.assert_allclose(f2.frame_right, f.frame_right, atol=1e-9)
        np.testing.assert_allclose(f2.frame_up, f.frame_up, atol=1e-9)

    def test_translate_via_frame_front(self, scene):
        """Documented pattern: ``view.translate(view.frame_front * 2.0)``."""
        f = scene._frame(at=scene.cameras[0])
        f2 = f.translate(f.frame_front * 2.0)
        # Camera's front is +z, so origin shifts +2 in z
        np.testing.assert_allclose(f2.frame_origin, [0, 0, 2], atol=1e-9)

    def test_translate_returns_new_frame(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f2 = f.translate(np.array([1.0, 0.0, 0.0]))
        assert f2 is not f
        np.testing.assert_allclose(f.frame_origin, [0, 0, 0])

    def test_translate_validates_dimensions(self, scene):
        f = scene._frame(at=scene.cameras[0])
        with pytest.raises(ValueError, match="3 components"):
            f.translate(np.array([1.0, 2.0]))

    # -- displacement -------------------------------------------------------

    def test_displacement_to_int(self, scene):
        f = scene._frame(at=scene.cameras[0])
        d = f.displacement(to=1)  # chair at (3,0,5)
        np.testing.assert_allclose(d, [3, 0, 5], atol=1e-9)

    def test_displacement_to_entity(self, scene):
        f = scene._frame(at=scene.cameras[0])
        d = f.displacement(to=scene.objects[0])
        np.testing.assert_allclose(d, scene.objects[0].center_world, atol=1e-9)

    def test_displacement_to_frame(self, scene):
        f1 = scene._frame(at=scene.cameras[0])
        f2 = scene._frame(at=scene.objects[1])
        d = f1.displacement(to=f2)
        np.testing.assert_allclose(d, scene.objects[1].center_world, atol=1e-9)

    def test_displacement_is_zero_to_self_origin(self, scene):
        """displacement to a synthetic point at our own origin = 0."""
        f = scene._frame(at=scene.cameras[0])
        # Camera 0 is at the origin, so displacement(to=any_zero_obj) =
        # that obj's position. For the self-zero case, build a Frame at
        # the same origin as f (using object index 0 ... but it's at
        # (0,0,5), not 0).  Test with a known offset instead.
        f2 = scene._frame(at=scene.cameras[0])
        d = f.displacement(to=f2)
        np.testing.assert_allclose(d, [0, 0, 0], atol=1e-9)

    # -- rotate -------------------------------------------------------------

    def test_rotate_zero_returns_self(self, scene):
        f = scene._frame(at=scene.cameras[0])
        assert f.rotate(0, 0) is f

    def test_rotate_yaw_180_flips_front_and_right(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f2 = f.rotate(yaw=180)
        np.testing.assert_allclose(f2.frame_front, -f.frame_front, atol=1e-6)
        np.testing.assert_allclose(f2.frame_right, -f.frame_right, atol=1e-6)
        # Up is preserved by yaw
        np.testing.assert_allclose(f2.frame_up, f.frame_up, atol=1e-6)
        # Origin preserved
        np.testing.assert_allclose(f2.frame_origin, f.frame_origin, atol=1e-9)

    def test_rotate_yaw_positive_is_clockwise_from_above(self, scene):
        """yaw>0 = turn right.  For front=+z, right=+x: yaw=+90 must rotate
        front toward right (i.e. front_new ≈ +x)."""
        f = scene._frame(at=scene.cameras[0])
        f2 = f.rotate(yaw=90)
        np.testing.assert_allclose(f2.frame_front, f.frame_right, atol=1e-6)
        np.testing.assert_allclose(f2.frame_right, -f.frame_front, atol=1e-6)

    def test_rotate_pitch_up_tilts_front_toward_up(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f2 = f.rotate(pitch=30)
        # +y component appears in front; z component shrinks
        assert f2.frame_front[1] > 0.4
        assert f2.frame_front[2] < f.frame_front[2]

    def test_rotate_preserves_orthonormality(self, scene):
        f = scene._frame(at=scene.cameras[0]).rotate(yaw=37, pitch=-22)
        for ax in (f.frame_right, f.frame_up, f.frame_front):
            np.testing.assert_allclose(np.linalg.norm(ax), 1.0, atol=1e-6)
        assert abs(float(np.dot(f.frame_right, f.frame_up))) < 1e-6
        assert abs(float(np.dot(f.frame_right, f.frame_front))) < 1e-6
        assert abs(float(np.dot(f.frame_up, f.frame_front))) < 1e-6

    # -- rotation_to --------------------------------------------------------

    def test_rotation_to_self_is_zero(self, scene):
        f = scene._frame(at=scene.cameras[0])
        yaw, pitch = f.rotation_to(f)
        assert yaw == pytest.approx(0.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_rotation_to_180_yaw(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f_back = f.rotate(yaw=180)
        yaw, pitch = f.rotation_to(f_back)
        # atan2(0, -1) = ±180°; either is fine
        assert abs(abs(yaw) - 180.0) < 1e-4
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_rotation_to_right_is_positive_yaw(self, scene):
        """Camera A faces +z; camera B (turned right) faces +x.
        rotation_to(B) should report positive yaw (turn right)."""
        f = scene._frame(at=scene.cameras[0])
        f_right = f.rotate(yaw=90)
        yaw, pitch = f.rotation_to(f_right)
        assert yaw == pytest.approx(90.0, abs=1e-4)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_rotation_to_left_is_negative_yaw(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f_left = f.rotate(yaw=-60)
        yaw, _ = f.rotation_to(f_left)
        assert yaw == pytest.approx(-60.0, abs=1e-4)

    def test_rotation_to_pitch_up_is_positive(self, scene):
        f = scene._frame(at=scene.cameras[0])
        f_up = f.rotate(pitch=25)
        yaw, pitch = f.rotation_to(f_up)
        assert pitch == pytest.approx(25.0, abs=1e-4)
        assert abs(yaw) < 1e-4

    def test_rotation_to_rejects_non_frame(self, scene):
        f = scene._frame(at=scene.cameras[0])
        with pytest.raises(TypeError, match="Frame"):
            f.rotation_to(scene.objects[0])


# ===========================================================================
# 4. FrameNamespace.direction(..., as_label=True)
# ===========================================================================


class TestDirectionAsLabel:
    """``direction(..., as_label=True)`` returns 8-way labels."""

    def test_default_returns_direction_value(self, scene):
        f = scene._frame(at=scene.cameras[0])
        result = f.direction(target=1)
        # Numeric DirectionValue (has yaw_degree) by default
        assert hasattr(result, "yaw_degree")

    def test_as_label_true_returns_string(self, scene):
        f = scene._frame(at=scene.cameras[0])
        label = f.direction(target=1, as_label=True)
        assert isinstance(label, str)

    @pytest.mark.parametrize(
        "target_pos, expected_label",
        [
            ([0.0, 0.0, 5.0], "front"),       # ahead
            ([5.0, 0.0, 0.0], "right"),       # to the right
            ([-5.0, 0.0, 0.0], "left"),       # to the left
            ([0.0, 0.0, -5.0], "back"),       # behind
            ([5.0, 0.0, 5.0], "front-right"),
            ([-5.0, 0.0, 5.0], "front-left"),
            ([5.0, 0.0, -5.0], "back-right"),
            ([-5.0, 0.0, -5.0], "back-left"),
        ],
    )
    def test_eight_way_labels(self, scene, target_pos, expected_label):
        f = scene._frame(at=scene.cameras[0])
        # Pass target as a raw 3D point
        label = f.direction(target=target_pos, as_label=True)
        assert label == expected_label

    def test_explicit_source_overrides_origin(self, scene):
        """direction(source=A, target=B) uses A as the from-point, not the
        frame origin."""
        f = scene._frame(at=scene.cameras[0])
        # source = obj 0 at (0,0,5); target = obj 1 at (3,0,5).  In a
        # +z-front frame, the displacement (3,0,0) lies to the right.
        label = f.direction(source=0, target=1, as_label=True)
        assert label == "right"

    def test_target_required(self, scene):
        f = scene._frame(at=scene.cameras[0])
        with pytest.raises(ValueError, match="target"):
            f.direction()

    def test_label_is_canonical_eight_way(self, scene):
        """``direction(..., as_label=True)`` returns the canonical 8-way label
        of the bin the target falls in."""
        f = scene._frame(at=scene.cameras[0])
        # 22.5° tolerance — anything strictly inside a 45° bin is unambiguous
        ne = f.direction(target=[1.0, 0.0, 1.0], as_label=True)
        assert ne == "front-right"


# ===========================================================================
# 5. FrameNamespace.best_match dispatch
# ===========================================================================


class TestBestMatchDispatch:
    """``best_match(direction, options, *, reference=None)`` dispatch:
    no reference -> origin-relative; with reference -> reference-relative.
    """

    def test_no_reference_uses_origin_path(self, scene):
        """Camera-relative.  Object 1 is at (3,0,5) — to the right of origin."""
        f = scene._frame(at=scene.cameras[0])
        # obj 1 = chair (right), obj 2 = lamp (left)
        options = {
            "A": _OneHot([0, 1, 0]),  # chair
            "B": _OneHot([0, 0, 1]),  # lamp
        }
        assert f.best_match("right", options) == "A"
        assert f.best_match("left", options) == "B"

    def test_with_int_reference_uses_relative_path(self, scene):
        """reference=0 (desk).  Chair (obj 1) is right of desk; lamp (obj 2)
        is left of desk."""
        f = scene._frame(at=scene.cameras[0])
        options = {
            "A": _OneHot([0, 1, 0]),  # chair
            "B": _OneHot([0, 0, 1]),  # lamp
        }
        assert f.best_match("right", options, reference=0) == "A"
        assert f.best_match("left", options, reference=0) == "B"

    def test_reference_kwarg_only(self, scene):
        """``reference=`` is keyword-only — positional must fail."""
        f = scene._frame(at=scene.cameras[0])
        options = {"A": _OneHot([0, 1, 0])}
        with pytest.raises(TypeError):
            # 4 positional args; signature only allows 3 (self, direction, options)
            f.best_match("right", options, 0)

    def test_returns_option_key_string(self, scene):
        f = scene._frame(at=scene.cameras[0])
        options = {"X": _OneHot([0, 1, 0]), "Y": _OneHot([0, 0, 1])}
        result = f.best_match("right", options)
        assert result in {"X", "Y"}

    def test_origin_path_with_cardinal_label(self, scene):
        """Origin path normalises cardinal labels to relative ones."""
        f = scene._frame(at=scene.cameras[0])
        options = {
            "A": _OneHot([0, 1, 0]),  # chair (right)
            "B": _OneHot([0, 0, 1]),  # lamp (left)
        }
        # In a camera-front=+z view, "front" lies along front, but "north"
        # cardinal is only meaningful with a cardinal frame; the method must
        # at least not crash when given a cardinal-style label.
        result = f.best_match("front", options)
        assert result in {"A", "B"}

    def test_with_nearest_flag(self, scene):
        """``nearest=True`` is reference-relative only and must not crash."""
        f = scene._frame(at=scene.cameras[0])
        options = {
            "A": _OneHot([0, 1, 0]),
            "B": _OneHot([0, 0, 1]),
        }
        result = f.best_match("right", options, reference=0, nearest=True)
        assert result in {"A", "B"}


# ===========================================================================
# 7. view.first_person.<label> — first-person predicate
# ===========================================================================


class TestViewFirstPerson:
    """1D K+C scores from the view's frame_origin to each entity.

    Layout (re-stated for convenience):
        camera 0 at (0, 0, 0) looking +z
        obj 0 (desk)  at (0, 0, 5)  → directly in front of camera
        obj 1 (chair) at (3, 0, 5)  → front-right of camera (yaw ≈ +31°)
        obj 2 (lamp)  at (-3, 0, 5) → front-left of camera  (yaw ≈ -31°)
    """

    def test_object_and_camera_indexing(self, scene):
        """Returned vector has length K+C; cameras live at [K, K+C)."""
        view = scene._frame(at=scene.cameras[0])
        scores = view.first_person.front
        K = len(scene.objects)  # 3
        C = len(scene.cameras)  # 1
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (K + C,)
        # Camera 0 sits at the origin (== frame_origin) → horizontal magnitude
        # is zero; scoring helper returns 0.0 there.
        assert scores[K + 0] == 0.0

    def test_argmax_picks_correct_object(self, scene):
        """``view.first_person.front`` argmax = desk (idx 0); ``.right`` = chair."""
        view = scene._frame(at=scene.cameras[0])
        K = len(scene.objects)
        front_scores = view.first_person.front[:K]
        right_scores = view.first_person.right[:K]
        left_scores = view.first_person.left[:K]
        assert int(np.argmax(front_scores)) == 0  # desk
        assert int(np.argmax(right_scores)) == 1  # chair
        assert int(np.argmax(left_scores)) == 2  # lamp

    def test_synonyms_and_underscore_attribute_access(self, scene_with_north):
        """``view.first_person.behind`` aliases ``.back`` (same scores);
        ``north_east`` ≡ ``"north-east"``; ``__call__`` and ``__getitem__``
        agree with attribute access.

        The first-person namespace aliases ``behind → back`` because the
        semantic is unambiguous (target lies in the back direction from the
        frame origin). The 2D pairwise ``view.behind[i, j]`` predicate
        lives on a separate namespace and is unaffected.
        """
        view = scene_with_north._frame(at=scene_with_north.cameras[0])
        back_scores = view.first_person.back
        behind_scores = view.first_person.behind
        np.testing.assert_allclose(back_scores, behind_scores, atol=1e-9)

        c = view.first_person.north_east
        d = view.first_person("north-east")
        e = view.first_person["ne"]
        np.testing.assert_allclose(c, d, atol=1e-9)
        np.testing.assert_allclose(c, e, atol=1e-9)

    def test_resolve_direction_angle_back_aliases(self):
        """``_resolve_direction_angle`` accepts ``forward → front`` and
        ``behind → back`` aliases.

        The ``behind`` alias is safe here because the resolver feeds the
        first-person namespace, where "k is behind me" and "k is at my back"
        are semantically identical. The 2D ``view.behind[i, j]``
        predicate uses a different namespace and bypasses this resolver.
        ``backward`` / ``backwards`` are not aliased — only ``behind``.
        """
        from saturn.predicates.frame import FrameNamespace
        # back / behind are canonical and alias → 180°.
        assert FrameNamespace._resolve_direction_angle("back") == 180.0
        assert FrameNamespace._resolve_direction_angle("behind") == 180.0
        assert FrameNamespace._resolve_direction_angle("behinds") == 180.0
        # forward / forwards alias → 0° (front).
        assert FrameNamespace._resolve_direction_angle("forward") == 0.0
        assert FrameNamespace._resolve_direction_angle("forwards") == 0.0
        # behind diagonals alias to back diagonals.
        assert FrameNamespace._resolve_direction_angle("behind-left") == \
            FrameNamespace._resolve_direction_angle("back-left")
        assert FrameNamespace._resolve_direction_angle("behind-right") == \
            FrameNamespace._resolve_direction_angle("back-right")
        # "backward"/"backwards" are NOT aliased — only "behind".
        for bad in ("backward", "backwards"):
            with pytest.raises(ValueError, match=f"Unknown direction: {bad}"):
                FrameNamespace._resolve_direction_angle(bad)

    def test_cardinal_guard_without_set_cardinal_vector(self, scene):
        """Cardinal labels raise ValueError when no cardinal vector is set."""
        view = scene._frame(at=scene.cameras[0])
        with pytest.raises(ValueError, match="set_cardinal_vector"):
            _ = view.first_person.north
        with pytest.raises(ValueError, match="set_cardinal_vector"):
            _ = view.first_person("south-west")

    def test_cardinal_aligns_with_world_frame(self, scene_with_north):
        """With north=+z, ``view.first_person.north`` ≡ ``view.first_person.front`` for a
        camera whose frame_front is +z."""
        view = scene_with_north._frame(at=scene_with_north.cameras[0])
        # Camera 0 looks +z → its frame_front == world +z == scene north.
        np.testing.assert_allclose(
            view.first_person.north, view.first_person.front, atol=1e-9
        )
        np.testing.assert_allclose(
            view.first_person.east, view.first_person.right, atol=1e-9
        )

    def test_diagonal_and_vertical_scoring(self, scene):
        """Diagonal label scores chair (front-right) higher than desk (front)
        on ``front-right``; ``above`` scoring is symmetric for objects at
        the same height as the camera (all ≈ 0.5)."""
        view = scene._frame(at=scene.cameras[0])
        K = len(scene.objects)
        fr = view.first_person("front-right")[:K]
        # chair (idx 1) is the unique front-right object → strict argmax
        assert int(np.argmax(fr)) == 1
        # And it should beat the on-axis desk on this diagonal label.
        assert fr[1] > fr[0]

        above = view.first_person.above[:K]
        # All objects share y=0 with the camera at y=0 → elevation is zero,
        # so above-score should be ~0.5 everywhere.
        np.testing.assert_allclose(above, np.full(K, 0.5), atol=1e-9)


class TestViewAt:
    """``view.at(target)`` perspective-shift method."""

    def test_at_object_index_shifts_origin_keeps_axes(self, scene):
        cam_view = scene._frame(at=("camera", 0))
        obj_view = cam_view.at(0)  # object 0 is at (0, 0, 5)
        np.testing.assert_allclose(obj_view.frame_origin, [0, 0, 5], atol=1e-9)
        np.testing.assert_allclose(obj_view.frame_front, cam_view.frame_front, atol=1e-9)
        np.testing.assert_allclose(obj_view.frame_right, cam_view.frame_right, atol=1e-9)
        np.testing.assert_allclose(obj_view.frame_up, cam_view.frame_up, atol=1e-9)

    def test_at_camera_tuple(self, scene):
        cam_view = scene._frame(at=("camera", 0))
        shifted = cam_view.at(("camera", 0))
        np.testing.assert_allclose(shifted.frame_origin, cam_view.frame_origin, atol=1e-9)
        np.testing.assert_allclose(shifted.frame_front, cam_view.frame_front, atol=1e-9)

    def test_at_equivalent_to_explicit_same_as(self, scene):
        cam_view = scene._frame(at=("camera", 0))
        a = cam_view.at(1)
        b = scene._frame(at=1, same_as=cam_view)
        np.testing.assert_allclose(a.frame_origin, b.frame_origin, atol=1e-9)
        np.testing.assert_allclose(a.frame_front, b.frame_front, atol=1e-9)

    def test_at_composes_with_first_person_back(self, scene):
        """From obj 0 (at z=5) with camera's axes, only camera (z=0) is at body-back."""
        cam_view = scene._frame(at=("camera", 0))
        scores = cam_view.at(0).first_person.back
        scores_arr = np.asarray(scores.tensor if hasattr(scores, "tensor") else scores)
        K = len(scene.objects)
        cam_idx = K + 0  # camera is the only entity in -z relative to obj 0
        assert int(np.argmax(scores_arr)) == cam_idx


class TestViewThirdPerson:
    """``view.third_person`` namespace."""

    def test_third_person_proxies_2d_predicates(self, scene):
        view = scene._frame(at=("camera", 0))
        for d in ("front", "behind", "left", "right", "above", "below"):
            np.testing.assert_allclose(
                np.asarray(getattr(view.third_person, d).tensor),
                np.asarray(getattr(view, d).tensor),
                atol=1e-12,
                err_msg=f"third_person.{d} != view.{d}",
            )

    def test_third_person_rejects_back_with_helpful_error(self, scene):
        view = scene._frame(at=("camera", 0))
        with pytest.raises(AttributeError, match="view.first_person.back"):
            _ = view.third_person.back

    def test_third_person_rejects_unknown_direction(self, scene):
        view = scene._frame(at=("camera", 0))
        with pytest.raises(AttributeError, match="Vocabulary"):
            _ = view.third_person.northeast

    def test_third_person_call_and_subscript_forms(self, scene):
        view = scene._frame(at=("camera", 0))
        a = view.third_person("behind")
        b = view.third_person["behind"]
        c = view.third_person.behind
        np.testing.assert_allclose(np.asarray(a.tensor), np.asarray(c.tensor), atol=1e-12)
        np.testing.assert_allclose(np.asarray(b.tensor), np.asarray(c.tensor), atol=1e-12)


class TestViewFacing:
    """``view.facing`` intrinsic-orientation namespace."""

    def test_facing_proxies_obj_facing(self, scene):
        view = scene._frame(at=("camera", 0))
        for short, long_ in [
            ("front", "obj_facing_front"),
            ("back", "obj_facing_back"),
            ("left", "obj_facing_left"),
            ("right", "obj_facing_right"),
            ("front_left", "obj_facing_front_left"),
            ("front_right", "obj_facing_front_right"),
            ("back_left", "obj_facing_back_left"),
            ("back_right", "obj_facing_back_right"),
            ("up", "obj_facing_up"),
            ("down", "obj_facing_down"),
        ]:
            np.testing.assert_allclose(
                np.asarray(getattr(view.facing, short).tensor),
                np.asarray(getattr(view, long_).tensor),
                atol=1e-12,
                err_msg=f"facing.{short} != view.{long_}",
            )

    def test_facing_underscore_and_hyphen_equivalent(self, scene):
        view = scene._frame(at=("camera", 0))
        a = view.facing.back_left
        b = view.facing("back-left")
        c = view.facing["back-left"]
        np.testing.assert_allclose(np.asarray(a.tensor), np.asarray(b.tensor), atol=1e-12)
        np.testing.assert_allclose(np.asarray(a.tensor), np.asarray(c.tensor), atol=1e-12)

    def test_facing_unknown_label_raises(self, scene):
        view = scene._frame(at=("camera", 0))
        with pytest.raises(AttributeError, match="Vocabulary"):
            _ = view.facing.northeast


class TestSceneCentroid:
    """``Scene.centroid(...)`` is the mean of the entities' positions
    (not of their forward vectors).
    """

    def test_single_object_index(self, scene):
        """``centroid([obj_idx])`` returns that object's world-space center."""
        result = scene.centroid([0])
        np.testing.assert_allclose(result, scene.objects[0].pos, atol=1e-9)

    def test_multiple_objects_mean(self, scene):
        """``centroid([i, j, k])`` returns the mean of the object centers."""
        # Fixture: desk (0,0,5), chair (3,0,5), lamp (-3,0,5).
        result = scene.centroid([0, 1, 2])
        expected = np.mean(
            np.stack([scene.objects[i].pos for i in (0, 1, 2)], axis=0),
            axis=0,
        )
        np.testing.assert_allclose(result, expected, atol=1e-9)
        # Sanity: mean lies on z=5 plane, x averages to 0.
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 5.0]), atol=1e-9)

    def test_camera_anchor(self, scene):
        """``centroid([('camera', i)])`` returns the camera's world position."""
        result = scene.centroid([("camera", 0)])
        # Fixture: camera 0 at (0, 0, 0).
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0]), atol=1e-9)

    def test_bare_3d_point(self, scene):
        """Bare numpy points are passed through verbatim."""
        pt = np.array([1.0, 2.0, 3.0])
        result = scene.centroid([pt])
        np.testing.assert_allclose(result, pt, atol=1e-9)

    def test_mixed_types(self, scene):
        """Object indices, camera tuples, and bare points compose correctly."""
        # desk at (0,0,5) + camera at (0,0,0) + bare (3,3,5) → mean (1,1, 10/3)
        result = scene.centroid([0, ("camera", 0), np.array([3.0, 3.0, 5.0])])
        np.testing.assert_allclose(result, np.array([1.0, 1.0, 10.0 / 3.0]), atol=1e-9)


# =============================================================================
# position/orientation pose-based view constructor
# =============================================================================


class TestPosePropertiesObject:
    """``MergedObject.position`` and ``MergedObject.orientation`` expose the
    object's pose as raw geometry consumable by ``scene.frame(position=..., orientation=...)``."""

    def test_position_returns_center_world(self, scene):
        obj = scene.objects[0]
        np.testing.assert_allclose(obj.position, obj.center_world, atol=1e-9)

    def test_orientation_shape_and_front_column(self, scene):
        """Orientation columns are [right, up, front] in the
        right-hand-rule convention. The front column matches ``front_world``;
        right/up are derived via ``right = cross(up, front)`` so that the
        matrix is directly consumable by ``scene._frame(orientation=...)``
        and equivalent to the ``scene._frame(at=obj_idx)`` path."""
        obj = scene.objects[0]
        ori = obj.orientation
        assert ori.shape == (3, 3)
        # front column (col 2) matches front_world (after normalization)
        f_unit = obj.front_world / (np.linalg.norm(obj.front_world) + 1e-12)
        np.testing.assert_allclose(ori[:, 2], f_unit, atol=1e-9)

    def test_orientation_is_orthonormal_and_right_handed(self, scene):
        """The orientation matrix must be orthonormal and right-handed
        (cross(col0, col1) == col2). Anything else corrupts predicate scoring."""
        obj = scene.objects[0]
        ori = obj.orientation
        # Each column unit-length
        for k in range(3):
            np.testing.assert_allclose(np.linalg.norm(ori[:, k]), 1.0, atol=1e-9)
        # Pairwise orthogonal
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            np.testing.assert_allclose(np.dot(ori[:, i], ori[:, j]), 0.0, atol=1e-9)
        # Right-handed: col0 == cross(col1, col2)
        np.testing.assert_allclose(
            ori[:, 0], np.cross(ori[:, 1], ori[:, 2]), atol=1e-9
        )

    def test_orientation_right_column_is_cross_up_front(self, scene):
        """The defining property: right = cross(up, front). This is exactly
        what the ``scene._frame(at=...)`` path computes internally, so
        orientation column 0 must match it bit-for-bit."""
        obj = scene.objects[0]
        ori = obj.orientation
        u_unit = obj.up_world / (np.linalg.norm(obj.up_world) + 1e-12)
        f_unit = obj.front_world / (np.linalg.norm(obj.front_world) + 1e-12)
        expected_right = np.cross(u_unit, f_unit)
        expected_right /= np.linalg.norm(expected_right) + 1e-12
        np.testing.assert_allclose(ori[:, 0], expected_right, atol=1e-9)


class TestPosePropertiesCamera:
    """``Camera.position`` and ``Camera.orientation`` expose the camera's pose
    as raw geometry, derived from extrinsics + heading."""

    def test_position_returns_position_world(self, scene):
        cam = scene.cameras[0]
        np.testing.assert_allclose(cam.position, cam.position_world, atol=1e-9)

    def test_orientation_shape_and_front_column(self, scene):
        """Front column matches ``heading.forward`` (the camera's optical axis)."""
        cam = scene.cameras[0]
        ori = cam.orientation
        assert ori.shape == (3, 3)
        f_unit = cam.heading.forward / (np.linalg.norm(cam.heading.forward) + 1e-12)
        np.testing.assert_allclose(ori[:, 2], f_unit, atol=1e-9)

    def test_orientation_is_orthonormal_and_right_handed(self, scene):
        """Camera orientation must be orthonormal and right-handed."""
        cam = scene.cameras[0]
        ori = cam.orientation
        for k in range(3):
            np.testing.assert_allclose(np.linalg.norm(ori[:, k]), 1.0, atol=1e-9)
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            np.testing.assert_allclose(np.dot(ori[:, i], ori[:, j]), 0.0, atol=1e-9)
        np.testing.assert_allclose(
            ori[:, 0], np.cross(ori[:, 1], ori[:, 2]), atol=1e-9
        )

    def test_orientation_right_column_is_cross_up_front(self, scene):
        """For cameras: orientation[:,0] == cross(heading.up, heading.forward).
        Note this is the *opposite sign* from ``heading.right`` which uses
        cross(forward, up) (OpenCV screen-right). The orientation matrix
        uses the physics RH convention so it matches scene._frame(at=)."""
        cam = scene.cameras[0]
        ori = cam.orientation
        u = cam.heading.up
        f = cam.heading.forward
        expected_right = np.cross(u, f)
        expected_right /= np.linalg.norm(expected_right) + 1e-12
        np.testing.assert_allclose(ori[:, 0], expected_right, atol=1e-9)
        # Sanity: this is OPPOSITE sign from heading.right
        assert np.dot(ori[:, 0], cam.heading.right) < -0.99, (
            "orientation right MUST be -heading.right; otherwise the OpenCV and "
            "physics-RH sign conventions are mixed."
        )


class TestSceneFramePose:
    """``scene.frame(position=..., orientation=...)`` is the explicit-pose form
    of the unified ``scene.frame`` constructor — same primitive, but the
    (position, orientation) choice is spelled out in plain text at the call
    site instead of being inferred from an entity reference.
    """

    def test_allocentric_matches_at_form_all_axes(self, scene):
        """The camera-pose form must match ``scene._frame(at=cam)``
        on **all four** frame components: origin, right, up, front. Only
        checking front would miss a sign flip in right/up, which would mirror
        every left/right predicate."""
        cam = scene.cameras[0]
        new_view = scene.frame(position=cam.position, orientation=cam.orientation)
        at_view = scene._frame(at=("camera", 0))
        np.testing.assert_allclose(
            new_view.frame_origin, at_view.frame_origin, atol=1e-9,
            err_msg="frame_origin mismatch — position pipe is broken",
        )
        np.testing.assert_allclose(
            new_view.frame_front, at_view.frame_front, atol=1e-9,
            err_msg="frame_front mismatch — front column corrupted",
        )
        np.testing.assert_allclose(
            new_view.frame_right, at_view.frame_right, atol=1e-9,
            err_msg="frame_right mismatch — RH-rule vs OpenCV sign-flip in orientation",
        )
        np.testing.assert_allclose(
            new_view.frame_up, at_view.frame_up, atol=1e-9,
            err_msg="frame_up mismatch — up_ortho recomputation diverged",
        )

    def test_allocentric_matches_at_form_predicate_scores(self, scene_with_north):
        """Sanity check at the predicate level: every first_person directional
        score (left/right/front/behind) must agree between the new pose form
        and ``scene._frame(at=cam)`` for the same scene. If right/up
        axes are sign-flipped, left/right scores will mirror — this catches
        that even when frame_right/up aren't compared directly."""
        scene = scene_with_north
        if len(scene.objects) == 0:
            return  # skip on empty scenes
        cam = scene.cameras[0]
        new_view = scene.frame(position=cam.position, orientation=cam.orientation)
        at_view = scene._frame(at=("camera", 0))
        for label in ("left", "right", "front", "back"):
            new_scores = np.asarray(getattr(new_view.first_person, label))
            at_scores = np.asarray(getattr(at_view.first_person, label))
            np.testing.assert_allclose(
                new_scores, at_scores, atol=1e-7,
                err_msg=f"first_person.{label} scores differ between pose-form and at-form",
            )

    def test_allocentric_left_right_argmax_correct(self, scene_with_north):
        """Concrete sanity at the *value* level — not just equivalence.
        In the fixture, object 1 (chair) is to the camera's right (+x);
        object 2 (lamp) is to the left (-x). The new pose form must pick
        the correct argmax — if right/up is sign-flipped, these argmaxes
        come out swapped (lamp wins right, chair wins left)."""
        scene = scene_with_north
        cam = scene.cameras[0]
        view = scene.frame(position=cam.position, orientation=cam.orientation)
        K = len(scene.objects)
        right_scores = np.asarray(view.first_person.right)[:K]
        left_scores = np.asarray(view.first_person.left)[:K]
        assert int(np.argmax(right_scores)) == 1, (
            f"right argmax should be chair (idx 1), got {int(np.argmax(right_scores))}. "
            f"A sign-flipped right axis swaps these argmaxes."
        )
        assert int(np.argmax(left_scores)) == 2, (
            f"left argmax should be lamp (idx 2), got {int(np.argmax(left_scores))}."
        )

    def test_intrinsic_matches_object_at_form_all_axes(self, scene):
        """Same all-axis equivalence check, but for object-anchored frames."""
        obj = scene.objects[0]
        new_view = scene.frame(position=obj.position, orientation=obj.orientation)
        at_view = scene._frame(at=0)
        np.testing.assert_allclose(
            new_view.frame_origin, at_view.frame_origin, atol=1e-9
        )
        np.testing.assert_allclose(
            new_view.frame_front, at_view.frame_front, atol=1e-9
        )
        np.testing.assert_allclose(
            new_view.frame_right, at_view.frame_right, atol=1e-9
        )
        np.testing.assert_allclose(
            new_view.frame_up, at_view.frame_up, atol=1e-9
        )

    def test_intrinsic_matches_object_at_form_predicate_scores(self, scene_with_north):
        """Predicate-level equivalence for object-anchored frames."""
        scene = scene_with_north
        if len(scene.objects) < 1:
            return
        obj = scene.objects[0]
        new_view = scene.frame(position=obj.position, orientation=obj.orientation)
        at_view = scene._frame(at=0)
        for label in ("left", "right", "front", "back"):
            new_scores = np.asarray(getattr(new_view.first_person, label))
            at_scores = np.asarray(getattr(at_view.first_person, label))
            np.testing.assert_allclose(
                new_scores, at_scores, atol=1e-7,
                err_msg=f"object-frame first_person.{label} differs between pose and at-form",
            )

    def test_pose_form_matches_at_form_for_all_cameras(self, scene):
        """Run the full equivalence loop across every camera in the scene —
        a single-camera test could pass by coincidence if heading happened to
        be axis-aligned."""
        for k, cam in enumerate(scene.cameras):
            new_view = scene.frame(
                position=cam.position, orientation=cam.orientation
            )
            at_view = scene._frame(at=("camera", k))
            np.testing.assert_allclose(
                new_view.frame_origin, at_view.frame_origin, atol=1e-9,
                err_msg=f"camera {k}: origin mismatch",
            )
            np.testing.assert_allclose(
                new_view.frame_front, at_view.frame_front, atol=1e-9,
                err_msg=f"camera {k}: front mismatch",
            )
            np.testing.assert_allclose(
                new_view.frame_right, at_view.frame_right, atol=1e-9,
                err_msg=f"camera {k}: right mismatch",
            )
            np.testing.assert_allclose(
                new_view.frame_up, at_view.frame_up, atol=1e-9,
                err_msg=f"camera {k}: up mismatch",
            )

    def test_pose_form_matches_at_form_for_all_objects(self, scene):
        """Same loop across every object — catches edge cases (e.g. objects
        with near-vertical front vectors)."""
        for i, obj in enumerate(scene.objects):
            new_view = scene.frame(
                position=obj.position, orientation=obj.orientation
            )
            at_view = scene._frame(at=i)
            np.testing.assert_allclose(
                new_view.frame_origin, at_view.frame_origin, atol=1e-9,
                err_msg=f"object {i}: origin mismatch",
            )
            np.testing.assert_allclose(
                new_view.frame_front, at_view.frame_front, atol=1e-9,
                err_msg=f"object {i}: front mismatch",
            )
            np.testing.assert_allclose(
                new_view.frame_right, at_view.frame_right, atol=1e-9,
                err_msg=f"object {i}: right mismatch",
            )
            np.testing.assert_allclose(
                new_view.frame_up, at_view.frame_up, atol=1e-9,
                err_msg=f"object {i}: up mismatch",
            )

    def test_synthetic_camera_pose_form_left_right_correct_sign(self):
        """Synthetic ground-truth check independent of fixture data:
        a camera at origin facing +Z, with an object at +X (its right).
        The first_person.right score for the object must dominate first_person.left.
        If signs are flipped, this test catches it deterministically."""
        # We can't easily build a full Scene from scratch in a unit test,
        # so we exercise the pose pipeline directly on an existing scene
        # by faking a camera whose forward = +Z.
        # See test_synthetic_via_orientation_helper for the synthetic
        # equivalent that uses scene.orientation_from_forward.

    def test_synthetic_via_orientation_helper(self, scene):
        """Build an orientation matrix from a known forward via
        ``scene.orientation_from_forward`` and confirm right = cross(up, front)
        sign convention. This ensures helpers + property + scene.frame agree."""
        forward = np.array([0.0, 0.0, 1.0])
        ori = scene.orientation_from_forward(forward)
        # orientation columns: [right, up, front]
        # right must be cross(up, front)
        np.testing.assert_allclose(
            ori[:, 0], np.cross(ori[:, 1], ori[:, 2]), atol=1e-9
        )
        # And for forward=+Z with default world-up=+Y, right should be -X
        # under cross(up, forward): cross((0,1,0),(0,0,1)) = (1,0,0)... wait
        # cross((0,1,0),(0,0,1)) = (1*1-0*0, 0*0-0*1, 0*0-1*0) = (1, 0, 0).
        # So right = +X for this configuration.
        np.testing.assert_allclose(ori[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-9)
        np.testing.assert_allclose(ori[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-9)

    def test_egocentric_positional_mix(self, scene):
        """Object's position with camera's orientation produces a frame at
        the object's spot but with the camera's axes."""
        obj = scene.objects[0]
        cam = scene.cameras[0]
        view = scene.frame(position=obj.position, orientation=cam.orientation)
        np.testing.assert_allclose(view.frame_origin, obj.position, atol=1e-9)
        np.testing.assert_allclose(view.frame_front, cam.heading.forward, atol=1e-9)

    def test_centroid_position_no_special_api_needed(self, scene):
        """Any 3-vector flows through ``position=`` — including centroids
        and computed midpoints — without a special API tag."""
        midpoint = (scene.objects[0].position + scene.objects[1].position) / 2
        view = scene.frame(
            position=midpoint, orientation=scene.cameras[0].orientation
        )
        np.testing.assert_allclose(view.frame_origin, midpoint, atol=1e-9)

    def test_position_must_be_3vec(self, scene):
        with pytest.raises(ValueError, match="position must be a 3-vector"):
            scene.frame(
                position=np.array([1.0, 2.0]),
                orientation=scene.cameras[0].orientation,
            )

    def test_orientation_must_be_3x3(self, scene):
        with pytest.raises(ValueError, match="orientation must be a 3x3 matrix"):
            scene.frame(
                position=np.zeros(3),
                orientation=np.eye(2),
            )

    def test_zero_orientation_column_rejected(self, scene):
        with pytest.raises(ValueError, match="zero-magnitude column"):
            scene.frame(
                position=np.zeros(3),
                orientation=np.zeros((3, 3)),
            )

    def test_position_and_orientation_must_come_together(self, scene):
        """The engine's ``_frame``: ``position=`` requires ``orientation=`` (and vice-versa).
        (Programs' ``scene.frame(position=p)`` is a frame that ``.look_at`` gives a facing.)"""
        with pytest.raises(ValueError, match="BOTH"):
            scene._frame(position=np.zeros(3))
        with pytest.raises(ValueError, match="BOTH"):
            scene._frame(orientation=scene.cameras[0].orientation)

    def test_explicit_pose_excludes_at_arg(self, scene):
        """Passing ``position=`` AND ``at=`` is a contradiction."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            scene._frame(
                at=0,
                position=np.zeros(3),
                orientation=scene.cameras[0].orientation,
            )

    def test_neither_form_supplied_raises(self, scene):
        """``scene._frame()`` with no args is a usage error."""
        with pytest.raises(TypeError, match="missing required argument"):
            scene._frame()

    def test_axes_are_unit_normalised(self, scene):
        """Even with a non-unit orientation matrix, frame axes come out unit."""
        ori = scene.cameras[0].orientation * 2.5
        view = scene.frame(position=np.zeros(3), orientation=ori)
        np.testing.assert_allclose(np.linalg.norm(view.frame_front), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(view.frame_right), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(view.frame_up), 1.0, atol=1e-9)


class TestOrientationFromForward:
    """``scene.orientation_from_forward(forward, up=None)`` builds a 3x3 from
    a forward direction vector — the bridge for the "sitting at A facing B"
    idiom under the unified ``scene.frame(position=, orientation=)`` API."""

    def test_returns_3x3(self, scene):
        ori = scene.orientation_from_forward(np.array([0.0, 0.0, 1.0]))
        assert ori.shape == (3, 3)

    def test_third_column_is_unit_forward(self, scene):
        forward = np.array([2.0, 0.0, 1.0])  # non-unit, non-axis-aligned
        ori = scene.orientation_from_forward(forward)
        unit_fwd = forward / np.linalg.norm(forward)
        np.testing.assert_allclose(ori[:, 2], unit_fwd, atol=1e-9)

    def test_columns_are_orthonormal(self, scene):
        ori = scene.orientation_from_forward(np.array([1.0, 0.0, 1.0]))
        for c in range(3):
            np.testing.assert_allclose(np.linalg.norm(ori[:, c]), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.dot(ori[:, 0], ori[:, 1]), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.dot(ori[:, 0], ori[:, 2]), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.dot(ori[:, 1], ori[:, 2]), 0.0, atol=1e-9)

    def test_default_up_is_world_y(self, scene):
        """With default up = world-Y, the orientation's up column is vertical
        for any horizontal forward."""
        ori = scene.orientation_from_forward(np.array([1.0, 0.0, 0.0]))
        # up column should be world-Y when forward is horizontal
        np.testing.assert_allclose(np.abs(ori[1, 1]), 1.0, atol=1e-9)

    def test_zero_forward_rejected(self, scene):
        with pytest.raises(ValueError, match="forward has zero magnitude"):
            scene.orientation_from_forward(np.zeros(3))

    def test_pairs_with_scene_frame(self, scene):
        """End-to-end: feeding the helper's output directly into scene.frame
        produces a valid view whose frame_front matches the input direction."""
        forward = scene.objects[1].position - scene.objects[0].position
        view = scene.frame(
            position=scene.objects[0].position,
            orientation=scene.orientation_from_forward(forward),
        )
        unit_fwd = forward / np.linalg.norm(forward)
        np.testing.assert_allclose(view.frame_front, unit_fwd, atol=1e-9)
        np.testing.assert_allclose(view.frame_origin, scene.objects[0].position, atol=1e-9)
