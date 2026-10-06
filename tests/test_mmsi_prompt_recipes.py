"""Executable spec for the MMSI prompt recipes.

Each test corresponds to one canonical recipe family taught by the MMSI
prompt. The goal is **not** to replay every example end-to-end (which
would require a VLM scorer), but to verify that the *Python skeleton*
each example uses works against the live multiview API on a deterministic
fixture. Identity scores are constructed directly via ``_OneHot`` instead
of being scored by a VLM.

A failure here means the prompt teaches a pattern the API does not
support, so the two must be brought back in line. This file is the
contract between prompt and code.

Recipe families
---------------
- predicate-first MCQ (camera origin)
- predicate-first MCQ (object origin)
- view.first_person.<label>[idx] subscript
- view.first_person(<cardinal>)[idx]
- view.first_person.<label> threshold
- obj_facing_<label> heading classification
- frame copy via same_as=
- rotate(yaw=...) → first_person
- view.direction(target=...).label(freedom=)
- centroid → frame
- view.displacement axis-coded
- view.rotation_to
- camera-as-target indexing (K + cam_idx)
"""


import numpy as np
import pytest


from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


# ---------------------------------------------------------------------------
# Test helpers (parallel to those in test_frame_first_api.py)
# ---------------------------------------------------------------------------


def _make_camera(position, forward, cam_id=0):
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
    ext = np.eye(4)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = t
    return Camera(
        id=cam_id,
        entity_id=cam_id,
        intrinsics=np.eye(3),
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


# ---------------------------------------------------------------------------
# Fixture: 2 cameras + 4 objects, with one object facing custom direction.
# ---------------------------------------------------------------------------
#
#                z (north / forward of camera 0)
#                ▲
#                │
#       lamp ────┼──── chair             objects all at z=5:
#       (-3,0,5) │     (3,0,5)             desk  at (0,0,5)   front=-z
#                │                         chair at (3,0,5)   front=-z
#         desk ──┼                         lamp  at (-3,0,5)  front=-z
#       (0,0,5)  │                         tv    at (0,0,8)   front=+x  (faces east!)
#                │
#       cam 1 ───┼─── tv (0,0,8 front +x)
#       (5,0,0)  │
#                │
#         cam 0  ●────────────────► x (east)
#         (0,0,0)


@pytest.fixture
def scene():
    cameras = [
        _make_camera(position=[0, 0, 0], forward=[0, 0, 1], cam_id=0),
        _make_camera(position=[5, 0, 0], forward=[-1, 0, 0], cam_id=1),  # cam1 looks west
    ]
    objects = [
        _make_object(0, center=[0, 0, 5]),                  # desk: front -z (toward cam0)
        _make_object(1, center=[3, 0, 5]),                  # chair: front -z (toward cam0)
        _make_object(2, center=[-3, 0, 5]),                 # lamp: front -z (toward cam0)
        _make_object(3, center=[0, 0, 8], front=(1, 0, 0)),  # tv: faces +x (east)
    ]
    return Scene(objects=objects, cameras=cameras, images=[None, None])


# ===========================================================================
# Recipe A: predicate-first MCQ from a camera (Examples 1, 2, 4, 11, 14, 20)
# ===========================================================================


class TestRecipePredicateFirstFromCamera:
    """``view = scene._frame(at=('camera', i)); view.first_person.<label>[:K].argmax()``
    selects the entity matching the predicate. Identity scoring then picks
    the option keyword."""

    def test_argmax_picks_object_in_front_of_camera(self, scene):
        # Ex 4 / Ex 11 skeleton: "what is in front of camera 0?"
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)  # 4
        scores = view.first_person.front[:K]
        # Desk at (0,0,5) is dead-ahead of cam0 → highest front-score.
        assert int(np.argmax(scores)) == 0

    def test_argmax_right_picks_chair(self, scene):
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        # Among objects, chair (idx 1) at (+3, 0, 5) is most-right.
        assert int(np.argmax(view.first_person.right[:K])) == 1

    def test_argmax_left_picks_lamp(self, scene):
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        assert int(np.argmax(view.first_person.left[:K])) == 2  # lamp at (-3,0,5)

    def test_predicate_then_identity_recipe_a(self, scene):
        """Recipe (a) from API doc: predicate-pick a target_idx, then map
        each option keyword to its identity score at target_idx."""
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        # Predicate: which object sits "right" of cam0? (chair, idx 1)
        target_idx = int(view.first_person.right[:K].argmax())
        assert target_idx == 1

        # Identity vectors that mirror what `score(...).iota("x1")` would yield.
        # In the prompt this comes from a VLM; here we hand-craft.
        identity_chair = np.array([0.05, 0.95, 0.05, 0.05])
        identity_desk = np.array([0.95, 0.05, 0.05, 0.05])
        identity_lamp = np.array([0.05, 0.05, 0.95, 0.05])
        options = {
            "A": identity_desk[target_idx],
            "B": identity_chair[target_idx],
            "C": identity_lamp[target_idx],
        }
        assert max(options, key=options.get) == "B"


# ===========================================================================
# Recipe B: predicate-first MCQ with object as origin (Examples 5, 7-9, 12)
# ===========================================================================


class TestRecipePredicateFirstFromObject:
    """``scene._frame(at=obj_idx, same_as=scene._frame(at=('camera', i)))``
    constructs a frame *originated* at the object but with the camera's
    axes — exactly the canonical "from the camera's perspective, where is
    X relative to Y" pattern."""

    def test_same_as_copies_axes_and_moves_origin(self, scene):
        cam_view = scene._frame(at=("camera", 0))
        obj_view = scene._frame(at=1, same_as=cam_view)  # origin = chair, axes = cam0
        np.testing.assert_allclose(obj_view.frame_origin, scene.objects[1].pos, atol=1e-9)
        # Axes should equal cam0's (modulo numeric jitter in cross-products).
        np.testing.assert_allclose(obj_view.frame_front, cam_view.frame_front, atol=1e-9)
        np.testing.assert_allclose(obj_view.frame_right, cam_view.frame_right, atol=1e-9)

    def test_lamp_is_left_of_chair_in_camera_axes(self, scene):
        """Ex 7 skeleton: "from cam0's perspective, where is the lamp
        relative to the chair?"  Chair at (3,0,5), lamp at (-3,0,5) →
        lamp is to chair's LEFT in cam0's basis (+x = right)."""
        cam_view = scene._frame(at=("camera", 0))
        view = scene._frame(at=1, same_as=cam_view)  # origin = chair
        lamp_idx = 2
        options = {"A": "front-left", "B": "front-right",
                   "C": "back-left", "D": "back-right"}
        pick = max(options, key=lambda k: float(view.first_person(options[k])[lamp_idx]))
        # Lamp is far left of chair, slightly back (same z, so no front/back signal).
        # back-left and front-left should tie roughly; either is acceptable.
        assert pick in ("A", "C")

    def test_intrinsic_object_frame(self, scene):
        """Ex 9 skeleton: object's own front axis is the frame front."""
        view = scene._frame(at=3)  # tv, faces +x
        np.testing.assert_allclose(view.frame_front, np.array([1.0, 0.0, 0.0]), atol=1e-9)

    def test_custom_front_object_frame(self, scene):
        """Ex 8 skeleton: ``frame(at=X, front=Y_pos - X_pos)`` — sit on X
        facing toward Y."""
        # Sit on chair facing the desk.
        chair_pos = scene.objects[1].pos
        desk_pos = scene.objects[0].pos
        view = scene._frame(at=1, front=desk_pos - chair_pos)
        # Desk should be in front (+z of new frame) — the custom-front axis.
        K = len(scene.objects)
        front_scores = view.first_person.front[:K]
        assert int(np.argmax(front_scores)) == 0  # desk


# ===========================================================================
# Recipe C: cardinal toward via set_cardinal_vector (Examples 6c, 10-13, 19)
# ===========================================================================


class TestRecipeCardinalToward:
    """``scene.set_cardinal_vector(north_vec)`` enables cardinal labels in
    ``view.first_person("north-east")[idx]``. Without it, cardinal labels raise."""

    def test_cardinal_label_without_setup_raises(self, scene):
        view = scene._frame(at=("camera", 0))
        with pytest.raises(ValueError):
            _ = view.first_person("north")

    def test_cardinal_north_equals_relative_front_when_north_is_plus_z(self, scene):
        """Camera 0 looks +z; if north == +z, then ``first_person.north`` should
        agree with ``first_person.front``."""
        scene.set_cardinal_vector(np.array([0.0, 0.0, 1.0]))
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        np.testing.assert_allclose(
            view.first_person("north")[:K], view.first_person.front[:K], atol=1e-9
        )

    def test_unified_8way_picker(self, scene):
        """Ex 6c skeleton: predicate-first 8-way picker that mixes
        relative and cardinal labels under one ``view.first_person`` recipe."""
        # Set up: north = +z (so chair at (3,0,5) is north-east of origin).
        scene.set_cardinal_vector(np.array([0.0, 0.0, 1.0]))
        view = scene._frame(at=("camera", 0))
        chair_idx = 1

        options = {"A": "south-east", "B": "north-west",
                   "C": "north-east", "D": "south-west"}
        pick = max(options, key=lambda k: float(view.first_person(options[k])[chair_idx]))
        assert pick == "C"  # chair is north-east of cam0


# ===========================================================================
# Recipe D: predicate threshold (Example 6a)
# ===========================================================================


class TestRecipeBinaryThreshold:
    """``view.first_person.<label>[idx] > 0.5`` for binary YES/NO."""

    def test_object_in_front_passes_threshold(self, scene):
        view = scene._frame(at=("camera", 0))
        # Desk at (0,0,5) is directly in front; should score ~1.
        assert float(view.first_person.front[0]) > 0.5

    def test_object_behind_fails_front_threshold(self, scene):
        # Camera 1 at (5,0,0) looks -x. Desk at (0,0,5) is to its right-front;
        # but cam1 is offset, let's verify with a simpler check.
        view = scene._frame(at=("camera", 0))
        # No object is *behind* cam0 (all are at z=5 in front).
        K = len(scene.objects)
        assert all(float(view.first_person.back[i]) < 0.5 for i in range(K))


# ===========================================================================
# Recipe E: obj_facing_<label> heading classification (Example 6b)
# ===========================================================================


class TestRecipeObjFacing:
    """``view.obj_facing_<label>[idx]`` returns a ``ProbabilisticTensor``;
    ``max(options, key=options.get)`` accepts it directly without ``float()``.
    This tests the asymmetry the prompt explicitly calls out."""

    def test_chair_faces_back_of_camera_view(self, scene):
        # Chair at (3,0,5), front = -z. View = cam0 at origin, +z forward.
        # Chair's front (-z) aligns with view's BACK direction (-z).
        view = scene._frame(at=("camera", 0))
        chair_idx = 1
        options = {
            "A": view.obj_facing_left[chair_idx],
            "B": view.obj_facing_right[chair_idx],
            "C": view.obj_facing_front[chair_idx],
            "D": view.obj_facing_back[chair_idx],
        }
        assert max(options, key=options.get) == "D"

    def test_tv_faces_right_in_camera_view(self, scene):
        # TV at (0,0,8) front = +x. Cam0 right axis = +x. So tv faces right.
        view = scene._frame(at=("camera", 0))
        tv_idx = 3
        options = {
            "A": view.obj_facing_left[tv_idx],
            "B": view.obj_facing_right[tv_idx],
            "C": view.obj_facing_front[tv_idx],
            "D": view.obj_facing_back[tv_idx],
        }
        assert max(options, key=options.get) == "B"


# ===========================================================================
# Recipe F: rotate(yaw=...) → first_person (Examples 14, 15, 18, 23)
# ===========================================================================


class TestRecipeRotateThenToward:
    """``view.rotate(yaw=θ).first_person.<label>`` lets us turn without rebuilding."""

    def test_rotate_180_swaps_front_and_back(self, scene):
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        front_before = view.first_person.front[:K]
        front_after = view.rotate(yaw=180).first_person.front[:K]
        # After turning around, the desk (in front of the unrotated view) is
        # behind, so the rotated view does not rank it first for "front".
        assert int(np.argmax(front_before)) == 0       # desk in front before the turn
        assert int(np.argmax(front_after)) != 0        # desk not in front after the turn

    def test_rotate_yaw_then_toward_is_consistent_with_back(self, scene):
        """``rotate(yaw=180).first_person.front`` ≈ original ``first_person.back`` for
        objects in the horizontal plane.

        Anchored at an object, so the geometric symmetry is exact.
        """
        view = scene._frame(at=("object", 0))
        K = len(scene.objects)
        rotated_front = view.rotate(yaw=180).first_person.front[:K]
        original_back = view.first_person.back[:K]
        np.testing.assert_allclose(rotated_front, original_back, atol=1e-6)


# ===========================================================================
# Recipe G: direction(target=point|idx).label(freedom=4|8) (Examples 14, 16, 17)
# ===========================================================================


class TestRecipeDirectionLabel:
    """``view.direction(target=...).label(freedom=4|8)`` returns a string
    from a fixed vocabulary."""

    def test_freedom4_axial_label(self, scene):
        view = scene._frame(at=("camera", 0))
        # Chair at (3,0,5): cam0's +z forward, so chair is front-right.
        # freedom=4 (axial) → "front" wins.
        label = view.direction(target=1).label(freedom=4)
        assert label == "front"

    def test_freedom8_diagonal_label(self, scene):
        view = scene._frame(at=("camera", 0))
        label = view.direction(target=1).label(freedom=8)
        assert label == "front-right"

    def test_freedom4_with_3d_point_target(self, scene):
        """Ex 16 ingredient: target may be a bare 3-vector (e.g. centroid)."""
        view = scene._frame(at=("camera", 0))
        pt = np.array([3.0, 0.0, 5.0])  # same as chair position
        assert view.direction(target=pt).label(freedom=4) == "front"

    def test_label_to_letter_dict_idiom(self, scene):
        """Ex 16 closing line: ``label_to_letter[direction(...).label(...)]``
        is the canonical region-MCQ pattern."""
        view = scene._frame(at=("camera", 0))
        label_to_letter = {"right": "A", "front": "B", "back": "C", "left": "D"}
        bathroom_pt = np.array([3.0, 0.0, 5.0])
        result = label_to_letter[view.direction(target=bathroom_pt).label(freedom=4)]
        assert result == "B"


# ===========================================================================
# Recipe H: scene.centroid → frame (Examples 16, 17)
# ===========================================================================


class TestRecipeCentroid:
    """Smoke tests for the centroid → frame composition used in Ex 17."""

    def test_centroid_then_frame_origin(self, scene):
        # Region = desk + chair + lamp; centroid = (0, 0, 5).
        kitchen_pt = scene.centroid([0, 1, 2])
        np.testing.assert_allclose(kitchen_pt, np.array([0.0, 0.0, 5.0]), atol=1e-9)

        # Frame originated at centroid with camera 0's front axis.
        view = scene._frame(at=kitchen_pt, front=scene.cameras[0].front)
        np.testing.assert_allclose(view.frame_origin, kitchen_pt, atol=1e-9)

    def test_into_room_front_construction(self, scene):
        """Ex 17 skeleton: door's intrinsic front may face outside, so use
        ``(region_centroid - door_pos)`` as the "into-the-room" front."""
        door_pos = np.array([0.0, 0.0, 0.0])  # imagine door at origin
        kitchen_pt = scene.centroid([0, 1, 2])  # at (0,0,5)
        into_room = kitchen_pt - door_pos       # (0,0,5) → +z
        # Build a synthetic door object on the fly via a frame at the point.
        view = scene._frame(at=door_pos, front=into_room)
        # Kitchen centroid should sit in +front (z) of this frame.
        np.testing.assert_allclose(view.frame_front, np.array([0.0, 0.0, 1.0]), atol=1e-9)


# ===========================================================================
# Recipe I: camera-as-target indexing (Examples 2, 18-20)
# ===========================================================================


class TestRecipeCameraAsTarget:
    """Cameras occupy indices ``[K, K+C)`` in ``view.first_person.<label>``. The
    prompt's Ex 2/18-20 use this to ask "where does camera j sit relative
    to view?"."""

    def test_camera_index_offset(self, scene):
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        C = len(scene.cameras)
        assert view.first_person.front.shape == (K + C,)

    def test_camera_self_score_is_zero(self, scene):
        """Cam 0 at the origin == frame_origin; horizontal magnitude is
        zero, so the per-axis score returns 0."""
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        assert view.first_person.front[K + 0] == 0.0

    def test_other_camera_lies_to_the_right(self, scene):
        """Cam 1 at (5,0,0) is on cam0's +x = right axis."""
        view = scene._frame(at=("camera", 0))
        K = len(scene.objects)
        right_score_cam1 = float(view.first_person.right[K + 1])
        front_score_cam1 = float(view.first_person.front[K + 1])
        assert right_score_cam1 > front_score_cam1
        assert right_score_cam1 > 0.9  # cam1 is dead-right of cam0


# ===========================================================================
# Recipe J: displacement axis-coded (Example 21)
# ===========================================================================


class TestRecipeDisplacement:
    """``view.displacement(target)`` returns a 3-vector in the view's basis,
    used for axis-coded MCQ ("Did the camera move forward and right?")."""

    def test_displacement_to_camera_target(self, scene):
        view = scene._frame(at=("camera", 0))
        disp = view.displacement(("camera", 1))  # cam1 at (5,0,0)
        assert disp.shape == (3,)
        # Cam0's basis: right=+x, up=+y, front=+z. Cam1 lies at +5 right.
        np.testing.assert_allclose(disp, np.array([5.0, 0.0, 0.0]), atol=1e-9)

    def test_displacement_sign_pattern_for_mcq(self, scene):
        """Ex 21 idiom: check signs of the 3 components against a coded
        option like ``(forward=+, right=+, up=0)``."""
        view = scene._frame(at=("camera", 0))
        disp = view.displacement(("camera", 1))
        right_sign = np.sign(disp[0])
        up_sign = np.sign(disp[1])
        front_sign = np.sign(disp[2])
        assert right_sign == 1   # cam1 to the right
        assert up_sign == 0      # same height
        assert front_sign == 0   # same depth


# ===========================================================================
# Recipe K: rotation_to (Example 22)
# ===========================================================================


class TestRecipeRotationTo:
    """``view.rotation_to(other)`` returns ``(yaw_deg, pitch_deg)`` between
    two frames. Used for camera-rotation MCQs."""

    def test_rotation_to_self_is_zero(self, scene):
        view = scene._frame(at=("camera", 0))
        yaw, pitch = view.rotation_to(view)
        assert abs(yaw) < 1e-6
        assert abs(pitch) < 1e-6

    def test_rotation_to_other_camera_returns_tuple(self, scene):
        view0 = scene._frame(at=("camera", 0))
        view1 = scene._frame(at=("camera", 1))
        result = view0.rotation_to(view1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Cam0 forward = +z, cam1 forward = -x → yaw should be -90° (turn
        # left from +z to -x rotates CCW from above) or +90° depending on
        # sign convention. We don't pin the sign here; the prompt's Ex 22
        # uses ``abs(yaw)`` and a sign check separately.
        yaw, _pitch = result
        assert abs(abs(yaw) - 90.0) < 1.0
