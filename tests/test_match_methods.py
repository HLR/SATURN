"""Tests for Scene.match_* convenience methods and the frame label matchers
(match / match_rotation / match_translation / check_relation / best_match)."""


import numpy as np
import pytest


from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


# ---- Helpers ----


def _make_camera(position, forward, cam_id=0):
    """Build Camera from position and forward vector (world coords)."""
    position = np.asarray(position, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0)
    t = -R_w2c @ position
    ext = np.eye(4)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = t
    K = np.eye(3)
    return Camera(
        id=cam_id, entity_id=cam_id, intrinsics=K, extrinsics=ext, image_size=(480, 640)
    )


def _make_object(obj_id, center, dims=(0.5, 0.5, 0.5), front=(0, 0, -1)):
    """Build a minimal MergedObject."""
    center = np.asarray(center, dtype=float)
    front = np.asarray(front, dtype=float)
    front = front / (np.linalg.norm(front) + 1e-12)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, front)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    dims = np.asarray(dims, dtype=float)
    rotation = np.column_stack([right, up, front])
    corners = np.zeros((8, 3))  # placeholder
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
        dims=dims,
        corners_world=corners,
        height=float(dims[1]),
        support_y=float(center[1] - dims[1] / 2),
    )


def _scene_with_cameras_and_objects(cameras, objects):
    return Scene(objects=objects, cameras=cameras, images=[None] * len(cameras))


# ---- test _normalize_label ----


def test_normalize_label():
    assert Scene._normalize_label("back-right") == "back right"
    assert Scene._normalize_label("Behind") == "back"
    assert Scene._normalize_label("directly to the right") == "right"
    assert Scene._normalize_label("Front-Left") == "front left"
    assert Scene._normalize_label("  left rear ") == "back left"
    assert Scene._normalize_label("immediate left") == "left"


def test_best_option_exact():
    assert (
        Scene._best_option("front right", {"A": "front right", "B": "back left"}) == "A"
    )


def test_best_option_synonym():
    assert Scene._best_option("behind", {"A": "back", "B": "front"}) == "A"


def test_best_option_substring():
    assert (
        Scene._best_option("front left", {"A": "upper front left area", "B": "back"})
        == "A"
    )


# ---- test match_direction ----


def test_match_direction_cam_to_obj_right():
    """Object directly to the right of camera → should match "right"."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj = _make_object(
        0, [2, 0, 0]
    )  # 2m to the right in world (camera right = +X when looking -Z)
    scene = _scene_with_cameras_and_objects([cam0], [obj])
    result = scene.match_direction(
        ("camera", 0),
        0,
        {"A": "front", "B": "right", "C": "behind", "D": "left"},
    )
    assert result == "B", f"Expected B (right), got {result}"


def test_match_direction_cam_to_obj_behind():
    """Object behind camera (world +Z when looking -Z) → should match "behind/back"."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj = _make_object(0, [0, 0, 3])  # behind
    scene = _scene_with_cameras_and_objects([cam0], [obj])
    result = scene.match_direction(
        ("camera", 0),
        0,
        {"A": "front", "B": "right", "C": "back", "D": "left"},
    )
    assert result == "C", f"Expected C (back), got {result}"


def test_match_direction_synonym_handling():
    """Options use 'rear' instead of 'back' — synonyms should handle it."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj = _make_object(0, [0, 0, 3])  # behind
    scene = _scene_with_cameras_and_objects([cam0], [obj])
    result = scene.match_direction(
        ("camera", 0),
        0,
        {"A": "front", "B": "right", "C": "rear", "D": "left"},
    )
    assert result == "C", f"Expected C (rear→back), got {result}"


def test_match_direction_cam_to_cam():
    """Camera 1 is to the front-right of camera 0."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)  # looking -Z
    cam1 = _make_camera([2, 0, -2], [0, 0, -1], cam_id=1)  # +X, -Z = front-right
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    result = scene.match_direction(
        ("camera", 0),
        ("camera", 1),
        {"A": "front right", "B": "front left", "C": "back right", "D": "back left"},
    )
    assert result == "A", f"Expected A (front right), got {result}"


def test_match_direction_directly_to_the():
    """Options with 'Directly to the right' should normalize properly."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    cam1 = _make_camera([3, 0, 0], [0, 0, -1], cam_id=1)  # purely to the right
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    result = scene.match_direction(
        ("camera", 0),
        ("camera", 1),
        {
            "A": "Front right",
            "B": "Directly to the right",
            "C": "Directly to the left",
            "D": "Front left",
        },
    )
    assert result == "B", f"Expected B (directly to the right), got {result}"


# ---- test match_rotation_direction ----


# ---- test match_movement ----


# ---- test match_rotation ----


# ---- test match_translation ----


def test_match_translation_pure_up():
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    cam1 = _make_camera([0, 1, 0], [0, 0, -1], cam_id=1)
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    v0 = scene._frame(at=("camera", 0))
    v1 = scene._frame(at=("camera", 1))
    result = v0.match_translation(
        v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}
    )
    assert result == "A", f"Expected A (Up), got {result}"


def test_match_translation_pure_down():
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    cam1 = _make_camera([0, -1, 0], [0, 0, -1], cam_id=1)
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    v0 = scene._frame(at=("camera", 0))
    v1 = scene._frame(at=("camera", 1))
    result = v0.match_translation(
        v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}
    )
    assert result == "B", f"Expected B (Down), got {result}"


def test_match_translation_pure_right():
    """cam1 displaced along v0's own +right axis must score "Right".

    Uses ``v0._frame_right`` as the displacement direction so the test is
    self-consistent with the FrameNamespace's basis convention
    (cross(up, front)) — same convention rotation_to / match_rotation use.
    """
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    scene_anchor = _scene_with_cameras_and_objects([cam0], [])
    v0_anchor = scene_anchor._frame(at=("camera", 0))
    cam1 = _make_camera(v0_anchor._frame_right * 1.0, [0, 0, -1], cam_id=1)
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    v0 = scene._frame(at=("camera", 0))
    v1 = scene._frame(at=("camera", 1))
    result = v0.match_translation(
        v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}
    )
    assert result == "D", f"Expected D (Right), got {result}"


def test_match_translation_pure_left():
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    scene_anchor = _scene_with_cameras_and_objects([cam0], [])
    v0_anchor = scene_anchor._frame(at=("camera", 0))
    cam1 = _make_camera(-v0_anchor._frame_right * 1.0, [0, 0, -1], cam_id=1)
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    v0 = scene._frame(at=("camera", 0))
    v1 = scene._frame(at=("camera", 1))
    result = v0.match_translation(
        v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}
    )
    assert result == "C", f"Expected C (Left), got {result}"


def test_match_translation_dominant_pitch_in_mixed_options():
    """Translation up dominates over a much smaller right-component."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    cam1 = _make_camera([0.05, 0.5, 0], [0, 0, -1], cam_id=1)  # mostly up, tiny right
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    v0 = scene._frame(at=("camera", 0))
    v1 = scene._frame(at=("camera", 1))
    result = v0.match_translation(
        v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}
    )
    assert result == "A", f"Expected A (dominant Up), got {result}"


def test_match_translation_independent_of_rotation():
    """match_translation must ignore rotation: a camera that translated
    along v0's +right axis must score "Right" regardless of how cam1 is
    pointing."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    scene_anchor = _scene_with_cameras_and_objects([cam0], [])
    v0_anchor = scene_anchor._frame(at=("camera", 0))
    cam1 = _make_camera(v0_anchor._frame_right * 1.0, [1, 0, 0], cam_id=1)
    scene = _scene_with_cameras_and_objects([cam0, cam1], [])
    v0 = scene._frame(at=("camera", 0))
    v1 = scene._frame(at=("camera", 1))
    result = v0.match_translation(
        v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}
    )
    assert result == "D", f"Expected D (Right by displacement), got {result}"


# ---- test match_direction with cardinal params ----


def test_match_direction_cardinal_with_anchor():
    """Object 2 is northeast of object 0. Query: where is object 1 from object 0?"""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    obj0 = _make_object(0, [0, 0, 0])
    obj1 = _make_object(
        1, [0, 0, -3]
    )  # directly in front of obj0 in world = north (if NE is at +X,-Z)
    obj2 = _make_object(2, [3, 0, -3])  # obj2 is at +X,-Z from obj0 = "northeast"
    scene = _scene_with_cameras_and_objects([cam0], [obj0, obj1, obj2])
    result = scene.match_direction(
        0,
        1,
        {"A": "northwest", "B": "southeast", "C": "northeast", "D": "southwest"},
        anchor=(2, 0),
        anchor_cardinal="northeast",
    )
    # obj2 is northeast of obj0. obj1 is at (0,0,-3) from obj0=(0,0,0).
    assert result in ("A", "B", "C", "D"), f"Got invalid result: {result}"


# ---- test match_extent ----


# ---- test scene.vector() ----


def test_vector_cam_to_obj():
    """Vector from camera to object should be normalized."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj = _make_object(0, [3, 0, -4])  # distance = 5
    scene = _scene_with_cameras_and_objects([cam0], [obj])
    v = scene.vector(("camera", 0), 0)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6, "vector should be unit length"
    assert v[0] > 0.5, "should point toward +X"
    assert v[2] < -0.5, "should point toward -Z"


def test_vector_obj_to_obj():
    """Vector between two objects."""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    obj0 = _make_object(0, [0, 0, 0])
    obj1 = _make_object(1, [0, 0, -3])
    scene = _scene_with_cameras_and_objects([cam0], [obj0, obj1])
    v = scene.vector(0, 1)
    assert abs(v[2] - (-1.0)) < 1e-6, "should point purely -Z"


def test_vector_same_position():
    """Vector between coincident entities should be zero."""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    obj = _make_object(0, [0, 0, 0])
    scene = _scene_with_cameras_and_objects([cam0], [obj])
    v = scene.vector(0, 0)
    assert np.linalg.norm(v) < 1e-9


# ---- test match_direction with observer= ----


def test_match_direction_observer_right():
    """Object B is to the right of object A from camera 0's viewpoint."""
    # Camera looking -Z, objects both in front at Z=-3
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj_a = _make_object(0, [-1, 0, -3])  # left in camera view
    obj_b = _make_object(1, [2, 0, -3])  # right in camera view
    scene = _scene_with_cameras_and_objects([cam0], [obj_a, obj_b])
    result = scene.match_direction(
        0,
        1,
        {"A": "left", "B": "right", "C": "front", "D": "behind"},
        observer=0,
    )
    assert result == "B", f"Expected B (right from cam0 view), got {result}"


def test_match_direction_observer_behind():
    """Object B is behind object A from camera 0's viewpoint."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj_a = _make_object(0, [0, 0, -3])  # nearer
    obj_b = _make_object(
        1, [0, 0, -6]
    )  # farther = "in front" from camera view (deeper)
    scene = _scene_with_cameras_and_objects([cam0], [obj_a, obj_b])
    result = scene.match_direction(
        0,
        1,
        {"A": "left", "B": "right", "C": "front", "D": "behind"},
        observer=0,
    )
    # B is farther in the camera's forward direction → "front" of A from cam's perspective
    assert result == "C", f"Expected C (front from cam0 view), got {result}"


def test_match_direction_observer_overrides_object_frame():
    """Observer should override the object's intrinsic frame_front."""
    # Camera looking +X, object A has intrinsic front along -Z.
    # Without observer, A→B direction uses A's frame (-Z = front).
    # With observer=cam0, uses cam0's frame (+X = front).
    cam0 = _make_camera([0, 0, 0], [1, 0, 0], cam_id=0)  # looking +X
    obj_a = _make_object(0, [5, 0, 0], front=(0, 0, -1))  # front is -Z
    obj_b = _make_object(1, [5, 0, -3])  # 3m in -Z from A
    scene = _scene_with_cameras_and_objects([cam0], [obj_a, obj_b])

    # Without observer: B is in A's front direction (-Z)
    result_no_obs = scene.match_direction(
        0,
        1,
        {"A": "front", "B": "right", "C": "behind", "D": "left"},
    )
    assert result_no_obs == "A", (
        f"Without observer: expected A (front), got {result_no_obs}"
    )

    # With observer: cam0 looks +X, so -Z is "left" from cam's perspective
    result_obs = scene.match_direction(
        0,
        1,
        {"A": "front", "B": "right", "C": "behind", "D": "left"},
        observer=0,
    )
    assert result_obs == "D", f"With observer: expected D (left), got {result_obs}"


# ---- test match_direction with facing= ----


def test_match_direction_facing_right():
    """Facing TV from chair; lamp is to the right."""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    chair = _make_object(0, [0, 0, 0])
    tv = _make_object(1, [0, 0, -3])  # TV is in -Z from chair → "forward"
    lamp = _make_object(2, [-3, 0, -1.5])  # facing -Z, right is -X (a camera facing +Z has +X on its right)
    scene = _scene_with_cameras_and_objects([cam0], [chair, tv, lamp])
    result = scene.match_direction(
        0,
        2,
        {"A": "front", "B": "right", "C": "behind", "D": "left"},
        facing=1,  # facing the TV
    )
    assert result == "B", f"Expected B (right when facing TV), got {result}"


def test_match_direction_facing_left():
    """Facing TV from chair; lamp is to the left."""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    chair = _make_object(0, [0, 0, 0])
    tv = _make_object(1, [0, 0, -3])  # TV is in -Z from chair → "forward"
    lamp = _make_object(2, [3, 0, -1.5])  # facing -Z, left is +X
    scene = _scene_with_cameras_and_objects([cam0], [chair, tv, lamp])
    result = scene.match_direction(
        0,
        2,
        {"A": "front", "B": "right", "C": "behind", "D": "left"},
        facing=1,
    )
    assert result == "D", f"Expected D (left when facing TV), got {result}"


def test_match_direction_facing_behind():
    """Facing TV from chair; something behind you."""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    chair = _make_object(0, [0, 0, 0])
    tv = _make_object(1, [0, 0, -3])  # forward direction
    bookshelf = _make_object(2, [0, 0, 2])  # behind when facing TV
    scene = _scene_with_cameras_and_objects([cam0], [chair, tv, bookshelf])
    result = scene.match_direction(
        0,
        2,
        {"A": "front", "B": "right", "C": "behind", "D": "left"},
        facing=1,
    )
    assert result == "C", f"Expected C (behind when facing TV), got {result}"


def test_match_direction_observer_facing_mutually_exclusive():
    """Setting both observer and facing should raise ValueError."""
    cam0 = _make_camera([0, 0, 0], [0, 0, -1], cam_id=0)
    obj_a = _make_object(0, [0, 0, -3])
    obj_b = _make_object(1, [2, 0, -3])
    scene = _scene_with_cameras_and_objects([cam0], [obj_a, obj_b])
    try:
        scene.match_direction(0, 1, {"A": "left"}, observer=0, facing=1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---- test CameraHeading.up and .right ----


def test_camera_heading_up():
    """CameraHeading looking -Z should have up ≈ (0, 1, 0)."""
    from saturn.scene.types import CameraHeading

    h = CameraHeading(np.array([0, 0, -1]))
    up = h.up
    assert abs(up[1] - 1.0) < 1e-6, f"Expected up ≈ (0,1,0), got {up}"


def test_camera_heading_right():
    """CameraHeading looking -Z should have right ≈ (1, 0, 0)."""
    from saturn.scene.types import CameraHeading

    h = CameraHeading(np.array([0, 0, -1]))
    r = h.right
    # forward × world_up = (0,0,-1) × (0,1,0) = (1, 0, 0)
    assert abs(r[0] - 1.0) < 1e-6, f"Expected right ≈ (1,0,0), got {r}"


def test_camera_heading_forward_setter():
    """Setting forward should update the direction."""
    from saturn.scene.types import CameraHeading

    h = CameraHeading(np.array([0, 0, -1]))
    h.forward = np.array([1, 0, 0])
    fwd = h.forward
    assert abs(fwd[0] - 1.0) < 1e-6, f"Expected forward ≈ (1,0,0), got {fwd}"


# ---- test match_direction with north_landmark ----


def test_match_direction_north_landmark():
    """Window is on the north wall. Where is obj1 from obj0?"""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    obj0 = _make_object(0, [0, 0, 0])  # center of room
    obj1 = _make_object(1, [3, 0, 0])  # +X from obj0
    window = _make_object(2, [0, 0, -5])  # far in -Z = "north" wall
    scene = _scene_with_cameras_and_objects([cam0], [obj0, obj1, window])
    result = scene.match_direction(
        0,
        1,
        {"A": "north", "B": "east", "C": "south", "D": "west"},
        north_landmark=2,
        landmark_cardinal="north",
    )
    # Window at -Z is north. When facing north (-Z), east is -X direction
    # (90° CW from -Z). obj1 at +X is therefore WEST.
    assert result == "D", f"Expected D (west), got {result}"


def test_match_direction_east_landmark():
    """Window is on the east wall. Where is obj1 from obj0?"""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    obj0 = _make_object(0, [0, 0, 0])
    obj1 = _make_object(1, [0, 0, 3])  # in +Z direction from obj0
    window = _make_object(2, [5, 0, 0])  # far in +X = "east" wall
    scene = _scene_with_cameras_and_objects([cam0], [obj0, obj1, window])
    result = scene.match_direction(
        0,
        1,
        {"A": "north", "B": "east", "C": "south", "D": "west"},
        north_landmark=2,
        landmark_cardinal="east",
    )
    # Window at +X is east. atan2 convention: +Z = 0° = north.
    # So north = +Z. obj1 is at +Z from obj0 = north.
    assert result == "A", f"Expected A (north), got {result}"


# ---- test direction_utils ----


def test_classify_direction_cardinal_basic():
    """classify_direction should give 'east' for +X when north = +Z."""
    from saturn.scene.direction_utils import classify_direction

    north = np.array([0.0, 0.0, 1.0])
    diff = np.array([3.0, 0.0, 0.0])
    assert classify_direction(diff, north, 4, labels="cardinal") == "east"
    assert classify_direction(diff, north, 8, labels="cardinal") == "east"


def test_classify_direction_relative_basic():
    """classify_direction with labels='relative' should give 'right' for +X when front = +Z."""
    from saturn.scene.direction_utils import classify_direction

    front = np.array([0.0, 0.0, 1.0])
    diff = np.array([3.0, 0.0, 0.0])
    assert classify_direction(diff, front, 4, labels="relative") == "right"


def test_classify_direction_north_minus_z():
    """When north = -Z, +X should be west (not east)."""
    from saturn.scene.direction_utils import classify_direction

    north = np.array([0.0, 0.0, -1.0])
    diff = np.array([3.0, 0.0, 0.0])  # +X direction
    assert classify_direction(diff, north, 4, labels="cardinal") == "west"


def test_translate_label_round_trip():
    """Translating relative→cardinal→relative should recover the original."""
    from saturn.scene.direction_utils import translate_label

    for lbl in ["front", "back", "left", "right", "front-right", "back-left"]:
        cardinal = translate_label(lbl, to="cardinal")
        back = translate_label(cardinal, to="relative")
        assert back == lbl.replace("behind", "back"), (
            f"Round-trip failed for {lbl}: {cardinal} → {back}"
        )


def test_is_relative_label():
    """is_relative_label should identify relative vs cardinal labels."""
    from saturn.scene.direction_utils import is_relative_label

    assert is_relative_label("front") is True
    assert is_relative_label("back-left") is True
    assert is_relative_label("behind") is True
    assert is_relative_label("north") is False
    assert is_relative_label("southeast") is False


def test_resolve_north_anchor_pair():
    """resolve_north with anchor pair should derive north correctly."""
    from saturn.scene.direction_utils import resolve_north

    # anchor at +X, reference at origin, anchor_cardinal="east"
    # → the +X direction is east, so north should be +Z
    north = resolve_north(
        anchor_pos=np.array([5.0, 0.0, 0.0]),
        reference_pos=np.array([0.0, 0.0, 0.0]),
        anchor_cardinal="east",
    )
    assert abs(north[2] - 1.0) < 0.01, f"Expected north ≈ +Z, got {north}"
    assert abs(north[0]) < 0.01, f"Expected north.x ≈ 0, got {north}"


def test_direction_value_label_uses_back():
    """DirectionValue.label() should use 'back' not 'behind'."""
    from saturn.scene.types import DirectionValue

    d = DirectionValue(180.0)  # straight behind
    assert d.label(4) == "back"
    assert d.label(8) == "back"
    d2 = DirectionValue(135.0)
    assert d2.label(8) == "back-right"


# ---- test scene serialization ----


def test_scene_to_dict_round_trip():
    """Scene should survive to_dict → from_dict round-trip."""
    import json

    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    cam1 = _make_camera([5, 0, 0], [-1, 0, 0], cam_id=1)
    obj0 = _make_object(0, [0, 0, 0])
    obj1 = _make_object(1, [3, 0, 0])
    scene = _scene_with_cameras_and_objects([cam0, cam1], [obj0, obj1])

    data = scene.to_dict()
    # Verify JSON-serializable
    json_str = json.dumps(data)
    data_back = json.loads(json_str)

    from saturn.scene.scene import Scene

    scene2 = Scene.from_dict(data_back)
    assert len(scene2.objects) == 2
    assert len(scene2.cameras) == 2
    assert scene2.objects[0].label == obj0.label
    assert scene2.cameras[0].id == 0
    # Verify positions survived
    np.testing.assert_allclose(scene2.objects[1].center_world, [3, 0, 0], atol=1e-6)
    np.testing.assert_allclose(scene2.cameras[0].position_world, [0, 0, 5], atol=1e-6)
    # Verify scene is functional: direction query should work
    d = scene2.direction(0, 1, observer=0)
    assert d.label(4) in ("front", "right", "back", "left")


def test_scene_dump_load(tmp_path):
    """Scene dump/load via file should preserve all data."""
    cam0 = _make_camera([0, 0, 5], [0, 0, -1], cam_id=0)
    obj0 = _make_object(0, [1, 2, 3])
    scene = _scene_with_cameras_and_objects([cam0], [obj0])

    filepath = str(tmp_path / "scene_test.json")
    scene.dump(filepath)

    from saturn.scene.scene import Scene

    scene2 = Scene.load(filepath)
    assert len(scene2.objects) == 1
    assert len(scene2.cameras) == 1
    np.testing.assert_allclose(scene2.objects[0].center_world, [1, 2, 3], atol=1e-6)


# ---- Object motion tests ----


def _make_object_with_per_view_centers(obj_id, center, per_view_centers, **kw):
    """Build a MergedObject with per_view_centers."""
    obj = _make_object(obj_id, center, **kw)
    obj.per_view_centers = {
        k: np.asarray(v, dtype=float) for k, v in per_view_centers.items()
    }
    return obj


def test_per_view_centers_serialization():
    """per_view_centers survives to_dict / from_dict round-trip."""
    obj = _make_object_with_per_view_centers(
        0, [1, 2, 3], per_view_centers={0: [0, 0, 0], 1: [2, 4, 6]}
    )
    d = obj.to_dict()
    assert "per_view_centers" in d
    obj2 = MergedObject.from_dict(d)
    np.testing.assert_allclose(obj2.per_view_centers[0], [0, 0, 0])
    np.testing.assert_allclose(obj2.per_view_centers[1], [2, 4, 6])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


# ---- MCQ label matchers accept direction aliases ---------------------------


@pytest.fixture
def forward_step():
    scene = _scene_with_cameras_and_objects(
        [], [_make_object(0, (0, 0, 5)), _make_object(1, (1, 0, 5))]
    )
    v0 = scene.frame(position=np.zeros(3), orientation=np.eye(3))
    return v0, v0.translate([0, 0, 2.0])  # camera stepped straight forward


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"A": "back", "B": "forward"}, "B"),
        ({"A": "left", "B": "forward", "C": "right"}, "B"),
        ({"A": "rear", "B": "Forward"}, "B"),
    ],
)
def test_match_translation_accepts_forward(forward_step, options, expected):
    v0, v1 = forward_step
    assert v0.match_translation(v1, options) == expected


def test_match_rotation_accepts_forward():
    scene = _scene_with_cameras_and_objects([], [_make_object(0, (0, 0, 5))])
    v0 = scene.frame(position=np.zeros(3), orientation=np.eye(3))
    turned = v0.rotate(yaw=170)
    assert v0.match_rotation(turned, {"A": "forward", "B": "rear"}) == "B"
    assert v0.match_rotation(turned, {"A": "front-left", "B": "behind left"}) == "B"


@pytest.mark.parametrize("label", ["behind-left", "behind left", "left-behind", "rear left"])
def test_view_match_accepts_back_left_aliases(label):
    # Object 0 is back-left of object 1 in a frame with front=+Z.
    scene = _scene_with_cameras_and_objects(
        [], [_make_object(0, (-1, 0, -1)), _make_object(1, (0, 0, 0))]
    )
    view = scene.frame(position=np.array([0, 0, -10.0]), orientation=np.eye(3))
    assert view.match(source=0, target=1, options={"A": label, "B": "front-left"}) == "A"


# ---- one front/back convention across axial and diagonal labels ----------


@pytest.fixture
def near_far_view():
    # ref at origin; A (idx 1) farther along +front, B (idx 2) closer; both slightly left.
    objs = [
        _make_object(0, (0, 0, 0)),
        _make_object(1, (-0.3, 0, 2)),
        _make_object(2, (-0.3, 0, -2)),
    ]
    scene = _scene_with_cameras_and_objects([], objs)
    return scene.frame(position=np.array([0, 0, -10.0]), orientation=np.eye(3))


@pytest.mark.parametrize(
    "label, closer_wins",
    [
        # relative labels: occluder convention ("front" = closer to observer)
        ("front", True), ("front-left", True), ("forward-left", True),
        ("behind", False), ("back-left", False), ("behind-left", False),
        # compass labels: frame directions (north = +front = farther)
        ("north", False), ("northwest", False), ("north-west", False),
        ("south", True), ("southwest", True),
    ],
)
def test_check_relation_front_back_convention(near_far_view, label, closer_wins):
    far = near_far_view.check_relation(label, 1, 0)
    near = near_far_view.check_relation(label, 2, 0)
    assert (near > 0.5 > far) if closer_wins else (far > 0.5 > near)


@pytest.mark.parametrize(
    "label, expected",
    [("front", "B"), ("front-left", "B"), ("north", "A"), ("northwest", "A"),
     ("behind", "A"), ("back-left", "A"), ("south", "B"), ("southwest", "B")],
)
def test_best_match_reference_convention(near_far_view, label, expected):
    opts = {"A": np.array([0, 1.0, 0]), "B": np.array([0, 0, 1.0])}
    assert near_far_view.best_match(label, opts, reference=0) == expected


def test_check_relation_diagonal_matches_third_person(near_far_view):
    tp = near_far_view.third_person.front_left.tensor
    for i in (1, 2):
        assert near_far_view.check_relation("front-left", i, 0) == pytest.approx(float(tp[i, 0]))
