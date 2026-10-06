"""Conventions of obj_<dir> and view.first_person.

These tests pin the semantics of the two primitive families that the
``Anchor`` predicates build on:

  - ``obj_left/right/front/behind[i, j]``  — subject-first 2D matrices over
    objects, computed by
    ``spatial_engine_v2.relations.metrics.compute_obj_relative_from_axes``.
  - ``view.first_person.<dir>[k]``         — 1D arrays over (objects ∪ cameras),
    accessed via ``scene._frame(at=...).first_person.<dir>``.

Generated programs rely on these conventions. If one of them has to change,
revisit the design rather than relax the test.
"""

import math
import os
import sys

import numpy as np
import pytest



from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject
from saturn.predicates.metrics import (
    compute_obj_relative_from_axes,
)


# ---------------------------------------------------------------------------
# Helpers — same shape as tests/test_frame_first_api.py for parity
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
    K = np.eye(3)
    return Camera(
        id=cam_id,
        entity_id=cam_id,
        intrinsics=K,
        extrinsics=ext,
        image_size=(480, 640),
    )


def _make_object(obj_id, center, front=(0, 0, -1)):
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


@pytest.fixture
def scene_three_objects_one_camera():
    """Camera at origin looking +z; three objects forming an L in front.

    World frame: +x right, +y up, +z forward (camera-relative).

    Layout:
      camera 0 at (0,0,0), forward = +z         → cam.right = +x
      object 0 (desk)  at (0,0,5)  facing -z    → desk.right = cross(+y, -z) = -x
      object 1 (chair) at (3,0,5)  facing -z    → at desk's LEFT (chair is at desk's -right direction)
      object 2 (lamp)  at (-3,0,5) facing -z    → at desk's RIGHT
    """
    cameras = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0)]
    objects = [
        _make_object(0, [0, 0, 5]),   # desk
        _make_object(1, [3, 0, 5]),   # chair
        _make_object(2, [-3, 0, 5]),  # lamp
    ]
    return Scene(objects=objects, cameras=cameras, images=[None])


# ===========================================================================
# CONVENTION 1: obj_<dir>[i, j] is subject-first
# ===========================================================================


def test_obj_left_is_subject_first(scene_three_objects_one_camera):
    """``obj_left[i, j]`` = "i is to the left of j, from j's perspective".

    The reference (j) is the SECOND index. This is the documented convention
    at ``metrics.py:252``.
    """
    scene = scene_three_objects_one_camera
    K = len(scene.objects)

    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])

    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    # Sanity: shape is (K, K) — OBJECTS ONLY, no cameras
    assert rel["obj_left"].shape == (K, K)
    assert rel["obj_right"].shape == (K, K)
    assert rel["obj_front"].shape == (K, K)
    assert rel["obj_behind"].shape == (K, K)

    desk, chair, lamp = 0, 1, 2

    # Desk faces -z so desk.right = -x. Chair is at +x → chair is in desk's
    # -right direction → chair is LEFT of desk → obj_left[chair, desk] HIGH.
    assert rel["obj_left"][chair, desk] > 0.7, (
        f"chair should be LEFT of desk (subject-first); got "
        f"obj_left[chair, desk]={rel['obj_left'][chair, desk]:.3f}"
    )
    assert rel["obj_right"][chair, desk] < 0.1, (
        f"chair should NOT be right of desk; got "
        f"obj_right[chair, desk]={rel['obj_right'][chair, desk]:.3f}"
    )

    # Symmetric check: lamp at -x → lamp is in desk's +right direction → RIGHT of desk.
    assert rel["obj_right"][lamp, desk] > 0.7
    assert rel["obj_left"][lamp, desk] < 0.1

    # And the swap test: obj_left[desk, chair] is "is desk left of chair (in
    # chair's frame)?"  Chair faces -z, so chair.right = -x, chair sits at +x.
    # Desk at origin is in chair's +right direction → desk is RIGHT of chair.
    # So obj_left[desk, chair] should be LOW.
    assert rel["obj_left"][desk, chair] < 0.1, (
        "Subject-first convention violated: swapping indices should flip the "
        "answer, not give the same value."
    )
    assert rel["obj_right"][desk, chair] > 0.7


def test_obj_front_is_subject_first(scene_three_objects_one_camera):
    """``obj_front[i, j]`` = "i is in front of j, from j's perspective".

    Desk faces -z. An entity in desk's +front (i.e. at z < 5) would be in
    front of desk. Lamp/chair are at z=5, same as desk → no front signal.
    Build a custom 2-object scene to isolate the front axis.
    """
    # Desk at z=5 facing -z. Probe at z=3 (so it sits in desk's +front direction).
    desk = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    probe = _make_object(1, [0, 0, 3], front=(0, 0, -1))

    positions = np.array([desk.center_world, probe.center_world])
    fronts = np.array([desk.front_world, probe.front_world])
    rights = np.array([desk.right_world, probe.right_world])

    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    # probe at z=3 < desk at z=5; desk faces -z → probe is in desk's +front direction.
    # obj_front[probe, desk] = "probe is in front of desk" → HIGH.
    assert rel["obj_front"][1, 0] > 0.7
    assert rel["obj_behind"][1, 0] < 0.1

    # Symmetrically: probe also faces -z, so from probe's perspective, desk
    # at z=5 is in probe's -front direction → desk is BEHIND probe.
    assert rel["obj_behind"][0, 1] > 0.7
    assert rel["obj_front"][0, 1] < 0.1


# ===========================================================================
# CONVENTION 2: obj_<dir> diagonal is zero
# ===========================================================================


def test_obj_dir_diagonal_is_zero(scene_three_objects_one_camera):
    """``obj_<dir>[i, i] == 0`` for every i.

    ``compute_obj_relative_from_axes`` skips ``i == j`` (``metrics.py:312``),
    so its diagonal is zero; the Anchor predicates use NaN on the self index
    instead.
    """
    scene = scene_three_objects_one_camera
    K = len(scene.objects)
    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])

    rel = compute_obj_relative_from_axes(positions, fronts, rights)
    for key in ("obj_left", "obj_right", "obj_front", "obj_behind"):
        diag = np.diag(rel[key])
        assert np.allclose(diag, 0.0), (
            f"{key} diagonal should be zero; got {diag}"
        )


# ===========================================================================
# CONVENTION 3: view.first_person index space includes cameras
# ===========================================================================


def test_first_person_array_length_is_K_plus_C(scene_three_objects_one_camera):
    """``view.first_person.<dir>`` returns a 1D array of length K + C.

    Per the docstring at ``frame.py:1364-1367``: "Returned arrays have length
    K + C where K = len(scene.objects) and C = len(scene.cameras). Camera c
    is at index K + c."
    """
    scene = scene_three_objects_one_camera
    K = len(scene.objects)
    C = len(scene.cameras)

    view = scene._frame(at=scene.cameras[0])

    for dir_name in ("left", "right", "front", "back"):
        arr = getattr(view.first_person, dir_name)
        # Accept ndarray OR ProbabilisticTensor (project's tensor wrapper)
        as_np = np.asarray(arr)
        assert as_np.ndim == 1, f"{dir_name} should be 1D, got shape {as_np.shape}"
        assert as_np.shape[0] == K + C, (
            f"{dir_name} length should be K+C={K + C}, got {as_np.shape[0]}"
        )


def test_first_person_camera_at_index_K_plus_c(scene_three_objects_one_camera):
    """Camera ``c`` is addressable at index ``K + c`` in first_person arrays.

    Probe: build a 2-camera scene where camera 1 is to the right of camera 0
    (looking the same way). Then from camera 0's first-person, camera 1's
    score on `right` should be HIGH at index K + 1.
    """
    cameras = [
        _make_camera([0, 0, 0], [0, 0, 1], cam_id=0),
        _make_camera([5, 0, 0], [0, 0, 1], cam_id=1),  # 5m to the right of cam 0
    ]
    objects = [_make_object(0, [0, 0, 5])]
    scene = Scene(objects=objects, cameras=cameras, images=[None])

    K = len(scene.objects)
    view = scene._frame(at=scene.cameras[0])

    # camera 1 (right of camera 0) is at index K + 1
    right_arr = np.asarray(view.first_person.right)
    cam1_right_score = float(right_arr[K + 1])
    assert cam1_right_score > 0.9, (
        f"camera 1 at index K+1={K + 1} should be far-right of camera 0; "
        f"got first_person.right[K+1]={cam1_right_score:.3f}"
    )

    # camera 1 on the LEFT score should be low
    left_arr = np.asarray(view.first_person.left)
    assert float(left_arr[K + 1]) < 0.1


# ===========================================================================
# CONVENTION 4: view.at(obj).first_person.<dir> agrees in SIGN with
#               obj_<dir>[k, obj_idx]
# ===========================================================================


def test_view_at_obj_first_person_agrees_with_obj_dir_in_sign(
    scene_three_objects_one_camera,
):
    """``view.at(obj).first_person.left[k]`` and ``obj_left[k, obj_idx]``
    answer the same question: "is k to the left of obj, in obj's body frame?"

    The two paths may score differently in absolute terms, but the argmax
    (which is what generated programs use) must agree.
    """
    scene = scene_three_objects_one_camera

    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])
    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    desk, chair, lamp = 0, 1, 2

    # From desk's perspective:
    #   obj_left[:, desk] should argmax at chair (chair is most left of desk).
    obj_left_col = rel["obj_left"][:, desk]
    # Mask self-index (which is 0 by convention).
    obj_left_col_masked = obj_left_col.copy()
    obj_left_col_masked[desk] = -np.inf
    obj_left_argmax = int(np.argmax(obj_left_col_masked))
    assert obj_left_argmax == chair

    # Same question via view.at(desk).first_person.left
    desk_view = scene._frame(at=scene.objects[desk])
    fp_left = np.asarray(desk_view.first_person.left)[: len(scene.objects)]
    fp_left_masked = fp_left.copy()
    fp_left_masked[desk] = -np.inf
    fp_left_argmax = int(np.argmax(fp_left_masked))

    assert fp_left_argmax == obj_left_argmax, (
        f"view.at(desk).first_person.left.argmax = obj_{fp_left_argmax} but "
        f"obj_left[:, desk].argmax = obj_{obj_left_argmax}. The two APIs "
        f"answer the same question and must agree on argmax."
    )


# ===========================================================================
# CONVENTION 5: MergedObject.rotate(yaw=...) is non-mutating
# ===========================================================================


def test_merged_object_rotate_is_non_mutating():
    """``obj.rotate(yaw=90)`` returns a new ``MergedObject`` with rotated axes;
    the original object is untouched. Pose-derived fields (``corners_world``,
    ``euler_world_deg``, ``rotation_world``) are updated atomically by
    ``_sync_pose``; semantic fields (``label``, ``dims``) carry through.
    """
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    front_before = obj.front_world.copy()
    label_before = obj.label
    dims_before = obj.dims.copy()

    rotated = obj.rotate(yaw=90)

    # rotate returns the same concrete type (no _OrientedView side-class).
    assert isinstance(rotated, MergedObject)
    assert rotated is not obj

    # Original is unchanged
    assert np.allclose(obj.front_world, front_before)

    # Rotated has a different front axis. Axis vectors are ``*_vec``;
    # ``rotated.front`` is an Anchor predicate.
    assert not np.allclose(rotated.front_vec, front_before)

    # Semantic fields carried through unchanged.
    assert rotated.label == label_before
    np.testing.assert_allclose(rotated.dims, dims_before)


# ===========================================================================
# CONVENTION 6: first_person.right of an object at +x from a camera looking +z
# ===========================================================================


def test_first_person_cosine_known_geometry():
    """Anchor at origin facing +z. Target at +x scores ``right == 1.0`` and
    ``left == 0.0`` under the cosine formula.

    This pins ``first_person`` as cosine-scored.
    """
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    target = _make_object(0, [3, 0, 0], front=(0, 0, -1))  # straight to the right
    scene = Scene(objects=[target], cameras=[cam], images=[None])

    view = scene._frame(at=scene.cameras[0])
    fp = view.first_person
    target_idx = 0

    right_score = float(np.asarray(fp.right)[target_idx])
    left_score = float(np.asarray(fp.left)[target_idx])
    front_score = float(np.asarray(fp.front)[target_idx])
    back_score = float(np.asarray(fp.back)[target_idx])

    # Cosine: target at +x from anchor facing +z → yaw = +pi/2.
    # score(right)  = (1 + cos(pi/2 - pi/2)) / 2 = 1.0
    # score(left)   = (1 + cos(pi/2 - (-pi/2))) / 2 = (1 + cos(pi))/2 = 0.0
    # score(front)  = (1 + cos(pi/2 - 0)) / 2 = (1 + 0)/2 = 0.5
    # score(back)   = (1 + cos(pi/2 - pi)) / 2 = (1 + 0)/2 = 0.5
    assert right_score == pytest.approx(1.0, abs=0.01)
    assert left_score == pytest.approx(0.0, abs=0.01)
    assert front_score == pytest.approx(0.5, abs=0.01)
    assert back_score == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# CONVENTION 7: obj_right and obj_behind are subject-first (mirror)
# ===========================================================================


def test_obj_right_is_subject_first(scene_three_objects_one_camera):
    """``obj_right[i, j]`` = "i is to the right of j, from j's perspective".
    Mirrors the obj_left test on the +right axis to catch per-axis inversions.
    """
    scene = scene_three_objects_one_camera
    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])
    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    desk, chair, lamp = 0, 1, 2
    # Desk faces -z → desk.right = -x. Lamp at -x → in desk's +right → RIGHT of desk.
    assert rel["obj_right"][lamp, desk] > 0.7
    assert rel["obj_left"][lamp, desk] < 0.1
    # Swap: lamp.right = -x. Desk at +x → in lamp's -right → LEFT of lamp.
    assert rel["obj_left"][desk, lamp] > 0.7
    assert rel["obj_right"][desk, lamp] < 0.1


def test_obj_behind_is_subject_first():
    """``obj_behind[i, j]`` = "i is behind j, from j's perspective".
    Uses an isolated 2-object scene to keep the +front/-front axis clean."""
    j = _make_object(0, [0, 0, 5], front=(0, 0, -1))  # faces -z
    i = _make_object(1, [0, 0, 7], front=(0, 0, -1))  # behind j along j's -front
    positions = np.array([j.center_world, i.center_world])
    fronts = np.array([j.front_world, i.front_world])
    rights = np.array([j.right_world, i.right_world])
    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    # i at z=7; j at z=5 facing -z. (pos_i - pos_j) = +z, j's front = -z, so
    # i lies in j's -front direction → i is BEHIND j → obj_behind[i, j] high.
    assert rel["obj_behind"][1, 0] > 0.7
    assert rel["obj_front"][1, 0] < 0.1


# ===========================================================================
# CONVENTION 8: obj_<dir> scores live in [0, 1]
# ===========================================================================


def test_obj_dir_scores_in_unit_interval(scene_three_objects_one_camera):
    """Every ``obj_<dir>`` entry is in [0, 1]."""
    scene = scene_three_objects_one_camera
    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])
    rel = compute_obj_relative_from_axes(positions, fronts, rights)
    for key in ("obj_left", "obj_right", "obj_front", "obj_behind"):
        mat = rel[key]
        assert np.all(mat >= 0.0), f"{key} has negative entries"
        assert np.all(mat <= 1.0), f"{key} has entries > 1.0"


# ===========================================================================
# CONVENTION 9: cosine opposite-pair invariant on first_person
# ===========================================================================


def test_first_person_opposite_pair_sums_to_one():
    """Under the cosine formula ``(1 + cos(yaw - target_yaw))/2``, opposite
    direction pairs (left/right and front/back) sum to 1 for ANY target.

    Formally: cos(yaw - 0) + cos(yaw - pi) = 0, so the scores add to 1.

    This is an invariant of the scoring formula, not a property of any
    specific geometry. Locking it in pins ``first_person`` as cosine-scored
    on every target, not just the synthetic axial case in test 6.
    """
    cam = _make_camera([0, 0, 0], [0, 0, 1])  # at origin, facing +z

    # Probe targets at arbitrary off-axis horizontal positions
    targets = [
        _make_object(0, [3, 0, 5]),    # front-right-ish
        _make_object(1, [-2, 0, 7]),   # front-left-ish
        _make_object(2, [5, 0, -1]),   # back-right-ish
    ]
    scene = Scene(objects=targets, cameras=[cam], images=[None])
    view = scene._frame(at=scene.cameras[0])

    front = np.asarray(view.first_person.front)
    back = np.asarray(view.first_person.back)
    left = np.asarray(view.first_person.left)
    right = np.asarray(view.first_person.right)

    K = len(targets)
    for k in range(K):
        assert float(front[k] + back[k]) == pytest.approx(1.0, abs=0.01), (
            f"front+back should sum to 1 at idx {k}; got {float(front[k]+back[k]):.3f}"
        )
        assert float(left[k] + right[k]) == pytest.approx(1.0, abs=0.01), (
            f"left+right should sum to 1 at idx {k}; got {float(left[k]+right[k]):.3f}"
        )


# ===========================================================================
# CONVENTION 10: first_person diagonals (front_right, back_left, etc.)
# ===========================================================================


def test_first_person_diagonal_attribute_names():
    """``view.first_person.front_right[k]`` works (underscore form is
    canonicalized to ``front-right``). Target sitting exactly along the
    front-right diagonal scores 1.0 there and 0.0 on back-left."""
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    # Target at (1, 0, 1) from cam facing +z → yaw = +pi/4 = front-right exactly.
    target = _make_object(0, [1, 0, 1])
    scene = Scene(objects=[target], cameras=[cam], images=[None])
    view = scene._frame(at=scene.cameras[0])

    fr = float(np.asarray(view.first_person.front_right)[0])
    bl = float(np.asarray(view.first_person.back_left)[0])
    br = float(np.asarray(view.first_person.back_right)[0])
    fl = float(np.asarray(view.first_person.front_left)[0])
    assert fr == pytest.approx(1.0, abs=0.01)
    assert bl == pytest.approx(0.0, abs=0.01)
    # back_right and front_left are 90° off the diagonal → score 0.5
    assert br == pytest.approx(0.5, abs=0.01)
    assert fl == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# CONVENTION 11: first_person vertical (above / below)
# ===========================================================================


def test_first_person_above_below_use_elevation():
    """``above`` / ``below`` use a different formula (elevation, not yaw).
    Target directly overhead scores ``above=1.0``, ``below=0.0``."""
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    # Target at (0, 3, 0) — straight up from cam.
    target = _make_object(0, [0, 3, 0])
    scene = Scene(objects=[target], cameras=[cam], images=[None])
    view = scene._frame(at=scene.cameras[0])

    above = float(np.asarray(view.first_person.above)[0])
    below = float(np.asarray(view.first_person.below)[0])
    assert above == pytest.approx(1.0, abs=0.01)
    assert below == pytest.approx(0.0, abs=0.01)
    # Vertical opposite-pair invariant.
    assert above + below == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# CONVENTION 12: first_person rejects "behind" (it belongs to third_person)
# ===========================================================================


def test_first_person_back_vs_behind_distinction(scene_three_objects_one_camera):
    """``view.first_person.back`` and ``view.first_person.behind`` BOTH work
    and return identical scores (the resolver aliases ``behind → back`` for
    the first-person namespace).

    The 2D pairwise ``view.behind[i, j]`` predicate ("i is occluded by j",
    observer semantics) lives on a separate namespace and is unaffected.
    Both spellings of "back" are accepted so codegen can use natural English
    ("what's behind me") interchangeably with the canonical "back".
    """
    scene = scene_three_objects_one_camera
    view = scene._frame(at=scene.cameras[0])
    back_scores = np.asarray(view.first_person.back)
    behind_scores = np.asarray(view.first_person.behind)
    np.testing.assert_allclose(back_scores, behind_scores, atol=1e-9)


def test_third_person_behind_works(scene_three_objects_one_camera):
    """``view.third_person.behind[i, j]`` returns a 2D pairwise matrix —
    distinct shape from ``first_person`` 1D arrays. Locks down that
    third_person is structurally different and stays so."""
    scene = scene_three_objects_one_camera
    view = scene._frame(at=scene.cameras[0])
    behind = np.asarray(view.third_person.behind)
    K = len(scene.objects)
    C = len(scene.cameras)
    # 2D pairwise matrix over the entity space
    assert behind.ndim == 2
    assert behind.shape[0] == behind.shape[1]
    # Camera-inclusive entity space
    assert behind.shape[0] >= K  # may include cameras at K..K+C-1
    assert behind.shape[0] <= K + C


# ===========================================================================
# CONVENTION 13: view.facing namespace returns one score per entity
# ===========================================================================


def test_view_facing_namespace_exists(scene_three_objects_one_camera):
    """``view.facing.<dir>[k]`` answers "does k face <dir> in the view's
    frame?" This smoke-checks that it returns a 1D array of the right length.
    """
    scene = scene_three_objects_one_camera
    K = len(scene.objects)
    C = len(scene.cameras)
    view = scene._frame(at=scene.cameras[0])
    for dir_name in ("front", "back", "left", "right"):
        arr = np.asarray(getattr(view.facing, dir_name))
        assert arr.ndim == 1
        # facing is K-only (cameras don't have orientation in the facing model)
        # OR K+C — accept both.
        assert arr.shape[0] in (K, K + C), (
            f"view.facing.{dir_name}.shape={arr.shape}; expected length K={K} or K+C={K+C}"
        )


# ===========================================================================
# CONVENTION 14: empty scene → (0, 0) matrices
# ===========================================================================


def test_obj_dir_empty_scene_returns_zero_shape():
    """``compute_obj_relative_from_axes`` on K=0 returns ``(0, 0)`` matrices.
    Edge case that must not crash (e.g. a scene with only cameras)."""
    rel = compute_obj_relative_from_axes(
        positions=np.zeros((0, 3)),
        front_directions=np.zeros((0, 3)),
        right_directions=np.zeros((0, 3)),
    )
    for key in ("obj_left", "obj_right", "obj_front", "obj_behind"):
        assert rel[key].shape == (0, 0)


# ===========================================================================
# CONVENTION 15: view.at(obj).first_person.<dir> uses obj's intrinsic axes
# ===========================================================================


def test_view_at_object_uses_objects_intrinsic_axes():
    """``view.at(obj).first_person.<dir>`` answers "from obj's POV, in obj's
    body frame". Specifically, anchored at desk (facing -z, so desk's right
    = -x), a probe at +x should score `left=1.0` because +x is in desk's
    -right direction.

    Numerical (not just argmax) check: pins the cosine formula AND the
    intrinsic-axis choice.
    """
    desk = _make_object(0, [0, 0, 5], front=(0, 0, -1))  # right = -x
    probe = _make_object(1, [3, 0, 5])  # at +x relative to desk → desk's LEFT
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    scene = Scene(objects=[desk, probe], cameras=[cam], images=[None])

    desk_view = scene._frame(at=scene.objects[0])
    fp_left = float(np.asarray(desk_view.first_person.left)[1])  # probe idx = 1
    fp_right = float(np.asarray(desk_view.first_person.right)[1])

    # Cosine: probe at +x; desk's right = -x; yaw of probe in desk's frame
    # is atan2(+x · desk.right, +x · desk.front) = atan2(-1, 0) = -pi/2
    # → target on the "left" target_yaw=-pi/2 → score = 1.0.
    assert fp_left == pytest.approx(1.0, abs=0.01), (
        f"view.at(desk).first_person.left[probe] should be 1.0 (probe at +x is "
        f"in desk's -right = LEFT direction); got {fp_left:.3f}"
    )
    assert fp_right == pytest.approx(0.0, abs=0.01)


# ===========================================================================
# CONVENTION 16: multiple cameras give distinct first_person answers
# ===========================================================================


def test_multiple_cameras_have_independent_first_person():
    """Two cameras at different positions facing the SAME direction
    classify the same target into DIFFERENT directions in their own frames.
    Locks down per-camera frame independence — the predicates are stored
    jointly in (K+C, K+C), and column c must be camera c's frame, not a
    copy of camera 0's.
    """
    cam_a = _make_camera([0, 0, 0], [0, 0, 1], cam_id=0)   # at origin, facing +z
    cam_b = _make_camera([6, 0, 5], [0, 0, 1], cam_id=1)   # to the right of A, facing +z
    target = _make_object(0, [3, 0, 5])  # in front of A, to the left of B
    scene = Scene(objects=[target], cameras=[cam_a, cam_b], images=[None])

    view_a = scene._frame(at=scene.cameras[0])
    view_b = scene._frame(at=scene.cameras[1])

    a_right = float(np.asarray(view_a.first_person.right)[0])
    b_left = float(np.asarray(view_b.first_person.left)[0])

    # From cam_a (origin, facing +z): target at (3,0,5) is mostly front-right.
    assert a_right > 0.7, f"target should be to A's right; got {a_right:.3f}"

    # From cam_b (at +x=6, facing +z): target at x=3 is to B's LEFT.
    assert b_left > 0.7, f"target should be to B's left; got {b_left:.3f}"
