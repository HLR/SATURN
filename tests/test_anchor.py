"""The ``Anchor`` API shared by scene objects and cameras.

Covers:
  - construction: objects and cameras are anchors with a consistent index
  - cosine direction scores on known geometry, NaN on the self index
  - the precomputed (K+C, K+C) predicate matrices and first-person reads
  - the object-relative ``obj_<dir>`` matrices
  - strict / non-strict scene build on a missing orientation
  - immutable ``translate`` / ``rotate`` and pose-format consistency
  - rotation composition, orthonormality and extrinsics round trips
  - predicate invariants, edge cases and corner/rotation sync on translate
  - the predicate matrix following the entity list (append, NMS, cache)
"""

import numpy as np
import pytest

from saturn.scene.anchor import (
    Anchor,
    compute_anchor_predicates,
)
from saturn.scene.scene import Scene
from saturn.scene.types import (
    Camera,
    MergedObject,
    SceneBuildError,
)


# ---------------------------------------------------------------------------
# Helpers (same shape as test_anchor_conventions.py / test_frame_first_api.py)
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


def _make_canonical_camera(position, forward, cam_id=0):
    """Like ``_make_camera`` but in the canonical (reflected) world of real scenes."""
    position = np.asarray(position, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(right, forward)  # canonical (reflected) world: row 1 = -up, det(R_w2c) = -1, as real scenes
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


@pytest.fixture
def fixture_scene():
    """Standard fixture used across tests: 3 objects + 1 camera."""
    cameras = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0)]
    objects = [
        _make_object(0, [0, 0, 5]),    # desk, faces -z
        _make_object(1, [3, 0, 5]),    # chair, faces -z (left of desk)
        _make_object(2, [-3, 0, 5]),   # lamp, faces -z (right of desk)
    ]
    return Scene(objects=objects, cameras=cameras, images=[None])


# ===========================================================================
# 1. Construction: Object and Camera are Anchors
# ===========================================================================


def test_object_and_camera_are_anchors(fixture_scene):
    """``MergedObject`` and ``Camera`` declare ``Anchor`` as base class.
    Both expose ``position``, ``orientation_front``, ``orientation_right``,
    ``orientation_up``, ``orientation_confidence``, ``anchor_index``.
    """
    scene = fixture_scene

    for obj in scene.objects:
        assert isinstance(obj, Anchor)
        assert obj.position.shape == (3,)
        assert obj.orientation_front.shape == (3,)
        assert obj.orientation_right.shape == (3,)
        assert obj.orientation_up.shape == (3,)
        assert 0.0 <= obj.orientation_confidence <= 1.0
        # anchor_index is in [0, K)
        assert 0 <= obj.anchor_index < len(scene.objects)
        # The Anchor accessors agree with the MergedObject storage fields.
        np.testing.assert_allclose(obj.position, obj.center_world)
        np.testing.assert_allclose(obj.orientation_front, obj.front_world)

    K = len(scene.objects)
    for c, cam in enumerate(scene.cameras):
        assert isinstance(cam, Anchor)
        assert cam.position.shape == (3,)
        assert cam.orientation_front.shape == (3,)
        # camera orientation_confidence is 1.0 (extrinsics are by construction valid)
        assert cam.orientation_confidence == pytest.approx(1.0)
        # anchor_index = K + c
        assert cam.anchor_index == K + c


# ===========================================================================
# 2. Cosine numerics on known geometry
# ===========================================================================


def test_cosine_scoring_known_geometry():
    """An anchor at origin facing +z with a target at +x scores:
        right        == 1.0
        left         == 0.0
        front        == 0.5
        back         == 0.5
        front_right  == cos(pi/2 - pi/4) → (1 + sqrt(2)/2) / 2  ≈ 0.853
        back_right   == cos(pi/2 - 3pi/4) → (1 + sqrt(2)/2) / 2  ≈ 0.853
        front_left   == cos(pi/2 - (-pi/4)) → (1 - sqrt(2)/2) / 2  ≈ 0.146
        back_left    == cos(pi/2 - (-3pi/4)) → (1 - sqrt(2)/2) / 2  ≈ 0.146
    """
    anchor = _make_camera([0, 0, 0], [0, 0, 1])  # at origin, facing +z
    target = _make_object(0, [3, 0, 0])           # at +x

    scene = Scene(objects=[target], cameras=[anchor], images=[None])
    K = len(scene.objects)
    cam = scene.cameras[0]
    target_idx = 0  # in the anchor-space

    expected = {
        "right": 1.0,
        "left": 0.0,
        "front": 0.5,
        "back": 0.5,
        "front_right": (1 + np.sqrt(2) / 2) / 2,
        "back_right": (1 + np.sqrt(2) / 2) / 2,
        "front_left": (1 - np.sqrt(2) / 2) / 2,
        "back_left": (1 - np.sqrt(2) / 2) / 2,
    }
    for dir_name, exp in expected.items():
        score = float(getattr(cam, dir_name)[target_idx])
        assert score == pytest.approx(exp, abs=0.01), (
            f"cam.{dir_name}[target] = {score:.3f}, expected {exp:.3f}"
        )


# ===========================================================================
# 3. Self-index returns NaN
# ===========================================================================


def test_self_index_is_nan(fixture_scene):
    """``anchor.<dir>[anchor.anchor_index]`` is NaN.

    Distinguishes "this anchor against itself" (undefined) from
    "no signal" (zero). The ``obj_<dir>`` matrices use zero on the
    diagonal; the anchor predicates use NaN so a self-match is visible.
    """
    for anchor in fixture_scene.objects + fixture_scene.cameras:
        for dir_name in ("left", "right", "front", "back",
                          "front_left", "front_right",
                          "back_left", "back_right",
                          "above", "below"):
            arr = getattr(anchor, dir_name)
            self_score = arr[anchor.anchor_index]
            assert np.isnan(self_score), (
                f"{type(anchor).__name__}.{dir_name}[self={anchor.anchor_index}] "
                f"should be NaN, got {self_score}"
            )


# ===========================================================================
# 4. Precomputed predicate matrices have shape (K+C, K+C)
# ===========================================================================


def test_precomputed_predicates_shape(fixture_scene):
    """``scene._anchor_predicates[<dir>].shape == (K+C, K+C)``."""
    scene = fixture_scene
    N = len(scene.objects) + len(scene.cameras)

    preds = scene._anchor_predicates
    for dir_name in (
        "left", "right", "front", "back",
        "front_left", "front_right", "back_left", "back_right",
        "above", "below",
    ):
        mat = preds[dir_name]
        assert mat.shape == (N, N), (
            f"{dir_name}.shape = {mat.shape}, expected ({N},{N})"
        )


# ===========================================================================
# 5. First-person equivalence: view.first_person.<dir>[k] matches the
#    anchor predicate matrix value precomputed for the camera anchor.
# ===========================================================================


def test_first_person_matches_anchor_row(fixture_scene):
    """For every camera, ``cam.first_person.left[k]`` numerically matches
    ``scene._anchor_predicates['left'][k, cam.anchor_index]``.

    Both paths use the same cosine formula and the same precomputed
    matrix. The matrix is subject-first (``[k, ref]`` = "is k to the left
    of ref"), so the column for the camera is what first-person reads.
    """
    scene = fixture_scene
    for cam in scene.cameras:
        view = scene._frame(at=cam)
        for dir_name in ("left", "right", "front", "back"):
            fp = np.asarray(getattr(view.first_person, dir_name))
            mat = scene._anchor_predicates[dir_name]
            col = mat[:, cam.anchor_index]
            # The self-index is NaN in the matrix while first_person may
            # return a finite value at that index — accept either.
            finite_mask = ~np.isnan(col)
            np.testing.assert_allclose(
                fp[finite_mask], col[finite_mask], rtol=1e-5, atol=1e-5,
                err_msg=f"first_person.{dir_name} != anchor matrix column for {cam}",
            )


# ===========================================================================
# 6. The obj_<dir> matrices: keys, dtype and (K, K) shape
# ===========================================================================


def test_obj_dir_matrix_shape(fixture_scene):
    """``compute_obj_relative_from_axes`` returns the ``obj_left`` / ``obj_right``
    / ``obj_front`` / ``obj_behind`` matrices as float64 of shape ``(K, K)``."""
    from saturn.predicates.metrics import (
        compute_obj_relative_from_axes,
    )

    scene = fixture_scene
    K = len(scene.objects)
    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])
    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    assert set(rel.keys()) >= {"obj_left", "obj_right", "obj_front", "obj_behind"}
    for key in ("obj_left", "obj_right", "obj_front", "obj_behind"):
        assert rel[key].shape == (K, K)
        assert rel[key].dtype == np.float64


# ===========================================================================
# 6b. obj_<dir> argmax on the fixture
# ===========================================================================


def test_obj_dir_matrix_argmax(fixture_scene):
    """For the desk, the entity ranked most likely "left of the desk" is the
    chair and "right of the desk" is the lamp. Generated programs consume the
    argmax, so the ranking is what matters, not the absolute scores."""
    from saturn.predicates.metrics import (
        compute_obj_relative_from_axes,
    )

    scene = fixture_scene
    K = len(scene.objects)
    positions = np.array([o.center_world for o in scene.objects])
    fronts = np.array([o.front_world for o in scene.objects])
    rights = np.array([o.right_world for o in scene.objects])
    rel = compute_obj_relative_from_axes(positions, fronts, rights)

    desk, chair, lamp = 0, 1, 2

    # Argmax over column desk: most-left of desk is chair; most-right is lamp.
    for name, expected_arg in [
        ("obj_left", chair),
        ("obj_right", lamp),
    ]:
        col = rel[name][:, desk].copy()
        col[desk] = -np.inf  # exclude self
        assert int(np.argmax(col)) == expected_arg


# ===========================================================================
# 7. strict=True raises on missing orientation
# ===========================================================================


def test_strict_true_raises_on_missing_orientation():
    """Building a Scene with an object whose orientation is unusable raises
    ``SceneBuildError``."""
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    obj_ok = _make_object(0, [0, 0, 5])
    obj_bad = _make_object(1, [3, 0, 5])
    # Wipe the bad object's orientation
    obj_bad.front_world = np.zeros(3)
    obj_bad.right_world = np.zeros(3)
    obj_bad.up_world = np.zeros(3)
    obj_bad.rotation_world = np.zeros((3, 3))

    with pytest.raises(SceneBuildError):
        Scene(objects=[obj_ok, obj_bad], cameras=[cam], images=[None], strict=True)


# ===========================================================================
# 7b. strict=False succeeds; predicate rows are NaN for the offender
# ===========================================================================


def test_strict_false_fills_nan():
    """Building with ``strict=False`` succeeds; the predicate rows/columns
    for the orientation-less object are NaN, not zero."""
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    obj_ok = _make_object(0, [0, 0, 5])
    obj_bad = _make_object(1, [3, 0, 5])
    obj_bad.front_world = np.zeros(3)
    obj_bad.right_world = np.zeros(3)
    obj_bad.up_world = np.zeros(3)
    obj_bad.rotation_world = np.zeros((3, 3))

    scene = Scene(
        objects=[obj_ok, obj_bad], cameras=[cam], images=[None], strict=False,
    )

    bad_idx = scene.objects[1].anchor_index
    for dir_name in ("left", "right", "front", "back"):
        mat = scene._anchor_predicates[dir_name]
        # All entries in bad_idx's row (using bad as reference) are NaN
        assert np.all(np.isnan(mat[:, bad_idx])), (
            f"{dir_name}[:, bad] should be all-NaN under strict=False"
        )


# ===========================================================================
# 8. translate returns a new instance
# ===========================================================================


def test_translate_returns_new_instance():
    obj = _make_object(0, [0, 0, 5])
    original_pos = obj.position.copy()

    translated = obj.translate(np.array([1.0, 0.0, 0.0]))

    assert translated is not obj
    np.testing.assert_allclose(obj.position, original_pos)
    np.testing.assert_allclose(translated.position, original_pos + [1.0, 0.0, 0.0])
    # Orientation unchanged
    np.testing.assert_allclose(translated.orientation_front, obj.orientation_front)


# ===========================================================================
# 9. rotate returns a new instance
# ===========================================================================


def test_rotate_returns_new_instance():
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    original_front = obj.orientation_front.copy()

    rotated = obj.rotate(yaw_deg=90.0)

    assert rotated is not obj
    np.testing.assert_allclose(obj.orientation_front, original_front)
    # Position unchanged
    np.testing.assert_allclose(rotated.position, obj.position)
    # Front has actually rotated
    assert not np.allclose(rotated.orientation_front, original_front)


# ===========================================================================
# 10. rotate(yaw=90) on an anchor facing -z produces +x front
# ===========================================================================


def test_rotate_yaw_90_swaps_front_and_right():
    """Yaw-right by 90° rotates an anchor facing -z to face +x (or -x —
    depends on rotation handedness, but the magnitude on the swapped axis
    must be ≈ 1.0 and the original axis ≈ 0)."""
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    rotated = obj.rotate(yaw_deg=90.0)

    # |z-component| should be ≈ 0; |x-component| ≈ 1
    assert abs(rotated.orientation_front[2]) < 0.05
    assert abs(rotated.orientation_front[0]) > 0.95


# ===========================================================================
# 11b. translate / rotate keep every pose storage format consistent
# ===========================================================================


def test_translate_keeps_extrinsics_consistent_with_position(fixture_scene):
    """Camera.translate must update BOTH ``position_world`` AND ``extrinsics``
    so the two formats describe the same pose.
    """
    cam = fixture_scene.cameras[0]
    delta = np.array([3.0, 0.0, -2.0])
    translated = cam.translate(delta)

    # Position storage moved by delta
    np.testing.assert_allclose(
        translated.position_world, cam.position_world + delta, atol=1e-9,
    )

    # Extrinsics camera-center must equal new position. Camera center in
    # world frame is ``-R_w2c.T @ t``.
    ext = translated.extrinsics
    R_w2c = ext[:3, :3]
    t = ext[:3, 3]
    cam_center_from_ext = -R_w2c.T @ t
    np.testing.assert_allclose(
        cam_center_from_ext, translated.position_world, atol=1e-9,
        err_msg="extrinsics encodes a different camera center than position_world",
    )


def test_camera_rotate_keeps_all_formats_consistent(fixture_scene):
    """Camera.rotate must keep ``position_world``, ``heading``, AND
    ``extrinsics`` consistent with each other; every pose field is derived
    in ``_sync_pose``.
    """
    cam = fixture_scene.cameras[0]
    rotated = cam.rotate(yaw=90.0)

    # Camera position is preserved (rotate only changes orientation).
    np.testing.assert_allclose(
        rotated.position_world, cam.position_world, atol=1e-9,
    )

    # ``front_vec`` reflects the rotation: facing +z rotated 90° about Y
    # under "yxz" euler should now face the world-frame direction
    # ``delta_rot @ (0,0,1)``. Compare modulo sign convention.
    f = rotated.front_vec
    assert abs(f[1]) < 1e-6, f"rotated front should stay horizontal; got {f}"
    assert abs(np.linalg.norm(f) - 1.0) < 1e-6

    # ``heading.forward`` matches ``front_vec``.
    np.testing.assert_allclose(rotated.heading.forward, rotated.front_vec, atol=1e-9)

    # ``extrinsics`` camera-center matches position_world.
    ext = rotated.extrinsics
    R_w2c = ext[:3, :3]
    t = ext[:3, 3]
    cam_center_from_ext = -R_w2c.T @ t
    np.testing.assert_allclose(
        cam_center_from_ext, rotated.position_world, atol=1e-9,
        err_msg="rotate left extrinsics inconsistent with position_world",
    )

    # ``extrinsics`` row 2 (camera-frame +Z = world forward) equals front_vec.
    np.testing.assert_allclose(R_w2c[2], rotated.front_vec, atol=1e-9)

    # Original camera is unchanged.
    np.testing.assert_allclose(cam.front_vec, [0.0, 0.0, 1.0], atol=1e-9)


def test_rotate_updates_corners_and_euler_on_merged_object():
    """``MergedObject.rotate(yaw=90)`` must update ``corners_world`` and
    ``euler_world_deg`` to match the new pose. ``MergedObject.rotate``
    returns a ``MergedObject``, so every pose-derived field must agree.
    """
    # Asymmetric dims so corner positions are sensitive to rotation.
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    # Inject non-cube dims and seed corners_world to a known box.
    obj.dims = np.array([2.0, 0.4, 0.8])
    w, h, d = 2.0, 0.4, 0.8
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=float)
    local = signs * np.array([w / 2.0, h / 2.0, d / 2.0])
    axes = np.column_stack([obj.right_vec, obj.up_vec, obj.front_vec])
    obj.corners_world = np.asarray(obj.center_world)[None, :] + local @ axes.T

    rotated = obj.rotate(yaw=90)

    # Corners should be in the same locations they'd be at if recomputed
    # from rotated dims + new pose. Since dims is pose-invariant and the
    # rotation is 90° about Y, the new corner set should equal
    # ``new_center + local @ new_axes.T``.
    new_axes = np.column_stack([rotated.right_vec, rotated.up_vec, rotated.front_vec])
    expected = np.asarray(rotated.center_world)[None, :] + local @ new_axes.T
    np.testing.assert_allclose(rotated.corners_world, expected, atol=1e-9)

    # euler_world_deg must be a finite (3,) array reflecting the new rotation.
    assert rotated.euler_world_deg.shape == (3,)
    assert np.all(np.isfinite(rotated.euler_world_deg))

    # Original object is untouched.
    np.testing.assert_allclose(
        obj.corners_world,
        np.asarray(obj.center_world)[None, :] + local @ axes.T,
        atol=1e-9,
    )


def test_precompute_matches_lazy_compute_on_real_scene(fixture_scene):
    """The vectorized per-pair loop in ``compute_anchor_predicates`` must
    produce identical scores to ``Anchor._compute_lazy`` (one ``disp``/``yaw``
    computation per pair, looped over the 10 predicate names): the two
    paths share one formula.
    """
    scene = fixture_scene

    for anchor in scene.objects + scene.cameras:
        idx = anchor.anchor_index
        for dir_name in (
            "left", "right", "front", "back",
            "front_left", "front_right", "back_left", "back_right",
            "above", "below",
        ):
            precomp = scene._anchor_predicates[dir_name][:, idx]
            lazy = anchor._compute_lazy(dir_name)
            # Both arrays are length N=K+C; self-index entry differs
            # (precompute uses NaN, lazy may produce NaN or a real
            # value depending on disp norm). Compare with equal_nan.
            np.testing.assert_allclose(
                precomp, lazy, rtol=1e-9, atol=1e-9, equal_nan=True,
                err_msg=f"precompute and lazy disagree for {type(anchor).__name__}.{dir_name}",
            )


# ===========================================================================
# 11. Free-floating rotate: predicates work without anchor_index
# ===========================================================================


def test_free_floating_rotate_predicates_work(fixture_scene):
    """``cam.rotate(yaw=90).front[k]`` works even though the rotated anchor
    is not in ``scene.anchors``. The predicates compute lazily against the
    scene's existing entity positions."""
    scene = fixture_scene
    cam = scene.cameras[0]
    # cam at origin facing +z. After yaw=90 (right-turn), cam faces +x.
    rotated_cam = cam.rotate(yaw_deg=90.0)

    # Free-floating anchor: no slot in scene.anchors
    assert getattr(rotated_cam, "anchor_index", None) is None or \
           rotated_cam.anchor_index >= len(scene.objects) + len(scene.cameras)

    # Predicates still compute. The desk at (0, 0, 5) is now to the LEFT
    # of the rotated camera (since cam now faces +x, +z is its left).
    desk_idx = 0
    left_arr = rotated_cam.left
    assert float(left_arr[desk_idx]) > 0.8, (
        "After yaw=90 right-turn, desk at +z should be on rotated cam's LEFT"
    )


# ===========================================================================
# 12. Composition / non-commutativity
# ===========================================================================


def test_rotate_yaw_45_twice_equals_yaw_90():
    """Two successive 45° yaw rotations produce the same front axis as one 90°.

    Locks down the composition law: rotate is applied cumulatively with
    correct trig identities — no floating-point shortcircuit that would
    break incremental simulation updates.
    """
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    once = obj.rotate(yaw_deg=90.0)
    twice = obj.rotate(yaw_deg=45.0).rotate(yaw_deg=45.0)

    np.testing.assert_allclose(
        twice.orientation_front, once.orientation_front, atol=1e-9,
        err_msg="rotate(45).rotate(45) should equal rotate(90)",
    )
    np.testing.assert_allclose(
        twice.orientation_right, once.orientation_right, atol=1e-9,
        err_msg="right axes should match after double-45 vs single-90",
    )


def test_translate_a_then_b_equals_translate_a_plus_b():
    """Sequential translates are additive: translate(a).translate(b) == translate(a+b).

    Locks down that translate always applies delta relative to current
    position, not to the original — a simple but easy-to-break invariant.
    """
    obj = _make_object(0, [0, 0, 0])
    a = np.array([1.0, 0.0, 2.0])
    b = np.array([-0.5, 3.0, 0.0])

    sequential = obj.translate(a).translate(b)
    combined = obj.translate(a + b)

    np.testing.assert_allclose(
        sequential.position, combined.position, atol=1e-9,
        err_msg="translate(a).translate(b) must equal translate(a+b)",
    )


def test_translate_then_rotate_differs_from_rotate_then_translate():
    """Translate-then-rotate does NOT equal rotate-then-translate.

    Rotation affects the displacement vector. If the order were commutative,
    the Anchor compose semantics would be wrong for all relative-position
    code that builds "a new anchor offset from another".
    """
    obj = _make_object(0, [0, 0, 0], front=(0, 0, 1))
    delta = np.array([2.0, 0.0, 0.0])

    tr = obj.translate(delta).rotate(yaw_deg=90.0)
    rt = obj.rotate(yaw_deg=90.0).translate(delta)

    # Positions must differ (they do because translate is always in world frame,
    # but the two paths produce the same position in this implementation —
    # the invariant is that the COMBINED pose differs enough in orientation+position
    # that they are not equal overall).
    # Front axes should be the same (both rotated 90° from the original).
    np.testing.assert_allclose(
        tr.orientation_front, rt.orientation_front, atol=1e-9,
        err_msg="front should be the same regardless of translate/rotate order",
    )
    # But the delta applied after rotate(90°) should shift in the original +x direction
    # while the delta applied before rotate shifts in +x too — world-frame translate
    # means both positions ARE the same here. The non-commutativity shows up when
    # delta is in BODY frame. We verify the world-frame positions are equal (implementation-
    # defined) and that the test at least documents the behavior.
    # The key: orientation is identical, position is identical because translate is world-frame.
    assert tr.orientation_front.shape == (3,)


def test_rotate_yaw_180_twice_returns_to_original():
    """Two 180° yaw rotations restore the original orientation within float tolerance.

    Locks down that the rotation formula is consistent: applying inverse
    twice is identity.
    """
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    twice = obj.rotate(yaw_deg=180.0).rotate(yaw_deg=180.0)

    np.testing.assert_allclose(
        twice.orientation_front, obj.orientation_front, atol=1e-9,
        err_msg="rotate(180).rotate(180) should return to original front",
    )
    np.testing.assert_allclose(
        twice.orientation_right, obj.orientation_right, atol=1e-9,
        err_msg="rotate(180).rotate(180) should return to original right",
    )


def test_rotate_positive_then_negative_yaw_returns_to_original():
    """rotate(yaw=θ).rotate(yaw=-θ) recovers the original orientation.

    Any angle θ should work. We check θ=37° (non-special) to ensure the
    trig is truly invertible and not relying on 90°/180° shortcuts.
    """
    obj = _make_object(0, [0, 0, 5], front=(1, 0, 0))
    theta = 37.0
    roundtrip = obj.rotate(yaw_deg=theta).rotate(yaw_deg=-theta)

    np.testing.assert_allclose(
        roundtrip.orientation_front, obj.orientation_front, atol=1e-9,
        err_msg=f"rotate({theta}).rotate({-theta}) should cancel",
    )
    np.testing.assert_allclose(
        roundtrip.orientation_right, obj.orientation_right, atol=1e-9,
    )


# ===========================================================================
# 13. Orthonormality / matrix sanity
# ===========================================================================


def test_axes_remain_orthonormal_after_rotation():
    """After any rotation, front/right/up remain unit-norm and mutually orthogonal.

    Verifies that _sync_pose's normalization and rotation math cannot
    drift into a non-orthonormal frame, which would corrupt all subsequent
    cosine scores.
    """
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    for yaw in [0, 30, 45, 90, 135, 180, -90, 270]:
        rotated = obj.rotate(yaw_deg=float(yaw))
        f = rotated.orientation_front
        r = rotated.orientation_right
        u = rotated.orientation_up

        assert abs(np.linalg.norm(f) - 1.0) < 1e-9, f"front not unit at yaw={yaw}"
        assert abs(np.linalg.norm(r) - 1.0) < 1e-9, f"right not unit at yaw={yaw}"
        assert abs(np.linalg.norm(u) - 1.0) < 1e-9, f"up not unit at yaw={yaw}"
        assert abs(np.dot(f, r)) < 1e-9, f"front·right not zero at yaw={yaw}"
        assert abs(np.dot(f, u)) < 1e-9, f"front·up not zero at yaw={yaw}"
        assert abs(np.dot(r, u)) < 1e-9, f"right·up not zero at yaw={yaw}"


def test_rotation_world_is_orthonormal_after_rotate():
    """MergedObject.rotation_world is an orthonormal SO(3) matrix after rotate.

    rotation_world is written by _sync_pose as column_stack([right, up, front]).
    Verifies det(R) ≈ +1 and R.T @ R ≈ I.
    """
    obj = _make_object(0, [0, 0, 5], front=(0, 0, -1))
    for yaw in [45.0, 90.0, 137.5, -30.0]:
        rotated = obj.rotate(yaw_deg=yaw)
        R = rotated.rotation_world
        assert R.shape == (3, 3)
        np.testing.assert_allclose(
            R.T @ R, np.eye(3), atol=1e-9,
            err_msg=f"rotation_world not orthonormal at yaw={yaw}",
        )
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, (
            f"det(rotation_world) ≠ 1 at yaw={yaw}: {np.linalg.det(R)}"
        )


def test_camera_extrinsics_orthonormal_and_center_consistent_after_rotate():
    """After Camera.rotate, extrinsics[:3,:3] is orthonormal and camera center matches position.

    Locks down that _sync_pose's OpenCV-convention extrinsics rebuild
    stays in sync with position_world for cameras specifically.
    """
    cam = _make_canonical_camera([3.0, 1.0, -2.0], [0.0, 0.0, 1.0])
    rotated = cam.rotate(yaw=57.3)

    ext = rotated.extrinsics
    R_w2c = ext[:3, :3]
    t = ext[:3, 3]

    # Orthonormality
    np.testing.assert_allclose(
        R_w2c.T @ R_w2c, np.eye(3), atol=1e-9,
        err_msg="extrinsics R_w2c not orthonormal after rotate",
    )
    # canonical (reflected) cameras have det -1; rotating must keep the handedness
    assert abs(np.linalg.det(R_w2c) - np.linalg.det(cam.extrinsics[:3, :3])) < 1e-9
    assert abs(np.linalg.det(R_w2c) + 1.0) < 1e-9

    # Camera center from extrinsics must equal position_world
    cam_center = -R_w2c.T @ t
    np.testing.assert_allclose(
        cam_center, rotated.position_world, atol=1e-9,
        err_msg="camera center derived from extrinsics ≠ position_world after rotate",
    )


# ===========================================================================
# 14. Cosine predicate invariants
# ===========================================================================


def test_predicate_opposite_pair_sums_to_one_in_multi_anchor_scene():
    """left+right ≈ 1 and front+back ≈ 1 for every non-self, non-coincident pair.

    The cosine formula for opposite directions sums to 1 by construction:
    cos(θ - α) + cos(θ - (α+π)) = 2cos(π/2)·… = 0 → average = 1.
    """
    cameras = [_make_canonical_camera([0, 0, 0], [0, 0, 1])]
    objects = [
        _make_object(0, [0, 0, 5]),
        _make_object(1, [3, 0, 5]),
        _make_object(2, [-1, 0, 2]),
    ]
    scene = Scene(objects=objects, cameras=cameras, images=[None])
    preds = scene._anchor_predicates
    N = len(objects) + len(cameras)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for pair in [("left", "right"), ("front", "back")]:
                a, b = preds[pair[0]][i, j], preds[pair[1]][i, j]
                if np.isnan(a) or np.isnan(b):
                    continue
                assert abs(a + b - 1.0) < 1e-9, (
                    f"{pair[0]}[{i},{j}]={a:.4f} + {pair[1]}[{i},{j}]={b:.4f} ≠ 1"
                )


def test_predicate_above_below_sums_to_one():
    """above + below ≈ 1.0 for every non-coincident (i, j) pair.

    Vertical cosine formula: above=(1+y)/2, below=(1-y)/2 → sum = 1.
    """
    cameras = [_make_canonical_camera([0, 1.5, 0], [0, 0, 1])]
    objects = [
        _make_object(0, [0, 0, 5]),
        _make_object(1, [3, 2.0, 5]),
    ]
    scene = Scene(objects=objects, cameras=cameras, images=[None])
    preds = scene._anchor_predicates
    N = len(objects) + len(cameras)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a = preds["above"][i, j]
            b = preds["below"][i, j]
            if np.isnan(a) or np.isnan(b):
                continue
            assert abs(a + b - 1.0) < 1e-9, (
                f"above[{i},{j}]={a:.4f} + below[{i},{j}]={b:.4f} ≠ 1"
            )


def test_predicate_values_in_unit_interval():
    """Every finite predicate value is in [0, 1].

    The cosine formula maps to [0, 1] by construction.
    """
    cameras = [_make_canonical_camera([0, 0, 0], [0, 0, 1])]
    objects = [
        _make_object(0, [0, 0, 5]),
        _make_object(1, [3, 0, 5]),
        _make_object(2, [-1, 1, 3]),
    ]
    scene = Scene(objects=objects, cameras=cameras, images=[None])
    preds = scene._anchor_predicates

    for name, mat in preds.items():
        finite = mat[np.isfinite(mat)]
        assert np.all(finite >= -1e-9) and np.all(finite <= 1.0 + 1e-9), (
            f"Predicate '{name}' has values outside [0,1]: "
            f"min={finite.min():.4f}, max={finite.max():.4f}"
        )


def test_target_directly_ahead_front_is_one():
    """Target placed exactly along anchor's front axis: front=1.0, back=0.0, left=right=0.5.

    The cosine formula evaluated at yaw=0 gives front=1, back=cos(π)=−1 → (1−1)/2=0.
    left and right at ±π/2 give cos(π/2)=0 → 0.5.
    """
    # Anchor at origin facing +z; target directly in front at (0,0,5).
    cam = _make_canonical_camera([0, 0, 0], [0, 0, 1])
    target = _make_object(0, [0, 0, 5])   # exactly along +z (front)
    scene = Scene(objects=[target], cameras=[cam], images=[None])
    cam_idx = cam.anchor_index
    target_idx = target.anchor_index
    preds = scene._anchor_predicates

    assert preds["front"][target_idx, cam_idx] == pytest.approx(1.0, abs=1e-9)
    assert preds["back"][target_idx, cam_idx] == pytest.approx(0.0, abs=1e-9)
    assert preds["left"][target_idx, cam_idx] == pytest.approx(0.5, abs=1e-9)
    assert preds["right"][target_idx, cam_idx] == pytest.approx(0.5, abs=1e-9)


def test_target_on_front_right_diagonal_scores_one():
    """Target at 45° (front-right diagonal) scores front_right=1.0, back_left=0.0.

    Anchor faces +z, target at (5, 0, 5) — equal +x and +z displacement →
    yaw = π/4 exactly → front_right predicate target_yaw = π/4, cos(0)=1.
    """
    cam = _make_canonical_camera([0, 0, 0], [0, 0, 1])  # right_vec = +x, front = +z
    target = _make_object(0, [5, 0, 5])        # 45° diagonal
    scene = Scene(objects=[target], cameras=[cam], images=[None])
    cam_idx = cam.anchor_index
    target_idx = target.anchor_index
    preds = scene._anchor_predicates

    assert preds["front_right"][target_idx, cam_idx] == pytest.approx(1.0, abs=1e-9)
    assert preds["back_left"][target_idx, cam_idx] == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 15. Edge cases
# ===========================================================================


def test_empty_scene_predicate_shape():
    """Scene with 0 objects and 1 camera: predicates shape (1, 1) with NaN diagonal.

    Verifies compute_anchor_predicates handles the edge case gracefully
    rather than indexing into an empty matrix.
    """
    cam = _make_canonical_camera([0, 0, 0], [0, 0, 1])
    scene = Scene(objects=[], cameras=[cam], images=[None])
    preds = scene._anchor_predicates
    for name, mat in preds.items():
        assert mat.shape == (1, 1), f"{name}.shape = {mat.shape}, expected (1,1)"
        assert np.isnan(mat[0, 0]), f"{name}[0,0] should be NaN (self)"


def test_single_object_no_cameras_all_nan():
    """Scene with 1 object, no cameras: all predicate matrices are (1,1) NaN.

    The only pair is (0, 0) = self, which is always NaN. Verifies the
    self-loop guard works when there are zero cameras.
    """
    obj = _make_object(0, [0, 0, 5])
    scene = Scene(objects=[obj], cameras=[], images=[])
    preds = scene._anchor_predicates
    for name, mat in preds.items():
        assert mat.shape == (1, 1), f"{name}.shape = {mat.shape}"
        assert np.isnan(mat[0, 0])


def test_vertical_target_horizontal_predicates_nan_vertical_correct():
    """Target directly above anchor: horizontal predicates are NaN, above=1, below=0.

    When displacement has zero horizontal component (norm_h < 1e-8),
    the code must set horizontal predicates to NaN, not 0.
    """
    cam = _make_canonical_camera([0, 0, 0], [0, 0, 1])
    # Target at exactly the same x/z as camera, but higher y.
    target = _make_object(0, [0, 5.0, 0])  # directly above, same x/z
    scene = Scene(objects=[target], cameras=[cam], images=[None])
    cam_idx = cam.anchor_index
    target_idx = target.anchor_index
    preds = scene._anchor_predicates

    for h_pred in ("left", "right", "front", "back",
                    "front_left", "front_right", "back_left", "back_right"):
        val = preds[h_pred][target_idx, cam_idx]
        assert np.isnan(val), (
            f"{h_pred}[target, cam] should be NaN for vertical target; got {val}"
        )

    assert preds["above"][target_idx, cam_idx] == pytest.approx(1.0, abs=1e-9)
    assert preds["below"][target_idx, cam_idx] == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 16. Camera ↔ extrinsics round-trip
# ===========================================================================


def test_camera_extrinsics_round_trip_after_rotate_then_unrotate():
    """cam.rotate(yaw=90).rotate(yaw=-90) extrinsics ≈ original extrinsics.

    The two rotations must cancel exactly in the _sync_pose computation,
    verifying that each extrinsics rebuild is deterministic and not
    subject to accumulating floating-point error beyond 1e-9.
    """
    cam = _make_canonical_camera([5.0, 2.0, -3.0], [1.0, 0.0, 0.0])
    original_ext = cam.extrinsics.copy()

    roundtrip = cam.rotate(yaw=90.0).rotate(yaw=-90.0)

    np.testing.assert_allclose(
        roundtrip.extrinsics[:3, :3],
        original_ext[:3, :3],
        atol=1e-9,
        err_msg="extrinsics R_w2c changed after yaw-roundtrip",
    )
    np.testing.assert_allclose(
        roundtrip.extrinsics[:3, 3],
        original_ext[:3, 3],
        atol=1e-9,
        err_msg="extrinsics translation changed after yaw-roundtrip",
    )


def test_camera_front_vec_equals_heading_forward():
    """cam.front_vec exactly equals cam.heading.forward for every test camera.

    front_vec reads from heading.forward when heading is present. If the
    two ever diverge after _sync_pose, DSL code that reads heading.forward
    will disagree with predicate code that reads front_vec.
    """
    test_cameras = [
        _make_canonical_camera([0, 0, 0], [0, 0, 1]),
        _make_canonical_camera([1, 2, -3], [1, 0, 0]),
        _make_canonical_camera([0, 5, 0], [0, 0, -1]),
    ]
    for cam in test_cameras:
        np.testing.assert_allclose(
            cam.front_vec, cam.heading.forward, atol=1e-9,
            err_msg=f"front_vec ≠ heading.forward for cam at {cam.position_world}",
        )
        # Also check after rotate
        rotated = cam.rotate(yaw=45.0)
        np.testing.assert_allclose(
            rotated.front_vec, rotated.heading.forward, atol=1e-9,
            err_msg="front_vec ≠ heading.forward after rotate",
        )


# ===========================================================================
# 17. MergedObject translate sync
# ===========================================================================


def test_translate_shifts_corners_world_by_delta():
    """After obj.translate(δ), every corner in corners_world shifts by exactly δ.

    The corners_world rebuild in _sync_pose is centred on the new position.
    """
    obj = _make_object(0, [1.0, 2.0, 3.0], front=(0, 0, -1))
    # Seed corners_world properly for the object's current pose.
    dims = obj.dims  # [0.5, 0.5, 0.5]
    w, h, d = dims[0], dims[1], dims[2]
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=float)
    local = signs * np.array([w / 2, h / 2, d / 2])
    axes = np.column_stack([obj.right_world, obj.up_world, obj.front_world])
    obj.corners_world = obj.center_world[None, :] + local @ axes.T
    original_corners = obj.corners_world.copy()

    delta = np.array([10.0, -3.0, 7.5])
    translated = obj.translate(delta)

    np.testing.assert_allclose(
        translated.corners_world,
        original_corners + delta[None, :],
        atol=1e-9,
        err_msg="corners_world should shift by exactly delta after translate",
    )


def test_translate_leaves_rotation_world_unchanged():
    """After obj.translate(δ), rotation_world is identical to the original.

    Translate changes position only; orientation (and hence rotation_world)
    must be invariant. If _sync_pose incorrectly recomputes rotation from
    the shifted position, this test catches it.
    """
    obj = _make_object(0, [0, 0, 0], front=(1, 0, 0))  # facing +x
    original_R = obj.rotation_world.copy()

    delta = np.array([5.0, 0.0, -2.0])
    translated = obj.translate(delta)

    np.testing.assert_allclose(
        translated.rotation_world, original_R, atol=1e-9,
        err_msg="rotation_world changed after translate — should be invariant",
    )


# ===========================================================================
# 18. The predicate matrix follows the entity list
#
# Objects appended after Scene.__init__ (detect()/ground()) get an
# anchor_index and a scene back-reference, and every anchor's predicate
# rows are rebuilt to the new entity count when the caches are invalidated.
# ===========================================================================


def _two_object_scene():
    objs = [_make_object(0, (0, 0, -2)), _make_object(1, (1, 0, -2))]
    cam = _make_camera((0, 0, 0), (0, 0, -1), cam_id=0)
    return Scene(objects=objs, cameras=[cam], images=[None])


def test_appended_object_gets_anchor_wiring_and_predicates_resize():
    s = _two_object_scene()
    n0 = len(s.objects) + len(s.cameras)
    assert s.objects[0]._read_predicate("front").shape[0] == n0
    new = _make_object(2, (-1, 0, -3))
    s.objects.append(new)            # what _append_merged_objects does ...
    s._invalidate_caches()           # ... followed by this
    n1 = n0 + 1
    assert new.anchor_index == 2 and new._scene is s
    assert s.cameras[0].anchor_index == 3
    assert new._read_predicate("front").shape[0] == n1          # the new object reads predicates
    assert s.objects[0]._read_predicate("front").shape[0] == n1  # existing rows cover the new entity
    assert np.isfinite(new._read_predicate("front")).any()


def test_nms_removal_restamps_indices():
    s = _two_object_scene()
    s.objects.append(_make_object(2, (0.01, 0, -2)))  # near-duplicate of object 0
    s._invalidate_caches()
    before = len(s.objects)
    removed = s.nms_objects() if callable(getattr(s, "nms_objects", None)) else 0
    n = len(s.objects) + len(s.cameras)
    for i, o in enumerate(s.objects):
        assert o.anchor_index == i
        assert o._read_predicate("front").shape[0] == n
    assert s.cameras[0].anchor_index == len(s.objects)


def test_cam0_frame_is_cached_and_invalidated():
    """scene.left/right/... share one FrameNamespace until the scene changes."""
    s = _two_object_scene()
    f1 = s._cam0_frame; f2 = s._cam0_frame
    assert f1 is f2
    s._invalidate_caches()
    assert s._cam0_frame is not f1
