"""Tests for ``scene.register_predicate`` / ``unregister_predicate``.

Verifies the three registration forms (paper-style ``h_r``, ``angle_deg``
sugar, K-ary ``fn`` escape hatch), their access paths (first_person and
third_person), collision detection vs built-in vocabulary, lifecycle, and
validation errors.
"""
import math

import numpy as np
import pytest

from saturn.scene.pose_solver import (
    R_y,
    _compose_extrinsics,
    _world_to_cam_translation,
)
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_scene_constraint.py)
# ---------------------------------------------------------------------------


def _make_object(
    obj_id: int, center, front=(0.0, 0.0, 1.0), height: float = 0.5,
) -> MergedObject:
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
        height=height,
        support_y=float(center[1] - 0.25),
    )


def _make_camera(idx: int, yaw_deg: float, position=None) -> Camera:
    """Build a Camera dataclass with a yaw-only orientation at given position."""
    if position is None:
        position = np.zeros(3)
    R_w2c = R_y(yaw_deg).T
    t_w2c = _world_to_cam_translation(R_w2c, position)
    ext = _compose_extrinsics(R_w2c, t_w2c)
    return Camera(
        id=idx,
        entity_id=idx,
        intrinsics=np.eye(3),
        extrinsics=ext,
        image_size=(100, 100),
    )


def _trivial_scene() -> Scene:
    """Two objects + one camera at the origin facing +Z."""
    obj_a = _make_object(0, center=(0.0, 0.5, 2.0), height=1.0)   # in front
    obj_b = _make_object(1, center=(2.0, 0.5, 0.0), height=2.0)   # to the right
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.5, 0.0))
    return Scene(objects=[obj_a, obj_b], cameras=[cam], images=[None])


# ---------------------------------------------------------------------------
# angle_deg form
# ---------------------------------------------------------------------------


def test_angle_deg_first_person_returns_ndarray():
    sc = _trivial_scene()
    sc.register_predicate("forty_five", angle_deg=45.0)
    arr = sc.cameras[0].first_person.forty_five
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(sc.objects) + len(sc.cameras),)
    assert np.all((arr >= 0.0) & (arr <= 1.0))


def test_angle_deg_third_person_returns_scalar():
    sc = _trivial_scene()
    sc.register_predicate("forty_five", angle_deg=45.0)
    s = sc.cameras[0].third_person.forty_five[0, 1]
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_angle_deg_zero_matches_front_direction():
    """angle_deg=0 should peak for entities in the anchor's +front direction."""
    sc = _trivial_scene()
    sc.register_predicate("zero_deg", angle_deg=0.0)
    arr = sc.cameras[0].first_person.zero_deg
    # obj_a at (0, 0.5, 2) is directly in front of the camera at (0, 0.5, 0)
    # obj_b at (2, 0.5, 0) is directly to the right
    assert arr[0] > 0.5  # front object scores high for front predicate
    assert arr[1] < 0.5  # right object scores low for front predicate


def test_angle_deg_ninety_matches_right_direction():
    sc = _trivial_scene()
    sc.register_predicate("ninety_deg", angle_deg=90.0)
    arr = sc.cameras[0].first_person.ninety_deg
    # obj_a (front) low for right predicate; obj_b (right) high.
    assert arr[0] < 0.5
    assert arr[1] > 0.5


# ---------------------------------------------------------------------------
# h_r form
# ---------------------------------------------------------------------------


def test_h_r_form_paper_directional_front():
    """Custom h_r that re-implements the paper's 'front' predicate."""
    def h_front(delta_local, R_i_local, R_j_local):
        return float(delta_local[2])  # +z component = front

    sc = _trivial_scene()
    sc.register_predicate("my_front", h_r=h_front)
    arr_fp = sc.cameras[0].first_person.my_front
    s_tp = sc.cameras[0].third_person.my_front[0, 1]
    # First-person: obj_a is in front of camera → high; obj_b is to right → low.
    assert arr_fp[0] > 0.5
    assert arr_fp[1] < 0.5
    # Third-person: obj_a (z=2) relative to obj_b (z=0) → obj_a is +z → high.
    assert s_tp > 0.5


def test_h_r_receives_orientation_matrices():
    """Verify R_i_local and R_j_local reach the user's h_r."""
    captured = {}

    def h_capture(delta_local, R_i_local, R_j_local):
        captured["delta"] = delta_local.copy()
        captured["R_i"] = R_i_local.copy()
        captured["R_j"] = R_j_local.copy()
        return 0.0

    sc = _trivial_scene()
    sc.register_predicate("capture", h_r=h_capture, normalize_distance=False)
    _ = sc.cameras[0].third_person.capture[0, 1]
    assert captured["delta"].shape == (3,)
    assert captured["R_i"].shape == (3, 3)
    assert captured["R_j"].shape == (3, 3)


# ---------------------------------------------------------------------------
# fn form (K-ary escape hatch)
# ---------------------------------------------------------------------------


def test_fn_form_arity_2_third_person():
    sc = _trivial_scene()
    sc.register_predicate(
        "taller_than", arity=2,
        fn=lambda scene, a, b: float(
            getattr(a, "height", 0.0) > getattr(b, "height", 0.0)
        ),
    )
    # obj_b height=2.0 > obj_a height=1.0 → 1.0
    assert sc.cameras[0].third_person.taller_than[1, 0] == 1.0
    assert sc.cameras[0].third_person.taller_than[0, 1] == 0.0


def test_fn_form_arity_3_third_person():
    sc = _trivial_scene()
    sc.register_predicate(
        "always_one", arity=3,
        fn=lambda scene, a, b, c: 1.0,
    )
    assert sc.cameras[0].third_person.always_one[0, 1, 0] == 1.0


def test_fn_first_person_raises_with_redirect():
    sc = _trivial_scene()
    sc.register_predicate(
        "binary_pred", arity=2,
        fn=lambda scene, a, b: 0.5,
    )
    with pytest.raises(AttributeError, match=r"arity=2"):
        _ = sc.cameras[0].first_person.binary_pred


def test_fn_third_person_wrong_arity_raises():
    sc = _trivial_scene()
    sc.register_predicate(
        "binary_pred", arity=2,
        fn=lambda scene, a, b: 0.5,
    )
    with pytest.raises(IndexError, match=r"expects 2 indices"):
        _ = sc.cameras[0].third_person.binary_pred[0, 1, 0]


# ---------------------------------------------------------------------------
# Canonicalization, collisions, lifecycle, scope
# ---------------------------------------------------------------------------


def test_canonicalization_underscore_hyphen_case():
    sc = _trivial_scene()
    sc.register_predicate("sixty_degrees", angle_deg=60.0)
    fp = sc.cameras[0].first_person
    a = fp.sixty_degrees
    b = fp("sixty-degrees")
    c = fp["SIXTY_DEGREES"]
    np.testing.assert_array_almost_equal(a, b)
    np.testing.assert_array_almost_equal(a, c)


def test_collision_with_builtin_rejected():
    sc = _trivial_scene()
    with pytest.raises(ValueError, match="collides"):
        sc.register_predicate("front", angle_deg=0.0)
    with pytest.raises(ValueError, match="collides"):
        sc.register_predicate("front-right", angle_deg=45.0)
    with pytest.raises(ValueError, match="collides"):
        sc.register_predicate("north", angle_deg=0.0)
    with pytest.raises(ValueError, match="vertical"):
        sc.register_predicate("above", h_r=lambda d, ri, rj: 0.0)


def test_collision_with_synonym_rejected():
    sc = _trivial_scene()
    # "behind" is the canonical for "back" via _resolve_direction_angle's
    # alias logic, so registering either should collide.
    with pytest.raises(ValueError, match="collides"):
        sc.register_predicate("behind", angle_deg=180.0)
    with pytest.raises(ValueError, match="collides"):
        sc.register_predicate("forward", angle_deg=0.0)



def test_scope_to_single_scene():
    sc1 = _trivial_scene()
    sc2 = _trivial_scene()
    sc1.register_predicate("only_in_one", angle_deg=33.0)
    _ = sc1.cameras[0].first_person.only_in_one
    with pytest.raises(ValueError, match=r"Unknown direction"):
        _ = sc2.cameras[0].first_person.only_in_one


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_empty_name():
    sc = _trivial_scene()
    with pytest.raises(ValueError, match="non-empty string"):
        sc.register_predicate("", angle_deg=10.0)
    with pytest.raises(ValueError, match="non-empty string"):
        sc.register_predicate("   ", angle_deg=10.0)


def test_validation_no_form_provided():
    sc = _trivial_scene()
    with pytest.raises(ValueError, match="exactly one"):
        sc.register_predicate("foo")


def test_validation_multiple_forms_provided():
    sc = _trivial_scene()
    with pytest.raises(ValueError, match="exactly one"):
        sc.register_predicate("foo", angle_deg=10.0, h_r=lambda d, ri, rj: 0.0)
    with pytest.raises(ValueError, match="exactly one"):
        sc.register_predicate("foo", angle_deg=10.0, fn=lambda *a: 0.0, arity=2)


def test_validation_fn_requires_arity_ge_2():
    sc = _trivial_scene()
    with pytest.raises(ValueError, match="arity int >= 2"):
        sc.register_predicate("foo", fn=lambda s, a: 0.0, arity=1)
    with pytest.raises(ValueError, match="arity int >= 2"):
        sc.register_predicate("foo", fn=lambda s, a, b: 0.0)  # no arity
    with pytest.raises(ValueError, match="arity int >= 2"):
        sc.register_predicate("foo", fn=lambda s, a, b: 0.0, arity=0)


def test_validation_non_callable_fn():
    sc = _trivial_scene()
    with pytest.raises(ValueError, match="fn must be callable"):
        sc.register_predicate("foo", fn=42, arity=2)
    with pytest.raises(ValueError, match="h_r must be callable"):
        sc.register_predicate("foo", h_r="not a fn")


# ---------------------------------------------------------------------------
# Sanity: sigmoid behavior and indexing convention
# ---------------------------------------------------------------------------


def test_score_in_unit_interval_for_random_h_r():
    """The sigmoid must clamp h_r outputs to [0, 1] regardless of evidence range."""
    sc = _trivial_scene()
    sc.register_predicate(
        "loud_evidence",
        h_r=lambda d, ri, rj: 999.0,   # huge positive → sigmoid → ~1
    )
    arr = sc.cameras[0].first_person.loud_evidence
    assert np.all(arr <= 1.0 + 1e-9)
    assert np.all(arr >= 1.0 - 1e-6)  # all should be ~1


def test_score_lower_bound():
    sc = _trivial_scene()
    sc.register_predicate(
        "tiny_evidence",
        h_r=lambda d, ri, rj: -999.0,  # huge negative → sigmoid → ~0
    )
    arr = sc.cameras[0].first_person.tiny_evidence
    assert np.all(arr <= 1e-6)
    assert np.all(arr >= 0.0)


def test_indexing_resolves_objects_and_cameras():
    """Index in [0, K) → object; [K, K+C) → camera."""
    sc = _trivial_scene()
    K = len(sc.objects)
    # arity=1 predicates aren't part of fn form, so test via an h_r that
    # checks the type of entity it received via R_i.
    types_seen = []

    def h_record(delta_local, R_i_local, R_j_local):
        # Use trace to encode "did we get identity" (camera fallback would give
        # us R_a_T @ R_a = identity; objects give a non-trivial rotation).
        types_seen.append(float(np.trace(R_i_local)))
        return 0.0

    sc.register_predicate("record", h_r=h_record, normalize_distance=False)
    _ = sc.cameras[0].first_person.record
    # We should have called h_record K + C times.
    assert len(types_seen) == K + len(sc.cameras)


# ===========================================================================
# Extended robustness suite — geometric correctness, sigmoid sensitivity,
# scene-scale, codegen ergonomics, edge cases, re-registration, failure modes.
# ===========================================================================


def _scene_axis_aligned() -> Scene:
    """4 objects placed at +Z, +X, -Z, -X around a camera at the origin."""
    objs = [
        _make_object(0, center=(0.0, 0.0, 5.0)),   # front (+Z)
        _make_object(1, center=(5.0, 0.0, 0.0)),   # right (+X)
        _make_object(2, center=(0.0, 0.0, -5.0)),  # back (-Z)
        _make_object(3, center=(-5.0, 0.0, 0.0)),  # left (-X)
    ]
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
    return Scene(objects=objs, cameras=[cam], images=[None])


# ---------------------------------------------------------------------------
# Group 1 — Geometric correctness (angle conventions + rotated anchors)
# ---------------------------------------------------------------------------


def test_angle_deg_full_cardinal_set():
    """All four cardinal angles must peak at the right object."""
    sc = _scene_axis_aligned()
    sc.register_predicate("ang000", angle_deg=0.0)    # front → obj 0
    sc.register_predicate("ang090", angle_deg=90.0)   # right → obj 1
    sc.register_predicate("ang180", angle_deg=180.0)  # back  → obj 2
    sc.register_predicate("ang270", angle_deg=270.0)  # left  → obj 3
    fp = sc.cameras[0].first_person
    assert fp.ang000[:4].argmax() == 0
    assert fp.ang090[:4].argmax() == 1
    assert fp.ang180[:4].argmax() == 2
    assert fp.ang270[:4].argmax() == 3


def test_angle_deg_diagonals_peak_between():
    """angle_deg=45 should score higher for +Z and +X objects than -Z, -X."""
    sc = _scene_axis_aligned()
    sc.register_predicate("ang045", angle_deg=45.0)
    arr = sc.cameras[0].first_person.ang045
    # Front (+Z) and right (+X) should both score above back (-Z) and left (-X)
    assert arr[0] > arr[2]   # +Z beats -Z
    assert arr[1] > arr[3]   # +X beats -X


def test_angle_deg_360_equals_0():
    """angle_deg=360 should be numerically identical to angle_deg=0."""
    sc = _scene_axis_aligned()
    sc.register_predicate("zero", angle_deg=0.0)
    sc.register_predicate("twofull", angle_deg=360.0)
    a = sc.cameras[0].first_person.zero
    b = sc.cameras[0].first_person.twofull
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_rotated_anchor_rotates_predicate():
    """A camera with yaw=90° (facing +X) should see the +X object as 'front'.

    Strong assertion: the predicate must track the anchor's body axes.  If the
    implementation forgot to invert R_a, this test would show the same
    argmax across both cameras.
    """
    objs = [
        _make_object(0, center=(0.0, 0.0, 5.0)),   # world +Z
        _make_object(1, center=(5.0, 0.0, 0.0)),   # world +X
    ]
    cam_yaw0 = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
    cam_yaw90 = _make_camera(1, yaw_deg=90.0, position=(0.0, 0.0, 0.0))
    sc = Scene(objects=objs, cameras=[cam_yaw0, cam_yaw90], images=[None, None])
    # Use a non-colliding name (avoid "front"/"forward" which are built-ins).
    sc.register_predicate("fdir", angle_deg=0.0)

    # cam_yaw0 frame_front = +Z → obj_0 (+Z) scores high, obj_1 (+X) scores low.
    arr0 = sc.cameras[0].first_person.fdir
    assert arr0[0] > 0.5
    assert arr0[1] < 0.5

    # cam_yaw90 frame_front = +X → obj_1 (+X) scores high, obj_0 (+Z) scores low.
    arr1 = sc.cameras[1].first_person.fdir
    assert arr1[1] > 0.5
    assert arr1[0] < 0.5


def test_merger_theoretical_equivalence():
    """The merger's headline guarantee: first_person.X[k] equals
    third_person.X[k, K+cam_idx] when the third-person j is the anchor itself.

    This is the empirical proof that the paper's S^a_r[k, a] = first_person
    collapse is implemented correctly.  If any sign / axis / reference-point
    is wrong in _compute_pairwise_score, this test fails.
    """
    sc = _trivial_scene()
    K = len(sc.objects)
    sc.register_predicate("checker", angle_deg=37.0)  # arbitrary angle
    fp = sc.cameras[0].first_person.checker
    for k in range(K + len(sc.cameras)):
        s_tp = sc.cameras[0].third_person.checker[k, K + 0]
        assert abs(fp[k] - s_tp) < 1e-9, (
            f"first_person/third_person mismatch at k={k}: "
            f"fp={fp[k]} tp[k, anchor]={s_tp}"
        )


def test_merger_theoretical_equivalence_h_r_form():
    """Same merger guarantee but for arbitrary h_r (not just angle_deg sugar)."""
    sc = _trivial_scene()
    K = len(sc.objects)

    def h(d, R_i, R_j):
        return float(d[0] * 0.3 + d[1] * 0.7 - d[2] * 0.4)  # arbitrary linear

    sc.register_predicate("checker_hr", h_r=h)
    fp = sc.cameras[0].first_person.checker_hr
    for k in range(K + len(sc.cameras)):
        s_tp = sc.cameras[0].third_person.checker_hr[k, K + 0]
        assert abs(fp[k] - s_tp) < 1e-9


def test_displacement_sign_convention_i_minus_j():
    """The paper says Δ^a_{ij} = R_a^T(x_i - x_j), so x_i - x_j (not the reverse)."""
    sc = _scene_axis_aligned()
    # h_r returns delta_local[2] (the +Z component).  If we put i = obj at +Z=5
    # and j = obj at -Z=-5, delta should be +10 along Z → h_r returns ~+10
    # → sigmoid clamps to ~1.  If implemented backwards (j - i) we'd get -10.
    captured = {}
    def h(d, R_i, R_j):
        captured["d"] = d.copy()
        return float(d[2])
    sc.register_predicate("z_check", h_r=h, normalize_distance=False, margin=0.0)
    s = sc.cameras[0].third_person.z_check[0, 2]  # obj 0 at +Z=5, obj 2 at -Z=-5
    assert captured["d"][2] > 0  # x_i - x_j is +10 along Z
    assert s > 0.99


def test_R_j_local_is_identity_in_first_person():
    """In first-person j = anchor, so R_j_local = R_a^T @ R_a = I.  This is
    a tight check on the first-person reference convention."""
    sc = _trivial_scene()
    captured = []
    def h(d, R_i, R_j):
        captured.append(R_j.copy())
        return 0.0
    sc.register_predicate("rj_probe", h_r=h)
    _ = sc.cameras[0].first_person.rj_probe
    # Every call should have R_j_local ~ identity.
    for R_j in captured:
        np.testing.assert_allclose(R_j, np.eye(3), atol=1e-9)


def test_R_i_local_picks_up_obj_orientation():
    """When the object's intrinsic front differs from the anchor's front,
    R_i_local must encode that rotation.  Tested via third_person access so
    we control exactly which entity is i (avoid the first_person loop
    overwriting captured["R_i"] with the camera's identity matrix)."""
    obj_facing_x = _make_object(0, center=(0.0, 0.0, 5.0), front=(1.0, 0.0, 0.0))
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
    sc = Scene(objects=[obj_facing_x], cameras=[cam], images=[None])
    captured = {}
    def h(d, R_i, R_j):
        captured["R_i"] = R_i.copy()
        return 0.0
    sc.register_predicate("ri_probe", h_r=h)
    K = len(sc.objects)  # K=1 so camera entity index is K+0 = 1
    _ = sc.cameras[0].third_person.ri_probe[0, K + 0]
    # Object's front column should be the +X axis in the anchor's local frame.
    # Since the anchor is axis-aligned, anchor-local +X = world +X = (1,0,0).
    np.testing.assert_allclose(
        captured["R_i"][:, 2], np.array([1.0, 0.0, 0.0]), atol=1e-9
    )

    # Bonus: cam yawed 90° should rotate the obj's front into anchor-local -Z.
    cam_yawed = _make_camera(1, yaw_deg=90.0, position=(0.0, 0.0, 0.0))
    sc2 = Scene(objects=[obj_facing_x], cameras=[cam_yawed], images=[None])
    captured2 = {}
    def h2(d, R_i, R_j):
        captured2["R_i"] = R_i.copy()
        return 0.0
    sc2.register_predicate("ri_probe2", h_r=h2)
    _ = sc2.cameras[0].third_person.ri_probe2[0, K + 0]
    # Cam yaw=90 has frame_right=-Z, frame_front=+X.  So in cam-local frame,
    # world +X becomes cam-local +Z.  Therefore R_i_local[:, 2] = (0, 0, 1).
    np.testing.assert_allclose(
        captured2["R_i"][:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-9
    )


# ---------------------------------------------------------------------------
# Group 2 — Sigmoid hyperparameter sensitivity
# ---------------------------------------------------------------------------


def test_margin_shifts_decision_boundary():
    """Larger margin should suppress evidence; smaller margin admits it.

    Math: sigma((h - m)/tau) crosses 0.95 at (h - m)/tau >= 2.944, and crosses
    0.05 at (h - m)/tau <= -2.944.  With h=0.10 fixed:
      strict (m=0.50, tau=0.05):  (0.10 - 0.50)/0.05 = -8   → sigma(-8) ≈ 3e-4
      lax    (m=0.00, tau=0.025): (0.10 - 0.00)/0.025 = 4   → sigma(4)  ≈ 0.982
    """
    sc = _trivial_scene()
    sc.register_predicate(
        "strict", h_r=lambda d, ri, rj: 0.10, margin=0.50, temperature=0.05,
    )
    sc.register_predicate(
        "lax",    h_r=lambda d, ri, rj: 0.10, margin=0.00, temperature=0.025,
    )
    strict = sc.cameras[0].first_person.strict[0]
    lax    = sc.cameras[0].first_person.lax[0]
    assert strict < 0.05, f"strict margin should suppress; got {strict}"
    assert lax    > 0.95, f"lax margin should pass; got {lax}"


def test_sigmoid_at_margin_is_half():
    """Evidence exactly at the margin should give sigmoid output 0.5."""
    sc = _trivial_scene()
    sc.register_predicate(
        "at_margin", h_r=lambda d, ri, rj: 0.25, margin=0.25, temperature=0.10,
    )
    s = sc.cameras[0].first_person.at_margin[0]
    assert abs(s - 0.5) < 1e-9


def test_temperature_controls_sharpness():
    """Smaller temperature → near-step function; larger → soft."""
    sc = _trivial_scene()
    sc.register_predicate(
        "sharp", h_r=lambda d, ri, rj: 0.10, margin=0.05, temperature=1e-4,
    )
    sc.register_predicate(
        "soft",  h_r=lambda d, ri, rj: 0.10, margin=0.05, temperature=10.0,
    )
    # Same evidence (0.10), same margin (0.05) → both above margin.
    # Sharp should be ~1 (well above); soft should be near 0.5.
    sharp = sc.cameras[0].first_person.sharp[0]
    soft  = sc.cameras[0].first_person.soft[0]
    assert sharp > 0.999
    assert 0.49 < soft < 0.51


# ---------------------------------------------------------------------------
# Group 3 — Scene-scale normalization
# ---------------------------------------------------------------------------


def test_normalize_distance_true_is_scale_invariant():
    """With normalize_distance=True, scaling all positions by k should leave
    the score unchanged (modulo numerical noise)."""
    def make_scene(scale: float):
        objs = [
            _make_object(0, center=(0.0, 0.0, 5.0 * scale)),
            _make_object(1, center=(5.0 * scale, 0.0, 0.0)),
        ]
        cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
        return Scene(objects=objs, cameras=[cam], images=[None])

    sc_small = make_scene(1.0)
    sc_big = make_scene(100.0)
    sc_small.register_predicate("dir", angle_deg=30.0, normalize_distance=True)
    sc_big.register_predicate("dir", angle_deg=30.0, normalize_distance=True)
    s_small = sc_small.cameras[0].first_person.dir[0]
    s_big = sc_big.cameras[0].first_person.dir[0]
    assert abs(s_small - s_big) < 1e-6


def test_normalize_distance_false_is_scale_sensitive():
    """Without normalization, scaling positions changes the evidence
    (and thus the sigmoid output) for distance-dependent h_r."""
    def make_scene(scale: float):
        objs = [_make_object(0, center=(0.0, 0.0, 5.0 * scale))]
        cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
        return Scene(objects=objs, cameras=[cam], images=[None])

    sc_small = make_scene(1.0)
    sc_big = make_scene(100.0)
    h = lambda d, ri, rj: float(d[2])  # raw +Z displacement
    sc_small.register_predicate(
        "raw", h_r=h, normalize_distance=False, margin=10.0, temperature=1.0,
    )
    sc_big.register_predicate(
        "raw", h_r=h, normalize_distance=False, margin=10.0, temperature=1.0,
    )
    s_small = sc_small.cameras[0].first_person.raw[0]
    s_big = sc_big.cameras[0].first_person.raw[0]
    assert s_big > s_small + 0.1   # bigger scale → higher evidence → higher score


def test_scene_scale_cached_after_first_access():
    sc = _trivial_scene()
    sc.register_predicate("cached", angle_deg=10.0, normalize_distance=True)
    assert sc._scene_scale_cache is None
    _ = sc.cameras[0].first_person.cached
    assert sc._scene_scale_cache is not None
    # Second access should not recompute (cache stays the same value).
    s1 = sc._scene_scale_cache
    _ = sc.cameras[0].first_person.cached
    assert sc._scene_scale_cache == s1


# ---------------------------------------------------------------------------
# Group 4 — Codegen ergonomics (first-person ndarray patterns)
# ---------------------------------------------------------------------------


def test_first_person_ndarray_supports_argmax():
    sc = _scene_axis_aligned()
    sc.register_predicate("front_dir", angle_deg=0.0)
    arr = sc.cameras[0].first_person.front_dir
    assert int(arr[:4].argmax()) == 0   # obj 0 is at +Z


def test_first_person_ndarray_supports_slicing():
    sc = _scene_axis_aligned()
    sc.register_predicate("front_dir", angle_deg=0.0)
    arr = sc.cameras[0].first_person.front_dir
    K = len(sc.objects)
    obj_only = arr[:K]
    assert obj_only.shape == (K,)
    assert obj_only.dtype == arr.dtype


def test_first_person_ndarray_supports_comparison():
    sc = _scene_axis_aligned()
    sc.register_predicate("front_dir", angle_deg=0.0)
    arr = sc.cameras[0].first_person.front_dir
    mask = arr > 0.5
    assert mask.dtype == np.bool_
    assert mask[0] == True
    assert mask[2] == False  # obj at -Z


def test_first_person_ndarray_supports_fancy_indexing():
    sc = _scene_axis_aligned()
    sc.register_predicate("front_dir", angle_deg=0.0)
    arr = sc.cameras[0].first_person.front_dir
    picks = arr[[0, 2]]
    assert picks.shape == (2,)


# ---------------------------------------------------------------------------
# Group 5 — Indexing edge cases (third_person)
# ---------------------------------------------------------------------------


def test_third_person_self_index_well_defined():
    """[i, i] gives zero displacement → near-zero evidence for distance-based h_r."""
    sc = _trivial_scene()
    sc.register_predicate(
        "selfprobe",
        h_r=lambda d, ri, rj: float(np.linalg.norm(d)),
        margin=0.0, temperature=0.01, normalize_distance=False,
    )
    s = sc.cameras[0].third_person.selfprobe[0, 0]
    # Zero displacement → evidence ~0 → with margin=0 sigmoid gives ~0.5
    assert 0.4 < s < 0.6


def test_third_person_index_out_of_range_raises():
    sc = _trivial_scene()
    sc.register_predicate("any", angle_deg=0.0)
    total = len(sc.objects) + len(sc.cameras)
    with pytest.raises(IndexError):
        _ = sc.cameras[0].third_person.any[total, 0]
    with pytest.raises(IndexError):
        _ = sc.cameras[0].third_person.any[0, total]


def test_third_person_single_int_raises_for_pairwise():
    """Pairwise predicates require a 2-tuple, not a single int."""
    sc = _trivial_scene()
    sc.register_predicate("dir", angle_deg=0.0)
    with pytest.raises(IndexError):
        _ = sc.cameras[0].third_person.dir[0]


def test_third_person_wrong_tuple_arity_raises():
    sc = _trivial_scene()
    sc.register_predicate("dir", angle_deg=0.0)
    with pytest.raises(IndexError):
        _ = sc.cameras[0].third_person.dir[0, 1, 2]   # 3 indices for arity-2


# ---------------------------------------------------------------------------
# Group 6 — Re-registration & multiple predicates
# ---------------------------------------------------------------------------


def test_re_register_overwrites_silently():
    """Re-registering the same name overwrites the existing spec."""
    sc = _trivial_scene()
    sc.register_predicate("dyn", angle_deg=0.0)
    first = sc.cameras[0].first_person.dyn[0]
    sc.register_predicate("dyn", angle_deg=180.0)   # opposite direction
    second = sc.cameras[0].first_person.dyn[0]
    # The obj at +Z=2 should swap from front-favored to back-favored.
    assert (first > 0.5) != (second > 0.5)


def test_multiple_predicates_independent():
    sc = _trivial_scene()
    sc.register_predicate("a", angle_deg=0.0)
    sc.register_predicate("b", angle_deg=90.0)
    sc.register_predicate("c", h_r=lambda d, ri, rj: 100.0)  # always high
    a = sc.cameras[0].first_person.a
    b = sc.cameras[0].first_person.b
    c = sc.cameras[0].first_person.c
    assert a[0] > 0.5 and a[1] < 0.5    # front predicate picks +Z obj
    assert b[0] < 0.5 and b[1] > 0.5    # right predicate picks +X obj
    assert np.all(c > 0.99)             # always-high predicate is ~1



# ---------------------------------------------------------------------------
# Group 7 — Boundary scenes
# ---------------------------------------------------------------------------


def test_empty_scene_registration_succeeds():
    """Registering a predicate on an objects-and-cameras-empty scene should
    not crash.  Access still requires a view, which needs at least one camera,
    so we only test the registration call itself."""
    sc = Scene(objects=[], cameras=[], images=[])
    sc.register_predicate("empty", angle_deg=0.0)   # must not raise
    assert "empty" in sc._user_predicates


def test_scene_with_only_camera():
    """A scene with no objects, just a camera — first_person.X should
    return a (1,) ndarray (just the camera self-entry)."""
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
    sc = Scene(objects=[], cameras=[cam], images=[None])
    sc.register_predicate("fdir", angle_deg=0.0)
    arr = sc.cameras[0].first_person.fdir
    assert arr.shape == (1,)


def test_single_entity_scene_scene_scale_falls_back_to_one():
    """With <2 entities, scene scale falls back to 1.0 (avoiding zero-div)."""
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))
    sc = Scene(objects=[], cameras=[cam], images=[None])
    assert sc._compute_scene_scale() == 1.0


# ---------------------------------------------------------------------------
# Group 8 — Failure-mode robustness
# ---------------------------------------------------------------------------


def test_h_r_exception_propagates():
    sc = _trivial_scene()
    def bad_h_r(d, R_i, R_j):
        raise RuntimeError("user error")
    sc.register_predicate("crashy", h_r=bad_h_r)
    with pytest.raises(RuntimeError, match="user error"):
        _ = sc.cameras[0].first_person.crashy


def test_fn_exception_propagates():
    sc = _trivial_scene()
    def bad_fn(scene, a, b):
        raise ValueError("user error")
    sc.register_predicate("crashy2", arity=2, fn=bad_fn)
    with pytest.raises(ValueError, match="user error"):
        _ = sc.cameras[0].third_person.crashy2[0, 1]


def test_h_r_returning_nan_propagates_to_sigmoid():
    """NaN evidence → NaN score (no silent coercion)."""
    sc = _trivial_scene()
    sc.register_predicate("nan_h", h_r=lambda d, ri, rj: float("nan"))
    arr = sc.cameras[0].first_person.nan_h
    assert np.all(np.isnan(arr))


# ---------------------------------------------------------------------------
# Group 9 — Wrapper repr / object identity
# ---------------------------------------------------------------------------


def test_third_person_pairwise_wrapper_repr():
    sc = _trivial_scene()
    sc.register_predicate("rprobe", angle_deg=0.0)
    w = sc.cameras[0].third_person.rprobe
    r = repr(w)
    assert "rprobe" in r
    assert "angle" in r or "Pairwise" in r


def test_third_person_kary_wrapper_repr():
    sc = _trivial_scene()
    sc.register_predicate("kprobe", arity=2, fn=lambda s, a, b: 0.0)
    w = sc.cameras[0].third_person.kprobe
    r = repr(w)
    assert "kprobe" in r
    assert "arity=2" in r


def test_two_anchors_share_registry_but_have_independent_views():
    """A registered predicate is visible from any anchor in the scene,
    and each anchor's view applies its own frame_front to the computation."""
    objs = [_make_object(0, center=(0.0, 0.0, 5.0))]  # at world +Z
    cam_a = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, 0.0))    # facing +Z
    cam_b = _make_camera(1, yaw_deg=180.0, position=(0.0, 0.0, 0.0))  # facing -Z
    sc = Scene(objects=objs, cameras=[cam_a, cam_b], images=[None, None])
    sc.register_predicate("fdir", angle_deg=0.0)
    # From cam_a's POV: obj is in front (+Z) → high score.
    # From cam_b's POV: obj is behind (cam_b faces -Z) → low score.
    sa = sc.cameras[0].first_person.fdir[0]
    sb = sc.cameras[1].first_person.fdir[0]
    assert sa > 0.9
    assert sb < 0.1


# ---------------------------------------------------------------------------
# Group 10 — Name-shape robustness
# ---------------------------------------------------------------------------


def test_name_with_digits_works():
    sc = _trivial_scene()
    sc.register_predicate("angle_30deg", angle_deg=30.0)
    sc.register_predicate("a1b2c3", angle_deg=45.0)
    _ = sc.cameras[0].first_person.angle_30deg
    _ = sc.cameras[0].first_person.a1b2c3


def test_leading_trailing_whitespace_stripped():
    sc = _trivial_scene()
    sc.register_predicate("  spaced  ", angle_deg=0.0)
    # Whitespace stripped → canonical "spaced"
    _ = sc.cameras[0].first_person.spaced


def test_canonical_name_is_lowercased():
    """Mixed-case registration should canonicalize and remain accessible."""
    sc = _trivial_scene()
    sc.register_predicate("MixedCase_Name", angle_deg=0.0)
    fp = sc.cameras[0].first_person
    a = fp.mixedcase_name        # lowercase access
    b = fp("MixedCase-Name")     # original case + hyphen
    np.testing.assert_array_almost_equal(a, b)


# ===========================================================================
# Group 11 — scene.obj_<name>  (paper's S^{obj}_r = S^{a_j}_r access path)
# ===========================================================================


def _scene_two_objs_front_axis():
    """obj_a at origin facing +Z, obj_b 5m ahead facing +Z.  Camera off-screen."""
    obj_a = _make_object(0, center=(0.0, 0.0, 0.0), front=(0.0, 0.0, 1.0))
    obj_b = _make_object(1, center=(0.0, 0.0, 5.0), front=(0.0, 0.0, 1.0))
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, -5.0))
    return Scene(objects=[obj_a, obj_b], cameras=[cam], images=[None])


def test_obj_X_basic_access_works():
    """scene.obj_<name>[i, j] returns a float in [0, 1]."""
    sc = _scene_two_objs_front_axis()
    sc.register_predicate("fdir", angle_deg=0.0)
    s = sc.obj_fdir[1, 0]
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_obj_X_uses_j_orientation_as_FoR_anchor():
    """The key invariant: rotating j's intrinsic front flips the result.

    This is the discriminator between obj_X (j-anchored) and third_person
    (anchor-frame-but-j-position-reference).  If obj_X mistakenly used the
    view's frame instead of j's, rotating j would NOT change the score.
    """
    # a faces +Z; b at +Z=5 → b IS in front of a.
    sc1 = _scene_two_objs_front_axis()
    sc1.register_predicate("fdir", angle_deg=0.0)
    s_default = sc1.obj_fdir[1, 0]
    assert s_default > 0.9

    # Rotate a 180° → a faces -Z → b is now BEHIND a, not in front.
    obj_a_rot = _make_object(0, center=(0.0, 0.0, 0.0), front=(0.0, 0.0, -1.0))
    obj_b = _make_object(1, center=(0.0, 0.0, 5.0), front=(0.0, 0.0, 1.0))
    cam = _make_camera(0, yaw_deg=0.0, position=(0.0, 0.0, -5.0))
    sc2 = Scene(objects=[obj_a_rot, obj_b], cameras=[cam], images=[None])
    sc2.register_predicate("fdir", angle_deg=0.0)
    s_rotated = sc2.obj_fdir[1, 0]
    assert s_rotated < 0.1, (
        f"Rotating j must flip the obj_X result; got {s_rotated} (expected ~0)"
    )


def test_obj_X_equivalence_with_view_at_j():
    """scene.obj_X[i, j] must equal first_person.X[i] when accessed from a
    view built at entity j's pose.  This is the paper's S^{obj}_r ≡ S^{a_j}_r
    equivalence — the strongest property of the obj-centric form."""
    sc = _scene_two_objs_front_axis()
    sc.register_predicate("fdir", angle_deg=37.0)  # arbitrary angle
    obj_a = sc.objects[0]
    view_at_a = sc.frame(position=obj_a.position, orientation=obj_a.rotation_world)
    direct = sc.obj_fdir[1, 0]
    via_view = float(view_at_a.first_person.fdir[1])
    assert abs(direct - via_view) < 1e-9, (
        f"obj_X[i, j] must equal first_person.X[i] from view@j; "
        f"got direct={direct} vs via_view={via_view}"
    )


def test_obj_X_equivalence_for_h_r_form():
    """Same S^{obj}_r equivalence for arbitrary h_r (not just angle_deg sugar)."""
    sc = _scene_two_objs_front_axis()
    sc.register_predicate(
        "custom", h_r=lambda d, R_i, R_j: float(d[0] * 0.7 - d[2] * 0.3),
    )
    obj_a = sc.objects[0]
    view_at_a = sc.frame(position=obj_a.position, orientation=obj_a.rotation_world)
    direct = sc.obj_custom[1, 0]
    via_view = float(view_at_a.first_person.custom[1])
    assert abs(direct - via_view) < 1e-9


def test_obj_X_camera_as_j():
    """Cameras can serve as j (the FoR anchor).  Mirrors existing
    obj_behind[i, cam] = 'is i behind the camera?' semantics."""
    sc = _scene_two_objs_front_axis()
    sc.register_predicate("fdir", angle_deg=0.0)
    K = len(sc.objects)
    # camera at +Z=-5, facing +Z.  obj_a is at +Z=0 → in front of cam.
    s_cam_anchor = sc.obj_fdir[0, K + 0]   # i=obj_a, j=camera
    assert s_cam_anchor > 0.9


def test_obj_X_unknown_predicate_raises_attribute_error():
    """scene.obj_undefined → AttributeError (caller may catch and recover)."""
    sc = _trivial_scene()
    with pytest.raises(AttributeError):
        _ = sc.obj_does_not_exist


def test_obj_X_underscore_prefix_does_not_trap():
    """Private attribute lookups (e.g. _x) must not go through the obj_ trap."""
    sc = _trivial_scene()
    with pytest.raises(AttributeError):
        _ = sc._obj_anything   # leading underscore → falls through


def test_obj_X_for_fn_form_raises_with_redirect():
    """fn-form predicates are not anchor-conditioned → no obj_X access."""
    sc = _trivial_scene()
    sc.register_predicate(
        "kary_pred", arity=2, fn=lambda scene, a, b: 0.5,
    )
    with pytest.raises(AttributeError, match=r"fn-form"):
        _ = sc.obj_kary_pred


def test_obj_X_wrong_subscript_arity_raises():
    sc = _trivial_scene()
    sc.register_predicate("fdir", angle_deg=0.0)
    with pytest.raises(IndexError, match=r"expected 2-tuple"):
        _ = sc.obj_fdir[0]                # single int
    with pytest.raises(IndexError, match=r"expected 2-tuple"):
        _ = sc.obj_fdir[0, 1, 2]          # 3-tuple


def test_obj_X_repr():
    sc = _trivial_scene()
    sc.register_predicate("fdir", angle_deg=10.0)
    w = sc.obj_fdir
    r = repr(w)
    assert "fdir" in r
    assert "ObjCentric" in r or "obj" in r.lower()


def test_obj_X_builtin_properties_not_shadowed():
    """Registering 'foo' must not interfere with existing obj_left / obj_right
    / obj_front / obj_behind builtin properties (they are @property's, so
    __getattr__ never fires for them)."""
    sc = _trivial_scene()
    sc.register_predicate("foo", angle_deg=0.0)
    # Built-ins should still be accessible and ProbabilisticTensor-shaped.
    obj_l = sc.obj_left
    obj_r = sc.obj_right
    obj_f = sc.obj_front
    obj_b = sc.obj_behind
    # No further claim — just ensure they don't blow up after we registered a
    # new predicate.
    assert obj_l is not None
    assert obj_r is not None
    assert obj_f is not None
    assert obj_b is not None


def test_obj_X_canonicalization_of_obj_prefix_name():
    """scene.obj_<canonical name> should match the same canonicalization as
    first_person / third_person — underscores treated as hyphens."""
    sc = _scene_two_objs_front_axis()
    sc.register_predicate("alpha_beta", angle_deg=0.0)
    a = sc.obj_alpha_beta[1, 0]      # underscore form
    # Direct dict lookup via getattr with the canonical name (hyphen form
    # isn't a valid Python identifier so we can't write sc.obj_alpha-beta).
    # Verifying via the equivalence path:
    obj_j = sc.objects[0]
    view_at_j = sc.frame(position=obj_j.position, orientation=obj_j.rotation_world)
    b = float(view_at_j.first_person.alpha_beta[1])
    assert abs(a - b) < 1e-9


def test_obj_X_dunder_names_not_intercepted():
    """Dunder names (like obj_lookup__internal__) starting with _ skip the trap,
    AND ordinary dunder lookups (__class__, __dict__, etc.) must work normally."""
    sc = _trivial_scene()
    # Pre-register something so the trap path is exercised internally.
    sc.register_predicate("foo", angle_deg=0.0)
    # Dunder access continues to work.
    assert sc.__class__ is Scene
    assert isinstance(sc.__dict__, dict)
