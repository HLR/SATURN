"""Gauge and co-transform behaviour of stated camera-pose constraints.

A ``same_position`` constraint says several cameras share one world position.
Any position in the set is a legal gauge for that shared point, but the choice
is not free: ``pose_solver`` documents that camera 0 stays at the origin with
the canonical rotation, and the world-up derivation, the cam0 frame,
first-person frames and the projection fallback all read it. The solver
therefore co-locates the set on its anchor camera rather than on the mean;
moving camera 0 would flip front/back and above/below for the whole scene.
"""

import numpy as np
import pytest

from saturn.scene.pose_solver import solve_camera_poses


def _ext(R_c2w: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """World-to-camera extrinsics from a camera-to-world rotation and centre."""
    E = np.eye(4)
    E[:3, :3] = R_c2w.T
    E[:3, 3] = -R_c2w.T @ centre
    return E


def _pos(ext: np.ndarray) -> np.ndarray:
    return -ext[:3, :3].T @ ext[:3, 3]


def _yaw(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@pytest.fixture()
def rotate_in_place():
    """Three cameras that truly share a spot, with spurious baselines."""
    return [
        _ext(np.eye(3), np.zeros(3)),
        _ext(_yaw(np.pi / 2), np.array([0.05, 0.0, 0.02])),
        _ext(_yaw(np.pi), np.array([-0.03, 0.0, 0.04])),
    ]


SP = [{"type": "same_position", "cams": [0, 1, 2]}]


def test_same_position_preserves_anchor_camera(rotate_in_place):
    """Co-locating cameras keeps camera 0 at the origin."""
    out = solve_camera_poses(rotate_in_place, SP, method="average", strength=1.0)
    assert np.allclose(_pos(out[0]), np.zeros(3), atol=1e-12)


def test_same_position_actually_co_locates(rotate_in_place):
    out = solve_camera_poses(rotate_in_place, SP, method="average", strength=1.0)
    p0 = _pos(out[0])
    for e in out[1:]:
        assert np.allclose(_pos(e), p0, atol=1e-12)


def test_same_position_preserves_orientations(rotate_in_place):
    """The constraint speaks about position only; yaw must survive untouched."""
    out = solve_camera_poses(rotate_in_place, SP, method="average", strength=1.0)
    for old, new in zip(rotate_in_place, out):
        assert np.allclose(new[:3, :3], old[:3, :3], atol=1e-12)


def test_same_position_is_a_pure_gauge_on_relative_geometry(rotate_in_place):
    """Object-camera bearings must be preserved for each camera's own points.

    A point reconstructed by camera k keeps its camera-frame coordinates when
    the camera moves and the point moves with it, so what the predicates read
    (direction and distance from that camera) is unchanged.
    """
    out = solve_camera_poses(rotate_in_place, SP, method="average", strength=1.0)
    rng = np.random.default_rng(0)
    for k, (old, new) in enumerate(zip(rotate_in_place, out)):
        p_world_old = rng.normal(size=3) * 2.0
        x_cam = old[:3, :3] @ p_world_old + old[:3, 3]          # camera frame
        p_world_new = new[:3, :3].T @ (x_cam - new[:3, 3])      # re-projected
        # bearing and range from that camera are invariant
        assert np.allclose(
            np.linalg.norm(p_world_old - _pos(old)),
            np.linalg.norm(p_world_new - _pos(new)),
            atol=1e-9,
        )
        d_old = (p_world_old - _pos(old)) / np.linalg.norm(p_world_old - _pos(old))
        d_new = (p_world_new - _pos(new)) / np.linalg.norm(p_world_new - _pos(new))
        assert np.allclose(old[:3, :3] @ d_old, new[:3, :3] @ d_new, atol=1e-9)


def test_anchor_is_the_lowest_index_in_the_set():
    """A set that excludes camera 0 anchors on its own lowest member."""
    exts = [
        _ext(np.eye(3), np.zeros(3)),
        _ext(_yaw(0.3), np.array([1.0, 0.0, 0.0])),
        _ext(_yaw(0.6), np.array([1.4, 0.0, 0.3])),
    ]
    out = solve_camera_poses(
        exts, [{"type": "same_position", "cams": [1, 2]}],
        method="average", strength=1.0,
    )
    assert np.allclose(_pos(out[0]), np.zeros(3), atol=1e-12)   # untouched
    assert np.allclose(_pos(out[1]), _pos(exts[1]), atol=1e-12)  # anchor holds
    assert np.allclose(_pos(out[2]), _pos(exts[1]), atol=1e-12)  # follower moves


def test_per_camera_world_delta_is_inv_new_compose_old():
    """The co-transform must be inv(ext_new) @ ext_old, not the reverse order."""
    from saturn.scene.constraints import _ConstraintNamespace as CN

    rng = np.random.default_rng(7)
    for _ in range(5):
        a = _ext(_yaw(rng.uniform(-np.pi, np.pi)), rng.normal(size=3))
        b = _ext(_yaw(rng.uniform(-np.pi, np.pi)), rng.normal(size=3))
        T = CN._per_camera_world_delta(a, b)
        assert np.allclose(T, np.linalg.inv(b) @ a, atol=1e-9)
        # a point keeps its camera-frame coordinates under T
        p = np.append(rng.normal(size=3), 1.0)
        x_cam = a @ p
        assert np.allclose(b @ (T @ p), x_cam, atol=1e-9)


def test_single_camera_set_is_a_noop(rotate_in_place):
    out = solve_camera_poses(
        rotate_in_place, [{"type": "same_position", "cams": [1]}],
        method="average", strength=1.0,
    )
    for old, new in zip(rotate_in_place, out):
        assert np.allclose(new, old, atol=1e-12)


# ------------------------------------------- rotation constraints are in-place
ROT = [
    {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90.0, "axis": "up"},
    {"type": "rotation", "from_cam": 0, "to_cam": 2, "yaw": 180.0, "axis": "up"},
]


def test_rotation_constraint_preserves_camera_positions(rotate_in_place):
    """A stated "the camera turned N degrees" is a rotation IN PLACE.

    Extrinsics are world-to-camera, so position = -R_w2c.T @ t_w2c depends on R:
    the solver rewrites t together with R so that the camera turns about its own
    centre, not about the world origin.
    """
    before = [_pos(e) for e in rotate_in_place]
    out = solve_camera_poses(rotate_in_place, ROT, method="average", strength=1.0)
    for b, e in zip(before, out):
        assert np.allclose(_pos(e), b, atol=1e-12)


def test_rotation_constraint_still_applies_the_stated_yaw(rotate_in_place):
    """Preserving position must not weaken the constraint itself.

    Measure the RELATIVE rotation cam0->cam1 (R1 @ R0.T) rather than absolute
    headings: the solver canonicalises the world frame, so absolute angles are
    not a stable reference.
    """
    out = solve_camera_poses(rotate_in_place, ROT, method="average", strength=1.0)
    rel = out[1][:3, :3] @ out[0][:3, :3].T
    yaw_deg = np.degrees(np.arctan2(rel[0, 2], rel[0, 0]))
    assert abs(abs(yaw_deg) - 90.0) < 1e-6
    rel2 = out[2][:3, :3] @ out[0][:3, :3].T
    yaw2 = np.degrees(np.arctan2(rel2[0, 2], rel2[0, 0]))
    assert abs(abs(yaw2) - 180.0) < 1e-6


def test_rotation_plus_same_position_compose(rotate_in_place):
    """Both constraints together: co-located on the anchor, relative yaws kept.

    Sign convention: the solver's propagated yaw is measured about the world-up
    axis in its own handedness, so assert the MAGNITUDE of the relative angle
    rather than a signed value.
    """
    out = solve_camera_poses(rotate_in_place, ROT + SP, method="average", strength=1.0)
    for e in out:
        assert np.allclose(_pos(e), np.zeros(3), atol=1e-12)
    fwd = lambda e: e[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    ang = lambda v: np.degrees(np.arctan2(v[0], v[2]))
    rel = abs((ang(fwd(out[1])) - ang(fwd(out[0]))) % 360)
    assert min(abs(rel - 90.0), abs(rel - 270.0)) < 1e-6


# ===========================================================================
# Pose invariants.
#
# The solver holds a camera pose as (R_c2w, t_w2c). Because
# t_w2c = -R_w2c @ position, changing R without t would move the camera. These
# tests assert the invariants directly: rotations happen in place, only
# same_position moves cameras, and every output is a valid rigid transform.
# ===========================================================================

ROT_ONLY = ROT
BOTH = ROT + SP


def _scene(centres, yaws):
    return [_ext(_yaw(y), np.asarray(c, dtype=float)) for c, y in zip(centres, yaws)]


@pytest.mark.parametrize("method", ["average", "anchor"])
@pytest.mark.parametrize(
    "centres",
    [
        [(0, 0, 0), (0.05, 0, 0.02), (-0.03, 0, 0.04)],      # canonical: cam0 at origin
        [(1, 0, 2), (1.05, 0, 2.02), (0.97, 0, 2.04)],       # cam0 away from origin
    ],
    ids=["cam0-at-origin", "cam0-off-origin"],
)
def test_rotation_constraints_never_move_any_camera(method, centres):
    """A rotation constraint is a rotation IN PLACE — for EVERY camera.

    The off-origin case matters: with cam0 at the origin, position =
    -R_c2w @ t is 0 for any R when t = 0, so only an off-origin anchor shows
    whether its rotation is applied in place.
    """
    scene = _scene(centres, [0.0, np.pi / 2 * 0.9, np.pi * 0.95])
    before = [_pos(e) for e in scene]
    out = solve_camera_poses(scene, ROT_ONLY, method=method, strength=1.0)
    for b, e in zip(before, out):
        assert np.allclose(_pos(e), b, atol=1e-12)


def test_rotation_constraints_preserve_a_non_cam0_anchor():
    """When no constraint mentions cam 0 the anchor is cam 1 — it must not move."""
    scene = _scene([(0, 0, 0), (1.0, 0, 0.5), (1.4, 0, 0.9)], [0.0, 0.3, 1.2])
    before = [_pos(e) for e in scene]
    out = solve_camera_poses(
        scene, [{"type": "rotation", "from_cam": 1, "to_cam": 2, "yaw": 90.0, "axis": "up"}],
        method="average", strength=1.0,
    )
    for b, e in zip(before, out):
        assert np.allclose(_pos(e), b, atol=1e-12)


def test_strength_zero_is_a_true_noop(rotate_in_place):
    """strength=0 is documented as "ignore constraint, return original".

    Neither the anchor's orientation nor any position changes at strength 0.
    """
    out = solve_camera_poses(rotate_in_place, ROT_ONLY, method="average", strength=0.0)
    for old, new in zip(rotate_in_place, out):
        assert np.allclose(new, old, atol=1e-12)


def test_strength_one_fully_applies_the_constraint(rotate_in_place):
    out = solve_camera_poses(rotate_in_place, ROT_ONLY, method="average", strength=1.0)
    rel = out[1][:3, :3] @ out[0][:3, :3].T
    assert abs(abs(np.degrees(np.arctan2(rel[0, 2], rel[0, 0]))) - 90.0) < 1e-6


def test_intermediate_strength_lies_between_the_endpoints(rotate_in_place):
    """SLERP must interpolate, not jump to an endpoint or overshoot."""
    from saturn.scene.pose_solver import angular_distance_deg

    lo = solve_camera_poses(rotate_in_place, ROT_ONLY, method="average", strength=0.0)
    hi = solve_camera_poses(rotate_in_place, ROT_ONLY, method="average", strength=1.0)
    mid = solve_camera_poses(rotate_in_place, ROT_ONLY, method="average", strength=0.5)
    span = angular_distance_deg(lo[1][:3, :3], hi[1][:3, :3])
    d_lo = angular_distance_deg(lo[1][:3, :3], mid[1][:3, :3])
    d_hi = angular_distance_deg(mid[1][:3, :3], hi[1][:3, :3])
    assert d_lo > 1e-6 and d_hi > 1e-6
    assert d_lo + d_hi <= span + 1e-6


def test_constraint_application_is_order_independent(rotate_in_place):
    """rotation-then-same_position must equal same_position-then-rotation.

    Rotations never move cameras, so the position that same_position copies
    does not depend on whether rotations ran first.
    """
    a = solve_camera_poses(rotate_in_place, ROT + SP, method="average", strength=1.0)
    b = solve_camera_poses(rotate_in_place, SP + ROT, method="average", strength=1.0)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-12)


def test_same_position_is_the_only_thing_that_moves_cameras(rotate_in_place):
    """Positions change under SP, and only onto the anchor's position."""
    out = solve_camera_poses(rotate_in_place, BOTH, method="average", strength=1.0)
    anchor_pos = _pos(rotate_in_place[0])
    for e in out:
        assert np.allclose(_pos(e), anchor_pos, atol=1e-12)


@pytest.mark.parametrize("constraints", [ROT, SP, ROT + SP, []], ids=["rot", "sp", "both", "none"])
def test_output_extrinsics_stay_valid_rigid_transforms(rotate_in_place, constraints):
    """R must stay orthogonal with its determinant sign intact.

    The canonical SaPy frame uses R = diag(+1,-1,+1) (det = -1, a reflection).
    A solver that quietly flips that sign mirrors left/right for the whole
    scene, so pin it.
    """
    out = solve_camera_poses(rotate_in_place, constraints, method="average", strength=1.0)
    for old, new in zip(rotate_in_place, out):
        R = new[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.sign(np.linalg.det(R)) == np.sign(np.linalg.det(old[:3, :3]))
        assert np.allclose(new[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_translation_is_always_derived_from_rotation_and_position(rotate_in_place):
    """The (R, position) representation must round-trip through extrinsics.

    Directly pins the coupling: t_w2c is a *derived*
    quantity, so for every output camera, -R_w2c.T @ t_w2c must reproduce the
    position the solver intended.
    """
    out = solve_camera_poses(rotate_in_place, BOTH, method="average", strength=1.0)
    for e in out:
        R_w2c, t_w2c = e[:3, :3], e[:3, 3]
        assert np.allclose(t_w2c, -R_w2c @ (-R_w2c.T @ t_w2c), atol=1e-12)


def test_empty_and_degenerate_inputs(rotate_in_place):
    assert solve_camera_poses([], ROT, method="average") == []
    out = solve_camera_poses(rotate_in_place, [], method="average", strength=1.0)
    for old, new in zip(rotate_in_place, out):
        assert np.allclose(new, old, atol=1e-12)
