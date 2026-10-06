"""Unit tests for saturn.scene.pose_solver.

Pure-numpy tests — no scene, no VLM, no async. Verifies:
  - rotation primitives produce the expected axes
  - SO(3) averaging recovers the centroid for tightly-clustered rotations
  - solve_camera_poses (anchor + average) reproduces the rotation_group015
    scenario's expected refined headings
  - same_position constraint places the cameras on the anchor camera
"""
import numpy as np
import pytest

from saturn.scene.pose_solver import (
    R_x,
    R_y,
    R_z,
    angular_distance_deg,
    weighted_average_so3,
    solve_camera_poses,
    _compose_extrinsics,
    _world_to_cam_translation,
)


# ----- Rotation primitives -----

def test_R_y_takes_north_to_east():
    """R_y(+90) should map +Z (north) to +X (east) — clockwise viewed from above."""
    out = R_y(90) @ np.array([0, 0, 1.0])
    assert np.allclose(out, [1, 0, 0], atol=1e-9)


def test_R_y_180_reverses_front():
    out = R_y(180) @ np.array([0, 0, 1.0])
    assert np.allclose(out, [0, 0, -1], atol=1e-9)


def test_R_x_around_right_axis():
    out = R_x(90) @ np.array([0, 1, 0.0])  # up rotated about right
    assert np.allclose(out, [0, 0, 1], atol=1e-9)


def test_R_z_around_front_axis():
    out = R_z(90) @ np.array([1, 0, 0.0])  # right rotated about front
    assert np.allclose(out, [0, 1, 0], atol=1e-9)


def _camera_position_world(ext):
    """Camera center in world frame, -R^T t, from a world-to-camera extrinsic."""
    ext = np.asarray(ext, dtype=float)
    return -ext[:3, :3].T @ ext[:3, 3]



# ----- Angular distance -----

def test_angular_distance_zero_for_identity():
    R = R_y(37)
    assert angular_distance_deg(R, R) < 1e-6


def test_angular_distance_90_for_quarter_turn():
    d = angular_distance_deg(np.eye(3), R_y(90))
    assert abs(d - 90) < 1e-6


# ----- SO(3) averaging -----

def test_weighted_average_recovers_centroid_for_close_cluster():
    """3 rotations close to each other should average back to roughly the centroid."""
    Rs = [R_y(0), R_y(10), R_y(-5)]
    avg = weighted_average_so3(Rs)
    # Recovered yaw should be close to the arithmetic mean (1.67°) within
    # the SO(3) approximation regime
    forward = avg @ np.array([0, 0, 1])
    yaw = np.degrees(np.arctan2(forward[0], forward[2]))
    assert abs(yaw - 1.67) < 1.0


def test_weighted_average_with_high_weight_dominates():
    """A high-weight rotation should dominate the average."""
    Rs = [R_y(0), R_y(90)]
    avg = weighted_average_so3(Rs, weights=[1.0, 1e9])
    fwd = avg @ np.array([0, 0, 1])
    yaw = np.degrees(np.arctan2(fwd[0], fwd[2]))
    assert abs(yaw - 90) < 0.5


def test_weighted_average_empty_raises():
    with pytest.raises(ValueError):
        weighted_average_so3([])


def test_weighted_average_zero_weight_sum_raises():
    with pytest.raises(ValueError):
        weighted_average_so3([R_y(0), R_y(90)], weights=[0.0, 0.0])


# ----- solve_camera_poses: anchor method -----

def _make_extrinsics_for_yaw(yaw_deg: float, position: np.ndarray = None) -> np.ndarray:
    """Build a 4x4 world-to-cam extrinsic for a camera at given yaw + position."""
    if position is None:
        position = np.zeros(3)
    # Loader form (canonicalize_y_up): camera y points down, so R_c2w = R_y(yaw) @ diag(1,-1,1)
    # and the image-up is world +Y. A bare R_y(yaw) camera is upside down and the solver
    # (correctly) reads gravity from it as -Y.
    R_w2c = (R_y(yaw_deg) @ np.diag([1.0, -1.0, 1.0])).T  # extrinsics is world-to-cam, so transpose
    t_w2c = _world_to_cam_translation(R_w2c, position)
    return _compose_extrinsics(R_w2c, t_w2c)


def test_anchor_solver_propagates_from_cam0():
    """Anchor method: cam0 trusted, cam1/cam2 derived from constraints."""
    ext_in = [
        _make_extrinsics_for_yaw(0),     # cam0: 0°
        _make_extrinsics_for_yaw(78),    # cam1: VGGT noisy (-12° from true 90°)
        _make_extrinsics_for_yaw(172),   # cam2: VGGT noisy (-8° from true 180°)
    ]
    constraints = [
        {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90},
        {"type": "rotation", "from_cam": 0, "to_cam": 2, "yaw": 180},
    ]
    out = solve_camera_poses(ext_in, constraints, method="anchor")

    # cam0 unchanged
    assert np.allclose(out[0], ext_in[0], atol=1e-9)
    # cam1 derived from cam0 + 90°
    fwd1 = out[1][:3, :3].T @ np.array([0, 0, 1])
    yaw1 = np.degrees(np.arctan2(fwd1[0], fwd1[2]))
    assert abs(yaw1 - 90) < 1e-6
    # cam2 derived from cam0 + 180°
    fwd2 = out[2][:3, :3].T @ np.array([0, 0, 1])
    yaw2 = np.degrees(np.arctan2(fwd2[0], fwd2[2]))
    assert abs(abs(yaw2) - 180) < 1e-6


# ----- solve_camera_poses: average method -----

def test_average_solver_distributes_noise_across_cameras():
    """Average method: anchor refined from back-derived estimates."""
    # True: cam0=0, cam1=90, cam2=180. VGGT errors: +5, +12, -8 deg.
    ext_in = [
        _make_extrinsics_for_yaw(5),
        _make_extrinsics_for_yaw(102),
        _make_extrinsics_for_yaw(172),
    ]
    constraints = [
        {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90},
        {"type": "rotation", "from_cam": 0, "to_cam": 2, "yaw": 180},
    ]
    out = solve_camera_poses(ext_in, constraints, method="average")

    # cam0 should be refined toward (5 + (102-90) + (172-180))/3 ≈ +3°
    fwd0 = out[0][:3, :3].T @ np.array([0, 0, 1])
    yaw0 = np.degrees(np.arctan2(fwd0[0], fwd0[2]))
    assert abs(yaw0 - 3.0) < 0.5

    # cam1 = anchor + 90 ≈ 93°
    fwd1 = out[1][:3, :3].T @ np.array([0, 0, 1])
    yaw1 = np.degrees(np.arctan2(fwd1[0], fwd1[2]))
    assert abs(yaw1 - 93.0) < 0.5

    # cam2 = anchor + 180 ≈ 183° (= -177° in atan2 wrap)
    fwd2 = out[2][:3, :3].T @ np.array([0, 0, 1])
    yaw2 = np.degrees(np.arctan2(fwd2[0], fwd2[2]))
    # Either +183 or -177 (equivalent)
    assert abs(((yaw2 + 360) % 360) - 183.0) < 0.5


def test_average_solver_recovers_when_cam0_is_outlier():
    """The user's specific worry: if cam0 is the noisy one, averaging should
    pull the anchor toward the truth via cam1/cam2's back-derived estimates.
    """
    # cam0 is wildly wrong (-25° from true 0°); cam1/cam2 are clean.
    ext_in = [
        _make_extrinsics_for_yaw(-25),
        _make_extrinsics_for_yaw(95),    # +5° from true 90°
        _make_extrinsics_for_yaw(177),   # -3° from true 180°
    ]
    constraints = [
        {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90},
        {"type": "rotation", "from_cam": 0, "to_cam": 2, "yaw": 180},
    ]
    out = solve_camera_poses(ext_in, constraints, method="average")
    fwd0 = out[0][:3, :3].T @ np.array([0, 0, 1])
    yaw0 = np.degrees(np.arctan2(fwd0[0], fwd0[2]))
    # Anchor should be ≈ (-25 + 5 + (-3))/3 ≈ -7.7° — much better than
    # the -25° we'd get from method='anchor'.
    assert abs(yaw0 - (-7.67)) < 1.0


# ----- solve_camera_poses: same_position constraint -----

def test_same_position_anchors_on_lowest_camera():
    """same_position co-locates the listed cameras ON THE ANCHOR (lowest index).

    Any member position is a legal gauge; the anchor keeps camera 0 in place,
    which preserves the cam0-at-origin contract that world-up, the cam0 frame
    and the projection fallback read (see tests/test_pose_constraint_gauge.py).
    """
    ext_in = [
        _make_extrinsics_for_yaw(0,   position=np.array([0.0, 0.0, 0.0])),
        _make_extrinsics_for_yaw(90,  position=np.array([1.0, 0.0, 0.0])),
        _make_extrinsics_for_yaw(180, position=np.array([0.0, 0.0, 1.0])),
    ]
    constraints = [
        {"type": "same_position", "cams": [0, 1, 2]},
    ]
    out = solve_camera_poses(ext_in, constraints)

    pos = [_camera_position_world(o) for o in out]
    assert np.allclose(pos[0], pos[1], atol=1e-9)
    assert np.allclose(pos[1], pos[2], atol=1e-9)
    # The shared position is camera 0's, which must not have moved.
    assert np.allclose(pos[0], np.array([0.0, 0.0, 0.0]), atol=1e-9)


# ----- Empty / pass-through -----

def test_no_constraints_returns_input_unchanged():
    ext_in = [_make_extrinsics_for_yaw(d) for d in (5, 95, 177)]
    out = solve_camera_poses(ext_in, [])
    for a, b in zip(out, ext_in):
        assert np.allclose(a, b, atol=1e-9)


def test_solve_unknown_method_raises():
    ext_in = [_make_extrinsics_for_yaw(0), _make_extrinsics_for_yaw(90)]
    with pytest.raises(ValueError):
        solve_camera_poses(
            ext_in,
            [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90}],
            method="bogus",
        )


# ----- Constraint validation -----

def test_out_of_range_camera_constraint_is_ignored_not_fatal():
    """A constraint naming a camera the scene does not have is dropped; the rest still chain."""
    from saturn.scene.pose_solver import _propagate_constraint_chains, _valid_constraints
    cons = [{"from_cam": 0, "to_cam": 1, "yaw": 90.0}, {"from_cam": 1, "to_cam": 3, "yaw": 45.0}]
    assert len(_valid_constraints(cons, 2)) == 1
    chains = _propagate_constraint_chains(cons, 0, 2)
    assert chains[0] == 0.0 and abs(chains[1] - 90.0) < 1e-9 and 3 not in chains


# --- components of the constraint graph ---------------------------------------------

def _canonical_cam(x: float = 0.0) -> np.ndarray:
    """Canonical Y-up extrinsics of a camera whose centre is at world x."""
    m = np.diag([1.0, -1.0, 1.0, 1.0])
    m[0, 3] = -float(x)
    return m


@pytest.mark.parametrize("method", ["anchor", "average"])
def test_rotation_constraints_apply_in_every_component(method):
    """Two constraint pairs with no camera in common: each pair gets its stated yaw."""
    from saturn.scene.pose_solver import angular_distance_deg
    ext = [_canonical_cam() for _ in range(4)]
    cons = [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90.0},
            {"type": "rotation", "from_cam": 2, "to_cam": 3, "yaw": 45.0}]
    out = solve_camera_poses(ext, cons, method=method)
    assert angular_distance_deg(out[0][:3, :3], out[1][:3, :3]) == pytest.approx(90.0, abs=1e-6)
    assert angular_distance_deg(out[2][:3, :3], out[3][:3, :3]) == pytest.approx(45.0, abs=1e-6)


def test_same_position_groups_merge_regardless_of_order():
    """Cameras 0~1 and 1~2 share one position whichever record comes first."""
    ext = [_canonical_cam(x) for x in (0.0, 1.0, 2.0)]
    a = {"type": "same_position", "cams": [0, 1]}
    b = {"type": "same_position", "cams": [1, 2]}
    centres = lambda cons: [-float(m[0, 3]) for m in solve_camera_poses(ext, cons)]
    assert centres([a, b]) == pytest.approx([0.0, 0.0, 0.0])
    assert centres([b, a]) == pytest.approx([0.0, 0.0, 0.0])
