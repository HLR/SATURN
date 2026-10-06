"""Pose constraint solver — pure-numpy, no LLM, no scipy.

Given (a) per-camera 4x4 extrinsics estimated by VGGT and (b) a list of
relative-pose constraints stated in question text (e.g. "camera 2 is rotated
180 degrees clockwise from camera 0"), produce refined extrinsics that better
satisfy both sources of information.

The default solver is **rotation averaging**: each constraint gives a
back-derived estimate of the anchor camera's pose; we average those estimates
on SO(3) via quaternion mean and propagate the constrained rotations from the
averaged anchor. This handles the case where ANY camera (including the anchor)
is noisy, because the noise spreads across the back-derived pool instead of
biasing the result toward one camera.

Sign convention for yaw:
    yaw = +90 means camera B is rotated 90° clockwise from camera A when
    viewed from above the scene (right-hand rule about world +Y).

This module is intentionally framework-agnostic — it operates on numpy arrays
and Python dicts so it can be unit-tested without loading a Scene.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from saturn.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rotation primitives
# ---------------------------------------------------------------------------

def R_x(deg: float) -> np.ndarray:
    """3x3 rotation about world +X axis by `deg` degrees (right-hand rule)."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def R_y(deg: float) -> np.ndarray:
    """3x3 rotation about world +Y (up) axis by `deg` degrees.

    Positive deg = clockwise when viewed from above (looking down at the floor).
    R_y(+90) takes world +Z (north) to world +X (east).
    """
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def R_z(deg: float) -> np.ndarray:
    """3x3 rotation about world +Z axis by `deg` degrees (right-hand rule)."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def R_axis_angle(axis_vec: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis ``axis_vec`` by ``deg``.

    Sign convention: positive ``deg`` means right-hand rule about ``axis_vec``
    (looking ALONG ``axis_vec`` from origin toward its tip, rotation appears
    clockwise). This makes "yaw=+90 clockwise viewed from above" work for
    any world convention: pass the world-up axis (gravity-down for Y-down
    world, +Y for Y-up world), and the sign on yaw is invariant.
    """
    a = np.asarray(axis_vec, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([
        [0,    -a[2],  a[1]],
        [a[2],  0,    -a[0]],
        [-a[1], a[0],  0],
    ], dtype=float)
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _cam0_is_canonical(extrinsics: Sequence[np.ndarray], tol: float = 0.05) -> bool:
    """True if cam 0 satisfies the VGGT L1 contract: cam 0 at world origin
    AND R ≈ diag(+1, -1, +1).

    Note: diag(+1, -1, +1) has det = -1 (technically a reflection, not a
    rotation), but this is what ``adapters.canonicalize_y_up`` produces by
    negating the Y column of an identity R from VGGT's first-view frame.

    When True, world +Y axis IS gravity-up (image 1's image-up direction
    is the world frame's +Y axis), so hardcoded `world_up = [0,1,0]` in
    pose_solver is exactly correct. When False (cam 0 at another position
    or orientation), `world_up = [0,1,0]` need not be gravity in the
    reconstruction frame, and `empirical_world_up` should be used instead.
    """
    if not extrinsics:
        return False
    ext0 = np.asarray(extrinsics[0], dtype=float)
    if ext0.shape == (3, 4):
        m = np.eye(4)
        m[:3, :] = ext0
        ext0 = m
    R0 = ext0[:3, :3]
    t0 = ext0[:3, 3]
    # cam 0 position in world = -R^T @ t
    pos = -R0.T @ t0
    if float(np.linalg.norm(pos)) > tol:
        return False
    # R deviation from canonical diag(+1, -1, +1)
    R_canonical = np.diag([1.0, -1.0, 1.0])
    if float(np.linalg.norm(R0 - R_canonical)) > tol:
        return False
    return True


def empirical_world_up(extrinsics: Sequence[np.ndarray]) -> np.ndarray:
    """Derive the scene's world-up direction from camera consensus.

    Each camera's image-down direction in world is ``R_w2c[1, :]`` (the
    second row of the world-to-camera rotation). World-up is the opposite:
    averaged across cameras and normalized. Returns ``(0, 1, 0)`` as a
    safe fallback if cameras disagree wildly (e.g., extrinsics all zero).

    Hardcoding world-up as +Y assumes a Y-UP world; in a Y-DOWN world
    (VGGT/OpenCV) a rotation about +Y is *gravity-down* and flips the sign
    of every "clockwise from above" rotation. Deriving it from the cameras
    gives the right axis in either frame.
    """
    ups = []
    for ext in extrinsics:
        E = np.asarray(ext, dtype=float)
        if E.shape == (3, 4) or E.shape == (4, 4):
            R_w2c_row1 = E[1, :3]
            up = -R_w2c_row1
            n = float(np.linalg.norm(up))
            if n > 1e-9:
                ups.append(up / n)
    if not ups:
        return np.array([0.0, 1.0, 0.0])
    mean_up = np.mean(np.stack(ups, axis=0), axis=0)
    n = float(np.linalg.norm(mean_up))
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    return mean_up / n


def angular_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance between two SO(3) rotations, in degrees."""
    M = R1 @ R2.T
    cos_th = np.clip((np.trace(M) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_th)))


# ---------------------------------------------------------------------------
# SO(3) averaging via quaternions (no scipy dependency required)
# ---------------------------------------------------------------------------

def _rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def _slerp_rotation(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two SO(3) rotations.

    t=0 returns R0, t=1 returns R1. For t in between, interpolates along
    the shorter geodesic on the unit-quaternion 3-sphere.

    Two det=-1 inputs (canonical cameras) are handled like
    :func:`weighted_average_so3`: factor out P = diag(1,1,-1), interpolate the
    proper parts, re-apply P. Quaternions cannot represent a reflection.
    """
    R0 = np.asarray(R0, dtype=float)
    R1 = np.asarray(R1, dtype=float)
    if np.linalg.det(R0) < 0 and np.linalg.det(R1) < 0:
        P = np.diag([1.0, 1.0, -1.0])
        return P @ _slerp_rotation(P @ R0, P @ R1, t)
    q0 = _rotation_to_quat(R0)
    q1 = _rotation_to_quat(R1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return _quat_to_rotation(out / np.linalg.norm(out))
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_to_rotation(s0 * q0 + s1 * q1)


def weighted_average_so3(
    rotations: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted mean rotation via quaternion averaging.

    Handles both SO(3) (det=+1) and improper-orthogonal (det=-1) inputs.
    Quaternions can only represent SO(3); to handle det=-1 inputs (which
    VGGT produces post-canonicalize_y_up since the canonical
    R = diag(+1,-1,+1) has det=-1) we factor out a
    fixed reflection P = diag(1,1,-1), average the proper parts via
    quaternion, then re-apply P. All inputs must share the same det sign.

    Algorithm (SO(3) path): convert each rotation to a unit quaternion,
    hemisphere-fix (negate quats whose dot with the first is negative —
    quaternions q and -q represent the same rotation), compute the weighted
    mean in 4-space, re-normalize, convert back. Exact only when rotations
    are tightly clustered; for spread > ~30° this is a first-order
    approximation.

    Args:
        rotations: list of 3x3 orthogonal matrices (det ±1).
        weights: optional non-negative weights (default: equal weighting).

    Returns:
        3x3 orthogonal matrix; det matches the inputs' shared sign.
    """
    if not rotations:
        raise ValueError("weighted_average_so3: empty rotations list")
    n = len(rotations)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} mismatched to rotations count {n}")
    if np.any(w < 0):
        raise ValueError("weighted_average_so3: weights must be non-negative")
    if w.sum() <= 0:
        raise ValueError("weighted_average_so3: weights sum to zero")

    Rs = [np.asarray(R, dtype=float) for R in rotations]
    dets = np.array([np.linalg.det(R) for R in Rs])
    # Factor out a fixed reflection P for det=-1 inputs so the quaternion
    # average operates on proper rotations (quats can't represent reflections;
    # naive conversion of a det=-1 matrix silently produces a det=+1 result
    # that can be 90°+ off from the input).
    if np.all(dets < 0):
        P = np.diag([1.0, 1.0, -1.0])  # det = -1
        Rs = [P @ R for R in Rs]       # all become det = +1
        re_apply_P = True
    elif np.all(dets > 0):
        re_apply_P = False
    else:
        # Mixed signs: O(3) average isn't well-defined; fall back to naive
        # quaternion average. Callers should ensure consistent chirality.
        re_apply_P = False

    quats = np.array([_rotation_to_quat(R) for R in Rs])
    # Hemisphere fix: align all quats to the same hemisphere as the first.
    for i in range(1, n):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    q_mean = (quats * w[:, None]).sum(axis=0) / w.sum()
    q_mean = q_mean / np.linalg.norm(q_mean)
    R_mean = _quat_to_rotation(q_mean)
    return (np.diag([1.0, 1.0, -1.0]) @ R_mean) if re_apply_P else R_mean


# ---------------------------------------------------------------------------
# Constraint records (plain dicts so this module is JSON-serializable)
# ---------------------------------------------------------------------------

# A rotation constraint:
#   {"type": "rotation", "from_cam": int, "to_cam": int, "yaw": float, "axis": str}
# A same-position constraint:
#   {"type": "same_position", "cams": [int, int, ...]}

ROTATION = "rotation"
SAME_POSITION = "same_position"


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _decompose_extrinsics(ext: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 world-to-camera extrinsics into (R_w2c, t_w2c)."""
    ext = np.asarray(ext, dtype=float)
    return ext[:3, :3].copy(), ext[:3, 3].copy()


def _compose_extrinsics(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Build a 4x4 extrinsics matrix from rotation and translation."""
    out = np.eye(4, dtype=float)
    out[:3, :3] = R_w2c
    out[:3, 3] = t_w2c
    return out


def _world_to_cam_translation(R_w2c: np.ndarray, position_world: np.ndarray) -> np.ndarray:
    """Convert a desired world camera-center back to the t_w2c translation field."""
    return -R_w2c @ position_world


def _blend_rotation(
    R_orig: np.ndarray,
    R_target: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Blend a backbone rotation toward a constraint-derived one.

    Single definition of the ``strength`` semantics documented on
    :func:`solve_camera_poses`: 0 keeps the backbone estimate, 1 fully applies
    the constraint, in between SLERPs. Applies to the anchor and the
    propagated cameras alike.
    """
    if strength >= 1.0 - 1e-9:
        return R_target
    if strength <= 1e-9:
        return R_orig
    return _slerp_rotation(R_orig, R_target, strength)


def solve_camera_poses(
    extrinsics: Sequence[np.ndarray],
    constraints: Sequence[dict],
    method: str = "average",
    weights: Optional[Sequence[float]] = None,
    *,
    strength: float = 1.0,
) -> List[np.ndarray]:
    """Refine VGGT camera extrinsics to better satisfy stated constraints.

    Precondition:
        Extrinsics must live in the canonical Y-UP world (X-right,
        Y-up, Z-forward). Production loaders (``load.py`` / ``load_async.py``)
        pass raw VGGT extrinsics through ``adapters.canonicalize_y_up``
        before they reach the scene, so this assumption holds for all
        ``Scene`` inputs. Callers that synthesize extrinsics directly must
        ensure the same — passing raw Y-DOWN OpenCV extrinsics will produce
        sign-flipped rotations.

    Args:
        extrinsics: list of 4x4 world-to-camera matrices, one per camera.
        constraints: list of constraint records (rotation + same_position).
        method: 'anchor' (trust cam 0, derive others) or 'average' (rotation
                averaging via back-derivation; recommended default).
        weights: optional per-camera VGGT confidence weights for averaging.
        strength: SLERP weight in [0, 1] blending the original backbone
            rotation toward the constraint-propagated rotation.
            strength=1.0 → fully apply constraint.
            strength=0.0 → ignore constraint, return original rotation.
            Between → soft trust of the backbone's noisier pose. Useful when
            camera poses have residual error that hard constraint
            application amplifies.

    Returns:
        New list of 4x4 extrinsics, same length as input.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    n = len(extrinsics)
    if n == 0:
        return []

    rot_constraints = [c for c in constraints if c.get("type") == ROTATION]
    sp_constraints  = [c for c in constraints if c.get("type") == SAME_POSITION]

    # Canonical internal representation: (camera-to-world rotation, world
    # position). NOT (R, t_w2c) — the extrinsics translation is *derived*
    # (t_w2c = -R_w2c @ position), so rewriting R while leaving t alone would
    # move the camera around the world origin. Rotation constraints touch
    # only the rotations, ``same_position`` touches only the positions, and
    # extrinsics are recomposed exactly once, at the end.
    Rs_c2w, positions = _rotations_and_positions(extrinsics)
    if rot_constraints:
        Rs_c2w = _apply_rotation_constraints(
            Rs_c2w, extrinsics, rot_constraints, method, weights, strength,
        )
    if sp_constraints:
        positions = _apply_same_position_constraints(positions, sp_constraints)
    return _recompose_extrinsics(Rs_c2w, positions)


def _rotations_and_positions(
    extrinsics: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Camera-to-world rotation and camera centre in world, per camera."""
    n = len(extrinsics)
    Rs_c2w = [None] * n
    positions = [None] * n
    for i, ext in enumerate(extrinsics):
        R_w2c, t_w2c = _decompose_extrinsics(ext)
        Rs_c2w[i] = R_w2c.T
        positions[i] = -R_w2c.T @ t_w2c   # camera centre in world
    return Rs_c2w, positions


def _world_up_axis(extrinsics: Sequence[np.ndarray]) -> np.ndarray:
    """Axis that yaw constraints rotate about.

    When the backbone satisfies the L1 contract (cam 0 at origin + canonical
    R = diag(+1,-1,+1), as VGGT produces), world +Y IS gravity-up.
    Otherwise world +Y may be a tilted axis in the reconstruction frame, so
    gravity comes from camera consensus instead. For canonical backbones +Y
    is exact, while averaging across views would dilute it.
    """
    if _cam0_is_canonical(extrinsics):
        return np.array([0.0, 1.0, 0.0])
    return empirical_world_up(extrinsics)


def _apply_rotation_constraints(
    Rs_c2w: List[np.ndarray],
    extrinsics: Sequence[np.ndarray],
    rot_constraints: Sequence[dict],
    method: str,
    weights: Optional[Sequence[float]],
    strength: float,
) -> List[np.ndarray]:
    """Camera-to-world rotations after the rotation constraints.

    Each connected component of the constraint graph is solved on its own
    anchor (its lowest-index camera), which gets a target rotation:
    the backbone's own ('anchor') or the average of the anchor estimates
    back-derived from every constrained camera ('average'). The other
    constrained cameras follow from the refined anchor:
        R_c2w[to] = R_world_up(total_yaw) @ R_c2w[anchor]
    i.e. the camera body rotates by total_yaw about the scene's world-up axis
    (+yaw = right-hand rule about world_up = clockwise viewed from above).
    Every constraint-derived rotation, the anchor's included, is blended with
    the backbone rotation by ``strength`` (SLERP when below 1).
    A stated "the camera turned N degrees" is a rotation IN PLACE, so the
    camera positions do not change here.
    """
    n = len(Rs_c2w)
    world_up = _world_up_axis(extrinsics)
    Rs_c2w_orig = [R.copy() for R in Rs_c2w]   # pre-constraint snapshot
    Rs_c2w = list(Rs_c2w)

    if method not in ("anchor", "average"):
        raise ValueError(f"Unknown solver method '{method}' (use 'anchor' or 'average')")
    rot_constraints = _valid_constraints(rot_constraints, n)

    # Cameras linked by constraints form connected components; each component
    # is solved on its own anchor (its lowest-index camera), since constraints
    # say nothing about how separate components relate to each other.
    unsolved = {int(c[k]) for c in rot_constraints for k in ("from_cam", "to_cam")}
    while unsolved:
        anchor_cam = min(unsolved)
        if method == "anchor":
            R_anchor_target = Rs_c2w_orig[anchor_cam]      # unchanged: trust VGGT
        else:
            R_anchor_target = _averaged_anchor_rotation(
                Rs_c2w_orig, rot_constraints, anchor_cam, weights, world_up=world_up,
            )
        Rs_c2w[anchor_cam] = _blend_rotation(
            Rs_c2w_orig[anchor_cam], R_anchor_target, strength,
        )

        chains = _propagate_constraint_chains(rot_constraints, anchor_cam, n)
        for cam_idx, total_yaw in chains.items():
            if cam_idx == anchor_cam:
                continue
            R_propagated = R_axis_angle(world_up, total_yaw) @ Rs_c2w[anchor_cam]
            Rs_c2w[cam_idx] = _blend_rotation(
                Rs_c2w_orig[cam_idx], R_propagated, strength,
            )
        unsolved -= set(chains)
    return Rs_c2w


def _apply_same_position_constraints(
    positions: List[np.ndarray],
    sp_constraints: Sequence[dict],
) -> List[np.ndarray]:
    """Camera positions after the same_position constraints.

    Sets that share a camera are merged first (sharing a position is
    transitive), so the result does not depend on the order of the records.
    Every camera in a merged set moves onto the position of the set's
    lowest-index camera, not onto their mean: both give identical relative
    geometry, but the mean also displaces camera 0 and breaks the L1 contract
    (cam 0 at the origin with the canonical rotation) that downstream frames
    and projection rely on.
    """
    positions = list(positions)
    parent: dict = {}

    def root(i: int) -> int:
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for sp in sp_constraints:
        cams_in_set = [int(i) for i in sp.get("cams", []) if 0 <= int(i) < len(positions)]
        if len(cams_in_set) < 2:
            continue
        for i in cams_in_set[1:]:
            parent[root(i)] = root(cams_in_set[0])

    groups: dict = {}
    for i in parent:
        groups.setdefault(root(i), []).append(i)
    for members in groups.values():
        target = positions[min(members)]
        for i in members:
            positions[i] = target
    return positions


def _recompose_extrinsics(
    Rs_c2w: Sequence[np.ndarray],
    positions: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """4x4 world-to-camera extrinsics from (camera-to-world rotation, world
    position). t_w2c is derived here and nowhere else, so it cannot drift out
    of sync with R."""
    out = []
    for i in range(len(Rs_c2w)):
        R_w2c_i = Rs_c2w[i].T
        out.append(_compose_extrinsics(R_w2c_i, _world_to_cam_translation(R_w2c_i, positions[i])))
    return out


def _valid_constraints(rot_constraints: Sequence[dict], num_cams: int) -> list:
    """Drop constraints naming a camera the scene does not have (planner
    hallucination, e.g. "camera 3" in a 2-view scene) instead of crashing."""
    keep, bad = [], []
    for c in rot_constraints:
        f, t = int(c["from_cam"]), int(c["to_cam"])
        (keep if 0 <= f < num_cams and 0 <= t < num_cams else bad).append(c)
    if bad:
        log.warning(f"[pose] ignoring {len(bad)} constraint(s) with camera index outside 0..{num_cams-1}")
    return keep


def _propagate_constraint_chains(
    rot_constraints: Sequence[dict],
    anchor_cam: int,
    num_cams: int,
) -> dict:
    """Compute the cumulative yaw from anchor_cam to every reachable camera.

    Constraints can be expressed in either direction (from→to or to→from).
    We BFS the constraint graph from anchor_cam, accumulating signed yaw.

    Returns:
        {cam_idx: total_yaw_from_anchor}  for every cam reachable from anchor.
    """
    # Build adjacency list: cam -> [(neighbor, signed_yaw)]
    rot_constraints = _valid_constraints(rot_constraints, num_cams)
    adj: dict = {i: [] for i in range(num_cams)}
    for c in rot_constraints:
        f, t = int(c["from_cam"]), int(c["to_cam"])
        yaw = float(c["yaw"])
        adj[f].append((t, +yaw))   # f→t: rotating from f by +yaw lands on t
        adj[t].append((f, -yaw))   # t→f: rotating from t by -yaw lands on f

    chains = {anchor_cam: 0.0}
    queue = [anchor_cam]
    while queue:
        cur = queue.pop(0)
        for (nbr, dy) in adj[cur]:
            if nbr in chains:
                continue
            chains[nbr] = chains[cur] + dy
            queue.append(nbr)
    return chains


def _averaged_anchor_rotation(
    Rs_c2w: Sequence[np.ndarray],
    rot_constraints: Sequence[dict],
    anchor_cam: int,
    weights: Optional[Sequence[float]] = None,
    *,
    world_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Back-derive estimates of anchor_cam from each constrained camera,
    average them on SO(3) with optional per-camera confidence weights.

    Operates on R_c2w (camera-to-world) matrices. Constraint propagation rule:
        R_c2w[to] = R_world_up(total_yaw) @ R_c2w[anchor]
    so the inverse direction is:
        R_c2w[anchor] = R_world_up(-total_yaw) @ R_c2w[to]

    ``world_up`` is the unit axis vector to rotate about. If None, defaults
    to ``+Y``; see :func:`empirical_world_up` for deriving it from the
    cameras.
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0])
    rot_constraints = _valid_constraints(rot_constraints, len(Rs_c2w))
    chains = _propagate_constraint_chains(rot_constraints, anchor_cam, len(Rs_c2w))
    candidates: List[np.ndarray] = []
    candidate_weights: List[float] = []
    for cam_idx, total_yaw in chains.items():
        R_anchor_via = R_axis_angle(world_up, -total_yaw) @ Rs_c2w[cam_idx]
        candidates.append(R_anchor_via)
        if weights is not None:
            candidate_weights.append(float(weights[cam_idx]))
        else:
            candidate_weights.append(1.0)
    return weighted_average_so3(candidates, candidate_weights)
