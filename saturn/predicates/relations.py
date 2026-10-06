"""
Relation computation for arbitrary reference frames.

Computes directional relations (left/right/front/behind/above/below),
frame-independent relations (facing/parallel/perpendicular/between),
distance metrics, and per-object facing scores — all in world-frame 3D.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from saturn.log import get_logger
from saturn.predicates.scoring import steep_sigmoid_signed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frame-dependent directional relations
# ---------------------------------------------------------------------------

# Combined labels -> their two axial components (sigmoid family keys).
DIRECTIONAL_COMBINATIONS = {
    "front_left": ("front", "left"),
    "front_right": ("front", "right"),
    "behind_left": ("behind", "left"),
    "behind_right": ("behind", "right"),
}


# Every key of compute_frame_relations' result.
_FRAME_RELATION_NAMES = (
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "front_left",
    "front_right",
    "behind_left",
    "behind_right",
    "left_normalized",
    "right_normalized",
    "front_normalized",
    "behind_normalized",
    "above_normalized",
    "below_normalized",
)


def compute_frame_relations(
    positions: np.ndarray,
    frame_right: np.ndarray,
    frame_up: np.ndarray,
    frame_front: np.ndarray,
    scene_scale: float = 1.0,
    steepness: float = 14.0,
) -> Dict[str, np.ndarray]:
    """Compute directional relations for objects in a given reference frame.

    The signed projections of the pairwise displacement are divided by
    ``s_scene`` (the 0.9-quantile of pairwise entity distances) before the
    sigmoid, so that ``S_r = sigmoid((h_r(delta / s_scene) - 0) / tau)``
    with ``tau = 1 / steepness``.  The caller passes ``s_scene`` from
    ``Scene._compute_scene_scale``.  A non-finite or non-positive scale is
    treated as ``1.0`` (raw units, a no-op).

    Parameters
    ----------
    positions : (K, 3) world-frame positions of objects
    frame_right, frame_up, frame_front : (3,) unit vectors defining the frame
    scene_scale : ``s_scene`` divisor applied to the signed projections
    steepness : sigmoid steepness (``1 / tau_dir``)

    Returns
    -------
    dict with keys: left, right, front, behind, above, below,
                    front_left, front_right, behind_left, behind_right,
                    left_normalized, right_normalized, ...
    All values are (K, K) float arrays.
    """
    if len(positions) == 0:
        return {name: np.zeros((0, 0), dtype=float) for name in _FRAME_RELATION_NAMES}

    disp_right, disp_up, disp_front = _frame_displacements(
        positions, frame_right, frame_up, frame_front
    )
    s_scene = _scene_scale_divisor(scene_scale)
    result = _sigmoid_frame_relations(
        disp_right / s_scene, disp_up / s_scene, disp_front / s_scene, steepness
    )
    result.update(_normalized_frame_relations(disp_right, disp_up, disp_front))
    return result


def _frame_displacements(positions, frame_right, frame_up, frame_front):
    """Pairwise displacements along the frame axes, as three (K, K) arrays.

    Entry ``[i, j]`` of each array is ``pos_i - pos_j`` projected onto the
    frame's right / up / front axis; the diagonals are zero.
    """
    positions = np.asarray(positions, dtype=float)
    frame_right = np.asarray(frame_right, dtype=float).ravel()
    frame_up = np.asarray(frame_up, dtype=float).ravel()
    frame_front = np.asarray(frame_front, dtype=float).ravel()

    P = positions @ np.column_stack([frame_right, frame_up, frame_front])   # (K, 3)
    disp = P[:, None, :] - P[None, :, :]                                     # (K, K, 3)
    disp_right, disp_up, disp_front = (np.ascontiguousarray(disp[..., k]) for k in range(3))
    for m in (disp_right, disp_up, disp_front):
        np.fill_diagonal(m, 0.0)
    return disp_right, disp_up, disp_front


def _scene_scale_divisor(scene_scale) -> float:
    """``s_scene`` as a float, or 1.0 when it is non-finite or not positive."""
    try:
        s_scene = float(scene_scale)
    except (TypeError, ValueError):
        s_scene = float("nan")
    if not math.isfinite(s_scene) or s_scene <= 0.0:
        log.debug(
            "compute_frame_relations: invalid scene_scale=%r, using 1.0 (raw units)",
            scene_scale,
        )
        s_scene = 1.0
    return s_scene


def _sigmoid_frame_relations(n_right, n_up, n_front, steepness) -> Dict[str, np.ndarray]:
    """Axial and combined directional scores from scale-normalised displacements."""
    result = {}
    result["left"] = steep_sigmoid_signed(-n_right, steepness=steepness)
    result["right"] = steep_sigmoid_signed(n_right, steepness=steepness)
    result["above"] = steep_sigmoid_signed(n_up, steepness=steepness)
    result["below"] = steep_sigmoid_signed(-n_up, steepness=steepness)
    # "i is behind j" = i is farther from the observer than j (occluded by j).
    # disp_front[i,j] > 0 when i is further along the front axis → i IS behind j.
    result["front"] = steep_sigmoid_signed(-n_front, steepness=steepness)
    result["behind"] = steep_sigmoid_signed(n_front, steepness=steepness)

    # Directional combinations: the evidence of a combined label is the min
    # over its component evidences,
    #   h_{front-left} = min(h_front, h_left),
    # then the same sigmoid with m_comb = 0, tau_comb = 1 / steepness.
    h = {"front": -n_front, "behind": n_front, "left": -n_right, "right": n_right}
    for key, (c1, c2) in DIRECTIONAL_COMBINATIONS.items():
        result[key] = steep_sigmoid_signed(np.minimum(h[c1], h[c2]), steepness=steepness)

    for key in result:
        np.fill_diagonal(result[key], 0.0)
    return result


def _normalized_frame_relations(disp_right, disp_up, disp_front) -> Dict[str, np.ndarray]:
    """The ``*_normalized`` scores: angular components mapped to [0, 1].

    These rank by the angular component along an axis rather than the
    near-binary sigmoid:
      left/right:   sin(yaw) = dr / sqrt(dr² + df²)  (+1 pure right, -1 pure left)
      front/behind: cos(yaw) = df / sqrt(dr² + df²)
      above/below:  sin(elevation)
    each mapped to [0, 1] via (1 ± value) / 2. Pairs with no horizontal
    offset, and the diagonal, score 0.
    """
    K = disp_right.shape[0]
    horiz = np.sqrt(disp_right ** 2 + disp_front ** 2)
    valid = (horiz >= 1e-12) & ~np.eye(K, dtype=bool)
    safe = np.where(valid, horiz, 1.0)
    sin_yaw = np.where(valid, disp_right / safe, 0.0)      # +1 = pure right, -1 = pure left
    cos_yaw = np.where(valid, disp_front / safe, 0.0)      # +1 = i straight behind j, -1 = straight in front
    sin_elev = np.where(valid, np.sin(np.arctan2(disp_up, safe)), 0.0)
    return {
        "left_normalized": np.where(valid, (1.0 - sin_yaw) / 2.0, 0.0),
        "right_normalized": np.where(valid, (1.0 + sin_yaw) / 2.0, 0.0),
        "front_normalized": np.where(valid, (1.0 - cos_yaw) / 2.0, 0.0),
        "behind_normalized": np.where(valid, (1.0 + cos_yaw) / 2.0, 0.0),
        "above_normalized": np.where(valid, (1.0 + sin_elev) / 2.0, 0.0),
        "below_normalized": np.where(valid, (1.0 - sin_elev) / 2.0, 0.0),
    }


def compute_frame_obj_facing(
    front_directions: np.ndarray,
    elevations: np.ndarray,
    frame_right: np.ndarray,
    frame_up: np.ndarray,
    frame_front: np.ndarray,
    steepness: float = 14.0,
) -> Dict[str, np.ndarray]:
    """Compute per-object facing scores in a given frame.

    Parameters
    ----------
    front_directions : (K, 3) world-frame front directions per object
    elevations : (K,) elevation angles in degrees
    frame_right, frame_up, frame_front : (3,) unit vectors defining the frame

    Returns
    -------
    dict with keys: obj_facing_left, obj_facing_right, obj_facing_front,
                    obj_facing_back, obj_facing_up, obj_facing_down,
                    obj_facing_front_right, obj_facing_front_left,
                    obj_facing_back_right, obj_facing_back_left
    All values are (K,) float arrays.
    """
    _SQRT2 = float(np.sqrt(2))
    K = len(front_directions)
    result = {
        "obj_facing_left": np.zeros(K, dtype=float),
        "obj_facing_right": np.zeros(K, dtype=float),
        "obj_facing_front": np.zeros(K, dtype=float),
        "obj_facing_back": np.zeros(K, dtype=float),
        "obj_facing_up": np.zeros(K, dtype=float),
        "obj_facing_down": np.zeros(K, dtype=float),
        "obj_facing_front_right": np.zeros(K, dtype=float),
        "obj_facing_front_left": np.zeros(K, dtype=float),
        "obj_facing_back_right": np.zeros(K, dtype=float),
        "obj_facing_back_left": np.zeros(K, dtype=float),
    }
    if K == 0:
        return result

    front_directions = np.asarray(front_directions, dtype=float)
    elevations = np.asarray(elevations, dtype=float)

    for i in range(K):
        fwd = front_directions[i]
        if np.linalg.norm(fwd) < 1e-8:
            continue

        # Project onto frame axes
        dot_right = np.dot(fwd, frame_right)
        dot_front = np.dot(fwd, frame_front)
        el = elevations[i]

        # "facing left in this frame" = object's front has negative right-component
        result["obj_facing_left"][i] = float(
            steep_sigmoid_signed(np.array([-dot_right]), steepness=steepness)[0]
        )
        result["obj_facing_right"][i] = float(
            steep_sigmoid_signed(np.array([dot_right]), steepness=steepness)[0]
        )
        result["obj_facing_front"][i] = float(
            steep_sigmoid_signed(np.array([dot_front]), steepness=steepness)[0]
        )
        result["obj_facing_back"][i] = float(
            steep_sigmoid_signed(np.array([-dot_front]), steepness=steepness)[0]
        )

        sin_el = np.sin(np.deg2rad(el))
        result["obj_facing_up"][i] = float(
            steep_sigmoid_signed(np.array([sin_el]), steepness=steepness)[0]
        )
        result["obj_facing_down"][i] = float(
            steep_sigmoid_signed(np.array([-sin_el]), steepness=steepness)[0]
        )

        # Diagonal directions: project onto 45° unit vectors in the horizontal plane
        result["obj_facing_front_right"][i] = float(
            steep_sigmoid_signed(
                np.array([(dot_front + dot_right) / _SQRT2]), steepness=steepness
            )[0]
        )
        result["obj_facing_front_left"][i] = float(
            steep_sigmoid_signed(
                np.array([(dot_front - dot_right) / _SQRT2]), steepness=steepness
            )[0]
        )
        result["obj_facing_back_right"][i] = float(
            steep_sigmoid_signed(
                np.array([(-dot_front + dot_right) / _SQRT2]), steepness=steepness
            )[0]
        )
        result["obj_facing_back_left"][i] = float(
            steep_sigmoid_signed(
                np.array([(-dot_front - dot_right) / _SQRT2]), steepness=steepness
            )[0]
        )

    return result


# ---------------------------------------------------------------------------
# Frame-independent relations
# ---------------------------------------------------------------------------


def compute_frame_independent_relations(
    positions: np.ndarray,
    front_directions: np.ndarray,
    steepness: float = 14.0,
) -> Dict[str, np.ndarray]:
    """Compute frame-independent relations from world-space geometry.

    Parameters
    ----------
    positions : (K, 3) world positions
    front_directions : (K, 3) world front direction per object

    Returns
    -------
    dict with keys: facing, parallel, perpendicular, orientation_distance, between
    facing/parallel/perpendicular/orientation_distance: (K,K)
    between: (K,K,K)
    """
    K = len(positions)
    result = {
        "facing": np.zeros((K, K), dtype=float),
        "parallel": np.zeros((K, K), dtype=float),
        "perpendicular": np.zeros((K, K), dtype=float),
        "orientation_distance": np.zeros((K, K), dtype=float),
        "between": np.zeros((K, K, K), dtype=float),
    }
    if K == 0:
        return result

    positions = np.asarray(positions, dtype=float)
    front_directions = np.asarray(front_directions, dtype=float)

    # Pairwise facing: is object i facing toward object j?
    # Vectorised over all (i, j): facing needs a valid f_i and a nonzero
    # horizontal offset; the orientation relations also need a valid f_j.
    f_norm = np.linalg.norm(front_directions, axis=1)
    f_ok = f_norm >= 1e-8
    f_unit = np.where(f_ok[:, None], front_directions / np.where(f_ok, f_norm, 1.0)[:, None], 0.0)
    vec = positions[None, :, :] - positions[:, None, :]          # vec_ij = p_j - p_i
    vec_h = vec.copy()
    vec_h[..., 1] = 0.0
    dist_h = np.linalg.norm(vec_h, axis=-1)
    off_diag = ~np.eye(K, dtype=bool)
    ok_ij = f_ok[:, None] & off_diag & (dist_h >= 1e-8)
    dot = np.einsum("id,ijd->ij", f_unit, vec_h / np.where(ok_ij, dist_h, 1.0)[..., None])
    result["facing"] = np.where(ok_ij, steep_sigmoid_signed(dot, steepness=steepness), 0.0)
    ok_or = ok_ij & f_ok[None, :]
    a = np.abs(f_unit @ f_unit.T)
    result["parallel"] = np.where(ok_or, steep_sigmoid_signed(a, midpoint=0.7, steepness=steepness), 0.0)
    result["perpendicular"] = np.where(ok_or, steep_sigmoid_signed(0.3 - a, steepness=steepness), 0.0)
    # Signed dot: 0 = same direction, 1 = opposite (abs() above is only for
    # the axis-agnostic parallel / perpendicular tests).
    cos_ij = np.clip(f_unit @ f_unit.T, -1.0, 1.0)
    result["orientation_distance"] = np.where(ok_or, np.degrees(np.arccos(cos_ij)) / 180.0, 0.0)
    if K >= 3:
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        D_ji = distances[:, :, np.newaxis]
        D_ik = distances[:, np.newaxis, :]
        D_jk = distances[np.newaxis, :, :]
        deviation = (D_ji + D_ik) - D_jk
        normalized_deviation = deviation / (D_jk + 1e-6)
        between_scores = np.clip(1.0 - normalized_deviation, 0.0, 1.0)

        idxs = np.arange(K)
        i_idx, j_idx, k_idx = np.meshgrid(idxs, idxs, idxs, indexing="ij")
        between_scores *= (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        result["between"] = between_scores

    return result


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


def _stride(pc: np.ndarray, max_pts: int) -> np.ndarray:
    """Deterministic uniform subsample of at most max_pts rows."""
    pc = np.asarray(pc, dtype=float)
    if len(pc) <= max_pts:
        return pc
    return pc[np.linspace(0, len(pc) - 1, max_pts).astype(int)]


def compute_distance_matrices(
    positions: np.ndarray,
    point_clouds: Optional[List[Optional[np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """Compute center and edge distance matrices.

    Parameters
    ----------
    positions : (K, 3) world positions (centers)
    point_clouds : optional list of (N_i, 3) point clouds per object

    Returns
    -------
    dict with: distance_center_raw (K,K), distance_edge_raw (K,K),
               distance (K,K) normalized, distance_edge (K,K) normalized,
               closeness (K,K)
    """
    K = len(positions)
    result = {}
    if K == 0:
        for name in (
            "distance_center_raw",
            "distance_edge_raw",
            "distance",
            "distance_edge",
            "closeness",
        ):
            result[name] = np.zeros((0, 0), dtype=float)
        return result

    positions = np.asarray(positions, dtype=float)

    # Center-to-center
    center_dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    result["distance_center_raw"] = center_dists

    # Edge-to-edge (closest points between point clouds, or fallback to center)
    edge_dists = center_dists.copy()
    if point_clouds is not None:
        for i in range(K):
            for j in range(i + 1, K):
                pc_i = point_clouds[i]
                pc_j = point_clouds[j]
                if (
                    pc_i is not None
                    and pc_j is not None
                    and len(pc_i) > 0
                    and len(pc_j) > 0
                ):
                    # Deterministic uniform-stride subsample, then exact
                    # nearest-neighbour distance via a KD-tree.
                    pc_i, pc_j = _stride(pc_i, 2000), _stride(pc_j, 2000)
                    min_dist = float(cKDTree(pc_j).query(pc_i, k=1)[0].min())
                    edge_dists[i, j] = min_dist
                    edge_dists[j, i] = min_dist

    result["distance_edge_raw"] = edge_dists

    # Normalize. distance / distance_edge divide by the largest distance, so they keep
    # the full ranking (the farthest pair is 1). closeness is 1 - distance scaled by the
    # 95th percentile and clipped: a degree of "near" that saturates for far pairs.
    for prefix, raw in [("distance", center_dists), ("distance_edge", edge_dists)]:
        top = float(raw.max()) if raw.size else 0.0
        normalized = raw / top if top > 1e-6 else np.zeros_like(raw)
        np.fill_diagonal(normalized, 0.0)
        result[prefix] = normalized

    positive = center_dists[center_dists > 0]
    p95 = float(np.percentile(positive, 95)) + 1e-6 if len(positive) > 0 else 1.0
    near_scale = np.clip(center_dists / p95, 0.0, 1.0) if p95 > 1e-6 else np.zeros_like(center_dists)
    result["closeness"] = 1.0 - near_scale
    np.fill_diagonal(result["closeness"], 0.0)

    return result
