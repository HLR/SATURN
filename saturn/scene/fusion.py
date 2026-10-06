"""
Multi-view object fusion and ground plane leveling.

Merges per-view detections by keyword, fuses orientations, fits OBBs,
and optionally levels the ground plane.
"""

from __future__ import annotations

from saturn.settings import env
from collections import defaultdict
from itertools import combinations, groupby
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
from saturn.log import get_logger

log = get_logger(__name__)

# Debug flag (SAPY_FUSION_DEBUG): log the same-view dedup and each keyword's observations.
_FUSION_DEBUG = bool(env("SAPY_FUSION_DEBUG"))

# assign rule: max cross-view offset as a fraction of depth (bounds reconstruction drift).
_ASSIGN_K = 0.15
# assign rule, same-view dedup: a box is a duplicate if intersection-over-smaller
# >= this, and >= this fraction of the small box's points lie in the big box's
# depth range.
_SAMEVIEW_IOS_THRESHOLD = 0.7
_SAMEVIEW_DEPTH_INSIDE = 0.5


# ---------------------------------------------------------------------------
# Rotation matrix utilities
# ---------------------------------------------------------------------------


def _ensure_proper_rotation(mat: np.ndarray) -> np.ndarray:
    """Force a 3x3 matrix to be a proper rotation (det = +1, orthonormal).

    Uses SVD to find the closest proper rotation matrix.
    """
    mat = np.asarray(mat, dtype=float)
    U, _, Vt = np.linalg.svd(mat)
    # Ensure right-handed (det = +1)
    d = np.linalg.det(U @ Vt)
    S = np.diag([1.0, 1.0, np.sign(d)])
    return U @ S @ Vt


# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------


def denoise_point_cloud(
    points: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    mad_k: float = 3.5,
    min_points: int = 10,
) -> np.ndarray:
    """Remove outliers by per-axis percentile clipping + MAD rejection.

    Two-stage filtering:
    1. Per-axis percentile clipping (removes extreme outliers).
    2. MAD (Median Absolute Deviation) rejection on each axis — removes
       points farther than ``mad_k * 1.4826 * MAD`` from the median.
    """
    if len(points) < min_points:
        return points
    mask = np.ones(len(points), dtype=bool)
    for dim in range(3):
        lo = np.percentile(points[:, dim], percentile_low)
        hi = np.percentile(points[:, dim], percentile_high)
        mask &= (points[:, dim] >= lo) & (points[:, dim] <= hi)
    filtered = points[mask]
    if len(filtered) < min_points:
        return points

    # Stage 2: MAD rejection
    med = np.median(filtered, axis=0)
    mad = np.median(np.abs(filtered - med), axis=0)
    robust_scale = 1.4826 * mad + 1e-8
    mad_mask = np.all(np.abs(filtered - med) <= (mad_k * robust_scale), axis=1)
    result = filtered[mad_mask]
    return result if len(result) >= min_points else filtered


def downsample_point_cloud(points: np.ndarray, max_points: int = 512) -> np.ndarray:
    """Deterministically subsample a point cloud to at most ``max_points`` evenly spaced points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, num=max_points, dtype=int)
    return pts[idx]


def summarize_point_cloud(points: np.ndarray) -> Optional[Dict[str, Any]]:
    """Build a compact geometric summary for a detection point cloud."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return None

    pts = denoise_point_cloud(pts)
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    dims = hi - lo
    diag = float(np.linalg.norm(dims))
    center = np.median(pts, axis=0)
    sample = downsample_point_cloud(pts, max_points=512)

    normal = None
    planarity = 0.0
    if len(sample) >= 3:
        centered = sample - np.mean(sample, axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(sample) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        normal = eigvecs[:, 0]
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-8:
            normal = normal / normal_norm
        else:
            normal = None
        if eigvals[-1] > 1e-12:
            planarity = float((eigvals[1] - eigvals[0]) / (eigvals[-1] + 1e-12))

    return {
        "points": pts,
        "sample": sample,
        "center": center,
        "lo": lo,
        "hi": hi,
        "dims": dims,
        "diag": diag,
        "max_extent": float(np.max(dims)) if len(dims) else 0.0,
        "normal": normal,
        "planarity": planarity,
    }


def _ios_boxes(a, b) -> float:
    """Intersection over the SMALLER box. Nested boxes score ~1 even when IoU is low."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    a2 = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(a1, a2)


def _same_depth(small: Dict[str, Any], big: Dict[str, Any], cam_pos, thr: float) -> bool:
    """Is the SMALL detection at the same depth as the BIG one it is nested in?

    Fraction of the small cloud's points whose camera distance falls inside the
    big cloud's p5-p95 range. A part of an object scores ~1; an object that
    merely occludes another from this viewpoint scores ~0. Both clouds come
    from ONE depth map, so this has zero reconstruction error and needs no
    tolerance constant. Returns True (= treat as duplicate) when it cannot tell,
    which leaves the 2D test alone in charge.
    """
    sp, bp = small.get("world_points"), big.get("world_points")
    if cam_pos is None or sp is None or bp is None or len(sp) < 5 or len(bp) < 5:
        return True
    ds = np.linalg.norm(np.asarray(sp, float) - cam_pos, axis=1)
    db = np.linalg.norm(np.asarray(bp, float) - cam_pos, axis=1)
    lo, hi = np.percentile(db, 5), np.percentile(db, 95)
    return float(np.mean((ds >= lo) & (ds <= hi))) >= thr


# Cross-view admissibility gate of the assign rule (see assign_keyword_observations):
#   dist(c_i, c_j) < k * mean_depth + ASSIGN_EXTENT_COEF * f(diag_i, diag_j)
#   and dist(c_i, c_j) < ASSIGN_SCENE_CAP * s_scene
# f = (2 * min + max) / 3: both extents count, weighted toward the smaller one so a
# wall-sized blob does not buy itself a wall-sized radius. s_scene = scene_scale
# passed in (median camera-camera distance), else the camera span recomputed from
# cam_positions, else the 0.9-quantile of pairwise observation-centroid distances.
ASSIGN_EXTENT_COEF = 0.5
ASSIGN_SCENE_CAP = 0.5


def _extent_allowance(diag_i: float, diag_j: float) -> float:
    lo, hi = (diag_i, diag_j) if diag_i <= diag_j else (diag_j, diag_i)
    return ASSIGN_EXTENT_COEF * (2.0 * lo + hi) / 3.0


def split_unique_track(
    observations: List[Dict[str, Any]],
    cam_positions: Dict[int, np.ndarray],
    scene_scale: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Planner-unique keyword ("the race car"): there is one such object, so the
    most confident detection of each view is that object. Returns (track, rest).

    Distance cannot link these on its own: an object that MOVES between views
    sits far from itself, and a second, weaker box in each view defeats any
    one-box-per-view shortcut.
    A view's top box joins the track only within the same sanity cap the
    assign rule uses (max(scene scale, a quarter of the summed viewing depths))
    of the most confident box, so a phantom in a view that cannot see the
    object stays out of the track."""
    tops: Dict[int, Dict[str, Any]] = {}
    for o in observations:
        pts = o.get("world_points")
        if pts is None or len(pts) == 0:
            continue
        v = int(o.get("view_idx", -1))
        if v not in tops or float(o.get("score", 0.0)) > float(tops[v].get("score", 0.0)):
            tops[v] = o
    if len(tops) < 2:
        return [], observations
    anchor = max(tops.values(), key=lambda o: float(o.get("score", 0.0)))
    a_view = int(anchor.get("view_idx", -1))
    a_c = np.mean(np.asarray(anchor["world_points"], float), axis=0)
    centroids = [{"c": np.mean(np.asarray(o["world_points"], float), axis=0)} for o in tops.values()]
    s_scene = _assignment_scene_scale(scene_scale, cam_positions, centroids)
    track = [anchor]
    for v, o in tops.items():
        if o is anchor or v not in cam_positions or a_view not in cam_positions:
            continue
        c = np.mean(np.asarray(o["world_points"], float), axis=0)
        depth = np.linalg.norm(c - cam_positions[v]) + np.linalg.norm(a_c - cam_positions[a_view])
        if np.linalg.norm(c - a_c) < max(s_scene, 0.25 * depth):
            track.append(o)
    if len(track) < 2:
        return [], observations
    ids = {id(o) for o in track}
    return track, [o for o in observations if id(o) not in ids]


def assign_keyword_observations(
    observations: List[Dict[str, Any]],
    cam_positions: Dict[int, np.ndarray],
    k: float,
    scene_scale: float = 0.0,
    unique: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Multi-view ASSIGNMENT of same-keyword detections into object instances:
    observations in, list of groups (one per instance) out.

    Why not distance clustering: same-object cross-view offsets and
    different-object offsets overlap in absolute terms, so no threshold
    separates them. But the RELATIVE structure is clean -- every detection's
    true counterpart is nearer than any wrong one. An assignment uses that; a
    threshold cannot.

    Per view pair, Hungarian matching on centroid distance, so each detection
    pairs with at most ONE detection per other view -- the constraint that an
    object cannot appear twice in one image. Accepted pairs are unioned
    cheapest first, and a union that would put two detections of one view in
    the same track is skipped, so collapse is impossible across pairs too.
    A pair is admissible only if its offset is below k * mean depth + ASSIGN_EXTENT_COEF * f(diag_i, diag_j), and below
    ASSIGN_SCENE_CAP * scene scale (a wall-sized blob in a small scene must not
    reach a distant small object). k bounds the reconstruction's own
    cross-view drift; it says nothing about the scene. The extent term is what
    a partial view does to a centroid: each view sees a different part of a
    large object, and the centroid of a part lies anywhere inside the object's
    hull, so two views of ONE object can sit up to ~half its extent apart with
    zero drift. f = (2 min + max) / 3 weights the allowance toward the smaller
    extent so a wall-sized blob does not buy a wall-sized radius; two walls
    meeting at a corner (centroids ~0.7 L apart, extents L) still fail the
    0.5 L bound. The scene cap (ASSIGN_SCENE_CAP x camera span) bounds the
    gate independently of detection size.
    The gate is applied INSIDE the cost matrix: applied after solving,
    Hungarian -- obliged to match every row -- wrecks correct pairs to
    accommodate forced matches to far phantoms.

    Groups come out in spatial order (x, then z, then y of the group centre),
    NOT in detector-score order. The grounder's VLM disambiguation keeps the
    FIRST of tied candidates, so group order is a tie-break that can reach the
    answer.
    """
    dets, orphans = _assignment_detections(observations, cam_positions)
    s_scene = _assignment_scene_scale(scene_scale, cam_positions, dets)
    cap = ASSIGN_SCENE_CAP * s_scene if s_scene > 0 else 0.0
    pairs = _match_view_pairs(dets, k, cap, s_scene, unique)
    tracks = _union_pairs_one_per_view(dets, pairs)

    def _group_sort_key(idxs: List[int]) -> Tuple[float, float, float]:
        centre = np.mean([dets[i]["c"] for i in idxs], axis=0)
        return (float(centre[0]), float(centre[2]), float(centre[1]))

    out = [[dets[i]["obs"] for i in idxs]
           for idxs in sorted(tracks, key=_group_sort_key)]
    # observations with no usable geometry or no camera become their own group
    out.extend([[o] for o in observations if o.get("world_points") is None or len(o["world_points"]) == 0])
    if orphans:
        log.info(f"[fusion] {len(orphans)} observation(s) kept unmatched: no camera position for their view")
        out.extend([[o] for o in orphans])
    return out


def _assignment_detections(
    observations: List[Dict[str, Any]],
    cam_positions: Dict[int, np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Observations with points, as assignment entries (observation, view,
    centroid ``c``, viewing ``depth``, denoised extent ``diag``), plus the
    orphans: observations whose view has no camera position. Observations
    without points appear in neither list."""
    dets = []
    orphans: List[Dict[str, Any]] = []
    for o in observations:
        pts = o.get("world_points")
        if pts is None or len(pts) == 0:
            continue
        c = np.mean(np.asarray(pts, float), axis=0)
        cp = cam_positions.get(int(o.get("view_idx", -1)))
        if cp is None:
            # No camera position for this view: it cannot take part in the
            # assignment, but it must not disappear from the scene.
            orphans.append(o)
            continue
        sm = summarize_point_cloud(pts)  # denoised extent
        dets.append({"obs": o, "view": int(o.get("view_idx", -1)), "c": c,
                     "depth": float(np.linalg.norm(c - cp)),
                     "diag": float(sm["diag"]) if sm is not None else 0.0})
    return dets, orphans


def _assignment_scene_scale(
    scene_scale: float,
    cam_positions: Dict[int, np.ndarray],
    dets: List[Dict[str, Any]],
) -> float:
    """s_scene of the scene cap: ``scene_scale`` when positive, else the median
    camera-camera distance, else the 0.9-quantile of pairwise detection-centroid
    distances (3+ detections), else 0."""
    s_scene = float(scene_scale) if scene_scale and scene_scale > 0 else 0.0
    if s_scene <= 0:
        _cp = [np.asarray(v, float) for v in cam_positions.values()]
        _cd = [np.linalg.norm(_cp[i] - _cp[j]) for i in range(len(_cp)) for j in range(i + 1, len(_cp))]
        if _cd:
            s_scene = float(np.median(_cd))  # camera span
        elif len(dets) >= 3:
            _pd = [np.linalg.norm(dets[i]["c"] - dets[j]["c"])
                   for i in range(len(dets)) for j in range(i + 1, len(dets))]
            s_scene = float(np.quantile(_pd, 0.9))
    return s_scene


def _match_view_pairs(
    dets: List[Dict[str, Any]],
    k: float,
    cap: float,
    s_scene: float,
    unique: bool,
) -> List[Tuple[float, int, int]]:
    """Hungarian matching on centroid distance for every pair of views, with the
    admissibility gate inside the cost matrix. Returns the accepted pairs as
    (cost, det_a, det_b)."""
    from scipy.optimize import linear_sum_assignment

    views = sorted({d["view"] for d in dets})
    BIG = 1e6
    pairs: List[Tuple[float, int, int]] = []  # accepted (cost, det_a, det_b)
    for va, vb in combinations(views, 2):
        A = [i for i, d in enumerate(dets) if d["view"] == va]
        B = [i for i, d in enumerate(dets) if d["view"] == vb]
        C = np.array([[np.linalg.norm(dets[i]["c"] - dets[j]["c"]) for j in B] for i in A])
        G = np.array([[k * 0.5 * (dets[i]["depth"] + dets[j]["depth"])
                       + _extent_allowance(dets[i]["diag"], dets[j]["diag"]) for j in B] for i in A])
        if cap > 0:
            G = np.minimum(G, cap)
        Cg = np.where(C < G, C, BIG)
        # Planner-unique keyword ("the gripper"): when each of the two views
        # holds exactly one detection there is nothing to disambiguate, so the
        # pair is the same object even if it moved between the views. The gate
        # exists for multi-candidate views only; keep a sanity cap: the
        # scene-scale term guards room-scale scenes, the depth term guards
        # ego-video scenes where the cameras are near-static (camera span ~0
        # is the wrong yardstick there; the object's viewing depth is).
        if unique and len(A) == 1 and len(B) == 1:
            _cap = max(s_scene, 0.25 * (dets[A[0]]["depth"] + dets[B[0]]["depth"]))
            if _cap <= 0 or C[0, 0] < _cap:
                Cg = C.copy()
        for r, c_ in zip(*linear_sum_assignment(Cg)):
            if Cg[r, c_] < BIG:
                pairs.append((float(Cg[r, c_]), A[r], B[c_]))
    return pairs


def _union_pairs_one_per_view(
    dets: List[Dict[str, Any]],
    pairs: List[Tuple[float, int, int]],
) -> List[List[int]]:
    """Union accepted pairs into tracks (lists of detection indices).

    Hungarian is one-to-one per view PAIR only; chaining pairs (a1-b1, b1-c1,
    c1-a2) could still merge two detections of one view. Pairs are unioned
    cheapest first and a union whose tracks already share a view is skipped.
    Tracks come out in order of their first detection.
    """
    parent = list(range(len(dets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    track_views = [{d["view"]} for d in dets]
    for _, i, j in sorted(pairs):
        ri, rj = find(i), find(j)
        if ri != rj and not (track_views[ri] & track_views[rj]):
            parent[ri] = rj
            track_views[rj] |= track_views[ri]

    groups: Dict[int, List[int]] = {}
    for i in range(len(dets)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def rotation_matrix_from_vectors(
    vec_from: np.ndarray, vec_to: np.ndarray
) -> np.ndarray:
    """Rotation matrix that rotates vec_from to vec_to (Rodrigues)."""
    a = vec_from / (np.linalg.norm(vec_from) + 1e-12)
    b = vec_to / (np.linalg.norm(vec_to) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-10:
        return np.eye(3) if c > 0 else -np.eye(3)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-12))


# ---------------------------------------------------------------------------
# OBB fitting
# ---------------------------------------------------------------------------


def fit_bbox_to_rotation(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Fit an oriented bounding box to points given a rotation.

    Returns dict with: center, rotation, dims (w,h,d), corners (8,3), support_y.
    """
    pts = denoise_point_cloud(points)
    if len(pts) < 3:
        pts = points

    # Project into frame
    local = pts @ rotation_matrix  # (N, 3)
    lo = np.min(local, axis=0)
    hi = np.max(local, axis=0)
    center_local = (lo + hi) / 2.0
    dims = hi - lo

    center_world = center_local @ rotation_matrix.T

    # 8 corners
    offsets = (
        np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        * 0.5
    )
    corners_local = center_local[None, :] + offsets * dims[None, :]
    corners_world = corners_local @ rotation_matrix.T

    # Support Y (2nd percentile of Y in world frame)
    support_y = float(np.percentile(pts[:, 1], 2))

    return {
        "center": center_world,
        "rotation": rotation_matrix,
        "dims": dims,
        "corners": corners_world,
        "support_y": support_y,
    }


# ---------------------------------------------------------------------------
# Orientation fusion
# ---------------------------------------------------------------------------


def build_object_frame_from_front(front_world: np.ndarray) -> np.ndarray:
    """Build a right-handed frame (3x3 rotation matrix) from a front direction.

    Columns: [right, up, front]. Assumes Y-up world.
    """
    front = np.asarray(front_world, dtype=float)
    # Project to ground plane (zero Y)
    front_h = np.array([front[0], 0.0, front[2]])
    norm = np.linalg.norm(front_h)
    if norm < 1e-8:
        front_h = np.array([0.0, 0.0, -1.0])
    else:
        front_h = front_h / norm

    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(front_h, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    return np.column_stack([right, up, front_h])


def fuse_orientations(
    per_view_fronts: List[np.ndarray],
    per_view_confidences: List[float],
) -> Tuple[np.ndarray, float]:
    """Weighted average of per-view front directions.

    Returns (fused_front_world, mean_confidence).
    """
    if not per_view_fronts:
        return np.array([0.0, 0.0, -1.0]), 0.0

    total = np.zeros(3, dtype=float)
    weight_sum = 0.0
    for front, conf in zip(per_view_fronts, per_view_confidences):
        w = max(conf, 0.1)
        total += w * np.asarray(front, dtype=float)
        weight_sum += w

    fused = total / (weight_sum + 1e-12)
    norm = np.linalg.norm(fused)
    if norm < 1e-8:
        fused = np.array([0.0, 0.0, -1.0])
    else:
        fused = fused / norm

    mean_conf = weight_sum / max(len(per_view_fronts), 1)
    return fused, float(mean_conf)


# ---------------------------------------------------------------------------
# Multi-view object merging
# ---------------------------------------------------------------------------


def _scene_scale_from_cameras(cameras: Optional[List[Any]]) -> float:
    """Median pairwise camera-camera distance — scale-invariant reference.

    VGGT outputs scale-up-to-similarity (no metric ground truth), so any
    absolute meter threshold is meaningless. The median inter-camera
    distance is the most reliably-localized characteristic length in the
    scene, since cameras come from multi-view consensus rather than
    monocular depth. Returns 0.0 if fewer than two cameras with
    extractable positions are available.
    """
    if not cameras:
        return 0.0
    positions = []
    for c in cameras:
        pos = getattr(c, "position_world", None)
        if pos is None:
            pos = getattr(c, "pos", None)
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=float).ravel()
        if arr.size >= 3:
            positions.append(arr[:3])
    if len(positions) < 2:
        return 0.0
    n = len(positions)
    dists = [
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.median(dists))


def merge_objects_by_keyword(
    per_view_observations: List[Dict[str, List[Dict[str, Any]]]],
    *,
    cameras: Optional[List[Any]] = None,
    cam_positions: Optional[Dict[int, np.ndarray]] = None,
    scene_scale: Optional[float] = None,
    unique_keywords: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Merge per-view observations into 3D object instances per keyword.

    ``unique_keywords``: keywords the planner marked ``unique`` (one instance in
    the scene); for those, single-detection-per-view pairs are matched directly.

    Observations are first grouped by keyword; within each keyword group,
    same-view duplicates are dropped and the rest are matched across views
    (assign_keyword_observations).  Two detections with the same keyword that
    are far apart in 3D become separate objects.  Raises RuntimeError when no
    camera positions arrive through ``cameras`` or ``cam_positions``.

    Parameters
    ----------
    per_view_observations : list of dicts per view
        Each dict maps keyword -> list of observation dicts, each with:
            - 'world_points': (N, 3) ndarray
            - 'front_world': (3,) ndarray
            - 'orientation_confidence': float
            - 'bbox': [x1, y1, x2, y2]
            - 'mask': ndarray or None
            - 'score': float
            - 'view_idx': int

    Returns
    -------
    list of merged object dicts with:
        label, views, center_world, rotation_world, front_world, up_world,
        right_world, euler_world_deg, dims, corners_world, height, support_y,
        world_points, per_view_bboxes, per_view_masks, per_view_scores
    """
    if scene_scale is None:
        scene_scale = _scene_scale_from_cameras(cameras)
    positions = _camera_positions_by_view(cameras, cam_positions)

    # The same-view dedup and the drift gate need camera positions; without
    # them the merge raises.
    if not positions:
        raise RuntimeError(
            "Object fusion requires camera positions (for the depth-aware "
            "same-view dedup and the drift gate), but none reached "
            "merge_objects_by_keyword(). Pass cameras= or cam_positions=."
        )

    by_keyword = _dedup_same_view(per_view_observations, positions)
    fused_objects = []
    for keyword, observations in by_keyword.items():
        clusters = _keyword_instances(keyword, observations, positions, scene_scale, unique_keywords)
        # Only clusters with points become objects, so only those are counted and numbered.
        clusters = [c for c in clusters if any(
            o.get("world_points") is not None and len(o["world_points"]) > 0 for o in c)]
        for cluster_idx, cluster_observations in enumerate(clusters):
            fused = _fuse_instance(keyword, cluster_observations, cluster_idx, len(clusters))
            if fused is not None:
                fused_objects.append(fused)
    return fused_objects


def _camera_positions_by_view(
    cameras: Optional[List[Any]],
    cam_positions: Optional[Dict[int, np.ndarray]],
) -> Dict[int, np.ndarray]:
    """Camera centre per view (view_idx -> xyz), for the assignment.

    Callers may pass positions directly (the initial scene-build path has no
    Camera objects yet -- they are constructed *from* the fused objects).
    A Camera overrides a passed position of the same view.
    """
    positions: Dict[int, np.ndarray] = {
        int(k): np.asarray(v, dtype=float) for k, v in (cam_positions or {}).items()
    }
    for ci, cam in enumerate(cameras or []):
        try:
            pos = getattr(cam, "position_world", None)
            if pos is None:
                ext = np.asarray(cam.extrinsics, dtype=float)
                pos = -ext[:3, :3].T @ ext[:3, 3]
            positions[int(getattr(cam, "id", ci))] = np.asarray(pos, dtype=float)
        except Exception as e:
            log.warning(f"[fusion] camera {ci} has no usable position ({e}); its view cannot be matched")
            continue
    return positions


def _dedup_same_view(
    per_view_observations: List[Dict[str, List[Dict[str, Any]]]],
    cam_positions: Dict[int, np.ndarray],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group the observations by keyword (tagging each with ``keyword``) and
    drop same-view duplicates, highest score first within each view."""
    by_keyword: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for view_obs in per_view_observations:
        for keyword, obs_list in view_obs.items():
            for obs in obs_list:
                obs["keyword"] = keyword
            sorted_obs = sorted(obs_list, key=lambda o: o.get("view_idx", -1))
            for vid, grp in groupby(sorted_obs, key=lambda o: o.get("view_idx", -1)):
                grp_list = sorted(grp, key=lambda o: o.get("score", 0), reverse=True)
                kept = _drop_same_view_duplicates(grp_list, cam_positions.get(int(vid)))
                if _FUSION_DEBUG and len(kept) < len(grp_list):
                    log.debug(f"[assign-dedup] view {vid}: {len(grp_list)} -> {len(kept)}")
                by_keyword[keyword].extend(kept)
    return by_keyword


def _drop_same_view_duplicates(
    grp_list: List[Dict[str, Any]],
    cam_pos: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Keep each detection of one view unless it duplicates an already kept,
    higher-scoring one. ``grp_list`` is sorted by score, descending.

    Same-view observations are sliced from the SAME per-view point map, so two
    of them have exactly ZERO reconstruction drift between them: any 3D
    distance is just re-encoded mask geometry. Intersection-over-smaller (IoS)
    catches both classic overlaps and the NESTED case (one box fully inside
    another), where IoU is low but IoS ~= 1.0.
    Nested in 2D AND at the same depth => duplicate (a part).
    Nested but at a different depth => a separate object that happens to
    occlude this one from here. Keep it.
    """
    ios_thr = _SAMEVIEW_IOS_THRESHOLD
    dep_thr = _SAMEVIEW_DEPTH_INSIDE
    kept: List[Dict[str, Any]] = []
    for obs in grp_list:
        bb = obs.get("bbox")
        dup = bb is not None and any(
            kb.get("bbox") is not None
            and _ios_boxes(bb, kb["bbox"]) >= ios_thr
            and _same_depth(obs, kb, cam_pos, dep_thr)
            for kb in kept
        )
        if not dup:
            kept.append(obs)
    return kept


def _keyword_instances(
    keyword: str,
    observations: List[Dict[str, Any]],
    cam_positions: Dict[int, np.ndarray],
    scene_scale: float,
    unique_keywords: Optional[set],
) -> List[List[Dict[str, Any]]]:
    """Object instances of one keyword, each a list of observations: the
    unique track first (planner-unique keywords only), then the multi-view
    assignment of the remaining observations."""
    if _FUSION_DEBUG:
        log.debug(f"{'=' * 60}")
        log.debug(f"[merge] keyword='{keyword}': {len(observations)} observations")
        for i, obs in enumerate(observations):
            log.debug(
                f"  obs[{i}] view={obs.get('view_idx', '?')} pts={len(obs.get('world_points', []))}"
            )
    is_unique = bool(unique_keywords) and keyword in unique_keywords
    unique_track: List[Dict[str, Any]] = []
    if is_unique:
        unique_track, observations = split_unique_track(
            observations, cam_positions, scene_scale=scene_scale)
    if not observations:
        clusters = []
    else:
        clusters = assign_keyword_observations(
            observations, cam_positions,
            k=_ASSIGN_K,
            scene_scale=scene_scale,
            unique=is_unique,
        )
    if unique_track:
        clusters = [unique_track] + clusters
    return clusters


def _fuse_instance(
    keyword: str,
    cluster_observations: List[Dict[str, Any]],
    cluster_idx: int,
    num_clusters: int,
) -> Optional[Dict[str, Any]]:
    """One fused object from the observations of one instance: pooled points,
    fused front, an OBB in the frame of that front, and the per-view records.
    None when no observation has points."""
    point_sets = [
        obs["world_points"]
        for obs in cluster_observations
        if obs.get("world_points") is not None and len(obs["world_points"]) > 0
    ]
    if not point_sets:
        return None
    world_points = np.concatenate(point_sets, axis=0)

    fronts = [
        obs["front_world"]
        for obs in cluster_observations
        if obs.get("front_world") is not None
    ]
    confs = [
        obs.get("orientation_confidence", 0.5)
        for obs in cluster_observations
        if obs.get("front_world") is not None
    ]
    fused_front, mean_conf = fuse_orientations(fronts, confs)

    rotation = build_object_frame_from_front(fused_front)
    obb = fit_bbox_to_rotation(world_points, rotation)

    views = sorted(set(obs["view_idx"] for obs in cluster_observations))
    per_view = _per_view_records(cluster_observations)

    from saturn.perception.orientation.convention import (
        user_euler_from_rotation_matrix,
    )

    rotation = _ensure_proper_rotation(rotation)
    euler = user_euler_from_rotation_matrix(rotation)
    label = keyword if num_clusters == 1 else f"{keyword}_{cluster_idx}"

    return {
        "label": label,
        "views": views,
        "center_world": obb["center"],
        "rotation_world": rotation,
        # build_object_frame_from_front() puts fused_front into col2,
        # so front_world = col2 (no negation needed here).
        # The -col2 convention in load.py is for the RAW OrientAnything
        # rotation matrix, not the frame built from the fused front.
        "front_world": rotation[:, 2],
        "up_world": rotation[:, 1],
        "right_world": rotation[:, 0],
        "euler_world_deg": euler,
        "dims": obb["dims"],
        "corners_world": obb["corners"],
        "height": float(obb["dims"][1]),
        "support_y": obb["support_y"],
        "world_points": world_points,
        **per_view,
        "mean_orientation_confidence": mean_conf,
        "metadata": {
            "source_keyword": keyword,
            "cluster_index": cluster_idx,
            "num_keyword_clusters": num_clusters,
            "num_observations": len(cluster_observations),
        },
    }


def _per_view_records(cluster_observations: List[Dict[str, Any]]) -> Dict[str, Dict[int, Any]]:
    """Per-view fields of a fused object, keyed by view: the bbox, mask and
    score of the highest-scoring observation, its 3D centroid, and the front
    (world and camera frame) of the most orientation-confident observation."""
    per_view_bboxes = {}
    per_view_masks = {}
    per_view_scores = {}
    per_view_centers = {}
    per_view_fronts: Dict[int, np.ndarray] = {}
    per_view_fronts_camera: Dict[int, np.ndarray] = {}
    per_view_orientation_confidence: Dict[int, float] = {}
    for obs in cluster_observations:
        v = obs["view_idx"]
        score = float(obs.get("score", 0.0))
        # Compute per-view 3D centroid from this observation's points
        obs_pts = obs.get("world_points")
        if obs_pts is not None and len(obs_pts) > 0:
            obs_center = np.mean(obs_pts, axis=0)
            # Keep the center from the highest-scoring observation per view
            if v not in per_view_centers or score > per_view_scores.get(v, -1):
                per_view_centers[v] = obs_center
        # Keep the per-view front of the most orientation-confident observation that has one
        obs_front = obs.get("front_world")
        obs_front_cam = obs.get("front_camera")
        obs_orient_conf = float(obs.get("orientation_confidence", 0.0))
        if obs_front is not None:
            prior = per_view_orientation_confidence.get(v, -1.0)
            if obs_orient_conf > prior or v not in per_view_fronts:
                per_view_fronts[v] = np.asarray(obs_front, dtype=float)
                if obs_front_cam is not None:
                    per_view_fronts_camera[v] = np.asarray(
                        obs_front_cam, dtype=float
                    )
                per_view_orientation_confidence[v] = obs_orient_conf
        if v in per_view_scores and score < per_view_scores[v]:
            continue
        per_view_bboxes[v] = obs.get("bbox", [0, 0, 0, 0])
        if obs.get("mask") is not None:
            per_view_masks[v] = obs["mask"]
        elif v in per_view_masks:
            per_view_masks.pop(v, None)
        per_view_scores[v] = score
    return {
        "per_view_bboxes": per_view_bboxes,
        "per_view_masks": per_view_masks,
        "per_view_scores": per_view_scores,
        "per_view_centers": per_view_centers,
        "per_view_fronts": per_view_fronts,
        "per_view_fronts_camera": per_view_fronts_camera,
        "per_view_orientation_confidence": per_view_orientation_confidence,
    }


# ---------------------------------------------------------------------------
# Ground plane leveling
# ---------------------------------------------------------------------------


def fit_ground_plane_xz(
    points: np.ndarray,
    grid_cells: int = 28,
    iterations: int = 4,
    outlier_factor: float = 2.5,
) -> Dict[str, Any]:
    """Fit a ground plane from a dense scene point cloud.

    Extracts support points (lowest Y in each XZ grid cell), fits y = ax + bz + c
    iteratively with outlier rejection, and returns the leveling transform.

    Returns dict with: rotation (3,3), y_shift (float), normal (3,).
    """
    if len(points) < 10:
        return {"rotation": np.eye(3), "y_shift": 0.0, "normal": np.array([0, 1, 0])}

    # Grid-based support point extraction
    xz = points[:, [0, 2]]
    y = points[:, 1]

    x_min, z_min = xz.min(axis=0)
    x_max, z_max = xz.max(axis=0)
    x_range = max(x_max - x_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)

    support_points = []
    for xi in range(grid_cells):
        for zi in range(grid_cells):
            x_lo = x_min + (xi / grid_cells) * x_range
            x_hi = x_min + ((xi + 1) / grid_cells) * x_range
            z_lo = z_min + (zi / grid_cells) * z_range
            z_hi = z_min + ((zi + 1) / grid_cells) * z_range
            mask = (
                (xz[:, 0] >= x_lo)
                & (xz[:, 0] < x_hi)
                & (xz[:, 1] >= z_lo)
                & (xz[:, 1] < z_hi)
            )
            if mask.any():
                lowest_idx = np.argmin(y[mask])
                indices = np.where(mask)[0]
                support_points.append(points[indices[lowest_idx]])

    if len(support_points) < 3:
        return {"rotation": np.eye(3), "y_shift": 0.0, "normal": np.array([0, 1, 0])}

    support = np.array(support_points, dtype=float)

    # Iterative robust plane fitting: y = a*x + b*z + c
    mask = np.ones(len(support), dtype=bool)
    for _ in range(iterations):
        pts = support[mask]
        if len(pts) < 3:
            break
        A = np.column_stack([pts[:, 0], pts[:, 2], np.ones(len(pts))])
        result = np.linalg.lstsq(A, pts[:, 1], rcond=None)
        coeffs = result[0]  # [a, b, c]
        residuals = pts[:, 1] - (
            coeffs[0] * pts[:, 0] + coeffs[1] * pts[:, 2] + coeffs[2]
        )
        mad = np.median(np.abs(residuals))
        threshold = max(outlier_factor * 1.4826 * mad, 0.03)
        new_mask = np.abs(residuals) < threshold
        full_mask = np.zeros(len(support), dtype=bool)
        full_mask[np.where(mask)[0][new_mask]] = True
        mask = full_mask

    # Plane normal
    a, b, c = coeffs
    normal = np.array([-a, 1.0, -b], dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if np.dot(normal, [0, 1, 0]) < 0:
        normal = -normal

    # Rotation to align normal with Y-up
    rot = rotation_matrix_from_vectors(normal, np.array([0.0, 1.0, 0.0]))

    # Apply rotation to support points and compute Y-shift
    rotated_support = (rot @ support.T).T
    y_shift = -float(np.percentile(rotated_support[:, 1], 1))

    return {"rotation": rot, "y_shift": y_shift, "normal": normal}


def apply_ground_leveling(
    rotation: np.ndarray,
    y_shift: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Apply ground leveling transform to a set of positions.

    Parameters
    ----------
    rotation : (3,3) rotation matrix
    y_shift : vertical shift
    positions : (N, 3) array

    Returns (N, 3) leveled positions.
    """
    leveled = (rotation @ positions.T).T
    leveled[:, 1] += y_shift
    return leveled
