"""IO, trace-loading and Plotly-scene helpers shared by the debug-report renderer."""

import base64
import html as html_lib
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from saturn.settings import env



def _json_default(obj):
    """Coerce numpy / set / pathlike objects so json.dumps doesn't crash.

    Report dicts carry numpy scalars (int64 object indices, float64 scores)
    and arrays (grounded bboxes).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "__fspath__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════


def esc(text: str) -> str:
    """HTML-escape text."""
    return html_lib.escape(str(text))


def b64_image(path: str) -> str:
    """Read an image file and return a data-URI string."""
    ext = Path(path).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        ext.lstrip("."), "jpeg"
    )
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{data}"


def find_scene_dirs(result_json: str) -> List[Path]:
    """Return candidate directories for scene JSON files."""
    rp = Path(result_json)
    candidates = [rp.parent / "scenes"]
    parent_exp = rp.parent.parent
    try:
        for d in sorted(parent_exp.iterdir()):
            if d.is_dir() and (d / "scenes").exists():
                candidates.append(d / "scenes")
    except OSError:
        pass  # missing or unreadable experiment dir: own run's scenes only
    return candidates


_MINDCUBE_IMAGE_INDEX: Optional[Dict[str, List[Path]]] = None


def default_mmsi_image_dir() -> str:
    """Where the MMSI loader extracts the view images: $MMSI_DATA_ROOT/images, else <repo>/data/mmsi/images."""
    root = Path(env("MMSI_DATA_ROOT") or Path(__file__).resolve().parent.parent.parent / "data" / "mmsi")
    return str(root / "images")


def _build_mindcube_image_index() -> Dict[str, List[Path]]:
    """Map every MindCube sample id → resolved image Paths from the JSONLs.

    Reads the directory the MindCube loader reads (saturn.datasets.get_dataset):
    $MINDCUBE_DATA_ROOT, else <repo>/data/mindcube, where setup/30_datasets.sh downloads it.
    """
    global _MINDCUBE_IMAGE_INDEX
    if _MINDCUBE_IMAGE_INDEX is not None:
        return _MINDCUBE_IMAGE_INDEX
    index: Dict[str, List[Path]] = {}
    data_root = Path(env("MINDCUBE_DATA_ROOT") or Path(__file__).resolve().parent.parent.parent / "data" / "mindcube")
    for jl in (data_root / "raw").glob("MindCube*.jsonl"):
        try:
            with open(jl) as f:
                for line in f:
                    rec = json.loads(line)
                    sid = rec.get("id")
                    paths = rec.get("images") or []
                    if sid and paths and sid not in index:
                        index[sid] = [data_root / p for p in paths]
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    _MINDCUBE_IMAGE_INDEX = index
    return index


def resolve_image_paths_for_sample(item_id: str, image_dir: str) -> List[Path]:
    """Resolve image files for a sample, supporting MindCube + MMSI conventions.

    MindCube records list per-sample images in their JSONL (variable filenames
    like ``front_126.png`` under per-group subdirectories), so a directory
    glob can't find them — we look up the id in the dataset index.

    Otherwise uses the ``{image_dir}/{item_id}_*.{jpg,png}`` glob, for MMSI
    and other datasets that name files by sample id.
    """
    if item_id.startswith(("among_", "around_", "rotation_", "translation_")):
        idx = _build_mindcube_image_index()
        paths = idx.get(item_id)
        if paths:
            return [p for p in paths if p.exists()]
    img_dir = Path(image_dir)
    return sorted(img_dir.glob(f"{item_id}_*.jpg")) + sorted(
        img_dir.glob(f"{item_id}_*.png")
    )


def load_scene_for_sample(
    item_id: str, scene_dir: Optional[str], result_json: str
) -> Tuple[Optional[Dict], Optional[str]]:
    """Load the scene JSON dict for a sample. Returns (scene_data, scene_path), or
    (None, None) when no scene file is found.

    An explicit *scene_dir* is the only place searched: falling through to a
    sibling run would show another model's scene as this run's. Without one,
    the run's own ``scenes/`` dir is tried first, then sibling runs'
    (auto-detection; the report flags a scene from another run).
    """
    if scene_dir:
        candidates = [Path(scene_dir)]
    else:
        candidates = find_scene_dirs(result_json)

    for cd in candidates:
        p = cd / f"{item_id}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f), str(p)
    return None, None


def load_orientation_trace_for_sample(
    item_id: str, result_json: str
) -> Optional[Dict]:
    """Search `experiments/debug*/<category>/` dirs for `<id>_orientation_trace.json`.

    Returns the parsed trace dict (contents of the "trace" key), or None.
    The result-JSON path is expected to live under
    ``experiments/<category>/<model-dir>/<file>.json``; we extract <category>
    as the directory two levels above the result file.
    """
    rp = Path(result_json).resolve()
    # Find the "experiments" directory and infer category
    category = None
    exp_root = None
    parents = list(rp.parents)
    for i, anc in enumerate(parents):
        if anc.name == "experiments":
            exp_root = anc
            if i > 0:
                category = parents[i - 1].name
            break
    if exp_root is None:
        # Fallback: search for an "experiments" sibling
        for anc in parents:
            if (anc / "experiments").exists():
                exp_root = anc / "experiments"
                break
    if exp_root is None or not exp_root.exists():
        return None
    if category is None:
        # Guess: result file lives directly under experiments/<cat>/<model>/
        try:
            rel = rp.relative_to(exp_root)
            if len(rel.parts) >= 2:
                category = rel.parts[0]
        except ValueError:
            pass
    if not category:
        return None
    # Look in debug*, debug directories under experiments
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("debug"):
            continue
        cand = d / category / f"{item_id}_orientation_trace.json"
        if cand.exists():
            try:
                with open(cand) as f:
                    payload = json.load(f)
                return payload.get("trace", payload)
            except Exception:
                continue
    return None


def _collect_raw_azimuths_by_label(
    orient_trace: Dict,
) -> Dict[str, List[Dict]]:
    """Group raw OA detections by per-view-observation keyword.

    The trace stores raw detections in order across all OA calls; per-view
    observations are stored in the same global order. We align them 1:1 by
    position and index by keyword. Returns
    {keyword: [{view, az, polar, roll, conf}, ...]}.
    """
    if not orient_trace:
        return {}
    raw = orient_trace.get("raw_orientation_per_view") or []
    pv = orient_trace.get("per_view_object_orientation") or []
    # Flatten raw detections preserving call order
    flat_raw: List[Dict] = []
    for call in raw:
        for det in call.get("detections") or []:
            flat_raw.append(det)
    out: Dict[str, List[Dict]] = {}
    for i, entry in enumerate(pv):
        if i >= len(flat_raw):
            break
        raw_det = flat_raw[i]
        kw = entry.get("keyword") or ""
        out.setdefault(kw, []).append(
            {
                "view": entry.get("view_index"),
                "azimuth": raw_det.get("raw_azimuth_backend"),
                "polar": raw_det.get("raw_polar_backend"),
                "roll": raw_det.get("raw_rotation_backend"),
                "confidence": raw_det.get("raw_confidence"),
                "alpha": raw_det.get("raw_alpha"),
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════
# Plotly 3D scene builder (generates the JS inline)
# ═══════════════════════════════════════════════════════════════════════

OBJ_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]
CAM_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]


def _w2d(pts):
    # World → display reorder for the Plotly (z-up) scene.
    #
    # Saved scene data is in the canonical world frame: X-right, Y-UP,
    # Z-forward (``canonicalize_y_up`` in saturn/scene/adapters.py normalizes
    # the reconstruction to Y-up before the scene is built).
    #
    # Plotly's scene z-axis is the vertical (up) axis, so we map world Y → z.
    # A bare (X, Z, Y) mapping is a reflection (det = -1) and would mirror
    # left↔right; to keep a proper (det = +1) rotation we also negate depth:
    #   (X_right, Y_up, Z_fwd) → (X_right, -Z_fwd, Y_up)
    # This is a 180° rotation about the left-right axis: left/right is
    # preserved, up/down and front/back flip together, nothing is mirrored.
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    return np.column_stack([pts[:, 0], -pts[:, 2], pts[:, 1]])


def _w2d_dir(v):
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[2], v[1]])


# Corner index pairs of the 12 edges of an 8-corner box.
_BOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
# Edges of a camera frustum: the 4 image-plane corners, then each corner to the center (point 4).
_FRUSTUM_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
# (scene key, color, length, line width, cone arrowhead): front is thicker, longer and has a head.
_AXIS_SPECS = [
    ("front_world", "#1d4ed8", 0.5, 7, True),   # blue
    ("right_world", "#dc2626", 0.25, 3, False), # red
    ("up_world",    "#16a34a", 0.25, 3, False), # green
]


def _line_segments(pts, pairs):
    """x, y, z lists drawing each (a, b) pair as a separate segment (None breaks the line)."""
    xs, ys, zs = [], [], []
    for a, b in pairs:
        xs.extend([float(pts[a, 0]), float(pts[b, 0]), None])
        ys.extend([float(pts[a, 1]), float(pts[b, 1]), None])
        zs.extend([float(pts[a, 2]), float(pts[b, 2]), None])
    return xs, ys, zs


def _object_center_trace(c, col, label, i) -> Dict:
    return {
        "type": "scatter3d",
        "mode": "markers+text",
        "x": [c[0]],
        "y": [c[1]],
        "z": [c[2]],
        "marker": {"size": 6, "color": col},
        "text": [f"{label} [{i}]"],
        "textposition": "top center",
        "name": f"{label} [{i}]",
    }


def _object_points_traces(obj_d: Dict, col, label, i) -> List[Dict]:
    """The object's point cloud, when the scene dump has one."""
    if not ("world_points" in obj_d and obj_d["world_points"]):
        return []
    pts = np.array(obj_d["world_points"])
    if not (pts.ndim == 2 and pts.shape[0] > 0):
        return []
    pts_d = _w2d(pts)
    return [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": pts_d[:, 0].tolist(),
            "y": pts_d[:, 1].tolist(),
            "z": pts_d[:, 2].tolist(),
            "marker": {"size": 1.5, "color": col, "opacity": 0.4},
            "name": f"{label} [{i}] pts",
            "showlegend": False,
            "hoverinfo": "skip",
        }
    ]


def _object_axis_traces(obj_d: Dict, c) -> List[Dict]:
    """Front / right / up axis lines from the object center; the front one ends in a cone."""
    traces = []
    origin = np.array(c)
    for axis_key, axis_col, length, width, with_head in _AXIS_SPECS:
        if axis_key in obj_d and obj_d[axis_key] is not None:
            raw = np.asarray(obj_d[axis_key], dtype=float)
            n = float(np.linalg.norm(raw))
            if n < 1e-9:
                continue
            d = _w2d_dir(raw / n)
            end = (origin + d * length).tolist()
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "x": [c[0], end[0]],
                    "y": [c[1], end[1]],
                    "z": [c[2], end[2]],
                    "line": {"color": axis_col, "width": width},
                    "showlegend": False,
                    "hoverinfo": "skip",
                }
            )
            if with_head:
                traces.append(
                    {
                        "type": "cone",
                        "x": [end[0]],
                        "y": [end[1]],
                        "z": [end[2]],
                        "u": [float(d[0])],
                        "v": [float(d[1])],
                        "w": [float(d[2])],
                        "sizemode": "absolute",
                        "sizeref": 0.12,
                        "anchor": "tip",
                        "colorscale": [[0, axis_col], [1, axis_col]],
                        "showscale": False,
                        "showlegend": False,
                        "hoverinfo": "skip",
                    }
                )
    return traces


def _object_box_traces(obj_d: Dict, col) -> List[Dict]:
    """The object's 3D bounding box edges; skipped when the corners are missing or all zero."""
    if "corners_world" not in obj_d:
        return []
    corners = np.array(obj_d["corners_world"])
    if not (corners.shape == (8, 3) and not np.allclose(corners, 0)):
        return []
    xs, ys, zs = _line_segments(_w2d(corners), _BOX_EDGES)
    return [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "z": zs,
            "line": {"color": col, "width": 3},
            "showlegend": False,
            "hoverinfo": "skip",
        }
    ]


def _object_traces(i: int, obj_d: Dict) -> List[Dict]:
    c = _w2d(obj_d["center_world"])[0].tolist()
    col = OBJ_COLORS[i % len(OBJ_COLORS)]
    label = obj_d.get("label", f"obj_{i}")
    return (
        [_object_center_trace(c, col, label, i)]
        + _object_points_traces(obj_d, col, label, i)
        + _object_axis_traces(obj_d, c)
        + _object_box_traces(obj_d, col)
    )


def _camera_frustum_points(cam_d: Dict):
    """Display-frame frustum points (4 image-plane corners, then the center) and the center."""
    K = np.array(cam_d["intrinsics"])
    E = np.array(cam_d["extrinsics"])
    imsz = tuple(cam_d["image_size"])

    E4 = np.eye(4)
    E4[:3, :4] = E if E.shape == (3, 4) else E[:3, :4]
    c2w = np.linalg.inv(E4)
    center = c2w[:3, 3]
    center_d = _w2d(center)[0].tolist()

    w, h = imsz
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depth = 0.4
    if fx < 1e-3:
        # No usable intrinsics: draw a smaller frustum with a nominal focal length.
        depth = 0.2
        fx = fy = max(w, h, 1)
        cx, cy = w / 2, h / 2
    cam_corners = []
    for u, v in [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]:
        cam_corners.append([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])
    cam_corners = np.array(cam_corners)
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    world_corners = cam_corners @ R_c2w.T + t_c2w
    pts = np.vstack([world_corners, center[None, :]])
    return _w2d(pts), center_d


def _camera_traces(i: int, cam_d: Dict) -> List[Dict]:
    """The camera's frustum lines and a labelled marker at its center."""
    cc = CAM_COLORS[i % len(CAM_COLORS)]
    pts_d, center_d = _camera_frustum_points(cam_d)
    xs, ys, zs = _line_segments(pts_d, _FRUSTUM_EDGES)
    return [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "z": zs,
            "line": {"color": cc, "width": 4},
            "name": f"Camera {i}",
        },
        {
            "type": "scatter3d",
            "mode": "markers+text",
            "x": [center_d[0]],
            "y": [center_d[1]],
            "z": [center_d[2]],
            "marker": {"size": 5, "color": cc, "symbol": "diamond"},
            "text": [f"Cam {i}"],
            "textposition": "top center",
            "showlegend": False,
        },
    ]


def _room_center_trace(room_center) -> Dict:
    rc = _w2d(room_center)[0].tolist()
    return {
        "type": "scatter3d",
        "mode": "markers+text",
        "x": [rc[0]],
        "y": [rc[1]],
        "z": [rc[2]],
        "marker": {"size": 8, "color": "black", "symbol": "x"},
        "text": ["room_center"],
        "textposition": "bottom center",
        "name": "room_center",
    }


def build_plotly_traces_json(scene_data: Dict, room_center=None) -> str:
    """Build a JSON array of Plotly trace dicts for the 3D scene."""
    traces = []
    for i, obj_d in enumerate(scene_data.get("objects", [])):
        traces.extend(_object_traces(i, obj_d))
    for i, cam_d in enumerate(scene_data.get("cameras", [])):
        traces.extend(_camera_traces(i, cam_d))
    if room_center is not None:
        traces.append(_room_center_trace(room_center))
    return json.dumps(traces, default=_json_default)


# ═══════════════════════════════════════════════════════════════════════
# RF block reconstruction
# ═══════════════════════════════════════════════════════════════════════



def _extract_user_code(full_code: str) -> str:
    """Extract just the user-written body from the CODE_TEMPLATE wrapper."""
    lines = full_code.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if "objects_count" in line and "scene.objects_count" in line:
            body_start = i + 1
            break
    user_code = "\n".join(lines[body_start:])
    return textwrap.dedent(user_code).strip()
