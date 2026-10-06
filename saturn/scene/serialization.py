"""Scene JSON (de)serialization: to_dict/from_dict/dump/load bodies.

Imports: may import saturn.scene.types / saturn.scene.adapters; must not import saturn.scene.scene at module level; must not import saturn.perception, saturn.vlm, saturn.serving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # annotations only; a runtime import would be circular
    from saturn.scene.scene import Scene

import numpy as np

from .types import Camera, MergedObject


def _serialize_target(target) -> Any:
    """JSON-safe representation of a face() target spec, for constraint records."""
    if hasattr(target, "id") and hasattr(target, "extrinsics"):
        return {"kind": "camera", "id": int(getattr(target, "id", -1))}
    if hasattr(target, "id") and hasattr(target, "front_world"):
        return {"kind": "object", "id": int(getattr(target, "id", -1))}
    if isinstance(target, (tuple, list)) and len(target) == 2:
        return {"kind": str(target[0]), "id": int(target[1])}
    if isinstance(target, (int, np.integer)):
        return {"kind": "object", "id": int(target)}
    arr = np.asarray(target, dtype=float).ravel()[:3]
    return {"kind": "point", "xyz": arr.tolist()}


def scene_to_dict(
    scene, *, include_metadata: bool = True, include_points: bool = False
) -> Dict[str, Any]:
    """Serialize scene geometry to a JSON-compatible dict.

    Captures all information needed to reconstruct the Scene for offline
    debugging: object positions, orientations, extents; camera extrinsics
    and intrinsics.  Does **not** include images, per-view masks, or
    callable hooks (detect_fn).

    Parameters
    ----------
    include_metadata : bool
        If True, include ground_info and per-object/camera metadata.
    include_points : bool
        If True, include subsampled point clouds per object (max 2000 pts
        each).  Default False to keep JSON small.

    Returns
    -------
    dict — JSON-serializable representation.
    """
    data: Dict[str, Any] = {
        "version": 1,
        "num_objects": len(scene.objects),
        "num_cameras": len(scene.cameras),
        "objects": [
            obj.to_dict(include_points=include_points) for obj in scene.objects
        ],
        "cameras": [cam.to_dict() for cam in scene.cameras],
    }
    if include_metadata and scene.ground_info is not None:
        data["ground_info"] = scene.ground_info
    return data


def scene_from_dict(
    cls,
    data: Dict[str, Any],
    *,
    images: Optional[List[Any]] = None,
    detect_fn: Optional[Callable] = None,
) -> "Scene":
    """Reconstruct a Scene from a dict produced by ``to_dict``.

    Parameters
    ----------
    data : dict
        Output of ``scene.to_dict()``.
    images : list, optional
        If available, attach the original images. Otherwise an empty list
        of ``None`` placeholders is used.
    detect_fn : callable, optional
        Detection callback (usually not needed for offline debugging).

    Returns
    -------
    Scene
    """
    objects = [MergedObject.from_dict(d) for d in data["objects"]]
    cameras = [Camera.from_dict(d) for d in data["cameras"]]
    if images is None:
        images = [None] * len(cameras)
    return cls(
        objects=objects,
        cameras=cameras,
        images=images,
        ground_info=data.get("ground_info"),
        detect_fn=detect_fn,
    )


def scene_dump(
    scene,
    filepath: str,
    *,
    include_metadata: bool = True,
    include_points: bool = False,
) -> None:
    """Save scene to a JSON file.

    Parameters
    ----------
    filepath : str
        Path to write (e.g. ``"scene_debug.json"``).
    include_metadata : bool
        Passed to ``to_dict``.
    include_points : bool
        Passed to ``to_dict``.  If True, subsampled point clouds are
        included per object.
    """
    import json

    class _NumpyEncoder(json.JSONEncoder):
        """Handle numpy types that slip through _arr_to_list."""

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    data = scene.to_dict(
        include_metadata=include_metadata, include_points=include_points
    )
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def scene_load(
    cls,
    filepath: str,
    *,
    images: Optional[List[Any]] = None,
    detect_fn: Optional[Callable] = None,
) -> "Scene":
    """Load a Scene from a JSON file created by ``dump``.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.
    images : list, optional
        Original images, if available.
    detect_fn : callable, optional
        Detection callback.

    Returns
    -------
    Scene
    """
    import json

    with open(filepath, "r") as f:
        data = json.load(f)
    return cls.from_dict(data, images=images, detect_fn=detect_fn)
