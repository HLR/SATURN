"""
Multi-view 3D scene understanding for SATURN.

Public API
----------
load_scene_async : Build a Scene from one or more images (detection, reconstruction, orientation, fusion).
Scene      : Central scene object with relations, frames, distances.
Camera     : Camera model with clone/rotate.
MergedObject : Merged cross-view object.
DirectionValue : Spherical direction returned by direction queries.
CameraHeading  : Camera heading with cardinal discretization.
FrameNamespace : Frame-scoped relations returned by scene.frame().
VGGTReconstructor : VGGT-1B 3D reconstruction.

Direction utilities (shared constants and functions):
    classify_direction, resolve_north, translate_label,
    is_relative_label, CARDINAL_TO_ANGLE, RELATIVE_TO_CARDINAL, etc.

The names are re-exported lazily (PEP 562): importing ``saturn.scene`` does
not load the perception / reconstruction layers until a name that needs
them is actually accessed.
"""

import importlib

__all__ = [
    "load_scene_async",
    "Scene",
    "Camera",
    "CameraHeading",
    "DirectionValue",
    "MergedObject",
    "FrameNamespace",
    "Frame",
    "VGGTReconstructor",
    # Direction utilities
    "classify_direction",
    "resolve_north",
    "translate_label",
    "is_relative_label",
    "CARDINAL_TO_ANGLE",
    "CARDINAL_LABELS_4",
    "CARDINAL_LABELS_8",
    "RELATIVE_LABELS_4",
    "RELATIVE_LABELS_8",
    "RELATIVE_TO_CARDINAL",
    "CARDINAL_TO_RELATIVE",
]

# name -> defining module
_LAZY = {
    "load_scene_async": "saturn.scene.build.load_async",
    "Scene": "saturn.scene.scene",
    "Camera": "saturn.scene.types",
    "CameraHeading": "saturn.scene.types",
    "DirectionValue": "saturn.scene.types",
    "MergedObject": "saturn.scene.types",
    "FrameNamespace": "saturn.predicates.frame",
    "Frame": "saturn.predicates.frame",
    "VGGTReconstructor": "saturn.perception.reconstruction.vggt",
    "classify_direction": "saturn.scene.direction_utils",
    "resolve_north": "saturn.scene.direction_utils",
    "translate_label": "saturn.scene.direction_utils",
    "is_relative_label": "saturn.scene.direction_utils",
    "CARDINAL_TO_ANGLE": "saturn.scene.direction_utils",
    "CARDINAL_LABELS_4": "saturn.scene.direction_utils",
    "CARDINAL_LABELS_8": "saturn.scene.direction_utils",
    "RELATIVE_LABELS_4": "saturn.scene.direction_utils",
    "RELATIVE_LABELS_8": "saturn.scene.direction_utils",
    "RELATIVE_TO_CARDINAL": "saturn.scene.direction_utils",
    "CARDINAL_TO_RELATIVE": "saturn.scene.direction_utils",
}


def __getattr__(name):
    try:
        module_name = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
