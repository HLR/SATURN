"""The helper bound in every generated program: ``camera(n)``.

``camera(n)`` takes the question's own 1-based image number (no N vs N-1
arithmetic) and returns the camera's entity index, which also carries the
camera itself. Entities are named with the method's own ``score()``;
viewpoints are built with ``scene.frame(...)``.
"""

from __future__ import annotations

from typing import Callable, Dict


class CameraRef(int):
    """camera(n): the camera's entity index (use it to index a predicate,
    ``pred[camera(2)]``) that also carries the camera itself, so it builds
    anchors directly: ``scene.frame(position=camera(2).position,
    orientation=camera(2).orientation)``."""

    def __new__(cls, index: int, cam):
        obj = super().__new__(cls, index)
        obj.cam = cam
        return obj

    def __getattr__(self, name):
        if name == "cam":
            raise AttributeError(name)
        return getattr(self.cam, name)


def make_formula_helpers(score: Callable, scene) -> Dict[str, Callable]:
    """Build the helpers bound to this program's ``score`` and ``scene``."""
    K = scene.objects_count

    def camera(n: int) -> int:
        """Entity index of the question's 'image n' / 'view n' (1-based)."""
        n = int(n)
        if not 1 <= n <= len(scene.cameras):
            raise IndexError(f"camera({n}): the scene has images 1..{len(scene.cameras)}")
        return CameraRef(K + n - 1, scene.cameras[n - 1])

    return {"camera": camera}
