"""The planner's ``unique`` flag reaches fusion on BOTH detection paths.

When SAM3 finds nothing for a phrase ("left gripper"), the grounder falls back to
VLM box grounding. That fallback must carry ``unique`` like ``detect()`` does:
fusion links a unique object's box in every view into one track
(fusion.split_unique_track). Without it, multi-view assignment keeps a MOVING
object's boxes as separate objects, the grounder keeps one, and a program that
reads ``per_view_centers[v]`` for the other view fails (MMSI motion questions).
"""
from types import SimpleNamespace

from saturn.scene.scene import Scene
from saturn.vlm.grounding import ObjectGrounder


def test_scene_ground_forwards_unique_to_ground_fn():
    calls = []
    fake = SimpleNamespace(_vlm=object(), images=[None, None],
                           ground_fn=lambda scene, desc, vlm, camera=None, unique=False: calls.append(unique) or [0])
    assert Scene.ground(fake, "gripper", unique=True) == [0]
    assert Scene.ground(fake, "gripper") == [0]
    assert calls == [True, False]


class _SceneWithoutSam3Hits:
    """detect() finds nothing, so the grounder must use the VLM-grounding fallback."""

    def __init__(self):
        self.ground_calls = []
        self.ground_fn = True

    def detect(self, desc, camera=None, unique=False):
        return []

    def ground(self, desc, camera=None, unique=False):
        self.ground_calls.append((desc, camera, unique))
        return [0]


def test_grounder_fallback_keeps_unique():
    scene = _SceneWithoutSam3Hits()
    g = ObjectGrounder(vlm=None, scene=scene, question="q?", verbose=False)
    assert g._detect("gripper", None, "id", unique=True) == [0]
    assert g._detect("bowl", 1, "id", unique=False) == [0]
    assert scene.ground_calls == [("gripper", None, True), ("bowl", 1, False)]
