"""Scene runtime hooks must survive to_dict/from_dict round-trips.

The in-process scene cache (run_benchmark_async._build_scene) rebuilds every
scene via Scene.from_dict, which cannot serialize the detect/ground callbacks.
The callbacks are re-attached after the round trip, so scene.detect() works
and planner-driven grounding (MindCube/MMSI) builds populated scenes.
"""

import asyncio
import threading

import pytest

from saturn.scene.build import load_async as LA


class _StubScene:
    """Minimal stand-in: attach_runtime_hooks only sets attributes on it."""


def _loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop


def test_attach_sets_all_hooks():
    scene = _StubScene()
    loop = _loop_in_thread()
    try:
        out = LA.attach_runtime_hooks(scene, loop=loop, sam3="SAM", vlm_grounder="VLM")
        assert out is scene
        assert scene._detect_fn is not None
        assert scene.ground_fn is not None
        assert callable(scene.detect_async) and callable(scene.ground_async)
        assert scene._vlm == "VLM"
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_detect_fn_reaches_detector(monkeypatch):
    calls = {}

    async def fake_detect(scene, description, camera=None, sam3=None,
                          orientation_provider=None, unique=False):
        calls.update(description=description, camera=camera, sam3=sam3,
                     scene=scene)
        return [7]

    monkeypatch.setattr(LA, "_detect_new_objects_async", fake_detect)
    scene = _StubScene()
    loop = _loop_in_thread()
    try:
        LA.attach_runtime_hooks(scene, loop=loop, sam3="SAM3HANDLE")
        got = scene._detect_fn(scene, "red bottle", camera=1)
        assert got == [7]
        assert calls == {"description": "red bottle", "camera": 1,
                         "sam3": "SAM3HANDLE", "scene": scene}
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_from_dict_scene_raises_without_hooks():
    """Pin the failure mode: a hook-less scene must raise loudly, not return []."""
    from saturn.scene.scene import Scene

    scene = Scene.__new__(Scene)
    scene._detect_fn = None
    with pytest.raises(Exception, match="detect_fn"):
        scene.detect("anything")
