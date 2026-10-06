"""Scene build: world frames and caching.

- Back-projecting a per-view depth prediction lands every view's pixel on
  the same world point.
- Ground leveling moves per_view_centers and per_view_fronts with the scene.
- Post-build detect()/ground() fuse points and camera centres in the same
  (leveled) frame.
- Every question gets its own copy of a cached scene, including one whose
  camera looks Y-down.
- The fusion debug log passes no print-only kwargs (flush=) to the logger.
"""
import argparse
import asyncio
import threading
import types
from types import SimpleNamespace

import numpy as np

from saturn.scene.types import Camera


# ---------------------------------------------------------------- back-projection
def _cam(center, forward, right):
    forward = forward / np.linalg.norm(forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ center
    return E


def _depth_pred(E, P):
    x, y, z = E[:3, :3] @ P + E[:3, 3]
    K = np.eye(3)
    K[0, 2] = -x / z
    K[1, 2] = -y / z
    return SimpleNamespace(depth=np.array([[z]]), confidence=np.ones((1, 1)),
                           intrinsics=K, extrinsics=E[:3, :].copy(), is_metric=False)


def test_world_geometry_from_depth_backprojects_every_view_to_one_point():
    from saturn.scene.build.load import _world_geometry_from_depth

    P = np.array([0.0, 1.0, 3.0])
    E0 = _cam(np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    th = np.deg2rad(80)
    f1 = np.array([0, np.sin(th), np.cos(th)])
    E1 = _cam(P - 2.0 * f1, f1, np.array([1.0, 0, 0]))

    pts = [_world_geometry_from_depth(_depth_pred(E, P))["world_points"][0, 0] for E in (E0, E1)]
    np.testing.assert_allclose(pts[0], P, atol=1e-6)
    np.testing.assert_allclose(pts[1], P, atol=1e-6)


# ---------------------------------------------------------------- leveling
def _obs(v, c, rng, front=(0, 0, 1.0), conf=0.9):
    return {"world_points": np.asarray(c, float) + rng.normal(0, 0.01, (60, 3)),
            "front_world": np.asarray(front, float), "orientation_confidence": conf,
            "bbox": [0, 0, 10, 10], "mask": None, "score": 0.9, "view_idx": v}


def _rot_x(deg):
    t = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def test_leveling_moves_per_view_centers_and_fronts():
    from saturn.scene.build.load import _level_merged_dicts
    from saturn.scene.fusion import merge_objects_by_keyword

    rng = np.random.default_rng(0)
    m = merge_objects_by_keyword(
        [{"box": [_obs(0, (0, -1.2, 2), rng)]}, {"box": [_obs(1, (0.02, -1.2, 2), rng)]}],
        cam_positions={0: np.zeros(3), 1: np.array([1.0, 0, 0])})
    rot = _rot_x(5)
    raw_c0 = np.asarray(m[0]["per_view_centers"][0]).copy()
    raw_f0 = np.asarray(m[0]["per_view_fronts"][0]).copy()
    _level_merged_dicts(m, {"rotation": rot, "y_shift": 1.2})
    obj = m[0]
    np.testing.assert_allclose(obj["per_view_centers"][0], rot @ raw_c0 + [0, 1.2, 0])
    np.testing.assert_allclose(obj["per_view_fronts"][0], rot @ raw_f0)
    # both views agree with the leveled centre / front
    for v in (0, 1):
        assert np.linalg.norm(obj["per_view_centers"][v] - obj["center_world"]) < 0.05
    np.testing.assert_allclose(obj["per_view_fronts"][0], obj["front_world"], atol=1e-6)


def _ext_at(center):
    E = np.eye(4)
    E[:3, 3] = -np.asarray(center, float)
    return E


def test_detect_after_leveling_fuses_in_one_frame():
    """Two chairs 0.45 apart seen once per view stay 2 objects on the
    post-build detect() path of a leveled scene."""
    from saturn.scene.build.load import _merge_new_observations

    rot, y_shift = np.eye(3), 1.2
    cams = []
    for i, c in enumerate([(0, 0, 0), (1, 0, 0)]):     # leveled like load_async
        E = _ext_at(c)
        R_new = E[:3, :3] @ rot.T
        E[:3, 3] = E[:3, 3] - R_new @ np.array([0, y_shift, 0])
        E[:3, :3] = R_new
        cams.append(Camera(id=i, entity_id=i, intrinsics=np.eye(3), extrinsics=E, image_size=(10, 10)))
    scene = SimpleNamespace(cameras=cams, ground_info={"rotation": rot, "y_shift": y_shift})
    rng = np.random.default_rng(0)
    obs = [{"chair": [_obs(0, (0.5, -1.2, 2.0), rng, conf=0.0)]},
           {"chair": [_obs(1, (0.95, -1.2, 2.0), rng, conf=0.0)]}]
    merged = _merge_new_observations(obs, scene)
    assert len(merged) == 2
    # results are leveled
    assert all(abs(m["center_world"][1]) < 0.1 for m in merged)
    for m in merged:
        for c in m["per_view_centers"].values():
            assert abs(c[1]) < 0.1


# ---------------------------------------------------------------- scene cache
def test_scene_cache_does_not_store_uncacheable_values():
    from saturn.scene.build.cache import SceneCache

    c = SceneCache()
    built = []
    def make():
        built.append(1)
        return object()
    a = c.get_or_build("k", make, cacheable=lambda v: False)
    b = c.get_or_build("k", make, cacheable=lambda v: False)
    assert a is not b and len(built) == 2


def test_cached_scene_is_not_shared_across_questions(monkeypatch):
    import saturn.scene as sc_pkg
    from saturn.pipeline.build_scene import build_scene
    from saturn.scene.build.cache import SceneCache
    from saturn.scene.scene import Scene

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    n = {"builds": 0}

    async def fake_load(**kw):
        n["builds"] += 1
        cam = Camera(id=0, entity_id=0, intrinsics=np.eye(3),
                     extrinsics=np.eye(4),   # image-down points to world +Y
                     image_size=(480, 640))
        return Scene(objects=[], cameras=[cam], images=[None])

    monkeypatch.setattr(sc_pkg, "load_scene_async", fake_load)

    class Img:
        _sapy_path = "/data/scene_1/view0.png"

    models = types.SimpleNamespace(sam3=None, orientation_provider=None,
                                   vggt_reconstructor=None, vl_model=None)
    cache = SceneCache()
    args = argparse.Namespace()
    try:
        s1 = build_scene([Img()], models, loop=loop, args=args, scene_cache=cache, keywords=["chair"])
        s1.objects.append("question-1 object")
        s2 = build_scene([Img()], models, loop=loop, args=args, scene_cache=cache, keywords=["chair"])
    finally:
        loop.call_soon_threadsafe(loop.stop)
    assert s1 is not s2 and s2.objects == [] and n["builds"] == 1


# ---------------------------------------------------------------- logging
def test_fusion_debug_log_has_no_print_kwargs():
    import inspect
    import saturn.scene.build.load_async as la

    assert "flush=True" not in inspect.getsource(la)
