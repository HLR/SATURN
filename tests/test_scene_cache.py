import threading
import time

import pytest

from saturn.scene.build.cache import SceneCache, scene_key


def test_key_covers_images_keywords_boxes_and_flags():
    def k(**o):
        return scene_key(**{"image_paths": ["a.png", "b.png"], "keywords": ["object"],
                            "bboxes": None, "level_ground": True, **o})
    assert k() == k()
    assert k(image_paths=["a.png", "c.png"]) != k()
    assert k(keywords=["object", "car"]) != k() and k(keywords=["car", "object"]) == k(keywords=["object", "car"])
    assert k(bboxes={0: [[0, 0, 1, 1]]}) != k(bboxes={0: [[0, 0, 2, 2]]})
    assert k(level_ground=False) != k()


def test_concurrent_same_key_builds_once():
    c = SceneCache(max_entries=4)
    n = {"b": 0}

    def build():
        n["b"] += 1
        time.sleep(0.2)
        return {"scene": n["b"]}
    out = []
    ts = [threading.Thread(target=lambda: out.append(c.get_or_build("k", build))) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert n["b"] == 1 and all(o == {"scene": 1} for o in out) and c.hits == 2 and c.builds == 1


def test_distinct_keys_and_lru_eviction():
    c = SceneCache(max_entries=2)
    made = []
    for k in ("a", "b", "c"):
        c.get_or_build(k, lambda k=k: made.append(k) or k)
    assert made == ["a", "b", "c"] and "a" not in c._entries and "c" in c._entries
    c.get_or_build("b", lambda: "REBUILT")          # still cached
    assert c.get_or_build("b", lambda: "REBUILT") == "b"


def _fixture_scene(y_up: bool):
    """Scene from the anchor fixtures. Production cameras are canonical Y-up
    (image-down = -Y world); the raw fixture camera has image-down = +Y."""
    import importlib.util
    import os
    import numpy as np
    spec = importlib.util.spec_from_file_location("_fx", os.path.join(os.path.dirname(__file__), "test_anchor.py"))
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    from saturn.scene.scene import Scene
    objs = [fx._make_object(0, (0, 0, -2)), fx._make_object(1, (1, 0, -2)), fx._make_object(2, (-1, 0.5, -3))]
    cam = fx._make_camera((0, 0, 0), (0, 0, -1), cam_id=0)
    if y_up:
        ext = np.asarray(cam.extrinsics, float).copy()   # keep det, image-down -> -Y
        ext[1, :3] *= -1
        ext[0, :3] *= -1
        cam.extrinsics = ext
    return Scene(objects=objs, cameras=[cam], images=[None])


def _A(x):
    import numpy as np
    t = getattr(x, "tensor", x)
    return np.asarray(t.detach().cpu() if hasattr(t, "detach") else t, dtype=float)


@pytest.mark.parametrize("y_up", [True, False])
def test_scene_dict_round_trip_is_faithful(y_up):
    """What the scene cache relies on: a scene survives to_dict/from_dict unchanged."""
    import numpy as np
    from saturn.scene.scene import Scene
    from saturn.scene.adapters import _is_y_down_extrinsics
    s1 = _fixture_scene(y_up=y_up)
    assert _is_y_down_extrinsics(s1.cameras[0].extrinsics) != y_up
    s2 = Scene.from_dict(s1.to_dict(include_points=True), images=[None])
    for name in ("left", "right", "front", "behind", "above", "below"):
        np.testing.assert_allclose(_A(getattr(s2, name)), _A(getattr(s1, name)), atol=1e-9, err_msg=name)
    for k in ("front", "left", "above"):
        np.testing.assert_allclose(s2._anchor_predicates[k], s1._anchor_predicates[k], atol=1e-9, equal_nan=True)
    assert [o.anchor_index for o in s2.objects] == [0, 1, 2] and s2.cameras[0].anchor_index == 3
