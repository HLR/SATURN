"""Multi-view assignment fusion: behaviour pinned by construction.

Each case is a tiny synthetic scene with known cameras; the expected answers
follow from the rule's definition, not from a benchmark.
"""
import numpy as np
import pytest

from saturn.scene import fusion
from saturn.scene.fusion import merge_objects_by_keyword

rng = np.random.default_rng(0)


def _blob(center, r=0.02, n=60):
    return np.asarray(center, float) + rng.normal(0, r, (n, 3))


def _obs(center, view, bbox, score=0.9, **kw):
    return {"world_points": _blob(center), "bbox": list(bbox), "score": score,
            "view_idx": view, "front_world": np.array([0, 0, 1.0]),
            "up_world": np.array([0, 1.0, 0]), "right_world": np.array([1.0, 0, 0]),
            "front_camera": None, "orientation_confidence": 0.0, "mask": None, **kw}


CAMS = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0])}


def _run(views, monkeypatch, k=0.15):
    monkeypatch.setattr(fusion, "_ASSIGN_K", k)
    return merge_objects_by_keyword(views, cam_positions=CAMS)


def test_same_object_three_views_fuses(monkeypatch):
    # one object at depth ~1, seen from 3 cameras with 2% drift
    c = np.array([0.0, 0.0, 1.0])
    views = [{"object": [_obs(c + [0.01, 0, 0], 0, (0, 0, 10, 10))]},
             {"object": [_obs(c + [0, 0.01, 0], 1, (0, 0, 10, 10))]},
             {"object": [_obs(c - [0.01, 0, 0], 2, (0, 0, 10, 10))]}]
    assert len(_run(views, monkeypatch)) == 1


def test_adjacent_objects_resolve_by_assignment_not_threshold(monkeypatch):
    # two objects 0.08 apart -- INSIDE the drift band -- but mutual-nearest
    # structure is intact, so the assignment must still get both right.
    a, b = np.array([0.0, 0.0, 1.0]), np.array([0.08, 0.0, 1.0])
    views = [{"object": [_obs(a, 0, (0, 0, 10, 10)), _obs(b, 0, (20, 0, 30, 10))]},
             {"object": [_obs(a + [0.01, 0, 0], 1, (0, 0, 10, 10)),
                         _obs(b + [0.01, 0, 0], 1, (20, 0, 30, 10))]}]
    out = _run(views, monkeypatch)
    assert len(out) == 2


def test_collapse_is_impossible(monkeypatch):
    # 8 distinct objects per view, all within one huge k: a cluster can still
    # hold at most one detection per view, so 8 objects must survive.
    pts = [np.array([i * 0.05, 0.0, 1.0]) for i in range(8)]
    views = [{"object": [_obs(p, v, (i * 20, 0, i * 20 + 10, 10)) for i, p in enumerate(pts)]}
             for v in range(3)]
    out = _run(views, monkeypatch, k=5.0)
    assert len(out) == 8


def test_far_apart_stays_split(monkeypatch):
    views = [{"object": [_obs([0, 0, 1.0], 0, (0, 0, 10, 10))]},
             {"object": [_obs([0.5, 0, 1.0], 1, (0, 0, 10, 10))]}]
    assert len(_run(views, monkeypatch)) == 2


def test_part_box_is_deduped_but_occluder_is_kept(monkeypatch):
    # view 0: a big object at depth 1, plus (a) a PART of it -- nested box, same
    # depth -- and (b) a small object nested in its box but at depth 0.5.
    big = _obs([0, 0, 1.0], 0, (0, 0, 100, 100), score=0.9)
    part = _obs([0.01, 0.01, 1.0], 0, (40, 40, 60, 60), score=0.5)
    occluder = _obs([0, 0, 0.5], 0, (40, 40, 60, 60), score=0.5)
    out = _run([{"object": [big, part, occluder]}], monkeypatch)
    assert len(out) == 2, "part merged into host; occluder kept as its own object"


def test_missing_cameras_is_loud():
    with pytest.raises(RuntimeError, match="fusion requires camera positions"):
        merge_objects_by_keyword([{"object": [_obs([0, 0, 1.0], 0, (0, 0, 10, 10))]}])


def test_adjacent_objects_do_not_collapse_at_default_k():
    # many adjacent objects with no gap between them: one detection per view
    # per track keeps all 8 apart at the shipped k.
    pts = [np.array([i * 0.05, 0.0, 1.0]) for i in range(8)]
    views = [{"object": [_obs(p, v, (i * 20, 0, i * 20 + 10, 10)) for i, p in enumerate(pts)]}
             for v in range(3)]
    assert len(merge_objects_by_keyword(views, cam_positions=CAMS)) == 8


def test_assignment_keeps_observations_whose_camera_is_missing():
    """An observation whose view has no camera position stays in the scene as its own group."""
    from saturn.scene.fusion import assign_keyword_observations
    rng = np.random.default_rng(0)
    def obs(view, c): return {"world_points": np.asarray(c) + rng.normal(0, 0.01, (30, 3)), "bbox": [0, 0, 5, 5], "score": 0.9, "view_idx": view}
    observations = [obs(0, [0, 0, 1.0]), obs(1, [0.01, 0, 1.0]), obs(2, [5, 5, 5])]   # view 2 has no camera
    groups = assign_keyword_observations(observations, {0: np.zeros(3), 1: np.array([1.0, 0, 0])}, k=0.15)
    flat = [o for g in groups for o in g]
    assert len(flat) == 3 and any(o["view_idx"] == 2 for o in flat)
    assert any(len(g) == 2 for g in groups)   # views 0 and 1 still fuse
