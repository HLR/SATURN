"""Scene loading: fusion labelling, the unique-pair cap, the reconstruction
view count and the SAM3 mask fallback of scene.ground()."""
import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from saturn.scene.build import load_async
from saturn.scene.fusion import assign_keyword_observations, merge_objects_by_keyword

rng = np.random.default_rng(0)
CAMS = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}


def _obs(center, view, bbox, points=True):
    pts = np.asarray(center, float) + rng.normal(0, 0.01, (50, 3)) if points else np.zeros((0, 3))
    return {"world_points": pts, "bbox": list(bbox), "score": 0.9, "view_idx": view,
            "front_world": np.array([0, 0, 1.0]), "front_camera": None,
            "orientation_confidence": 0.0, "mask": None}


# ---------------------------------------------------------------- fusion labels
def test_pointless_detection_is_not_counted_in_labels():
    """One real chair plus a point-less detection is labelled 'chair' with
    num_keyword_clusters=1: a detection without points is not a cluster."""
    c = np.array([0.0, 0.0, 1.0])
    views = [{"chair": [_obs(c, 0, (0, 0, 10, 10)), _obs(c, 0, (50, 50, 60, 60), points=False)]},
             {"chair": [_obs(c + [0.01, 0, 0], 1, (0, 0, 10, 10))]}]
    objs = merge_objects_by_keyword(views, cam_positions=CAMS)
    assert [o["label"] for o in objs] == ["chair"]
    assert objs[0]["metadata"]["num_keyword_clusters"] == 1
    assert objs[0]["metadata"]["cluster_index"] == 0


def test_pointless_detection_is_not_counted_with_two_real_objects():
    """Two real chairs plus a point-less detection report num_keyword_clusters=2."""
    a, b = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 0.0, 2.0])
    views = [{"chair": [_obs(a, 0, (0, 0, 10, 10)), _obs(b, 0, (30, 0, 40, 10)),
                        _obs(a, 0, (50, 50, 60, 60), points=False)]},
             {"chair": [_obs(a, 1, (0, 0, 10, 10)), _obs(b, 1, (30, 0, 40, 10))]}]
    objs = merge_objects_by_keyword(views, cam_positions=CAMS)
    assert sorted(o["label"] for o in objs) == ["chair_0", "chair_1"]
    assert {o["metadata"]["num_keyword_clusters"] for o in objs} == {2}


# ---------------------------------------------------------------- unique-pair cap
def test_unique_pair_cap_uses_scene_scale_fallback():
    """With scene_scale None or 0, the scene term of the unique-pair cap uses
    the camera span, as the rest of the function does."""
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([2.0, 0.0, 0.0])}   # camera span 2.0
    # One unique object that moved 1.8 between the views: beyond the drift gate and
    # beyond a quarter of the summed depths (~1.5), within the camera span (2.0).
    obs = [_obs([0.0, 0.0, 3.0], 0, (0, 0, 10, 10)), _obs([1.8, 0.0, 3.0], 1, (0, 0, 10, 10))]
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=2.0, unique=True)) == 1
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=None, unique=True)) == 1
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=0.0, unique=True)) == 1


# ---------------------------------------------------------------- reconstruction view count
def _vggt_result(num_views, h=4, w=4):
    from saturn.perception.reconstruction.vggt import VGGTResult

    ext = np.tile(np.eye(4), (num_views, 1, 1))
    ext[:, 0, 3] = np.arange(num_views, dtype=float)
    K = np.tile(np.array([[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1.0]]), (num_views, 1, 1))
    return VGGTResult(
        world_points=np.ones((num_views, h, w, 3)), world_points_conf=np.ones((num_views, h, w)),
        depth=np.ones((num_views, h, w)), depth_conf=np.ones((num_views, h, w)),
        extrinsics=ext, intrinsics=K, transform_info=[{} for _ in range(num_views)],
        num_views=num_views,
    )


def _load(num_images, num_views):
    recon = SimpleNamespace(reconstruct=lambda images: _vggt_result(num_views))
    images = [Image.new("RGB", (4, 4)) for _ in range(num_images)]
    return asyncio.run(load_async.load_scene_async(images, vggt_reconstructor=recon, level_ground=False))


@pytest.mark.parametrize("num_images,num_views", [(3, 2), (2, 3)])
def test_reconstruction_view_count_mismatch_raises(num_images, num_views):
    """A reconstruction must return one view per image; any other count is a
    ValueError rather than a scene built from a mismatched camera set."""
    with pytest.raises(ValueError, match="one view per image"):
        _load(num_images, num_views)


def test_reconstruction_view_count_match_builds_scene():
    scene = _load(2, 2)
    assert len(scene.cameras) == 2


# ---------------------------------------------------------------- SAM3 mask fallback
def test_ground_logs_sam3_mask_failure_once_and_falls_back(monkeypatch, caplog):
    """A failing SAM3 predict_masks is logged once; each view keeps its box, without a mask."""
    captured = {}

    async def _fake_extract(images, preds, geoms, per_view_detections, orientation_provider=None):
        captured["dets"] = per_view_detections
        return [{} for _ in images]

    monkeypatch.setattr(load_async, "_extract_per_view_observations_async", _fake_extract)
    monkeypatch.setattr(load_async, "_merge_new_observations", lambda obs, scene, unique_keyword=None: [])
    monkeypatch.setattr(load_async, "_append_merged_objects", lambda scene, merged, source="": [])

    def _boom(image, boxes):
        raise RuntimeError("sam3 replica down")

    scene = SimpleNamespace(images=[Image.new("RGB", (20, 20)) for _ in range(2)],
                            _depth_predictions=[None, None], _world_geometries=[None, None])
    vlm = SimpleNamespace(ground=lambda image, description: (2, 2, 10, 10))
    sam3 = SimpleNamespace(predict_masks=_boom)
    caplog.set_level(logging.WARNING, logger="saturn")
    asyncio.run(load_async._ground_new_objects_async(scene, "the mug", vlm=vlm, sam3=sam3))

    warnings = [r for r in caplog.records if "predict_masks" in r.getMessage()]
    assert len(warnings) == 1
    assert "sam3 replica down" in warnings[0].getMessage() and "2 view(s)" in warnings[0].getMessage()
    # the fallback is kept: each view still contributes its box, without a mask
    assert [d["the mug"][0]["mask"] for d in captured["dets"]] == [None, None]


def test_split_unique_track_scene_scale_none_uses_the_fallback():
    # the unique-track cap shares the assignment cap's scene-scale fallback, so None does not raise
    import numpy as np
    from saturn.scene.fusion import split_unique_track
    rng = np.random.default_rng(0)
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([2.0, 0.0, 0.0])}
    obs = [{"view_idx": v, "score": 0.9, "world_points": rng.normal([1.0 + 0.5 * v, 0.0, 3.0], 0.05, (50, 3))}
           for v in (0, 1)]
    track_none, _ = split_unique_track(obs, cams, scene_scale=None)
    track_span, _ = split_unique_track(obs, cams, scene_scale=2.0)   # median camera distance
    assert len(track_none) == len(track_span) == 2
