"""Planner-unique keywords.

A single detection per view is matched across views regardless of the
drift/extent gate (moving objects), bounded by the scene scale; with several
boxes per view, ``split_unique_track`` links the top box of each view.
"""
import numpy as np
from saturn.scene.fusion import assign_keyword_observations, split_unique_track


def _obs(view, centre, n=50):
    pts = np.asarray(centre, float) + np.random.RandomState(view).randn(n, 3) * 0.01
    return {"view_idx": view, "world_points": pts, "bbox": [0, 0, 10, 10], "score": 0.9}


def test_unique_singletons_match_despite_motion():
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 0.0])}
    obs = [_obs(0, [0.0, 0.0, 1.0]), _obs(1, [0.6, 0.0, 1.0])]   # moved 0.6 at depth 1: far beyond k*depth
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=2.0)) == 2
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=2.0, unique=True)) == 1


def test_unique_singletons_respect_scene_cap():
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 0.0])}
    obs = [_obs(0, [0.0, 0.0, 1.0]), _obs(1, [5.0, 0.0, 1.0])]
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=2.0, unique=True)) == 2


def test_unique_does_not_bypass_gate_with_multiple_candidates():
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 0.0])}
    obs = [_obs(0, [0.0, 0.0, 1.0]), _obs(0, [1.0, 0.0, 1.0]), _obs(1, [0.6, 0.0, 1.0])]
    groups = assign_keyword_observations(obs, cams, k=0.15, scene_scale=2.0, unique=True)
    assert len(groups) == 3   # two candidates in view 0 -> gate applies, none within k*depth


def test_unique_singletons_merge_in_ego_video_with_static_cameras():
    # First-person video: cameras barely move (scene_scale ~0.082), but the
    # object moved 0.195 at depth ~0.83.  The depth term of the cap must let
    # the singleton pair merge; without ``unique`` the drift gate still holds.
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.05, 0.0, 0.065])}
    obs = [_obs(0, [0.0, 0.0, 0.83]), _obs(1, [0.195, 0.0, 0.83])]
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=0.082, unique=True)) == 1
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=0.082)) == 2


def test_unique_singletons_room_scale_cap_still_rejects():
    # Room-scale scene: two singletons 5.0 apart at depth ~1.0 each.
    # cap = max(scene_scale=2.0, 0.5 * 1.0) = 2.0 -> still not merged.
    cams = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([5.0, 0.0, 0.0])}
    obs = [_obs(0, [0.0, 0.0, 1.0]), _obs(1, [5.0, 0.0, 1.0])]
    assert len(assign_keyword_observations(obs, cams, k=0.15, scene_scale=2.0, unique=True)) == 2


# ---------------------------------------------------------------- split_unique_track

def _scored_obs(view, center, score, n=20):
    rng = np.random.default_rng(view * 100 + int(score * 100))
    return {"view_idx": view, "score": score,
            "world_points": np.asarray(center, float) + 0.01 * rng.standard_normal((n, 3))}


TRACK_CAMS = {0: np.zeros(3), 1: np.array([0.1, 0.0, 0.0])}


def test_moving_unique_object_links_top_box_of_each_view():
    # The car moves 0.8 between frames; each view also has a weaker second box.
    obs = [_scored_obs(0, [0, 0, 3], 0.91), _scored_obs(0, [1.5, 0, 3], 0.80),
           _scored_obs(1, [0.8, 0, 3], 0.96), _scored_obs(1, [-1.2, 0, 3], 0.53)]
    track, rest = split_unique_track(obs, TRACK_CAMS, scene_scale=0.1)
    assert sorted(o["view_idx"] for o in track) == [0, 1]
    assert {o["score"] for o in track} == {0.91, 0.96}
    assert len(rest) == 2


def test_far_phantom_stays_out_of_track():
    # Top box of view 1 is 10 units away: beyond max(scene, depth/4), no link.
    obs = [_scored_obs(0, [0, 0, 3], 0.9), _scored_obs(1, [10, 0, 3], 0.5)]
    track, rest = split_unique_track(obs, TRACK_CAMS, scene_scale=0.1)
    assert track == [] and len(rest) == 2


def test_single_view_is_untouched():
    obs = [_scored_obs(0, [0, 0, 3], 0.9), _scored_obs(0, [1, 0, 3], 0.6)]
    track, rest = split_unique_track(obs, TRACK_CAMS)
    assert track == [] and rest == obs
