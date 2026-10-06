"""Grounding / verification robustness (ObjectGrounder + pre_detect_objects).

1. An unparseable verify response is an explicit "unparsed" outcome (score
   0.0 + diagnostic, after one retry), never a neutral 0.5 that would clear
   the 0.4 verify threshold.
2. A region with no member carries a planner-retry diagnostic, and first
   falls back to grounding the region phrase as one object.
3. A grounded region is also ONE scene entity (label = phrase, merged
   geometry, unknown orientation) that the grounding result points at.
"""

import re

import numpy as np
import pytest
from PIL import Image

from saturn.pipeline.ground import pre_detect_objects
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject
from saturn.vlm.grounding import UNPARSED_FEEDBACK, ObjectGrounder

NAN = float("nan")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _obj(label, center, bboxes, half=0.25, views=None, points=True, mask_shape=None):
    center = np.asarray(center, dtype=float)
    offs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners = center + half * offs
    masks = {}
    if mask_shape is not None:
        for v, b in bboxes.items():
            m = np.zeros(mask_shape, dtype=bool)
            x1, y1, x2, y2 = [int(c) for c in b]
            m[y1:y2, x1:x2] = True
            masks[v] = m
    return MergedObject(
        id=0,
        label=label,
        views=list(views if views is not None else bboxes),
        center_world=center,
        rotation_world=np.eye(3),
        front_world=np.array([0.0, 0.0, 1.0]),
        up_world=np.array([0.0, 1.0, 0.0]),
        right_world=np.array([1.0, 0.0, 0.0]),
        euler_world_deg=np.zeros(3),
        dims=np.full(3, 2 * half),
        corners_world=corners,
        height=2 * half,
        support_y=float(center[1] - half),
        world_points=corners.copy() if points else None,
        per_view_bboxes={v: list(b) for v, b in bboxes.items()},
        per_view_masks=masks,
        per_view_scores={v: 0.9 for v in bboxes},
        per_view_orientation_confidence={v: 0.8 for v in bboxes},
    )


class _FakeScene:
    """Scene stand-in: ``detect(query)`` appends the objects wired in ``table``."""

    def __init__(self, table=None, n_views=2):
        self.table = dict(table or {})
        self.objects = []
        self.images = [Image.new("RGB", (64, 64), (i * 40, 0, 0)) for i in range(n_views)]
        self.detect_queries = []
        self.ground_fn = None

    @property
    def objects_count(self):
        return len(self.objects)

    def detect(self, desc, camera=None, **kw):
        self.detect_queries.append(desc)
        new = []
        for spec in self.table.get(desc, []):
            o = _obj(**spec)
            o.id = len(self.objects)
            self.objects.append(o)
            new.append(o.id)
        return new

    def nms_objects(self, **_kw):
        return 0

    def _invalidate_caches(self):
        pass


class _VerifyVLM:
    """Answers verify prompts with a per-phrase P(Yes); refine probes get nothing."""

    _P = {"yes": 1.0, "no": 0.0}

    def __init__(self, verdicts=None, default="yes", raw=None):
        self.verdicts = dict(verdicts or {})
        self.default = default
        self.raw = list(raw or [])  # scripted P(Yes) values / exceptions, consumed first
        self.verify_calls = 0

    def _query(self, images, prompt, max_new_tokens=256):
        return ""  # refine probe: unparseable -> no revision

    def _score_simple(self, images, prompt):
        self.verify_calls += 1
        if self.raw:
            r = self.raw.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        phrase = re.search(r'I need to find "([^"]+)"', prompt).group(1)
        return self._P[self.verdicts.get(phrase, self.default)]


def _grounder(scene, vlm, **kw):
    kw.setdefault("max_refine", 0)
    return ObjectGrounder(vlm=vlm, scene=scene, question="q?", verbose=False, **kw)


# ---------------------------------------------------------------------------
# 1. Unparsed verification
# ---------------------------------------------------------------------------


def _one_candidate(vlm):
    scene = _FakeScene()
    scene.objects = [_obj("cup", [0, 0, 0], {0: [1, 1, 20, 20]})]
    return _grounder(scene, vlm)


def test_unparsed_verify_retries_once_then_scores_zero():
    vlm = _VerifyVLM(raw=[NAN, NAN, 1.0])
    score, fb = _one_candidate(vlm)._score_candidate(0, "the cup", "cup", "t")
    assert score == 0.0
    assert fb.startswith(UNPARSED_FEEDBACK)
    assert vlm.verify_calls == 2  # one retry, then give up


def test_unparsed_verify_recovers_on_retry():
    vlm = _VerifyVLM(raw=[NAN, 1.0])
    score, _ = _one_candidate(vlm)._score_candidate(0, "the cup", "cup", "t")
    assert score == 1.0
    assert vlm.verify_calls == 2


def test_verify_error_is_retried_then_unparsed():
    vlm = _VerifyVLM(raw=[RuntimeError("timeout"), RuntimeError("timeout")])
    score, fb = _one_candidate(vlm)._score_candidate(0, "the cup", "cup", "t")
    assert (score, fb.startswith(UNPARSED_FEEDBACK)) == (0.0, True)


def test_score_path_error_is_retried_then_unparsed():
    class ScoreVLM:
        calls = 0

        def _score_simple(self, images, prompt):
            ScoreVLM.calls += 1
            raise RuntimeError("boom")

    score, fb = _one_candidate(ScoreVLM())._score_candidate(0, "the cup", "cup", "t")
    assert score == 0.0 and fb.startswith(UNPARSED_FEEDBACK)
    assert ScoreVLM.calls == 2


def test_unparsed_verification_does_not_verify_the_object():
    scene = _FakeScene({"cup": [dict(label="x", center=[0, 0, 0], bboxes={0: [1, 1, 9, 9]})]})
    vlm = _VerifyVLM(raw=[NAN] * 4)
    r = _grounder(scene, vlm).ground_all(
        [{"phrase": "the cup", "description": "cup", "cam_id": 0}]
    )[0]
    assert not r.verified
    assert r.last_diagnostic == "no_verify"
    assert r.verify_feedback.startswith(UNPARSED_FEEDBACK)


def test_view_blind_candidate_without_bbox_is_not_verified():
    scene = _FakeScene()
    o = _obj("cup", [0, 0, 0], {0: [1, 1, 9, 9]})
    o.per_view_bboxes = {}
    scene.objects = [o]
    score, _ = _grounder(scene, _VerifyVLM())._score_candidate(0, "cup", "cup", "t")
    assert score == 0.0


# ---------------------------------------------------------------------------
# 2. Region diagnostics + whole-phrase fallback
# ---------------------------------------------------------------------------

REGION = {"phrase": "the kitchen area", "description": "stove, sink",
          "cam_id": 0, "is_region": True}


def test_region_with_nothing_found_sets_no_candidates():
    scene = _FakeScene()
    r = _grounder(scene, _VerifyVLM()).ground_all([REGION])[0]
    assert r.obj_indices == []
    assert r.last_diagnostic == "no_candidates"
    assert r.n_candidates == 0
    # the region phrase itself was tried as one object before giving up
    assert "kitchen area" in scene.detect_queries


def test_region_with_unverified_candidates_sets_no_verify():
    scene = _FakeScene({"stove": [dict(label="x", center=[0, 0, 0], bboxes={0: [1, 1, 9, 9]})]})
    r = _grounder(scene, _VerifyVLM(default="no")).ground_all([REGION])[0]
    assert r.last_diagnostic == "no_verify"
    assert r.n_candidates == 1


def test_region_falls_back_to_phrase_as_one_object():
    scene = _FakeScene({"kitchen area": [
        dict(label="x", center=[0, 0, 0], bboxes={0: [1, 1, 30, 30]})]})
    r = _grounder(scene, _VerifyVLM()).ground_all([REGION])[0]
    assert r.last_diagnostic == ""
    assert r.verified
    assert len(r.obj_indices) == 1
    assert scene.objects[r.obj_indices[0]].label == "kitchen area"
    assert r.region_members == []  # no merged entity: the object IS the region


def test_failed_region_reaches_planner_retry_and_salvage():
    scene = _FakeScene()
    scene._vlm = _VerifyVLM()

    class Planner:
        seen = []

        def refine_groundings(self, question, images, prior_groundings, failed_groundings):
            Planner.seen.extend(failed_groundings)
            # the planner renames nothing useful; salvage must still run
            return []

    salvage_queries = []
    orig_detect = scene.detect

    def detect(desc, camera=None, **kw):
        salvage_queries.append((desc, camera))
        return orig_detect(desc, camera)

    scene.detect = detect
    out = {}
    pre_detect_objects(scene, {"object_groundings": [dict(REGION, cam_id=[0])]},
                       "t", out, question="q?", planner=Planner(), images=scene.images)
    assert [f["phrase"] for f in Planner.seen] == ["the kitchen area"]
    assert Planner.seen[0]["diagnostic"] == "no_candidates"
    # salvage grounds the region PHRASE (not "stove, sink") with an int cam_id
    assert salvage_queries[-1] == ("kitchen area", 0)


def test_planner_retry_pairs_results_with_their_own_grounding():
    """A skipped grounding ("already in scene") must not shift the pairing."""
    scene = _FakeScene()
    scene.objects = [_obj("chair", [0, 0, 0], {0: [1, 1, 9, 9]})]
    scene._vlm = _VerifyVLM()

    class Planner:
        seen = []

        def refine_groundings(self, question, images, prior_groundings, failed_groundings):
            Planner.seen.extend(failed_groundings)
            return []

    groundings = [
        {"phrase": "the chair", "description": "wooden chair", "cam_id": 0},
        {"phrase": "the lamp", "description": "tall floor lamp", "cam_id": 1},
    ]
    pre_detect_objects(scene, {"object_groundings": groundings}, "t", {},
                       question="q?", planner=Planner(), images=scene.images)
    assert len(Planner.seen) == 1
    assert Planner.seen[0]["phrase"] == "the lamp"
    assert Planner.seen[0]["description"] == "tall floor lamp"


# ---------------------------------------------------------------------------
# 3. Region as one scene entity
# ---------------------------------------------------------------------------

WALKWAY = {"phrase": "the walkway between table and trash bins",
           "description": "table, trash bins", "cam_id": 0, "is_region": True}


def _walkway_scene():
    return _FakeScene({
        "table": [dict(label="x", center=[0, 0, 2], bboxes={0: [10, 10, 30, 30]},
                       mask_shape=(64, 64))],
        "trash bins": [dict(label="x", center=[2, 0, 4],
                            bboxes={0: [40, 20, 60, 50], 1: [5, 5, 15, 15]},
                            mask_shape=(64, 64))],
    })


def test_region_becomes_one_merged_entity():
    scene = _walkway_scene()
    r = _grounder(scene, _VerifyVLM()).ground_all([WALKWAY])[0]

    assert len(scene.objects) == 3  # both members kept + the region entity
    assert r.region_members == [0, 1]
    assert r.obj_indices == [2]
    assert r.verified and r.last_diagnostic == ""

    ent = scene.objects[2]
    table, bins = scene.objects[0], scene.objects[1]
    assert ent.label == "walkway between table and trash bins"
    np.testing.assert_allclose(ent.center_world, [1.0, 0.0, 3.0])
    assert ent.per_view_bboxes == {0: [10, 10, 60, 50], 1: [5.0, 5.0, 15.0, 15.0]}
    assert ent.views == [0, 1]
    assert ent.per_view_masks[0].sum() == (
        table.per_view_masks[0] | bins.per_view_masks[0]).sum()
    assert len(ent.world_points) == len(table.world_points) + len(bins.world_points)
    np.testing.assert_allclose(ent.rotation_world, np.eye(3))
    assert ent.orientation_confidence == 0.0
    assert ent.metadata["is_region"] is True
    assert ent.metadata["member_indices"] == [0, 1]
    # the box spans both members
    np.testing.assert_allclose(ent.corners_world.min(axis=0), [-0.25, -0.25, 1.75])
    np.testing.assert_allclose(ent.corners_world.max(axis=0), [2.25, 0.25, 4.25])


def test_region_entity_is_reused_not_regrounded():
    scene = _walkway_scene()
    g = _grounder(scene, _VerifyVLM())
    g.ground_all([WALKWAY])
    assert g.ground_all([WALKWAY]) == []  # "already in scene"
    assert len(scene.objects) == 3


def _camera():
    ext = np.eye(4)
    return Camera(id=0, entity_id=0, intrinsics=np.eye(3), extrinsics=ext,
                  image_size=(64, 64))


def test_region_entity_behaves_like_an_object_in_a_real_scene():
    table = _obj("table", [0, 0, 2], {0: [10, 10, 30, 30]})
    bins = _obj("trash bins", [2, 0, 4], {0: [40, 20, 60, 50]})
    far = _obj("sofa", [10, 0, 10], {0: [0, 0, 5, 5]})
    scene = Scene(objects=[table, bins, far], cameras=[_camera()],
                  images=[Image.new("RGB", (64, 64))])
    g = _grounder(scene, _VerifyVLM())
    r = g._ground_region(
        "the walkway between table and trash bins", "table, trash bins", 0, "t",
    )
    g._add_region_entity(r, "t")

    assert r.obj_indices == [3]
    n = len(scene.objects) + len(scene.cameras)
    assert tuple(scene.closeness.tensor.shape) == (n, n)
    close = scene.closeness.tensor
    # the walkway is closer to its members than to the far sofa
    assert close[3, 0] > close[3, 2] and close[3, 1] > close[3, 2]
    assert tuple(scene.left.tensor.shape) == (n, n)
    assert scene.objects[3].anchor_index == 3
    scene._frame(at=scene.objects[3])  # usable as a frame anchor


def test_nms_keeps_region_entity_and_repoints_indices():
    """Real Scene NMS: a duplicate before the region shifts indices; the
    region (whose union box ~= its dominant member's box) must survive and
    every result / member index must follow its object."""
    dup_a = _obj("chair", [5, 0, 5], {0: [0, 0, 10, 10]})
    dup_b = _obj("chair", [5, 0, 5], {0: [0, 0, 10, 10]})
    dup_b.per_view_scores = {0: 0.1}
    stove = _obj("stove", [0, 0, 2], {0: [10, 10, 50, 50]})
    scene = Scene(objects=[dup_a, dup_b, stove], cameras=[_camera()],
                  images=[Image.new("RGB", (64, 64))])
    g = _grounder(scene, _VerifyVLM())
    r = g._ground_region("the stove area", "stove", 0, "t")
    g._session_results.append(r)
    g._add_region_entity(r, "t")
    region = scene.objects[3]
    assert r.obj_indices == [3] and r.region_members == [2]

    removed = g._nms("t", "test")
    assert removed == 1  # only the duplicate chair
    assert region in scene.objects and region.label == "stove area"
    assert scene.objects[r.obj_indices[0]] is region
    assert scene.objects[r.region_members[0]] is stove
    assert region.metadata["member_indices"] == [scene.objects.index(stove)]
