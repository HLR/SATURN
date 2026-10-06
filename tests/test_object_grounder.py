"""ObjectGrounder: NMS after each grounding phase, cam_id normalisation,
verification on the unique and non-unique paths, and salvage of the
top-scoring box."""
import types

import pytest
from PIL import Image

from saturn.vlm.grounding import GroundingResult, ObjectGrounder


class _NmsScene:
    """Each NMS call removes one object."""

    def __init__(self):
        self.objects = []
        self.images = []
        self.nms_calls = 0

    @property
    def objects_count(self):
        return len(self.objects)

    def nms_objects(self, **kw):
        self.nms_calls += 1
        self.objects.pop()
        return 1


def test_phase2_nms_runs_after_phase1_nms_removed_objects():
    g = ObjectGrounder(vlm=None, scene=_NmsScene(), question="q", verbose=False)

    def fake_process(gr, item_id):
        ph = gr["phrase"]
        n_new = 2 if ph == "chair" else 1  # phase 1: 2 duplicates; phase 2: 1 object
        for _ in range(n_new):
            g.scene.objects.append(types.SimpleNamespace(label=ph, metadata={}))
        return GroundingResult(phrase=ph, description=ph, cam_id=None,
                               is_region=gr.get("is_region", False))

    g._process_one = fake_process
    g.ground_all([{"phrase": "chair"}, {"phrase": "desk area", "is_region": True}])
    assert g.scene.nms_calls == 2


class _CamScene:
    objects = []
    images = [None] * 4
    objects_count = 0

    def __init__(self):
        self.seen = []

    def detect(self, d, camera=None, **kw):
        self.seen.append(camera)
        if camera is not None and not (isinstance(camera, int) and 0 <= camera < 4):
            raise ValueError(f"camera={camera} out of range")
        return []


@pytest.mark.parametrize("raw,expected", [
    ([4], None), (4, None), (-1, None), ([], None), ("x", None),
    ("2", 2), ([3, 1], 3), (0, 0),
])
def test_process_one_normalizes_cam_id(raw, expected):
    scene = _CamScene()
    g = ObjectGrounder(vlm=types.SimpleNamespace(), scene=scene, question="q",
                       verbose=False, max_refine=0)
    g._process_one({"phrase": "the sofa", "description": "grey sofa", "cam_id": raw}, "x")
    assert scene.seen and all(c == expected for c in scene.seen)


class _DetectScene:
    """``detect`` appends one candidate per entry of ``det_scores`` (in that order)."""

    def __init__(self, det_scores):
        self.images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
        self.objects = []
        self.det_scores = det_scores
        self.detect_calls = []

    @property
    def objects_count(self):
        return len(self.objects)

    def detect(self, desc, camera=None, unique=False):
        self.detect_calls.append((desc, camera, unique))
        view = 0 if camera is None else camera
        start = len(self.objects)
        for k, s in enumerate(self.det_scores):
            self.objects.append(types.SimpleNamespace(
                id=start + k, label=desc, views=[view],
                per_view_bboxes={view: [2 + k, 2, 12 + k, 12]},
                per_view_scores={view: s},
            ))
        return list(range(start, start + len(self.det_scores)))


class _CountingVLM:
    """Verify scores by call order within each pass over the candidates."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def _score_simple(self, images, prompt):
        s = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return s


@pytest.mark.parametrize("cam_id", [None, 1])
@pytest.mark.parametrize("flags", [{"unique": False}, {"multi_view": True}])
def test_failed_non_unique_verification_asks_the_vlm_once_per_candidate(cam_id, flags):
    scene = _DetectScene(det_scores=[0.9, 0.8, 0.7])
    vlm = _CountingVLM(scores=[0.10, 0.30, 0.20])
    g = ObjectGrounder(vlm=vlm, scene=scene, question="q", max_refine=0, verbose=False)
    result = g._ground_object("the mug", "white mug", cam_id, "item", **flags)
    assert vlm.calls == 3
    assert not result.verified
    assert result.last_diagnostic == "no_verify"
    assert result.n_candidates == 3
    assert result.verify_score == pytest.approx(0.30)
    assert scene.objects == []


def test_non_unique_verification_keeps_passing_candidates_without_rescoring():
    scene = _DetectScene(det_scores=[0.9, 0.8])
    vlm = _CountingVLM(scores=[0.9, 0.1])
    g = ObjectGrounder(vlm=vlm, scene=scene, question="q", max_refine=0, verbose=False)
    result = g._ground_object("the mug", "white mug", None, "item", unique=False)
    assert vlm.calls == 2
    assert result.verified and result.obj_indices == [0]


def test_unique_path_verifies_once_per_candidate():
    scene = _DetectScene(det_scores=[0.9, 0.8])
    vlm = _CountingVLM(scores=[0.2, 0.7])
    g = ObjectGrounder(vlm=vlm, scene=scene, question="q", max_refine=0, verbose=False)
    result = g._ground_object("the mug", "white mug", None, "item")
    assert vlm.calls == 2
    assert result.verified and result.verify_score == pytest.approx(0.7)
    assert len(scene.objects) == 1 and scene.objects[0].per_view_scores == {0: 0.8}


def test_salvage_keeps_the_highest_scoring_detection_not_the_first():
    # Fusion orders candidates spatially; the most confident box is the middle one.
    scene = _DetectScene(det_scores=[0.4, 0.95, 0.6])
    g = ObjectGrounder(vlm=None, scene=scene, question="q", verbose=False)
    keep = g.salvage("the lamp", "floor lamp", None, "item")
    assert keep == 0
    assert len(scene.objects) == 1
    assert scene.objects[0].per_view_scores == {0: 0.95}
    assert scene.objects[0].label == "lamp"


def test_salvage_keeps_fusion_order_on_tied_scores():
    scene = _DetectScene(det_scores=[0.7, 0.7])
    g = ObjectGrounder(vlm=None, scene=scene, question="q", verbose=False)
    assert g.salvage("the lamp", "floor lamp", None, "item") == 0
    assert scene.objects[0].per_view_bboxes == {0: [2, 2, 12, 12]}
