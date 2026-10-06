"""Perception components: SAM3 thresholds and box prompts, Orient-Anything
confidence and symmetry, roll in orientation extraction, and depth-based
orientation refinement."""
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


# ------------------------------------------- SAM3 per-request threshold -----
def _sam3_with_stub_processor(prob):
    """Real SAM3 methods + real Sam3Processor._forward_grounding; network stubbed."""
    sam3_proc = pytest.importorskip("sam3.model.sam3_image_processor")
    from saturn.perception.detection.sam3 import SAM3

    class Prompt:
        def append_boxes(self, b, l):
            pass

    class Model:
        backbone = types.SimpleNamespace(forward_text=lambda t, device: {})

        def _get_dummy_prompt(self):
            return Prompt()

        def forward_grounding(self, **kw):
            return {
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
                "pred_logits": torch.logit(torch.tensor(prob)).view(1, 1, 1),
                "pred_masks": torch.full((1, 1, 8, 8), 5.0),
                "presence_logit_dec": torch.tensor([[20.0]]),
            }

    proc = sam3_proc.Sam3Processor.__new__(sam3_proc.Sam3Processor)
    proc.model = Model()
    proc.device = "cpu"
    proc.find_stage = None
    proc.confidence_threshold = 0.5  # Sam3Processor default on a fresh replica
    proc.set_image = lambda pil: {"backbone_out": {}, "original_height": 32, "original_width": 32}
    proc.set_text_prompt = lambda state, prompt: proc._forward_grounding(
        {**state, "geometric_prompt": Prompt()}
    )
    det = SAM3.__new__(SAM3)
    det.processor = proc
    det.threshold = 0.2
    return det


def test_sam3_threshold_not_sticky_across_requests():
    det = _sam3_with_stub_processor(prob=0.35)
    img = Image.new("RGB", (32, 32))
    box = [[4, 4, 12, 12]]
    fresh = det.mask_from_boxes(img, box)[0].sum()
    det.predict_with_masks(img, text="chair", threshold=0.0)
    after_zero = det.mask_from_boxes(img, box)[0].sum()
    det.predict_with_masks(img, text="chair", threshold=0.5)
    after_half = det.mask_from_boxes(img, box)[0].sum()
    assert fresh == after_zero == after_half == 32 * 32  # real mask at self.threshold=0.2


def test_sam3_predict_with_masks_explicit_threshold_applies():
    det = _sam3_with_stub_processor(prob=0.35)
    img = Image.new("RGB", (32, 32))
    assert len(det.predict_with_masks(img, "chair", threshold=0.5)["scores"]) == 0
    assert len(det.predict_with_masks(img, "chair", threshold=0.2)["scores"]) == 1
    assert len(det.predict_with_masks(img, "chair")["scores"]) == 1  # None -> 0.2


# --------------------------- Orient-Anything confidence and symmetry -----
def _oa_logits(peaks, kappa=8.0):
    x = np.arange(360)
    d = sum(np.exp(kappa * np.cos(np.deg2rad(x - p))) for p in peaks)
    d = d / d.max()
    p = torch.full((900,), -10.0)
    p[:360] = torch.logit(torch.tensor(d * 0.98 + 0.01))
    return p


def test_oa_confidence_highest_for_unique_front():
    from saturn.perception.orientation.orient_anything import OrientAnythingEstimator

    est = OrientAnythingEstimator.__new__(OrientAnythingEstimator)
    unique = est._parse_predictions(_oa_logits([30]))
    fourfold = est._parse_predictions(_oa_logits([100, 190, 280, 10]))
    assert unique.dir_num == 1 and fourfold.dir_num == 4
    assert unique.confidence == 1.0
    assert fourfold.confidence == 0.25


def test_oa_tta_keeps_symmetry_alpha():
    from saturn.perception.orientation.orient_anything import (
        OrientAnythingEstimator,
        OrientationResult,
    )

    est = OrientAnythingEstimator.__new__(OrientAnythingEstimator)
    est._init_model = lambda: None
    est.estimate_orientations_batch = lambda imgs, rb=False: [
        OrientationResult(10.0, 0.0, 0.0, 0.25, dir_num=4) for _ in imgs
    ]
    out = est.estimate_orientations_batch_with_tta([Image.new("RGB", (40, 40))], num_crops=3)[0]
    assert out.dir_num == 4
    single = est.estimate_orientation_with_tta(Image.new("RGB", (40, 40)), num_crops=3)
    assert single.dir_num == 4


# -------------------------- in-process orientation keeps the roll -----
def test_in_process_orientation_keeps_roll():
    from saturn.perception.geometry.object_extraction import extract_objects_with_providers
    from saturn.perception.orientation.orient_anything import OrientationResult
    from saturn.perception.types import DepthEstimate, Detection2D

    raw = OrientationResult(azimuth=40.0, polar=10.0, rotation=35.0, confidence=1.0, dir_num=1)

    class LocalProvider:
        estimator = object()
        extractor = SimpleNamespace(extract_object_images=lambda img, b, masks=None: [img])

        def _estimate_raw(self, imgs):
            return [raw]

    class Depth:
        def predict(self, image):
            return DepthEstimate(depth_map=np.full((64, 64), 2.0), confidence_map=None, is_metric=True)

    det = Detection2D(bbox_xyxy=(10, 10, 50, 50), mask=np.pad(np.ones((40, 40), bool), 12)[:64, :64])
    objs, _, _ = extract_objects_with_providers(
        Image.new("RGB", (64, 64)), [det], depth_provider=Depth(), orientation_provider=LocalProvider(),
        config=SimpleNamespace(trust_model_elevation=1.0, trust_pca_horizontal=0.0),
    )
    np.testing.assert_allclose(objs[0].pose.orientation.euler_deg, [40.0, 10.0, 35.0], atol=1e-6)


# ------------------------------ depth refinement of the orientation -----
def _front_face_points():
    rng = np.random.default_rng(0)
    return np.column_stack(
        [rng.uniform(-0.4, 0.4, 2000), rng.uniform(-0.3, 0.3, 2000), 2.0 + rng.normal(0, 0.01, 2000)]
    )


@pytest.mark.parametrize("az", [0, 45, 90, 135])
def test_depth_refinement_does_not_tilt_up_toward_surface_normal(az):
    from saturn.perception.geometry.pose_fusion import refine_orientation_with_depth
    from saturn.perception.orientation.convention import rotation_matrix_from_user_euler

    R = rotation_matrix_from_user_euler([az, 0, 0])
    Rr = refine_orientation_with_depth(_front_face_points(), R, 0.85, 0.0)
    tilt = np.degrees(np.arccos(np.clip(Rr[:, 1] @ R[:, 1], -1, 1)))
    assert tilt < 1.0


def test_depth_refinement_independent_of_eigenvector_sign(monkeypatch):
    from saturn.perception.geometry.pose_fusion import refine_orientation_with_depth
    from saturn.perception.orientation.convention import rotation_matrix_from_user_euler

    pts, R = _front_face_points(), rotation_matrix_from_user_euler([45, 0, 0])
    base = refine_orientation_with_depth(pts, R, 0.85, 0.0)
    orig = np.linalg.eigh

    def flipped(c):
        w, v = orig(c)
        v = v.copy()
        v[:, 0] *= -1
        return w, v

    monkeypatch.setattr(np.linalg, "eigh", flipped)
    np.testing.assert_allclose(refine_orientation_with_depth(pts, R, 0.85, 0.0), base, atol=1e-9)


# ------------------------------------ MaskerSAM3 box-prompted masks -----
def test_masker_sam3_masks_follow_input_boxes():
    from saturn.perception.drawing import MaskerSAM3

    calls = []

    class Proc:
        confidence_threshold = 0.5

        def set_image(self, img):
            return {}

        def add_geometric_prompt(self, box, label, state):
            calls.append((box, label))
            if len(calls) == 2:
                return {"masks": [], "scores": []}  # SAM3 finds nothing -> bbox fallback
            m = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
            m[1, 0, :4, :4] = True
            return {"masks": m, "scores": torch.tensor([0.1, 0.9])}

        def set_text_prompt(self, prompt, state):
            raise AssertionError("must not fall back to a text prompt")

    m = MaskerSAM3.__new__(MaskerSAM3)
    m.processor, m.use_local_sam3, m.threshold = Proc(), True, 0.3
    masks = m.mask_image(Image.new("RGB", (32, 32)), [[0, 0, 16, 16], [12, 12, 30, 30]])
    assert masks.shape == (2, 32, 32)
    assert [c[1] for c in calls] == [True, True]
    np.testing.assert_allclose(calls[0][0], [0.25, 0.25, 0.5, 0.5])
    assert masks[0].sum() == 16  # argmax-score mask
    assert masks[1].sum() == 18 * 18  # rectangle of the second box
