"""Scene-scale normalisation of the built-in directional predicates.

Paper (appendix_predicate.tex, "Scene-scale normalization"):

    s_scene = quantile_0.9({ ||x_p - x_q|| : p < q })
    delta_bar = delta / s_scene
    S_r[i, j] = sigmoid((h_r(delta_bar) - m_dir) / tau_dir),  m_dir = 0, tau_dir = 1/14

``compute_frame_relations`` divides the signed projections by ``scene_scale``
before the sigmoid; ``FrameNamespace._ensure_directional`` threads
``Scene._compute_scene_scale()`` into it.
"""
import numpy as np
import pytest

from saturn.predicates.relations import compute_frame_relations
from saturn.predicates.scoring import steep_sigmoid_signed
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject

_KEYS = ("left", "right", "front", "behind", "above", "below")


def _np(t):
    """ProbabilisticTensor / torch tensor / ndarray -> float ndarray."""
    t = getattr(t, "tensor", t)
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=float)


def _make_camera(position, forward, cam_id=0):
    position = np.asarray(position, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0)
    ext = np.eye(4)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = -R_w2c @ position
    return Camera(id=cam_id, entity_id=cam_id, intrinsics=np.eye(3),
                  extrinsics=ext, image_size=(480, 640))


def _make_object(obj_id, center, front=(0, 0, -1)):
    center = np.asarray(center, dtype=float)
    front = np.asarray(front, dtype=float)
    front = front / (np.linalg.norm(front) + 1e-12)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)
    return MergedObject(
        id=obj_id, label=f"obj_{obj_id}", views=[0], center_world=center,
        rotation_world=np.column_stack([right, up, front]), front_world=front,
        up_world=up, right_world=right, euler_world_deg=np.zeros(3),
        dims=np.array([0.5, 0.5, 0.5]), corners_world=np.zeros((8, 3)),
        height=0.5, support_y=float(center[1] - 0.25),
    )


def _scene(centres, cam_pos=(0.0, 0.0, 0.0)):
    objs = [_make_object(i, c) for i, c in enumerate(centres)]
    return Scene(objects=objs, cameras=[_make_camera(cam_pos, [0, 0, 1])], images=[None])


_FIVE = np.array([
    [0.3, 0.1, 1.2],
    [-1.1, 0.4, 2.7],
    [2.2, -0.3, 0.6],
    [0.9, 0.8, -1.4],
    [-0.5, -0.6, 0.2],
])


# (i) scene_scale=2 == unnormalised with all positions halved --------------
def test_scene_scale_equals_halved_positions():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(6, 3))
    r, u, f = np.eye(3)
    a = compute_frame_relations(P, r, u, f, scene_scale=2.0)
    b = compute_frame_relations(P / 2.0, r, u, f)  # default scale 1.0
    for k in _KEYS:
        np.testing.assert_allclose(a[k], b[k], atol=1e-12, err_msg=k)


def test_invalid_scale_is_noop():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(4, 3))
    r, u, f = np.eye(3)
    ref = compute_frame_relations(P, r, u, f)
    for bad in (0.0, -3.0, float("nan"), float("inf")):
        out = compute_frame_relations(P, r, u, f, scene_scale=bad)
        for k in _KEYS:
            np.testing.assert_allclose(out[k], ref[k], atol=1e-12, err_msg=f"{k} scale={bad}")


# (ii) _compute_scene_scale == numpy quantile 0.9 of pairwise distances -----
def test_compute_scene_scale_matches_numpy_quantile():
    objs = [_make_object(i, c) for i, c in enumerate(_FIVE)]
    sc = Scene(objects=objs, cameras=[], images=[])
    d = np.linalg.norm(_FIVE[:, None, :] - _FIVE[None, :, :], axis=-1)
    iu = np.triu_indices(5, k=1)
    expected = float(np.quantile(d[iu], 0.9))
    assert sc._compute_scene_scale() == pytest.approx(expected, abs=1e-12)


# (iii) third_person matrices are scale invariant ------------------------
def test_third_person_scale_invariant():
    # Decimetre-scale layout so the raw (unnormalised) sigmoids are not yet
    # saturated and the pre-fix non-invariance is visible.
    small = _FIVE * 0.1
    base = _scene(small)
    big = _scene(small * 3.0, cam_pos=(0.0, 0.0, 0.0))
    v0 = base._frame(at=base.cameras[0])
    v3 = big._frame(at=big.cameras[0])
    K = len(base.objects)
    for k in ("left", "right", "front", "behind", "above", "below"):
        m0 = _np(getattr(v0.third_person, k))[:K, :K]
        m3 = _np(getattr(v3.third_person, k))[:K, :K]
        np.testing.assert_allclose(m3, m0, atol=1e-6, err_msg=k)

    # The pre-normalisation form (raw projections into the sigmoid) is NOT
    # scale invariant: recompute it inline and assert it differs.
    right = np.asarray(v0.frame_right)
    P0 = np.asarray(base._entity_positions())
    P3 = np.asarray(big._entity_positions())
    raw0 = steep_sigmoid_signed(-(P0 @ right)[:, None] + (P0 @ right)[None, :], steepness=14.0)
    raw3 = steep_sigmoid_signed(-(P3 @ right)[:, None] + (P3 @ right)[None, :], steepness=14.0)
    np.fill_diagonal(raw0, 0.0)
    np.fill_diagonal(raw3, 0.0)
    assert np.max(np.abs(raw3[:K, :K] - raw0[:K, :K])) > 1e-3


def test_third_person_equals_hand_formula():
    """view.third_person.left[i, j] == sigmoid(14 * (-(x_i - x_j).right) / s_scene)."""
    sc = _scene(_FIVE)
    v = sc._frame(at=sc.cameras[0])
    s = sc._compute_scene_scale()
    P = sc._entity_positions()
    r = P @ np.asarray(v.frame_right)
    fr = P @ np.asarray(v.frame_front)
    exp_left = steep_sigmoid_signed(-(r[:, None] - r[None, :]) / s, steepness=14.0)
    exp_front = steep_sigmoid_signed(-(fr[:, None] - fr[None, :]) / s, steepness=14.0)
    np.fill_diagonal(exp_left, 0.0)
    np.fill_diagonal(exp_front, 0.0)
    np.testing.assert_allclose(_np(v.third_person.left), exp_left, atol=1e-6)
    np.testing.assert_allclose(_np(v.third_person.front), exp_front, atol=1e-6)
