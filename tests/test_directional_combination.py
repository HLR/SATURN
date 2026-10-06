"""Directional combinations as min-conjunction (sigmoid / third-person family).

Paper (appendix_predicate.tex, "Directional combinations"):

    h_r(delta_bar) = min_{c in C(r)} h_c(delta_bar)
    S_r = sigmoid((h_r - m_comb) / tau_comb),  m_comb = 0, tau_comb = 1/14

Implemented in ``compute_frame_relations`` and exposed as
``view.third_person.front_left`` etc.  The cosine family (first_person,
Anchor.*) keeps its 45-degree-target cosine kernel and is not covered here.
"""
import numpy as np
import pytest

from saturn.predicates.relations import DIRECTIONAL_COMBINATIONS, compute_frame_relations
from saturn.predicates.scoring import steep_sigmoid_signed
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject


def _np(t):
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


def _random_frame(rng):
    yaw = rng.uniform(-np.pi, np.pi)
    front = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    return front, np.cross(up, front), up


# (i) S_diag == min(S_c1, S_c2) on 20 random configurations
def test_min_conjunction_random_configs():
    rng = np.random.default_rng(42)
    for _ in range(20):
        front, right, up = _random_frame(rng)
        P = rng.normal(size=(5, 3))
        rel = compute_frame_relations(P, right, up, front, scene_scale=rng.uniform(0.5, 3.0))
        for diag, (c1, c2) in DIRECTIONAL_COMBINATIONS.items():
            np.testing.assert_allclose(
                rel[diag], np.minimum(rel[c1], rel[c2]), atol=1e-12, err_msg=diag
            )


# (ii) monotonicity: S_diag <= each component
def test_diagonal_bounded_by_components():
    rng = np.random.default_rng(7)
    for _ in range(20):
        front, right, up = _random_frame(rng)
        rel = compute_frame_relations(rng.normal(size=(4, 3)), right, up, front)
        for diag, (c1, c2) in DIRECTIONAL_COMBINATIONS.items():
            assert np.all(rel[diag] <= rel[c1] + 1e-12)
            assert np.all(rel[diag] <= rel[c2] + 1e-12)


# (iii) view.third_person.front_left exists and equals sigmoid(14*min(h_front,h_left))
def test_third_person_front_left_hand_computed():
    cam = _make_camera([0, 0, 0], [0, 0, 1])
    objs = [_make_object(0, [0.0, 0.0, 0.0]), _make_object(1, [0.05, 0.0, 0.12])]
    sc = Scene(objects=objs, cameras=[cam], images=[None])
    v = sc._frame(at=sc.cameras[0])
    s = sc._compute_scene_scale()

    fl = _np(v.third_person.front_left)
    assert fl.shape[0] == len(objs) + 1

    # i = obj0, j = obj1.  Frame: right = +x, front = +z.
    # disp = pos_i - pos_j = (-0.05, 0, -0.12)
    # h_front = -disp_front / s = 0.12 / s ;  h_left = -disp_right / s = 0.05 / s
    h_front = 0.12 / s
    h_left = 0.05 / s
    expected = float(steep_sigmoid_signed(np.array([min(h_front, h_left)]), steepness=14.0)[0])
    assert fl[0, 1] == pytest.approx(expected, abs=1e-6)
    assert fl[0, 1] == pytest.approx(
        min(_np(v.third_person.front)[0, 1], _np(v.third_person.left)[0, 1]), abs=1e-9)

    # aliases and vocabulary
    np.testing.assert_allclose(_np(v.third_person("front-left")), fl, atol=1e-12)
    np.testing.assert_allclose(_np(v.third_person["behind_right"]),
                               _np(v.third_person.behind_right), atol=1e-12)
    with pytest.raises(AttributeError):
        v.third_person.back_left
