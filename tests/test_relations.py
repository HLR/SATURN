"""Pairwise relation matrices (saturn.predicates.relations).

Every matrix is checked two ways: against known values for a small hand-placed
configuration, and entry by entry against its defining formula on random
configurations that include the degenerate cases (coincident objects, a purely
vertical offset, an object without an orientation).
"""
import math

import numpy as np
import pytest

from saturn.predicates import relations
from saturn.predicates.relations import (
    compute_distance_matrices,
    compute_frame_independent_relations,
)

R, U, F = np.eye(3)          # frame axes: right = +X, up = +Y, front = +Z


def _sig(x, midpoint=0.0, steepness=14.0):
    """The scoring sigmoid, 1 / (1 + exp(-steepness * (x - midpoint))), clipped at +-60."""
    z = min(max(steepness * (x - midpoint), -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- frame relations

# Three objects within 0.15 of each other, so the sigmoid scores are graded:
#   0 at the origin, 1 right / up / slightly ahead of 0, 2 left / down / well ahead of 0.
# Entry [i, j] reads "i is <relation> of j" in the frame (right +X, up +Y, front +Z);
# "front" means closer to the observer (smaller front coordinate).
FRAME_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.1, 0.05, 0.02], [-0.04, -0.03, 0.15]])
FRAME_EXPECTED = {
    "left": [[0.0, 0.8021838885585818, 0.36354745971843366],
        [0.19781611144141822, 0.0, 0.12346704756522399],
        [0.6364525402815664, 0.8765329524347759, 0.0]],
    "right": [[0.0, 0.19781611144141822, 0.6364525402815664],
        [0.8021838885585818, 0.0, 0.8765329524347759],
        [0.36354745971843366, 0.12346704756522399, 0.0]],
    "above": [[0.0, 0.3318122278318339, 0.6034832498647263],
        [0.6681877721681662, 0.0, 0.7539887164489482],
        [0.3965167501352737, 0.2460112835510519, 0.0]],
    "below": [[0.0, 0.6681877721681662, 0.3965167501352737],
        [0.3318122278318339, 0.0, 0.2460112835510519],
        [0.6034832498647263, 0.7539887164489482, 0.0]],
    "front": [[0.0, 0.569546223939229, 0.8909031788043871],
        [0.43045377606077095, 0.0, 0.86056612703835],
        [0.10909682119561293, 0.13943387296165005, 0.0]],
    "behind": [[0.0, 0.43045377606077095, 0.10909682119561293],
        [0.569546223939229, 0.0, 0.13943387296165005],
        [0.8909031788043871, 0.86056612703835, 0.0]],
    "left_normalized": [[0.0, 0.9902903378454601, 0.37116867471983384],
        [0.009709662154539944, 0.0, 0.13360325418685043],
        [0.6288313252801662, 0.8663967458131496, 0.0]],
    "right_normalized": [[0.0, 0.009709662154539944, 0.6288313252801662],
        [0.9902903378454601, 0.0, 0.8663967458131496],
        [0.37116867471983384, 0.13360325418685043, 0.0]],
    "front_normalized": [[0.0, 0.598058067569092, 0.9831174698006231],
        [0.401941932430908, 0.0, 0.8402255496836388],
        [0.016882530199376855, 0.15977445031636112, 0.0]],
    "behind_normalized": [[0.0, 0.401941932430908, 0.016882530199376855],
        [0.598058067569092, 0.0, 0.15977445031636112],
        [0.9831174698006231, 0.8402255496836388, 0.0]],
    "above_normalized": [[0.0, 0.2798872734185941, 0.5948683298050514],
        [0.7201127265814059, 0.0, 0.693121819834107],
        [0.4051316701949486, 0.30687818016589297, 0.0]],
    "below_normalized": [[0.0, 0.7201127265814059, 0.4051316701949486],
        [0.2798872734185941, 0.0, 0.30687818016589297],
        [0.5948683298050514, 0.693121819834107, 0.0]],
}


def test_frame_relations_on_a_known_configuration():
    got = relations.compute_frame_relations(FRAME_POSITIONS, R, U, F)
    for key, expected in FRAME_EXPECTED.items():
        np.testing.assert_allclose(got[key], expected, atol=1e-12, rtol=0, err_msg=key)


@pytest.mark.parametrize("seed,K", [(0, 1), (1, 2), (2, 7), (3, 26)])
def test_frame_relations_match_their_definition(seed, K):
    rng = np.random.default_rng(seed); P = rng.normal(size=(K, 3))
    P[1 % K] = P[0]                                   # a coincident pair (no horizontal offset)
    got = relations.compute_frame_relations(P, R, U, F)
    assert set(FRAME_EXPECTED) | set(relations.DIRECTIONAL_COMBINATIONS) <= set(got)
    for i in range(K):
        for j in range(K):
            entry = {k: got[k][i, j] for k in got}
            if i == j:
                assert all(v == 0.0 for v in entry.values()), (i, j)
                continue
            dr, du, df = P[i] - P[j]                  # i relative to j, along right / up / front
            expected = {
                "left": _sig(-dr), "right": _sig(dr), "above": _sig(du), "below": _sig(-du),
                "front": _sig(-df), "behind": _sig(df),
                # a combined label scores the weaker of its two axial evidences
                "front_left": _sig(min(-df, -dr)), "front_right": _sig(min(-df, dr)),
                "behind_left": _sig(min(df, -dr)), "behind_right": _sig(min(df, dr)),
            }
            horiz = math.hypot(dr, df)
            if horiz < 1e-12:                         # straight above / below: no yaw
                expected.update({k + "_normalized": 0.0 for k in
                                 ("left", "right", "front", "behind", "above", "below")})
            else:
                sin_yaw, cos_yaw = dr / horiz, df / horiz
                sin_elev = math.sin(math.atan2(du, horiz))
                expected.update({
                    "left_normalized": (1 - sin_yaw) / 2, "right_normalized": (1 + sin_yaw) / 2,
                    "front_normalized": (1 - cos_yaw) / 2, "behind_normalized": (1 + cos_yaw) / 2,
                    "above_normalized": (1 + sin_elev) / 2, "below_normalized": (1 - sin_elev) / 2,
                })
            for k, v in expected.items():
                assert entry[k] == pytest.approx(v, abs=1e-12), (k, i, j)


# ---------------------------------------------------------------- frame-independent relations

# Four objects on the floor plane (object 1 raised by 0.5) with headings:
#   0 faces almost +Z, 1 faces mostly -X, 2 faces almost -Z (nearly opposite 0),
#   3 faces the +X/+Z diagonal (45 deg from 0).
# facing[i, j]: i's heading points at j.  parallel / perpendicular ignore the sign of
# the headings; orientation_distance is signed (0 same, 0.5 perpendicular, 1 opposite).
# between[i, j, k]: i lies on the segment from j to k.
INDEPENDENT_POSITIONS = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 1.0], [-1.0, 0.0, 3.0], [0.5, 0.0, 1.2]])
INDEPENDENT_FRONTS = np.array([[0.1, 0.0, 1.0], [-1.0, 0.0, 0.3], [0.2, 0.1, -1.0], [0.7, 0.0, 0.7]])
INDEPENDENT_EXPECTED = {
    "facing": [[0.0, 0.9994337462697463, 0.9999971697156476, 0.9999984768544639],
        [0.9999626372844461, 0.0, 0.9999984679801454, 0.999999008073118],
        [0.99999901002005, 0.999947373688012, 0.0, 0.9999951919667913],
        [2.3869461618521105e-06, 0.9997974460081426, 0.7803145649990264, 0.0]],
    "parallel": [[0.0, 0.000798963327608628, 0.9713323731716321, 0.7379618815580898],
        [0.000798963327608628, 0.0, 0.037083277338610623, 0.04059756734319475],
        [0.9713323731716321, 0.037083277338610623, 0.0, 0.11192116920591387],
        [0.7379618815580898, 0.04059756734319475, 0.11192116920591387, 0.0]],
    "perpendicular": [[0.0, 0.8222105072599296, 0.0001091257860862098, 0.0013113287037961625],
        [0.8222105072599296, 0.0, 0.08760785715842484, 0.08036504225082296],
        [0.0001091257860862098, 0.08760785715842484, 0.0, 0.028505618850199973],
        [0.0013113287037961625, 0.08036504225082296, 0.028505618850199973, 0.0]],
    "orientation_distance": [[0.0, 0.43895193835281127, 0.9005983624799555, 0.21827448256944634],
        [0.43895193835281127, 0.0, 0.6547988965050912, 0.6572264209222577],
        [0.9005983624799555, 0.6547988965050912, 0.0, 0.6861551872076451],
        [0.21827448256944634, 0.6572264209222577, 0.6861551872076451, 0.0]],
    "between": [[[0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5017904695458557, 0.0],
        [0.0, 0.5017904695458557, 0.0, 0.09554668933719046],
        [0.0, 0.0, 0.09554668933719046, 0.0]],
        [[0.0, 0.0, 0.1243449962748201, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.1243449962748201, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.7370694235097706, 0.8479586101441117, 0.0],
        [0.7370694235097706, 0.0, 0.918474411466226, 0.0],
        [0.8479586101441117, 0.918474411466226, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]]],
}


def test_frame_independent_relations_on_a_known_configuration():
    got = compute_frame_independent_relations(INDEPENDENT_POSITIONS, INDEPENDENT_FRONTS)
    assert set(got) == set(INDEPENDENT_EXPECTED)
    for key, expected in INDEPENDENT_EXPECTED.items():
        np.testing.assert_allclose(got[key], expected, atol=1e-12, rtol=0, err_msg=key)


@pytest.mark.parametrize("seed,K", [(0, 1), (1, 3), (2, 8), (3, 20)])
def test_frame_independent_relations_match_their_definition(seed, K):
    rng = np.random.default_rng(seed); P = rng.normal(size=(K, 3)); Fr = rng.normal(size=(K, 3))
    if K > 2: Fr[2] = 0.0                             # an object without an orientation
    if K > 1: P[1] = P[0] + [0, 1.0, 0]               # a purely vertical offset
    got = compute_frame_independent_relations(P, Fr)
    norms = np.linalg.norm(Fr, axis=1)
    for i in range(K):
        for j in range(K):
            entry = {k: got[k][i, j] for k in ("facing", "parallel", "perpendicular", "orientation_distance")}
            expected = dict.fromkeys(entry, 0.0)
            offset = P[j] - P[i]
            offset[1] = 0.0                           # horizontal offset from i to j
            dist_h = np.linalg.norm(offset)
            if i != j and norms[i] >= 1e-8 and dist_h >= 1e-8:
                fi = Fr[i] / norms[i]
                expected["facing"] = _sig(float(fi @ offset / dist_h))
                if norms[j] >= 1e-8:
                    cos_ij = float(fi @ (Fr[j] / norms[j]))
                    expected["parallel"] = _sig(abs(cos_ij), midpoint=0.7)
                    expected["perpendicular"] = _sig(0.3 - abs(cos_ij))
                    expected["orientation_distance"] = math.degrees(math.acos(min(1.0, max(-1.0, cos_ij)))) / 180.0
            for k, v in expected.items():
                assert entry[k] == pytest.approx(v, abs=1e-12), (k, i, j)
    assert got["between"].shape == (K, K, K)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    for i in range(K):
        for j in range(K):
            for k in range(K):
                if K < 3 or len({i, j, k}) < 3:
                    expected = 0.0
                else:                                 # detour through i relative to the direct j-k path
                    expected = min(max(1.0 - (D[j, i] + D[i, k] - D[j, k]) / (D[j, k] + 1e-6), 0.0), 1.0)
                assert got["between"][i, j, k] == pytest.approx(expected, abs=1e-12), (i, j, k)


# ---------------------------------------------------------------- distances

# closeness for five random points: 1 - distance / (95th-percentile distance), 0 on the diagonal.
CLOSENESS_EXPECTED = [[0.0, 0.8724800948410678, 0.4576744084197478, 0.586419831833382, 0.19641467213481356],
    [0.8724800948410678, 0.0, 0.43265156532279014, 0.6337800857701126, 0.23879210729977474],
    [0.4576744084197478, 0.43265156532279014, 0.0, 0.1942742028136788, 2.597351266286907e-07],
    [0.586419831833382, 0.6337800857701126, 0.1942742028136788, 0.0, 0.554392833611556],
    [0.19641467213481356, 0.23879210729977474, 2.597351266286907e-07, 0.554392833611556, 0.0]]


def test_distance_matrices_centres_and_deterministic_edges():
    rng = np.random.default_rng(0); P = rng.normal(size=(5, 3)); clouds = [P[i] + rng.normal(0, 0.05, (3000, 3)) for i in range(5)]
    b = compute_distance_matrices(P, None)
    raw = b["distance_center_raw"]
    np.testing.assert_allclose(raw, np.linalg.norm(P[:, None] - P[None], axis=-1), atol=1e-12)
    np.testing.assert_allclose(b["distance_edge_raw"], raw, atol=1e-12)    # no clouds: edge = centre
    np.testing.assert_allclose(b["closeness"], CLOSENESS_EXPECTED, atol=1e-12, rtol=0)
    # distance / distance_edge divide by the largest distance (unclipped), so they keep the full ranking
    expect = raw / raw.max(); np.fill_diagonal(expect, 0.0)
    np.testing.assert_allclose(b["distance"], expect, atol=1e-12)
    np.testing.assert_allclose(b["distance_edge"], expect, atol=1e-12)
    assert np.isclose(b["distance"].max(), 1.0) and len(np.unique(np.round(b["distance"][np.triu_indices(5, 1)], 9))) == 10
    e1 = compute_distance_matrices(P, clouds); e2 = compute_distance_matrices(P, clouds)
    np.testing.assert_array_equal(e1["distance_edge_raw"], e2["distance_edge_raw"])   # deterministic subsample
    # exact minimum over full clouds is a lower bound on any subsample's minimum
    for i in range(5):
        for j in range(i + 1, 5):
            exact = np.min(np.linalg.norm(clouds[i][:, None] - clouds[j][None], axis=-1))
            assert e1["distance_edge_raw"][i, j] >= exact - 1e-9


# ---- relation matrices stay in [0, 1] with the right meaning --------------


def test_closeness_stays_in_unit_interval():
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0.5, 0, 0.5], [6, 0, 6]], float)
    d = compute_distance_matrices(P)
    for key in ("distance", "distance_edge", "closeness"):
        assert d[key].min() >= 0.0 and d[key].max() <= 1.0


def test_orientation_distance_opposite_is_one():
    pos = np.array([[0, 0, 0], [2, 0, 0.0]])
    opposite = compute_frame_independent_relations(pos, np.array([[0, 0, 1], [0, 0, -1.0]]))
    same = compute_frame_independent_relations(pos, np.array([[0, 0, 1], [0, 0, 1.0]]))
    perp = compute_frame_independent_relations(pos, np.array([[0, 0, 1], [1, 0, 0.0]]))
    assert opposite["orientation_distance"][0, 1] == pytest.approx(1.0)
    assert same["orientation_distance"][0, 1] == pytest.approx(0.0)
    assert perp["orientation_distance"][0, 1] == pytest.approx(0.5)
