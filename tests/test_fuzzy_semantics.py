"""Fuzzy-logic semantics of the soft-logic engine, each checked against its textbook definition.

forall is the minimum over its variable; comparisons and arithmetic align operands by
variable name; count counts entities, not tuples; exists / argmax bind objects only;
best() is a restricted superlative (rank by a measure among an eligible set).
"""
import math

import numpy as np
import pytest
import torch

from saturn.soft_logic.tensor import BINDABLE_ENTITIES, ProbabilisticTensor as PT


def pt(data, vars):
    return PT(torch.as_tensor(data, dtype=torch.float64), vars=vars)


@pytest.fixture
def training():
    PT.training = True
    yield
    PT.training = False


@pytest.fixture
def two_objects_one_camera():
    token = BINDABLE_ENTITIES.set(2)
    yield
    BINDABLE_ENTITIES.reset(token)


# ---------------------------------------------------------------- forall
def test_forall_is_the_minimum():
    assert float(pt([0.6, 0.8], ["x1"]).forall("x1")) == pytest.approx(0.6)
    assert float(pt([0.6, 0.8], ["x1"]).forall()) == pytest.approx(0.6)


def test_forall_skips_self_pairs_of_distinct_variables():
    # R(x1, x2) for every x2 other than x1: the diagonal (x1 == x2) never counts against it.
    r = pt([[0.0, 0.8, 0.7], [0.9, 0.0, 0.3], [0.6, 0.5, 0.0]], ["x1", "x2"])
    np.testing.assert_allclose(r.forall("x2").tensor.cpu().numpy(), [0.7, 0.3, 0.5])
    assert r.forall("x2").vars == ["x1"]


def test_forall_ignores_non_object_slots(two_objects_one_camera):
    assert float(pt([0.7, 0.9, 0.0], ["x1"]).forall("x1")) == pytest.approx(0.7)


# ---------------------------------------------------------------- count
def test_count_counts_entities_not_tuples():
    # One chair (x1 = 0) satisfies the formula with three anchors x2: it is one chair.
    j = pt([[0.0, 0.9, 0.9, 0.9], [0.1, 0.0, 0.2, 0.1], [0.1, 0.2, 0.0, 0.3], [0.2, 0.1, 0.1, 0.0]], ["x1", "x2"])
    assert float(j.count()) == 1.0


def test_count_unary_is_the_half_cut():
    assert float(pt([0.9, 0.5, 0.49, 0.1], ["x1"]).count()) == 2.0


def test_count_in_training_is_the_sigma_count(training):
    assert float(pt([0.9, 0.5, 0.4], ["x1"]).count()) == pytest.approx(1.8)


# ---------------------------------------------------------------- comparisons and arithmetic
def test_comparison_of_two_variables_is_a_relation():
    a, b = pt([0.2, 0.8], ["x1"]), pt([0.7, 0.3], ["x2"])
    rel = a >= b
    assert rel.vars == ["x1", "x2"]
    np.testing.assert_allclose(rel.tensor.cpu().numpy(), [[0.0, 0.0], [1.0, 1.0]])


def test_soft_comparison_is_symmetric(training):
    a, b = pt([0.654], ["x1"]), pt([0.615], ["x1"])
    gt, lt = float((a > b).tensor[0]), float((b > a).tensor[0])
    assert gt + lt == pytest.approx(1.0)
    assert gt == pytest.approx(1 / (1 + math.exp(-0.039 / 0.05)), rel=1e-6)


def test_threshold_test_keeps_its_variable():
    t = pt([0.2, 0.8], ["x2"]) > 0.5
    assert t.vars == ["x2"] and t.tensor.cpu().tolist() == [0.0, 1.0]


def test_power_keeps_the_variable_name():
    very = pt([0.5, 0.9], ["x2"]) ** 2          # Zadeh's concentration ("very")
    assert very.vars == ["x2"]
    np.testing.assert_allclose(very.tensor.cpu().numpy(), [0.25, 0.81])


def test_lukasiewicz_implication_aligns_variables():
    imp = pt([0.9, 0.2], ["x1"]).implies(pt([0.3, 0.6], ["x2"]), logic="lukasiewicz")
    assert imp.vars == ["x1", "x2"]
    np.testing.assert_allclose(imp.tensor.cpu().numpy(), [[0.4, 0.7], [1.0, 1.0]])


def test_unknown_implication_logic_is_an_error():
    with pytest.raises(ValueError):
        pt([0.5], ["x1"]).implies(pt([0.5], ["x1"]), logic="goedel-typo")


# ---------------------------------------------------------------- iota
def test_iota_backpropagates():
    x = torch.tensor([0.2, 0.9, 0.5], dtype=torch.float64, requires_grad=True)
    PT(x, vars=["x1"]).iota("x1").tensor.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------- objects only
def test_exists_and_argmax_bind_objects_only(two_objects_one_camera):
    j = pt([0.1, 0.2, 0.99], ["x1"])            # two objects, then a camera
    assert float(j.exists()) == pytest.approx(0.2)
    assert j.argmax("x1") == 1
    assert j.assign()["x1"] == 1


# ---------------------------------------------------------------- best (restricted superlative)
def test_best_ranks_only_the_eligible_entities():
    # The cat scene: chair barely left but farther, shelf straight left, table on the right.
    far = pt([0.654, 0.615, 0.9], ["x1"])        # the table is farthest of all, but not on the left
    left = pt([0.587, 1.0, 0.0], ["x1"])
    j = far.best("x1", among=left)
    np.testing.assert_allclose(j.tensor.cpu().numpy(), [1.0, 0.615 / 0.654, 0.0])
    assert j.vars == ["x1"]


def test_best_projects_out_an_anchor_variable():
    # closeness(x1, x2) & the sink(x2): the measure is closeness to the sink.
    close = pt([[0.0, 0.2, 0.9], [0.2, 0.0, 0.5], [0.9, 0.5, 0.0]], ["x1", "x2"])
    sink = pt([0.0, 0.0, 1.0], ["x2"])
    j = (close & sink).best("x1", among=pt([1.0, 1.0, 0.0], ["x1"]))
    np.testing.assert_allclose(j.tensor.cpu().numpy(), [1.0, 0.5 / 0.9, 0.0])


def test_best_soft_eligibility_and_ties():
    m = pt([0.8, 0.8, 0.4], ["x1"])
    np.testing.assert_allclose(m.best("x1").tensor.cpu().numpy(), [1.0, 1.0, 0.5])   # a tie stays a tie
    soft = m.best("x1", among=pt([0.55, 0.45, 1.0], ["x1"]), tau=0.05)
    s = [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1)), 1 / (1 + math.exp(-10))]
    scores = [0.8 * s[0], 0.8 * s[1], 0.4 * s[2]]
    np.testing.assert_allclose(soft.tensor.cpu().numpy(), [v / max(scores) for v in scores], rtol=1e-6)


def test_best_with_nobody_eligible_is_all_zero():
    assert pt([0.3, 0.9], ["x1"]).best("x1", among=pt([0.1, 0.2], ["x1"])).tensor.cpu().tolist() == [0.0, 0.0]


def test_best_never_picks_a_camera(two_objects_one_camera):
    j = pt([0.3, 0.4, 0.99], ["x1"]).best("x1")
    np.testing.assert_allclose(j.tensor.cpu().numpy(), [0.75, 1.0, 0.0])
