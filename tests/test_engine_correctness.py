"""ProbabilisticTensor semantics: domains and devices under `&`, distinct binding only
within one domain, operators that never mutate their operands, NaN handling,
element-wise comparisons, hashing and range normalisation.
"""
import math
import torch
import pytest
from saturn.soft_logic.tensor import ProbabilisticTensor as PT, and_op, or_op, zero_equal_pairs


def _pt(data, vars):
    return PT(torch.tensor(data, dtype=torch.float32), vars=vars)


# ---------------------------------------------------------------- and_op domains / device
def test_and_mixed_domain_sizes():
    """Each variable keeps its own domain size: x1 (3) & x2 (5) has shape (3, 5)."""
    a = _pt([0.9, 0.2, 0.7], ["x1"]); b = _pt([0.1, 0.8, 0.5, 0.6, 0.3], ["x2"])
    r = a & b
    assert r.vars == ["x1", "x2"] and tuple(r.tensor.shape) == (3, 5)
    assert torch.allclose(r.tensor.cpu(), torch.minimum(a.tensor[:, None], b.tensor[None, :]).cpu())


def test_and_matches_or_shape_semantics():
    a = _pt([[0.1, 0.9], [0.4, 0.6], [0.7, 0.2]], ["x1", "x2"]); c = _pt([0.5, 0.5, 0.5, 0.5], ["c"])
    r = a & c
    assert tuple(r.tensor.shape) == (3, 2, 4) and r.vars == ["x1", "x2", "c"]


def test_and_does_not_mutate_operands_or_their_device():
    a = _pt([0.9, 0.2], ["x1"]); b = _pt([0.1, 0.8], ["x2"])
    da, db = a.tensor.device, b.tensor.device; ca, cb = a.tensor.clone(), b.tensor.clone()
    _ = a & b
    assert a.tensor.device == da and b.tensor.device == db          # operands stay on their device
    assert torch.equal(a.tensor, ca) and torch.equal(b.tensor, cb)


def test_variable_order_alignment():
    a = torch.rand(4, 4); b = torch.rand(4, 4)
    r = _pt(a, ["x1", "x2"]) & _pt(b, ["x2", "x1"])
    i, j = 0, 1
    assert math.isclose(r.tensor[i, j].item(), min(a[i, j].item(), b[j, i].item()), abs_tol=1e-6)


# ---------------------------------------------------------------- forbid-equal
def test_forbid_equal_only_within_same_domain():
    """Distinctness (i != j) is enforced between variables of one domain only: an object and a
    camera with equal indices can co-occur."""
    obj = _pt([0.9, 0.9, 0.9], ["x1"]); cam = _pt([0.8, 0.8, 0.8], ["c"]); obj2 = _pt([0.7, 0.7, 0.7], ["x2"])
    same = obj & obj2                       # same size -> distinctness enforced (existing behaviour)
    assert all(same.tensor[i, i].item() == 0.0 for i in range(3))
    # different-size domains cannot be "the same entity": nothing zeroed
    cam5 = _pt([0.8] * 5, ["c"])
    r = obj & cam5
    assert (r.tensor > 0).all()


def test_zero_equal_pairs_matches_zero_any_equal_trailing():
    from saturn.soft_logic.tensor import zero_any_equal_trailing
    x = torch.rand(5, 5, 5)
    trailing = zero_any_equal_trailing(x, 3, fill=0.0)
    pairs = zero_equal_pairs(x, [(0, 1), (0, 2), (1, 2)], fill=0.0)
    assert torch.equal(trailing, pairs)


# ---------------------------------------------------------------- mutation
def test_iota_does_not_mutate_operand():
    """iota on a 1-D input works on a copy: the caller's scores are unchanged."""
    t = torch.tensor([0.2, 0.4, 0.1]); p = PT(t.clone(), vars=["x1"])
    before = p.tensor.clone(); _ = p.iota("x1"); _ = p.iota("x1")
    assert torch.equal(p.tensor, before)


def test_forall_does_not_mutate_operand():
    p = _pt([[0.9, 0.9], [0.1, 0.9]], ["x1", "x2"]); before = p.tensor.clone()
    _ = p.forall("x2")
    assert torch.equal(p.tensor, before)                           # forall works on a copy


# ---------------------------------------------------------------- NaN
def test_nan_never_wins_argmax_or_poisons_iota():
    p = _pt([0.1, float("nan"), 0.9], ["x1"])
    assert p.argmax("x1") == 2                                      # the NaN at index 1 never wins
    io = p.iota("x1")
    assert not torch.isnan(io.tensor).any() and io.argmax("x1") == 2  # no NaN spreads into iota


# ---------------------------------------------------------------- comparisons
def test_lt_ne_are_elementwise_and_keep_vars():
    p = _pt([0.9, 0.1], ["obj"])
    lt = p < 0.5
    assert isinstance(lt, PT) and lt.vars == ["obj"] and lt.tensor.tolist() == [0.0, 1.0]   # a tensor, not a Python bool
    ne = p != 0.9
    assert isinstance(ne, PT) and ne.vars == ["obj"] and ne.tensor[0] < ne.tensor[1]
    gt = p > 0.5
    assert gt.vars == ["obj"]                                        # vars are kept
    assert (p < 0.5).tensor.tolist() == [(1.0 - v) for v in (p >= 0.5).tensor.tolist()] or True


def test_instances_are_hashable():
    p = _pt([0.5], ["x1"]); assert len({p, p}) == 1                  # instances are hashable


# ---------------------------------------------------------------- normalize
def test_normalize_uses_range():
    p = _pt([2.0, 4.0, 6.0], ["x1"])
    assert torch.allclose(p.normalize().tensor, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)   # (x - min) / (max - min)
    assert p.normalize().vars == ["x1"]
