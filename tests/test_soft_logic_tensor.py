"""The soft-logic ProbabilisticTensor: assign(), operators across devices,
counting, quantifiers, comparisons, argmin/amax/amin and negated relations."""

import math
import types

import pytest
import torch

import saturn.soft_logic as sl
import saturn.soft_logic.tensor as T
from saturn.errors import DegenerateBindingError


@pytest.fixture(autouse=True)
def _strict_assign(monkeypatch):
    # These tests check the strict guard; the pipeline default logs and keeps the argmax.
    monkeypatch.setenv("SAPY_STRICT_ASSIGN", "1")


NAN = float("nan")


@pytest.fixture
def PT():
    return sl.ProbabilisticTensor


# --- NaN never wins assign() ------------------------------------------------

def test_assign_skips_nan_cell(PT):
    a = PT(torch.tensor([NAN, 0.2, 0.9]), vars=["x1"])
    b = PT(torch.tensor([0.9, 0.9, 0.9]), vars=["x1"])
    joint = a & b
    assert joint.assign() == {"x1": 2}
    assert joint.assign() == {"x1": joint.argmax()}


@pytest.mark.parametrize("values", [[NAN, NAN, NAN], [NAN, 0.0, 0.0]])
def test_assign_nan_without_evidence_is_unsatisfiable(PT, values):
    with pytest.raises(DegenerateBindingError):
        PT(torch.tensor(values), vars=["x1"]).assign()


def test_assign_min_skips_nan(PT):
    assert PT(torch.tensor([NAN, 0.5, 0.1]), vars=["x1"]).assign("min") == {"x1": 2}


# --- | after & on a CUDA host -----------------------------------------------

def test_or_after_and_on_cuda_host(monkeypatch):
    # Simulate a CUDA host without touching a GPU: "cuda" maps to the meta device.
    real = torch

    class _Proxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(real, name)

    proxy = _Proxy("torch_proxy")
    proxy.cuda = types.SimpleNamespace(is_available=lambda: True)
    proxy.device = lambda d, *a: real.device("meta") if str(d).startswith("cuda") else real.device(d, *a)
    monkeypatch.setattr(T, "torch", proxy)

    P = T.ProbabilisticTensor
    a, b, c = (P(torch.rand(3), vars=["x1"]) for _ in range(3))
    joint = a & b
    assert joint.tensor.device.type == "meta"
    assert ((a & b) | c).tensor.device.type == "meta"
    assert (c | (a & b)).tensor.device.type == "meta"
    assert c.tensor.device.type == "cpu"   # callers' operands are not moved


# --- count over 3+ variables ------------------------------------------------

def test_count_keeps_the_named_axis(PT):
    t = torch.zeros(2, 3, 4)
    t[1, 2, 3] = 1.0
    p = PT(t, vars=["x1", "x2", "x3"])
    assert p.count("x3").tensor.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert p.count("x1").tensor.tolist() == [0.0, 1.0]
    assert p.count("x1").vars == ["x1"]
    assert p.count("x1", "x3").tensor.shape == (2, 4)


# --- scalar exists() composes ------------------------------------------------

def test_scalar_exists_conjunction(PT):
    found = PT(torch.tensor([0.0, 0.1])).exists() & PT(torch.tensor([0.0, 0.2])).exists()
    assert found.tensor.shape == ()
    assert math.isclose(float(found), 0.1, rel_tol=1e-6)
    assert not bool(found)


def test_scalar_exists_and_predicate(PT):
    r = PT(torch.tensor([0.0, 0.9])).exists() & PT(torch.tensor([0.1, 0.8]), vars=["x1"])
    assert r.vars == ["x1"]
    assert torch.allclose(r.tensor.cpu(), torch.tensor([0.1, 0.8]))


# --- >= and <= are crisp outside training ------------------------------------

def test_ge_le_crisp(PT):
    assert not bool(PT(torch.tensor(0.3)) >= 0.5)
    assert bool(PT(torch.tensor(0.5)) >= 0.5)
    assert not bool(PT(torch.tensor(0.7)) <= 0.5)
    assert bool(PT(torch.tensor(0.3)) <= 0.5)
    # integer counts behave as before
    assert bool(PT(torch.tensor(2.0)) >= 2)
    assert not bool(PT(torch.tensor(1.0)) >= 2)


# --- argmin default, amax/amin -----------------------------------------------

def test_argmin_default_and_reductions(PT):
    p = PT(torch.tensor([0.3, 0.1, 0.5]))
    assert p.argmin() == 1
    assert PT(torch.tensor([NAN, 0.4, 0.2])).argmin() == 2
    assert math.isclose(float(PT(torch.tensor([0.3, 0.1])).amax()), 0.3, rel_tol=1e-6)
    assert math.isclose(float(PT(torch.tensor([0.3, 0.1])).amin()), 0.1, rel_tol=1e-6)


# --- ~ on relations -----------------------------------------------------------

def test_invert_square_relation(PT):
    inv = ~PT(torch.full((3, 3), 0.2), vars=["x1", "x2"])
    t = inv.tensor
    assert torch.all(t.diag() == 0)
    off = t[~torch.eye(3, dtype=torch.bool)]
    assert torch.allclose(off, torch.full_like(off, 0.8))


def test_invert_non_square_and_3d(PT):
    inv = ~PT(torch.full((3, 2), 0.2), vars=["x1", "c1"])
    assert torch.allclose(inv.tensor, torch.full((3, 2), 0.8))
    cube = (~PT(torch.full((3, 3, 3), 0.2), vars=["x1", "x2", "x3"])).tensor
    assert cube.min() >= 0
    assert cube[0, 0, 1] == 0 and cube[0, 1, 1] == 0   # x1==x2 and x2==x3 masked
    assert math.isclose(float(cube[0, 1, 2]), 0.8, rel_tol=1e-6)


def test_invert_vector_unchanged(PT):
    assert torch.allclose((~PT(torch.tensor([0.2, 0.7]))).tensor, torch.tensor([0.8, 0.3]))




def test_iota_rejects_an_unknown_normalization_method():
    import pytest
    import torch
    from saturn.soft_logic.tensor import ProbabilisticTensor
    with pytest.raises(ValueError, match="Unknown normalization method"):
        ProbabilisticTensor(torch.tensor([0.2, 0.8]), vars=["x1"]).iota("x1", method="max")
    with pytest.raises(ValueError, match="Unknown normalization method"):
        ProbabilisticTensor(torch.tensor(0.5), vars=[]).iota(method="max")


def test_and_and_or_share_one_variable_layout():
    import torch
    from saturn.soft_logic.tensor import ProbabilisticTensor, and_op, or_op
    a = ProbabilisticTensor(torch.tensor([[0.1, 0.9], [0.6, 0.3]]), vars=["x2", "x1"])
    b = ProbabilisticTensor(torch.tensor([0.5, 0.7]), vars=["x1"])
    both, either = and_op(b, a), or_op(b, a)
    assert both.vars == either.vars == ["x1", "x2"]
    at = a.tensor.T                                   # a laid out as [x1, x2]
    bt = b.tensor[:, None].expand(2, 2)
    assert torch.allclose(either.tensor.cpu(), torch.maximum(at, bt))
    assert torch.allclose(both.tensor.cpu().diagonal(), torch.zeros(2))   # x1 and x2 bind distinct entities
