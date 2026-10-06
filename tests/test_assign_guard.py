"""assign() refuses a formula with no satisfying assignment.

argmax of an all-zero joint is index 0 for every variable, which reads as a
real answer. assign() raises DegenerateBindingError instead.
"""

import pytest
import torch

import saturn.soft_logic as sl
from saturn.errors import DegenerateBindingError, ExecutionError
from saturn.soft_logic import ProbabilisticTensor


@pytest.fixture(autouse=True)
def _strict_assign(monkeypatch):
    # These tests check the strict guard; the pipeline default logs and keeps the argmax.
    monkeypatch.setenv("SAPY_STRICT_ASSIGN", "1")



@pytest.fixture
def PT():
    return sl.ProbabilisticTensor


def test_normal_assign_unchanged(PT):
    rel = PT(torch.tensor([[0.0, 0.2], [0.9, 0.0]]), vars=["x1", "x2"])
    assert rel.assign() == {"x1": 1, "x2": 0}


def test_fewer_entities_than_variables_raises(PT):
    # One entity, two variables: & binds distinct entities, so the joint is all zero.
    a = PT(torch.tensor([1.0]))
    joint = a("x1") & a("x2")
    assert float(joint.tensor.max()) == 0.0
    with pytest.raises(DegenerateBindingError) as ei:
        joint.assign()
    msg = str(ei.value)
    assert "x1" in msg and "x2" in msg and "no satisfying assignment" in msg


def test_zero_predicate_raises(PT):
    joint = PT(torch.tensor([0.8, 0.6, 0.3]))("x1") & PT(torch.zeros(3))("x1")
    with pytest.raises(DegenerateBindingError):
        joint.assign()


def test_is_an_execution_error():
    assert issubclass(DegenerateBindingError, ExecutionError)
    with pytest.raises(ExecutionError):
        ProbabilisticTensor(torch.zeros(3)).assign()


def test_min_mode_not_guarded():
    assert ProbabilisticTensor(torch.zeros(3)).assign(mode="min") == {"x1": 0}
