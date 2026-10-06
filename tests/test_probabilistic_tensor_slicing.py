"""Tests for ProbabilisticTensor.__getitem__ slicing.

The contract under test:
  - A slice returns a *new ProbabilisticTensor* (not a raw torch.Tensor).
  - The new tensor's `.vars` list correctly reflects which axes were dropped,
    kept, or inserted:
      * int / 0-d tensor / np scalar       → axis collapsed, var dropped
      * slice / list / 1-d tensor / mask   → axis kept, var preserved
      * None / np.newaxis                  → new axis inserted (placeholder var)
      * Ellipsis                           → expands to enough slice(None)s
  - The underlying tensor data matches `torch.Tensor.__getitem__`.
  - String-keyed slicing (`pt["var_name"]`) drops that var.

"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from saturn.soft_logic.tensor import ProbabilisticTensor


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def pt_3d():
    """A 3-D ProbabilisticTensor with explicit, distinct var names."""
    t = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    return ProbabilisticTensor(t, vars=["cam", "obj", "rel"])


@pytest.fixture
def pt_2d():
    t = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    return ProbabilisticTensor(t, vars=["subject", "reference"])


# ----------------------------------------------------------------------
# Return type
# ----------------------------------------------------------------------

def test_returns_probabilistic_tensor(pt_3d):
    assert isinstance(pt_3d[0], ProbabilisticTensor)
    assert isinstance(pt_3d[0, 1], ProbabilisticTensor)
    assert isinstance(pt_3d[:, 1], ProbabilisticTensor)
    assert isinstance(pt_3d[0:1], ProbabilisticTensor)


# ----------------------------------------------------------------------
# Vars propagation: axis-collapsing keys
# ----------------------------------------------------------------------

def test_int_on_first_axis_drops_first_var(pt_3d):
    s = pt_3d[1]
    assert s.vars == ["obj", "rel"]
    assert s.tensor.shape == (3, 4)
    assert torch.equal(s.tensor, pt_3d.tensor[1])


def test_int_on_second_axis_drops_second_var(pt_3d):
    s = pt_3d[:, 2]
    assert s.vars == ["cam", "rel"]
    assert s.tensor.shape == (2, 4)
    assert torch.equal(s.tensor, pt_3d.tensor[:, 2])


def test_two_ints_drops_two_vars(pt_3d):
    s = pt_3d[1, 2]
    assert s.vars == ["rel"]
    assert s.tensor.shape == (4,)
    assert torch.equal(s.tensor, pt_3d.tensor[1, 2])


def test_three_ints_collapses_to_scalar(pt_3d):
    s = pt_3d[1, 2, 3]
    assert s.vars == []
    assert s.tensor.shape == ()
    assert s.tensor.item() == pt_3d.tensor[1, 2, 3].item()


# ----------------------------------------------------------------------
# Vars propagation: axis-preserving keys
# ----------------------------------------------------------------------

def test_slice_on_first_axis_preserves_all_vars(pt_3d):
    s = pt_3d[0:2]
    assert s.vars == ["cam", "obj", "rel"]
    assert s.tensor.shape == (2, 3, 4)


def test_slice_on_first_axis_resized_preserves_all_vars(pt_3d):
    s = pt_3d[0:1]  # narrower slice
    assert s.vars == ["cam", "obj", "rel"]
    assert s.tensor.shape == (1, 3, 4)


def test_slice_on_middle_axis_preserves_all_vars(pt_3d):
    s = pt_3d[:, 1:3]
    assert s.vars == ["cam", "obj", "rel"]
    assert s.tensor.shape == (2, 2, 4)


def test_slice_int_mix_preserves_outer_drops_middle(pt_3d):
    s = pt_3d[0:1, 1]
    assert s.vars == ["cam", "rel"]
    assert s.tensor.shape == (1, 4)


def test_fancy_index_list_keeps_axis(pt_3d):
    s = pt_3d[[0, 1]]
    assert s.vars == ["cam", "obj", "rel"]
    assert s.tensor.shape == (2, 3, 4)


def test_fancy_index_1d_tensor_keeps_axis(pt_3d):
    s = pt_3d[torch.tensor([0, 1])]
    assert s.vars == ["cam", "obj", "rel"]
    assert s.tensor.shape == (2, 3, 4)


def test_boolean_mask_keeps_axis(pt_3d):
    s = pt_3d[torch.tensor([True, False])]
    assert s.vars == ["cam", "obj", "rel"]
    assert s.tensor.shape == (1, 3, 4)


# ----------------------------------------------------------------------
# Vars propagation: Ellipsis and newaxis
# ----------------------------------------------------------------------

def test_ellipsis_then_int_drops_last_var(pt_3d):
    s = pt_3d[..., 2]
    assert s.vars == ["cam", "obj"]
    assert s.tensor.shape == (2, 3)


def test_int_then_ellipsis_drops_first_var(pt_3d):
    s = pt_3d[1, ...]
    assert s.vars == ["obj", "rel"]
    assert s.tensor.shape == (3, 4)


def test_ellipsis_alone_keeps_all_vars(pt_3d):
    s = pt_3d[...]
    assert s.vars == ["cam", "obj", "rel"]


def test_newaxis_inserts_placeholder(pt_3d):
    s = pt_3d[None]
    # New axis prepended; original vars stay shifted in by one placeholder.
    assert len(s.vars) == 4
    assert s.vars[1:] == ["cam", "obj", "rel"]
    assert s.tensor.shape == (1, 2, 3, 4)


def test_newaxis_in_middle(pt_3d):
    s = pt_3d[:, None]
    assert len(s.vars) == 4
    assert s.vars[0] == "cam"
    assert s.vars[2:] == ["obj", "rel"]
    assert s.tensor.shape == (2, 1, 3, 4)


# ----------------------------------------------------------------------
# 0-d torch tensor key (existing code path)
# ----------------------------------------------------------------------

def test_zero_d_tensor_key_drops_axis(pt_3d):
    idx = torch.tensor(1)
    s = pt_3d[idx]
    assert s.vars == ["obj", "rel"]
    assert s.tensor.shape == (3, 4)


def test_tuple_with_zero_d_tensor_keys(pt_3d):
    s = pt_3d[torch.tensor(1), torch.tensor(2)]
    assert s.vars == ["rel"]
    assert s.tensor.shape == (4,)


# ----------------------------------------------------------------------
# String-keyed indexing (named-axis lookup)
# ----------------------------------------------------------------------

def test_string_key_drops_named_axis(pt_3d):
    # "cam" is vars[0]; pt["cam"] is currently coerced to pt[0]; this means
    # the returned tensor selects the 0-th slice of axis 0 — equivalent to
    # pt_3d[0]. Vars after: ["obj", "rel"].
    s = pt_3d["cam"]
    assert s.vars == ["obj", "rel"]


# ----------------------------------------------------------------------
# vars=None default still works
# ----------------------------------------------------------------------

def test_default_vars_no_crash():
    """When vars is auto-generated (default __init__ behavior), slicing
    still propagates a sensible vars list (e.g. drops the auto name x1
    after an int-index)."""
    t = torch.zeros(2, 3)
    pt = ProbabilisticTensor(t)  # vars defaults to ['x1', 'x2']
    s = pt[0]
    # Should not crash; vars list should drop x1.
    assert isinstance(s, ProbabilisticTensor)
    assert s.vars == ["x2"]


# ----------------------------------------------------------------------
# Underlying tensor data still correct
# ----------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    1,
    (1, 2),
    (slice(None), 2),
    slice(0, 1),
    (slice(None), slice(1, 3)),
    (Ellipsis, 2),
    (1, Ellipsis),
    ([0, 1],),
    (torch.tensor([True, False]),),
])
def test_underlying_tensor_matches_torch(pt_3d, key):
    expected = pt_3d.tensor[key]
    got = pt_3d[key].tensor
    assert torch.equal(got, expected)
