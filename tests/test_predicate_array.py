"""PredicateArray: ``first_person`` scores are numpy-identical AND composable.

``anchor.first_person.<dir>`` is a PredicateArray: it behaves as the plain
ndarray of scores in numpy code and is also callable, so
``rect("x1") & view.first_person.left("x1")`` composes in the algebra.
"""

import os
import sys

import numpy as np
import pytest
import torch

from saturn.soft_logic import PredicateArray, ProbabilisticTensor, and_op, or_op
from saturn.scene.scene import Scene

sys.path.insert(0, os.path.dirname(__file__))
from test_frame_first_api import _make_camera, _make_object  # noqa: E402

VALUES = np.array([0.1, 0.9, 0.3, 0.9, 0.0])

# Usage patterns found in the cached program corpus (~40k programs).
NUMPY_EXPRS = [
    "X[1]", "X[np.int64(1)]", "X.max()", "X.argmax()", "np.max(X)", "X*2", "X>0.5", "float(X[1])",
    "sum(X)", "X[1]+X[2]", "sorted(range(5), key=lambda i: X[i])", 'f"{X.max():.3f}"', "X.tolist()",
    "np.argmax(X)", "X.mean()", "X[:3].argmax()", "np.round(X, 2)", "X-X", "-X", "abs(X)", "X/X.sum()",
    "np.where(X>0.5)[0]", "list(X)", "len(X)", "np.clip(X, 0, .5)", "(X>0.5).any()", "max(X)", "X.sum()",
    "np.exp(X)", "X@X", "X.std()", "np.argsort(X)", "X.argsort()[::-1]", "repr(X[1])", "X[:3].max()",
    "X[[0, 2]].tolist()", "X[X > 0.2].tolist()", "X.shape", "X.dtype",
]


@pytest.mark.parametrize("expr", NUMPY_EXPRS)
def test_numpy_behaviour_is_identical(expr):
    plain = eval(expr.replace("X", "a"), {"np": np, "a": VALUES.copy()})
    wrapped = eval(expr.replace("X", "a"), {"np": np, "a": PredicateArray(VALUES.copy())})
    if isinstance(wrapped, PredicateArray):  # only slices/copies keep the subclass
        assert type(plain) is np.ndarray and np.array_equal(plain, np.asarray(wrapped))
    else:
        # reading one entity gives a Degree: a np.float64 subclass that also supports & | ~
        assert type(plain) is type(wrapped) or (np.isscalar(plain) and isinstance(wrapped, type(plain)))
        if isinstance(plain, np.ndarray):
            assert np.array_equal(plain, wrapped)
        else:
            assert plain == wrapped


def test_is_an_ndarray():
    assert isinstance(PredicateArray(VALUES), np.ndarray)


def test_call_lifts_to_named_tensor():
    t = PredicateArray(VALUES)("x1")
    assert isinstance(t, ProbabilisticTensor) and t.vars == ["x1"]
    assert torch.allclose(t.tensor.double().cpu(), torch.tensor(VALUES))


def test_and_with_tensor_both_orders():
    other = ProbabilisticTensor(torch.tensor([0.5, 0.5, 0.5, 0.2, 1.0]))
    left = PredicateArray(VALUES)
    expected = np.minimum(VALUES, [0.5, 0.5, 0.5, 0.2, 1.0])
    for joint in (left & other("x1"), other("x1") & left, left("x1") & other("x1")):
        assert np.allclose(joint.tensor.double().cpu().numpy(), expected)


def test_and_with_pairwise_then_exists_iota():
    rel = ProbabilisticTensor(torch.rand(5, 5), vars=["x1", "x2"])
    ref = ProbabilisticTensor(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    target = (PredicateArray(VALUES)("x1") & rel("x1", "x2") & ref("x2")).exists("x2").iota("x1")
    assert target.vars == ["x1"] and target.tensor.shape == (5,)


def test_invert_and_pt_only_methods():
    left = PredicateArray(VALUES)
    assert np.allclose((~left).tensor.double().cpu().numpy(), 1 - VALUES)
    assert int(left.iota("x1").argmax()) == 1
    assert float(left.exists().tensor) == pytest.approx(0.9)


def test_numpy_methods_win_over_same_named_tensor_methods():
    left = PredicateArray(VALUES)
    assert type(left.max()) is np.float64 and type(left.argmax()) is np.int64


def test_and_op_or_op_accept_predicate_array():
    other = ProbabilisticTensor(torch.tensor([0.5, 0.5, 0.5, 0.2, 1.0]))
    left = PredicateArray(VALUES)
    assert np.allclose(and_op(left, other).tensor.double().cpu().numpy(),
                       np.minimum(VALUES, [0.5, 0.5, 0.5, 0.2, 1.0]))
    assert np.allclose(or_op(left, other).tensor.double().cpu().numpy(),
                       np.maximum(VALUES, [0.5, 0.5, 0.5, 0.2, 1.0]))


def test_lift_builds_the_tensor_class():
    import saturn.soft_logic as sl
    assert type(PredicateArray(VALUES)("x1")) is sl.ProbabilisticTensor
    joint = PredicateArray(VALUES) & sl.ProbabilisticTensor(torch.ones(5))("x1")
    assert np.allclose(joint.tensor.double().cpu().numpy(), VALUES)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        PredicateArray(VALUES).no_such_thing


@pytest.fixture
def scene():
    cameras = [_make_camera(position=[0, 0, 0], forward=[0, 0, 1], cam_id=0)]
    objects = [_make_object(0, center=[0, 0, 5]), _make_object(1, center=[3, 0, 5]),
               _make_object(2, center=[-3, 0, 5])]
    return Scene(objects=objects, cameras=cameras, images=[None])


def _skip_unless_wired(view):
    # saturn.predicates.frame switches first_person to PredicateArray separately;
    # these turn on automatically once it does.
    if not isinstance(view.first_person.left, PredicateArray):
        pytest.skip("first_person does not return PredicateArray yet")


def test_first_person_returns_predicate_array(scene):
    view = scene._frame(at=scene.cameras[0])
    _skip_unless_wired(view)
    for acc in (view.first_person.left, view.first_person("left"), view.first_person["left"]):
        assert isinstance(acc, PredicateArray) and acc.shape == (4,)


def test_first_person_composes_in_the_algebra(scene):
    """``rect & first_person.<dir>(var)`` builds a formula over the variable."""
    view = scene._frame(at=scene.cameras[0])
    _skip_unless_wired(view)
    is_obj = ProbabilisticTensor(torch.tensor([1.0, 1.0, 1.0, 0.0]))
    picked = (is_obj("x1") & view.first_person.left("x1")).iota("x1").argmax()
    assert picked == 2  # the lamp at (-3, 0, 5) is the leftmost object
    near_desk = (view.first_person.right("x1") & scene.closeness("x1", "x2")
                 & ProbabilisticTensor(torch.tensor([1.0, 0, 0, 0]))("x2")).exists("x2")
    assert near_desk.tensor.shape == (4,)
