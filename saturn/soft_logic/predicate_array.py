"""PredicateArray: a 1D predicate score vector that is BOTH a numpy array and a
member of the soft-logic algebra.

``anchor.first_person.<dir>`` returns a PredicateArray, so a program can both
index or ``argmax`` it and write ``rect("x1") & left("x1")``, ``~left`` or
``left.iota("x1")``, as with the other predicates (pairwise relations,
``facing``, ``score()``, ``closeness``), which return ``ProbabilisticTensor``.

This subclass keeps every numpy behaviour identical to a plain ndarray
(indexing returns numpy scalars, arithmetic / comparisons / reductions /
``np.*`` return what a plain ndarray returns) and adds the algebra entry
points by lifting to ``ProbabilisticTensor(vars=["x1"])``:

    left("x1")          -> ProbabilisticTensor over x1
    left & other        -> left("x1") & other
    ~left               -> 1 - left
    left.iota("x1"), left.exists(), left.topk(k=2), left.count(), ...

The lifted tensor goes through the normal ``ProbabilisticTensor`` constructor,
like every other predicate. Only slicing / copying keeps the subclass, so
``left[:K]`` can still enter the algebra.

Reading one entity, ``left[i]`` (or ``left(i)``, the constant form), gives a
``Degree``: a numpy float64 that also composes with ``& | ~`` (min, max, 1-x),
so ``left[i] & below[i]`` is the truth of "i is left and below". Arrays built by
``anchor.first_person`` can also be read at a 3D point, ``front[p]``: the same
direction score for the location ``p`` (e.g. where an object was in one image).
"""

from __future__ import annotations

import numpy as np
import torch

# ProbabilisticTensor methods ndarray lacks; forwarded to the lifted tensor.
# Names both define (max, argmax, sum, ...) keep numpy semantics.
_PT_ONLY = frozenset({"iota", "exists", "forall", "assign", "topk", "count",
                      "normalize", "implies", "not_implies", "mask", "tensor", "vars"})


class PredicateArray(np.ndarray):
    """Score vector over entities (length K + C) usable as numpy AND as a predicate."""

    def __new__(cls, arr):
        return np.asarray(arr, dtype=float).view(cls)

    # Results of ufuncs / reductions come back exactly as for a plain ndarray
    # (0-d -> numpy scalar, else base ndarray): arithmetic output is a number,
    # not a predicate.
    def __array_wrap__(self, obj, context=None, return_scalar=False):
        arr = np.asarray(obj).view(np.ndarray)
        return arr[()] if arr.ndim == 0 else arr

    def argsort(self, *args, **kwargs):  # an index array is not a predicate
        return np.asarray(self).argsort(*args, **kwargs)

    # ---- algebra --------------------------------------------------------
    def to_tensor(self, var: str = "x1"):
        # the same tensor class every other predicate uses
        import saturn.soft_logic as _sl
        return _sl.ProbabilisticTensor(torch.from_numpy(np.array(self, dtype=np.float64)), vars=[var])

    def __call__(self, *vars):
        if all(isinstance(v, str) for v in vars):
            if len(vars) > 1:
                raise TypeError(
                    f"this predicate scores ONE entity relative to the anchor; called with {len(vars)} "
                    f"variables {list(vars)!r}. Use pred(\"x1\"); for a relation between two entities use "
                    f"anchor.third_person.<dir>(\"x1\", \"x2\").")
            return self.to_tensor(*(vars or ("x1",)))
        if len(vars) == 1 and _is_point(vars[0]):
            import saturn.soft_logic as _sl
            return _sl.ProbabilisticTensor(torch.tensor(float(self[vars[0]]), dtype=torch.float64), vars=[])
        # entity constants (camera(2), an object index): the formula's value at that entity
        for v in vars:
            self._check_defined(v)
        return self.to_tensor()(*vars)

    def __getitem__(self, key):
        if _is_point(key):
            scorer = getattr(self, "_point_scorer", None)
            if scorer is None:
                raise TypeError(
                    "this score vector is indexed by entity (an object index or camera(N)); only "
                    "anchor.first_person.<dir> can be read at a 3D point.")
            return Degree(scorer(np.asarray(key, dtype=float).reshape(1, 3))[0])
        self._check_defined(key)
        out = super().__getitem__(key)
        return Degree(out) if type(out) is np.float64 else out

    def _check_defined(self, key) -> None:
        """A program reading camera(N) where the array has no value for it
        (``_undefined``: index -> reason, set by the predicate that built the
        array) gets an error, not a 0: a silent 0 for every option makes max()
        return the first option. Plain integer indexing keeps numpy semantics."""
        undefined = getattr(self, "_undefined", None)
        if not undefined or not hasattr(key, "cam"):  # only camera(N) references
            return
        if int(key) in undefined:
            raise ValueError(undefined[int(key)])

    def __and__(self, other):
        return self.to_tensor() & _lift(other)

    def __rand__(self, other):
        return _lift(other) & self.to_tensor()

    def __or__(self, other):
        return self.to_tensor() | _lift(other)

    def __ror__(self, other):
        return _lift(other) | self.to_tensor()

    def __invert__(self):
        return ~self.to_tensor()

    def __getattr__(self, name):
        # Only reached for names ndarray does not define. Dunder/private
        # probes (numpy internals) must fail normally.
        if name in _PT_ONLY:
            return getattr(self.to_tensor(), name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


def _lift(x):
    return x.to_tensor() if isinstance(x, PredicateArray) else x


def _is_point(key) -> bool:
    """A 3D location (a float 3-vector), as opposed to an entity index / index array."""
    if isinstance(key, (np.ndarray, list, tuple)) and not isinstance(key, PredicateArray):
        try:
            a = np.asarray(key)
        except Exception:
            return False
        return a.shape == (3,) and a.dtype.kind == "f"
    return False


def _is_truth(x) -> bool:
    return isinstance(x, (bool, int, float, np.number, np.bool_))


class Degree(np.float64):
    """One entity's truth degree, ``pred[i]``: a numpy float64 in every numeric
    use, and a closed formula for the connectives: ``&`` = min, ``|`` = max,
    ``~`` = 1 - x (the same t-norm the algebra uses). Combined with a formula
    it lifts into the algebra; ``.exists()`` of a closed formula is its value."""

    def _pt(self):
        import saturn.soft_logic as _sl
        return _sl.ProbabilisticTensor(torch.tensor(float(self), dtype=torch.float64), vars=[])

    def __and__(self, other):
        if _is_truth(other):
            return Degree(min(float(self), float(other)))
        if hasattr(other, "vars") or hasattr(other, "to_tensor"):
            return self._pt() & other
        return NotImplemented

    def __rand__(self, other):
        if _is_truth(other):
            return Degree(min(float(self), float(other)))
        if hasattr(other, "vars") or hasattr(other, "to_tensor"):
            return other & self._pt()
        return NotImplemented

    def __or__(self, other):
        if _is_truth(other):
            return Degree(max(float(self), float(other)))
        if hasattr(other, "vars") or hasattr(other, "to_tensor"):
            return self._pt() | other
        return NotImplemented

    def __ror__(self, other):
        if _is_truth(other):
            return Degree(max(float(self), float(other)))
        if hasattr(other, "vars") or hasattr(other, "to_tensor"):
            return other | self._pt()
        return NotImplemented

    def __invert__(self):
        return Degree(1.0 - float(self))

    def exists(self, *vars):
        return self
