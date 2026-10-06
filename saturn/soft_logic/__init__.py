"""
saturn.soft_logic
=================
Import point for the probabilistic tensor algebra.

    from saturn.soft_logic import ProbabilisticTensor, and_op, or_op
"""

from saturn.soft_logic.tensor import (
    ProbabilisticTensor,
    and_op,
    or_op,
    zero_superdiag_trailing,
    zero_any_equal_trailing,
    serializable,
)
from .predicate_array import PredicateArray

__all__ = [
    "ProbabilisticTensor",
    "and_op",
    "or_op",
    # numpy score vector that also composes in the algebra (first_person.<dir>)
    "PredicateArray",
    "zero_superdiag_trailing",
    "zero_any_equal_trailing",
    "serializable",
]
