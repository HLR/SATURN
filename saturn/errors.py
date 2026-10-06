"""Exception vocabulary for saturn stage boundaries."""
from __future__ import annotations


class SaturnError(Exception):
    """Base class for saturn-specific errors."""


class StageError(SaturnError):
    """An error attributed to one pipeline stage for one sample."""

    stage: str = ""

    def __init__(self, message: str = "", *, stage: str | None = None, sample_id: str | None = None):
        super().__init__(message)
        if stage is not None:
            self.stage = stage
        self.sample_id = sample_id



class ExecutionError(StageError):
    stage = "execution"


class DegenerateBindingError(ExecutionError):
    """``assign()`` on a formula whose every joint assignment has zero truth.

    Raised instead of silently returning index 0 for every variable (the
    argmax of an all-zero tensor), which would look like a real answer.
    """
