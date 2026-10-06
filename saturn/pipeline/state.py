"""Pre-execute -> execute hand-off state."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _PreExecState:
    """State produced by phase A (pre-execute) and consumed by phase B (execute).

    When ``early_return`` is True, ``result`` is a complete sample result and
    the execute phase MUST be skipped (e.g. missing question, no objects
    detected, code-gen failed). When False, all of ``clarification``, ``scene``,
    and ``code_snippet`` are populated and the execute phase will run.
    """
    early_return: bool
    result: Dict[str, Any]
    clarification: Optional[Dict[str, Any]] = None
    scene: Optional[Any] = None
    code_snippet: Optional[str] = None
