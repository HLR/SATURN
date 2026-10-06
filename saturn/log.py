"""Logging for saturn: one logger per module under the "saturn" namespace, configured once by an entry point."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("saturn") else f"saturn.{name}")


def configure(level: str | int | None = None, stream=sys.stderr) -> None:
    """Idempotent. Level from SATURN_LOG_LEVEL (registered setting) unless given. Format: time name level message."""
    if level is None:
        from saturn.settings import env

        level = env("SATURN_LOG_LEVEL") or "INFO"
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO
    root = logging.getLogger("saturn")
    root.setLevel(level)
    for h in root.handlers:
        if getattr(h, "_saturn_handler", False):
            h.setLevel(level)
            return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(level)
    handler._saturn_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def progress(msg: str) -> None:
    """User-facing progress line (accuracy tallies, 'Results will be saved to', per-sample verdict lines that the
    shell scripts grep). Goes to STDOUT, flushed, unformatted — this is the one sanctioned print."""
    print(msg, flush=True)
