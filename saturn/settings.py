"""Every environment variable SATURN reads, with its default and purpose.

The settings live in configs/settings.json: per variable, its default, a group
(runtime = deployment/paths/keys/seeds; debug = diagnostics) and a one-line
description. Code never calls ``os.environ`` for these directly; it calls
``env("NAME")``, which returns the environment value or the default from that
file (always a string; call sites parse).
``tests/test_settings_registry.py`` fails if code reads a variable that is not listed.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = REPO_ROOT / "configs" / "settings.json"
_PATH_DEFAULTS = {"SATURN_TOOLS_DIR"}   # defaults given relative to the repository root


def _load(path: Path = SETTINGS_FILE) -> dict:
    """{name: (default, group, help)} from the settings file."""
    out = {}
    for name, entry in json.loads(path.read_text()).items():
        default = entry["default"]
        if name in _PATH_DEFAULTS and default and not os.path.isabs(default):
            default = str(REPO_ROOT / default)
        out[name] = (default, entry["group"], entry["help"])
    return out


_R: dict = _load()


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Environment value for a registered variable, else its registered default."""
    if name not in _R:
        raise KeyError(f"{name} is not a registered setting (see configs/settings.json)")
    reg_default = _R[name][0]
    return os.environ.get(name, reg_default if default is None else default)


def registry() -> dict:
    return dict(_R)


def unknown_env() -> list:
    """SAPY_* variables set in the environment that no setting reads (a typo or a removed option)."""
    return sorted(k for k in os.environ if k.startswith("SAPY_") and k not in _R)
