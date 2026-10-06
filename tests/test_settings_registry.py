"""configs/settings.json lists every environment variable SATURN reads."""
import glob
import re

import pytest

from saturn import settings

# the files that ship: the release's import closure (same function the build uses)
import importlib.util
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_bs = os.path.join(_root, "scripts", "build_release.py")
if os.path.exists(_bs):                      # dev repo: exactly the files that ship
    _spec = importlib.util.spec_from_file_location("_build_release", _bs)
    _br = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_br)
    SHIPPED = [f for f in _br.import_closure() if f != "saturn/settings.py"]
else:                                        # inside the release: everything present is shipped
    SHIPPED = [f for f in glob.glob(os.path.join(_root, "**", "*.py"), recursive=True)
               if not any(part in f for part in ("/tools/", "/.venv/", "/tests/")) and not f.endswith("settings.py")]

def test_no_direct_environ_reads_for_registered_vars():
    pat = re.compile(r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z_][A-Z0-9_]*)["\']|os\.environ\[\s*["\']([A-Z_][A-Z0-9_]*)["\']')
    offenders = [(f, m.group(1) or m.group(2)) for f in SHIPPED for m in pat.finditer(open(f, errors="ignore").read()) if (m.group(1) or m.group(2)) in settings._R]
    assert not offenders, offenders

def test_every_read_var_is_registered():
    pat = re.compile(r'\benv\(\s*["\']([A-Z_][A-Z0-9_]*)["\']')
    unknown = {(f, v) for f in SHIPPED for v in pat.findall(open(f, errors="ignore").read()) if v not in settings._R}
    assert not unknown, unknown

def test_defaults_and_tags_are_well_formed():
    for name, (default, tag, doc) in settings.registry().items():
        assert default is None or isinstance(default, str), name
        assert tag in ("runtime", "debug"), name
        assert doc and len(doc) > 10, name

def test_env_override_and_default(monkeypatch):
    monkeypatch.delenv("SAPY_CODEGEN_SEED", raising=False)
    assert settings.env("SAPY_CODEGEN_SEED") == "0"
    monkeypatch.setenv("SAPY_CODEGEN_SEED", "1")
    assert settings.env("SAPY_CODEGEN_SEED") == "1"
    with pytest.raises(KeyError):
        settings.env("NOT_A_SETTING")


def test_unknown_sapy_variables_are_reported(monkeypatch):
    monkeypatch.setenv("SAPY_FUSION_RULE", "legacy")   # not a registered setting
    monkeypatch.setenv("SAPY_CODEGEN_SEED", "1")       # a real setting
    assert "SAPY_FUSION_RULE" in settings.unknown_env()
    assert "SAPY_CODEGEN_SEED" not in settings.unknown_env()


def test_settings_file_entries_are_complete():
    import json
    from saturn.settings import SETTINGS_FILE
    entries = json.loads(SETTINGS_FILE.read_text())
    assert entries, "configs/settings.json is empty"
    for name, entry in entries.items():
        assert set(entry) == {"default", "group", "help"}, name
        assert entry["group"] in {"runtime", "debug"}, name
        assert isinstance(entry["default"], str) and entry["help"], name
