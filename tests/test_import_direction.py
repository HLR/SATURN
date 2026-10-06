"""Pin the import direction of the ``saturn`` package.

The pure-geometry layers (``saturn.soft_logic``, ``saturn.scene``, ``saturn.predicates``)
must stay importable without dragging in the VLM / serving layers or the model-side
wheels (``transformers``, ``ray``, ``vllm``).  The checks run in *subprocesses* with a
clean interpreter because ``sys.modules`` is already polluted by the time the rest of
the suite runs.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = ["saturn.vlm", "saturn.serving", "transformers", "ray", "vllm"]


def _run(code):
    """Run ``code`` in a fresh interpreter rooted at the repo; return stdout."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


def test_geometry_layers_do_not_load_vlm_serving_or_model_libs():
    out = _run(
        f"""
        import sys
        import saturn.soft_logic
        import saturn.scene.scene
        import saturn.predicates.frame
        forbidden = {FORBIDDEN!r}
        loaded = sorted(m for m in sys.modules
                        if any(m == f or m.startswith(f + ".") for f in forbidden))
        print("LOADED:" + ",".join(loaded))
        """
    )
    loaded = out.strip().split("LOADED:")[-1].strip()
    assert loaded == "", "geometry layers pulled in forbidden modules: " + loaded


def test_bare_import_saturn_loads_no_submodule():
    out = _run(
        """
        import sys
        import saturn
        subs = sorted(m for m in sys.modules if m.startswith("saturn."))
        print("SUBS:" + ",".join(subs))
        print("SETTINGS:" + str("saturn.settings" in sys.modules))
        """
    )
    subs = out.split("SUBS:")[1].split("\n")[0].strip()
    assert subs == "", "import saturn eagerly loaded submodules: " + subs
    assert "SETTINGS:False" in out, "saturn.settings must not be auto-imported by `import saturn`"


def test_report_perception_importer_chain():
    """Diagnostic only (never fails): does saturn.scene.scene pull saturn.perception, and via what chain?"""
    out = _run(
        """
        import sys, builtins
        chain = {}
        stack = []
        _orig = builtins.__import__
        def _imp(name, globals=None, locals=None, fromlist=(), level=0):
            importer = (globals or {}).get("__name__", "?")
            if name not in sys.modules:
                chain.setdefault(name, importer)
            return _orig(name, globals, locals, fromlist, level)
        builtins.__import__ = _imp
        import saturn.scene.scene
        builtins.__import__ = _orig
        loaded = sorted(m for m in sys.modules if m.startswith("saturn.perception"))
        print("PERCEPTION_LOADED:" + ",".join(loaded))
        for m in loaded:
            path = [m]; cur = m
            seen = set()
            while cur in chain and chain[cur] not in seen and chain[cur] != "?":
                seen.add(cur); cur = chain[cur]; path.append(cur)
            print("CHAIN:" + " <- ".join(path))
        """
    )
    print(out)


def test_predicates_frame_imports_first():
    """``import saturn.predicates.frame`` must work in a fresh interpreter (no circular import via saturn.scene)."""
    out = _run(
        """
        import saturn.predicates.frame
        print("FRAME_FIRST_OK")
        """
    )
    assert "FRAME_FIRST_OK" in out


def test_scene_scene_does_not_load_reconstruction_vlm_or_serving():
    out = _run(
        """
        import sys
        import saturn.scene.scene
        bad = sorted(m for m in sys.modules
                     if m == "saturn.perception.reconstruction" or m.startswith("saturn.perception.reconstruction.")
                     or m == "saturn.vlm" or m.startswith("saturn.vlm.")
                     or m == "saturn.serving" or m.startswith("saturn.serving."))
        print("BAD:" + ",".join(bad))
        """
    )
    bad = out.strip().split("BAD:")[-1].strip()
    assert bad == "", "saturn.scene.scene loaded: " + bad


def test_scene_package_lazy_names_resolve():
    """Every name in ``saturn.scene.__all__`` resolves through the lazy re-export table."""
    out = _run(
        """
        import sys
        import saturn.scene as s
        eager = sorted(m for m in sys.modules if m.startswith("saturn.perception"))
        print("EAGER_PERCEPTION:" + ",".join(eager))
        for n in s.__all__:
            getattr(s, n)
        assert set(s.__all__) <= set(dir(s))
        print("RESOLVED:" + str(len(s.__all__)))
        """
    )
    assert "EAGER_PERCEPTION:\n" in out, "import saturn.scene eagerly loaded perception: " + out
    assert "RESOLVED:20" in out, out
