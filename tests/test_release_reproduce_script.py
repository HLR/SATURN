"""scripts/release_reproduce.sh exits non-zero, naming the runs, when a run's process exits with an error."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_reproduce.sh"


def _fake_release(tmp_path: Path, mmsi_exit: int) -> Path:
    """A release tree whose run_mmsi.sh exits with ``mmsi_exit``."""
    (tmp_path / "scripts").mkdir()
    runner = tmp_path / "scripts" / "run_mmsi.sh"
    runner.write_text(f"#!/usr/bin/env bash\nexit {mmsi_exit}\n")
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    return tmp_path


@pytest.mark.skipif(not SCRIPT.exists(), reason="script not present")
@pytest.mark.parametrize("child_exit, ok", [(0, True), (1, False)])
def test_exit_status_follows_the_runs(tmp_path, child_exit, ok):
    rel = _fake_release(tmp_path, child_exit)
    proc = subprocess.run(["bash", str(SCRIPT), "mmsi"], capture_output=True, text=True,
                          env={**os.environ, "REL": str(rel)}, timeout=60)
    assert (proc.returncode == 0) is ok
    assert ("All 2 runs exited with status 0." in proc.stdout) is ok
    assert "[mmsi_s0] exited with status %d" % child_exit in proc.stdout
    if not ok:
        assert "Runs that exited with an error: mmsi_s0 (status 1), mmsi_s1 (status 1)" in proc.stderr
