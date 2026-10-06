"""--programs_by_id: programs exported from a results file replay by sample id."""
import json
import subprocess
import sys
from pathlib import Path

from saturn.pipeline.models import load_pinned_programs
from saturn.pipeline.template import CODE_TEMPLATE

ROOT = Path(__file__).resolve().parent.parent


def test_export_unwraps_the_header_and_load_keys_by_string_id(tmp_path):
    body = "anchor = scene.frame(position=camera(1).position, orientation=camera(1).orientation)\nreturn 'A'"
    wrapped = CODE_TEMPLATE.format(code=body.replace("\n", "\n    "))
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"results": [{"id": 7, "program_code": wrapped}, {"id": "q2", "program_code": None}]}))
    out = tmp_path / "programs.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_programs.py"), str(out), str(results)], check=True)
    programs = load_pinned_programs(str(out))
    assert programs == {"7": body}          # int ids become strings; empty programs are skipped


def test_no_path_means_no_replay():
    assert load_pinned_programs(None) is None


def test_export_reads_the_two_line_header(tmp_path):
    body = "return 'B'"
    program = ("def logic_executor(query, score_fn, query_fn, scene, images, history):\n"
           "    _h = formula_helpers(score, scene)\n"
           "    camera, view, position, answer, holds, dir_label, choice = _h[\"camera\"], _h[\"view\"], _h[\"position\"], "
           "_h[\"answer\"], _h[\"holds\"], _h[\"dir_label\"], _h[\"choice\"]\n    " + body + "\n")
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"id": "among_1", "program_code": program}]))
    out = tmp_path / "programs.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_programs.py"), str(out), str(results)], check=True)
    assert load_pinned_programs(str(out)) == {"among_1": body}
