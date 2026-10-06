"""The per-sample debug report (saturn/reports)."""

import json
import re

from saturn.codegen.generator import CodeGenerator
from saturn.planning.query_planner import QueryPlanner
from saturn.reports import sections as S
from saturn.reports.helpers import load_scene_for_sample

GROUNDINGS = [{"phrase": "the red chair", "description": "a red chair", "role": "anchor"}]


def _ctx(tmp_path, sample, **kw):
    args = dict(
        result_json=str(tmp_path / "run" / "r.json"), prompt_template=None,
        program_cache=None, planner_cache=None,
        image_dir=str(tmp_path), scene_dir=None,
    )
    args.update(kw)
    return S.build_context(sample, 0, 1, **args)


def _write_scene(root, sid):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{sid}.json").write_text(json.dumps({"objects": [], "cameras": []}))


# the program cache section must find the key codegen really used.
def test_program_cache_hit_on_recorded_key(tmp_path):
    sample = {"id": "s1", "query": "Q?", "program_cache_key": "prog-v2:abc"}
    html = "".join(S.section_program_cache(_ctx(tmp_path, sample, program_cache={"prog-v2:abc": "print(1)"})))
    assert "CACHE HIT" in html and "recorded in result" in html


def test_program_cache_hit_on_content_addressed_key(tmp_path):
    ran = CodeGenerator._finalize("x = 1")
    cache = {"prog-v2:abc": "x = 1", "prog-v2:other": "y = 2"}
    sample = {"id": "s1", "query": "Q?", "raw_llm_code": ran}
    html = "".join(S.section_program_cache(_ctx(tmp_path, sample, program_cache=cache)))
    assert "CACHE HIT" in html and "matched by program text" in html


def test_program_cache_miss(tmp_path):
    sample = {"id": "s1", "query": "Q?", "raw_llm_code": "z = 3"}
    cache = {"prog-v2:abc": "x = 1"}
    html = "".join(S.section_program_cache(_ctx(tmp_path, sample, program_cache=cache)))
    assert "CACHE MISS" in html


# the assembled prompt must fill the placeholders codegen fills.
def test_assembled_prompt_fills_object_groundings(tmp_path):
    sample = {"id": "s1", "query": "Q?", "planner": {
        "object_groundings": GROUNDINGS}}
    tpl = "{query}\n{object_groundings_block}\n{clarified_query_block}"
    html = "".join(S.section_assembled_prompt(_ctx(tmp_path, sample, prompt_template=tpl)))
    full = html.split("Full Prompt")[1]
    assert "{object_groundings_block}" not in full
    expected = QueryPlanner.format_groundings_block(GROUNDINGS)
    assert S.esc(expected) in full
    assert "{clarified_query_block}" not in full


def test_scene_facts_block_is_rebuilt(tmp_path):
    _write_scene(tmp_path / "run" / "scenes", "s1")
    sample = {"id": "s1", "query": "Q?"}
    assert _ctx(tmp_path, sample).scene_facts_block


# an explicit scene_dir must not fall through to a sibling run.
def test_explicit_scene_dir_does_not_use_sibling_run(tmp_path):
    _write_scene(tmp_path / "exp" / "modelA" / "scenes", "s1")
    (tmp_path / "exp" / "modelB" / "scenes").mkdir(parents=True)
    data, path = load_scene_for_sample(
        "s1", str(tmp_path / "exp" / "modelB" / "scenes"), str(tmp_path / "exp" / "modelB" / "r.json"))
    assert data is None and path is None


def test_autodetected_sibling_scene_is_flagged(tmp_path):
    _write_scene(tmp_path / "exp" / "modelA" / "scenes", "s1")
    (tmp_path / "exp" / "modelB").mkdir(parents=True)
    ctx = _ctx(tmp_path, {"id": "s1", "query": "Q?"},
               result_json=str(tmp_path / "exp" / "modelB" / "r.json"))
    assert "modelA" in ctx.scene_path
    assert "SCENE FROM ANOTHER RUN" in "".join(S.section_scene_3d(ctx))


def test_own_scene_not_flagged(tmp_path):
    _write_scene(tmp_path / "run" / "scenes", "s1")
    ctx = _ctx(tmp_path, {"id": "s1", "query": "Q?"})
    assert "SCENE FROM ANOTHER RUN" not in "".join(S.section_scene_3d(ctx))


# REGION badge background must be valid CSS.
def test_region_badge_css(tmp_path):
    sample = {"id": "s1", "query": "Q?", "planner": {
        "object_groundings": [{"phrase": "p", "is_region": True}]}}
    html = "".join(S.section_planned_vs_detected(_ctx(tmp_path, sample)))
    assert re.search(r'background:#a855f7">REGION', html)
    assert "background::" not in html


# a grounding score that is not a number renders as text (as the planner
# section does) in the Detected Objects and Planned-vs-Detected sections.
def _ctx_with_objects(tmp_path, objects, grounding_results, **kw):
    sample = {"id": "s1", "query": "Q?", "grounding_results": grounding_results,
              "planner": {"object_groundings": [{"phrase": gr["phrase"]} for gr in grounding_results]}}
    ctx = _ctx(tmp_path, sample, **kw)
    ctx.scene_data = {"objects": objects, "cameras": []}
    return ctx


def test_non_numeric_grounding_score_renders(tmp_path):
    grs = [{"phrase": "a", "score": "n/a", "obj_indices": [0]},
           {"phrase": "b", "score": None, "obj_indices": [1]},
           {"phrase": "c", "score": 0.5, "obj_indices": [2]}]
    objects = [{"label": x} for x in "abc"]
    ctx = _ctx_with_objects(tmp_path, objects, grs)
    for section in (S.section_detected_objects, S.section_planned_vs_detected, S.section_planner):
        html = "".join(section(ctx))
        assert "n/a" in html and "None" in html and "0.500" in html


# a detected object whose metadata is None renders (source shown as "?").
def test_detected_object_with_null_metadata(tmp_path):
    ctx = _ctx_with_objects(tmp_path, [{"label": "chair", "metadata": None}], [])
    html = "".join(S.section_detected_objects(ctx))
    assert "<strong>chair</strong></td><td>?</td>" in html


# the overlay skips a per-view box that is not four numbers (the table lists it).
def test_overlay_skips_malformed_bbox(tmp_path):
    from PIL import Image

    Image.new("RGB", (64, 48), (0, 0, 0)).save(tmp_path / "s1_0.png")
    objects = [{"label": "bad3", "per_view_bboxes": {"0": [1, 2, 3]}},
               {"label": "bad5", "per_view_bboxes": {"0": [1, 2, 3, 4, 5]}},
               {"label": "flip", "per_view_bboxes": {"0": [30, 2, 10, 4]}},
               {"label": "ok", "per_view_bboxes": {"0": [10, 10, 40, 30]}}]
    img = Image.new("RGB", (64, 48), (0, 0, 0))
    S._draw_bboxes(img, objects, 0)
    assert img.getpixel((10, 20)) == S._OVERLAY_COLORS[3]   # the valid box is drawn
    assert img.getpixel((2, 3)) == (0, 0, 0)                # the malformed ones are not
    html = "".join(S.section_detected_objects(_ctx_with_objects(tmp_path, objects, [])))
    assert "Per-View Bbox + Mask Overlays" in html and "<img" in html
    assert "v0:[1, 2, 3]" in html                           # the table still lists the box


def test_bbox_with_non_numeric_coordinate(tmp_path):
    from PIL import Image

    Image.new("RGB", (64, 48), (0, 0, 0)).save(tmp_path / "s1_0.png")
    objects = [{"label": "none", "per_view_bboxes": {"0": [1, None, 3, 4]}}]
    html = "".join(S.section_detected_objects(_ctx_with_objects(tmp_path, objects, [])))
    assert "v0:[1, None, 3, 4]" in html and "<img" in html


# load_scene_for_sample's annotation matches the (data, path) pair it returns.
def test_load_scene_for_sample_return_annotation():
    import typing

    hint = typing.get_type_hints(load_scene_for_sample)["return"]
    assert typing.get_origin(hint) is tuple and len(typing.get_args(hint)) == 2


# MindCube images are looked up where the loader reads them ($MINDCUBE_DATA_ROOT).
def test_mindcube_images_follow_data_root_env(tmp_path, monkeypatch):
    from saturn.reports import helpers

    root = tmp_path / "mc"
    (root / "raw").mkdir(parents=True)
    (root / "imgs").mkdir()
    rel = ["imgs/front_1.png", "imgs/left_2.png"]
    for r in rel:
        (root / r).write_bytes(b"x")
    rec = {"id": "among_test_q1", "images": rel}
    (root / "raw" / "MindCube_tinybench.jsonl").write_text(json.dumps(rec) + "\n")
    monkeypatch.setenv("MINDCUBE_DATA_ROOT", str(root))
    monkeypatch.setattr(helpers, "_MINDCUBE_IMAGE_INDEX", None)
    got = helpers.resolve_image_paths_for_sample("among_test_q1", str(tmp_path / "nowhere"))
    assert got == [root / r for r in rel]


def test_mmsi_report_images_follow_data_root(monkeypatch, tmp_path):
    # reports look for MMSI images where the loader extracts them
    from saturn.reports.helpers import default_mmsi_image_dir
    monkeypatch.setenv("MMSI_DATA_ROOT", str(tmp_path))
    assert default_mmsi_image_dir() == str(tmp_path / "images")
    monkeypatch.delenv("MMSI_DATA_ROOT")
    assert default_mmsi_image_dir().endswith("data/mmsi/images")
