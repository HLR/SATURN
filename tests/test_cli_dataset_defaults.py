"""The CLI defaults equal the released configurations, per dataset (configs/benchmarks.json)."""
import pytest

from saturn.cli.args import parse_args


def test_shared_defaults():
    a = parse_args(["--dataset", "mmsi"])
    assert (a.code_gen_provider, a.code_gen_model_name, a.vlm_model_name) == (
        "deepseek", "deepseek-chat", "Qwen/Qwen3-VL-8B-Instruct")


@pytest.mark.parametrize("dataset,pose", [("mindcube-among", True), ("mindcube-rotation", True), ("mmsi", False),
                                          ("mmsi-MSR", False)])
def test_mindcube_and_mmsi(dataset, pose):
    a = parse_args(["--dataset", dataset])
    assert a.code_prompt == "prompts/vqa.txt"
    assert a.use_pose_constraints is pose


@pytest.mark.parametrize("dataset,stem", [("force3d-ref", "ref"), ("force3d-puzzle", "sag")])
def test_force3d_non_gt(dataset, stem):
    a = parse_args(["--dataset", dataset])
    assert a.code_prompt == f"prompts/force3d_{stem}.txt"
    assert a.program_cache == "cache/programs.json"   # no shipped programs: written fresh
    assert a.use_pose_constraints is False


def test_flags_override():
    a = parse_args(["--dataset", "mindcube-among", "--no-use_pose_constraints", "--code_prompt", "p.txt"])
    assert (a.use_pose_constraints, a.code_prompt) == (False, "p.txt")


def test_planner_flags_are_gone():
    # one planner path: the unified prompt plans every benchmark
    for flag in (["--use_planner"], ["--planner_prompt", "dataset"]):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "mmsi", *flag])


def test_benchmarks_file_rejects_unknown_keys(tmp_path):
    import json
    import pytest
    from saturn.cli.args import _load_benchmarks
    bad = tmp_path / "benchmarks.json"
    bad.write_text(json.dumps({"default": {"code_prompt": "p.txt"},
                               "benchmarks": {"x": {"code_prompt": "p.txt", "fov_falloff": True}}}))
    with pytest.raises(ValueError, match="fov_falloff"):
        _load_benchmarks(bad)
