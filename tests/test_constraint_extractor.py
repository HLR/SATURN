"""Tests for vision_agents.multiview.constraint_extractor.

Covers the regex pre-filter, the JSON parser (with markdown fence handling),
and the validator (camera-index range checks, malformed-record dropping).
The actual VLM call is mocked — these tests run with no network/GPU.
"""
import asyncio

import pytest

from saturn.planning.constraint_extractor import (
    extract_camera_constraints_async,
    question_has_pose_keywords,
    _parse_constraints_json,
    _validate_constraints,
)


# ----- Regex pre-filter -----

def test_prefilter_catches_rotation_phrases():
    assert question_has_pose_keywords("rotated 90 degrees clockwise")
    assert question_has_pose_keywords("Image 2 was turned 60 degrees")
    assert question_has_pose_keywords("Image 3 is the opposite direction of image 1")
    assert question_has_pose_keywords("All photos were taken from the same spot")


def test_prefilter_skips_plain_questions():
    assert not question_has_pose_keywords("Where is the wine bottle?")
    assert not question_has_pose_keywords("What color is the chair?")
    assert not question_has_pose_keywords("Pick the option closest to the table")


# ----- JSON parser -----

def test_parser_handles_markdown_fence():
    raw = """```json
{"constraints": [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90}]}
```"""
    parsed = _parse_constraints_json(raw)
    assert parsed == [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90}]


def test_parser_handles_bare_array():
    raw = '[{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": -45}]'
    parsed = _parse_constraints_json(raw)
    assert parsed == [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": -45}]


def test_parser_handles_object_wrapper():
    raw = '{"constraints": []}'
    assert _parse_constraints_json(raw) == []


def test_parser_handles_leading_prose():
    raw = 'Here are the constraints:\n{"constraints": [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90}]}'
    parsed = _parse_constraints_json(raw)
    assert parsed == [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90}]


def test_parser_returns_empty_on_garbage():
    assert _parse_constraints_json("") == []
    assert _parse_constraints_json("not json at all") == []


# ----- Validator -----

def test_validator_drops_out_of_range_indices():
    bad = [{"type": "rotation", "from_cam": 5, "to_cam": 1, "yaw": 90}]
    assert _validate_constraints(bad, num_cameras=3) == []


def test_validator_drops_self_constraint():
    bad = [{"type": "rotation", "from_cam": 1, "to_cam": 1, "yaw": 90}]
    assert _validate_constraints(bad, num_cameras=3) == []


def test_validator_drops_malformed_rotation():
    bad = [{"type": "rotation", "from_cam": "foo", "to_cam": 1, "yaw": 90}]
    assert _validate_constraints(bad, num_cameras=3) == []


def test_validator_dedupes_and_sorts_same_position_cams():
    rec = [{"type": "same_position", "cams": [2, 0, 1, 1]}]
    out = _validate_constraints(rec, num_cameras=3)
    assert out == [{"type": "same_position", "cams": [0, 1, 2]}]


def test_validator_keeps_same_position():
    """Extracted same-position groups are applied (K = K_r u K_p)."""
    recs = [
        {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90},
        {"type": "same_position", "cams": [0, 1, 2]},
    ]
    out = _validate_constraints(recs, num_cameras=3)
    assert [r["type"] for r in out] == ["rotation", "same_position"]


def test_validator_drops_singleton_same_position():
    rec = [{"type": "same_position", "cams": [0]}]
    assert _validate_constraints(rec, num_cameras=3) == []


def test_validator_accepts_well_formed_records(monkeypatch):
    recs = [
        {"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90},
        {"type": "rotation", "from_cam": 0, "to_cam": 2, "yaw": 180},
        {"type": "same_position", "cams": [0, 1, 2]},
    ]
    out = _validate_constraints(recs, num_cameras=3)
    assert len(out) == 3
    assert out[0]["yaw"] == 90.0 and out[0]["axis"] == "up"
    assert out[2]["cams"] == [0, 1, 2]


# ----- End-to-end with a fake VLM -----

def test_extract_skips_when_no_keywords():
    """If the regex pre-filter rejects the question, no VLM call is made
    (we set up a fake VLM that would fail if called)."""
    async def fake_vlm(images, text, **kw):
        raise AssertionError("VLM should not be called when prefilter fails")

    out = asyncio.run(
        extract_camera_constraints_async(
            "Where is the bottle?", images=[], vlm_generate=fake_vlm,
        )
    )
    assert out == []


def test_extract_calls_vlm_and_parses():
    """When the prefilter passes, the VLM is called and its response parsed."""
    async def fake_vlm(images, text, **kw):
        return ('{"constraints": [{"type": "rotation", "from_cam": 0, '
                '"to_cam": 1, "yaw": 90}]}')

    out = asyncio.run(
        extract_camera_constraints_async(
            "Image 2 was rotated 90 degrees clockwise from image 1",
            images=[],
            vlm_generate=fake_vlm,
            num_cameras=2,
        )
    )
    assert out == [{
        "type": "rotation",
        "from_cam": 0,
        "to_cam": 1,
        "yaw": 90.0,
        "axis": "up",
    }]


def test_extract_returns_empty_on_vlm_exception():
    async def fake_vlm(images, text, **kw):
        raise RuntimeError("simulated VLM failure")

    out = asyncio.run(
        extract_camera_constraints_async(
            "Image 2 was rotated 90 degrees from image 1",
            images=[],
            vlm_generate=fake_vlm,
        )
    )
    assert out == []
