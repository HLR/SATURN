"""Tests for the program-style planner prompt and its output parsing.

Mocks the VLM so we can exercise the prompt + parsing without hitting a
live server.
"""
import os
import pytest

from saturn.planning.query_planner import (
    QueryPlanner,
    _PLANNER_CACHE_VERSION,
)


class _FakeVLM:
    """Captures the prompt sent + returns a configurable canned response."""
    def __init__(self, reply: str):
        self.reply = reply
        self.last_prompt = None
        self.last_images = None

    def _query_thinking(self, images, prompt, max_new_tokens=None):
        self.last_images = images
        self.last_prompt = prompt
        return ("", self.reply)


@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "cache.json")


def _make_planner(vlm, cache_path):
    return QueryPlanner(
        vl_model=vlm, cache_path=cache_path, write_cache=False,
        max_new_tokens=512, verbose=False,
    )


# --- prompt ------------------------------------------------------------


def test_program_invokes_program_prompt(tmp_cache):
    fake = _FakeVLM(
        '{"program_sketch":"Ground the chair.","object_groundings":[{"phrase":"the chair","cam_id":0,"is_region":false,"unique":true}],"skipped_phrases":[]}'
    )
    p = _make_planner(fake, tmp_cache)
    out = p.clarify("Where is the chair?", [None])
    assert out is not None
    # Program prompt mentions scene.objects[i] in the data-source section.
    assert "scene.objects[i]" in fake.last_prompt
    assert "scene.cameras[k]" in fake.last_prompt


# --- parse compatibility ------------------------------------------------



def test_parse_program_shape_uses_sketch_as_reasoning():
    raw = (
        '{"program_sketch":"Ground the chair.",'
        '"object_groundings":[{"phrase":"the chair","cam_id":0,"is_region":false}],'
        '"skipped_phrases":[]}'
    )
    out = QueryPlanner.parse(raw)
    # program_sketch → reasoning so downstream code that reads .reasoning works.
    assert out["reasoning"] == "Ground the chair."
    assert out["object_groundings"][0]["phrase"] == "the chair"


def test_parse_program_shape_with_empty_sketch_yields_empty_reasoning():
    raw = '{"program_sketch":"","object_groundings":[]}'
    out = QueryPlanner.parse(raw)
    assert out["reasoning"] == ""
    assert out["object_groundings"] == []



# --- cache version ------------------------------------------------------


def test_cache_version_is_current():
    """Every planner-output change bumps the cache version so stale entries are
    re-planned (version 38: groundings carry the paired `image` field)."""
    assert _PLANNER_CACHE_VERSION == 38
