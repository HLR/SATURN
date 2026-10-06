"""The object-motion worked examples in prompts/vqa.txt run on the engine and answer correctly.

The programs are read from the prompt itself, so an edit to an example that breaks it fails here.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_frame_first_api import _make_camera, _make_object  # noqa: E402

from saturn.pipeline.execute import execute_code  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402
from saturn.soft_logic import ProbabilisticTensor  # noqa: E402

PROMPT = os.path.join(os.path.dirname(__file__), "..", "prompts", "vqa.txt")


def _example(stem: str) -> str:
    """The program of the worked example whose question starts with ``stem``."""
    lines = open(PROMPT).read().split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"Q: {stem}"))
    body = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        body.append(line)
    return "\n".join(body)


class StubVL:
    """score(...) is 1.0 for the object whose label the question names."""

    def score_multiview(self, question, num_objects=1, type=None, scene=None, images=None, cam_id=None, **kw):
        v = torch.zeros(scene.objects_count + len(scene.cameras), dtype=torch.float64)
        for i, o in enumerate(scene.objects):
            if o.label in question.lower():
                v[i] = 1.0
        return ProbabilisticTensor(v)

    def query_multiview(self, question, **kw):
        return ""


def _scene(obj):
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0), _make_camera([1, 0, 0], [0, 0, 1], cam_id=1)]
    other = _make_object(1, [3, 0, 9])
    other.label = "box"
    return Scene(objects=[obj, other], cameras=cams, images=[None, None])


def _axes(scene):
    """Camera 1's right and front in world coordinates."""
    c = scene.cameras[0]
    a = scene.frame(position=c.position, orientation=c.orientation)
    return np.asarray(a._frame_right, float), np.asarray(a._frame_front, float)


@pytest.mark.parametrize("move, expected", [("left", "A"), ("toward", "B"), ("stay", "C")])
def test_translation_example(move, expected):
    ball = _make_object(0, [0, 0, 5], dims=(0.4, 0.4, 0.4))
    ball.label = "ball"
    scene = _scene(ball)
    right, front = _axes(scene)
    c0 = np.array([0.0, 0.0, 5.0])
    c1 = {"left": c0 - 1.5 * right, "toward": c0 - 1.5 * front, "stay": c0 + 0.03 * right}[move]
    ball.per_view_centers = {0: c0, 1: c1}
    ans, _, err = execute_code(_example("Between picture 1 and picture 2"), "where did the ball roll?",
                               StubVL(), scene, [None, None])
    assert err is None, err
    assert ans == expected


@pytest.mark.parametrize("yaw, expected", [(60, "A"), (-60, "B")])
def test_rotation_example(yaw, expected):
    stool = _make_object(0, [0, 0, 5], front=(1, 0, 0))
    stool.label = "stool"
    scene = _scene(stool)
    f0 = np.array([1.0, 0.0, 0.0])
    turned = scene.frame(position=stool.center_world, orientation=scene.orientation_from_forward(f0)).rotate(yaw=yaw)
    stool.per_view_fronts = {0: f0, 1: np.asarray(turned.front_vec, float)}
    ans, _, err = execute_code(_example("Looking down on the room, did the stool turn"), "which way did the stool turn?",
                               StubVL(), scene, [None, None])
    assert err is None, err
    assert ans == expected
