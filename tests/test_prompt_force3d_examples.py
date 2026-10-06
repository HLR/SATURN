"""Every worked example in the 3D-FORCE prompts runs on the engine and returns the right answer type.

The programs are read from the prompts themselves, so an edit that breaks an example fails here.
"""

import os
import re
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

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _examples(name):
    """(question, program) for every worked example in the prompt."""
    text = open(os.path.join(PROMPTS, name)).read().split("=== EXAMPLES ===")[1].split("=== OUTPUT ===")[0]
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().split("\n")
        if lines and lines[0].startswith("Q: "):
            out.append((lines[0][3:], "\n".join(lines[1:])))
    return out


class StubVL:
    """score(...) is 1.0 for every object whose label word appears in the question."""

    def score_multiview(self, question, num_objects=1, type=None, scene=None, images=None, cam_id=None, **kw):
        v = torch.zeros(scene.objects_count + len(scene.cameras), dtype=torch.float64)
        q = question.lower()
        for i, o in enumerate(scene.objects):
            if any(w in q for w in o.label.split()):
                v[i] = 1.0
        return ProbabilisticTensor(v)

    def query_multiview(self, question, **kw):
        return ""


def _scene(question):
    words = [w for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 3]
    labels = list(dict.fromkeys(words))[:10]
    rng = np.random.default_rng(0)
    objs = []
    for i, lab in enumerate(labels):
        o = _make_object(i, rng.uniform(-3, 3, 3) * [1, 0.2, 1] + [0, 0, 6], front=tuple(rng.normal(size=3) * [1, 0, 1]))
        o.label = lab
        objs.append(o)
    cams = [_make_camera([x, 0, 0], [0, 0, 1], cam_id=k) for k, x in enumerate((0.0, 1.0, -1.0))]
    return Scene(objects=objs, cameras=cams, images=[None] * len(cams))


@pytest.mark.parametrize("question, program", _examples("force3d_ref.txt"))
def test_ref_examples_return_an_object_index(question, program):
    scene = _scene(question)
    ans, _, err = execute_code(program, question, StubVL(), scene, [None] * 3)
    assert err is None, err
    assert int(ans) in range(scene.objects_count)


@pytest.mark.parametrize("question, program", _examples("force3d_sag.txt"))
def test_sag_examples_return_true_or_false(question, program):
    scene = _scene(question)
    ans, _, err = execute_code(program, question, StubVL(), scene, [None] * 3)
    assert err is None, err
    assert ans in ("true", "false")
