"""The prompt's object-anchored examples, written with scene.frame(at=<the .iota description>), answer exactly
as the equivalent programs that index scene.objects directly (INDEX_FORM below).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_frame_first_api import _make_camera, _make_object  # noqa: E402
from test_prompt_motion_examples import StubVL  # noqa: E402

from saturn.pipeline.execute import execute_code  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STEMS = ("Seen from camera 2, which item hangs", "From the perspective of the cat", "Sitting on the piano bench",
         "Camera 1 looks at the front of the mailbox", "The bookcase stands with its back",
         "The piano is on the east side", "I sit at the desk")
LABELS = ("fireplace", "painting", "clock", "mirror", "cat", "chair", "shelf", "table", "piano bench", "lamp",
          "plant", "door", "mailbox", "bookcase", "window", "piano", "couch", "rug", "desk", "printer")


def _example(text: str, stem: str) -> str:
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"Q: {stem}"))
    body = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        body.append(line)
    return "\n".join(body)


def _question(text: str, stem: str) -> str:
    return next(line for line in text.split("\n") if line.startswith(f"Q: {stem}"))[3:]


PROMPT = open(os.path.join(ROOT, "prompts", "vqa.txt")).read()

# Each example written with explicit object indices (scene.objects[...] / .assign()):
# the description form in the prompt must answer exactly as its index form does.
INDEX_FORM = {
    'Seen from camera 2, which item hangs': 'fp = score("is the object in the red bounding box a fireplace?").iota("x1").assign()["x1"]\nanchor = scene.frame(position=scene.objects[fp].position, orientation=camera(2).orientation)\noptions = {"A": "painting", "B": "clock", "C": "mirror"}\nreturn max(options, key=lambda k: float((anchor.first_person.above("x1") & score(f"is the object in the red bounding box a {options[k]}?").iota("x1")).exists()))',
    'From the perspective of the cat': 'c = score("is the object in the red bounding box a cat?").iota("x1").assign()["x1"]\nanchor = scene.frame(position=scene.objects[c].position, orientation=scene.objects[c].orientation)\ncat = score("is the object in the red bounding box a cat?").iota("x2")\nfar_cat = (scene.distance("x1", "x2") & cat).exists("x2")                                                 # how far x1 is from the cat\nS = score("is the object in the red bounding box made of wood?")("x1") & anchor.first_person.left("x1")   # who competes\nJ = far_cat.best("x1", among=S)                                                                           # the farthest of them\noptions = {"A": "chair", "B": "shelf", "C": "table"}\nreturn max(options, key=lambda k: float((J & score(f"is the object in the red bounding box a {options[k]}?").iota("x1")).exists()))',
    'Sitting on the piano bench': 'J = score("is the object in the red bounding box a piano bench?").iota("x2")\np = J.assign()["x2"]\nanchor = scene.frame(position=scene.objects[p].position, orientation=camera(1).orientation)\noptions = {"A": "lamp", "B": "plant", "C": "door"}\nreturn max(options, key=lambda k: float((anchor.first_person.right("x1") & score(f"is the object in the red bounding box a {options[k]}?").iota("x1")).exists()))',
    'Camera 1 looks at the front of the mailbox': 'm = score("is the object in the red bounding box a mailbox?").iota("x1").assign()["x1"]\nf = scene.frame(position=scene.objects[m].position, orientation=scene.objects[m].orientation).look_at(camera(1))   # its front faces camera 1\nlabels = {"A": "left", "B": "back", "C": "right"}\nreturn max(labels, key=lambda k: float(f.first_person(labels[k])[camera(3)]))',
    'The bookcase stands with its back': 'b = score("is the object in the red bounding box a bookcase?").iota("x2").assign()["x2"]\nw = score("is the object in the red bounding box a window?").iota("x2").assign()["x2"]\nanchor = scene.frame(position=scene.objects[b].position, orientation=scene.objects[b].orientation).look_at(w).rotate(yaw=180)   # its back is toward the window\noptions = {"A": "lamp", "B": "chair", "C": "plant"}\nreturn max(options, key=lambda k: float((anchor.first_person.right("x1") & score(f"is the object in the red bounding box a {options[k]}?").iota("x1")).exists()))',
    'The piano is on the east side': 'p = score("is the object in the red bounding box a piano?").iota("x2").assign()["x2"]\nc = score("is the object in the red bounding box a couch?").iota("x2").assign()["x2"]\nscene.set_cardinal_vector(scene.cardinalize(scene.objects[p].position - scene.objects[c].position, known="east"))   # the stated fact: couch -> piano points east\nanchor = scene.frame(position=scene.objects[c].position, orientation=scene.objects[c].orientation)\noptions = {"A": "rug", "B": "shelf", "C": "lamp"}\nreturn max(options, key=lambda k: float((anchor.first_person.north("x1") & score(f"is the object in the red bounding box a {options[k]}?").iota("x1")).exists()))',
    'I sit at the desk': 'd = score("is the object in the red bounding box a desk?").iota("x1").assign()["x1"]\nme = scene.frame(position=scene.objects[d].position, orientation=scene.objects[d].orientation).rotate(yaw=180)   # working at the desk: I face it\noptions = {"A": "lamp", "B": "printer", "C": "plant"}\nreturn max(options, key=lambda k: float((me.first_person.right("x1") & score(f"is the object in the red bounding box a {options[k]}?").iota("x1")).exists()))',
}


def _scene(seed):
    rng = np.random.RandomState(seed)
    objs = []
    for i, label in enumerate(LABELS):
        o = _make_object(i, rng.randn(3) * np.array([3.0, 0.5, 3.0]), front=rng.randn(3) * np.array([1.0, 0.0, 1.0]))
        o.label = label
        objs.append(o)
    cams = [_make_camera(rng.randn(3) * np.array([4.0, 0.2, 4.0]), rng.randn(3) * np.array([1.0, 0.0, 1.0]), cam_id=c)
            for c in range(3)]
    return Scene(objects=objs, cameras=cams, images=[None] * 3)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("stem", STEMS)
def test_anchor_examples_match_the_index_form(stem, seed):
    described, indexed = _example(PROMPT, stem), INDEX_FORM[stem]
    frame_line = next(line for line in described.split("\n") if "scene.frame(at=" in line)
    assert ".assign()" not in frame_line and "scene.objects[" not in frame_line
    q = _question(PROMPT, stem)
    ans, _, err = execute_code(described, q, StubVL(), _scene(seed), [None] * 3)
    assert err is None, err
    ref, _, err = execute_code(indexed, q, StubVL(), _scene(seed), [None] * 3)
    assert err is None, err
    assert ans == ref
