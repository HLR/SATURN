"""anchor.turned("+x" / "-y" / ...): a turn about the question's declared axes, as a soft turned score.

Right-hand rule in the declared right-handed system: + about an axis pointing up = turned left,
+ about an axis pointing right = turned above; an axis pointing down / left swaps them; the
forward axis (roll) is not measured (0.5)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_frame_first_api import _make_camera, _make_object  # noqa: E402

from saturn.pipeline.formula import make_formula_helpers  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402


def _two_cameras(yaw=0.0, pitch=0.0):
    th, ph = np.radians(yaw), np.radians(pitch)                       # camera 2 turned right by yaw, up by pitch
    f2 = np.array([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)])
    cams = [_make_camera([0, 0, 0], [0, 0, 1.0], cam_id=0), _make_camera([0.5, 0, 0], f2, cam_id=1)]
    scene = Scene(objects=[_make_object(0, [0, 0, 3.0])], cameras=cams, images=[None, None])
    camera = make_formula_helpers(None, scene)["camera"]
    return scene, camera, scene.frame(position=camera(1).position, orientation=camera(1).orientation)


CONVENTIONS = [   # declared axes -> which label each axis's "+" turn is (up/down/right/left/roll)
    (dict(up="+Y", forward="-Z"), {"x": "above", "y": "left", "z": None}),           # "+Y up, -Z forward"
    (dict(right="+X", up="-Y", forward="+Z"), {"x": "above", "y": "right", "z": None}),   # y down (OpenCV)
    (dict(right="+X", up="+Z", forward="+Y"), {"x": "above", "y": None, "z": "left"}),
    (dict(right="-X", up="+Y", forward="+Z"), {"x": "below", "y": "left", "z": None}),   # x pointing left
]
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}


@pytest.mark.parametrize("conv,plus", CONVENTIONS)
@pytest.mark.parametrize("yaw,pitch", [(30, 0), (-40, 0), (0, 20), (0, -25), (25, 15)])
def test_axis_turns_are_the_physical_turned_labels(conv, plus, yaw, pitch):
    scene, camera, a1 = _two_cameras(yaw, pitch)
    scene.set_axis_convention(**conv)
    for axis, label in plus.items():
        for sign in "+-":
            got = float(a1.turned(sign + axis)[camera(2)])
            if label is None:
                assert got == pytest.approx(0.5)                          # roll: not measured
            else:
                want = label if sign == "+" else OPPOSITE[label]
                assert got == pytest.approx(float(a1.turned(want)[camera(2)]), abs=1e-12)


def test_the_questions_example_convention():
    """'+Y up, -Z forward, right-handed': a right turn is a negative angle about Y, a tilt up positive about X."""
    scene, camera, a1 = _two_cameras(yaw=30)
    scene.set_axis_convention(up="+Y", forward="-Z")
    assert float(a1.turned("-y")[camera(2)]) > 0.7 > 0.3 > float(a1.turned("+y")[camera(2)])
    scene, camera, a1 = _two_cameras(pitch=20)
    scene.set_axis_convention(up="+Y", forward="-Z")
    assert float(a1.turned("+x")[camera(2)]) > float(a1.turned("-x")[camera(2)])


def test_axis_turns_compose():
    """'positive about Y, then negative about X' is a conjunction of two soft turns."""
    scene, camera, a1 = _two_cameras(yaw=-30, pitch=-20)            # turned left and down
    scene.set_axis_convention(up="+Y", forward="-Z")
    both = a1.turned("+y")[camera(2)] & a1.turned("-x")[camera(2)]
    assert float(both) == pytest.approx(min(float(a1.turned("+y")[camera(2)]), float(a1.turned("-x")[camera(2)])))
    assert float(both) > float(a1.turned("-y")[camera(2)] & a1.turned("-x")[camera(2)])


def test_without_a_convention_the_axes_are_right_up_forward():
    scene, camera, a1 = _two_cameras(yaw=30)
    scene.set_axis_convention()
    assert float(a1.turned("-y")[camera(2)]) == pytest.approx(float(a1.turned.right[camera(2)]))
