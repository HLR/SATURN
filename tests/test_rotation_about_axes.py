import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_frame_first_api import _make_camera, _make_object  # noqa: E402
from saturn.scene.scene import Scene  # noqa: E402


@pytest.fixture
def scene():
    cams = [_make_camera([0, 0, 0], [0, 0, 1], cam_id=0)]
    return Scene(objects=[_make_object(0, [0, 0, 5])], cameras=cams, images=[None])


def _anchor(scene):
    c = scene.cameras[0]
    return scene.frame(position=c.position, orientation=c.orientation)


def test_turn_right_is_negative_about_up_axis_opengl(scene):
    scene.set_axis_convention(up="+Y", forward="-Z")
    a = _anchor(scene)
    r = a.rotation_about_axes(a.rotate(yaw=30))
    assert r["Y"] == pytest.approx(-30, abs=0.5) and abs(r["X"]) < 0.5 and abs(r["Z"]) < 0.5
    r = a.rotation_about_axes(a.rotate(yaw=-40))
    assert r["Y"] == pytest.approx(40, abs=0.5)


def test_tilt_up_is_positive_about_right_axis_opengl(scene):
    scene.set_axis_convention(up="+Y", forward="-Z")
    a = _anchor(scene)
    r = a.rotation_about_axes(a.rotate(pitch=20))
    assert r["X"] == pytest.approx(20, abs=0.5) and abs(r["Y"]) < 0.5


def test_other_right_handed_convention_z_up(scene):
    # right=+X, up=+Z, forward=+Y is right-handed; turning right = negative about Z.
    scene.set_axis_convention(right="+X", up="+Z", forward="+Y")
    a = _anchor(scene)
    assert a.rotation_about_axes(a.rotate(yaw=25))["Z"] == pytest.approx(-25, abs=0.5)


def test_left_handed_declaration_is_refused(scene):
    scene.set_axis_convention()  # right=+X, up=+Y, forward=+Z is left-handed
    a = _anchor(scene)
    with pytest.raises(ValueError, match="left-handed"):
        a.rotation_about_axes(a.rotate(yaw=10))


def test_cardinal_vector_inverts_cardinalize(scene):
    north = scene.cardinalize(np.array([1.0, 0.0, 0.3]), known="east")
    scene.set_cardinal_vector(north)
    assert np.allclose(scene.cardinal_vector("north"), north / np.linalg.norm(north))
    for d in ("east", "south-west", "northwest"):
        assert np.allclose(scene.cardinalize(scene.cardinal_vector(d), known=d.replace("-", "")), scene.cardinal_vector())
    assert np.allclose(scene.cardinal_vector("south"), -scene.cardinal_vector("north"))
