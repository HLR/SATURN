"""Direction-label aliases: one canonical spelling, alias-aware comparison.

``view.facing.behind`` is an alias of ``view.facing.back``, and the engine's
``"back-left"`` compares equal to a ``"behind-left"`` option.
"""

import os
import sys

import numpy as np
import pytest

from saturn.scene.direction_utils import (
    RELATIVE_LABELS_8,
    canonical_direction,
    is_relative_label,
    translate_label,
)
from saturn.scene.scene import Scene
from saturn.scene.types import DirectionValue
from saturn.soft_logic import PredicateArray

sys.path.insert(0, os.path.dirname(__file__))
from test_frame_first_api import _make_camera, _make_object  # noqa: E402


@pytest.mark.parametrize("alias, canon", [
    ("behind", "back"), ("rear", "back"), ("forward", "front"),
    ("behind-left", "back-left"), ("behind_right", "back-right"),
    ("Behind Left", "back-left"), ("rear left", "back-left"),
    ("forward-left", "front-left"), ("forward_right", "front-right"),
    ("left-front", "front-left"), ("right back", "back-right"),
    ("north-east", "northeast"), ("north_west", "northwest"),
    ("back", "back"), ("front-left", "front-left"), ("above", "above"),
])
def test_canonical_direction(alias, canon):
    assert canonical_direction(alias) == canon


def test_distinct_directions_stay_distinct():
    assert canonical_direction("behind-left") != canonical_direction("back-right")
    assert canonical_direction("front") != canonical_direction("back")


def test_relative_helpers_accept_aliases():
    assert is_relative_label("rear") and is_relative_label("forward-right")
    assert not is_relative_label("north")
    assert translate_label("behind-left") == "southwest"
    assert translate_label("forward", to="cardinal") == "north"
    assert translate_label("north-east", to="cardinal") == "northeast"


def test_engine_labels_are_canonical_and_compare_alias_aware():
    lab = DirectionValue(225.0).label(8)
    assert str(lab) == "back-left"  # canonical spelling unchanged
    assert lab == "behind-left" and "behind-left" == lab and lab != "back-right"
    assert lab in ["A) front", "behind left"]
    assert {lab: 1}["back-left"] == 1  # hashes as its canonical spelling
    assert RELATIVE_LABELS_8[4] == "behind" and RELATIVE_LABELS_8[0] == "forward"


@pytest.fixture
def scene():
    cameras = [_make_camera(position=[0, 0, 0], forward=[0, 0, 1], cam_id=0)]
    objects = [_make_object(0, center=[0, 0, 5]), _make_object(1, center=[3, 0, -4]),
               _make_object(2, center=[-3, 0, -2], front=(1, 0, 0))]
    return Scene(objects=objects, cameras=cameras, images=[None])


def test_view_direction_label_matches_behind_option(scene):
    view = scene._frame(at=scene.cameras[0])
    lab = view.direction(target=2, as_label=True)
    assert str(lab) == "back-left" and lab == "behind-left"


@pytest.mark.parametrize("alias, canon", [
    ("behind", "back"), ("rear", "back"), ("forward", "front"),
    ("behind_left", "back_left"), ("behind_right", "back_right"),
    ("forward_left", "front_left"), ("forward_right", "front_right"),
])
def test_first_person_aliases_are_identical(scene, alias, canon):
    fp = scene._frame(at=scene.cameras[0]).first_person
    expected = np.asarray(getattr(fp, canon))
    for got in (getattr(fp, alias), fp(alias), fp[alias.replace("_", "-")]):
        assert isinstance(got, PredicateArray)
        assert np.array_equal(np.asarray(got), expected)


def test_first_person_is_bit_identical_to_raw_scores(scene):
    fp = scene._frame(at=scene.cameras[0]).first_person
    raw = fp._compute("left")
    assert type(raw) is np.ndarray
    assert np.array_equal(np.asarray(fp.left), raw)


def test_unhyphenated_cardinal_is_a_cardinal(scene):
    """``northeast`` is a cardinal label: it needs the cardinal vector, it is not front-right."""
    fp = scene._frame(at=scene.cameras[0]).first_person
    with pytest.raises(ValueError, match="set_cardinal_vector"):
        fp.northeast
    scene.set_cardinal_vector([1.0, 0.0, 0.0])
    fp = scene._frame(at=scene.cameras[0]).first_person
    assert np.array_equal(np.asarray(fp.northeast), np.asarray(fp.north_east))


@pytest.mark.parametrize("alias, canon", [
    ("behind", "back"), ("rear", "back"), ("forward", "front"),
    ("behind_left", "back_left"), ("forward_right", "front_right"),
])
def test_facing_aliases_are_identical(scene, alias, canon):
    facing = scene._frame(at=scene.cameras[0]).facing
    expected = getattr(facing, canon).tensor
    for got in (getattr(facing, alias), facing(alias), facing[alias]):
        assert np.array_equal(got.tensor.cpu().numpy(), expected.cpu().numpy())


def test_facing_unknown_label_raises(scene):
    with pytest.raises(AttributeError, match="unknown direction"):
        scene._frame(at=scene.cameras[0]).facing.sideways
