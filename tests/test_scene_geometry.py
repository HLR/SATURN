"""Scene geometry: assignment fusion, observer/facing handedness, constraints on
canonical cameras, and direction/rotation option matching."""

import numpy as np
import pytest

from saturn.scene.fusion import assign_keyword_observations
from saturn.scene.pose_solver import R_y, solve_camera_poses
from saturn.scene.scene import Scene
from saturn.scene.types import Camera
from test_match_methods import _make_object  # noqa: E402

_FLIP = np.diag([1.0, -1.0, 1.0])  # canonicalize_y_up reflection


def _canonical_camera(cam_id, position, yaw_deg):
    """Canonical (det=-1) camera at ``position`` turned ``yaw_deg`` about +Y."""
    R_c2w = R_y(yaw_deg) @ _FLIP.T
    ext = np.eye(4)
    ext[:3, :3] = R_c2w.T
    ext[:3, 3] = -R_c2w.T @ np.asarray(position, float)
    return Camera(id=cam_id, entity_id=cam_id, intrinsics=np.eye(3), extrinsics=ext,
                  image_size=(480, 640))


# ---- the assign rule never chains two detections of one view ----

def test_assign_never_merges_two_detections_of_one_view():
    rng = np.random.default_rng(0)

    def ob(view, x, uid):
        return {"world_points": rng.normal([x, 0, 3.0], 0.03, (200, 3)),
                "view_idx": view, "_uid": uid}

    obs = [ob(0, 0.00, "a1"), ob(0, 0.40, "a2"), ob(1, 0.12, "b1"), ob(2, 0.30, "c1")]
    cams = {0: np.zeros(3), 1: np.array([1.0, 0, 0]), 2: np.array([-1.0, 0, 0])}
    groups = assign_keyword_observations(obs, cams, k=0.15)
    for g in groups:
        views = [o["view_idx"] for o in g]
        assert len(views) == len(set(views)), [o["_uid"] for o in g]
    # cheapest pairs win: a1-b1 (0.12) and a2-c1 (0.10)
    assert sorted(sorted(o["_uid"] for o in g) for g in groups) == [["a1", "b1"], ["a2", "c1"]]


# ---- facing= uses right = up x forward ----

def test_direction_facing_matches_observer_handedness():
    cam = _canonical_camera(0, [0, 0, 0], 0.0)
    me = _make_object(0, [0, 0, 0], front=(0, 0, 1))
    tv = _make_object(1, [0, 0, 3])
    lamp = _make_object(2, [2, 0, 1])  # camera image-right
    scene = Scene(objects=[me, tv, lamp], cameras=[cam], images=[None])
    assert scene.direction(0, 2, observer=0).label(4) == "right"
    assert scene.direction(0, 2, facing=1).label(4) == "right"
    above = np.array([0, 2.0, 1.0])
    assert scene.direction(0, above, facing=1).elevation_degree() == pytest.approx(
        scene.direction(0, above, observer=0).elevation_degree())
    assert scene.direction(0, above, facing=1).elevation_degree() > 0


# ---- face() holds after a later camera constraint ----

def test_face_survives_rotation_constraint():
    c0 = _canonical_camera(0, [0, 0, 0], 0.0)
    c1 = _canonical_camera(1, [0, 0, 0], 85.0)
    statue = _make_object(0, [0, 0, 3], front=(1, 0, 0))
    statue.per_view_scores = {0: 0.9}
    scene = Scene(objects=[statue], cameras=[c0, c1], images=[None, None])
    scene.constraint.face(scene.objects[0], toward=scene.cameras[0])
    scene.constraint.rotation(scene.cameras[0], scene.cameras[1], yaw=90)
    obj = scene.objects[0]
    to_cam = scene.cameras[0].position_world - obj.center_world
    to_cam /= np.linalg.norm(to_cam)
    assert np.allclose(obj.front_world, to_cam, atol=1e-6)
    scene.constraint.clear()
    assert np.allclose(scene.objects[0].front_world, [1, 0, 0])


# ---- strength in (0, 1) keeps canonical cameras det=-1 ----

@pytest.mark.parametrize("method", ["anchor", "average"])
def test_partial_strength_keeps_reflection(method):
    ext0 = np.eye(4)
    ext0[:3, :3] = _FLIP
    ext1 = np.eye(4)
    ext1[:3, :3] = (R_y(80) @ _FLIP.T).T
    cons = [{"type": "rotation", "from_cam": 0, "to_cam": 1, "yaw": 90.0, "axis": "up"}]
    out = solve_camera_poses([ext0, ext1], cons, method=method, strength=0.5)
    for ext in out:
        assert np.linalg.det(ext[:3, :3]) < 0
    f = out[1][:3, :3].T[:, 2]
    assert abs(f[1]) < 1e-6  # no pitch
    yaw = np.degrees(np.arctan2(f[0], f[2]))
    assert 80.0 <= yaw <= 90.0


# ---- hyphenated / aliased options in match_direction ----

def test_match_direction_hyphenated_and_alias_options():
    north = np.array([0, 0, 1.0])
    th = np.radians(120)  # south-east
    a = _make_object(0, [0, 0, 0], front=(0, 0, 1))
    se = _make_object(1, [2 * np.sin(th), 0, 2 * np.cos(th)])
    br = _make_object(2, [2, 0, -2])  # back-right of a
    scene = Scene(objects=[a, se, br], cameras=[], images=[])
    opts = {"A": "north-east", "B": "south-east", "C": "west", "D": "north"}
    assert scene.match_direction(0, 1, opts, north_vector=north) == "B"
    opts_sp = {"A": "north east", "B": "south east", "C": "west", "D": "north"}
    assert scene.match_direction(0, 1, opts_sp, north_vector=north) == "B"
    opts_rel = {"A": "right", "B": "behind-right", "C": "front-left", "D": "left"}
    assert scene.match_direction(0, 2, opts_rel) == "B"


# ---- "clockwise" inside "counterclockwise" does not match it ----

@pytest.mark.parametrize("direction,options,expected", [
    ("counterclockwise", {"A": "clockwise", "B": "counter-clockwise"}, "B"),
    ("counterclockwise", {"A": "clockwise", "B": "anticlockwise"}, "B"),
    ("clockwise", {"A": "Rotated counterclockwise", "B": "Rotated clockwise"}, "B"),
    ("clockwise", {"A": "left", "B": "right"}, "B"),
    ("counterclockwise", {"A": "left", "B": "right"}, "A"),
    ("clockwise", {"A": "clockwise", "B": "counterclockwise"}, "A"),
])
def test_match_object_rotation_direction_spellings(direction, options, expected):
    scene = Scene(objects=[_make_object(0, [0, 0, 0])], cameras=[], images=[])
    rot = {"top_down_direction": direction,
           "signed_angle_deg": 30.0 if direction == "clockwise" else -30.0}
    scene.object_rotation = lambda *a, **k: rot
    scene.object_rotation_camera_frame = lambda *a, **k: rot
    assert scene.match_object_rotation_direction(0, 0, 1, options) == expected
    assert scene.match_object_rotation_direction_camera_frame(0, 0, 1, options) == expected
