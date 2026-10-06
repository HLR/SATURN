"""Orient-Anything V2 orientation convention.

This module defines the canonical (az, polar, roll) <-> rotation-matrix
conversion for OA V2 outputs, using the formula of V2's reference pipeline
(``pipelineV2.py``):

    R = scipy.spatial.transform.Rotation.from_euler(
            "yxz", [azimuth_deg, polar_deg, rotation_deg], degrees=True
        ).as_matrix()

Column order of the returned 3x3 matrix is `[right_obj, up_obj, forward_obj]`
expressed in the OpenCV camera frame (+X right, +Y down, +Z into scene).

The weights are Orient-Anything V2
(`Viglong/OriAnyV2_ckpt::demo_ckpts/rotmod_realrotaug_best.pt`); V1 uses a
different (look_at) construction.

Conventions
-----------
OA V2 angles (user-facing = backend; no az-flip):
  azimuth_deg : [0, 360)      - camera's horizontal angle around the object
  polar_deg   : [-90, 90]     - camera's elevation
  rotation_deg: [-180, 180]   - in-plane roll

Camera frame: OpenCV (+X right, +Y down, +Z into scene).

World azimuth semantics — *semantic front* (after world-frame fusion, with
initial camera at origin looking +Z_world):
  az = 0    -> front_world = -Z_world   (toward initial camera; object faces viewer)
  az = 90   -> front_world = -X_world   (world left)
  az = 180  -> front_world = +Z_world   (away from initial camera)
  az = 270  -> front_world = +X_world   (world right)

Note: the *raw* rotation matrix returned by
:func:`rotation_matrix_from_user_euler` has column 2 = +Z_obj_local, which
points INTO the scene at (az=0). The object's FRONT axis is ``-column 2``.
This sign flip is applied centrally in :mod:`saturn.scene.adapters`.
"""
import numpy as np
from scipy.spatial.transform import Rotation as _R


# ---------------------------------------------------------------------------
# User <-> backend (identity for OA V2)
# ---------------------------------------------------------------------------
def backend_azimuth_to_user(azimuth_deg: float) -> float:
    return float(float(azimuth_deg) % 360.0)


def backend_euler_to_user(euler_deg) -> np.ndarray:
    euler = np.asarray(euler_deg, dtype=float)
    return np.array(
        [backend_azimuth_to_user(euler[0]), euler[1], euler[2]], dtype=float
    )



# ---------------------------------------------------------------------------
# Core geometry (V2 convention: scipy yxz intrinsic Euler)
# ---------------------------------------------------------------------------
def rotation_matrix_from_user_euler(euler_deg) -> np.ndarray:
    """OA V2 user-Euler (az, polar, roll) in degrees -> camera-frame 3x3 rotation.

    Columns of the returned matrix are ``[right_obj, up_obj, +Z_obj_local]``
    in OpenCV camera frame.  Note that column 2 is the object's local +Z axis
    — **not** its semantic front direction.  Under the OA V2 formula, az=0
    corresponds to the object facing the camera, so the semantic front axis
    is ``-column 2``. This sign flip is applied in
    :mod:`saturn.scene.adapters` on ingestion.

    Matches the V2 reference pipeline:
        R = Rotation.from_euler("yxz", [az, el, rot], degrees=True).as_matrix()
    which is equivalent to ``Ry(az) @ Rx(el) @ Rz(rot)`` (intrinsic yxz).
    """
    euler = np.asarray(euler_deg, dtype=float).reshape(3)
    return _R.from_euler("yxz", euler, degrees=True).as_matrix()


def user_euler_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """Inverse of `rotation_matrix_from_user_euler`.

    Input: 3x3 rotation matrix whose columns are
    ``[right_obj, up_obj, +Z_obj_local]`` in OpenCV camera (or world) frame.

    Output: ``[azimuth_deg, polar_deg, roll_deg]`` with
      az  in [0, 360),  polar in [-90, 90],  roll in [-180, 180].
    """
    R_mat = np.asarray(rotation_matrix, dtype=float)
    if R_mat.shape != (3, 3):
        raise ValueError(f"rotation_matrix must be (3,3); got {R_mat.shape}")

    euler = _R.from_matrix(R_mat).as_euler("yxz", degrees=True)
    az_deg = float(euler[0]) % 360.0
    polar_deg = float(np.clip(euler[1], -90.0, 90.0))
    roll_deg = float(euler[2])
    # Normalize roll to (-180, 180]
    if roll_deg > 180.0:
        roll_deg -= 360.0
    elif roll_deg <= -180.0:
        roll_deg += 360.0
    return np.array([az_deg, polar_deg, roll_deg], dtype=float)


# ---------------------------------------------------------------------------
# Cardinal-direction helpers for object-centric relations
# ---------------------------------------------------------------------------
# Semantic: OA V2 returns a rotation R whose columns are arbitrary local
# axes [X_obj, Y_obj, Z_obj] in the camera frame. The model is trained so
# that az=0 corresponds to the camera viewing the object's FRONT face
# (object facing toward the viewer). With scipy yxz at
# (az=0, pol=0, rot=0), R = I, so R[:, 2] = (0, 0, 1) = into the scene
# AWAY from the viewer. Therefore the object's front axis = -R[:, 2].
#
# Consistency requirement:
#   -rotation_matrix_from_user_euler([az, 0, 0])[:, 2]
#   == front_direction_3d_from_user_azimuth(az)
#
# At (az, 0, 0), R = Ry(az), R[:, 2] = (sin(az), 0, cos(az)),
# so front = -R[:, 2] = (-sin(az), 0, -cos(az)). World/camera-frame semantics
# (when c2w = I):
#   az = 0   -> ( 0, 0, -1)  (toward camera)
#   az = 90  -> (-1, 0,  0)  (world -X)
#   az = 180 -> ( 0, 0, +1)  (away from camera)
#   az = 270 -> (+1, 0,  0)  (world +X)


def front_direction_3d_from_user_azimuth(azimuth_deg: float) -> np.ndarray:
    """Front-axis direction (unit vector) in the canonical camera/world frame.

    Matches -column 2 of `rotation_matrix_from_user_euler([az, 0, 0])`.
    az=0 means the object faces toward the camera (toward the viewer).
    """
    az = np.deg2rad(float(azimuth_deg))
    return np.array([-np.sin(az), 0.0, -np.cos(az)], dtype=float)


def front_direction_2d_from_user_azimuth(azimuth_deg: float) -> np.ndarray:
    """2D top-down front direction (x, z) in the canonical frame."""
    az = np.deg2rad(float(azimuth_deg))
    return np.array([-np.sin(az), -np.cos(az)], dtype=float)
