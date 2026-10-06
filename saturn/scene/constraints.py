"""Pose-constraint namespace (scene.constraint.rotation / same_position / face).

Imports: may import saturn.settings / saturn.scene.serialization / saturn.scene.pose_solver; must not import saturn.scene.scene at module level; must not import saturn.perception, saturn.vlm, saturn.serving."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .serialization import _serialize_target
from saturn.log import get_logger

log = get_logger(__name__)


class _ConstraintNamespace:
    """Accumulator for pose constraints stated in question text.

    Constraints are added one at a time via the public methods. Camera-pose
    constraints (``rotation``, ``same_position``) trigger an eager re-solve
    via ``saturn.scene.pose_solver`` and write the refined
    extrinsics back into ``scene.cameras[i]``. Object-orientation constraints
    (``face``) directly mutate the object's world-frame axes; they do not
    re-solve camera poses.

    Camera-pose methods (``rotation`` / ``same_position``) accept only Camera
    entities; passing an object raises ``NotImplementedError``. The object-
    facing method (``face``) accepts only object entities for its first arg
    and any entity (camera, object, or 3D point) for ``toward``.
    """

    def __init__(self, scene):
        self._scene = scene
        # List of constraint records consumable by pose_solver.solve_camera_poses.
        # Each record is a plain dict to keep the namespace JSON-serializable
        # for cache writes / debug logging.
        self._constraints: List[Dict[str, Any]] = []
        # Snapshot of the original extrinsics so re-solves always start from
        # VGGT's estimates (not from the previous solve's output) when
        # constraints are added incrementally. Captured lazily on first add.
        self._original_extrinsics: Optional[List[np.ndarray]] = None
        # Parallel snapshot of object world-frame geometry so co-transform
        # always starts from VGGT's estimate. Mirrors the camera snapshot
        # contract: lazy on first add, restored on clear(). Stored as a
        # list of dicts (one per object) to keep the snapshot dataclass-free.
        self._original_objects: Optional[List[Dict[str, Any]]] = None
        # (object index, live ``toward`` entity) per face() call, replayed by
        # _solve() after the co-transform (which starts from the snapshot).
        self._faces: List[Tuple[int, Any]] = []

    # ----- Public constraint-adder methods -----

    def rotation(self, from_entity, to_entity, *, yaw: float, axis: str = "up") -> None:
        """Constrain ``to_entity`` = ``from_entity`` rotated by ``yaw`` degrees
        about the given world axis (default: up).

        Both entities must be Camera instances; passing an object entity
        raises ``NotImplementedError``.

        Sign convention: yaw=+90 means clockwise viewed from above (right-hand
        rule about world +Y), matching the natural-language sense of
        "rotated 90 degrees clockwise" used in question text.
        """
        from_idx = self._require_camera(from_entity, "from_entity")
        to_idx   = self._require_camera(to_entity, "to_entity")
        self._snapshot_originals_if_needed()
        self._constraints.append({
            "type": "rotation",
            "from_cam": from_idx,
            "to_cam": to_idx,
            "yaw": float(yaw),
            "axis": str(axis),
        })
        self._solve()

    def same_position(self, *entities) -> None:
        """Constrain that all given Camera entities share the same world
        position. Useful for "all photos taken from the same spot" cases.
        """
        if len(entities) < 2:
            raise ValueError("same_position requires at least 2 entities")
        idxs = [self._require_camera(e, f"entities[{i}]") for i, e in enumerate(entities)]
        self._snapshot_originals_if_needed()
        self._constraints.append({"type": "same_position", "cams": idxs})
        self._solve()

    def face(self, obj_entity, *, toward) -> None:
        """Constrain a scene object's intrinsic +front axis to point at
        ``toward``.

        Use for question text that redefines an object's front via a viewing
        reference, e.g. "Figure 1 shows the front of the statue" — the question
        is asserting that the statue's front faces camera 0. After this call,
        the object's ``front_world`` / ``up_world`` / ``right_world`` /
        ``rotation_world`` are rewritten so that ``front_world`` points from
        the object toward ``toward``. World up (+Y) is preserved (falls back to
        +Z if the new forward is parallel to +Y). ``obj.orientation`` and any
        anchor built via ``scene.frame(orientation=obj.orientation)`` reflect
        the constraint.

        Object-orientation refinement does NOT re-solve camera poses (object
        orientation is independent of camera extrinsics), but pose-derived
        scene caches are invalidated so subsequent queries recompute.
        ``scene.constraint.clear()`` restores the pre-constraint orientation.

        Parameters
        ----------
        obj_entity : MergedObject, int, or ("object", idx)
            The object whose facing is being constrained.
        toward : entity specifier
            The target the object's front should point at. Accepts any form
            ``scene._resolve_entity`` accepts: Camera / MergedObject instance,
            ``("camera", N)`` / ``("object", N)``, ``int`` (object index),
            or a bare 3D world point.
        """
        scene = self._scene
        obj = scene._resolve_entity_object(obj_entity)
        if obj is None:
            raise ValueError(
                f"face(obj_entity): could not resolve {obj_entity!r} to a scene "
                f"object. Pass scene.objects[i] or ('object', i)."
            )
        try:
            obj_idx = scene.objects.index(obj)
        except ValueError:
            raise ValueError(
                "face(obj_entity): object is not in this scene's objects list."
            )

        axes = self._facing_axes(obj, toward)
        self._snapshot_originals_if_needed()
        self._set_object_axes(obj, *axes)

        self._constraints.append({
            "type": "face",
            "object": obj_idx,
            "toward": _serialize_target(toward),
        })
        # _solve() rebuilds objects from the pre-face snapshot; keep the live
        # target so a later camera constraint can re-apply this face.
        self._faces.append((obj_idx, toward))
        self._invalidate_scene_caches()

    def _facing_axes(self, obj, toward):
        """(front, up, right) that turn ``obj``'s front toward ``toward``."""
        tgt_pos, _, _ = self._scene._resolve_entity(toward, label="toward")
        obj_pos = np.asarray(obj.center_world, dtype=float)
        fwd = np.asarray(tgt_pos, dtype=float) - obj_pos
        nf = float(np.linalg.norm(fwd))
        if nf < 1e-9:
            raise ValueError(
                f"face(): object and target share the same position "
                f"(obj={obj_pos.tolist()}, target={np.asarray(tgt_pos).tolist()})."
            )
        fwd = fwd / nf

        up_world = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(fwd, up_world))) > 0.999:
            up_world = np.array([0.0, 0.0, 1.0])

        right = np.cross(up_world, fwd)
        right = right / float(np.linalg.norm(right))
        up = np.cross(fwd, right)
        up = up / float(np.linalg.norm(up))
        return fwd, up, right

    @staticmethod
    def _set_object_axes(obj, fwd, up, right) -> None:
        obj.front_world = fwd
        obj.up_world = up
        obj.right_world = right
        obj.rotation_world = np.column_stack([right, up, fwd])
        az = float(np.degrees(np.arctan2(fwd[0], fwd[2])))
        el = float(np.degrees(np.arcsin(np.clip(fwd[1], -1.0, 1.0))))
        obj.euler_world_deg = np.array([az, el, 0.0])

    def clear(self) -> None:
        """Drop all accumulated constraints and restore VGGT's original
        extrinsics on every camera AND original world-geometry on every
        object. Useful for unit tests and ablations.
        """
        if self._original_extrinsics is not None:
            for cam, ext in zip(self._scene.cameras, self._original_extrinsics):
                cam.extrinsics = np.asarray(ext, dtype=float).copy()
                cam.position_world = cam._extract_position()
                cam.heading = cam._extract_heading()
        if self._original_objects is not None:
            for obj, snap in zip(self._scene.objects, self._original_objects):
                self._restore_object_from_snapshot(obj, snap)
        self._constraints.clear()
        self._faces.clear()
        self._original_extrinsics = None
        self._original_objects = None
        self._invalidate_scene_caches()

    # ----- Read-only accessors for diagnostics -----

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Read-only view of the accumulated constraint records."""
        return list(self._constraints)

    # ----- Internal helpers -----

    def _require_camera(self, entity, role: str) -> int:
        """Validate that ``entity`` is a Camera in this scene; return its index."""
        entity = getattr(entity, "cam", entity)  # camera(N) carries the camera itself
        for i, cam in enumerate(self._scene.cameras):
            if cam is entity:
                return i
        # Detect "object entity" misuse early with a precise error.
        if hasattr(entity, "center_world") and not hasattr(entity, "extrinsics"):
            raise NotImplementedError(
                f"scene.constraint.* with object entities is not yet supported "
                f"(got {role}={type(entity).__name__}). v1 supports cameras only; "
                f"pass scene.cameras[i] instead."
            )
        raise ValueError(
            f"{role} (type {type(entity).__name__}) is not one of this scene's "
            f"cameras. Pass scene.cameras[i] explicitly."
        )

    def _snapshot_originals_if_needed(self) -> None:
        if self._original_extrinsics is None:
            self._original_extrinsics = [
                np.asarray(cam.extrinsics, dtype=float).copy()
                for cam in self._scene.cameras
            ]
        # Object snapshot mirrors the camera one — captured on the first
        # constraint add so co-transform always rebases from VGGT's
        # estimate, not from the previous solve's output.
        if self._original_objects is None:
            self._original_objects = [
                self._snapshot_object(obj) for obj in self._scene.objects
            ]

    @staticmethod
    def _snapshot_object(obj) -> Dict[str, Any]:
        """Capture the fields _solve will co-transform.

        Excludes image-plane data (per_view_bboxes, per_view_masks,
        per_view_scores, per_view_fronts_camera) — those live in the
        camera's pixel frame and are not affected by world-pose refinement.
        """
        snap: Dict[str, Any] = {
            "center_world": np.asarray(obj.center_world, dtype=float).copy(),
            "corners_world": np.asarray(obj.corners_world, dtype=float).copy(),
            "rotation_world": np.asarray(obj.rotation_world, dtype=float).copy(),
            "front_world": np.asarray(obj.front_world, dtype=float).copy(),
            "up_world": np.asarray(obj.up_world, dtype=float).copy(),
            "right_world": np.asarray(obj.right_world, dtype=float).copy(),
        }
        if obj.world_points is not None:
            snap["world_points"] = np.asarray(obj.world_points, dtype=float).copy()
        if obj.per_view_centers:
            snap["per_view_centers"] = {
                int(k): np.asarray(v, dtype=float).copy()
                for k, v in obj.per_view_centers.items()
            }
        if obj.per_view_fronts:
            snap["per_view_fronts"] = {
                int(k): np.asarray(v, dtype=float).copy()
                for k, v in obj.per_view_fronts.items()
            }
        # Keep per-view confidences (read-only, used for weighting) and
        # the camera-frame fronts (which are pose-independent).
        snap["per_view_scores"] = dict(obj.per_view_scores)
        snap["views"] = list(obj.views)
        return snap

    @staticmethod
    def _restore_object_from_snapshot(obj, snap: Dict[str, Any]) -> None:
        """Inverse of _snapshot_object — used by clear()."""
        obj.center_world = snap["center_world"].copy()
        obj.corners_world = snap["corners_world"].copy()
        obj.rotation_world = snap["rotation_world"].copy()
        obj.front_world = snap["front_world"].copy()
        obj.up_world = snap["up_world"].copy()
        obj.right_world = snap["right_world"].copy()
        if "world_points" in snap and obj.world_points is not None:
            obj.world_points = snap["world_points"].copy()
        if "per_view_centers" in snap:
            obj.per_view_centers = {
                int(k): v.copy() for k, v in snap["per_view_centers"].items()
            }
        if "per_view_fronts" in snap:
            obj.per_view_fronts = {
                int(k): v.copy() for k, v in snap["per_view_fronts"].items()
            }

    @staticmethod
    def _ext_to_homo(ext: np.ndarray) -> np.ndarray:
        """3x4 or 4x4 world-to-camera extrinsics → 4x4 homogeneous."""
        E = np.asarray(ext, dtype=float)
        if E.shape == (4, 4):
            return E.copy()
        out = np.eye(4)
        out[:3, :4] = E[:3, :4]
        return out

    @classmethod
    def _per_camera_world_delta(
        cls,
        old_ext: np.ndarray,
        new_ext: np.ndarray,
    ) -> np.ndarray:
        """Compute the world-frame correction induced by refining one camera.

        Intuition: a point ``p_old`` in the original world frame projects to
        camera coordinates via ``p_cam = R_w2c_old @ p_old + t_w2c_old``.
        After refinement, we want pixels observed by this camera (i.e.,
        points at ``p_cam`` in its frame) to live in world coordinates
        consistent with the new extrinsics:
        ``p_new = R_c2w_new @ p_cam + c_new``.

        Substituting yields ``p_new = T_c2w_new @ T_w2c_old @ p_old``, so
        ``T_delta = T_c2w_new @ T_w2c_old`` is the 4×4 transform that
        carries old-frame points to new-frame points *for points reconstructed
        by this camera*.
        """
        T_w2c_old = cls._ext_to_homo(old_ext)
        T_c2w_new = np.linalg.inv(cls._ext_to_homo(new_ext))
        return T_c2w_new @ T_w2c_old

    @staticmethod
    def _object_view_weights(
        snap: Dict[str, Any],
        num_cams: int,
    ) -> np.ndarray:
        """Build per-camera weights for applying camera-delta transforms.

        Winner-takes-all: returns a one-hot weight vector pointing at the
        object's primary observing camera (highest ``per_view_scores``;
        falls back to the first listed ``view`` if no scores are present).
        Returns the zero vector if the object has no detections at all
        (caller should leave that object alone).

        Why one-hot rather than a weighted blend: pixel-registration in
        camera *i* is preserved iff the object's new world position is
        exactly ``T_delta_i @ P_old``. A linear blend across cameras
        breaks the invariant in every camera simultaneously. Picking the
        primary observer keeps the projection exact in that camera; for
        objects also visible in other (refined) cameras, those secondaries
        do not agree exactly, which is inherent to the over-determined joint
        problem.
        """
        weights = np.zeros(num_cams, dtype=float)
        scores = snap.get("per_view_scores") or {}
        if scores:
            # Argmax over scored cameras.
            best_cam: Optional[int] = None
            best_score = -1.0
            for v, s in scores.items():
                vi = int(v)
                if not (0 <= vi < num_cams):
                    continue
                sv = float(s)
                if sv > best_score:
                    best_score = sv
                    best_cam = vi
            if best_cam is not None and best_score > 0.0:
                weights[best_cam] = 1.0
                return weights
        # Fallback: first listed view (object was detected somewhere but
        # carries no per-view score).
        for v in snap.get("views") or []:
            vi = int(v)
            if 0 <= vi < num_cams:
                weights[vi] = 1.0
                return weights
        return weights

    @staticmethod
    def _blend_apply_point(
        p_old: np.ndarray,
        deltas: List[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """Weighted-average application of camera deltas to a world point.

        new_p = sum_i w_i * (R_i @ p_old + t_i). Linear blend — well-defined
        for positions.
        """
        out = np.zeros(3, dtype=float)
        p_homo = np.append(np.asarray(p_old, dtype=float), 1.0)
        for w, T in zip(weights, deltas):
            if w <= 0.0:
                continue
            out += w * (T @ p_homo)[:3]
        return out

    @staticmethod
    def _blend_apply_dir(
        d_old: np.ndarray,
        deltas: List[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """Weighted application of rotation components to a unit direction.

        Applies only the 3×3 rotation portion of each delta, weighted-averages
        the results, then renormalizes. Falls back to ``d_old`` if the blend
        collapses to a zero vector (e.g., when two cameras' rotations are
        antipodal — a hint that the constraints are inconsistent).
        """
        out = np.zeros(3, dtype=float)
        d = np.asarray(d_old, dtype=float)
        for w, T in zip(weights, deltas):
            if w <= 0.0:
                continue
            out += w * (T[:3, :3] @ d)
        n = float(np.linalg.norm(out))
        if n < 1e-9:
            return d.copy()
        return out / n

    @staticmethod
    def _project_to_so3(M: np.ndarray) -> np.ndarray:
        """Find the closest rotation matrix to ``M`` via SVD.

        Required because the weighted blend of rotation matrices is
        generally not itself a rotation; SVD-based projection is the
        standard rotation-averaging trick (orthogonal Procrustes).
        """
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        # Ensure det=+1 (proper rotation, not reflection).
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
        return R

    def _co_transform_objects(
        self,
        deltas: List[np.ndarray],
    ) -> None:
        """Apply the per-camera delta transforms to every object.

        Each object is blended with weights derived from its per-view
        detection confidence. Position fields use linear weighted blends;
        rotation/direction fields blend then re-orthogonalize via SVD so
        the result remains a valid SO(3) element. Fields that live in the
        camera's pixel frame (bboxes, masks) are left untouched.
        """
        if self._original_objects is None:
            return
        num_cams = len(deltas)
        for obj, snap in zip(self._scene.objects, self._original_objects):
            weights = self._object_view_weights(snap, num_cams)
            if weights.sum() <= 1e-9:
                continue  # no observations to weight by

            # Positions
            obj.center_world = self._blend_apply_point(
                snap["center_world"], deltas, weights,
            )
            corners_old = snap["corners_world"]  # (8, 3)
            obj.corners_world = np.stack(
                [self._blend_apply_point(p, deltas, weights) for p in corners_old],
                axis=0,
            )
            if "world_points" in snap and obj.world_points is not None:
                pts_old = snap["world_points"]
                obj.world_points = np.stack(
                    [self._blend_apply_point(p, deltas, weights) for p in pts_old],
                    axis=0,
                )
            if "per_view_centers" in snap:
                obj.per_view_centers = {
                    k: self._blend_apply_point(v, deltas, weights)
                    for k, v in snap["per_view_centers"].items()
                }

            # Directions (apply weighted rotation portion, renormalize)
            obj.front_world = self._blend_apply_dir(
                snap["front_world"], deltas, weights,
            )
            obj.up_world = self._blend_apply_dir(
                snap["up_world"], deltas, weights,
            )
            obj.right_world = self._blend_apply_dir(
                snap["right_world"], deltas, weights,
            )
            if "per_view_fronts" in snap:
                obj.per_view_fronts = {
                    k: self._blend_apply_dir(v, deltas, weights)
                    for k, v in snap["per_view_fronts"].items()
                }

            # Rotation matrix — blend rotation columns then SVD-project
            R_old = snap["rotation_world"]
            R_blend = np.zeros((3, 3), dtype=float)
            for w, T in zip(weights, deltas):
                if w <= 0.0:
                    continue
                R_blend += w * (T[:3, :3] @ R_old)
            obj.rotation_world = self._project_to_so3(R_blend)

            # Derived fields: support_y from new corners; leave euler alone
            # (rarely used downstream, and a clean re-derivation requires
            # picking an Euler convention we don't currently expose).
            try:
                obj.support_y = float(np.min(obj.corners_world[:, 1]))
            except Exception:
                log.debug("suppressed: support_y re-derivation failed", exc_info=True)
                pass

    def _solve(self) -> None:
        """Run pose_solver against the original VGGT extrinsics + accumulated
        constraints, write refined extrinsics back into scene.cameras, AND
        co-transform every object's world-frame geometry so the scene stays
        internally consistent (cameras + objects agree on a common frame).
        """
        # Local import keeps pose_solver fully optional at scene-import time.
        from .pose_solver import solve_camera_poses

        if self._original_extrinsics is None:
            return  # nothing to solve
        new_extrinsics = solve_camera_poses(
            self._original_extrinsics,
            self._constraints,
            method="average",
        )

        # Per-camera 4×4 world deltas — must be computed BEFORE we mutate
        # cam.extrinsics, since they use the pre-update extrinsics on one side.
        deltas: List[np.ndarray] = []
        for old_ext, new_ext in zip(self._original_extrinsics, new_extrinsics):
            deltas.append(self._per_camera_world_delta(old_ext, new_ext))

        for cam, ext in zip(self._scene.cameras, new_extrinsics):
            cam.extrinsics = np.asarray(ext, dtype=float)
            cam.position_world = cam._extract_position()
            cam.heading = cam._extract_heading()

        self._co_transform_objects(deltas)
        # Re-apply face() constraints on top: the co-transform restarted
        # from the pre-face snapshot. Targets are re-resolved, so a face
        # toward a camera follows that camera's refined pose.
        for obj_idx, toward in self._faces:
            obj = self._scene.objects[obj_idx]
            self._set_object_axes(obj, *self._facing_axes(obj, toward))

        self._invalidate_scene_caches()

    def _invalidate_scene_caches(self) -> None:
        """Clear pose-derived caches on the parent scene (incl. anchor predicates)."""
        scene = self._scene
        scene._frame_cache = {}
        scene._distance_cache = None
        scene._frame_independent_cache = None
        scene._obj_relative_cache = None
        if hasattr(scene, "_rebuild_anchor_predicates"):
            scene._rebuild_anchor_predicates()
