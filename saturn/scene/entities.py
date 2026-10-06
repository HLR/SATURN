"""Scene entity accessors (object/camera positions, axes, index resolution) — mixin for Scene.

Imports: may import saturn.soft_logic / saturn.scene.types; must not import saturn.perception, saturn.vlm, saturn.serving."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from saturn.soft_logic.tensor import ProbabilisticTensor
from .types import Camera, MergedObject


class _EntityAccessMixin:
    def _object_positions(self) -> np.ndarray:
        """(N, 3) world-frame centers of all objects."""
        if not self.objects:
            return np.zeros((0, 3), dtype=float)
        return np.array([obj.center_world for obj in self.objects], dtype=float)


    def _object_front_directions(self) -> np.ndarray:
        """(N, 3) world-frame unit front direction per object (``obj.front_vec``)."""
        if not self.objects:
            return np.zeros((0, 3), dtype=float)
        return np.array([obj.front_vec for obj in self.objects], dtype=float)


    def _object_right_directions(self) -> np.ndarray:
        """(N, 3) world-frame unit body-right direction per object (``obj.right_vec``)."""
        if not self.objects:
            return np.zeros((0, 3), dtype=float)
        return np.array([obj.right_vec for obj in self.objects], dtype=float)



    def _object_elevations(self) -> np.ndarray:
        """(N,) elevation angles in degrees per object (positive = front tilted up).

        Read from the front vector: ``euler_world_deg[1]`` is the middle angle
        of a scipy "yxz" decomposition, whose sign depends on the azimuth and
        which is 0 for side-facing objects, so it is not an elevation.
        """
        if not self.objects:
            return np.zeros(0, dtype=float)
        return np.array(
            [
                math.degrees(math.asin(float(np.clip(
                    obj.front_vec[1] / (np.linalg.norm(obj.front_vec) + 1e-12), -1.0, 1.0))))
                if obj.euler_world_deg is not None else 0.0
                for obj in self.objects
            ],
            dtype=float,
        )


    def _entity_front_directions(self) -> np.ndarray:
        """(N+C, 3) world-frame front direction per entity.

        For objects: ``obj.front_vec``.
        For cameras: the camera's view forward axis in world coordinates,
        i.e. third row of the world-to-camera rotation matrix.
        """
        obj_fronts = self._object_front_directions()
        if not self.cameras:
            return obj_fronts
        cam_fronts = []
        for cam in self.cameras:
            try:
                _, _, front_world, _ = self._camera_frame_axes(cam)
                cam_fronts.append(np.asarray(front_world, dtype=float))
            except Exception:
                cam_fronts.append(np.zeros(3, dtype=float))
        cam_fronts = np.array(cam_fronts, dtype=float) if cam_fronts else np.zeros(
            (0, 3), dtype=float
        )
        if len(obj_fronts) == 0:
            return cam_fronts
        return np.concatenate([obj_fronts, cam_fronts], axis=0)


    def _entity_right_directions(self) -> np.ndarray:
        """(N+C, 3) world-frame right direction per entity.

        For objects: ``obj.right_vec``.  For cameras: the right axis of
        the camera frame in world coordinates (first row of the world-to-
        camera rotation matrix = image-right, same sign as
        ``cam.right_vec``; unlike ``cam.right_vec`` it keeps camera roll).
        """
        obj_rights = self._object_right_directions()
        if not self.cameras:
            return obj_rights
        cam_rights = []
        for cam in self.cameras:
            try:
                right_world, _, _, _ = self._camera_frame_axes(cam)
                cam_rights.append(np.asarray(right_world, dtype=float))
            except Exception:
                cam_rights.append(np.zeros(3, dtype=float))
        cam_rights = np.array(cam_rights, dtype=float) if cam_rights else np.zeros(
            (0, 3), dtype=float
        )
        if len(obj_rights) == 0:
            return cam_rights
        return np.concatenate([obj_rights, cam_rights], axis=0)


    def _entity_elevations(self) -> np.ndarray:
        """(N+C,) elevation angles in degrees per entity.

        Cameras are assumed to have 0° elevation (horizontal); they are
        rarely the subject of obj_facing_up/down queries.
        """
        obj_elev = self._object_elevations()
        if not self.cameras:
            return obj_elev
        cam_elev = np.zeros(len(self.cameras), dtype=float)
        if len(obj_elev) == 0:
            return cam_elev
        return np.concatenate([obj_elev, cam_elev], axis=0)


    def _num_entities(self) -> int:
        """Total number of entities in the mixed entity space."""
        return len(self.objects) + len(self.cameras)


    def _wrap_tensor(self, arr: np.ndarray, ndim: int) -> ProbabilisticTensor:
        """Wrap a numpy array as a ProbabilisticTensor with standard variable names.

        Parameters
        ----------
        arr : numpy array
        ndim : 1 -> vars=["x1"], 2 -> vars=["x1","x2"], 3 -> vars=["x1","x2","x3"]
        """
        t = torch.tensor(arr, dtype=torch.float32)
        if ndim == 1:
            var_names = ["x1"]
        elif ndim == 2:
            var_names = ["x1", "x2"]
        elif ndim == 3:
            var_names = ["x1", "x2", "x3"]
        else:
            var_names = [f"x{i + 1}" for i in range(ndim)]
        return ProbabilisticTensor(t, vars=var_names)


    def _camera_frame_axes(self, camera: Camera):
        """Extract right/up/front axes and origin for a camera frame."""
        ext = np.asarray(camera.extrinsics, dtype=float)
        R_w2c = ext[:3, :3]
        # Camera frame in world coordinates:
        #   camera X (right)  = R_w2c^T @ [1,0,0] = first column of R_w2c^T = first row of R_w2c
        #   camera Y (down)   = R_w2c^T @ [0,1,0] = second row of R_w2c
        #   camera Z (forward)= R_w2c^T @ [0,0,1] = third row of R_w2c
        # We use Y-up convention: right = cam_X, up = -cam_Y, front = cam_Z
        right_world = R_w2c[0, :]
        down_world = R_w2c[1, :]
        front_world = R_w2c[2, :]
        up_world = -down_world

        origin = np.asarray(camera.position_world, dtype=float)
        return right_world, up_world, front_world, origin


    def _object_frame_axes(self, obj: MergedObject):
        """Extract right/up/front axes and origin for an object frame."""
        right, up, front = obj.right_vec, obj.up_vec, obj.front_vec
        origin = np.asarray(obj.center_world, dtype=float)
        return right, up, front, origin


    @staticmethod
    def _camera_hfov_deg(camera: Camera) -> Optional[float]:
        """Compute horizontal field-of-view in degrees from intrinsics.

        Returns None if intrinsics are unavailable or degenerate.
        """
        try:
            K = np.asarray(camera.intrinsics, dtype=float)
            fx = K[0, 0]
            if fx < 1e-6:
                return None
            W = float(camera.image_size[1])  # (H, W)
            if W <= 0:  # e.g. MergedObject.clone() cameras: image_size=(0, 0)
                return None
            hfov = float(2.0 * math.degrees(math.atan(W / (2.0 * fx))))
            return hfov if math.isfinite(hfov) and hfov > 0 else None
        except Exception:
            return None


    def _entity_at(self, idx: int):
        """Resolve a unified (K+C) index to the underlying object or camera.

        Mirrors the indexing convention used by ``anchor.first_person.<label>``:
        ``0..K-1`` selects ``scene.objects[i]``; ``K..K+C-1`` selects
        ``scene.cameras[i-K]``.
        """
        idx = int(idx)
        K = len(self.objects)
        if 0 <= idx < K:
            return self.objects[idx]
        C = len(self.cameras)
        if K <= idx < K + C:
            return self.cameras[idx - K]
        raise IndexError(
            f"Scene._entity_at: index {idx} out of range [0, {K + C})"
        )


    def _resolve_entity_object(self, entity) -> Optional[MergedObject]:
        """Return the MergedObject if *entity* resolves to one, else None."""
        if isinstance(entity, MergedObject):
            return entity
        if isinstance(getattr(entity, "cam", None), Camera):
            return None  # camera(N): a camera, not an object
        if isinstance(entity, (int, np.integer)):
            return self.objects[int(entity)]
        if (
            isinstance(entity, (tuple, list))
            and len(entity) == 2
            and isinstance(entity[0], str)
        ):
            kind, idx = entity
            if kind.lower() in ("object", "obj"):
                return self.objects[idx]
        return None


    def _resolve_entity(
        self,
        entity,
        *,
        label: str = "entity",
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Resolve an entity specification to (position, forward_vec, right_vec).

        Accepted forms:
          - A ``Camera`` instance
          - A ``MergedObject`` instance
          - ``("camera", cam_id)``  or  ``("cam", cam_id)``
          - ``("object", obj_id)``  or  ``("obj", obj_id)``
          - ``int``                — treated as object index
          - a description (``score(...).iota(v)``) — the object it names
          - ``np.ndarray`` / list of 3 numbers — a 3D world point (no axes)

        Returns (position, forward, right).  forward/right are None for
        bare 3D points.
        """
        # A named object's description: the object it names, description.assign()
        # (exactly what scene.frame(at=description) stands at).
        if isinstance(entity, ProbabilisticTensor):
            entity = self._described_object(entity)
        # scene.frame(position=p) before it has a facing: its position.
        if getattr(entity, "_unfaced_position", None) is not None:
            return np.asarray(entity._unfaced_position, dtype=float), None, None
        # camera(N) from generated programs: an int index that carries its
        # Camera. A bare int means an OBJECT index, so unwrap it first.
        if isinstance(getattr(entity, "cam", None), Camera):
            entity = entity.cam
        # --- Direct instance forms ---
        if isinstance(entity, Camera):
            right, up, front, origin = self._camera_frame_axes(entity)
            return origin, front, right
        if isinstance(entity, MergedObject):
            right, up, front, origin = self._object_frame_axes(entity)
            return origin, front, right

        # --- Tuple forms: ("camera", idx) / ("object", idx) ---
        if (
            isinstance(entity, (tuple, list))
            and len(entity) == 2
            and isinstance(entity[0], str)
        ):
            kind, idx = entity
            kind = kind.lower()
            if kind in ("camera", "cam"):
                cam = self.cameras[idx]
                right, up, front, origin = self._camera_frame_axes(cam)
                return origin, front, right
            elif kind in ("object", "obj"):
                obj = self.objects[idx]
                right, up, front, origin = self._object_frame_axes(obj)
                return origin, front, right
            else:
                raise ValueError(
                    f"Unknown entity kind '{kind}'. Use 'camera' or 'object'."
                )

        # --- Bare integer → object index ---
        if isinstance(entity, (int, np.integer)):
            obj = self.objects[int(entity)]
            right, up, front, origin = self._object_frame_axes(obj)
            return origin, front, right

        # --- Otherwise it must be a 3D point (3 coordinates; never a predicate's scores) ---
        if type(entity).__name__ == "PredicateArray":
            raise TypeError(f"{label}: a predicate's scores are not a place; name the object with its "
                            "description, score(...).iota(\"x2\"), or pass camera(N) or a 3D point.")
        try:
            pos = np.asarray(entity, dtype=float).ravel()
        except (TypeError, ValueError):
            pos = None
        if pos is None or pos.size != 3:
            raise TypeError(f"{label}: expected a named object's description, camera(N) or a 3D point; "
                            f"got {type(entity).__name__}.")
        return pos, None, None


    @property
    def is_camera(self) -> ProbabilisticTensor:
        """1D predicate: 1.0 for camera entities, 0.0 for objects.

        Shape is ``(N+C,)``.  Use this to constrain quantification to
        cameras when both objects and cameras live in the same tensor:

            # Which camera is behind the chair?
            chair = score("is the object a chair?").iota("x1").argmax()
            view = scene.frame(at=score("is the object a chair?").iota("x2"))
            cam_behind = (
                scene.is_camera & view.behind.iota("x2", int(chair))
            ).argmax()  # entity index of the winning camera
        """
        N = len(self.objects)
        total = self._num_entities()
        arr = np.zeros(total, dtype=np.float32)
        arr[N:] = 1.0
        return self._wrap_tensor(arr, ndim=1)


    @property
    def is_object(self) -> ProbabilisticTensor:
        """1D predicate: 1.0 for object entities, 0.0 for cameras.

        Shape ``(N+C,)``.  Useful for ensuring an object-class score
        (``car``, ``chair``, ...) only fires on object slots.
        """
        N = len(self.objects)
        total = self._num_entities()
        arr = np.zeros(total, dtype=np.float32)
        arr[:N] = 1.0
        return self._wrap_tensor(arr, ndim=1)


    def object(self, idx: int) -> ProbabilisticTensor:
        """One-hot predicate over the entity space (objects + cameras).

        ``idx`` is an object index in ``[0, len(self.objects))``.  Returns
        a ``(N+C,)`` tensor with 1.0 at position ``idx`` and 0.0 elsewhere.
        Sized to the mixed entity space so it composes with the spatial
        relation tensors produced by ``scene.frame(...)``.
        """
        total = self._num_entities()
        arr = np.zeros(total, dtype=np.float32)
        if 0 <= idx < total:
            arr[idx] = 1.0
        return self._wrap_tensor(arr, ndim=1)


    def camera(self, idx: int) -> ProbabilisticTensor:
        """One-hot predicate selecting a single camera entity.

        ``idx`` is a camera index in ``[0, len(self.cameras))``; the
        returned tensor places 1.0 at entity slot ``len(self.objects) + idx``.
        Use this to refer to a specific camera in symbolic queries:

            ``scene.camera(2) & view.behind("x1", chair_idx)``
        """
        total = self._num_entities()
        arr = np.zeros(total, dtype=np.float32)
        cam_entity_id = len(self.objects) + idx
        if 0 <= cam_entity_id < total:
            arr[cam_entity_id] = 1.0
        return self._wrap_tensor(arr, ndim=1)
