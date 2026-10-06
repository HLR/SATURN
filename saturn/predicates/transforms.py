"""FrameNamespace transform helpers (pos / project / match_rotation / match_translation) — mixin for FrameNamespace.

Imports: may import saturn.scene.direction_utils; must not import saturn.predicates.frame at module level; must not import saturn.perception, saturn.vlm, saturn.serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; a runtime import would be circular
    from saturn.predicates.frame import FrameNamespace

import math

import numpy as np

from .scoring import label_yaw, warn_unresolved_labels


class _TransformMixin:
    @property
    def pos(self) -> np.ndarray:
        """Frame origin in world coordinates (alias for ``frame_origin``)."""
        return self._frame_origin


    def project(self, vec) -> tuple:
        """Project a world-frame vector onto this anchor's local axes,
        returning ``(x, y, z)`` in the scene's current axis convention.

        Without a convention set, the result is ``(sx, sy, sz)`` along
        the anchor's (right, up, front) — equivalent to manually doing
        ``np.dot(vec, anchor.frame_right)`` etc.

        With a convention set via ``scene.set_axis_convention(...)``, the
        returned scalars are signed/permuted to match the question's
        labeled (X, Y, Z) axes — so option patterns like ``(+1, -1)`` for
        ``"+X, -Z"`` can be matched directly against ``(x, z)``.
        """
        v = np.asarray(vec, dtype=float)
        sx = float(np.dot(v, self._frame_right))
        sy = float(np.dot(v, self._frame_up))
        sz = float(np.dot(v, self._frame_front))
        return self._scene._axis_project((sx, sy, sz))



    def match_rotation(self, to: "FrameNamespace", options: dict) -> str:
        """MCQ helper: which option best labels the rotation from self to ``to``?

        Handles both yaw (left/right/cardinal) and pitch (up/down) labels.
        For each option label, computes a signed alignment score on the
        appropriate axis, then returns the key with the highest score.

        Typical usage for camera-rotation MCQ items::

            return v0.match_rotation(v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"})
        """
        yaw_deg, pitch_deg = self.rotation_to(to)
        yaw_deg_norm = yaw_deg % 360.0

        _PITCH_UP = {"up", "upward", "upwards", "tiltup", "tiltsup", "updirection"}
        _PITCH_DOWN = {
            "down", "downward", "downwards", "tiltdown", "tiltsdown",
            "downdirection",
        }

        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(" ", "").strip()

        def _ang_dist(a: float, b: float) -> float:
            d = abs(a - b) % 360.0
            return d if d <= 180.0 else 360.0 - d

        # Classify each option as pitch-axis or yaw-axis, then compare within axis.
        pitch_opts: dict = {}   # key -> signed score (positive = good match)
        yaw_opts: dict = {}     # key -> angular distance (lower = better)
        label_yaws: dict = {}   # key -> label yaw (None = not understood)
        for key, lab in options.items():
            n = _norm(str(lab))
            if n in _PITCH_UP:
                pitch_opts[key] = pitch_deg
            elif n in _PITCH_DOWN:
                pitch_opts[key] = -pitch_deg
            else:
                y = label_yaw(lab)
                label_yaws[key] = y
                yaw_opts[key] = 180.0 if y is None else _ang_dist(yaw_deg_norm, y)
        warn_unresolved_labels("match_rotation", options, label_yaws)

        if pitch_opts and yaw_opts:
            # Mixed option set — use the dominant rotation axis to decide.
            if abs(pitch_deg) >= abs(yaw_deg):
                return max(pitch_opts, key=lambda k: pitch_opts[k])
            else:
                return min(yaw_opts, key=lambda k: yaw_opts[k])
        elif pitch_opts:
            return max(pitch_opts, key=lambda k: pitch_opts[k])
        elif yaw_opts:
            return min(yaw_opts, key=lambda k: yaw_opts[k])
        return next(iter(options))


    def match_translation(self, to: "FrameNamespace", options: dict) -> str:
        """MCQ helper: which option best labels the translation from self to ``to``?

        Companion to :meth:`match_rotation` for camera motion that is
        dominated by translation rather than rotation (e.g., the operator
        stepped up/down/left/right while keeping the same heading).  Uses
        the same Up/Down/cardinal/relative label vocabulary; scoring is
        identical to :meth:`match_rotation` but the underlying angles come
        from :meth:`translation_to`.

        Typical usage::

            return v0.match_translation(v1, {"A": "Up", "B": "Down", "C": "Left", "D": "Right"})
        """
        yaw_deg, pitch_deg = self.translation_to(to)
        yaw_deg_norm = yaw_deg % 360.0
        # For translation, axis dominance is by projected length, not by
        # angle: a displacement of (ε, 1, 0) yields a huge yaw angle even
        # though the vertical component dominates.  Recompute raw
        # projections to drive the mixed-axis tie-breaker.
        disp = np.asarray(to._frame_origin, dtype=float) - self._frame_origin
        r_proj = float(np.dot(disp, self._frame_right))
        u_proj = float(np.dot(disp, self._frame_up))
        f_proj = float(np.dot(disp, self._frame_front))
        horiz_len = math.hypot(r_proj, f_proj)
        vert_len = abs(u_proj)

        _PITCH_UP = {"up", "upward", "upwards", "above", "updirection"}
        _PITCH_DOWN = {
            "down", "downward", "downwards", "below", "downdirection",
        }

        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(" ", "").strip()

        def _ang_dist(a: float, b: float) -> float:
            d = abs(a - b) % 360.0
            return d if d <= 180.0 else 360.0 - d

        pitch_opts: dict = {}
        yaw_opts: dict = {}
        label_yaws: dict = {}
        for key, lab in options.items():
            n = _norm(str(lab))
            if n in _PITCH_UP:
                pitch_opts[key] = pitch_deg
            elif n in _PITCH_DOWN:
                pitch_opts[key] = -pitch_deg
            else:
                y = label_yaw(lab)
                label_yaws[key] = y
                yaw_opts[key] = 180.0 if y is None else _ang_dist(yaw_deg_norm, y)
        warn_unresolved_labels("match_translation", options, label_yaws)

        if pitch_opts and yaw_opts:
            if vert_len >= horiz_len:
                return max(pitch_opts, key=lambda k: pitch_opts[k])
            else:
                return min(yaw_opts, key=lambda k: yaw_opts[k])
        elif pitch_opts:
            return max(pitch_opts, key=lambda k: pitch_opts[k])
        elif yaw_opts:
            return min(yaw_opts, key=lambda k: yaw_opts[k])
        return next(iter(options))
