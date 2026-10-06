"""
Shared direction constants and utilities.

Single source of truth for:
- Cardinal/relative label lists and angle mappings
- Bidirectional label translation (cardinal ↔ relative)
- North-vector resolution from anchor pairs or explicit vectors
- Direction classification (diff vector → label string)
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

# ======================================================================
# Label constants
# ======================================================================

CARDINAL_TO_ANGLE = {
    "north": 0.0,
    "n": 0.0,
    "northeast": 45.0,
    "ne": 45.0,
    "east": 90.0,
    "e": 90.0,
    "southeast": 135.0,
    "se": 135.0,
    "south": 180.0,
    "s": 180.0,
    "southwest": 225.0,
    "sw": 225.0,
    "west": 270.0,
    "w": 270.0,
    "northwest": 315.0,
    "nw": 315.0,
}

class DirectionLabel(str):
    """A label the engine emits: a plain ``str`` in canonical spelling whose
    ``==`` is alias-aware, so ``view.direction(...).label(8) == "behind-left"``
    is True when the engine says ``"back-left"``.  Hashes as its canonical
    spelling (dict keys / sets should use canonical strings)."""

    __slots__ = ()

    def __eq__(self, other):
        if isinstance(other, str):
            return canonical_direction(self) == canonical_direction(other)
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = str.__hash__


CARDINAL_LABELS_8 = [DirectionLabel(x) for x in (
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
)]
CARDINAL_LABELS_4 = [DirectionLabel(x) for x in ("north", "east", "south", "west")]

# Relative labels — "back" (not "behind") is the canonical spelling; aliases
# (behind, rear, forward, ...) are folded by ``canonical_direction`` below.
RELATIVE_LABELS_8 = [DirectionLabel(x) for x in (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)]
RELATIVE_LABELS_4 = [DirectionLabel(x) for x in ("front", "right", "back", "left")]

# ======================================================================
# Bidirectional translation tables
# ======================================================================

RELATIVE_TO_CARDINAL = {
    "front": "north",
    "back": "south",
    "left": "west",
    "right": "east",
    "front-left": "northwest",
    "front-right": "northeast",
    "front left": "northwest",
    "front right": "northeast",
    "back-left": "southwest",
    "back-right": "southeast",
    "back left": "southwest",
    "back right": "southeast",
}

CARDINAL_TO_RELATIVE = {
    "north": "front",
    "south": "back",
    "west": "left",
    "east": "right",
    "northwest": "front-left",
    "northeast": "front-right",
    "southwest": "back-left",
    "southeast": "back-right",
}


# ======================================================================
# Direction-label aliases
# ======================================================================

# ONE canonical spelling per direction: the one the engine emits
# (RELATIVE_LABELS_8 / CARDINAL_LABELS_8): "front" / "back" and
# unhyphenated cardinals ("northeast").  Every alias below means exactly the
# canonical word it maps to.  ("behind" in the THIRD-person namespace,
# ``view.third_person.behind[i, j]``, is a different, pairwise predicate and
# never goes through this table.)
_DIRECTION_WORD_ALIASES = {
    "behind": "back",
    "behinds": "back",
    "rear": "back",
    "forward": "front",
    "forwards": "front",
}


def canonical_direction(label: str) -> str:
    """Canonical spelling of a direction label; aliases collapse to one string.

    Case, whitespace, ``_`` and ``-`` are normalised, alias words are
    replaced, and front/back is put before left/right::

        canonical_direction("behind_left")   -> "back-left"
        canonical_direction("Forward Right") -> "front-right"
        canonical_direction("rear")          -> "back"
        canonical_direction("left-front")    -> "front-left"
        canonical_direction("north-east")    -> "northeast"

    Labels that are not directions (``"above"``, user predicate names, typos)
    come back normalised but otherwise unchanged, so callers keep their own
    "unknown label" errors.
    """
    words = [w for w in re.split(r"[\s_\-]+", str(label).strip().lower()) if w]
    words = [_DIRECTION_WORD_ALIASES.get(w, w) for w in words]
    if len(words) == 2 and words[0] in ("left", "right") and words[1] in ("front", "back"):
        words.reverse()
    if len(words) > 1 and "".join(words) in CARDINAL_TO_ANGLE:
        return "".join(words)  # "north-east" -> "northeast"
    return "-".join(words)



def is_relative_label(label: str) -> bool:
    """Return True if *label* is a relative direction (front/back/left/right/...),
    including aliases (behind, rear, forward, behind-left, ...)."""
    return canonical_direction(label) in RELATIVE_TO_CARDINAL


def translate_label(label: str, *, to: str = "cardinal") -> str:
    """Translate a label between relative and cardinal systems.

    Parameters
    ----------
    label : str
        A direction label (e.g. "front", "north", "back-left", "southeast");
        aliases are accepted (see :func:`canonical_direction`).
    to : str
        ``"cardinal"`` — translate relative → cardinal (no-op if already cardinal).
        ``"relative"`` — translate cardinal → relative (no-op if already relative).

    Returns
    -------
    str — the translated label, in canonical spelling.

    Raises
    ------
    ValueError if the label is unrecognized in either system.
    """
    key = canonical_direction(label)
    if to == "cardinal":
        if key in RELATIVE_TO_CARDINAL:
            return RELATIVE_TO_CARDINAL[key]
        # Already cardinal (or abbreviation)?
        if key in CARDINAL_TO_ANGLE:
            return key
        raise ValueError(
            f"Unknown direction label '{label}'. "
            f"Expected relative (front/back/left/right/...) or cardinal."
        )
    elif to == "relative":
        if key in CARDINAL_TO_RELATIVE:
            return CARDINAL_TO_RELATIVE[key]
        # Already relative?
        if key in RELATIVE_TO_CARDINAL:
            return key
        raise ValueError(
            f"Unknown direction label '{label}'. "
            f"Expected cardinal (north/south/east/west/...) or relative."
        )
    else:
        raise ValueError(f"'to' must be 'cardinal' or 'relative', got '{to}'.")


# ======================================================================
# Core direction classification
# ======================================================================


def classify_direction(
    diff: np.ndarray,
    north_vec: np.ndarray,
    freedom: int = 8,
    *,
    labels: str = "cardinal",
) -> str:
    """Classify a horizontal displacement vector into a direction label.

    This is the **single implementation** behind the scene's cardinal
    labels (cardinals are scene-level only).

    Parameters
    ----------
    diff : ndarray (3,)
        World-space displacement vector (target − source).
    north_vec : ndarray (3,)
        Unit vector defining "north" (or "front" in relative mode) on the
        horizontal plane.  Must be pre-normalized and Y-flattened.
    freedom : int
        4 or 8 — number of output buckets.
    labels : str
        ``"cardinal"`` — output north/south/east/west/... labels.
        ``"relative"`` — output front/back/left/right/... labels.

    Returns
    -------
    str — the classified label.
    """
    # Project to horizontal
    diff_h = diff.copy()
    diff_h[1] = 0.0

    # East = cross(up, north)  — right-hand rule: Y_up × north = east
    # This convention is used consistently everywhere.
    east_vec = np.cross(np.array([0.0, 1.0, 0.0]), north_vec)
    en = np.linalg.norm(east_vec)
    east_vec = east_vec / en if en > 1e-9 else np.array([1.0, 0.0, 0.0])

    north_comp = np.dot(diff_h, north_vec)
    east_comp = np.dot(diff_h, east_vec)

    # Angle: 0° = north, increasing clockwise through east
    angle_deg = math.degrees(math.atan2(east_comp, north_comp)) % 360.0

    # Select label set
    if labels == "cardinal":
        label_list = CARDINAL_LABELS_8 if freedom == 8 else CARDINAL_LABELS_4
    elif labels == "relative":
        label_list = RELATIVE_LABELS_8 if freedom == 8 else RELATIVE_LABELS_4
    else:
        raise ValueError(f"labels must be 'cardinal' or 'relative', got '{labels}'")

    if freedom not in (4, 8):
        raise ValueError(f"freedom must be 4 or 8, got {freedom}")

    step = 360.0 / freedom
    half = step / 2.0
    idx = int((angle_deg + half) / step) % len(label_list)
    return label_list[idx]


# ======================================================================
# North-vector resolution
# ======================================================================


def resolve_north(
    *,
    anchor_pos: Optional[np.ndarray] = None,
    reference_pos: Optional[np.ndarray] = None,
    anchor_cardinal: Optional[str] = None,
    north_vector: Optional[np.ndarray] = None,
    fallback_forward: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Determine the world-space north vector from user-supplied cardinal frame.

    Three modes (checked in order):

    **Mode A — Anchor pair:**
        ``anchor_pos``, ``reference_pos``, ``anchor_cardinal`` are all provided.
        The vector reference → anchor is at the given cardinal angle,
        and north is derived by rotating back.

    **Mode B — Explicit north:**
        ``north_vector`` is provided directly.

    **Mode C — Fallback:**
        ``fallback_forward`` is used as north.

    Returns
    -------
    ndarray (3,) — unit north vector on the horizontal plane.
    """
    if (
        anchor_pos is not None
        and reference_pos is not None
        and anchor_cardinal is not None
    ):
        anchor_vec = anchor_pos - reference_pos
        anchor_vec[1] = 0.0
        anchor_norm = np.linalg.norm(anchor_vec)
        if anchor_norm < 1e-9:
            # Degenerate — fall through to fallback
            return _flatten_or_default(fallback_forward)
        anchor_dir = anchor_vec / anchor_norm
        angle = CARDINAL_TO_ANGLE.get(anchor_cardinal.lower().strip())
        if angle is None:
            raise ValueError(f"Unknown cardinal label: '{anchor_cardinal}'")
        ax, az = anchor_dir[0], anchor_dir[2]
        world_angle = math.degrees(math.atan2(ax, az))
        north_angle = world_angle - angle
        nr = math.radians(north_angle)
        return np.array([math.sin(nr), 0.0, math.cos(nr)], dtype=float)

    if north_vector is not None:
        nv = np.asarray(north_vector, dtype=float).copy()
        nv[1] = 0.0
        n = np.linalg.norm(nv)
        if n < 1e-9:
            return _flatten_or_default(fallback_forward)
        return nv / n

    return _flatten_or_default(fallback_forward)


def _flatten_or_default(vec: Optional[np.ndarray]) -> np.ndarray:
    """Flatten to horizontal and normalize; default to +Z if degenerate."""
    if vec is not None:
        nv = np.asarray(vec, dtype=float).copy()
        nv[1] = 0.0
        n = np.linalg.norm(nv)
        if n > 1e-9:
            return nv / n
    return np.array([0.0, 0.0, 1.0], dtype=float)


def as_vector3(value, where: str) -> np.ndarray:
    """``value`` as a float 3-vector, or a TypeError that says what was passed.
    A method passed without calling it (``scene.cardinal_vector``) is the usual
    mistake, so it is named explicitly."""
    if callable(value) and not isinstance(value, np.ndarray):
        name = getattr(value, "__name__", type(value).__name__)
        raise TypeError(f"{where}: expected a 3-vector but got the function/method {name!r}; "
                        f"call it to get the vector, e.g. {name}(...).")
    try:
        v = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        raise TypeError(f"{where}: expected a 3-vector (x, y, z), got {type(value).__name__}.") from None
    if v.size < 3:
        raise TypeError(f"{where}: expected a 3-vector (x, y, z), got {v.size} number(s).")
    return v[:3].astype(float)

