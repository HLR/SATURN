"""User-registered predicate wrappers (pairwise / k-ary / object-centric) returned by scene.register_predicate access paths.

Imports: may import saturn.predicates.scoring; must not import saturn.predicates.frame or saturn.scene.scene at module level; must not import saturn.perception, saturn.vlm, saturn.serving."""

from __future__ import annotations

from typing import Any, Dict

from .scoring import _compute_pairwise_score, _entity_pose


class _RegisteredPairwiseWrapper:
    """Indexable handle for a registered ``h_r`` / ``angle_deg`` predicate.

    Returned from ``anchor.third_person.<name>`` (which expects 2D
    ``[i, j]`` indexing); the corresponding 1D ``anchor.first_person.<name>``
    is materialized eagerly as a ``np.ndarray`` and does NOT go through this
    wrapper.  See ``_FirstPersonNamespace._compute`` for the eager path.
    """

    __slots__ = ("_scene", "_spec", "_view", "_name")

    def __init__(self, scene, spec: Dict[str, Any], view, name: str):
        self._scene = scene
        self._spec = spec
        self._view = view
        self._name = name

    def __getitem__(self, key) -> float:
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError(
                f"view.third_person.{self._name}: expected 2-tuple [i, j], "
                f"got {key!r}"
            )
        return _compute_pairwise_score(
            self._view, self._scene, self._spec, int(key[0]), int(key[1])
        )

    def __repr__(self) -> str:
        return (
            f"<RegisteredPairwisePredicate name={self._name!r} "
            f"kind={self._spec['kind']!r}>"
        )


class _RegisteredKaryWrapper:
    """Indexable handle for a registered K-ary ``fn`` predicate (K>=2).

    Only reachable via ``anchor.third_person.<name>``.  Each subscript dispatches
    to ``fn(scene, e_{i_1}, ..., e_{i_K})`` and returns ``float``.  No sigmoid
    is applied — ``fn`` is expected to return a final score in ``[0, 1]``.
    """

    __slots__ = ("_scene", "_fn", "_arity", "_name")

    def __init__(self, scene, fn, arity: int, name: str):
        self._scene = scene
        self._fn = fn
        self._arity = arity
        self._name = name

    def __getitem__(self, key) -> float:
        if self._arity == 1:
            # arity=1 fn predicates go through first_person only; this branch
            # is here defensively but should not be reached.
            if isinstance(key, tuple):
                if len(key) != 1:
                    raise IndexError(
                        f"predicate {self._name!r}: arity=1 expects a single "
                        f"index, got tuple of length {len(key)}"
                    )
                key = key[0]
            ent = self._scene._entity_at(int(key))
            return float(self._fn(self._scene, ent))
        if not isinstance(key, tuple):
            raise IndexError(
                f"view.third_person.{self._name}: arity={self._arity} expects "
                f"a {self._arity}-tuple, got single index {key!r}"
            )
        if len(key) != self._arity:
            raise IndexError(
                f"view.third_person.{self._name}: arity={self._arity} expects "
                f"{self._arity} indices, got {len(key)}"
            )
        ents = [self._scene._entity_at(int(k)) for k in key]
        return float(self._fn(self._scene, *ents))

    def __repr__(self) -> str:
        return (
            f"<RegisteredKaryPredicate name={self._name!r} arity={self._arity}>"
        )


class _RegisteredObjCentricWrapper:
    """Indexable handle for the object-centric access path ``scene.obj_<name>``.

    Implements the paper's $S^{obj}_r[i, j] \\equiv S^{a_j}_r[i, j]$: each
    subscript ``[i, j]`` uses **entity j itself** as the FoR anchor (j supplies
    both the local frame axes and the reference position).  Mirrors the
    behavior of the built-in ``scene.obj_left`` / ``obj_right`` / ``obj_front``
    / ``obj_behind`` matrices — i.e., "i is <name>-of j, from j's perspective".

    Lazily builds a view per access via ``scene.frame(position=, orientation=)``.
    The computation reuses ``_compute_pairwise_score`` with the j-rooted view
    and j_idx=None (first-person collapse from j's view): with j as the anchor,
    the displacement is x_i - x_j, R_j_local becomes the identity, and R_i_local
    is i's orientation in j's local frame — exactly the paper's S^{a_j}_r.
    """

    __slots__ = ("_scene", "_spec", "_name")

    def __init__(self, scene, spec: Dict[str, Any], name: str):
        self._scene = scene
        self._spec = spec
        self._name = name

    def __getitem__(self, key) -> float:
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError(
                f"scene.obj_{self._name}: expected 2-tuple [i, j], got {key!r}"
            )
        i_idx, j_idx = int(key[0]), int(key[1])
        entity_j = self._scene._entity_at(j_idx)
        pos_j, R_j = _entity_pose(entity_j)
        view_at_j = self._scene.frame(position=pos_j, orientation=R_j)
        return _compute_pairwise_score(
            view_at_j, self._scene, self._spec, i_idx, None
        )

    def __repr__(self) -> str:
        return (
            f"<RegisteredObjCentricPredicate name={self._name!r} "
            f"kind={self._spec['kind']!r}>"
        )
