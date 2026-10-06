"""Continuous spatial metrics (distances, angles, overlaps) that the soft predicates are built from.

Pure geometry; must not import saturn.perception/vlm/serving.
"""
import numpy as np


def compute_obj_relative_matrices(
    front_comp: np.ndarray,
    right_comp: np.ndarray,
    confidences: np.ndarray = None,
    *,
    confidence_threshold: float = 0.0,
    fallback_left: np.ndarray = None,
    fallback_right: np.ndarray = None,
    fallback_front: np.ndarray = None,
    fallback_behind: np.ndarray = None,
    weight_by_confidence: bool = False,
) -> dict:
    """Frame-independent object-centric directional relations.

    Builds the ``obj_left/right/front/behind`` (K, K) matrices from the
    pre-computed orientation components with the egocentric direction-label
    kernel: all four directions are scored as ``(1 +/- component) / 2``.

    Conventions
    -----------
    ``front_comp[i, j] = (pos_j - pos_i) · front_i / |·|``  — where j sits
    along i's intrinsic forward axis.  Same for ``right_comp``.

    The output uses the **subject-first** convention::

        obj_left[i, j]   = "i is to the left of j, from j's perspective"
        obj_right[i, j]  = "i is to the right of j, from j's perspective"
        obj_front[i, j]  = "i is in front of j, from j's perspective"
        obj_behind[i, j] = "i is behind j, from j's perspective"

    Note the index swap below (``rc = right_comp[j, i]``) that converts
    "where j sits in i's frame" into "where i sits in j's frame".

    Parameters
    ----------
    front_comp, right_comp : (K, K) signed components.  Index 0 is the
        anchor whose intrinsic axes were used; index 1 is the other entity
        whose position was projected onto those axes.
    confidences : optional (K,) per-entity orientation confidences.  If an
        entry is below ``confidence_threshold`` (or ``None``), the
        corresponding column j of the output falls back to the matching
        camera-frame matrix (``fallback_*[:, j]``) when supplied; otherwise
        zeros.
    fallback_* : optional (K, K) camera-frame relation matrices used when
        an entity's orientation confidence is below threshold.  Must
        either all be supplied together or all be ``None``.
    weight_by_confidence : if True, multiply each non-fallback column by
        the column's anchor confidence.

    Returns
    -------
    Dict with keys ``obj_left``, ``obj_right``, ``obj_front``,
    ``obj_behind``; each ``(K, K)`` float array in ``[0, 1]``.
    """
    K = front_comp.shape[0]
    out = {
        "obj_left": np.zeros((K, K), dtype=float),
        "obj_right": np.zeros((K, K), dtype=float),
        "obj_front": np.zeros((K, K), dtype=float),
        "obj_behind": np.zeros((K, K), dtype=float),
    }
    if K == 0:
        return out

    if confidences is None:
        confidences = np.ones(K, dtype=float)

    use_fallback = (
        fallback_left is not None
        and fallback_right is not None
        and fallback_front is not None
        and fallback_behind is not None
    )

    for j in range(K):
        cj = float(confidences[j]) if confidences is not None else 1.0
        if cj < confidence_threshold:
            if use_fallback:
                out["obj_left"][:, j] = fallback_left[:, j]
                out["obj_right"][:, j] = fallback_right[:, j]
                out["obj_front"][:, j] = fallback_front[:, j]
                out["obj_behind"][:, j] = fallback_behind[:, j]
            continue
        weight = cj if weight_by_confidence else 1.0
        for i in range(K):
            if i == j:
                continue
            # ``right_comp[j, i]`` = where i lies along j's right axis.
            rc = right_comp[j, i]
            fc = front_comp[j, i]
            out["obj_left"][i, j] = (1.0 - rc) / 2.0 * weight
            out["obj_right"][i, j] = (1.0 + rc) / 2.0 * weight
            out["obj_front"][i, j] = (1.0 + fc) / 2.0 * weight
            out["obj_behind"][i, j] = (1.0 - fc) / 2.0 * weight
    return out


def compute_obj_relative_from_axes(
    positions: np.ndarray,
    front_directions: np.ndarray,
    right_directions: np.ndarray,
    confidences: np.ndarray = None,
    *,
    confidence_threshold: float = 0.0,
    weight_by_confidence: bool = False,
) -> dict:
    """Pure-array entry point for ``compute_obj_relative_matrices``.

    Useful for callers that already have ``(K, 3)`` arrays of world-space
    positions plus per-entity ``front_world`` / ``right_world`` axes — for
    example the multi-view ``Scene`` which exposes ``obj_*`` over a mixed
    object+camera entity space.

    The function projects horizontal-plane separations onto each anchor's
    front/right axes and then defers to
    :func:`compute_obj_relative_matrices` for the score assembly.

    Parameters
    ----------
    positions : (K, 3) world positions.
    front_directions : (K, 3) per-entity world-space forward axis.
    right_directions : (K, 3) per-entity world-space right axis.
    confidences : optional (K,) confidences in [0, 1].
    """
    K = len(positions)
    if K == 0:
        return {
            "obj_left": np.zeros((0, 0), dtype=float),
            "obj_right": np.zeros((0, 0), dtype=float),
            "obj_front": np.zeros((0, 0), dtype=float),
            "obj_behind": np.zeros((0, 0), dtype=float),
        }

    positions = np.asarray(positions, dtype=float)
    fronts = np.asarray(front_directions, dtype=float)
    rights = np.asarray(right_directions, dtype=float)

    front_comp = np.zeros((K, K), dtype=float)
    right_comp = np.zeros((K, K), dtype=float)
    for i in range(K):
        fi = fronts[i]
        ri = rights[i]
        nfi = float(np.linalg.norm(fi))
        nri = float(np.linalg.norm(ri))
        if nfi < 1e-8 and nri < 1e-8:
            continue
        if nfi >= 1e-8:
            fi_u = fi / nfi
        else:
            fi_u = None
        if nri >= 1e-8:
            ri_u = ri / nri
        else:
            ri_u = None
        for j in range(K):
            if i == j:
                continue
            vec = positions[j] - positions[i]
            vec_h = np.array([vec[0], 0.0, vec[2]], dtype=float)
            dist_h = float(np.linalg.norm(vec_h))
            if dist_h < 1e-8:
                continue
            if ri_u is not None:
                right_comp[i, j] = float(np.dot(vec_h, ri_u)) / dist_h
            if fi_u is not None:
                front_comp[i, j] = float(np.dot(vec_h, fi_u)) / dist_h

    return compute_obj_relative_matrices(
        front_comp,
        right_comp,
        confidences=confidences,
        confidence_threshold=confidence_threshold,
        weight_by_confidence=weight_by_confidence,
    )
