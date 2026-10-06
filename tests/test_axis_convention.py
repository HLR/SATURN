"""Exhaustive tests for scene.set_axis_convention + anchor.project.

Coverage strategy:
- All 48 signed permutations of (X,Y,Z) → math correctness parametrized
- Parsing rejection grid: ~30 malformed inputs
- Numerical edge cases: NaN, inf, zero, tiny, huge, int-mixed
- Round-trip property: M @ M^T = I over all 48 conventions
- Predicate-invariance: displacement / frame axes / origin unchanged across grid
- State invariants: mid-construction failure, copy/deepcopy
- MMSI scoring scenarios: cam-motion × option-pattern grid
- Named real-world conventions enumerated
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pytest

from saturn.scene.scene import Scene
from saturn.predicates.frame import FrameNamespace


# ==============================================================
# Helpers
# ==============================================================

def make_scene_shell() -> Scene:
    """Build a Scene with just enough state to exercise the convention API."""
    s = Scene.__new__(Scene)
    s._axis_convention_M = None
    return s


def make_frame(scene: Scene, right=(1, 0, 0), up=(0, 1, 0), front=(0, 0, 1),
               origin=(0, 0, 0)) -> FrameNamespace:
    f = FrameNamespace.__new__(FrameNamespace)
    f._scene = scene
    f._frame_right = np.asarray(right, dtype=float)
    f._frame_up = np.asarray(up, dtype=float)
    f._frame_front = np.asarray(front, dtype=float)
    f._frame_origin = np.asarray(origin, dtype=float)
    return f


# All 6 permutations of (X, Y, Z) axis assignments.
# Each entry is (axis_for_right, axis_for_up, axis_for_forward) — a permutation of (X, Y, Z).
_PERMS = list(itertools.permutations(["X", "Y", "Z"]))     # 6
_SIGN_TRIPLES = list(itertools.product([+1, -1], repeat=3))  # 8

# All 48 valid (distinct-axis) conventions: 6 permutations × 8 sign combos.
ALL_48_CONVENTIONS = []
for (axis_r, axis_u, axis_f), (s_r, s_u, s_f) in itertools.product(_PERMS, _SIGN_TRIPLES):
    ALL_48_CONVENTIONS.append({
        "right": ("+" if s_r > 0 else "-") + axis_r,
        "up":    ("+" if s_u > 0 else "-") + axis_u,
        "forward": ("+" if s_f > 0 else "-") + axis_f,
    })

assert len(ALL_48_CONVENTIONS) == 48


def expected_M(conv: dict) -> np.ndarray:
    """Build M[q, role] = sign matching the implementation's formula."""
    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    M = np.zeros((3, 3), dtype=float)
    for role_idx, role in enumerate(["right", "up", "forward"]):
        tok = conv[role]
        sign = +1 if tok[0] == "+" else -1
        q = axis_idx[tok[1].upper()]
        M[q, role_idx] = sign
    return M


# ==============================================================
# Section 1: Parsing & default behavior
# ==============================================================

class TestParsing:
    def test_default_no_args_is_identity(self):
        s = make_scene_shell()
        s.set_axis_convention()
        assert s._axis_convention_M is None

    def test_all_explicit_defaults_is_identity(self):
        s = make_scene_shell()
        s.set_axis_convention(right="+X", up="+Y", forward="+Z")
        assert s._axis_convention_M is None

    @pytest.mark.parametrize("kw,key,expected", [
        ({"right": "-X"}, (3, 4, 5), (-3, 4, 5)),
        ({"right": "+X"}, (3, 4, 5), (3, 4, 5)),
        ({"up":    "-Y"}, (3, 4, 5), (3, -4, 5)),
        ({"up":    "+Y"}, (3, 4, 5), (3, 4, 5)),
        ({"forward": "-Z"}, (3, 4, 5), (3, 4, -5)),
        ({"forward": "+Z"}, (3, 4, 5), (3, 4, 5)),
    ])
    def test_single_kwarg_sign(self, kw, key, expected):
        s = make_scene_shell()
        s.set_axis_convention(**kw)
        out = s._axis_project(tuple(float(x) for x in key))
        assert out == tuple(float(x) for x in expected)

    @pytest.mark.parametrize("token,kwarg", [
        # Use kwargs that don't conflict with the other defaults.
        # right defaults to +X, up to +Y, forward to +Z — choose non-default
        # tokens for the test inputs.
        ("-x", "right"), ("-y", "up"), ("-z", "forward"),     # lowercase + sign flip
        ("-X", "right"), ("-Y", "up"), ("-Z", "forward"),     # uppercase + sign flip
        ("  -X  ", "right"), ("\t-Y", "up"),                   # whitespace
        ("-X\n", "right"), ("\n-Z", "forward"),                # newline padding
    ])
    def test_whitespace_and_case_accepted(self, token, kwarg):
        s = make_scene_shell()
        # Should not raise
        s.set_axis_convention(**{kwarg: token})


# ==============================================================
# Section 2: Parsing rejections (the malformed-input grid)
# ==============================================================

class TestRejections:
    @pytest.mark.parametrize("bad_token", [
        "",            # empty
        "X",           # missing sign
        "+",           # only sign
        "-",
        "++X",         # double sign
        "+-X",
        "-+X",
        "+X+",         # trailing junk
        "X+",          # axis-first
        "+W",          # bad letter
        "-A",
        "+0",          # digit
        "+1",
        "+ X",         # internal space
        "+\tX",        # internal tab
        "+XY",         # extra letter
        "+XX",
        "+X1",
        "1X",
        ".X",
        "*X",
    ])
    def test_malformed_token_rejected(self, bad_token):
        s = make_scene_shell()
        with pytest.raises(ValueError, match="sign followed by"):
            s.set_axis_convention(forward=bad_token)

    @pytest.mark.parametrize("bad_token", [
        "−X",   # Unicode minus (U+2212), not ASCII '-'
        "＋X",  # Full-width plus (U+FF0B)
        "+Ｘ",  # Full-width X
    ])
    def test_unicode_lookalike_rejected(self, bad_token):
        s = make_scene_shell()
        with pytest.raises(ValueError, match="sign followed by"):
            s.set_axis_convention(forward=bad_token)

    @pytest.mark.parametrize("duplicate_kwargs", [
        {"right": "+X", "up": "+X", "forward": "+Y"},      # X twice
        {"right": "+X", "up": "+Y", "forward": "+Y"},      # Y twice
        {"right": "+Z", "up": "+Z", "forward": "+X"},      # Z twice
        {"right": "+X", "up": "+X", "forward": "+X"},      # all same
        {"right": "+X", "up": "-X", "forward": "+Y"},      # X with opposite signs
        {"up": "+Y", "forward": "-Y"},                      # default right conflicts? no, defaults to +X
    ])
    def test_duplicate_axes_rejected(self, duplicate_kwargs):
        s = make_scene_shell()
        with pytest.raises(ValueError, match="distinct"):
            s.set_axis_convention(**duplicate_kwargs)

    def test_default_right_conflicts_with_explicit_x(self):
        # right defaults to +X — explicit up=+X collides
        s = make_scene_shell()
        with pytest.raises(ValueError, match="distinct"):
            s.set_axis_convention(up="+X")

    def test_default_up_conflicts_with_explicit_y(self):
        s = make_scene_shell()
        with pytest.raises(ValueError, match="distinct"):
            s.set_axis_convention(forward="+Y")  # up defaults to +Y

    def test_default_forward_conflicts_with_explicit_z(self):
        s = make_scene_shell()
        with pytest.raises(ValueError, match="distinct"):
            s.set_axis_convention(right="+Z")  # forward defaults to +Z


# ==============================================================
# Section 3: Exhaustive math correctness — all 48 conventions
# ==============================================================

class TestExhaustiveMath:
    """For each of the 48 valid (distinct-axis) conventions, verify
    that set_axis_convention builds the expected M matrix and that
    _axis_project produces M @ vec for several test vectors."""

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_matrix_matches_formula(self, conv):
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        expected = expected_M(conv)
        if s._axis_convention_M is None:
            # Identity case
            np.testing.assert_array_equal(expected, np.eye(3))
        else:
            np.testing.assert_array_equal(s._axis_convention_M, expected)

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    @pytest.mark.parametrize("vec", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0),
        (-1.0, -2.0, -3.0),
        (7.5, -3.2, 11.1),
    ])
    def test_project_matches_matmul(self, conv, vec):
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        out = s._axis_project(vec)
        expected = tuple(float(x) for x in (expected_M(conv) @ np.array(vec, dtype=float)))
        assert out == expected

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_determinant_is_pm_one(self, conv):
        """All 48 signed permutations have determinant +1 (right-handed
        ordering) or -1 (left-handed ordering). OpenGL is left-handed in
        (right, up, forward) ordering, so both signs are valid conventions."""
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        if s._axis_convention_M is None:
            det = +1.0
        else:
            det = float(np.linalg.det(s._axis_convention_M))
        assert abs(abs(det) - 1.0) < 1e-12, f"det(M) = {det} for {conv}"

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_M_is_signed_permutation(self, conv):
        """Each M is a signed permutation matrix: exactly one ±1 per row
        and per column, all others zero."""
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        M = s._axis_convention_M if s._axis_convention_M is not None else np.eye(3)
        # Each row has exactly one non-zero entry of magnitude 1
        for row in range(3):
            nonzero = [abs(M[row, col]) for col in range(3) if M[row, col] != 0]
            assert len(nonzero) == 1, f"row {row}: {M[row]}"
            assert nonzero[0] == 1.0
        # Each column has exactly one non-zero entry of magnitude 1
        for col in range(3):
            nonzero = [abs(M[row, col]) for row in range(3) if M[row, col] != 0]
            assert len(nonzero) == 1, f"col {col}: {M[:, col]}"
            assert nonzero[0] == 1.0


# ==============================================================
# Section 4: Round-trip property — M @ M^T = I for signed permutations
# ==============================================================

class TestRoundTripProperty:
    """A signed permutation matrix is orthogonal: M @ M^T = I.
    So projecting with M then un-projecting with M^T recovers the input."""

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_M_orthogonal(self, conv):
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        M = s._axis_convention_M if s._axis_convention_M is not None else np.eye(3)
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(M.T @ M, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_round_trip_random_vector(self, conv, seed):
        """project(vec) followed by inverse-project should recover vec."""
        rng = np.random.default_rng(seed)
        vec = tuple(rng.normal(size=3))
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        forward = s._axis_project(vec)
        M = s._axis_convention_M if s._axis_convention_M is not None else np.eye(3)
        back = M.T @ np.array(forward, dtype=float)
        np.testing.assert_allclose(back, vec, atol=1e-12)


# ==============================================================
# Section 5: Numerical edge cases
# ==============================================================

class TestNumericalEdgeCases:
    def test_zero_vector(self):
        s = make_scene_shell()
        for conv in ALL_48_CONVENTIONS[::8]:  # spot-check 6 conventions
            s.set_axis_convention(**conv)
            assert s._axis_project((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)

    def test_nan_propagates(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        out = s._axis_project((float("nan"), 1.0, 2.0))
        assert math.isnan(out[0])
        assert out[1] == 1.0
        assert out[2] == -2.0

    def test_inf_propagates(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        out = s._axis_project((1.0, float("inf"), -5.0))
        assert out[0] == 1.0
        assert math.isinf(out[1])
        assert out[2] == 5.0

    def test_neg_inf_propagates(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        out = s._axis_project((1.0, 2.0, float("-inf")))
        assert math.isinf(out[2]) and out[2] > 0   # neg-inf × -1 = +inf

    def test_very_large_floats(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        big = 1e300
        out = s._axis_project((big, big, big))
        assert out[0] == big
        assert out[1] == big
        assert out[2] == -big

    def test_subnormal_floats(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        tiny = 5e-324  # smallest positive subnormal
        out = s._axis_project((tiny, tiny, tiny))
        assert out[0] == tiny
        assert out[1] == tiny
        assert out[2] == -tiny

    def test_int_input_returns_floats(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        out = s._axis_project((1, 2, 3))
        assert all(isinstance(x, float) for x in out)
        assert out == (1.0, 2.0, -3.0)

    def test_mixed_int_float_input(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        out = s._axis_project((1, 2.5, 3))
        assert out == (1.0, 2.5, -3.0)

    def test_numpy_array_input_via_project(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        f = make_frame(s)
        out = f.project(np.array([1.0, 2.0, 3.0]))
        assert out == (1.0, 2.0, -3.0)

    def test_list_input_via_project(self):
        s = make_scene_shell()
        f = make_frame(s)
        out = f.project([1.0, 2.0, 3.0])
        assert out == (1.0, 2.0, 3.0)

    def test_tuple_input_via_project(self):
        s = make_scene_shell()
        f = make_frame(s)
        out = f.project((1.0, 2.0, 3.0))
        assert out == (1.0, 2.0, 3.0)

    def test_negative_zero(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        out = s._axis_project((-0.0, 0.0, -0.0))
        # -1 * -0.0 = +0.0 in IEEE 754
        assert out[2] == 0.0


# ==============================================================
# Section 6: Reset and overwrite behavior
# ==============================================================

class TestStateTransitions:
    def test_reset_after_set(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, 2.0, -3.0)
        s.set_axis_convention()
        assert s._axis_convention_M is None
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)

    def test_overwrite_convention(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        s.set_axis_convention(up="-Y")
        # Last convention wins: right=+X (default), up=-Y, forward=+Z (default)
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, -2.0, 3.0)

    def test_repeated_reset_idempotent(self):
        s = make_scene_shell()
        for _ in range(10):
            s.set_axis_convention()
            assert s._axis_convention_M is None

    def test_set_then_set_same_idempotent(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        first = s._axis_convention_M.copy()
        s.set_axis_convention(forward="-Z")
        np.testing.assert_array_equal(s._axis_convention_M, first)

    def test_mid_construction_failure_preserves_prior_state(self):
        """If set_axis_convention raises mid-construction, the prior
        convention should be preserved (not partially overwritten)."""
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        prior = s._axis_convention_M.copy()
        # Second call has a bad token — should raise without altering state.
        with pytest.raises(ValueError):
            s.set_axis_convention(forward="-Z", up="BAD")
        # Prior convention intact?
        assert s._axis_convention_M is not None
        np.testing.assert_array_equal(s._axis_convention_M, prior)

    def test_mid_construction_validation_failure_preserves_state(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        prior = s._axis_convention_M.copy()
        # Duplicate axes — should raise after parsing
        with pytest.raises(ValueError):
            s.set_axis_convention(up="+Y", forward="+Y")
        np.testing.assert_array_equal(s._axis_convention_M, prior)


# ==============================================================
# Section 7: anchor.project end-to-end on real FrameNamespace
# ==============================================================

class TestAnchorProject:
    """Verify project() with various anchor orientations and conventions."""

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    @pytest.mark.parametrize("vec", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (3.0, -2.0, 4.0),
    ])
    def test_canonical_anchor_all_conventions(self, conv, vec):
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        f = make_frame(s)  # canonical axes
        out = f.project(vec)
        expected = tuple(float(x) for x in (expected_M(conv) @ np.array(vec, dtype=float)))
        assert out == expected

    def test_rotated_anchor_canonical_basis_check(self):
        """Anchor with non-canonical orientation: front pointing along world +X,
        up = +Y, right = -Z. Verify local-axis projection is correct under
        default convention."""
        s = make_scene_shell()
        f = make_frame(s, right=(0, 0, -1), up=(0, 1, 0), front=(1, 0, 0))
        # World vec [5, 0, 0] is along anchor's +front → sz=5
        assert f.project([5.0, 0.0, 0.0]) == (0.0, 0.0, 5.0)
        # World vec [0, 3, 0] is along anchor's +up → sy=3
        assert f.project([0.0, 3.0, 0.0]) == (0.0, 3.0, 0.0)
        # World vec [0, 0, -2] is along anchor's +right → sx=2
        assert f.project([0.0, 0.0, -2.0]) == (2.0, 0.0, 0.0)

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_rotated_anchor_all_conventions(self, conv):
        """A rotated anchor combined with any convention should give the
        product of (anchor rotation) and (M)."""
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        f = make_frame(s, right=(0, 0, -1), up=(0, 1, 0), front=(1, 0, 0))
        # World vec [5, 0, 0] gives anchor-local (sx=0, sy=0, sz=5)
        local = (0.0, 0.0, 5.0)
        expected = tuple(float(x) for x in (expected_M(conv) @ np.array(local, dtype=float)))
        assert f.project([5.0, 0.0, 0.0]) == expected


# ==============================================================
# Section 8: Predicate / world-geometry invariance
# ==============================================================

class TestInvariance:
    """The convention flag must not affect non-project primitives.

    We test the ones we can construct without a full Scene: displacement,
    frame axes, frame origin. The other primitives (first_person, etc.)
    require a full scene and are integration-level."""

    def setup_method(self):
        self.s = make_scene_shell()
        # Stub _resolve_entity so displacement works without a full scene
        self.s._resolve_entity = lambda to, label: (np.asarray(to, dtype=float), None, None)

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_displacement_unchanged(self, conv):
        f = make_frame(self.s)
        baseline = f.displacement([1.0, 2.0, 3.0]).copy()
        self.s.set_axis_convention(**conv)
        after = f.displacement([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(baseline, after)

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_frame_axes_unchanged(self, conv):
        f = make_frame(self.s, right=(0, 0, -1), up=(0, 1, 0), front=(1, 0, 0))
        baseline = (f._frame_right.copy(), f._frame_up.copy(), f._frame_front.copy())
        self.s.set_axis_convention(**conv)
        np.testing.assert_array_equal(f._frame_right, baseline[0])
        np.testing.assert_array_equal(f._frame_up,    baseline[1])
        np.testing.assert_array_equal(f._frame_front, baseline[2])

    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS)
    def test_frame_origin_unchanged(self, conv):
        f = make_frame(self.s, origin=(7.5, -3.2, 11.1))
        baseline = f._frame_origin.copy()
        self.s.set_axis_convention(**conv)
        np.testing.assert_array_equal(f._frame_origin, baseline)


# ==============================================================
# Section 9: Scene copy/deepcopy semantics
# ==============================================================

class TestCopySemantics:
    def test_independent_scenes_independent_conventions(self):
        s1 = make_scene_shell()
        s2 = make_scene_shell()
        s1.set_axis_convention(forward="-Z")
        assert s1._axis_convention_M is not None
        assert s2._axis_convention_M is None

    def test_copy_preserves_convention(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        s2 = copy.copy(s)
        # copy.copy is shallow — they share the matrix reference
        assert s2._axis_convention_M is s._axis_convention_M
        np.testing.assert_array_equal(s2._axis_project((1.0, 2.0, 3.0)),
                                      s._axis_project((1.0, 2.0, 3.0)))

    def test_deepcopy_preserves_convention(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        s2 = copy.deepcopy(s)
        # Deep copy — separate matrix, same values
        assert s2._axis_convention_M is not s._axis_convention_M
        np.testing.assert_array_equal(s2._axis_convention_M, s._axis_convention_M)

    def test_deepcopy_independent_mutation(self):
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        s2 = copy.deepcopy(s)
        s.set_axis_convention(up="-Y")  # mutate original
        # s2 should still have the OpenGL convention
        assert s2._axis_project((1.0, 2.0, 3.0)) == (1.0, 2.0, -3.0)
        # s now has Y-down convention
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, -2.0, 3.0)


# ==============================================================
# Section 10: Real-world named conventions (cross-check table)
# ==============================================================

class TestNamedConventions:
    @pytest.mark.parametrize("name,kwargs,vec,expected", [
        # (name, kwargs, (sx,sy,sz), (xq,yq,zq))
        ("OpenCV (default)",      {},                              (1, 2, 3), (1.0,  2.0,  3.0)),
        ("OpenGL camera",         {"up": "+Y", "forward": "-Z"},   (1, 2, 3), (1.0,  2.0, -3.0)),
        ("Y-down image",          {"up": "-Y"},                    (1, 2, 3), (1.0, -2.0,  3.0)),
        ("Z-up + Y-forward",      {"up": "+Z", "forward": "+Y"},   (1, 2, 3), (1.0,  3.0,  2.0)),
        ("Z-up + neg-Y-forward",  {"up": "+Z", "forward": "-Y"},   (1, 2, 3), (1.0, -3.0,  2.0)),
        ("Right-handed XZY swap", {"up": "+Z", "forward": "+Y", "right": "+X"}, (1, 2, 3), (1.0, 3.0, 2.0)),
        ("Cyclic XYZ→ZXY",        {"right": "+Z", "up": "+X", "forward": "+Y"}, (1, 2, 3), (2.0, 3.0, 1.0)),
        ("Cyclic XYZ→YZX",        {"right": "+Y", "up": "+Z", "forward": "+X"}, (1, 2, 3), (3.0, 1.0, 2.0)),
        ("All-negative",          {"right": "-X", "up": "-Y", "forward": "-Z"}, (1, 2, 3), (-1.0, -2.0, -3.0)),
        ("Mixed sign + permute",  {"right": "+Z", "up": "-X", "forward": "-Y"}, (1, 2, 3), (-2.0, -3.0, 1.0)),
    ])
    def test_named_conventions(self, name, kwargs, vec, expected):
        s = make_scene_shell()
        s.set_axis_convention(**kwargs)
        out = s._axis_project(tuple(float(x) for x in vec))
        assert out == expected, f"{name}: got {out}, expect {expected}"


# ==============================================================
# Section 11: MMSI-pattern scoring scenarios
# ==============================================================

class TestMMSIScoring:
    """End-to-end realistic scenarios: cam-cam axis-coded motion questions."""

    OPENGL_OPTIONS = {"A": (+1, -1), "B": (+1, +1), "C": (-1, +1), "D": (-1, -1)}

    @pytest.mark.parametrize("disp_world,expected_letter", [
        ((+1.0, 0.0, +5.0), "A"),  # +X +Z internal = +X -Z question → A
        ((+1.0, 0.0, -5.0), "B"),  # +X -Z internal = +X +Z question → B
        ((-1.0, 0.0, -5.0), "C"),  # -X -Z internal = -X +Z question → C
        ((-1.0, 0.0, +5.0), "D"),  # -X +Z internal = -X -Z question → D
        ((+0.1, 0.0, +5.0), "A"),  # small +X but still dominates over -1
        ((+5.0, 0.0, +0.1), "A"),  # dominant +X, small +Z (still on +/- side)
        ((-5.0, 0.0, -5.0), "C"),
        ((+5.0, 0.0, -5.0), "B"),
    ])
    def test_opengl_motion_picks_correct_option(self, disp_world, expected_letter):
        s = make_scene_shell()
        s.set_axis_convention(up="+Y", forward="-Z")
        anchor = make_frame(s)
        sx, sy, sz = anchor.project(disp_world)
        winner = max(self.OPENGL_OPTIONS,
                     key=lambda k: self.OPENGL_OPTIONS[k][0] * sx
                                  + self.OPENGL_OPTIONS[k][1] * sz)
        assert winner == expected_letter, f"disp={disp_world}: got {winner}, expect {expected_letter}"

    DEFAULT_OPTIONS = {"A": (+1, -1), "B": (+1, +1), "C": (-1, +1), "D": (-1, -1)}

    @pytest.mark.parametrize("disp_world,expected_letter", [
        ((+1.0, 0.0, +5.0), "B"),  # No convention: internal == question → +X +Z → B
        ((+1.0, 0.0, -5.0), "A"),  # +X -Z → A
        ((-1.0, 0.0, +5.0), "C"),  # -X +Z → C
        ((-1.0, 0.0, -5.0), "D"),  # -X -Z → D
    ])
    def test_default_convention_picks_correct_option(self, disp_world, expected_letter):
        s = make_scene_shell()
        # No set_axis_convention call → default (identity)
        anchor = make_frame(s)
        sx, sy, sz = anchor.project(disp_world)
        winner = max(self.DEFAULT_OPTIONS,
                     key=lambda k: self.DEFAULT_OPTIONS[k][0] * sx
                                  + self.DEFAULT_OPTIONS[k][1] * sz)
        assert winner == expected_letter


# ==============================================================
# Section 12: Property tests via random sampling
# ==============================================================

class TestRandomProperties:
    """Property-based tests using fixed seeds for reproducibility."""

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_random_vec_under_random_convention_round_trips(self, seed):
        rng = np.random.default_rng(seed)
        # Pick a random convention from the 48
        conv = ALL_48_CONVENTIONS[rng.integers(0, 48)]
        vec = tuple(rng.normal(size=3))
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        forward = s._axis_project(vec)
        M = s._axis_convention_M if s._axis_convention_M is not None else np.eye(3)
        back = M.T @ np.array(forward, dtype=float)
        np.testing.assert_allclose(back, vec, atol=1e-12)

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_random_vec_preserves_norm(self, seed):
        """Signed-permutation matrices preserve L2 norm."""
        rng = np.random.default_rng(seed)
        conv = ALL_48_CONVENTIONS[rng.integers(0, 48)]
        vec = tuple(rng.normal(size=3))
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        out = s._axis_project(vec)
        assert abs(np.linalg.norm(out) - np.linalg.norm(vec)) < 1e-12

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_random_vec_unchanged_under_default_convention(self, seed):
        rng = np.random.default_rng(seed)
        vec = tuple(rng.normal(size=3))
        s = make_scene_shell()
        s.set_axis_convention()  # default
        out = s._axis_project(vec)
        assert out == tuple(float(x) for x in vec)


# ==============================================================
# Section 13: Composition / chained operations
# ==============================================================

class TestComposition:
    @pytest.mark.parametrize("conv", ALL_48_CONVENTIONS[::4])  # spot-check 12
    def test_multiple_vectors_consistent(self, conv):
        """Projecting many vectors gives the same M for each."""
        s = make_scene_shell()
        s.set_axis_convention(**conv)
        M = s._axis_convention_M if s._axis_convention_M is not None else np.eye(3)
        for vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (-1, -1, -1)]:
            out = s._axis_project(tuple(float(x) for x in vec))
            expected = tuple(float(x) for x in (M @ np.array(vec, dtype=float)))
            assert out == expected

    def test_set_project_set_project_independent(self):
        """Switching conventions mid-session works without bleed-through."""
        s = make_scene_shell()
        s.set_axis_convention(forward="-Z")
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, 2.0, -3.0)
        s.set_axis_convention(up="-Y")
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, -2.0, 3.0)
        s.set_axis_convention()  # reset
        assert s._axis_project((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
        s.set_axis_convention(right="-X", up="-Y", forward="-Z")
        assert s._axis_project((1.0, 2.0, 3.0)) == (-1.0, -2.0, -3.0)


# ==============================================================
# Section 14: Permutation correctness — cyclic and reversed orderings
# ==============================================================

class TestPermutationCorrectness:
    """Verify the M-matrix indexing for non-trivial permutations.
    A single cyclic rotation could pass by coincidence, so each of the
    6 axis assignments is checked explicitly."""

    @pytest.mark.parametrize("axis_r,axis_u,axis_f,expected", [
        # Anchor scalars are (sx, sy, sz) = (1, 2, 3).
        # The result (x_q, y_q, z_q) places sx into the X-slot the right role
        # maps to, sy into the Y-slot, sz into the Z-slot.
        ("X", "Y", "Z", (1.0, 2.0, 3.0)),
        ("X", "Z", "Y", (1.0, 3.0, 2.0)),  # up=Z (sy→Z), forward=Y (sz→Y)
        ("Y", "X", "Z", (2.0, 1.0, 3.0)),  # right=Y (sx→Y), up=X (sy→X)
        ("Y", "Z", "X", (3.0, 1.0, 2.0)),
        ("Z", "X", "Y", (2.0, 3.0, 1.0)),
        ("Z", "Y", "X", (3.0, 2.0, 1.0)),
    ])
    def test_permutation_matrix_indexing(self, axis_r, axis_u, axis_f, expected):
        s = make_scene_shell()
        s.set_axis_convention(
            right="+" + axis_r, up="+" + axis_u, forward="+" + axis_f
        )
        out = s._axis_project((1.0, 2.0, 3.0))
        assert out == expected, f"({axis_r},{axis_u},{axis_f}): got {out}, expect {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
