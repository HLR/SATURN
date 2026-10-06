"""Tests for the cv2 compatibility shim (``saturn.perception.cv_compat``).

Two regimes are covered:

* with real OpenCV importable, every symbol must be the genuine cv2 callable
  (this is what keeps the Python 3.10 paper path byte-identical);
* the numpy fallback must reproduce cv2 to the tolerances measured by
  ``scripts/validate_cv_compat.py``.

Tolerances below are the *measured* worst cases, not aspirations:
  resize INTER_NEAREST  : bit-exact
  resize INTER_LINEAR   : float32 matches to ~2e-7 relative (float32 epsilon);
                          uint8 two-axis is ~99.5% bit-exact, remainder +-1 LSB
  cvtColor RGB<->BGR    : bit-exact
  erode (solid kernel)  : bit-exact
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SHIM_PATH = pathlib.Path(__file__).resolve().parents[1] / "saturn" / "perception" / "cv_compat.py"

cv2 = pytest.importorskip("cv2")

from saturn.perception import cv_compat  # noqa: E402


@pytest.fixture(scope="module")
def fb():
    """A second copy of the shim loaded with ``import cv2`` forced to fail."""

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "cv2" or name.startswith("cv2."):
                raise ImportError("cv2 blocked for fallback testing")
            return None

    saved = {k: v for k, v in sys.modules.items() if k == "cv2" or k.startswith("cv2.")}
    for k in saved:
        del sys.modules[k]
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        spec = importlib.util.spec_from_file_location("_cv_compat_fb", SHIM_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
    assert mod.USING_REAL_CV2 is False
    return mod


# --------------------------------------------------------------------------
# delegation
# --------------------------------------------------------------------------

def test_delegates_to_real_cv2_when_present():
    assert cv_compat.USING_REAL_CV2 is True
    assert cv_compat.resize is cv2.resize
    assert cv_compat.cvtColor is cv2.cvtColor
    assert cv_compat.erode is cv2.erode


def test_flags_match_opencv_values():
    for mod in (cv_compat,):
        assert mod.INTER_NEAREST == cv2.INTER_NEAREST
        assert mod.INTER_LINEAR == cv2.INTER_LINEAR
        assert mod.INTER_CUBIC == cv2.INTER_CUBIC
        assert mod.INTER_AREA == cv2.INTER_AREA
        assert mod.COLOR_RGB2BGR == cv2.COLOR_RGB2BGR
        assert mod.COLOR_BGR2RGB == cv2.COLOR_BGR2RGB


def test_fallback_flag_values_also_match(fb):
    """The fallback hardcodes the enum values; they must not drift from cv2."""
    assert (fb.INTER_NEAREST, fb.INTER_LINEAR, fb.INTER_CUBIC, fb.INTER_AREA) == (
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
    )
    assert fb.COLOR_RGB2BGR == cv2.COLOR_RGB2BGR
    assert fb.COLOR_BGR2RGB == cv2.COLOR_BGR2RGB


# --------------------------------------------------------------------------
# cvtColor
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["COLOR_RGB2BGR", "COLOR_BGR2RGB"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_cvtcolor_is_bit_exact(fb, code, dtype):
    rng = np.random.default_rng(0)
    arr = (rng.random((17, 23, 3)) * 255).astype(dtype)
    flag = getattr(cv2, code)
    np.testing.assert_array_equal(fb.cvtColor(arr, flag), cv2.cvtColor(arr, flag))


def test_cvtcolor_is_a_pure_channel_reversal(fb):
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    np.testing.assert_array_equal(
        fb.cvtColor(arr, cv2.COLOR_RGB2BGR), arr[..., ::-1]
    )


def test_cvtcolor_rejects_unsupported_codes(fb):
    arr = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(NotImplementedError):
        fb.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def test_cvtcolor_rejects_2d(fb):
    with pytest.raises(ValueError):
        fb.cvtColor(np.zeros((4, 4), np.uint8), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# erode -- the kernel actually used by erode_mask()
# --------------------------------------------------------------------------

KERNEL = np.ones((3, 3), np.uint8)


@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize(
    "name",
    ["random", "blob", "all_ones", "all_zero", "greyscale", "float32", "single_pixel"],
)
def test_erode_is_bit_exact(fb, name, iterations):
    rng = np.random.default_rng(3)
    arrays = {
        "random": (rng.random((41, 53)) > 0.4).astype(np.uint8),
        "blob": np.pad(np.ones((17, 29), np.uint8), 5),
        "all_ones": np.ones((12, 12), np.uint8),
        "all_zero": np.zeros((12, 12), np.uint8),
        "greyscale": rng.integers(0, 256, (30, 30), dtype=np.uint8),
        "float32": rng.random((30, 30)).astype(np.float32),
        "single_pixel": np.pad(np.ones((1, 1), np.uint8), 4),
    }
    arr = arrays[name]
    np.testing.assert_array_equal(
        fb.erode(arr, KERNEL, iterations=iterations),
        cv2.erode(arr, KERNEL, iterations=iterations),
    )


def test_erode_zero_iterations_is_identity(fb):
    arr = np.ones((5, 5), np.uint8)
    np.testing.assert_array_equal(fb.erode(arr, KERNEL, iterations=0), arr)


def test_erode_rejects_non_solid_kernel(fb):
    k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    with pytest.raises(NotImplementedError):
        fb.erode(np.ones((5, 5), np.uint8), k)


def test_erode_preserves_dtype(fb):
    for dt in (np.uint8, np.float32):
        assert fb.erode(np.ones((6, 6), dt), KERNEL).dtype == dt


# --------------------------------------------------------------------------
# resize
# --------------------------------------------------------------------------

SHAPES = [(9, 7), (23, 29), (48, 64)]
TARGETS = ["down", "down_odd", "up", "up_odd"]


def _target(shape, kind):
    h, w = shape
    return {
        "down": (max(1, w // 3), max(1, h // 3)),
        "down_odd": (max(1, int(w * 0.7)), max(1, int(h * 0.44))),
        "up": (w * 2, h * 2),
        "up_odd": (w * 2 + 1, int(h * 1.7)),
    }[kind]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", TARGETS)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [None, 3])
def test_resize_nearest_is_bit_exact(fb, shape, kind, dtype, channels):
    """INTER_NEAREST is the flag used for every mask resize in the orchestrator."""
    rng = np.random.default_rng(7)
    full = shape if channels is None else (*shape, channels)
    arr = (rng.random(full) * 255).astype(dtype)
    dsize = _target(shape, kind)
    got = fb.resize(arr, dsize, interpolation=cv2.INTER_NEAREST)
    ref = cv2.resize(arr, dsize, interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(got, ref)
    assert got.dtype == dtype
    assert got.shape[:2] == (dsize[1], dsize[0])


def test_resize_nearest_bit_exact_over_many_scale_pairs(fb):
    """The floor(dx * 1/(dst/src)) index map is fragile; sweep it broadly."""
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, (1, 61), dtype=np.uint8)
    for dst in range(2, 130):
        np.testing.assert_array_equal(
            fb.resize(arr, (dst, 1), interpolation=cv2.INTER_NEAREST),
            cv2.resize(arr, (dst, 1), interpolation=cv2.INTER_NEAREST),
            err_msg=f"nearest index map differs for 61 -> {dst}",
        )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", TARGETS)
@pytest.mark.parametrize("channels", [None, 3])
def test_resize_linear_float32_matches_to_float_epsilon(fb, shape, kind, channels):
    """INTER_LINEAR on float32 (depth-map-like arrays) matches cv2."""
    rng = np.random.default_rng(5)
    full = shape if channels is None else (*shape, channels)
    arr = rng.random(full).astype(np.float32) * 10.0  # metres-scale, like depth
    dsize = _target(shape, kind)
    got = fb.resize(arr, dsize, interpolation=cv2.INTER_LINEAR)
    ref = cv2.resize(arr, dsize, interpolation=cv2.INTER_LINEAR)
    assert got.dtype == np.float32
    # measured worst case is ~1.9e-7 relative, i.e. float32 machine epsilon
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)
    rel = np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-12)
    assert rel < 1e-6, f"relative error {rel:g} exceeds float32 epsilon budget"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", TARGETS)
def test_resize_linear_uint8_within_one_lsb(fb, shape, kind):
    """uint8 INTER_LINEAR: >=95% bit-exact, remainder at most +-1 LSB.

    OpenCV's 8-bit path is fixed-point and its fused two-pass buffer discards
    low bits in an order the shim does not reproduce exactly. Single-axis
    resizes are bit-exact (see the H-only/V-only sweeps in
    scripts/validate_cv_compat.py); two-axis resizes disagree on ~0.5% of
    pixels by 1 LSB, scattered rather than confined to borders. This flag is
    NOT used anywhere in the orchestrator -- masks use INTER_NEAREST and depth
    uses float32 INTER_LINEAR, both of which are exact.
    """
    rng = np.random.default_rng(13)
    arr = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    dsize = _target(shape, kind)
    got = fb.resize(arr, dsize, interpolation=cv2.INTER_LINEAR).astype(int)
    ref = cv2.resize(arr, dsize, interpolation=cv2.INTER_LINEAR).astype(int)
    diff = np.abs(got - ref)
    assert diff.max() <= 1
    assert (diff == 0).mean() >= 0.95


@pytest.mark.parametrize("kind", TARGETS)
def test_resize_area_and_cubic_stay_within_tolerance(fb, kind):
    rng = np.random.default_rng(17)
    arr = rng.random((23, 29)).astype(np.float32)
    dsize = _target((23, 29), kind)
    for flag in (cv2.INTER_AREA, cv2.INTER_CUBIC):
        got = fb.resize(arr, dsize, interpolation=flag)
        ref = cv2.resize(arr, dsize, interpolation=flag)
        rel = np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-12)
        assert rel < 1e-6, f"flag={flag} kind={kind} rel={rel:g}"


def test_resize_noop_returns_equal_copy(fb):
    arr = np.arange(35, dtype=np.uint8).reshape(5, 7)
    for flag in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC):
        out = fb.resize(arr, (7, 5), interpolation=flag)
        np.testing.assert_array_equal(out, arr)
        assert out is not arr


def test_resize_2d_stays_2d_and_3d_stays_3d(fb):
    a2 = np.zeros((10, 12), np.uint8)
    a3 = np.zeros((10, 12, 3), np.uint8)
    assert fb.resize(a2, (6, 5), interpolation=cv2.INTER_NEAREST).shape == (5, 6)
    assert fb.resize(a3, (6, 5), interpolation=cv2.INTER_NEAREST).shape == (5, 6, 3)


def test_resize_preserves_dtype(fb):
    for dt in (np.uint8, np.float32, np.float64):
        arr = np.ones((10, 10), dt)
        for flag in (cv2.INTER_NEAREST, cv2.INTER_LINEAR):
            assert fb.resize(arr, (5, 5), interpolation=flag).dtype == dt


def test_resize_accepts_fx_fy(fb):
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    got = fb.resize(arr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    ref = cv2.resize(arr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(got, ref)


def test_resize_rejects_unknown_flag(fb):
    with pytest.raises(ValueError):
        fb.resize(np.zeros((8, 8), np.uint8), (4, 4), interpolation=99)


# --------------------------------------------------------------------------
# call-site behaviour: the fallback must not change the orchestrator helpers
# --------------------------------------------------------------------------

def test_mask_resize_call_site_agrees_with_cv2(fb):
    """vision_agents/multiview/load.py::_resize_mask_to_prediction"""
    rng = np.random.default_rng(19)
    mask = (rng.random((93, 121)) > 0.5).astype(np.uint8)
    target = (37, 51)
    ref = cv2.resize(mask, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
    got = fb.resize(mask, (target[1], target[0]), interpolation=fb.INTER_NEAREST)
    np.testing.assert_array_equal(got.astype(bool), ref.astype(bool))


def test_erode_mask_call_site_agrees_with_cv2(fb):
    """vision_agents/spatial_engine_v2/geometry/filtering.py::erode_mask"""
    rng = np.random.default_rng(23)
    mask = rng.random((80, 80)) > 0.3  # area well above the 500-px threshold
    ref = cv2.erode(mask.astype(np.uint8), KERNEL, iterations=1)
    got = fb.erode(mask.astype(np.uint8), KERNEL, iterations=1)
    np.testing.assert_array_equal(got, ref)
