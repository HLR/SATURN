"""Minimal cv2 compatibility layer for the SATURN orchestrator.

The orchestrator only needs three OpenCV entry points (``resize``, ``cvtColor``,
``erode``) plus a handful of interpolation/colour flags. ``cv2`` wheels are
not available for free-threaded CPython (``cp314t``).

Import policy
-------------
``import cv2`` is attempted first and, when it succeeds, every symbol below is
the *genuine* OpenCV callable re-exported verbatim. The numpy fallback is
only installed when OpenCV is unimportable.

Use :data:`USING_REAL_CV2` to tell the two regimes apart.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "USING_REAL_CV2",
    "resize",
    "cvtColor",
    "erode",
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_CUBIC",
    "INTER_AREA",
    "COLOR_RGB2BGR",
    "COLOR_BGR2RGB",
]

# Flag values are the real OpenCV enum values, so a mixed environment (a caller
# that grabbed a flag from cv2 directly) still interoperates.
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3

COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4  # OpenCV uses the same code; the swap is its own inverse.

try:  # pragma: no cover - environment dependent
    import cv2 as _cv2
except Exception:  # noqa: BLE001 - any import failure means "no OpenCV"
    _cv2 = None

USING_REAL_CV2 = _cv2 is not None

if USING_REAL_CV2:
    resize = _cv2.resize
    cvtColor = _cv2.cvtColor
    erode = _cv2.erode

    INTER_NEAREST = _cv2.INTER_NEAREST
    INTER_LINEAR = _cv2.INTER_LINEAR
    INTER_CUBIC = _cv2.INTER_CUBIC
    INTER_AREA = _cv2.INTER_AREA
    COLOR_RGB2BGR = _cv2.COLOR_RGB2BGR
    COLOR_BGR2RGB = _cv2.COLOR_BGR2RGB

else:
    # ------------------------------------------------------------------
    # Fallback implementations (numpy only).
    #
    # These reproduce OpenCV's resampling *math* rather than approximating it
    # with PIL/torch: OpenCV uses half-pixel sample centres with replicated
    # borders, which neither PIL's BILINEAR nor torch's default nearest mode
    # reproduce. Weights are built once per axis and applied separably.
    # ------------------------------------------------------------------

    def _clip_idx(idx: np.ndarray, n: int) -> np.ndarray:
        return np.clip(idx, 0, n - 1)

    def _nearest_map(src: int, dst: int) -> np.ndarray:
        # OpenCV truncates without a half-pixel shift, and derives the step as
        # the reciprocal of the (dst/src) ratio -- not as src/dst. The double
        # round-trip can change the last bit and move a sample by one pixel,
        # so it is reproduced exactly here.
        step = 1.0 / (dst / src)
        return _clip_idx(np.floor(np.arange(dst) * step).astype(np.intp), src)

    def _linear_coeffs(src: int, dst: int):
        scale = 1.0 / (dst / src)
        fx = (np.arange(dst) + 0.5) * scale - 0.5
        sx = np.floor(fx).astype(np.intp)
        frac = fx - sx
        # OpenCV pins the fraction to 0 outside the source support.
        frac = np.where((sx < 0) | (sx >= src - 1), 0.0, frac)
        sx = _clip_idx(sx, src)
        idx = np.stack([_clip_idx(sx, src), _clip_idx(sx + 1, src)], axis=1)
        return idx, np.stack([1.0 - frac, frac], axis=1)

    def _cubic_coeffs(src: int, dst: int):
        A = -0.75  # OpenCV's Catmull-Rom parameter
        scale = 1.0 / (dst / src)
        fx = (np.arange(dst) + 0.5) * scale - 0.5
        sx = np.floor(fx).astype(np.intp)
        x = fx - sx
        w = np.empty((dst, 4), dtype=np.float64)
        w[:, 0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
        w[:, 1] = ((A + 2) * x - (A + 3)) * x * x + 1
        y = 1 - x
        w[:, 2] = ((A + 2) * y - (A + 3)) * y * y + 1
        w[:, 3] = 1.0 - w[:, 0] - w[:, 1] - w[:, 2]
        idx = _clip_idx(sx[:, None] + np.arange(-1, 3)[None, :], src)
        return idx, w

    def _area_shrink_coeffs(src: int, dst: int):
        """Exact fractional-area weights (OpenCV's true INTER_AREA path)."""
        scale = 1.0 / (dst / src)
        starts = np.arange(dst) * scale
        ends = starts + scale
        width = int(np.ceil(scale)) + 1
        base = np.floor(starts).astype(np.intp)
        idx = base[:, None] + np.arange(width)[None, :]
        lo = np.maximum(idx, starts[:, None])
        hi = np.minimum(idx + 1, ends[:, None])
        w = np.clip(hi - lo, 0.0, None)
        w /= w.sum(axis=1, keepdims=True)
        return _clip_idx(idx, src), w

    def _area_generic_coeffs(src: int, dst: int):
        """OpenCV's INTER_AREA fallback used when *either* axis is upscaled.

        It is a 2-tap filter like INTER_LINEAR but with area-style sample
        positions: ``sx = floor(dx*scale)`` and ``fx = (dx+1) - (sx+1)*inv_scale``, clamped to
        zero when negative. Matches OpenCV 4.12.
        """
        inv_scale = dst / src
        scale = 1.0 / inv_scale
        dx = np.arange(dst)
        sx = np.floor(dx * scale).astype(np.intp)
        fx = (dx + 1) - (sx + 1) * inv_scale
        fx = np.where(fx <= 0, 0.0, fx - np.floor(fx))
        fx = np.where((sx < 0) | (sx >= src - 1), 0.0, fx)
        idx = np.stack([_clip_idx(sx, src), _clip_idx(sx + 1, src)], axis=1)
        return idx, np.stack([1.0 - fx, fx], axis=1)

    def _axis_coeffs(src: int, dst: int, interpolation: int, area_generic: bool):
        if interpolation == INTER_LINEAR:
            return _linear_coeffs(src, dst)
        if interpolation == INTER_CUBIC:
            return _cubic_coeffs(src, dst)
        if interpolation == INTER_AREA:
            return _area_generic_coeffs(src, dst) if area_generic else _area_shrink_coeffs(src, dst)
        raise ValueError(f"unsupported interpolation flag: {interpolation!r}")

    def _gather(arr: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
        """arr -> shape with `axis` replaced by (dst, K)."""
        g = np.take(arr, idx.ravel(), axis=axis)
        shape = list(g.shape)
        shape[axis : axis + 1] = [idx.shape[0], idx.shape[1]]
        return g.reshape(shape)

    def _wshape(ndim: int, axis: int, idx: np.ndarray):
        s = [1] * ndim
        s[axis] = idx.shape[0]
        s[axis + 1] = idx.shape[1]
        return s

    _COEF_BITS = 11
    _COEF_SCALE = 1 << _COEF_BITS  # 2048, OpenCV's INTER_RESIZE_COEF_SCALE

    def _fixed_point_u8(arr, hidx, hw, vidx, vw):
        """Bit-exact replay of OpenCV's 8-bit fixed-point separable resize."""
        alpha = np.clip(np.rint(hw * _COEF_SCALE), -32768, 32767).astype(np.int32)
        beta = np.clip(np.rint(vw * _COEF_SCALE), -32768, 32767).astype(np.int32)
        src = arr.astype(np.int32, copy=False)
        # horizontal pass first, matching OpenCV's row-then-column order
        g = _gather(src, hidx, 1)
        row = (g * alpha.reshape(_wshape(g.ndim, 1, hidx))).sum(axis=2)
        # OpenCV's uchar VResizeLinear is *not* a plain 22-bit cast: it uses
        # the specialised  ((b*(S>>4))>>16) ... +2) >> 2  sequence, which loses
        # low bits differently. Reproduced literally.
        g = _gather(row, vidx, 0)
        shifted = (g >> 4) * beta.reshape(_wshape(g.ndim, 0, vidx))
        acc = (shifted >> 16).sum(axis=1)
        out = (acc + 2) >> 2
        return np.clip(out, 0, 255).astype(np.uint8)

    def resize(src, dsize=None, dst=None, fx=0, fy=0, interpolation=INTER_LINEAR):
        """numpy re-implementation of ``cv2.resize`` (subset)."""
        arr = np.asarray(src)
        if arr.ndim not in (2, 3):
            raise ValueError("resize expects a 2D or 3D array")
        src_h, src_w = arr.shape[:2]
        if dsize is not None and tuple(dsize) != (0, 0):
            dst_w, dst_h = int(dsize[0]), int(dsize[1])
        else:
            if not fx or not fy:
                raise ValueError("either dsize or both fx and fy must be given")
            dst_w = int(round(src_w * fx))
            dst_h = int(round(src_h * fy))
        if dst_w <= 0 or dst_h <= 0:
            raise ValueError("resize target must be positive")

        in_dtype = arr.dtype
        if (dst_w, dst_h) == (src_w, src_h):
            return arr.copy()

        if interpolation == INTER_NEAREST:
            out = np.take(arr, _nearest_map(src_w, dst_w), axis=1)
            out = np.take(out, _nearest_map(src_h, dst_h), axis=0)
            return np.ascontiguousarray(out.astype(in_dtype, copy=False))

        # OpenCV only takes the true-area path when BOTH axes shrink.
        area_generic = interpolation == INTER_AREA and not (
            src_w >= dst_w and src_h >= dst_h
        )
        hidx, hw = _axis_coeffs(src_w, dst_w, interpolation, area_generic)
        vidx, vw = _axis_coeffs(src_h, dst_h, interpolation, area_generic)

        # OpenCV's 8-bit generic path is fixed-point; the true-area shrink path
        # is not, so only the 2-tap flavours go through the integer replay.
        if in_dtype == np.uint8 and (
            interpolation == INTER_LINEAR
            or (interpolation == INTER_AREA and area_generic)
        ):
            return np.ascontiguousarray(_fixed_point_u8(arr, hidx, hw, vidx, vw))

        # Horizontal pass then vertical, matching OpenCV's ordering so that
        # float rounding accumulates the same way.
        acc_dtype = np.float64 if in_dtype == np.float64 else np.float32
        work = arr.astype(acc_dtype, copy=False)
        g = _gather(work, hidx, 1)
        work = (g * hw.astype(acc_dtype).reshape(_wshape(g.ndim, 1, hidx))).sum(axis=2)
        g = _gather(work, vidx, 0)
        work = (g * vw.astype(acc_dtype).reshape(_wshape(g.ndim, 0, vidx))).sum(axis=1)

        if np.issubdtype(in_dtype, np.integer):
            info = np.iinfo(in_dtype)
            work = np.clip(np.rint(work), info.min, info.max)
        return np.ascontiguousarray(work.astype(in_dtype, copy=False))

    def cvtColor(src, code, dst=None, dstCn=0):
        """Only the RGB<->BGR channel swap the orchestrator needs."""
        if code not in (COLOR_RGB2BGR, COLOR_BGR2RGB):
            raise NotImplementedError(
                f"cv_compat.cvtColor fallback only supports RGB<->BGR, got {code!r}"
            )
        arr = np.asarray(src)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("cvtColor expects an HxWx3 or HxWx4 array")
        out = arr.copy()
        out[..., :3] = arr[..., 2::-1]
        return np.ascontiguousarray(out)

    def erode(src, kernel, dst=None, anchor=None, iterations=1, **kwargs):
        """Binary/greyscale erosion via a sliding-window minimum.

        Matches ``cv2.erode`` for the rectangular all-ones kernels used in this
        repo. OpenCV's default ``borderValue`` for erode is +inf, i.e. the
        border never constrains the minimum, which is reproduced by padding
        with the dtype maximum.
        """
        arr = np.asarray(src)
        k = np.asarray(kernel)
        if k.ndim != 2:
            raise ValueError("erode expects a 2D kernel")
        if not np.all(k != 0):
            raise NotImplementedError(
                "cv_compat.erode fallback only supports fully-solid kernels"
            )
        if anchor not in (None, (-1, -1)):
            raise NotImplementedError("cv_compat.erode fallback assumes a centred anchor")
        if iterations <= 0:
            return arr.copy()

        kh, kw = k.shape
        ay, ax = kh // 2, kw // 2
        pad_val = (
            np.iinfo(arr.dtype).max
            if np.issubdtype(arr.dtype, np.integer)
            else np.inf
        )
        out = arr
        for _ in range(int(iterations)):
            pad = [(ay, kh - 1 - ay), (ax, kw - 1 - ax)] + [(0, 0)] * (out.ndim - 2)
            padded = np.pad(out, pad, mode="constant", constant_values=pad_val)
            windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(0, 1))
            out = windows.min(axis=(-2, -1))
        return np.ascontiguousarray(out.astype(arr.dtype, copy=False))
