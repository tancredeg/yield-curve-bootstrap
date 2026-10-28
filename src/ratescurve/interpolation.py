"""Curve interpolation schemes on pillar zero rates.

Every scheme is defined on pillars ``0 < t_1 < ... < t_n`` with continuously
compounded zero rates ``z_i`` and exposes the integrated forward
``H(t) = -ln P(0, t) = t z(t)`` and the instantaneous forward ``f(t) = H'(t)``.
All schemes reproduce the pillar discount factors exactly, so a curve
calibrated with one scheme reprices its instruments under that same scheme.

Schemes:

``linear_zero``
    Linear interpolation of ``z(t)``. Forwards jump at every pillar.
``log_linear_df``
    Linear interpolation of ``ln P``: piecewise-flat forwards equal to the
    discrete forwards between pillars (also called raw interpolation).
``monotone_convex``
    Hagan & West (2006). Interpolates the instantaneous forward so that it is
    continuous, preserves every discrete forward exactly (hence the pillar
    discount factors), is local, and stays positive when the discrete forwards
    are positive (amended version with the forward collar).
``natural_cubic_zero``
    Natural cubic spline on ``z(t)``. Smooth zeros, but non-local and prone to
    overshooting; kept as a counter-example.

Below the first pillar every scheme except monotone convex holds the zero rate
flat at ``z_1``; monotone convex extrapolates the forward to ``t = 0`` as in
Hagan & West. Beyond the last pillar every scheme raises ``ValueError``.

References:
    P. S. Hagan and G. West (2006), "Interpolation methods for curve
    construction", Applied Mathematical Finance 13(2), 89-129.
    P. S. Hagan and G. West (2008), "Methods for constructing a yield curve",
    Wilmott Magazine, May, 70-81.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

FloatArray = NDArray[np.float64]


class Interpolator(ABC):
    """Base class: validates pillars and the evaluation domain."""

    name: str = "base"

    def __init__(self, times: ArrayLike, zeros: ArrayLike) -> None:
        t = np.asarray(times, dtype=float)
        z = np.asarray(zeros, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or t.size == 0:
            raise ValueError("times and zeros must be 1-D arrays of equal, positive length")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("pillar times must be positive and strictly increasing")
        if not np.all(np.isfinite(z)):
            raise ValueError("zero rates must be finite")
        self.times = t
        self.zeros = z
        self.t_max = float(t[-1])

    def _check(self, t: ArrayLike) -> FloatArray:
        arr = np.asarray(t, dtype=float)
        if np.any(arr < 0) or np.any(~np.isfinite(arr)):
            raise ValueError("curve time must be finite and non-negative")
        if np.any(arr > self.t_max * (1 + 1e-12)):
            raise ValueError(f"extrapolation beyond last pillar {self.t_max:.4f}y is not allowed")
        return np.minimum(arr, self.t_max)

    def integrated_forward(self, t: ArrayLike) -> FloatArray:
        """``H(t) = -ln P(0, t)``."""
        return self._integrated(self._check(t))

    def forward(self, t: ArrayLike) -> FloatArray:
        """Instantaneous forward ``f(t)`` (right-continuous at pillars)."""
        return self._forward(self._check(t))

    @abstractmethod
    def _integrated(self, t: FloatArray) -> FloatArray: ...

    @abstractmethod
    def _forward(self, t: FloatArray) -> FloatArray: ...


class LinearZero(Interpolator):
    """Linear interpolation on continuously compounded zero rates."""

    name = "linear_zero"

    def _segment_slope(self, t: FloatArray) -> FloatArray:
        slopes = np.diff(self.zeros) / np.diff(self.times)
        idx = np.searchsorted(self.times, t, side="right") - 1
        out = np.zeros_like(t)
        inside = (idx >= 0) & (idx < len(slopes))
        out[inside] = slopes[idx[inside]]
        last = idx >= len(slopes)
        if len(slopes):
            out[last] = slopes[-1]
        return out

    def _integrated(self, t: FloatArray) -> FloatArray:
        return t * np.interp(t, self.times, self.zeros)

    def _forward(self, t: FloatArray) -> FloatArray:
        return np.interp(t, self.times, self.zeros) + t * self._segment_slope(t)


class LogLinearDF(Interpolator):
    """Linear interpolation on ln P: piecewise-flat instantaneous forwards."""

    name = "log_linear_df"

    def __init__(self, times: ArrayLike, zeros: ArrayLike) -> None:
        super().__init__(times, zeros)
        self._t0 = np.concatenate(([0.0], self.times))
        self._h0 = np.concatenate(([0.0], self.times * self.zeros))
        self._fd = np.diff(self._h0) / np.diff(self._t0)

    def _integrated(self, t: FloatArray) -> FloatArray:
        return np.interp(t, self._t0, self._h0)

    def _forward(self, t: FloatArray) -> FloatArray:
        idx = np.clip(np.searchsorted(self._t0, t, side="right") - 1, 0, len(self._fd) - 1)
        return self._fd[idx]


class NaturalCubicZero(Interpolator):
    """Natural cubic spline on zero rates (flat zero below the first pillar)."""

    name = "natural_cubic_zero"

    def __init__(self, times: ArrayLike, zeros: ArrayLike) -> None:
        super().__init__(times, zeros)
        if self.times.size < 3:
            raise ValueError("natural cubic spline needs at least three pillars")
        self._spline = CubicSpline(self.times, self.zeros, bc_type="natural", extrapolate=False)

    def _zero(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        z = np.where(t < self.times[0], self.zeros[0], self._spline(np.maximum(t, self.times[0])))
        dz = np.where(t < self.times[0], 0.0, self._spline(np.maximum(t, self.times[0]), 1))
        return z, dz

    def _integrated(self, t: FloatArray) -> FloatArray:
        z, _ = self._zero(t)
        return t * z

    def _forward(self, t: FloatArray) -> FloatArray:
        z, dz = self._zero(t)
        return z + t * dz


class MonotoneConvex(Interpolator):
    """Hagan-West monotone convex interpolation of instantaneous forwards.

    Discrete forwards ``fd_i = (H_i - H_{i-1}) / (t_i - t_{i-1})`` (with
    ``t_0 = 0``) are preserved exactly. Node forwards are estimated as
    length-weighted averages of adjacent discrete forwards, extrapolated at
    both ends, optionally collared to ``[0, 2 min(fd_i, fd_{i+1})]`` for
    positivity, and on each interval the deviation ``g(x) = f - fd_i`` follows
    one of the four Hagan-West shapes, each with zero integral on [0, 1].
    """

    name = "monotone_convex"

    def __init__(self, times: ArrayLike, zeros: ArrayLike, positive: bool = True) -> None:
        super().__init__(times, zeros)
        t0 = np.concatenate(([0.0], self.times))
        h0 = np.concatenate(([0.0], self.times * self.zeros))
        fd = np.diff(h0) / np.diff(t0)
        n = fd.size
        f = np.empty(n + 1)
        if n == 1:
            f[:] = fd[0]
        else:
            dt = np.diff(t0)
            f[1:n] = (dt[:-1] * fd[1:] + dt[1:] * fd[:-1]) / (dt[:-1] + dt[1:])
            f[0] = fd[0] - 0.5 * (f[1] - fd[0])
            f[n] = fd[-1] - 0.5 * (f[n - 1] - fd[-1])
        if positive:
            if np.any(fd < 0):
                raise ValueError("positivity amendment requires non-negative discrete forwards")
            f[0] = np.clip(f[0], 0.0, 2.0 * fd[0])
            f[n] = np.clip(f[n], 0.0, 2.0 * fd[-1])
            if n > 1:
                f[1:n] = np.clip(f[1:n], 0.0, 2.0 * np.minimum(fd[:-1], fd[1:]))
        self.positive = positive
        self._t0, self._h0, self._fd, self._fnode = t0, h0, fd, f
        self._g0 = f[:-1] - fd
        self._g1 = f[1:] - fd

    def node_forwards(self) -> FloatArray:
        """Estimated instantaneous forwards at ``0, t_1, ..., t_n``."""
        return self._fnode.copy()

    def _locate(self, t: FloatArray) -> tuple[NDArray[np.intp], FloatArray]:
        idx = np.clip(np.searchsorted(self._t0, t, side="right") - 1, 0, self._fd.size - 1)
        x = (t - self._t0[idx]) / (self._t0[idx + 1] - self._t0[idx])
        return idx, np.clip(x, 0.0, 1.0)

    @staticmethod
    def _g_and_integral(g0: FloatArray, g1: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Hagan-West deviation ``g(x)`` and ``G(x) = int_0^x g`` (vectorised)."""
        g = np.zeros_like(x)
        G = np.zeros_like(x)
        tiny = 1e-15
        zero = (np.abs(g0) < tiny) | (np.abs(g1) < tiny)
        quad = ~zero & (
            ((g0 < 0) & (g1 >= -0.5 * g0) & (g1 <= -2.0 * g0)) | ((g0 > 0) & (g1 <= -0.5 * g0) & (g1 >= -2.0 * g0))
        )
        case3 = ~zero & ~quad & (((g0 < 0) & (g1 > -2.0 * g0)) | ((g0 > 0) & (g1 < -2.0 * g0)))
        case4 = ~zero & ~quad & ~case3 & (((g0 > 0) & (g1 < 0)) | ((g0 < 0) & (g1 > 0)))
        case5 = ~zero & ~quad & ~case3 & ~case4

        m = quad
        a, b, s = g0[m], g1[m], x[m]
        g[m] = a * (1 - 4 * s + 3 * s**2) + b * (-2 * s + 3 * s**2)
        G[m] = a * (s - 2 * s**2 + s**3) + b * (-(s**2) + s**3)

        m = case3
        a, b, s = g0[m], g1[m], x[m]
        eta = (b + 2 * a) / (b - a)
        right = s > eta
        u = np.where(right, (s - eta) / (1 - eta), 0.0)
        g[m] = a + (b - a) * u**2
        G[m] = a * s + (b - a) * np.where(right, (s - eta) ** 3 / (3 * (1 - eta) ** 2), 0.0)

        m = case4
        a, b, s = g0[m], g1[m], x[m]
        eta = 3 * b / (b - a)
        left = s < eta
        u = np.where(left, (eta - s) / eta, 0.0)
        g[m] = b + (a - b) * u**2
        G[m] = b * s + (a - b) * np.where(left, (eta**3 - (eta - s) ** 3) / (3 * eta**2), eta / 3)

        m = case5
        a, b, s = g0[m], g1[m], x[m]
        big_a = -a * b / (a + b)
        eta = b / (a + b)
        left = s <= eta
        u_left = np.where(left, (eta - s) / eta, 0.0)
        u_right = np.where(left, 0.0, (s - eta) / (1 - eta))
        g[m] = np.where(left, big_a + (a - big_a) * u_left**2, big_a + (b - big_a) * u_right**2)
        G[m] = np.where(
            left,
            big_a * s + (a - big_a) * (eta**3 - (eta - s) ** 3) / (3 * eta**2),
            big_a * s + (a - big_a) * eta / 3 + (b - big_a) * (s - eta) ** 3 / (3 * (1 - eta) ** 2),
        )
        return g, G

    def _integrated(self, t: FloatArray) -> FloatArray:
        idx, x = self._locate(t)
        _, G = self._g_and_integral(self._g0[idx], self._g1[idx], x)
        width = self._t0[idx + 1] - self._t0[idx]
        return self._h0[idx] + width * (self._fd[idx] * x + G)

    def _forward(self, t: FloatArray) -> FloatArray:
        idx, x = self._locate(t)
        g, _ = self._g_and_integral(self._g0[idx], self._g1[idx], x)
        return self._fd[idx] + g


METHODS: dict[str, type[Interpolator]] = {
    cls.name: cls for cls in (LinearZero, LogLinearDF, MonotoneConvex, NaturalCubicZero)
}


def make_interpolator(method: str, times: ArrayLike, zeros: ArrayLike) -> Interpolator:
    """Factory: ``method`` is one of :data:`METHODS`."""
    try:
        cls = METHODS[method]
    except KeyError as exc:
        raise ValueError(f"unknown interpolation method {method!r}; choose from {sorted(METHODS)}") from exc
    return cls(times, zeros)
