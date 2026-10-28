"""Discount curve objects.

A :class:`DiscountCurve` is a set of pillar zero rates plus an interpolation
scheme; it returns discount factors, zero rates and instantaneous forwards
from the same object, so pricing, risk and plots can never disagree on the
curve. A :class:`ShiftedCurve` adds a deterministic zero-rate shift ``s(t)``
evaluated continuously in ``t`` (never re-interpolated), which is how
scenarios, key-rate bumps and parallel bumps are applied.

Conventions: time in years (ACT/365F from the curve date for dated
instruments), continuously compounded zero rates, ``P(0, t) = exp(-t z(t))``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .interpolation import FloatArray, Interpolator, make_interpolator

Shift = Callable[[FloatArray], FloatArray]


class Discounting(Protocol):
    """Anything that returns discount factors."""

    t_max: float

    def df(self, t: ArrayLike) -> FloatArray: ...


class DiscountCurve:
    """Pillar zero rates + interpolation scheme."""

    def __init__(self, times: ArrayLike, zeros: ArrayLike, method: str = "monotone_convex", label: str = "") -> None:
        self.method = method
        self.label: str = label
        self._interp: Interpolator = make_interpolator(method, times, zeros)
        self.times = self._interp.times
        self.zeros = self._interp.zeros
        self.t_max: float = self._interp.t_max

    def __repr__(self) -> str:
        return f"DiscountCurve({self.label!r}, method={self.method!r}, pillars={self.times.size})"

    def integrated_forward(self, t: ArrayLike) -> FloatArray:
        """``H(t) = -ln P(0, t)``."""
        return self._interp.integrated_forward(t)

    def df(self, t: ArrayLike) -> FloatArray:
        """Discount factor ``P(0, t)``."""
        return np.exp(-self.integrated_forward(t))

    def zero(self, t: ArrayLike) -> FloatArray:
        """Continuously compounded zero rate; ``zero(0)`` is the limit ``f(0)``."""
        arr = np.asarray(t, dtype=float)
        h = self.integrated_forward(arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(arr > 0, h / np.where(arr > 0, arr, 1.0), self.forward(np.zeros_like(arr)))
        return z

    def forward(self, t: ArrayLike) -> FloatArray:
        """Instantaneous forward rate ``f(t) = -d ln P / dt``."""
        return self._interp.forward(t)

    def discrete_forward(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between ``t1`` and ``t2``."""
        if t2 <= t1:
            raise ValueError("t2 must exceed t1")
        h = self.integrated_forward(np.array([t1, t2]))
        return float((h[1] - h[0]) / (t2 - t1))

    def with_zeros(self, zeros: ArrayLike) -> DiscountCurve:
        """Same pillars and scheme, new pillar zero rates."""
        return DiscountCurve(self.times, zeros, self.method, self.label)

    def shifted(self, shift: Shift) -> ShiftedCurve:
        """Curve with zero rates ``z(t) + shift(t)``."""
        return ShiftedCurve(self, shift)


class ShiftedCurve:
    """``z'(t) = z(t) + s(t)``, evaluated pointwise: ``P'(t) = P(t) exp(-t s(t))``."""

    def __init__(self, base: DiscountCurve | ShiftedCurve, shift: Shift) -> None:
        self.base = base
        self.shift = shift
        self.t_max: float = float(base.t_max)
        self.label: str = f"{base.label} (shifted)"

    def df(self, t: ArrayLike) -> FloatArray:
        arr = np.asarray(t, dtype=float)
        return self.base.df(arr) * np.exp(-arr * self.shift(arr))

    def zero(self, t: ArrayLike) -> FloatArray:
        arr = np.asarray(t, dtype=float)
        return self.base.zero(arr) + self.shift(arr)

    def forward(self, t: ArrayLike, h: float = 1e-6) -> FloatArray:
        """Forward of the shifted curve: ``f(t) + s(t) + t s'(t)`` (s' by central difference)."""
        arr = np.asarray(t, dtype=float)
        lo = np.maximum(arr - h, 0.0)
        hi = np.minimum(arr + h, self.t_max)
        ds = (self.shift(hi) - self.shift(lo)) / (hi - lo)
        return self.base.forward(arr) + self.shift(arr) + arr * ds

    def shifted(self, shift: Shift) -> ShiftedCurve:
        return ShiftedCurve(self, shift)


def parallel(bp: float) -> Shift:
    """Constant zero-rate shift of ``bp`` basis points."""
    amount = bp * 1e-4

    def shift(t: FloatArray) -> FloatArray:
        return np.full_like(np.asarray(t, dtype=float), amount)

    return shift


def triangular(key: float, left: float | None, right: float | None, bp: float = 1.0) -> Shift:
    """Key-rate bump (Ho, 1992): ``bp`` at ``key``, linear to 0 at the neighbouring keys.

    ``left=None`` keeps the bump flat below ``key``; ``right=None`` keeps it flat
    above. The bumps of a full set of key tenors sum to a parallel shift.
    """
    amount = bp * 1e-4

    def shift(t: FloatArray) -> FloatArray:
        arr = np.asarray(t, dtype=float)
        w = np.zeros_like(arr)
        below = arr <= key
        if left is None:
            w[below] = 1.0
        else:
            w[below] = np.clip((arr[below] - left) / (key - left), 0.0, 1.0)
        above = ~below
        if right is None:
            w[above] = 1.0
        else:
            w[above] = np.clip((right - arr[above]) / (right - key), 0.0, 1.0)
        return amount * w

    return shift
