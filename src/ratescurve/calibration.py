"""Exact calibration of pillar zero rates to instrument prices.

One pillar per instrument (at its maturity). The pillar zero rates ``z`` solve
the square system ``model_price_i(curve(z)) = market_price_i`` for all ``i``
simultaneously (MINPACK hybrid Powell method), so the result does not depend
on processing order and works for non-local schemes such as the natural cubic
spline. Each instrument is priced with the same interpolation scheme that will
be used downstream: calibration curve and pricing curve are one object.

The calibration is accepted only if every residual is below
``tol * max(1, |market price|)``; otherwise :class:`CalibrationError` is raised.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root

from .bonds import cashflow_times
from .curve import DiscountCurve, Discounting
from .instruments import FixedRateBond, ParInstrument

Pricer = Callable[[Discounting], float]


class CalibrationError(RuntimeError):
    """Raised when the solver does not reprice every instrument within tolerance."""


def bond_pricer(bond: FixedRateBond, settle: date, nominal: float = 1.0) -> Pricer:
    """Dirty price per 100 (times ``nominal``) from a curve; cash flows precomputed."""
    times, amounts = cashflow_times(bond, settle)
    weights = amounts * nominal

    def price(curve: Discounting) -> float:
        return float(np.dot(weights, curve.df(times)))

    return price


def par_pricer(instrument: ParInstrument) -> Pricer:
    """Price per unit face of a constant-maturity par instrument (par = 1)."""
    flows = instrument.cashflow_times()
    times = np.array([t for t, _ in flows])
    amounts = np.array([a for _, a in flows])

    def price(curve: Discounting) -> float:
        return float(np.dot(amounts, curve.df(times)))

    return price


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated curve and its repricing residuals (model minus market)."""

    curve: DiscountCurve
    residuals: np.ndarray
    evaluations: int

    @property
    def max_abs_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))


def calibrate(
    pricers: Sequence[Pricer],
    targets: ArrayLike,
    pillar_times: ArrayLike,
    method: str,
    guess: ArrayLike,
    label: str = "",
    tol: float = 1e-10,
) -> CalibrationResult:
    """Solve the pillar zero rates that reprice every instrument exactly.

    Args:
        pricers: one model pricer per instrument.
        targets: market prices (same units as the pricers).
        pillar_times: one pillar per instrument, strictly increasing.
        method: interpolation scheme (see :mod:`ratescurve.interpolation`).
        guess: starting zero rates (for instance yields converted to continuous).
        label: curve label.
        tol: relative repricing tolerance.
    """
    targets_arr = np.asarray(targets, dtype=float)
    times = np.asarray(pillar_times, dtype=float)
    if len(pricers) != times.size or targets_arr.size != times.size:
        raise ValueError("need exactly one pillar and one target per instrument")

    def residuals(z: np.ndarray) -> np.ndarray:
        curve = DiscountCurve(times, z, method, label)
        return np.array([p(curve) for p in pricers]) - targets_arr

    sol = root(
        residuals,
        np.asarray(guess, dtype=float),
        method="hybr",
        options={"xtol": 1e-15, "maxfev": 200 * (times.size + 1)},
    )
    z = sol.x
    res = residuals(z)
    limit = tol * np.maximum(1.0, np.abs(targets_arr))
    if not np.all(np.abs(res) <= limit):
        worst = int(np.argmax(np.abs(res) / limit))
        raise CalibrationError(
            f"{label} [{method}]: instrument {worst} residual {res[worst]:.3e} "
            f"exceeds {limit[worst]:.1e} ({sol.message})"
        )
    return CalibrationResult(DiscountCurve(times, z, method, label), res, int(sol.nfev))
