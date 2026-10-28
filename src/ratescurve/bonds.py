"""Bond pricing off a curve and yield-based analytics (ICMA convention).

Curve pricing::

    dirty(settle) = sum_j CF_j * P(0, t_j),   t_j = ACT/365F(settle, d_j)

Yield convention (ICMA / ISMA, compounding at the coupon frequency ``f``)::

    dirty = sum_j CF_j * (1 + y/f) ** (-e_j)

where ``e_j`` is the ACT/ACT ICMA time from settlement to payment ``j`` in
coupon periods. For annual Bunds this is the yield the Bundesbank publishes.

Risk measures under the same yield basis (so the second-order Taylor
expansion ``dP/P = -D_mod dy + 0.5 C dy^2`` is exact to second order)::

    D_mod = -(1/P) dP/dy = (1/P) sum_j CF_j (e_j/f) (1+y/f)^(-e_j-1)
    D_mac = D_mod (1 + y/f)
    C     =  (1/P) d2P/dy2 = (1/P) sum_j CF_j (e_j/f)((e_j+1)/f) (1+y/f)^(-e_j-2)
    DV01  = D_mod * P * 1e-4   (price change for a 1 bp fall in yield)

Reference: Fabozzi, *Bond Markets, Analysis and Strategies*, ch. 4; ICMA Rule 251.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import brentq

from .conventions import yf_act365f
from .curve import DiscountCurve
from .instruments import FixedRateBond


def cashflow_times(bond: FixedRateBond, settle: date) -> tuple[np.ndarray, np.ndarray]:
    """Payment times (ACT/365F years from ``settle``) and amounts per face."""
    flows = bond.cashflows(settle)
    times = np.array([yf_act365f(settle, d) for d, _ in flows])
    amounts = np.array([a for _, a in flows])
    return times, amounts


def dirty_price_on_curve(bond: FixedRateBond, curve: DiscountCurve, settle: date) -> float:
    """Arbitrage-free dirty price per ``face`` from discount factors."""
    times, amounts = cashflow_times(bond, settle)
    return float(np.dot(amounts, curve.df(times)))


def clean_price_on_curve(bond: FixedRateBond, curve: DiscountCurve, settle: date) -> float:
    """Dirty price on the curve minus accrued interest."""
    return dirty_price_on_curve(bond, curve, settle) - bond.accrued(settle)


def _yield_terms(bond: FixedRateBond, settle: date) -> tuple[np.ndarray, np.ndarray]:
    flows = bond.cashflows(settle)
    exps = np.array(bond.period_exponents(settle, [d for d, _ in flows]))
    amounts = np.array([a for _, a in flows])
    return exps, amounts


def dirty_from_yield(bond: FixedRateBond, ytm: float, settle: date) -> float:
    """Dirty price per ``face`` for an ICMA yield ``ytm`` (decimal)."""
    exps, amounts = _yield_terms(bond, settle)
    g = 1.0 + ytm / bond.frequency
    if g <= 0:
        raise ValueError("yield below -frequency")
    return float(np.dot(amounts, g ** (-exps)))


def yield_from_dirty(bond: FixedRateBond, dirty: float, settle: date) -> float:
    """ICMA yield solving ``dirty_from_yield(y) = dirty`` (Brent, |error| < 1e-14)."""
    if dirty <= 0:
        raise ValueError("price must be positive")

    def f(y: float) -> float:
        return dirty_from_yield(bond, y, settle) - dirty

    return float(brentq(f, -0.5, 1.0, xtol=1e-14, rtol=1e-14, maxiter=200))


@dataclass(frozen=True)
class YieldRisk:
    """Yield-based risk of one bond per ``face``."""

    dirty: float
    ytm: float
    duration_mac: float
    duration_mod: float
    convexity: float
    dv01: float


def yield_risk(bond: FixedRateBond, ytm: float, settle: date) -> YieldRisk:
    """Macaulay/modified duration, convexity and DV01 under the ICMA yield basis."""
    exps, amounts = _yield_terms(bond, settle)
    f = bond.frequency
    g = 1.0 + ytm / f
    pv = amounts * g ** (-exps)
    price = float(pv.sum())
    d_mod = float(np.sum(amounts * (exps / f) * g ** (-exps - 1.0))) / price
    conv = float(np.sum(amounts * (exps / f) * ((exps + 1.0) / f) * g ** (-exps - 2.0))) / price
    return YieldRisk(
        dirty=price, ytm=ytm, duration_mac=d_mod * g, duration_mod=d_mod, convexity=conv, dv01=d_mod * price * 1e-4
    )
