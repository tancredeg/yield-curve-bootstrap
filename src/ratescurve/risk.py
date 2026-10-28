"""Interest rate risk of a set of positions priced off a curve.

Sign convention: every DV01 here is the profit for a 1 bp *fall* in rates,

    DV01 = -[PV(+1bp) - PV(-1bp)] / 2,

so a long bond position has a positive DV01 (central differences, error O(h^2)).

* :func:`parallel_dv01` shifts the whole zero curve.
* :func:`key_rate_dv01` applies triangular zero-rate bumps centred on key tenors
  (Ho, 1992). The bumps sum to a parallel shift, so the key-rate DV01s add up to
  the parallel DV01 (to second order in the bump size).
* :func:`bucketed_dv01` bumps one *market quote* at a time and recalibrates the
  curve: the sensitivity to each hedging instrument. Unlike key-rate DV01 it
  depends on the interpolation scheme, which is how interpolation locality shows
  up in risk.

Reference: T. S. Y. Ho (1992), "Key rate durations: measures of interest rate
risks", Journal of Fixed Income 2(2), 29-44.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .curve import DiscountCurve, Discounting, parallel, triangular

Valuation = Callable[[Discounting], float]


def parallel_dv01(value: Valuation, curve: DiscountCurve, bp: float = 1.0) -> float:
    """Profit for a 1 bp parallel fall of the zero curve (per unit of ``value``)."""
    up = value(curve.shifted(parallel(bp)))
    down = value(curve.shifted(parallel(-bp)))
    return -(up - down) / (2.0 * bp)


def key_rate_shifts(keys: Sequence[float], bp: float = 1.0) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Triangular bumps on ``keys``; flat beyond the first and last key."""
    ks = list(keys)
    if any(b <= a for a, b in zip(ks[:-1], ks[1:])):
        raise ValueError("key tenors must be strictly increasing")
    shifts = []
    for i, k in enumerate(ks):
        left = ks[i - 1] if i > 0 else None
        right = ks[i + 1] if i < len(ks) - 1 else None
        shifts.append(triangular(k, left, right, bp))
    return shifts


def key_rate_dv01(value: Valuation, curve: DiscountCurve, keys: Sequence[float], bp: float = 1.0) -> np.ndarray:
    """Key-rate DV01 for each key tenor (profit for a 1 bp fall at that key)."""
    up = key_rate_shifts(keys, bp)
    down = key_rate_shifts(keys, -bp)
    return np.array([-(value(curve.shifted(u)) - value(curve.shifted(d))) / (2.0 * bp) for u, d in zip(up, down)])


def bucketed_dv01(
    value: Valuation, rebuild: Callable[[int, float], Discounting], n_quotes: int, bp: float = 1.0
) -> np.ndarray:
    """Sensitivity to each market quote: ``rebuild(i, bp)`` recalibrates with quote ``i`` bumped.

    The quote bump is in the quote's own units (yield or par rate, in bp), so the
    result is the profit for a 1 bp fall in each instrument's quoted rate.
    """
    return np.array([-(value(rebuild(i, bp)) - value(rebuild(i, -bp))) / (2.0 * bp) for i in range(n_quotes)])
