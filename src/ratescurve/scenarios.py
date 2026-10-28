"""BCBS IRRBB standardised interest rate shock scenarios (Basel Framework SRP31).

Shock shapes (SRP31.90-31.92), with ``t`` in years and ``x = 4``::

    parallel(t)  = R_parallel
    short(t)     = R_short * exp(-t / x)
    long(t)      = R_long * (1 - exp(-t / x))
    steepener(t) = -0.65 |short(t)| + 0.9 |long(t)|
    flattener(t) = +0.80 |short(t)| - 0.6 |long(t)|

Six scenarios: parallel up/down, steepener, flattener, short up/down.
Shock sizes are the recalibrated SRP31 Table 2 values published by the BCBS in
July 2024 (document d578, in force 1 January 2026), in basis points
(parallel, short, long). No post-shock floor is applied: SRP31.93 leaves floors
to national supervisors (the EU regulatory floor is not modelled).

Worked example (SRP31.92, ``R_short = R_long = 100`` bp, ``t = 3.5``):
short = +41.7 bp, steepener = +25.4 bp, flattener = -1.6 bp.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .curve import Shift
from .interpolation import FloatArray

SHOCK_SIZES_BP: dict[str, tuple[float, float, float]] = {
    "EUR": (225.0, 350.0, 200.0),
    "USD": (200.0, 300.0, 225.0),
}
SCENARIOS = ("parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down")


def shock_bp(scenario: str, t: ArrayLike, sizes: tuple[float, float, float], x: float = 4.0) -> FloatArray:
    """Zero-rate shock in basis points at maturities ``t``."""
    tt = np.asarray(t, dtype=float)
    r_par, r_short, r_long = sizes
    short = r_short * np.exp(-tt / x)
    long = r_long * (1.0 - np.exp(-tt / x))
    shapes = {
        "parallel_up": np.full_like(tt, r_par),
        "parallel_down": np.full_like(tt, -r_par),
        "steepener": -0.65 * np.abs(short) + 0.9 * np.abs(long),
        "flattener": 0.8 * np.abs(short) - 0.6 * np.abs(long),
        "short_up": short,
        "short_down": -short,
    }
    if scenario not in shapes:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    return np.asarray(shapes[scenario], dtype=float)


def scenario_shift(scenario: str, currency: str) -> Shift:
    """Zero-rate shift (decimal) for ``scenario`` with the currency's SRP31 sizes."""
    sizes = SHOCK_SIZES_BP[currency]

    def shift(t: FloatArray) -> FloatArray:
        return shock_bp(scenario, t, sizes) * 1e-4

    return shift
