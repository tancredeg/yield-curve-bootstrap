"""Interpolation schemes: exactness, forward properties, locality, independent oracles."""

import json
from pathlib import Path

import numpy as np
import pytest

from ratescurve.curve import DiscountCurve
from ratescurve.interpolation import METHODS, MonotoneConvex, make_interpolator

TIMES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30.0])
ZEROS = np.array([0.0410, 0.0420, 0.0440, 0.0465, 0.0475, 0.0480, 0.0490, 0.0498, 0.0520, 0.0535, 0.0530])
FIXTURE = Path(__file__).parent / "fixtures" / "monotone_convex_reference.json"
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(12)


def _integral(fn, a: float, b: float, pieces: int = 400) -> float:
    """Composite 12-point Gauss-Legendre quadrature (no adaptive tolerance games)."""
    edges = np.linspace(a, b, pieces + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    pts = (mid[:, None] + half[:, None] * _NODES[None, :]).ravel()
    vals = np.asarray(fn(pts)).reshape(pieces, _NODES.size)
    return float(np.sum(half * (vals @ _WEIGHTS)))


@pytest.mark.parametrize("method", sorted(METHODS))
def test_pillar_discount_factors_exact(method):
    curve = DiscountCurve(TIMES, ZEROS, method)
    assert np.allclose(curve.df(TIMES), np.exp(-ZEROS * TIMES), rtol=0, atol=1e-15)
    assert np.allclose(curve.zero(TIMES), ZEROS, atol=1e-15)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_forward_integrates_to_log_discount(method):
    curve = DiscountCurve(TIMES, ZEROS, method)
    breaks = np.concatenate(([0.0], TIMES))
    for t in (0.1, 0.8, 4.2, 12.5, 29.0):
        pieces = [b for b in breaks if b < t] + [t]
        integral = sum(_integral(curve.forward, a, b) for a, b in zip(pieces[:-1], pieces[1:]))
        assert integral == pytest.approx(float(curve.integrated_forward(t)), abs=1e-12)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_flat_curve_is_reproduced(method):
    curve = DiscountCurve(TIMES, np.full(TIMES.size, 0.03), method)
    grid = np.linspace(0.0, 30.0, 501)
    assert np.allclose(curve.forward(grid), 0.03, atol=1e-14)
    assert np.allclose(curve.zero(grid), 0.03, atol=1e-14)


def test_domain_is_enforced():
    curve = DiscountCurve(TIMES, ZEROS)
    with pytest.raises(ValueError):
        curve.df(30.5)
    with pytest.raises(ValueError):
        curve.df(-0.1)
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 0.5], [0.01, 0.01])
    with pytest.raises(ValueError):
        DiscountCurve([0.0, 1.0], [0.01, 0.01])
    with pytest.raises(ValueError):
        make_interpolator("cubic", TIMES, ZEROS)
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 2.0], [0.01, 0.01], "natural_cubic_zero")
    with pytest.raises(ValueError):
        curve.discrete_forward(5, 5)


def _jumps(curve: DiscountCurve, at: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.abs(curve.forward(at + eps) - curve.forward(at - eps))


def test_forward_regularity_by_method():
    """Linear-zero and log-linear forwards jump at pillars; monotone convex is continuous everywhere.

    The natural spline is continuous at interior pillars but jumps at the first pillar, where the
    flat-zero short end meets the spline (z + t z' with z' != 0).
    """
    interior, first = TIMES[1:-1], TIMES[:1]
    jump = {m: _jumps(DiscountCurve(TIMES, ZEROS, m), TIMES[:-1]) for m in METHODS}
    assert jump["linear_zero"].max() > 1e-3
    assert jump["log_linear_df"].max() > 1e-3
    assert jump["monotone_convex"].max() < 1e-7
    spline = DiscountCurve(TIMES, ZEROS, "natural_cubic_zero")
    assert _jumps(spline, interior).max() < 1e-7
    assert _jumps(spline, first).max() > 1e-4


def test_log_linear_forwards_are_discrete_forwards():
    curve = DiscountCurve(TIMES, ZEROS, "log_linear_df")
    t0 = np.concatenate(([0.0], TIMES))
    h0 = np.concatenate(([0.0], TIMES * ZEROS))
    fd = np.diff(h0) / np.diff(t0)
    mids = 0.5 * (t0[:-1] + t0[1:])
    assert np.allclose(curve.forward(mids), fd, atol=1e-15)


def test_monotone_convex_preserves_discrete_forwards_and_nodes():
    mc = MonotoneConvex(TIMES, ZEROS)
    t0 = np.concatenate(([0.0], TIMES))
    h0 = np.concatenate(([0.0], TIMES * ZEROS))
    fd = np.diff(h0) / np.diff(t0)
    for i in range(TIMES.size):
        integral = _integral(mc.forward, t0[i], t0[i + 1])
        assert integral / (t0[i + 1] - t0[i]) == pytest.approx(fd[i], abs=1e-13)
    assert np.allclose(mc.forward(t0), mc.node_forwards(), atol=1e-14)


def test_monotone_convex_matches_independent_implementation():
    """Without the positivity collar, identical to tf-quant-finance's Hagan-West implementation."""
    ref = json.loads(FIXTURE.read_text())
    for case in ref["cases"]:
        mc = MonotoneConvex(case["times"], case["zeros"], positive=False)
        t = np.asarray(case["t"])
        assert np.allclose(mc.forward(t), case["forward"], rtol=0, atol=1e-13)
        pos = t > 0
        h = mc.integrated_forward(t[pos])
        assert np.allclose(h / t[pos], np.asarray(case["zero"])[pos], rtol=0, atol=1e-13)


def test_monotone_convex_positivity_amendment():
    rng = np.random.default_rng(7)
    times = np.array([0.5, 1, 2, 3, 5, 7, 10])
    worst_plain = np.inf
    for _ in range(300):
        fd = rng.uniform(0.0005, 0.06, times.size)
        zeros = np.cumsum(fd * np.diff(np.concatenate(([0.0], times)))) / times
        grid = np.linspace(0, times[-1], 2001)
        amended = MonotoneConvex(times, zeros, positive=True).forward(grid)
        plain = MonotoneConvex(times, zeros, positive=False).forward(grid)
        assert amended.min() >= -1e-15
        worst_plain = min(worst_plain, plain.min())
    assert worst_plain < 0  # the collar is needed: plain Hagan-West can go negative
    with pytest.raises(ValueError):
        MonotoneConvex([1.0, 2.0], [0.02, 0.005], positive=True)


def test_locality_of_monotone_convex_versus_spline():
    """A pillar bump moves monotone convex forwards only nearby; the spline moves them everywhere."""
    bumped = ZEROS.copy()
    bumped[3] += 1e-4  # 2y pillar
    far = np.linspace(10.5, 29.5, 50)  # beyond the 7y pillar
    for method, local in (("monotone_convex", True), ("log_linear_df", True), ("natural_cubic_zero", False)):
        base = DiscountCurve(TIMES, ZEROS, method).forward(far)
        moved = DiscountCurve(TIMES, bumped, method).forward(far)
        change = np.max(np.abs(moved - base))
        if local:
            assert change < 1e-15
        else:
            assert change > 1e-7


def test_quantlib_oracle_for_linear_and_log_linear():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    days = [91, 182, 365, 730, 1095, 1826, 2557, 3652, 5479, 7305, 10957]
    dates = [today + ql.Period(d, ql.Days) for d in days]
    t = np.array(days) / 365.0
    grid = np.linspace(0.01, t[-1], 1500)
    linear = ql.ZeroCurve([today, *dates], [ZEROS[0], *ZEROS], dc, ql.NullCalendar(), ql.Linear(), ql.Continuous)
    loglin = ql.DiscountCurve([today, *dates], [1.0, *np.exp(-ZEROS * t)], dc, ql.NullCalendar(), ql.LogLinear())
    for method, ref in (("linear_zero", linear), ("log_linear_df", loglin)):
        mine = DiscountCurve(t, ZEROS, method).df(grid)
        theirs = np.array([ref.discount(float(s)) for s in grid])
        assert np.max(np.abs(mine - theirs)) < 1e-14


def test_shifted_curve_forward_and_zero():
    curve = DiscountCurve(TIMES, ZEROS)

    def shift(s):
        return 1e-4 * np.asarray(s, dtype=float)

    shifted = curve.shifted(shift)
    grid = np.linspace(0.5, 29.0, 20)
    assert np.allclose(shifted.zero(grid), curve.zero(grid) + 1e-4 * grid, atol=1e-15)
    assert np.allclose(shifted.forward(grid), curve.forward(grid) + 2e-4 * grid, atol=1e-9)
    assert np.allclose(shifted.df(grid), np.exp(-grid * shifted.zero(grid)), rtol=1e-14)
    twice = shifted.shifted(shift)
    assert np.allclose(twice.zero(grid), curve.zero(grid) + 2e-4 * grid, atol=1e-15)
