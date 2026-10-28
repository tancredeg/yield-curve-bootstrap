"""Risk identities: key-rate sum, bucketed sum, bucketed DV01 of a calibration bond."""

import numpy as np
import pytest

from ratescurve.bonds import yield_risk
from ratescurve.calibration import bond_pricer
from ratescurve.curve import DiscountCurve, parallel
from ratescurve.risk import bucketed_dv01, key_rate_dv01, key_rate_shifts, parallel_dv01

KEYS = (1, 2, 3, 5, 7, 10, 15, 20, 30)


def _portfolio(market, isins_nominals):
    quotes = market.instruments
    return [
        (bond_pricer(next(b for b, i in zip(quotes["bond"], quotes["isin"]) if i == isin), market.settle), n)
        for isin, n in isins_nominals
    ]


def test_key_rate_bumps_partition_a_parallel_shift():
    grid = np.linspace(0, 30, 301)
    total = sum(s(grid) for s in key_rate_shifts(KEYS))
    assert np.allclose(total, 1e-4, atol=1e-18)
    with pytest.raises(ValueError):
        key_rate_shifts((1, 1, 2))


def test_key_rate_dv01_sums_to_parallel(eur_markets):
    m = eur_markets["monotone_convex"]
    bond = m.instruments["bond"].iloc[9]
    price = bond_pricer(bond, m.settle)
    krd = key_rate_dv01(price, m.curve, KEYS)
    par = parallel_dv01(price, m.curve)
    assert par > 0
    assert krd.sum() == pytest.approx(par, rel=1e-6)  # equal up to O(bump^2) terms


@pytest.mark.parametrize("method", ["log_linear_df", "monotone_convex", "natural_cubic_zero"])
def test_bucketed_dv01_of_calibration_bond_is_its_own_yield_dv01(eur_markets, method):
    """A calibration bond reprices exactly, so its only sensitivity is to its own quote."""
    m = eur_markets[method]
    k = 6
    bond = m.instruments["bond"].iloc[k]
    price = bond_pricer(bond, m.settle)
    bucket = bucketed_dv01(price, m.rebuild, len(m.quotes))
    own = yield_risk(bond, m.quotes[k], m.settle).dv01
    assert bucket[k] == pytest.approx(own, rel=1e-6)
    others = np.delete(bucket, k)
    assert np.max(np.abs(others)) < 1e-6 * own


def test_bucketed_sum_equals_parallel_quote_shift(eur_markets):
    m = eur_markets["monotone_convex"]
    quotes = m.instruments
    target = quotes["bond"].iloc[3]
    # an off-pillar bond: move the valuation to a curve built on a shifted set of quotes
    price = bond_pricer(target, m.settle)
    bucket = bucketed_dv01(price, m.rebuild, len(m.quotes))
    total = bucket.sum()
    assert total == pytest.approx(yield_risk(target, m.quotes[3], m.settle).dv01, rel=1e-6)


def test_parallel_dv01_is_linear_in_nominal(eur_markets):
    m = eur_markets["log_linear_df"]
    bond = m.instruments["bond"].iloc[2]
    one = parallel_dv01(bond_pricer(bond, m.settle), m.curve)
    ten = parallel_dv01(bond_pricer(bond, m.settle, nominal=10.0), m.curve)
    assert ten == pytest.approx(10 * one, rel=1e-12)


def test_parallel_shift_of_curve():
    curve = DiscountCurve([1, 5, 10], [0.02, 0.025, 0.03])
    up = curve.shifted(parallel(10))
    assert np.allclose(up.zero([1, 3, 7]), curve.zero([1, 3, 7]) + 1e-3, atol=1e-15)
