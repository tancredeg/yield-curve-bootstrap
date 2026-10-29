"""Risk identities: key-rate sum, additivity of bucketed DV01, bucketed DV01 of calibration instruments."""

import numpy as np
import pytest

from ratescurve.bonds import yield_risk
from ratescurve.calibration import bond_pricer, par_pricer
from ratescurve.curve import DiscountCurve, parallel
from ratescurve.instruments import ParInstrument
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


OFF_PILLAR = "DE000BU2Z072"  # DBR 3 08/15/36: between two calibration Bunds, not one of them


def _off_pillar_pricer(market, quotes):
    row = quotes[quotes["isin"] == OFF_PILLAR].iloc[0]
    assert OFF_PILLAR not in set(market.instruments["isin"])
    return bond_pricer(row["bond"], market.settle)


@pytest.mark.parametrize("method", ["linear_zero", "log_linear_df", "monotone_convex", "natural_cubic_zero"])
def test_bucketed_dv01_of_off_pillar_bond_adds_up_to_joint_quote_shift(eur_markets, bund_quotes, method):
    """Small quote bumps are additive: the bucket sum equals a joint shift of every quote (first order)."""
    m = eur_markets[method]
    price = _off_pillar_pricer(m, bund_quotes)
    h = 0.01
    bucket = bucketed_dv01(price, m.rebuild, len(m.quotes), bp=h)
    joint = -(price(m.rebuild_all(h)) - price(m.rebuild_all(-h))) / (2 * h)
    assert joint > 0
    assert bucket.sum() == pytest.approx(joint, rel=1e-5)


def test_monotone_convex_is_only_piecewise_smooth_in_the_quotes(eur_markets, bund_quotes):
    """With 1 bp bumps the Hagan-West shape regions switch and monotone convex buckets stop adding up.

    This is why the published bucketed DV01s use 0.01 bp bumps; the linear schemes are additive at 1 bp.
    """
    gaps = {}
    for method in ("monotone_convex", "log_linear_df", "linear_zero"):
        m = eur_markets[method]
        price = _off_pillar_pricer(m, bund_quotes)
        bucket = bucketed_dv01(price, m.rebuild, len(m.quotes), bp=1.0)
        joint = -(price(m.rebuild_all(1.0)) - price(m.rebuild_all(-1.0))) / 2.0
        gaps[method] = abs(bucket.sum() / joint - 1.0)
    assert gaps["monotone_convex"] > 1e-2
    assert gaps["log_linear_df"] < 1e-5 and gaps["linear_zero"] < 1e-5


def test_usd_par_instrument_bucket_is_its_annuity(usd_markets):
    """A par instrument's only quote sensitivity is to its own par rate, equal to its annuity per bp."""
    m = usd_markets["log_linear_df"]
    k = int(np.argmin(np.abs(m.pillar_times - 10.0)))
    instrument = ParInstrument(float(m.pillar_times[k]), float(m.quotes[k]))
    price = par_pricer(instrument)
    bucket = bucketed_dv01(price, m.rebuild, len(m.quotes), bp=0.01)
    times = 0.5 * np.arange(1, 21)
    annuity = 0.5 * float(np.sum(m.curve.df(times)))
    assert bucket[k] == pytest.approx(annuity * 1e-4, rel=1e-5)
    assert np.max(np.abs(np.delete(bucket, k))) < 1e-6 * bucket[k]
    joint = -(price(m.rebuild_all(0.01)) - price(m.rebuild_all(-0.01))) / 0.02
    assert joint == pytest.approx(bucket.sum(), rel=1e-6)


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
