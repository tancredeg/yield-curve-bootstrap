"""Calibration: every instrument reprices under every scheme; independent re-derivation of quotes."""

import numpy as np
import pytest

from ratescurve.bonds import dirty_price_on_curve, yield_from_dirty
from ratescurve.calibration import CalibrationError, calibrate, par_pricer
from ratescurve.instruments import ParInstrument
from ratescurve.markets import EUR_TARGETS, MIN_ISSUE_VOLUME_BN

from .conftest import METHODS


def _par_rate(curve, tenor: float, frequency: int = 2) -> float:
    """Par rate from discount factors (independent of the pricing code path)."""
    step = 1.0 / frequency
    if tenor <= step + 1e-12:
        return float((1.0 / curve.df(tenor) - 1.0) / tenor)
    times = step * np.arange(1, round(tenor / step) + 1)
    return float((1.0 - curve.df(tenor)) / (step * np.sum(curve.df(times))))


@pytest.mark.parametrize("method", METHODS)
def test_usd_par_instruments_reprice(usd_markets, method):
    m = usd_markets[method]
    assert np.max(np.abs(m.residuals)) < 1e-12
    for tenor, quote in zip(m.pillar_times, m.quotes):
        assert abs(_par_rate(m.curve, tenor) - quote) * 1e4 < 1e-8  # bp


@pytest.mark.parametrize("method", METHODS)
def test_eur_bonds_reprice(eur_markets, method):
    m = eur_markets[method]
    assert np.max(np.abs(m.residuals)) < 1e-9  # per 100 nominal
    for (_, row), quote in zip(m.instruments.iterrows(), m.quotes):
        model = dirty_price_on_curve(row["bond"], m.curve, m.settle)
        assert abs(yield_from_dirty(row["bond"], model, m.settle) - quote) * 1e4 < 1e-8


def test_eur_selection_rule(eur_markets, bund_quotes):
    sel = eur_markets["monotone_convex"].instruments
    assert len(sel) == len(EUR_TARGETS)
    assert (sel["issue_volume_bn"] >= MIN_ISSUE_VOLUME_BN).all()
    assert (sel["coupon"] > 0).all()
    assert sel["tau"].is_monotonic_increasing
    assert not sel["description"].str.contains(r"\bG\b|I/L", regex=True).any()
    assert set(sel["isin"]) <= set(bund_quotes["isin"])


def test_calibration_does_not_depend_on_scheme_at_pillars_for_zero_coupon_inputs():
    """With zero-coupon instruments, every scheme returns the same pillar zeros (only pillars matter)."""
    tenors = [0.25, 0.5, 1.0]
    instruments = [ParInstrument(t, r) for t, r in zip(tenors, [0.04, 0.041, 0.043])]
    pricers = [par_pricer(i) for i in instruments[:2]]
    zeros = [
        calibrate(pricers, [1, 1], tenors[:2], m, [0.04, 0.04]).curve.zeros
        for m in ("linear_zero", "log_linear_df", "monotone_convex")
    ]
    assert np.allclose(zeros[0], zeros[1], atol=1e-15) and np.allclose(zeros[0], zeros[2], atol=1e-15)


def test_calibration_errors():
    pricers = [par_pricer(ParInstrument(1.0, 0.04))]
    with pytest.raises(ValueError):
        calibrate(pricers, [1.0, 1.0], [1.0], "log_linear_df", [0.04])
    with pytest.raises(CalibrationError):  # a negative price needs a negative discount factor
        calibrate(pricers, [-1.0], [1.0], "log_linear_df", [0.04])
