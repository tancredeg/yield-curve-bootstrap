"""Market construction: out-of-sample residuals, curve domain, selection inputs."""

import numpy as np
import pytest

from ratescurve.bonds import dirty_price_on_curve, yield_from_dirty
from ratescurve.markets import out_of_sample


def test_out_of_sample_residuals(eur_markets, bund_quotes):
    m = eur_markets["monotone_convex"]
    oos = out_of_sample(m, bund_quotes)
    assert len(oos) > 30
    assert not set(oos["isin"]) & set(m.instruments["isin"])
    assert (oos["tau"] >= 0.5).all()
    # residual = market minus model ICMA yield, in bp, re-derived independently for one bond
    r = oos.iloc[len(oos) // 2]
    row = bund_quotes[bund_quotes["isin"] == r["isin"]].iloc[0]
    model = yield_from_dirty(row["bond"], dirty_price_on_curve(row["bond"], m.curve, row["settle"]), row["settle"])
    market = yield_from_dirty(row["bond"], float(row["dirty"]), row["settle"])
    assert r["residual_bp"] == pytest.approx((market - model) * 1e4, abs=1e-9)
    assert np.sqrt(np.mean(oos["residual_bp"] ** 2)) < 10.0


def test_out_of_sample_needs_a_bond_market(usd_markets, bund_quotes):
    with pytest.raises(ValueError):
        out_of_sample(usd_markets["monotone_convex"], bund_quotes)


def test_curves_cover_the_instruments(usd_markets, eur_markets):
    for m in usd_markets.values():
        assert m.curve.t_max == pytest.approx(30.0)
        assert len(m.labels) == len(m.quotes) == m.pillar_times.size
    for m in eur_markets.values():
        assert m.curve.t_max == pytest.approx(float(m.instruments["tau"].max()))
        assert m.settle == m.instruments["settle"].iloc[0]
