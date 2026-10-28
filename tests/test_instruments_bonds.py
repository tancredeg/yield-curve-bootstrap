"""Bond cash flows, accrued interest, ICMA yields and yield risk."""

from datetime import date

import numpy as np
import pytest

from ratescurve.bonds import (
    clean_price_on_curve,
    dirty_from_yield,
    dirty_price_on_curve,
    yield_from_dirty,
    yield_risk,
)
from ratescurve.conventions import add_business_days
from ratescurve.curve import DiscountCurve
from ratescurve.data import load_bond_static, load_bund_quotes, make_bond
from ratescurve.instruments import FixedRateBond, ParInstrument

BUND = FixedRateBond("DE000BU2Z056", 0.026, date(2035, 8, 15))
LONG_FIRST = FixedRateBond("DE000BU2Z064", 0.029, date(2036, 2, 15), 1, date(2026, 1, 9), date(2027, 2, 15))
SETTLE = date(2026, 10, 29)


def test_regular_cashflows():
    flows = BUND.cashflows(SETTLE)
    assert flows[0] == (date(2027, 8, 15), pytest.approx(2.6))
    assert flows[-1] == (date(2035, 8, 15), pytest.approx(102.6))
    assert len(flows) == 9


def test_coupon_on_settlement_date_belongs_to_seller():
    flows = BUND.cashflows(date(2027, 8, 15))
    assert flows[0][0] == date(2028, 8, 15)
    assert BUND.accrued(date(2027, 8, 15)) == 0.0


def test_long_first_coupon_amount_and_accrual():
    flows = LONG_FIRST.cashflows(SETTLE)
    assert flows[0][0] == date(2027, 2, 15)
    assert flows[0][1] == pytest.approx(2.9 * (1 + 37 / 365))
    # 37 days in the previous quasi period + 256 days in the current one
    assert LONG_FIRST.accrued(SETTLE) == pytest.approx(2.9 * (37 / 365 + 256 / 365))
    assert LONG_FIRST.coupons_between(SETTLE, date(2027, 2, 15)) == [
        (date(2027, 2, 15), pytest.approx(3.19397, abs=1e-5))
    ]


def test_first_coupon_validation():
    with pytest.raises(ValueError):
        FixedRateBond("X", 0.02, date(2030, 1, 1), 1, None, date(2027, 1, 1))
    with pytest.raises(ValueError):
        FixedRateBond("X", 0.02, date(2030, 1, 1), 1, date(2026, 6, 1), date(2027, 6, 1))
    with pytest.raises(ValueError):
        LONG_FIRST.accrued(date(2025, 12, 1))


def test_matured_bond_has_no_cashflows():
    assert BUND.cashflows(date(2036, 1, 1)) == []
    assert BUND.accrued(date(2036, 1, 1)) == 0.0


def test_par_instrument_flows():
    bill = ParInstrument(0.25, 0.04)
    assert bill.cashflow_times() == [(0.25, pytest.approx(1.01))]
    note = ParInstrument(2.0, 0.05)
    flows = note.cashflow_times()
    assert len(flows) == 4 and flows[-1] == (2.0, pytest.approx(1.025))
    with pytest.raises(ValueError):
        ParInstrument(1.3, 0.05).cashflow_times()


def test_bundesbank_accrued_and_yield_reproduced_exactly():
    """Every conventional quote of the sample: accrued = dirty - clean, ICMA yield = published yield."""
    quotes = load_bund_quotes()
    static = load_bond_static()
    quotes = quotes[~quotes["description"].astype(str).str.contains("I/L")]
    assert len(quotes) > 1400
    worst_ai = worst_y = 0.0
    for _, r in quotes.iterrows():
        bond = make_bond(r, static)
        settle = add_business_days(r["price_date"], 2)
        worst_ai = max(worst_ai, abs(bond.accrued(settle) - (r["dirty"] - r["clean"])))
        worst_y = max(worst_y, abs(yield_from_dirty(bond, r["dirty"], settle) * 1e4 - r["yield"] * 100))
    assert worst_ai < 1e-9  # accrued per 100 nominal
    assert worst_y < 5.01e-5  # bp: the published yield is rounded to 1e-6 percent


@pytest.mark.parametrize("ytm", [-0.005, 0.0, 0.0333, 0.08])
def test_yield_price_roundtrip(ytm):
    dirty = dirty_from_yield(LONG_FIRST, ytm, SETTLE)
    assert yield_from_dirty(LONG_FIRST, dirty, SETTLE) == pytest.approx(ytm, abs=1e-13)


def test_yield_risk_matches_finite_differences():
    y, h = 0.0331, 1e-6
    risk = yield_risk(LONG_FIRST, y, SETTLE)
    p0 = dirty_from_yield(LONG_FIRST, y, SETTLE)
    up, dn = dirty_from_yield(LONG_FIRST, y + h, SETTLE), dirty_from_yield(LONG_FIRST, y - h, SETTLE)
    assert risk.dirty == pytest.approx(p0, rel=1e-14)
    assert risk.duration_mod == pytest.approx(-(up - dn) / (2 * h) / p0, rel=1e-8)
    assert risk.convexity == pytest.approx((up - 2 * p0 + dn) / h**2 / p0, rel=1e-4)
    assert risk.duration_mac == pytest.approx(risk.duration_mod * (1 + y), rel=1e-14)
    assert risk.dv01 == pytest.approx(risk.duration_mod * risk.dirty * 1e-4, rel=1e-14)
    dy = 1e-3
    taylor = -risk.duration_mod * dy + 0.5 * risk.convexity * dy**2
    exact = dirty_from_yield(LONG_FIRST, y + dy, SETTLE) / p0 - 1
    assert abs(exact - taylor) < 1e-6  # third-order remainder


def test_zero_coupon_duration():
    zero = FixedRateBond("Z", 0.0, date(2031, 10, 29))
    y = 0.03
    risk = yield_risk(zero, y, SETTLE)
    assert risk.duration_mac == pytest.approx(5.0, abs=1e-12)
    assert risk.duration_mod == pytest.approx(5.0 / 1.03, abs=1e-12)


def test_curve_price_on_flat_curve():
    curve = DiscountCurve([0.5, 10, 30], [0.03, 0.03, 0.03], "log_linear_df")
    dirty = dirty_price_on_curve(BUND, curve, SETTLE)
    times = np.array([(d - SETTLE).days / 365 for d, _ in BUND.cashflows(SETTLE)])
    amounts = np.array([a for _, a in BUND.cashflows(SETTLE)])
    assert dirty == pytest.approx(float(np.sum(amounts * np.exp(-0.03 * times))), rel=1e-14)
    assert clean_price_on_curve(BUND, curve, SETTLE) == pytest.approx(dirty - BUND.accrued(SETTLE))
    with pytest.raises(ValueError):
        yield_from_dirty(BUND, -1.0, SETTLE)
    with pytest.raises(ValueError):
        dirty_from_yield(BUND, -2.0, SETTLE)
