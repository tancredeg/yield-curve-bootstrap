"""Euro-Bund futures: conversion factors versus Eurex, calendar, basis identities, CTD switch."""

from datetime import date

import numpy as np
import pytest

from ratescurve.data import load_bond_static, load_deliverables, load_fgbl
from ratescurve.futures import (
    NOTIONAL_COUPON,
    basis,
    conversion_factor,
    ctd_switch,
    delivery_date,
    futures_dv01,
    last_trading_day,
)
from ratescurve.instruments import FixedRateBond

SETTLE = date(2026, 10, 29)
DELIVERY = date(2026, 12, 10)


def _bond(row, static):
    isin = row["isin"]
    start = static.loc[isin, "interest_start"] if isin in static.index else None
    first = static.loc[isin, "first_coupon"] if isin in static.index else None
    return FixedRateBond(isin, row["coupon"] / 100, row["maturity"], 1, start, first)


def test_conversion_factors_match_eurex_for_every_deliverable():
    dl = load_deliverables()
    static = load_bond_static()
    assert len(dl) >= 40
    for _, r in dl.iterrows():
        y, m = int(r["contract"][:4]), int(r["contract"][4:6])
        cf = conversion_factor(_bond(r, static), delivery_date(y, m), NOTIONAL_COUPON[r["product"]])
        assert round(cf, 6) == pytest.approx(r["conversion_factor"], abs=1e-12), (r["product"], r["isin"])


def test_contract_codes_are_last_trading_days():
    for code in load_deliverables()["contract"].unique():
        y, m = int(code[:4]), int(code[4:6])
        assert last_trading_day(y, m).strftime("%Y%m%d") == code
    assert delivery_date(2026, 12) == DELIVERY


def test_zero_coupon_conversion_factor_closed_form():
    bond = FixedRateBond("Z", 0.0, date(2031, 8, 15))
    f = 1 - (date(2026, 12, 10) - date(2026, 8, 15)).days / 365
    assert conversion_factor(bond, DELIVERY) == pytest.approx(1.06 ** -(f + 4), abs=1e-15)
    with pytest.raises(ValueError):
        conversion_factor(FixedRateBond("S", 0.02, date(2035, 1, 1), 2), DELIVERY)


def test_basis_identities():
    bond = FixedRateBond("DE000BU2Z056", 0.026, date(2035, 8, 15))
    cf = conversion_factor(bond, DELIVERY)
    b = basis(bond, 94.62, SETTLE, DELIVERY, 122.09, cf, repo=0.0244)
    assert b.days == 42
    assert b.net_basis == pytest.approx(b.forward_clean - 122.09 * cf, abs=1e-12)
    assert b.gross_basis - b.carry == pytest.approx(b.net_basis, abs=1e-12)
    # financing at the implied repo leaves no net basis
    at_irr = basis(bond, 94.62, SETTLE, DELIVERY, 122.09, cf, repo=b.implied_repo)
    assert at_irr.net_basis == pytest.approx(0.0, abs=1e-10)


def test_basis_with_intermediate_coupon():
    bond = FixedRateBond("X", 0.03, date(2036, 11, 20))
    cf = conversion_factor(bond, DELIVERY)
    b = basis(bond, 97.0, SETTLE, DELIVERY, 122.0, cf, repo=0.024)
    assert b.coupons == pytest.approx(3.0)
    at_irr = basis(bond, 97.0, SETTLE, DELIVERY, 122.0, cf, repo=b.implied_repo)
    assert at_irr.net_basis == pytest.approx(0.0, abs=1e-10)


def test_ctd_switch_follows_duration_rule():
    """Yields below the 6 % notional favour the short-duration bond, far above it the long one."""
    static = load_bond_static()
    dl = load_deliverables()
    dec = dl[(dl["product"] == "FGBL") & (dl["contract"] == "20261208")]
    bonds = [_bond(r, static) for _, r in dec.iterrows()]
    cfs = list(dec["conversion_factor"])
    quotes = {"DE000BU2Z056": 94.62, "DE000BU2Z064": 96.58, "DE000BU2Z072": 97.09}  # 27 Oct 2026
    cleans = [quotes[b.isin] for b in bonds]
    grid = ctd_switch(bonds, cleans, cfs, SETTLE, DELIVERY, 0.0244, np.array([-100.0, 0.0, 600.0]))
    mats = [b.maturity for b in bonds]
    assert mats[grid.ctd_index[0]] == min(mats)
    assert mats[grid.ctd_index[-1]] == max(mats)
    assert np.all(grid.net_basis >= -1e-12)
    assert futures_dv01(bonds, cleans, cfs, SETTLE, DELIVERY, 0.0244) > 0


def test_fgbl_sample_is_complete():
    fgbl = load_fgbl()
    front = fgbl[fgbl["contract"] == "20261208"]
    assert front["business_date"].is_unique and len(front) >= 19
    assert front["settlement"].between(110, 135).all()
    assert (front["open_interest"] > 1_000_000).all()
