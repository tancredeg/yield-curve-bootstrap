"""Shared fixtures: as-of date and calibrated markets (built once per session)."""

from __future__ import annotations

from datetime import date

import pytest

from ratescurve.markets import CalibratedMarket, build_eur_curve, build_usd_curve, eligible_bunds

ASOF = date(2026, 10, 27)
METHODS = ("linear_zero", "log_linear_df", "monotone_convex", "natural_cubic_zero")


@pytest.fixture(scope="session")
def asof() -> date:
    return ASOF


@pytest.fixture(scope="session")
def eur_markets() -> dict[str, CalibratedMarket]:
    return {m: build_eur_curve(ASOF, m) for m in METHODS}


@pytest.fixture(scope="session")
def usd_markets() -> dict[str, CalibratedMarket]:
    return {m: build_usd_curve(ASOF, m) for m in METHODS}


@pytest.fixture(scope="session")
def bund_quotes():
    return eligible_bunds(ASOF)
