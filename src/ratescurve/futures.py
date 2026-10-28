"""Euro government bond futures (Eurex): conversion factors, basis, carry, implied repo, CTD.

Contract facts (Eurex contract specifications): physical delivery on the 10th
calendar day of the contract month (next exchange day if not an exchange day);
last trading day two exchange days before delivery; notional coupon 6 % for
Euro-Schatz/Bobl/Bund, 4 % for Euro-Buxl; contract size EUR 100,000.

Conversion factor (Eurex, "Conversion factor calculation for EUR-denominated
bonds")::

    CF = 1/(1+n/100)^f * [ c/100 * di/act2 + c/n * (1 + n/100 - 1/(1+n/100)^N) + 1/(1+n/100)^N ]
         - c/100 * (di/act2 - de/act1)

with ``n`` the notional coupon, ``c`` the coupon (percent), ``de = NCD1y - DD``,
``di = NCD1y - LCD``, ``f = 1 + de/act1``, ``N`` the integer years from the
next coupon date (NCD) to maturity, ``LCD`` the last coupon date before
delivery or the interest commencement date if no coupon has been paid yet.

Cash-and-carry (per 100 nominal, simple money-market interest ACT/360)::

    invoice        = F * CF + AI(delivery)
    gross basis    = clean - F * CF
    forward clean  = dirty * (1 + r d/360) - sum_k C_k (1 + r d_k/360) - AI(delivery)
    carry          = clean - forward clean
    net basis      = gross basis - carry = forward clean - F * CF
    implied repo   = (invoice + sum_k C_k - dirty) / (dirty d/360 - sum_k C_k d_k/360)

where ``d`` is the number of days from spot settlement to delivery and ``C_k``
are coupons paid in between (``d_k`` days before delivery). The cheapest to
deliver (CTD) bond maximises the implied repo rate. The net basis bundles the
value of the seller's delivery options and any repo specialness; neither is
modelled here.

References: Eurex, "Euro-Bund Futures" contract specifications and conversion
factor methodology; G. Burghardt et al., *The Treasury Bond Basis*, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .bonds import dirty_from_yield, yield_from_dirty
from .conventions import add_business_days, add_months, following, yf_act360
from .instruments import FixedRateBond

NOTIONAL_COUPON = {"FGBS": 6.0, "FGBM": 6.0, "FGBL": 6.0, "FGBX": 4.0}
CONTRACT_SIZE = 100_000.0


def delivery_date(year: int, month: int) -> date:
    """Delivery day: 10th calendar day of the contract month, next business day if needed."""
    return following(date(year, month, 10))


def last_trading_day(year: int, month: int) -> date:
    """Two exchange days before the delivery day."""
    return add_business_days(delivery_date(year, month), -2)


def conversion_factor(bond: FixedRateBond, delivery: date, notional_coupon: float = 6.0) -> float:
    """Eurex conversion factor (unrounded) of an annual-coupon EUR bond."""
    if bond.frequency != 1:
        raise ValueError("Eurex EUR formula applies to annual-coupon bonds")
    c = bond.coupon * 100.0
    ncd = date(delivery.year, bond.maturity.month, bond.maturity.day)
    if ncd <= delivery:
        ncd = add_months(ncd, 12)
    ncd1, ncd2 = add_months(ncd, -12), add_months(ncd, -24)
    first = bond.first_coupon_date()
    if first is not None and first > delivery:
        if bond.interest_start is None:
            raise ValueError("interest_start required")
        lcd = bond.interest_start
    else:
        lcd = ncd1
    de = (ncd1 - delivery).days
    act1 = (ncd - ncd1).days if de < 0 else (ncd1 - ncd2).days
    di = (ncd1 - lcd).days
    act2 = (ncd - ncd1).days if di < 0 else (ncd1 - ncd2).days
    f = 1.0 + de / act1
    n_years = bond.maturity.year - ncd.year
    v = 1.0 / (1.0 + notional_coupon / 100.0)
    bracket = c / 100.0 * di / act2 + c / notional_coupon * (1.0 + notional_coupon / 100.0 - v**n_years) + v**n_years
    return float(v**f * bracket - c / 100.0 * (di / act2 - de / act1))


@dataclass(frozen=True)
class Basis:
    """Cash-and-carry analysis of one deliverable bond (per 100 nominal)."""

    isin: str
    cf: float
    clean: float
    dirty: float
    accrued_settle: float
    accrued_delivery: float
    coupons: float
    days: int
    gross_basis: float
    forward_clean: float
    carry: float
    net_basis: float
    implied_repo: float


def basis(
    bond: FixedRateBond, clean: float, settle: date, delivery: date, futures_price: float, cf: float, repo: float
) -> Basis:
    """Gross basis, carry, net basis and implied repo of ``bond`` against the futures."""
    ai_s = bond.accrued(settle)
    ai_d = bond.accrued(delivery)
    dirty = clean + ai_s
    d = (delivery - settle).days
    flows = bond.coupons_between(settle, delivery)
    coupons = sum(c for _, c in flows)
    reinvest = sum(c * yf_act360(dc, delivery) for dc, c in flows)
    invoice = futures_price * cf + ai_d
    fwd_dirty = dirty * (1.0 + repo * d / 360.0) - sum(c * (1.0 + repo * yf_act360(dc, delivery)) for dc, c in flows)
    fwd_clean = fwd_dirty - ai_d
    irr = (invoice + coupons - dirty) / (dirty * d / 360.0 - reinvest)
    gross = clean - futures_price * cf
    carry = clean - fwd_clean
    return Basis(bond.isin, cf, clean, dirty, ai_s, ai_d, coupons, d, gross, fwd_clean, carry, gross - carry, irr)


def forward_clean_from_yield(bond: FixedRateBond, ytm: float, settle: date, delivery: date, repo: float) -> float:
    """Forward clean price at delivery when the spot ICMA yield is ``ytm`` (repo carry, ACT/360)."""
    dirty = dirty_from_yield(bond, ytm, settle)
    d = (delivery - settle).days
    flows = bond.coupons_between(settle, delivery)
    fwd_dirty = dirty * (1.0 + repo * d / 360.0) - sum(c * (1.0 + repo * yf_act360(dc, delivery)) for dc, c in flows)
    return fwd_dirty - bond.accrued(delivery)


@dataclass(frozen=True)
class SwitchGrid:
    """CTD and net basis of every deliverable under parallel yield shifts."""

    shifts_bp: np.ndarray
    futures: np.ndarray
    net_basis: np.ndarray  # (n_shifts, n_bonds), forward clean - F * CF
    ctd_index: np.ndarray


def ctd_switch(
    bonds: list[FixedRateBond],
    cleans: list[float],
    cfs: list[float],
    settle: date,
    delivery: date,
    repo: float,
    shifts_bp: np.ndarray,
) -> SwitchGrid:
    """Theoretical futures ``min_i fwd_i / CF_i`` and CTD as bond yields shift in parallel.

    Delivery options are ignored: the theoretical futures price is the forward
    price of the cheapest bond divided by its conversion factor.
    """
    ytms = [yield_from_dirty(b, c + b.accrued(settle), settle) for b, c in zip(bonds, cleans)]
    fwd = np.array(
        [
            [forward_clean_from_yield(b, y + s * 1e-4, settle, delivery, repo) for b, y in zip(bonds, ytms)]
            for s in shifts_bp
        ]
    )
    ratio = fwd / np.asarray(cfs)
    fut = ratio.min(axis=1)
    nb = fwd - fut[:, None] * np.asarray(cfs)
    return SwitchGrid(np.asarray(shifts_bp, dtype=float), fut, nb, ratio.argmin(axis=1))


def futures_dv01(
    bonds: list[FixedRateBond],
    cleans: list[float],
    cfs: list[float],
    settle: date,
    delivery: date,
    repo: float,
    bp: float = 1.0,
) -> float:
    """Futures price change (per 100) for a 1 bp parallel fall in deliverable yields (CTD-driven)."""
    grid = ctd_switch(bonds, cleans, cfs, settle, delivery, repo, np.array([-bp, bp]))
    return float((grid.futures[0] - grid.futures[1]) / (2.0 * bp))
