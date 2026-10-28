"""Market curves built from the frozen sample: US Treasury par curve and German Bund curve.

USD
    The Treasury par yield curve of one date (14 constant-maturity tenors,
    bond-equivalent basis) is calibrated exactly: one pillar per tenor, each
    par instrument reprices to 1.

EUR
    The German government curve of one date is calibrated exactly to a
    benchmark-style subset of conventional Bunds, Bobls and Schatz (Bundesbank
    dirty reference prices, T+2 settlement). Selection rule, fixed in advance:
    for each target tenor, the eligible bond with the nearest residual maturity
    (ties: larger issue volume), where eligible means non-zero coupon, residual
    maturity of at least 0.5 years, issue volume of at least EUR 10bn, and a
    maturity within 0.5 years of the target up to 10 years and within 2 years
    beyond. Green bonds and inflation-linked bonds are excluded. All other
    conventional bonds are out of sample and are repriced off the curve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .bonds import dirty_from_yield, dirty_price_on_curve, yield_from_dirty
from .calibration import Pricer, bond_pricer, calibrate, par_pricer
from .conventions import yf_act365f
from .curve import DiscountCurve
from .data import SAMPLE_DIR, bunds_on, load_ust_par, tenor_to_years
from .instruments import FixedRateBond, ParInstrument

EUR_TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30)
MIN_ISSUE_VOLUME_BN = 10.0


@dataclass
class CalibratedMarket:
    """A calibrated curve with the quotes it was built from.

    ``quotes`` are the instruments' quoted rates (decimal): ICMA yields for
    bonds, par yields for the Treasury curve. ``rebuild(i, bp)`` recalibrates
    with quote ``i`` moved by ``bp`` basis points (used for bucketed DV01).
    """

    curve: DiscountCurve
    labels: list[str]
    quotes: np.ndarray
    pillar_times: np.ndarray
    residuals: np.ndarray
    rebuild: Callable[[int, float], DiscountCurve] = field(repr=False)
    settle: date | None = None
    instruments: pd.DataFrame | None = field(default=None, repr=False)


def _continuous(rates: ArrayLike, frequency: int) -> np.ndarray:
    """Periodic rates (compounding ``frequency``) to continuously compounded rates."""
    return frequency * np.log1p(np.asarray(rates, dtype=float) / frequency)


def build_usd_curve(day: date, method: str, sample_dir: Path = SAMPLE_DIR) -> CalibratedMarket:
    """Exact calibration of the US Treasury par curve of ``day``."""
    ust = load_ust_par(sample_dir)
    row = ust.loc[pd.Timestamp(day)].dropna()
    labels = list(row.index)
    rates = row.to_numpy(dtype=float) / 100.0
    tenors = np.array([tenor_to_years(k) for k in labels])
    guess = _continuous(rates, 2)

    def build(quotes: np.ndarray, start: ArrayLike) -> tuple[DiscountCurve, np.ndarray]:
        pricers = [par_pricer(ParInstrument(t, q)) for t, q in zip(tenors, quotes)]
        res = calibrate(pricers, np.ones(len(pricers)), tenors, method, start, label="USD Treasury")
        return res.curve, res.residuals

    curve, residuals = build(rates, guess)

    def rebuild(i: int, bp: float) -> DiscountCurve:
        bumped = rates.copy()
        bumped[i] += bp * 1e-4
        return build(bumped, curve.zeros)[0]

    return CalibratedMarket(curve, labels, rates, tenors, residuals, rebuild)


def eligible_bunds(day: date, sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """Conventional (non-green, non-linker) Federal securities of ``day`` with residual maturity."""
    quotes = bunds_on(day, sample_dir)
    quotes["tau"] = [yf_act365f(s, m) for s, m in zip(quotes["settle"], quotes["maturity"])]
    return quotes


def select_calibration_bunds(
    quotes: pd.DataFrame, targets: Sequence[float] = EUR_TARGETS, min_volume: float = MIN_ISSUE_VOLUME_BN
) -> pd.DataFrame:
    """Apply the selection rule of the module docstring; returns rows sorted by maturity."""
    pool = quotes[(quotes["coupon"] > 0) & (quotes["tau"] >= 0.5) & (quotes["issue_volume_bn"] >= min_volume)]
    picked: list[str] = []
    for k in targets:
        window = 0.5 if k <= 10 else 2.0
        cand = pool[(pool["tau"] - k).abs() <= window].copy()
        if cand.empty:
            continue
        cand["dist"] = (cand["tau"] - k).abs()
        best = cand.sort_values(["dist", "issue_volume_bn"], ascending=[True, False]).iloc[0]
        if best["isin"] not in picked:
            picked.append(str(best["isin"]))
    sel = quotes[quotes["isin"].isin(picked)].sort_values("tau")
    return sel.reset_index(drop=True)


def build_eur_curve(
    day: date, method: str, sample_dir: Path = SAMPLE_DIR, targets: Sequence[float] = EUR_TARGETS
) -> CalibratedMarket:
    """Exact calibration of the Bund curve of ``day`` to the selected benchmark bonds."""
    quotes = eligible_bunds(day, sample_dir)
    sel = select_calibration_bunds(quotes, targets)
    settle: date = sel["settle"].iloc[0]
    bonds: list[FixedRateBond] = list(sel["bond"])
    dirty = sel["dirty"].to_numpy(dtype=float)
    ytm = np.array([yield_from_dirty(b, p, settle) for b, p in zip(bonds, dirty)])
    taus = sel["tau"].to_numpy(dtype=float)
    pricers: list[Pricer] = [bond_pricer(b, settle) for b in bonds]
    res = calibrate(pricers, dirty, taus, method, _continuous(ytm, 1), label="EUR Bund")

    def rebuild(i: int, bp: float) -> DiscountCurve:
        bumped = dirty.copy()
        bumped[i] = dirty_from_yield(bonds[i], ytm[i] + bp * 1e-4, settle)
        return calibrate(pricers, bumped, taus, method, res.curve.zeros, label="EUR Bund").curve

    return CalibratedMarket(
        res.curve, [str(d) for d in sel["description"]], ytm, taus, res.residuals, rebuild, settle, sel
    )


def out_of_sample(market: CalibratedMarket, quotes: pd.DataFrame) -> pd.DataFrame:
    """Yield residuals (market minus curve, bp) of the bonds not used in the calibration.

    Positive residual: the bond yields more than the curve implies (cheap).
    """
    if market.instruments is None or market.settle is None:
        raise ValueError("out-of-sample residuals need a bond-calibrated market")
    used = set(market.instruments["isin"])
    rows = []
    for _, r in quotes.iterrows():
        if r["isin"] in used or r["tau"] < 0.5:
            continue
        model_dirty = dirty_price_on_curve(r["bond"], market.curve, r["settle"])
        model_y = yield_from_dirty(r["bond"], model_dirty, r["settle"])
        market_y = yield_from_dirty(r["bond"], float(r["dirty"]), r["settle"])
        rows.append(
            {
                "isin": r["isin"],
                "description": r["description"],
                "tau": r["tau"],
                "coupon": r["coupon"],
                "market_yield": market_y,
                "model_yield": model_y,
                "residual_bp": (market_y - model_y) * 1e4,
            }
        )
    return pd.DataFrame(rows)
