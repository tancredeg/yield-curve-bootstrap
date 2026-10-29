"""Regenerate every figure, table and number of the README from ``data/sample/``.

Outputs (regenerated, never edited by hand):

    docs/figures/*.png        four figures
    docs/results.md           detailed result tables, each with its interpretation
    docs/key_results.json     every number quoted in README.md and docs/results.md
    README.md                 the block between the ``<!-- results:key -->`` markers

Interpretive sentences are assembled from computed values. Statements that
depend on the data (a ranking, "the same CTD every day", ...) go through
:func:`ensure`, so the script fails rather than print a claim the data no
longer support.

``--check`` recomputes everything, writes nothing, and exits with status 1 if
``docs/key_results.json``, ``docs/results.md`` or the README block differ from
the committed files (the CI runs it to prove that the published numbers come
from this script).

Usage::

    python scripts/run_results.py            # regenerate
    python scripts/run_results.py --check    # verify, no writes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ratescurve.bonds import dirty_price_on_curve, yield_from_dirty, yield_risk
from ratescurve.calibration import bond_pricer
from ratescurve.conventions import add_business_days
from ratescurve.curve import DiscountCurve, Discounting, parallel
from ratescurve.data import (
    ASOF,
    bunds_on,
    load_bond_static,
    load_bund_quotes,
    load_deliverables,
    load_ecb_spot,
    load_estr,
    load_fgbl,
    load_ust_par,
    make_bond,
)
from ratescurve.futures import (
    CONTRACT_SIZE,
    NOTIONAL_COUPON,
    basis,
    conversion_factor,
    ctd_switch,
    delivery_date,
    futures_dv01,
)
from ratescurve.instruments import FixedRateBond
from ratescurve.markets import CalibratedMarket, build_eur_curve, build_usd_curve, eligible_bunds, out_of_sample
from ratescurve.pca import loading_similarity, principal_components
from ratescurve.risk import bucketed_dv01, key_rate_dv01, parallel_dv01
from ratescurve.scenarios import SCENARIOS, scenario_shift

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
KEY_JSON = DOCS / "key_results.json"
RESULTS_MD = DOCS / "results.md"
README = ROOT / "README.md"

METHODS = ("monotone_convex", "log_linear_df", "linear_zero", "natural_cubic_zero")
LABEL = {
    "monotone_convex": "Monotone convex (Hagan-West)",
    "log_linear_df": "Log-linear on discount factors",
    "linear_zero": "Linear on zero rates",
    "natural_cubic_zero": "Natural cubic spline on zeros",
}
FWD_WINDOWS = ((5, 5), (10, 10), (15, 5))  # (start, length) in years
KEYS = (1, 2, 3, 5, 7, 10, 15, 20, 30)
BULLET = "DE000BU2Z072"  # DBR 3 08/15/36, newest 10-year Bund (out of sample)
BARBELL = ("DE000BU22148", "DE000BU2D012")  # BKO 2.7 09/13/28 and DBR 2.9 08/15/56
NOTIONAL = 100e6
BUCKET_BP = 0.01  # quote bump for bucketed DV01 (see the 1 bp linearity check in R2)
FGBL_DEC26 = "20261208"
FUTURES_MOVE_CENTS = 10.0
USD_PCA_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# Chart tokens (light surface): categorical slots in a fixed order, recessive chrome.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DASH: tuple[Any, ...] = ("-", (0, (6, 2)), (0, (2, 1.5)), (0, (8, 2, 2, 2)))  # second encoding for print and CVD

SaveFn = Callable[[Any, str], None]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "lines.solid_capstyle": "round",
        }
    )


def write_png(fig: Any, name: str) -> None:
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def discard_png(fig: Any, name: str) -> None:
    plt.close(fig)


def r(x: float, nd: int) -> float:
    """Round for publication; ``-0.0`` becomes ``0.0``."""
    return float(round(float(x), nd)) + 0.0


def below(x: float, floor: float) -> float:
    """Smallest power of ten at or above ``x``, never below ``floor``.

    Errors at machine-precision level change with the CPU and BLAS kernel; publishing a bound keeps
    the published outputs identical on every machine while still proving the accuracy claimed.
    """
    bound = floor
    while bound < abs(x):
        bound *= 10.0
    return float(f"{bound:.0e}")


def ensure(condition: bool, claim: str) -> None:
    """Fail loudly instead of publishing a claim the data no longer support."""
    if not condition:
        raise RuntimeError(f"Data no longer support the claim: {claim}")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ── Validation: calibration and external reference values ───────────────────


def _par_rate(curve: DiscountCurve, tenor: float) -> float:
    if tenor <= 0.5 + 1e-12:
        return float((1.0 / curve.df(tenor) - 1.0) / tenor)
    times = 0.5 * np.arange(1, round(tenor / 0.5) + 1)
    return float((1.0 - curve.df(tenor)) / (0.5 * np.sum(curve.df(times))))


def repricing_error_bp(ccy: str, m: CalibratedMarket) -> float:
    """Largest |model - market| quoted rate of the calibration instruments, in bp."""
    if ccy == "USD":
        return max(abs(_par_rate(m.curve, t) - q) * 1e4 for t, q in zip(m.pillar_times, m.quotes))
    assert m.instruments is not None and m.settle is not None
    return max(
        abs(yield_from_dirty(b, dirty_price_on_curve(b, m.curve, m.settle), m.settle) - q) * 1e4
        for b, q in zip(m.instruments["bond"], m.quotes)
    )


def validation() -> dict[str, Any]:
    """Reproduction of published values: Bundesbank accrued and yields, Eurex conversion factors."""
    quotes = load_bund_quotes()
    static = load_bond_static()
    conv = quotes[~quotes["description"].astype(str).str.contains("I/L")]
    worst_ai = worst_y = 0.0
    for _, row in conv.iterrows():
        bond = make_bond(row, static)
        settle = add_business_days(row["price_date"], 2)
        worst_ai = max(worst_ai, abs(bond.accrued(settle) - (row["dirty"] - row["clean"])))
        worst_y = max(worst_y, abs(yield_from_dirty(bond, row["dirty"], settle) * 1e4 - row["yield"] * 100))
    dl = load_deliverables()
    matches = 0
    for _, row in dl.iterrows():
        isin = str(row["isin"])
        start = static.loc[isin, "interest_start"] if isin in static.index else None
        first = static.loc[isin, "first_coupon"] if isin in static.index else None
        bond = FixedRateBond(isin, float(row["coupon"]) / 100, row["maturity"], 1, start, first)
        y, mth = int(row["contract"][:4]), int(row["contract"][4:6])
        cf = conversion_factor(bond, delivery_date(y, mth), NOTIONAL_COUPON[str(row["product"])])
        matches += int(abs(round(cf, 6) - float(row["conversion_factor"])) < 5e-13)
    return {
        "bundesbank_quotes": len(conv),
        "bundesbank_days": int(conv["price_date"].nunique()),
        "bundesbank_securities": int(conv["isin"].nunique()),
        "max_accrued_error_below_per_100": below(worst_ai, 1e-12),
        "max_yield_error_below_bp": below(worst_y, 1e-4),
        "eurex_cf_total": len(dl),
        "eurex_cf_matched": matches,
        "eurex_products": sorted(str(p) for p in dl["product"].unique()),
    }


# ── R1: instantaneous forwards by interpolation scheme ──────────────────────


def r1(
    usd: dict[str, CalibratedMarket], eur: dict[str, CalibratedMarket], quotes: pd.DataFrame, save: SaveFn
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ccy, mk in (("USD", usd), ("EUR", eur)):
        t_max = min(m.curve.t_max for m in mk.values())
        grid = np.linspace(1 / 365, t_max, 6000)
        zeros = np.array([mk[m].curve.zero(grid) for m in METHODS])
        fwds = np.array([mk[m].curve.forward(grid) for m in METHODS])
        rows: dict[str, dict[str, Any]] = {}
        for m in METHODS:
            c = mk[m].curve
            pillars = c.times[:-1]
            jumps = np.abs(c.forward(pillars + 1e-9) - c.forward(pillars - 1e-9)) * 1e4
            f = c.forward(grid)
            row: dict[str, Any] = {
                "repricing_error_below_bp": below(repricing_error_bp(ccy, mk[m]), 1e-10),
                "forward_min_pct": r(f.min() * 100, 3),
                "forward_max_pct": r(f.max() * 100, 3),
                "max_forward_jump_bp": r(jumps.max(), 1),
            }
            for s, length in FWD_WINDOWS:
                row[f"fwd_{s}y{length}y_pct"] = r(c.discrete_forward(s, s + length) * 100, 3)
            if ccy == "EUR":
                oos = out_of_sample(mk[m], quotes)
                row["oos_rms_bp"] = r(np.sqrt(np.mean(oos["residual_bp"] ** 2)), 2)
                row["oos_n"] = len(oos)
            rows[m] = row
        spreads = {
            f"{s}y{length}y": r(
                (
                    max(v[f"fwd_{s}y{length}y_pct"] for v in rows.values())
                    - min(v[f"fwd_{s}y{length}y_pct"] for v in rows.values())
                )
                * 100,
                1,
            )
            for s, length in FWD_WINDOWS
        }
        widest = max(spreads, key=lambda k: spreads[k])
        out[ccy] = {
            "instruments": len(mk["monotone_convex"].quotes),
            "t_max": r(t_max, 2),
            "max_zero_spread_bp": r((zeros.max(0) - zeros.min(0)).max() * 1e4, 1),
            "max_forward_spread_bp": r((fwds.max(0) - fwds.min(0)).max() * 1e4, 0),
            "forward_window_spread_bp": spreads,
            "widest_window": widest,
            "methods": rows,
        }

    smooth = [m for m in METHODS if all(out[c]["methods"][m]["max_forward_jump_bp"] < 0.05 for c in out)]
    out["continuous_forward_schemes"] = smooth
    out["max_repricing_error_below_bp"] = max(
        out[c]["methods"][m]["repricing_error_below_bp"] for c in ("USD", "EUR") for m in METHODS
    )
    oos = {m: out["EUR"]["methods"][m]["oos_rms_bp"] for m in METHODS}
    out["best_oos"] = min(oos, key=lambda k: oos[k])

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
    for ax, ccy, mk, title in (
        (axes[0], "USD", usd, "US Treasury par curve, 27 Oct 2026: instantaneous forward"),
        (axes[1], "EUR", eur, "German government (Bund) curve, 27 Oct 2026: instantaneous forward"),
    ):
        t_max = min(m.curve.t_max for m in mk.values())
        grid = np.linspace(1 / 365, t_max, 3000)
        for i, m in enumerate(METHODS):
            ax.plot(grid, mk[m].curve.forward(grid) * 100, color=SLOTS[i], lw=1.6, ls=DASH[i], label=LABEL[m])
        piv = mk["monotone_convex"].pillar_times
        lo = ax.get_ylim()[0]
        ax.plot(piv, np.full(piv.size, lo), ls="none", marker="|", ms=9, color=MUTED, clip_on=False)
        ax.set_ylim(bottom=lo)
        ax.set_title(title)
        ax.set_ylabel("Forward rate (%, cont. comp.)")
        s = out[ccy]
        ax.text(
            0.99,
            0.04,
            f"Zero curves agree within {s['max_zero_spread_bp']:.1f} bp; forwards differ by up to "
            f"{s['max_forward_spread_bp']:.0f} bp. Ticks: calibration pillars.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=INK2,
            fontsize=8,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].set_xlabel("Maturity (years)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    save(fig, "r1_forwards_by_interpolation.png")
    return out


# ── R2: DV01 and key-rate DV01 of Bund portfolios ───────────────────────────


@dataclass
class Position:
    isin: str
    description: str
    nominal: float
    spread: float  # constant zero-rate spread (decimal) that reprices the market dirty price


def _row(quotes: pd.DataFrame, isin: str) -> pd.Series:
    return quotes[quotes["isin"] == isin].iloc[0]


def _with_spread(curve: Discounting, s: float) -> Discounting:
    return curve.shifted(parallel(s * 1e4)) if s else curve  # type: ignore[attr-defined]


def _spread(row: pd.Series, curve: Discounting) -> float:
    pricer = bond_pricer(row["bond"], row["settle"])
    return float(brentq(lambda s: pricer(_with_spread(curve, s)) - float(row["dirty"]), -0.05, 0.05, xtol=1e-14))


def portfolio_value(positions: list[Position], quotes: pd.DataFrame) -> Callable[[Discounting], float]:
    """Mark-to-market (EUR) of bond positions, each at its own constant spread to the curve."""
    legs = [(bond_pricer(_row(quotes, p.isin)["bond"], _row(quotes, p.isin)["settle"]), p) for p in positions]

    def value(curve: Discounting) -> float:
        return sum(pr(_with_spread(curve, p.spread)) * p.nominal / 100.0 for pr, p in legs)

    return value


def r2(eur: dict[str, CalibratedMarket], quotes: pd.DataFrame, save: SaveFn) -> dict[str, Any]:
    curve = eur["monotone_convex"].curve

    def pos(isin: str, nominal: float, c: Discounting = curve) -> Position:
        row = _row(quotes, isin)
        return Position(isin, str(row["description"]), nominal, _spread(row, c))

    bullet = [pos(BULLET, NOTIONAL)]
    v_b = portfolio_value(bullet, quotes)
    mv_b, dv_b = v_b(curve), parallel_dv01(v_b, curve)
    unit = [portfolio_value([pos(i, 1.0)], quotes) for i in BARBELL]
    a = np.array([[u(curve) for u in unit], [parallel_dv01(u, curve) for u in unit]])
    n_short, n_long = np.linalg.solve(a, np.array([mv_b, dv_b]))
    barbell = [pos(BARBELL[0], n_short), pos(BARBELL[1], n_long)]
    v_bb = portfolio_value(barbell, quotes)

    def stats(value: Callable[[Discounting], float], positions: list[Position]) -> dict[str, Any]:
        pv = value(curve)
        dv = parallel_dv01(value, curve)
        krd = key_rate_dv01(value, curve, KEYS)
        h = 10.0
        conv = (value(curve.shifted(parallel(h))) + value(curve.shifted(parallel(-h))) - 2 * pv) / (
            pv * (h * 1e-4) ** 2
        )
        return {
            "positions": [
                {"isin": p.isin, "description": p.description, "nominal_mn": r(p.nominal / 1e6, 2)} for p in positions
            ],
            "market_value_mn": r(pv / 1e6, 3),
            "dv01_k": r(dv / 1e3, 2),
            "krd_k": {str(k): r(v / 1e3, 2) for k, v in zip(KEYS, krd)},
            "krd_sum_k": r(krd.sum() / 1e3, 2),
            "convexity": r(conv, 1),
            "irrbb_pnl_mn": {s: r((value(curve.shifted(scenario_shift(s, "EUR"))) - pv) / 1e6, 2) for s in SCENARIOS},
        }

    out: dict[str, Any] = {"bullet": stats(v_b, bullet), "barbell": stats(v_bb, barbell), "keys": list(KEYS)}
    b, bb = out["bullet"], out["barbell"]
    ensure(abs(b["market_value_mn"] - bb["market_value_mn"]) < 1e-3, "barbell matches the bullet's market value")
    ensure(abs(b["dv01_k"] - bb["dv01_k"]) < 1e-2, "barbell matches the bullet's DV01")
    b["krd_10y_share_pct"] = r(b["krd_k"]["10"] / b["dv01_k"] * 100, 1)
    bb["krd_wings_share_pct"] = r((bb["krd_k"]["2"] + bb["krd_k"]["30"]) / bb["dv01_k"] * 100, 1)

    # Bucketed (quote) DV01 of the bullet: where each scheme puts the risk, and whether 1 bp bumps are linear.
    tau = float(_row(quotes, BULLET)["tau"])
    bucket: dict[str, Any] = {}
    for meth in METHODS:
        mk = eur[meth]
        vb = portfolio_value([pos(BULLET, NOTIONAL, mk.curve)], quotes)
        bk = bucketed_dv01(vb, mk.rebuild, len(mk.quotes), bp=BUCKET_BP)
        bk_1bp = bucketed_dv01(vb, mk.rebuild, len(mk.quotes), bp=1.0)
        assert mk.rebuild_all is not None
        joint = -(vb(mk.rebuild_all(1.0)) - vb(mk.rebuild_all(-1.0))) / 2.0
        lo = int(np.searchsorted(mk.pillar_times, tau)) - 1
        mag = np.abs(bk)
        total = mag.sum()
        bucket[meth] = {
            "by_quote_k": {lab: r(v / 1e3, 3) for lab, v in zip(mk.labels, bk)},
            "sum_k": r(bk.sum() / 1e3, 2),
            "bracket": [mk.labels[lo], mk.labels[lo + 1]],
            "share_bracket_pct": r((mag[lo] + mag[lo + 1]) / total * 100, 1),
            "share_four_nearest_pct": r(mag[max(lo - 1, 0) : lo + 3].sum() / total * 100, 1),
            "sign_changes": int(np.sum(np.diff(np.sign(bk[np.abs(bk) > 1.0])) != 0)),
            "sum_1bp_k": r(bk_1bp.sum() / 1e3, 2),
            "joint_1bp_k": r(joint / 1e3, 2),
            "gap_1bp_pct": r((bk_1bp.sum() / joint - 1.0) * 100, 2),
        }
    out["bucketed_bullet"] = bucket
    out["bucket_bp"] = BUCKET_BP
    # A 1 bp move in an annually compounded yield is about 1/(1+y) bp of continuously compounded zero rate.
    brow = _row(quotes, BULLET)
    y_b = yield_from_dirty(brow["bond"], float(brow["dirty"]), brow["settle"])
    out["bullet_yield_pct"] = r(y_b * 100, 3)
    out["zero_dv01_over_1py_k"] = r(b["dv01_k"] / (1.0 + y_b), 2)
    ensure(
        all(abs(bucket[m]["sum_k"] / out["zero_dv01_over_1py_k"] - 1.0) < 1e-3 for m in METHODS),
        "bucketed sums equal the zero-curve DV01 divided by (1 + y) within 0.1%",
    )

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(KEYS))
    width = 0.36
    labels = (
        ("bullet", f"Bullet: {b['positions'][0]['nominal_mn']:.0f}mn {b['positions'][0]['description']}"),
        ("barbell", f"Barbell: {bb['positions'][0]['description']} + {bb['positions'][1]['description']}"),
    )
    for i, (name, label) in enumerate(labels):
        vals = [out[name]["krd_k"][str(k)] for k in KEYS]
        ax.bar(x + (i - 0.5) * (width + 0.04), vals, width=width, color=SLOTS[i], label=label, zorder=3)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax.set_xticks(x, [f"{k}y" for k in KEYS])
    ax.set_ylabel("Key-rate DV01 (EUR thousand per bp)")
    ax.set_title("Same market value, same DV01, different curve exposure")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.99,
        0.95,
        f"Both: MV EUR {b['market_value_mn']:.1f}mn, DV01 EUR {b['dv01_k']:.1f}k/bp",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK2,
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "r2_key_rate_dv01.png")
    return out


# ── R3: PCA of daily curve changes ───────────────────────────────────────────


def r3(save: SaveFn) -> dict[str, Any]:
    ust = load_ust_par()[USD_PCA_TENORS].dropna()
    ecb = load_ecb_spot().dropna()
    out: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, (name, levels, title) in zip(
        axes, (("USD", ust, "US Treasury par yields"), ("EUR", ecb, "ECB AAA euro area spot curve"))
    ):
        changes = levels.diff().dropna() * 100.0  # percent -> bp
        res = principal_components(changes.to_numpy(), 3)
        half = len(changes) // 2
        first = principal_components(changes.iloc[:half].to_numpy(), 3)
        second = principal_components(changes.iloc[half:].to_numpy(), 3)
        sim = loading_similarity(first, second)
        out[name] = {
            "tenors": list(levels.columns),
            "start": str(changes.index[0].date()),
            "end": str(changes.index[-1].date()),
            "n_obs": res.n_obs,
            "explained_pct": [r(v * 100, 1) for v in res.explained],
            "cumulative_pct": r(res.explained.sum() * 100, 1),
            "pc_daily_sd_bp": [r(np.sqrt(v), 2) for v in res.eigenvalues],
            "subperiod_explained_pct": [
                [r(v * 100, 1) for v in first.explained],
                [r(v * 100, 1) for v in second.explained],
            ],
            "subperiod_loading_similarity": [r(v, 3) for v in sim],
            "least_stable_pc": int(np.argmin(sim)) + 1,
            "loadings": [[r(v, 4) for v in col] for col in res.loadings.T],
        }
        x = np.arange(len(levels.columns))
        for k, lab in enumerate(("PC1 level", "PC2 slope", "PC3 curvature")):
            ax.plot(
                x,
                res.loadings[:, k],
                color=SLOTS[k],
                lw=1.6,
                ls=DASH[k],
                marker="o",
                ms=4.5,
                mec=SURFACE,
                mew=1.2,
                label=f"{lab} ({res.explained[k] * 100:.1f}%)",
            )
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(x, list(levels.columns), fontsize=7.5)
        ax.set_title(title, pad=16)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.text(
            0.0,
            1.01,
            f"Daily changes {out[name]['start']} to {out[name]['end']}, n = {res.n_obs}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=INK2,
            fontsize=7.5,
        )
    axes[0].set_ylabel("Loading")
    fig.tight_layout()
    save(fig, "r3_pca_loadings.png")
    return out


# ── R4: Euro-Bund futures basis, implied repo, CTD ──────────────────────────


@dataclass
class Leg:
    isin: str
    description: str
    bond: FixedRateBond
    clean: float
    basis: Any


def r4(r2_out: dict[str, Any], save: SaveFn) -> dict[str, Any]:
    fgbl = load_fgbl()
    fgbl = fgbl[fgbl["contract"] == FGBL_DEC26].set_index("business_date")
    estr = load_estr()
    deliv = delivery_date(2026, 12)
    dl = load_deliverables()
    dec = dl[(dl["product"] == "FGBL") & (dl["contract"] == FGBL_DEC26)]
    isins = list(dec["isin"])
    published = dict(zip(dec["isin"], dec["conversion_factor"]))

    def day_basis(day: date) -> tuple[list[Leg], dict[str, Any]]:
        q = bunds_on(day).set_index("isin")
        fut = float(fgbl.loc[day, "settlement"])
        repo = float(estr.loc[pd.Timestamp(day)]) / 100.0
        settle = q["settle"].iloc[0]
        legs = []
        for isin in isins:
            bond = q.loc[isin, "bond"]
            cf = round(conversion_factor(bond, deliv), 6)
            clean = float(q.loc[isin, "clean"])
            legs.append(
                Leg(
                    isin, str(q.loc[isin, "description"]), bond, clean, basis(bond, clean, settle, deliv, fut, cf, repo)
                )
            )
        return legs, {"futures": fut, "repo": repo, "settle": settle}

    legs, meta = day_basis(ASOF)
    settle = meta["settle"]
    ctd = max(legs, key=lambda x: x.basis.implied_repo)
    rows = []
    for leg in legs:
        bs = leg.basis
        ytm = yield_from_dirty(leg.bond, leg.clean + leg.bond.accrued(settle), settle)
        ensure(abs(bs.cf - published[leg.isin]) < 5e-13, f"computed CF of {leg.isin} equals Eurex's")
        rows.append(
            {
                "isin": leg.isin,
                "description": leg.description,
                "cf": bs.cf,
                "cf_eurex": published[leg.isin],
                "clean": leg.clean,
                "yield_pct": r(ytm * 100, 3),
                "duration_mod": r(yield_risk(leg.bond, ytm, settle).duration_mod, 2),
                "gross_basis_cents": r(bs.gross_basis * 100, 1),
                "carry_cents": r(bs.carry * 100, 1),
                "net_basis_cents": r(bs.net_basis * 100, 1),
                "implied_repo_pct": r(bs.implied_repo * 100, 2),
                "ctd": leg.isin == ctd.isin,
            }
        )
    by_irr = sorted(legs, key=lambda x: -x.basis.implied_repo)
    margin = (by_irr[0].basis.implied_repo - by_irr[1].basis.implied_repo) * 100
    shortest = min(rows, key=lambda x: x["duration_mod"])
    ensure(shortest["isin"] == ctd.isin, "the CTD is the lowest-duration deliverable")
    ensure(all(x["yield_pct"] < NOTIONAL_COUPON["FGBL"] for x in rows), "every deliverable yields below 6%")
    ensure(all(x["clean"] > 100.0 * x["cf"] for x in rows), "the conversion factor understates every bond's value")
    ratios = {x["isin"]: x["clean"] / x["cf"] for x in rows}
    ensure(min(ratios, key=lambda k: ratios[k]) == ctd.isin, "the CTD has the lowest clean price / CF")

    # Implied repo sensitivity to the futures price: d(IRR)/dF = CF / (dirty * d/360) when no coupon is paid.
    cb = ctd.basis
    ensure(cb.coupons == 0.0, "no coupon between spot settlement and delivery for the CTD")
    irr_per_cent = cb.cf / (cb.dirty * cb.days / 360.0) / 100.0 * 100.0  # percentage points per cent of futures

    # CTD switch under parallel yield shifts.
    bonds = [x.bond for x in legs]
    cleans = [x.clean for x in legs]
    cfs = [x.basis.cf for x in legs]
    shifts = np.arange(-200.0, 400.0 + 1e-9, 2.5)
    grid = ctd_switch(bonds, cleans, cfs, settle, deliv, meta["repo"], shifts)
    changes = np.nonzero(np.diff(grid.ctd_index))[0]
    ensure(changes.size >= 1, "the CTD switches within the shift grid")
    k = int(changes[0])
    i0, i1 = int(grid.ctd_index[k]), int(grid.ctd_index[k + 1])

    def gap(s: float) -> float:
        g = ctd_switch(bonds, cleans, cfs, settle, deliv, meta["repo"], np.array([s]))
        return float(g.net_basis[0, i0] - g.net_basis[0, i1])

    s_star = float(brentq(gap, shifts[k], shifts[k + 1], xtol=1e-6))
    y_ctd = yield_from_dirty(bonds[i0], cleans[i0] + bonds[i0].accrued(settle), settle)
    switch = {
        "shift_bp": r(s_star, 1),
        "from": legs[i0].description,
        "to": legs[i1].description,
        "ctd_yield_at_switch_pct": r((y_ctd + s_star * 1e-4) * 100, 2),
    }
    dv01_contract = futures_dv01(bonds, cleans, cfs, settle, deliv, meta["repo"]) / 100.0 * CONTRACT_SIZE
    hedge = r2_out["bullet"]["dv01_k"] * 1e3 / dv01_contract

    history = []
    for day in sorted(fgbl.index):
        if pd.Timestamp(day) not in estr.index:
            continue
        try:
            day_legs, mt = day_basis(day)
        except (KeyError, ValueError):
            continue
        best = max(day_legs, key=lambda x: x.basis.implied_repo)
        history.append(
            {
                "date": day.isoformat(),
                "futures": mt["futures"],
                "estr_pct": r(mt["repo"] * 100, 3),
                "ctd": best.description,
                "ctd_implied_repo_pct": r(best.basis.implied_repo * 100, 3),
                "ctd_net_basis_cents": r(best.basis.net_basis * 100, 2),
            }
        )
    spread = np.array([h["ctd_implied_repo_pct"] - h["estr_pct"] for h in history])
    out = {
        "asof": ASOF.isoformat(),
        "contract": "FGBL Dec 2026",
        "delivery": deliv.isoformat(),
        "spot_settlement": settle.isoformat(),
        "days": int(legs[0].basis.days),
        "futures_settlement": meta["futures"],
        "estr_pct": r(meta["repo"] * 100, 3),
        "open_interest": int(fgbl.loc[ASOF, "open_interest"]),
        "basket": rows,
        "ctd": ctd.description,
        "ctd_isin": ctd.isin,
        "ctd_margin_vs_next_pct": r(margin, 2),
        "irr_pp_per_10_cents": r(irr_per_cent * FUTURES_MOVE_CENTS, 2),
        "switch": switch,
        "contract_dv01_eur": r(dv01_contract, 2),
        "bullet_hedge_contracts": r(hedge, 1),
        "history_start": history[0]["date"],
        "history_end": history[-1]["date"],
        "history_days": len(history),
        "history_ctd_unique": sorted({h["ctd"] for h in history}),
        "irr_minus_estr_mean_pct": r(spread.mean(), 2),
        "irr_minus_estr_sd_pct": r(spread.std(ddof=1), 2),
        "irr_minus_estr_min_pct": r(spread.min(), 2),
        "irr_minus_estr_max_pct": r(spread.max(), 2),
        "history": history,
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for i, leg in enumerate(legs):
        ax.plot(shifts, grid.net_basis[:, i] * 100, color=SLOTS[i], lw=1.6, ls=DASH[i], label=leg.description)
    ax.axvline(switch["shift_bp"], color=MUTED, lw=0.8)
    ax.text(
        switch["shift_bp"] + 6,
        ax.get_ylim()[1] * 0.92,
        f"CTD switch at {switch['shift_bp']:+.0f} bp\n(CTD yield {switch['ctd_yield_at_switch_pct']:.2f}%)",
        color=INK2,
        fontsize=7.5,
        va="top",
    )
    ax.set_xlabel("Parallel shift of deliverable yields (bp)")
    ax.set_ylabel("Theoretical net basis (cents per 100)")
    ax.set_title("FGBL Dec-26 deliverables: net basis vs yield level")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    d = pd.to_datetime([h["date"] for h in history])
    ax.plot(
        d,
        [h["ctd_implied_repo_pct"] for h in history],
        color=SLOTS[0],
        lw=1.6,
        marker="o",
        ms=4,
        mec=SURFACE,
        mew=1.0,
        label="CTD implied repo",
    )
    ax.plot(d, [h["estr_pct"] for h in history], color=SLOTS[1], lw=1.6, ls=DASH[1], label="€STR")
    ax.set_ylabel("Rate (%, ACT/360)")
    ax.set_title("CTD implied repo vs €STR, October 2026")
    ax.legend(loc="upper center", fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    save(fig, "r4_bund_futures_basis.png")
    return out


# ── Text: README key block and docs/results.md ──────────────────────────────


def _range(values: list[float], nd: int) -> str:
    return f"{min(values):.{nd}f}-{max(values):.{nd}f}"


def r1_text(o: dict[str, Any]) -> str:
    usd, eur = o["USD"], o["EUR"]
    rows = []
    for ccy, s in (("USD", usd), ("EUR", eur)):
        for m in METHODS:
            v = s["methods"][m]
            rows.append(
                [
                    ccy,
                    LABEL[m],
                    f"< {v['repricing_error_below_bp']:.0e}",
                    f"{v['max_forward_jump_bp']:.1f}",
                    f"{v['forward_min_pct']:.2f}-{v['forward_max_pct']:.2f}",
                    *[f"{v[f'fwd_{a}y{b}y_pct']:.3f}" for a, b in FWD_WINDOWS],
                    f"{v['oos_rms_bp']:.2f}" if "oos_rms_bp" in v else "-",
                ]
            )
    head = [
        "Curve",
        "Scheme",
        "Max repricing error (bp)",
        "Largest forward jump at a pillar (bp)",
        "Forward range (%)",
        *[f"{a}y{b}y fwd (%)" for a, b in FWD_WINDOWS],
        "Out-of-sample RMS (bp)",
    ]
    ensure(o["continuous_forward_schemes"] == ["monotone_convex"], "monotone convex is the only jump-free scheme")
    w_u = usd["widest_window"]
    best = o["best_oos"]
    others = [eur["methods"][m]["oos_rms_bp"] for m in METHODS if m != best]
    oos_n = eur["methods"][best]["oos_n"]
    return (
        table(head, rows)
        + "\n\n"
        + f"All four schemes reprice their own calibration instruments ({usd['instruments']} Treasury par points, "
        f"{eur['instruments']} Bunds) to within {o['max_repricing_error_below_bp']:.0e} bp and their zero curves differ by at "
        f"most {usd['max_zero_spread_bp']:.1f} bp (USD) and {eur['max_zero_spread_bp']:.1f} bp (EUR), yet their "
        f"instantaneous forwards differ by up to {usd['max_forward_spread_bp']:.0f} bp and {eur['max_forward_spread_bp']:.0f}"
        f" bp, and the USD {w_u} forward by {usd['forward_window_spread_bp'][w_u]:.1f} bp: the interpolation scheme is a "
        f"forward-rate decision. Monotone convex is the only scheme with no forward jump at any pillar; on the "
        f"{oos_n} Bunds left out of the calibration, the lowest RMS yield error comes from {LABEL[best]} "
        f"({eur['methods'][best]['oos_rms_bp']:.2f} bp, against {_range(others, 2)} bp for the others)."
    )


def r2_text(o: dict[str, Any]) -> str:
    b, bb = o["bullet"], o["barbell"]
    rows = [
        [
            name,
            ", ".join(f"{p['nominal_mn']:.1f}mn {p['description']}" for p in s["positions"]),
            f"{s['market_value_mn']:.1f}",
            f"{s['dv01_k']:.1f}",
            *[f"{s['krd_k'][str(k)]:.1f}" for k in KEYS],
            f"{s['convexity']:.0f}",
            f"{s['irrbb_pnl_mn']['steepener']:+.2f}",
            f"{s['irrbb_pnl_mn']['flattener']:+.2f}",
        ]
        for name, s in (("Bullet", b), ("Barbell", bb))
    ]
    bk = o["bucketed_bullet"]
    brows = [
        [
            LABEL[m],
            f"{bk[m]['sum_k']:.2f}",
            f"{bk[m]['share_bracket_pct']:.1f}%",
            f"{bk[m]['share_four_nearest_pct']:.1f}%",
            str(bk[m]["sign_changes"]),
            f"{bk[m]['sum_1bp_k']:.2f} vs {bk[m]['joint_1bp_k']:.2f} ({bk[m]['gap_1bp_pct']:+.2f}%)",
        ]
        for m in METHODS
    ]
    local = [m for m in METHODS if bk[m]["share_bracket_pct"] >= 97.0]
    ensure(set(local) == {"log_linear_df", "linear_zero"}, "only the two linear schemes keep >= 97% on the bracket")
    gaps = {m: abs(bk[m]["gap_1bp_pct"]) for m in METHODS}
    nonlinear = [m for m in METHODS if gaps[m] > 0.5]
    ensure(nonlinear == ["monotone_convex"], "only monotone convex is visibly non-linear under 1 bp quote bumps")
    steep_gap = b["irrbb_pnl_mn"]["steepener"] - bb["irrbb_pnl_mn"]["steepener"]
    mc, sp = bk["monotone_convex"], bk["natural_cubic_zero"]
    return (
        table(
            [
                "Portfolio",
                "Positions (EUR nominal)",
                "MV (EUR mn)",
                "DV01 (EUR k/bp)",
                *[f"KRD {k}y" for k in KEYS],
                "Convexity",
                "IRRBB steepener P&L (EUR mn)",
                "IRRBB flattener P&L (EUR mn)",
            ],
            rows,
        )
        + "\n\n"
        + f"The barbell matches the bullet's market value and DV01 exactly, but {b['krd_10y_share_pct']:.0f}% of the "
        f"bullet's DV01 sits on the 10y key rate while {bb['krd_wings_share_pct']:.0f}% of the barbell's sits on the 2y "
        f"and 30y keys; the single DV01 number hides a {abs(steep_gap):.2f}mn difference in P&L under the SRP31 "
        f"steepener and a convexity of {bb['convexity']:.0f} against {b['convexity']:.0f}.\n\n"
        + f"Bucketed DV01 of the bullet (sensitivity to each calibration Bund's yield, {o['bucket_bp']} bp bumps, "
        f"EUR k/bp; the bullet is not a calibration bond and sits between {mc['bracket'][0]} and {mc['bracket'][1]}):\n\n"
        + table(
            [
                "Scheme",
                "Sum",
                "Share on the two bracketing Bunds",
                "Share on the four nearest",
                "Sign changes",
                "1 bp bumps: sum of buckets vs joint shift",
            ],
            brows,
        )
        + "\n\n"
        + f"Both linear schemes keep at least {min(bk[m]['share_bracket_pct'] for m in local):.0f}% of the risk on the two "
        f"bracketing Bunds; monotone convex puts {mc['share_bracket_pct']:.0f}% there and "
        f"{mc['share_four_nearest_pct']:.1f}% on the four nearest, while the natural spline spreads "
        f"{100 - sp['share_four_nearest_pct']:.1f}% beyond them with {sp['sign_changes']} sign changes. Monotone convex "
        f"is only piecewise smooth in the quotes (its Hagan-West shape region can change under a 1 bp bump): 1 bp "
        f"bucket DV01s add up to {mc['sum_1bp_k']:.2f}k against {mc['joint_1bp_k']:.2f}k for a joint 1 bp shift "
        f"({mc['gap_1bp_pct']:+.1f}%), so its bucketed risk is computed with {o['bucket_bp']} bp bumps. The bucket sums "
        f"sit below the zero-curve DV01 ({b['dv01_k']:.2f}k) because a 1 bp move in an annually compounded yield is "
        f"1/(1+y) bp of continuously compounded zero rate: {b['dv01_k']:.2f}k / (1 + {o['bullet_yield_pct']:.3f}%) = "
        f"{o['zero_dv01_over_1py_k']:.2f}k."
    )


def r3_text(o: dict[str, Any]) -> str:
    names = {1: "level", 2: "slope", 3: "curvature"}
    rows = [
        [
            c,
            f"{s['start']} to {s['end']}",
            str(s["n_obs"]),
            *[f"{v:.1f}" for v in s["explained_pct"]],
            f"{s['cumulative_pct']:.1f}",
            " / ".join(f"{v:.2f}" for v in s["pc_daily_sd_bp"]),
            " / ".join(f"{v:.3f}" for v in s["subperiod_loading_similarity"]),
        ]
        for c, s in (("USD Treasury par", o["USD"]), ("EUR AAA spot (ECB)", o["EUR"]))
    ]
    usd, eur = o["USD"], o["EUR"]
    min_sim = min(min(s["subperiod_loading_similarity"]) for s in (usd, eur))
    return (
        table(
            [
                "Curve",
                "Window",
                "Daily changes",
                "PC1 %",
                "PC2 %",
                "PC3 %",
                "Cumulative %",
                "Factor daily sd (bp)",
                "Loading similarity, 1st vs 2nd half (PC1/PC2/PC3)",
            ],
            rows,
        )
        + "\n\n"
        + f"Three factors explain {usd['cumulative_pct']:.1f}% (USD) and {eur['cumulative_pct']:.1f}% (EUR) of the "
        f"variance of daily curve changes, level alone {usd['explained_pct'][0]:.1f}% and {eur['explained_pct'][0]:.1f}%; "
        f"the loadings estimated on each half of the sample have a cosine similarity of at least {min_sim:.2f} "
        f"(least stable factor: {names[usd['least_stable_pc']]} for USD, {names[eur['least_stable_pc']]} for EUR). "
        f"Both inputs are already smoothed (Treasury monotone convex par curve, ECB Svensson fit), which concentrates "
        f"variance in the first factors; the shares are an upper bound for raw bond yields."
    )


def r4_text(o: dict[str, Any]) -> str:
    rows = [
        [
            x["description"],
            x["isin"],
            f"{x['cf']:.6f}",
            f"{x['clean']:.2f}",
            f"{x['yield_pct']:.3f}",
            f"{x['duration_mod']:.2f}",
            f"{x['gross_basis_cents']:.1f}",
            f"{x['carry_cents']:.1f}",
            f"{x['net_basis_cents']:.1f}",
            f"{x['implied_repo_pct']:.2f}",
            "yes" if x["ctd"] else "",
        ]
        for x in o["basket"]
    ]
    sw = o["switch"]
    one_ctd = len(o["history_ctd_unique"]) == 1
    ensure(one_ctd, "the CTD is the same bond on every day of the history")
    ctd_row = next(x for x in o["basket"] if x["ctd"])
    return (
        table(
            [
                "Bond",
                "ISIN",
                "Conversion factor",
                "Clean",
                "Yield (%)",
                "Modified duration",
                "Gross basis (c)",
                "Carry (c)",
                "Net basis (c)",
                "Implied repo (%)",
                "CTD",
            ],
            rows,
        )
        + "\n\n"
        + f"Inputs: FGBL Dec-26 daily settlement {o['futures_settlement']:.2f} (Eurex, {o['asof']}; open interest "
        f"{o['open_interest']:,} contracts), Bundesbank reference prices of the same day with spot settlement "
        f"{o['spot_settlement']}, delivery {o['delivery']} ({o['days']} days), €STR {o['estr_pct']:.3f}% as the "
        f"financing rate for carry. Conversion factors are computed and equal Eurex's published values.\n\n"
        + f"The {o['ctd']} is cheapest to deliver with an implied repo of {ctd_row['implied_repo_pct']:.2f}% against "
        f"€STR at {o['estr_pct']:.2f}%, {o['ctd_margin_vs_next_pct']:.1f} points above the next bond, and it was the CTD "
        f"on all {o['history_days']} trading days from {o['history_start']} to {o['history_end']}. With every deliverable "
        f"yielding below the 6% notional coupon, the conversion factor (a price at a 6% yield) understates every bond's "
        f"value and understates the lowest-duration bond's the least, so that bond is cheapest; the CTD would switch to the {sw['to']} after "
        f"a {sw['shift_bp']:+.0f} bp parallel rise in yields (CTD yield {sw['ctd_yield_at_switch_pct']:.2f}%).\n\n"
        + f"Over the month the CTD implied repo minus €STR averaged {o['irr_minus_estr_mean_pct']:+.2f} points with a "
        f"daily standard deviation of {o['irr_minus_estr_sd_pct']:.2f} points. The bond prices (Bundesbank reference "
        f"prices of the Frankfurt Stock Exchange) and the futures price (Eurex settlement, volume-weighted price of the "
        f"minute before 17:15 CET) are separate fixings, and a {FUTURES_MOVE_CENTS:.0f}-cent discrepancy between them "
        f"moves the implied repo by {o['irr_pp_per_10_cents']:.2f} points: the daily dispersion corresponds to about "
        f"{o['irr_minus_estr_sd_pct'] / o['irr_pp_per_10_cents'] * FUTURES_MOVE_CENTS:.0f} cents of price discrepancy, so "
        f"single-day implied repos from these sources are not a trading signal. One contract carries EUR "
        f"{o['contract_dv01_eur']:.2f} of DV01, so the 100mn bullet of R2 is DV01-hedged by selling "
        f"{o['bullet_hedge_contracts']:.0f} contracts."
    )


def validation_text(v: dict[str, Any], r1o: dict[str, Any]) -> str:
    return table(
        ["Check", "Result"],
        [
            [
                "Calibration instruments repriced under each of the four schemes",
                f"{r1o['USD']['instruments']} Treasury par points and {r1o['EUR']['instruments']} Bunds, max error "
                f"below {r1o['max_repricing_error_below_bp']:.0e} bp",
            ],
            [
                "Bundesbank accrued interest (dirty - clean) reproduced, ACT/ACT ICMA, T+2",
                f"{v['bundesbank_quotes']} quotes ({v['bundesbank_securities']} securities, {v['bundesbank_days']} days), "
                f"max error below {v['max_accrued_error_below_per_100']:.0e} per 100",
            ],
            [
                "Bundesbank yields reproduced (ICMA annual compounding)",
                f"max error below {v['max_yield_error_below_bp']:.0e} bp (published to 1e-6 %)",
            ],
            [
                "Eurex conversion factors reproduced to six decimals",
                f"{v['eurex_cf_matched']} / {v['eurex_cf_total']} ({', '.join(v['eurex_products'])})",
            ],
        ],
    )


def key_block(res: dict[str, Any]) -> str:
    r1o, r2o, r3o, r4o, v = res["r1"], res["r2"], res["r3"], res["r4"], res["validation"]
    usd, eur = r1o["USD"], r1o["EUR"]
    w = usd["widest_window"]
    b, bb = r2o["bullet"], r2o["barbell"]
    ctd_row = next(x for x in r4o["basket"] if x["ctd"])
    steep_gap = abs(b["irrbb_pnl_mn"]["steepener"] - bb["irrbb_pnl_mn"]["steepener"])
    rows = [
        [
            "R1. Does the interpolation scheme matter if every scheme reprices the curve exactly?",
            f"Zero curves agree within {max(usd['max_zero_spread_bp'], eur['max_zero_spread_bp']):.1f} bp; instantaneous "
            f"forwards differ by up to {usd['max_forward_spread_bp']:.0f} bp (USD) and {eur['max_forward_spread_bp']:.0f} "
            f"bp (EUR); USD {w} forward spread {usd['forward_window_spread_bp'][w]:.1f} bp",
            "Yes for anything forward-starting: the scheme is a pricing and risk decision, not a plotting detail.",
        ],
        [
            f"R2. What does a single DV01 hide? (EUR {b['positions'][0]['nominal_mn']:.0f}mn "
            f"{b['positions'][0]['description']} vs a barbell with the same MV and DV01)",
            f"Same MV and DV01 (EUR {b['dv01_k']:.1f}k/bp); {b['krd_10y_share_pct']:.0f}% of the bullet's DV01 on the 10y "
            f"key vs {bb['krd_wings_share_pct']:.0f}% of the barbell's on 2y+30y; steepener P&L gap EUR "
            f"{steep_gap:.2f}mn",
            "Curve risk needs key-rate or bucketed DV01; the bucket split also depends on the interpolation scheme.",
        ],
        [
            "R3. How many factors drive daily curve moves? (2023-2026)",
            f"Level / slope / curvature: {' / '.join(f'{x:.1f}' for x in r3o['USD']['explained_pct'])}% (USD), "
            f"{' / '.join(f'{x:.1f}' for x in r3o['EUR']['explained_pct'])}% (EUR)",
            "Three factors carry about 96% of the variance, the basis for factor hedging and scenario design.",
        ],
        [
            f"R4. Which Bund is cheapest to deliver into FGBL Dec-26? ({r4o['asof']})",
            f"{r4o['ctd']}: implied repo {ctd_row['implied_repo_pct']:.2f}% vs €STR {r4o['estr_pct']:.2f}%, net basis "
            f"{ctd_row['net_basis_cents']:.1f} c; CTD switch after a {r4o['switch']['shift_bp']:+.0f} bp parallel rise",
            "Below the 6% notional coupon the lowest-duration bond is CTD; over the month its implied repo averaged "
            "within one daily standard deviation of €STR.",
        ],
    ]
    ensure(
        all(c["max_forward_spread_bp"] > 3 * c["max_zero_spread_bp"] for c in (usd, eur)),
        "forwards differ across schemes far more than zero rates do",
    )
    ensure(
        abs(r3o["USD"]["cumulative_pct"] - 96) < 1.5 and abs(r3o["EUR"]["cumulative_pct"] - 96) < 1.5,
        "three factors carry about 96% of the variance",
    )
    ensure(
        abs(r4o["irr_minus_estr_mean_pct"]) < r4o["irr_minus_estr_sd_pct"],
        "the CTD's IRR minus €STR is within one daily standard deviation of zero on average",
    )
    validation_line = (
        f"Validation: every calibration instrument reprices to within {r1o['max_repricing_error_below_bp']:.0e} bp under each scheme; "
        f"{v['bundesbank_quotes']} Bundesbank accrued amounts and yields are reproduced to within "
        f"{v['max_accrued_error_below_per_100']:.0e} per 100 and {v['max_yield_error_below_bp']:.0e} bp; "
        f"{v['eurex_cf_matched']} of {v['eurex_cf_total']} Eurex conversion factors are reproduced to six decimals."
    )
    return table(["Question", "Result (as of 27 Oct 2026)", "Reading"], rows) + "\n\n" + validation_line


def results_md(res: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Detailed results",
            "",
            "Generated by `python scripts/run_results.py` from `data/sample/` (as of "
            f"{res['asof']}). Do not edit by hand: the CI fails if this file differs from the script output.",
            "",
            "## Validation against published values",
            "",
            validation_text(res["validation"], res["r1"]),
            "",
            "## R1. Instantaneous forwards by interpolation scheme",
            "",
            "![Forwards by interpolation](figures/r1_forwards_by_interpolation.png)",
            "",
            r1_text(res["r1"]),
            "",
            "## R2. DV01, key-rate DV01 and bucketed DV01 of Bund portfolios",
            "",
            "![Key-rate DV01](figures/r2_key_rate_dv01.png)",
            "",
            r2_text(res["r2"]),
            "",
            "## R3. Principal components of daily curve changes",
            "",
            "![PCA loadings](figures/r3_pca_loadings.png)",
            "",
            r3_text(res["r3"]),
            "",
            "## R4. Euro-Bund futures: basket, conversion factors, CTD, net basis, implied repo",
            "",
            "![Bund futures basis](figures/r4_bund_futures_basis.png)",
            "",
            r4_text(res["r4"]),
            "",
        ]
    )


def json_diff(old: Any, new: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Leaf-level differences between two JSON documents (for the --check diagnostic)."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [d for k in sorted(set(old) | set(new)) for d in json_diff(old.get(k), new.get(k), f"{path}.{k}")]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in json_diff(a, b, f"{path}[{i}]")]
    return [] if old == new else [(path, old, new)]


def inject(readme: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- results:{name} -->\n)(.*?)(\n<!-- /results:{name} -->)", re.S)
    if not pat.search(readme):
        raise SystemExit(f"README block {name!r} not found")
    return pat.sub(lambda m: m.group(1) + body + m.group(3), readme)


def compute(save: SaveFn) -> dict[str, Any]:
    quotes = eligible_bunds(ASOF)
    usd = {m: build_usd_curve(ASOF, m) for m in METHODS}
    eur = {m: build_eur_curve(ASOF, m) for m in METHODS}
    res: dict[str, Any] = {"asof": ASOF.isoformat()}
    res["validation"] = validation()
    res["r1"] = r1(usd, eur, quotes, save)
    res["r2"] = r2(eur, quotes, save)
    res["r3"] = r3(save)
    res["r4"] = r4(res["r2"], save)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify committed outputs, write nothing")
    args = parser.parse_args()
    style()
    if not args.check:
        FIG.mkdir(parents=True, exist_ok=True)
    res = compute(discard_png if args.check else write_png)
    key_json = json.dumps(res, indent=1, ensure_ascii=False) + "\n"
    report = results_md(res)
    readme = README.read_text(encoding="utf-8")
    new_readme = inject(readme, "key", key_block(res))
    if args.check:
        stale = [
            name
            for name, path, text in (
                ("docs/key_results.json", KEY_JSON, key_json),
                ("docs/results.md", RESULTS_MD, report),
            )
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if new_readme != readme:
            stale.append("README.md")
        if stale:
            print("Out of date: " + ", ".join(stale) + ". Run: python scripts/run_results.py")
            if KEY_JSON.exists():
                for key, old, new in json_diff(json.loads(KEY_JSON.read_text(encoding="utf-8")), json.loads(key_json)):
                    print(f"  {key}: committed {old!r}, recomputed {new!r}")
            sys.exit(1)
        print("README, docs/results.md and docs/key_results.json are up to date.")
        return
    KEY_JSON.write_text(key_json, encoding="utf-8")
    RESULTS_MD.write_text(report, encoding="utf-8")
    README.write_text(new_readme, encoding="utf-8")
    print(
        f"Wrote {len(list(FIG.glob('*.png')))} figures, {RESULTS_MD.relative_to(ROOT)}, {KEY_JSON.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
