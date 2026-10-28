"""Loaders for the frozen public data in ``data/sample/`` and raw-file parsers.

The sample is produced by ``scripts/freeze_sample.py`` from the raw downloads
of ``scripts/fetch_data.py``; every file is listed with its SHA-256 and source
URL in ``data/sample/MANIFEST.json``. Rates are stored in percent, as
published; loaders return decimals where noted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from .bonds import yield_from_dirty
from .conventions import add_business_days, quasi_coupon_schedule
from .instruments import FixedRateBond

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
ASOF = date(2026, 10, 27)

UST_COLUMNS = {
    "1 Mo": "1M",
    "1.5 Month": "1.5M",
    "2 Mo": "2M",
    "3 Mo": "3M",
    "4 Mo": "4M",
    "6 Mo": "6M",
    "1 Yr": "1Y",
    "2 Yr": "2Y",
    "3 Yr": "3Y",
    "5 Yr": "5Y",
    "7 Yr": "7Y",
    "10 Yr": "10Y",
    "20 Yr": "20Y",
    "30 Yr": "30Y",
}


def tenor_to_years(label: str) -> float:
    """'3M' -> 0.25, '1.5M' -> 0.125, '10Y' -> 10.0."""
    value, unit = float(label[:-1]), label[-1].upper()
    if unit == "M":
        return value / 12.0
    if unit == "Y":
        return value
    raise ValueError(f"bad tenor label {label!r}")


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(sample_dir: Path = SAMPLE_DIR) -> dict[str, str]:
    """Check every file listed in MANIFEST.json against its SHA-256; raise on mismatch."""
    manifest = json.loads((sample_dir / "MANIFEST.json").read_text())
    for name, meta in manifest["files"].items():
        digest = sha256(sample_dir / name)
        if digest != meta["sha256"]:
            raise ValueError(f"checksum mismatch for {name}")
    return {name: meta["sha256"] for name, meta in manifest["files"].items()}


# ── Raw parsers (used by scripts/freeze_sample.py) ──────────────────────────


def parse_bundesbank_workbook(path: Path) -> pd.DataFrame:
    """One row per (price date, security) from a Bundesbank monthly workbook."""
    book = pd.ExcelFile(path)
    frames = []
    for sheet in book.sheet_names:
        df = book.parse(sheet, header=1)
        df.columns = [
            "isin",
            "coupon",
            "description",
            "maturity",
            "residual",
            "issue_volume_bn",
            "clean",
            "yield",
            "dirty",
        ]
        df = df[df["isin"].astype(str).str.match(r"^DE[0-9A-Z]{10}$")].copy()
        df["price_date"] = datetime.strptime(str(sheet), "%d.%m.%Y").date()
        df["maturity"] = pd.to_datetime(df["maturity"]).dt.date
        frames.append(df.drop(columns=["residual"]))
    out = pd.concat(frames, ignore_index=True)
    return out[
        ["price_date", "isin", "coupon", "description", "maturity", "issue_volume_bn", "clean", "yield", "dirty"]
    ]


@dataclass(frozen=True)
class Observation:
    """One Bundesbank quote: settlement date, accrued, dirty price, yield (percent)."""

    settle: date
    accrued: float
    dirty: float
    yield_pct: float


def _matches(bond: FixedRateBond, obs: list[Observation], ai_tol: float, y_tol_bp: float, check_yield: bool) -> bool:
    for o in obs:
        if abs(bond.accrued(o.settle) - o.accrued) > ai_tol:
            return False
    if check_yield:
        for o in obs:
            if abs(yield_from_dirty(bond, o.dirty, o.settle) * 1e4 - o.yield_pct * 100) > y_tol_bp:
                return False
    return True


def infer_first_period(
    isin: str,
    coupon_pct: float,
    maturity: date,
    obs: list[Observation],
    ai_tol: float = 2e-6,
    y_tol_bp: float = 0.01,
) -> tuple[date | None, date | None]:
    """Recover an irregular first coupon period from published accrued interest and yields.

    Returns ``(None, None)`` when the regular-schedule assumption reproduces every
    observation. Otherwise searches the interest commencement date ``S`` (day by
    day, up to two years before the first observation) and the first coupon date
    ``F`` (first or second quasi-coupon date after ``S``), keeps the hypotheses that
    reproduce all accrued amounts within ``ai_tol``, and disambiguates short versus
    long first coupons with the published yields (within ``y_tol_bp``). Raises if
    the answer is not unique.
    """
    regular = FixedRateBond(isin, coupon_pct / 100, maturity)
    if _matches(regular, obs, ai_tol, y_tol_bp, check_yield=True):
        return None, None
    first = min(o.settle for o in obs)
    candidates: list[FixedRateBond] = []
    for back in range(0, 800):
        start = first - timedelta(days=back)
        sched = quasi_coupon_schedule(maturity, 1, start)
        after = [d for d in sched if d > start][:2]
        for fc in after:
            bond = FixedRateBond(isin, coupon_pct / 100, maturity, 1, start, fc)
            if _matches(bond, obs, ai_tol, y_tol_bp, check_yield=False):
                candidates.append(bond)
    survivors = [b for b in candidates if _matches(b, obs, ai_tol, y_tol_bp, check_yield=True)]
    if len(survivors) != 1:
        raise ValueError(f"{isin}: {len(survivors)} first-period hypotheses fit the data")
    return survivors[0].interest_start, survivors[0].first_coupon


# ── Sample loaders ───────────────────────────────────────────────────────────


def load_ust_par(sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """US Treasury par yields (percent), date index, tenor-label columns."""
    df = pd.read_csv(sample_dir / "ust_par_yields.csv", parse_dates=["date"]).set_index("date")
    return df.sort_index()


def load_ecb_spot(sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """ECB AAA euro area government spot rates (percent, continuous compounding)."""
    df = pd.read_csv(sample_dir / "ecb_aaa_spot.csv", parse_dates=["date"]).set_index("date")
    return df.sort_index()


def load_estr(sample_dir: Path = SAMPLE_DIR) -> pd.Series:
    """€STR (percent, ACT/360) by date."""
    df = pd.read_csv(sample_dir / "estr.csv", parse_dates=["date"]).set_index("date")
    return df["rate"].sort_index()


def load_bond_static(sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """Irregular first coupon periods (interest commencement and first coupon dates)."""
    df = pd.read_csv(sample_dir / "bond_static.csv")
    for col in ("interest_start", "first_coupon"):
        df[col] = pd.to_datetime(df[col]).dt.date
    return df.set_index("isin")


def load_bund_quotes(sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """Bundesbank reference prices and yields of listed Federal securities."""
    df = pd.read_csv(sample_dir / "bundesbank_federal_securities.csv")
    df["price_date"] = pd.to_datetime(df["price_date"]).dt.date
    df["maturity"] = pd.to_datetime(df["maturity"]).dt.date
    return df


def make_bond(row: pd.Series, static: pd.DataFrame) -> FixedRateBond:
    """FixedRateBond from a quote row and the static first-period table."""
    isin = str(row["isin"])
    start: date | None = None
    first: date | None = None
    if isin in static.index:
        start = cast(date, static.loc[isin, "interest_start"])
        first = cast(date, static.loc[isin, "first_coupon"])
    return FixedRateBond(isin, float(row["coupon"]) / 100, cast(date, row["maturity"]), 1, start, first)


def bunds_on(
    price_date: date, sample_dir: Path = SAMPLE_DIR, include_linkers: bool = False, include_green: bool = False
) -> pd.DataFrame:
    """Quotes of one price date with a ``bond`` column and the T+2 settlement date."""
    quotes = load_bund_quotes(sample_dir)
    static = load_bond_static(sample_dir)
    day = quotes[quotes["price_date"] == price_date].copy()
    if day.empty:
        raise ValueError(f"no Bundesbank quotes on {price_date}")
    desc = day["description"].astype(str)
    if not include_linkers:
        day = day[~desc.str.contains("I/L")]
    if not include_green:
        day = day[~day["description"].astype(str).str.contains(r"\bG\b", regex=True)]
    day["settle"] = add_business_days(price_date, 2)
    day["bond"] = [make_bond(r, static) for _, r in day.iterrows()]
    return day.sort_values("maturity").reset_index(drop=True)


def load_fgbl(sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """Eurex Euro-Bund futures daily statistics (settlement, volume, open interest)."""
    df = pd.read_csv(sample_dir / "eurex_fgbl_daily.csv", dtype={"contract": str})
    df["business_date"] = pd.to_datetime(df["business_date"]).dt.date
    return df


def load_deliverables(sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """Eurex deliverable bonds and published conversion factors."""
    df = pd.read_csv(sample_dir / "eurex_deliverable_bonds.csv", dtype={"contract": str})
    df["maturity"] = pd.to_datetime(df["maturity"]).dt.date
    return df


def as_decimal(values: pd.DataFrame | pd.Series) -> np.ndarray:
    """Percent to decimal as a float array."""
    return np.asarray(values, dtype=float) / 100.0
