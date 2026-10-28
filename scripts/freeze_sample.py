"""Turn ``data/raw/`` (from ``fetch_data.py``) into the versioned snapshot ``data/sample/``.

Writes tidy CSV files, the inferred irregular first coupon periods, and
``MANIFEST.json`` with the SHA-256 of every file and its sources. Everything
downstream (tests, results, notebook) reads only ``data/sample/``.

Usage::

    python scripts/freeze_sample.py --asof 2026-10-27
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ratescurve.conventions import add_business_days
from ratescurve.data import (
    UST_COLUMNS,
    Observation,
    infer_first_period,
    parse_bundesbank_workbook,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sample"


def freeze_ust(asof: date, start: date) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((RAW / "ust").glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns=UST_COLUMNS)[["date", *UST_COLUMNS.values()]]
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= asof)]
    return df.sort_values("date").drop_duplicates("date")


def freeze_ecb_spot(asof: date) -> pd.DataFrame:
    raw = pd.read_csv(RAW / "ecb" / "yc_aaa_spot.csv")
    raw["tenor"] = raw["KEY"].str.rsplit(".", n=1).str[-1].str.replace("SR_", "", regex=False)
    df = raw.pivot(index="TIME_PERIOD", columns="tenor", values="OBS_VALUE")
    order = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    df = df[order].reset_index().rename(columns={"TIME_PERIOD": "date"})
    df = df[pd.to_datetime(df["date"]).dt.date <= asof]
    return df.sort_values("date")


def freeze_estr(asof: date) -> pd.DataFrame:
    raw = pd.read_csv(RAW / "ecb" / "estr.csv")
    df = raw[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "rate"})
    return df[pd.to_datetime(df["date"]).dt.date <= asof].sort_values("date")


def freeze_bundesbank(asof: date) -> pd.DataFrame:
    frames = [parse_bundesbank_workbook(p) for p in sorted((RAW / "bundesbank").glob("*.xlsx"))]
    df = pd.concat(frames, ignore_index=True)
    return df[df["price_date"] <= asof].sort_values(["price_date", "maturity", "isin"])


def freeze_static(quotes: pd.DataFrame) -> pd.DataFrame:
    """Infer irregular first periods for every conventional security; cross-check Borsa Italiana."""
    borsa = {}
    for p in (RAW / "borsaitaliana").glob("*.json"):
        j = json.loads(p.read_text())
        borsa[j["isin"]] = j["interest_commencement"]
    rows = []
    conventional = quotes[~quotes["description"].astype(str).str.contains("I/L")]
    for isin, grp in conventional.groupby("isin"):
        obs = [
            Observation(add_business_days(r.price_date, 2), float(r.dirty - r.clean), float(r.dirty), float(r["yield"]))
            for _, r in grp.iterrows()
        ]
        first = grp.iloc[0]
        start, fc = infer_first_period(isin, float(first.coupon), first.maturity, obs)
        if start is None:
            continue
        check = borsa.get(isin, "")
        if check and check != start.isoformat():
            raise ValueError(f"{isin}: inferred start {start} disagrees with Borsa Italiana {check}")
        rows.append(
            {
                "isin": isin,
                "description": first.description,
                "interest_start": start.isoformat(),
                "first_coupon": fc.isoformat() if fc else "",
                "method": "inferred from Bundesbank accrued interest and yields",
                "borsaitaliana_interest_commencement": check,
            }
        )
    return pd.DataFrame(rows).sort_values("isin")


def freeze_eurex(asof: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for p in sorted((RAW / "eurex").glob("FGBL_*.json")):
        payload = json.loads(p.read_text())
        bd = datetime.strptime(p.stem.split("_")[1], "%Y%m%d").date()
        if bd > asof:
            continue
        for r in payload["dataRows"]:
            rows.append(
                {
                    "business_date": bd.isoformat(),
                    "contract": r["date"],
                    "settlement": round(r["dSettle"], 4),
                    "open": round(r["open"], 4),
                    "high": round(r["high"], 4),
                    "low": round(r["low"], 4),
                    "last": round(r["last"], 4),
                    "volume": int(r["volume"]),
                    "open_interest": int(r["openInt"]),
                }
            )
    daily = pd.DataFrame(rows).sort_values(["business_date", "contract"])
    text = (RAW / "eurex" / "deliverable-bonds-and-conversion-factors.csv").read_text()
    deliverable = []
    for row in csv.reader(io.StringIO(text), delimiter=";"):
        if not row or row[0].startswith("#"):
            continue
        product, _, code, _ = row[0].split()
        if product not in {"FGBL", "FGBM", "FGBS", "FGBX"}:
            continue
        d, m, y = row[3].split(".")
        deliverable.append(
            {
                "product": product,
                "contract": code,
                "isin": row[1],
                "coupon": float(row[2]),
                "maturity": f"{y}-{m}-{d}",
                "conversion_factor": float(row[4]),
            }
        )
    return daily, pd.DataFrame(deliverable).sort_values(["product", "contract", "isin"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", required=True, type=date.fromisoformat)
    parser.add_argument("--start", default="2023-01-01", type=date.fromisoformat)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_log = json.loads((RAW / "fetch_log.json").read_text())

    quotes = freeze_bundesbank(args.asof)
    daily, deliverable = freeze_eurex(args.asof)
    tables = {
        "ust_par_yields.csv": (
            freeze_ust(args.asof, args.start),
            "US Treasury daily par yield curve, percent, bond-equivalent (semiannual) basis",
        ),
        "ecb_aaa_spot.csv": (
            freeze_ecb_spot(args.asof),
            "ECB AAA euro area government spot rates, percent, "
            "Svensson model, continuous compounding (provider id SV_C_YM)",
        ),
        "estr.csv": (freeze_estr(args.asof), "Euro short-term rate, volume-weighted trimmed mean, percent, ACT/360"),
        "bundesbank_federal_securities.csv": (
            quotes,
            "Bundesbank reference prices (around noon, Frankfurt) "
            "and yields of listed Federal securities; clean and dirty price "
            "per 100, yield in percent",
        ),
        "bond_static.csv": (
            freeze_static(quotes),
            "Irregular first coupon periods inferred from the Bundesbank "
            "accrued interest and yields; commencement dates cross-checked against Borsa "
            "Italiana where available",
        ),
        "eurex_fgbl_daily.csv": (
            daily,
            "Eurex Euro-Bund futures daily statistics: daily settlement price, "
            "OHLC, volume, open interest; contract = last trading day code",
        ),
        "eurex_deliverable_bonds.csv": (
            deliverable,
            "Eurex deliverable bonds and published conversion factors (FGBS, FGBM, FGBL, FGBX)",
        ),
    }
    files = {}
    for name, (df, description) in tables.items():
        path = OUT / name
        df.to_csv(path, index=False, lineterminator="\n")
        files[name] = {"sha256": sha256(path), "rows": len(df), "description": description}
    manifest = {
        "asof": args.asof.isoformat(),
        "history_start": args.start.isoformat(),
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "fetched_utc": fetch_log["fetched_utc"],
        "files": files,
        "sources": [
            *sorted(
                {f"{e['source']}: {e.get('listing', e['url'])}" for e in fetch_log["files"] if "FGBL_" not in e["file"]}
            ),
            "Eurex: https://www.eurex.com/api/v1/overallstatistics/34656?busdate=YYYYMMDD&filtertype=overview",
        ],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, meta in files.items():
        print(f"{name:40s} {meta['rows']:6d} rows  {meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
