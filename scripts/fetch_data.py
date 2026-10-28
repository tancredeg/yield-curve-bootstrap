"""Download the raw public data used by the project into ``data/raw/``.

Sources (all public, no key):

* US Treasury daily par yield curve, yearly CSV files (home.treasury.gov).
* ECB Data API: AAA euro area government spot rates (dataset ``YC``,
  provider ``SV_C_YM`` = Svensson, continuous compounding) and the euro
  short-term rate (dataset ``EST``, series ``EST.B.EU000A2X2A25.WT``).
* Deutsche Bundesbank: daily reference prices and yields of listed Federal
  securities (one XLSX workbook per month, one sheet per trading day).
* Eurex: deliverable bonds and conversion factors (CSV) and Euro-Bund
  futures (FGBL) daily statistics, including the daily settlement price.
* Borsa Italiana bond pages: interest commencement date of the bonds whose
  first coupon is still to be paid (validation source only).

``data/raw/`` is git-ignored. ``scripts/freeze_sample.py`` turns it into the
versioned, checksummed snapshot in ``data/sample/``.

Usage::

    python scripts/fetch_data.py --asof 2026-10-27 --start 2023-01-01
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "yield-curve-bootstrap research script (github.com/tancredeg/yield-curve-bootstrap)"

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata&startPeriod={start}&endPeriod={end}&detail=dataonly"
ECB_SPOT_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
BUNDESBANK_LIST = (
    "https://www.bundesbank.de/dynamic/action/en/service/federal-securities/"
    "prices-and-yields/810710/prices-and-yields-of-listed-federal-securities"
)
EUREX_BONDS_PAGE = "https://www.eurex.com/ex-en/data/clearing-files/notified-deliverable-bonds-conversion-factors"
EUREX_STATS = "https://www.eurex.com/api/v1/overallstatistics/{product_id}?busdate={busdate}&filtertype=overview"
EUREX_PRODUCTS = {"FGBL": "34656"}
BORSA_ITALIANA = "https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/{isin}-MOTX.html?lang=en"
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def get(url: str, timeout: int = 90, retries: int = 2) -> bytes:
    """HTTP GET with a small retry loop; raises on final failure."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return bytes(resp.read())
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries + 1} attempts: {url}") from last


def fetch_treasury(out: Path, start: date, asof: date, log: list[dict]) -> None:
    for year in range(start.year, asof.year + 1):
        url = TREASURY_URL.format(year=year)
        raw = get(url)
        if not raw.startswith(b"Date"):
            raise RuntimeError(f"Unexpected Treasury payload for {year}")
        (out / f"{year}.csv").write_bytes(raw)
        log.append({"source": "US Treasury", "url": url, "file": f"ust/{year}.csv"})


def fetch_ecb(out: Path, start: date, asof: date, log: list[dict]) -> None:
    key = "B.U2.EUR.4F.G_N_A.SV_C_YM." + "+".join(f"SR_{t}" for t in ECB_SPOT_TENORS)
    url = ECB_URL.format(flow="YC", key=key, start=start.isoformat(), end=asof.isoformat())
    (out / "yc_aaa_spot.csv").write_bytes(get(url))
    log.append({"source": "ECB Data API", "url": url, "file": "ecb/yc_aaa_spot.csv"})
    url = ECB_URL.format(flow="EST", key="B.EU000A2X2A25.WT", start=start.isoformat(), end=asof.isoformat())
    (out / "estr.csv").write_bytes(get(url))
    log.append({"source": "ECB Data API", "url": url, "file": "ecb/estr.csv"})


def fetch_bundesbank(out: Path, asof: date, log: list[dict]) -> None:
    html = get(BUNDESBANK_LIST).decode("utf-8", errors="replace")
    stem = f"{asof.year}-{asof.month:02d}-excel-data.xlsx"
    links = sorted(set(re.findall(r'href="(/resource/blob/[^"]+' + re.escape(stem) + ')"', html)))
    if len(links) != 1:
        raise RuntimeError(f"Expected one Bundesbank workbook for {stem}, found {links}")
    url = "https://www.bundesbank.de" + links[0]
    (out / f"{asof.year}-{asof.month:02d}.xlsx").write_bytes(get(url))
    log.append(
        {
            "source": "Deutsche Bundesbank",
            "url": url,
            "file": f"bundesbank/{asof.year}-{asof.month:02d}.xlsx",
            "listing": BUNDESBANK_LIST,
        }
    )


def fetch_eurex(out: Path, asof: date, log: list[dict]) -> list[str]:
    page = get(EUREX_BONDS_PAGE).decode("utf-8", errors="replace")
    links = sorted(set(re.findall(r'href="(/resource/blob/[^"]+deliverable-bonds-and-conversion-factors\.csv)"', page)))
    if len(links) != 1:
        raise RuntimeError(f"Expected one Eurex deliverable-bonds CSV link, found {links}")
    url = "https://www.eurex.com" + links[0]
    raw = get(url)
    (out / "deliverable-bonds-and-conversion-factors.csv").write_bytes(raw)
    log.append(
        {
            "source": "Eurex",
            "url": url,
            "file": "eurex/deliverable-bonds-and-conversion-factors.csv",
            "listing": EUREX_BONDS_PAGE,
        }
    )
    basket_isins = sorted(
        {row[1] for row in csv.reader(io.StringIO(raw.decode()), delimiter=";") if row and row[0].startswith("FGBL")}
    )
    for code, pid in EUREX_PRODUCTS.items():
        url = EUREX_STATS.format(product_id=pid, busdate=asof.strftime("%Y%m%d"))
        first = json.loads(get(url))
        dates = sorted(datetime.strptime(d[:10], "%d-%m-%Y").date() for d in first["header"]["tradingDates"])
        for d in dates:
            if d > asof:
                continue
            url_d = EUREX_STATS.format(product_id=pid, busdate=d.strftime("%Y%m%d"))
            payload = json.loads(get(url_d))
            if payload["meta"]["productCode"] != code:
                raise RuntimeError(f"Eurex product mismatch for {code} on {d}")
            (out / f"{code}_{d:%Y%m%d}.json").write_text(json.dumps(payload, indent=1))
            log.append({"source": "Eurex", "url": url_d, "file": f"eurex/{code}_{d:%Y%m%d}.json"})
    return basket_isins


def fetch_borsa_italiana(out: Path, isins: list[str], log: list[dict]) -> None:
    for isin in isins:
        url = BORSA_ITALIANA.format(isin=isin)
        text = get(url).decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", "", text, flags=re.S))
        match = re.search(r"Interest Commencement Date\s+(\d{2})/(\d{2})/(\d{2})", re.sub(r"\s+", " ", text))
        if not match:
            raise RuntimeError(f"No interest commencement date on Borsa Italiana page for {isin}")
        yy, mm, dd = match.groups()
        (out / f"{isin}.json").write_text(json.dumps({"isin": isin, "interest_commencement": f"20{yy}-{mm}-{dd}"}))
        log.append({"source": "Borsa Italiana", "url": url, "file": f"borsaitaliana/{isin}.json"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--asof", required=True, type=date.fromisoformat)
    parser.add_argument("--start", default="2023-01-01", type=date.fromisoformat)
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()
    out = Path(args.out)
    dirs = {name: out / name for name in ("ust", "ecb", "bundesbank", "eurex", "borsaitaliana")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    fetch_treasury(dirs["ust"], args.start, args.asof, log)
    fetch_ecb(dirs["ecb"], args.start, args.asof, log)
    fetch_bundesbank(dirs["bundesbank"], args.asof, log)
    basket = fetch_eurex(dirs["eurex"], args.asof, log)
    fetch_borsa_italiana(dirs["borsaitaliana"], basket, log)
    meta = {
        "asof": args.asof.isoformat(),
        "start": args.start.isoformat(),
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": log,
    }
    (out / "fetch_log.json").write_text(json.dumps(meta, indent=2))
    print(f"Fetched {len(log)} files into {out}")


if __name__ == "__main__":
    main()
