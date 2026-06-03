"""
Fetch Real Market Data — US Treasury + ECB (with diagnostic output)
====================================================================

USD: US Treasury daily par yield curve (home.treasury.gov)
     Tries multiple URL formats (CSV + XML, several parameter variants).
     Fallback: FRED per series. Manual fallback printed if all fail.
EUR: ECB AAA-rated euro area government zero-coupon rates (Svensson).

Usage:
    python fetch_market_data.py                 # latest available
    python fetch_market_data.py 2024-01-15      # specific date
"""

import csv
import io
import json
import os
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

DATA_DIR = "data"

# Treasury maturity field map — XML/CSV column → maturity in years.
# Includes legacy "BC_*" XML tags and current CSV header strings.
TREASURY_FIELD_MAT = {
    "BC_1MONTH": 1/12, "BC_2MONTH": 2/12, "BC_3MONTH": 0.25,
    "BC_4MONTH": 4/12, "BC_6MONTH": 0.5,  "BC_1YEAR":  1.0,
    "BC_2YEAR":  2.0,  "BC_3YEAR":  3.0,  "BC_5YEAR":  5.0,
    "BC_7YEAR":  7.0,  "BC_10YEAR": 10.0, "BC_20YEAR": 20.0,
    "BC_30YEAR": 30.0,
}
CSV_HEADER_MAT = {
    "1 mo": 1/12, "2 mo": 2/12, "3 mo": 0.25, "4 mo": 4/12, "6 mo": 0.5,
    "1 yr": 1.0,  "2 yr": 2.0,  "3 yr": 3.0,  "5 yr": 5.0,  "7 yr": 7.0,
    "10 yr": 10.0,"20 yr": 20.0,"30 yr": 30.0,
}
FRED_SERIES = [
    ("DGS1MO", 1/12), ("DGS3MO", 0.25), ("DGS6MO", 0.5),
    ("DGS1", 1.0), ("DGS2", 2.0), ("DGS3", 3.0), ("DGS5", 5.0),
    ("DGS7", 7.0), ("DGS10", 10.0), ("DGS20", 20.0), ("DGS30", 30.0),
]


def _get(url, timeout=25):
    req = urllib.request.Request(
        url, headers={
            "User-Agent": "Mozilla/5.0 yield-curve-project/1.0",
            "Accept": "*/*",
        })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _save_debug(name, content):
    path = os.path.join(DATA_DIR, f"_debug_{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ── Treasury parsers ────────────────────────────────────────

def _parse_treasury_csv(raw, target_date=None):
    """Treasury daily-treasury-rates.csv has header: Date, "1 Mo", "2 Mo", ..."""
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = [(f or "").strip() for f in (reader.fieldnames or [])]
    if "Date" not in fieldnames:
        return None, None
    # Map header → maturity (case-insensitive, lenient on spaces)
    header_map = {}
    for col in fieldnames:
        key = col.strip().lower().replace("  ", " ")
        if key in CSV_HEADER_MAT:
            header_map[col] = CSV_HEADER_MAT[key]
    if len(header_map) < 8:
        return None, None
    rows = list(reader)
    if not rows:
        return None, None
    # Treasury CSV: most recent first, so just take first row with enough data.
    for row in rows:
        date = (row.get("Date") or "").strip()
        # Treasury date format: "MM/DD/YYYY" → normalize to YYYY-MM-DD
        try:
            dt = datetime.strptime(date, "%m/%d/%Y")
        except ValueError:
            continue
        iso = dt.strftime("%Y-%m-%d")
        if target_date and iso > target_date:
            continue
        vals = {}
        for col, mat in header_map.items():
            v = (row.get(col) or "").strip()
            if v and v not in ("N/A", "-"):
                try:
                    vals[mat] = float(v)
                except ValueError:
                    pass
        if len(vals) >= 8:
            return iso, vals
    return None, None


def _parse_treasury_xml(raw):
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None, None
    entries = {}
    # Walk every element; collect entries grouped by date
    current_date = None
    current_vals = {}
    def local(t): return t.split("}")[-1] if "}" in t else t
    for el in root.iter():
        name = local(el.tag)
        if name == "entry":
            current_date = None
            current_vals = {}
            for child in el.iter():
                cname = local(child.tag)
                if cname in ("NEW_DATE", "INDEX_DATE") and child.text:
                    current_date = child.text[:10]
                elif cname in TREASURY_FIELD_MAT and child.text:
                    try:
                        current_vals[TREASURY_FIELD_MAT[cname]] = float(child.text)
                    except ValueError:
                        pass
            if current_date and len(current_vals) >= 8:
                entries[current_date] = current_vals
    if not entries:
        return None, None
    best = max(entries)
    return best, entries[best]


def fetch_usd_treasury(target_date=None):
    """Try multiple Treasury URL formats. Print diagnostics on failure."""
    now = datetime.now()
    year  = (target_date[:4] if target_date else now.strftime("%Y"))
    yearm = (target_date[:7].replace("-", "") if target_date else now.strftime("%Y%m"))
    prev_yearm = (now.replace(day=1) - timedelta(days=1)).strftime("%Y%m")

    attempts = [
        ("CSV year", f"https://home.treasury.gov/resource-center/data-chart-center/"
                     f"interest-rates/daily-treasury-rates.csv/{year}/all"
                     f"?type=daily_treasury_yield_curve&"
                     f"field_tdr_date_value_year={year}&page&_format=csv"),
        ("CSV month", f"https://home.treasury.gov/resource-center/data-chart-center/"
                      f"interest-rates/daily-treasury-rates.csv/all/{yearm}"
                      f"?type=daily_treasury_yield_curve&"
                      f"field_tdr_date_value_month={yearm}&page&_format=csv"),
        ("XML v1", f"https://home.treasury.gov/resource-center/data-chart-center/"
                   f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
                   f"&field_tdr_date_value_month={yearm}"),
        ("XML v2", f"https://home.treasury.gov/resource-center/data-chart-center/"
                   f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
                   f"&field_tdr_date_value={yearm}"),
        ("XML prev month v1", f"https://home.treasury.gov/resource-center/data-chart-center/"
                              f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
                              f"&field_tdr_date_value_month={prev_yearm}"),
    ]

    for label, url in attempts:
        print(f"\n  [{label}] {url[:90]}...")
        try:
            raw = _get(url)
        except Exception as e:
            print(f"    network error: {type(e).__name__}: {e}")
            continue
        print(f"    response: {len(raw)} bytes")
        snippet = raw[:200].replace("\n", " | ")
        print(f"    snippet : {snippet!r}")

        # Try CSV first (faster, simpler)
        if "Date" in raw[:200] or "date" in raw[:200].lower():
            date, vals = _parse_treasury_csv(raw, target_date)
            if vals:
                print(f"    [OK] CSV parsed: {len(vals)} tenors for {date}")
                return date, vals
        # Try XML
        if raw.lstrip().startswith("<"):
            date, vals = _parse_treasury_xml(raw)
            if vals:
                print(f"    [OK] XML parsed: {len(vals)} tenors for {date}")
                return date, vals
        # Save for inspection
        path = _save_debug(label.replace(" ", "_"), raw)
        print(f"    [no parse] response saved to {path}")

    print("\n  All Treasury.gov attempts failed.")
    return None, None


def fetch_fred_usd_fallback(target_date=None):
    from collections import Counter
    print("\n  FRED fallback (per series, may timeout on restricted networks)...")
    vals = {}; dates = {}
    for series_id, mat in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        print(f"    {series_id:8}...", end="", flush=True)
        try:
            raw = _get(url, timeout=10)
        except Exception as e:
            print(f"  [{type(e).__name__}]"); continue
        for row in reversed(list(csv.DictReader(io.StringIO(raw)))):
            date = row.get("DATE", ""); v = row.get(series_id, ".").strip()
            if v and v != "." and (target_date is None or date <= target_date):
                vals[mat] = float(v); dates[series_id] = date
                print(f"  {date}={v}%"); break
        else:
            print("  [no data]")
    if not vals:
        return None, None
    best = Counter(dates.values()).most_common(1)[0][0]
    return best, vals


def fetch_usd(target_date=None):
    date, vals = fetch_usd_treasury(target_date)
    if vals:
        return date, vals, "US Treasury CMT (home.treasury.gov)"
    date, vals = fetch_fred_usd_fallback(target_date)
    if vals:
        return date, vals, "FRED Treasury CMT (DGS series)"
    return None, None, None


def print_manual_fallback():
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  MANUAL FALLBACK — USD Treasury yields                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │  1. Open in browser:                                                │
  │     https://home.treasury.gov/resource-center/data-chart-center/    │
  │     interest-rates/TextView?type=daily_treasury_yield_curve         │
  │                                                                     │
  │  2. Pick the latest business day with all columns filled.           │
  │                                                                     │
  │  3. Create data/usd_yields.csv with this content                    │
  │     (replace rates with what you see; rates in percent):            │
  │                                                                     │
  │     maturity,rate                                                   │
  │     0.0833,5.50                                                     │
  │     0.25,5.40                                                       │
  │     0.5,5.20                                                        │
  │     1.0,4.80                                                        │
  │     2.0,4.40                                                        │
  │     3.0,4.20                                                        │
  │     5.0,4.00                                                        │
  │     7.0,4.00                                                        │
  │     10.0,4.00                                                       │
  │     20.0,4.20                                                       │
  │     30.0,4.10                                                       │
  │                                                                     │
  │  4. Re-run: python yield_curve.py                                   │
  └─────────────────────────────────────────────────────────────────────┘
""")


# ── ECB (unchanged — works) ──────────────────────────────────

ECB_MATURITIES = [
    ("SR_3M",  0.25), ("SR_6M",  0.5),  ("SR_1Y",  1.0),
    ("SR_2Y",  2.0),  ("SR_3Y",  3.0),  ("SR_5Y",  5.0),
    ("SR_7Y",  7.0),  ("SR_10Y", 10.0), ("SR_15Y", 15.0),
    ("SR_20Y", 20.0), ("SR_30Y", 30.0),
]
ECB_MATURITIES_ALT = [
    ("SR_0Y3M", 0.25), ("SR_0Y6M", 0.5),  ("SR_1Y0M", 1.0),
    ("SR_2Y0M", 2.0),  ("SR_3Y0M", 3.0),  ("SR_5Y0M", 5.0),
    ("SR_7Y0M", 7.0),  ("SR_10Y0M", 10.0),("SR_15Y0M", 15.0),
    ("SR_20Y0M", 20.0),("SR_30Y0M", 30.0),
]


def _try_ecb_fetch(mat_list, target_date=None):
    from collections import defaultdict
    code_to_mat = {c: m for c, m in mat_list}
    keys = "+".join(c for c, _ in mat_list)
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    period = (f"&startPeriod={target_date}&endPeriod={target_date}"
              if target_date else f"&startPeriod={start}")
    url = (f"https://data-api.ecb.europa.eu/service/data/"
           f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{keys}"
           f"?format=csvdata{period}&detail=dataonly")
    raw = _get(url)
    by_date = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(raw)):
        date = row.get("TIME_PERIOD", ""); key_full = row.get("KEY", "")
        val = row.get("OBS_VALUE", "")
        if not val: continue
        mat_code = key_full.rsplit(".", 1)[-1] if key_full else ""
        if mat_code in code_to_mat:
            by_date[date][code_to_mat[mat_code]] = float(val)
    if not by_date: return None, None
    best = max(by_date)
    return (best, by_date[best]) if len(by_date[best]) >= 6 else (None, None)


def fetch_ecb_eur(target_date=None):
    for label, mat_list in [("primary", ECB_MATURITIES), ("alt", ECB_MATURITIES_ALT)]:
        try:
            date, vals = _try_ecb_fetch(mat_list, target_date)
            if vals:
                print(f"  Found {len(vals)} tenors for {date}")
                return date, vals
        except Exception as e:
            print(f"  {label} codes: {type(e).__name__}: {e}")
    return None, None


def save_csv(filepath, data_dict):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["maturity", "rate"])
        for mat in sorted(data_dict):
            label = f"{mat:.4f}" if mat < 1 else f"{mat:.1f}"
            w.writerow([label, f"{data_dict[mat]:.4f}"])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Target date: {target or 'latest available'}")

    print("\n[1/2] USD — US Treasury daily par yield curve")
    usd_date, usd_data, usd_source = fetch_usd(target)
    if usd_data:
        save_csv(os.path.join(DATA_DIR, "usd_yields.csv"), usd_data)
        print(f"\n  Saved {len(usd_data)} tenors to data/usd_yields.csv "
              f"(source: {usd_source})")
    else:
        print_manual_fallback()

    print("\n[2/2] EUR — ECB AAA Govt zero-coupon rates")
    eur_date, eur_data = fetch_ecb_eur(target)
    if eur_data:
        save_csv(os.path.join(DATA_DIR, "eur_zeros.csv"), eur_data)
        print(f"  Saved {len(eur_data)} tenors to data/eur_zeros.csv")
    else:
        print("  EUR data NOT saved.")

    meta = {
        "usd_source": usd_source or "unknown",
        "usd_date":   usd_date,
        "eur_source": "ECB AAA Govt zero-coupon (Svensson)",
        "eur_date":   eur_date,
        "fetch_timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if usd_data and eur_data:
        print("\n[OK] Both curves fetched. Run: python yield_curve.py")
    else:
        print("\n[PARTIAL] See errors above. data/_debug_*.txt may contain "
              "the raw responses for inspection.")


if __name__ == "__main__":
    main()
