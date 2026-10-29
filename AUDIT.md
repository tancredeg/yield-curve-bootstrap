# AUDIT — yield-curve-bootstrap (Phase 0)

| | |
|---|---|
| Scope | Public `main` at `247592a` (2026-06-03): `yield_curve.py` (1409 lines), `test_yield_curve.py`, `fetch_market_data.py`, `README.md`, `requirements*.txt`, `data/`, `.vscode/` |
| Method | Full read of every file; pipeline and tests executed (Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, pytest 9.1.1); every suspected defect quantified by script |
| Pipeline | `python yield_curve.py` exits 0 |
| Tests | 22/22 pass; line coverage 42 % on `yield_curve.py`, 0 % on `fetch_market_data.py` |
| Reproduce the numbers below | `python scripts/audit/quantify_phase0.py` and `python scripts/audit/check_gapfill_claim.py` from the repo root |
| Severity scale | BLOCKER (false claim or wrong headline output) · MAJOR (correctness or design gap a rates reviewer will hit) · MINOR · NIT |

---

## 0. Preliminary finding — the public HEAD is the pre-audit version

Earlier working notes describe a later state of this repo: BCBS 368 shocks with exponential shapes and currency-specific sizes, a Nelson-Siegel τ grid on [0.5, 25] with a boundary warning, bootstrap verification split into two independent functions, data-driven commentary. **None of this is in `247592a`**, the only version a recruiter can see. Either that work exists locally and was never pushed, or it was never completed.

Action before Phase 1: if a later local version exists, push it and re-run this audit as a delta (about 30 min). Every finding below is otherwise open.

---

## 1. Findings

### BLOCKER

**B1 — Hard-coded narrative contradicts the data, in a figure and in the console output.**
- `yield_curve.py:1122-1126` annotates figure 04 with "USD inverted (rate cuts priced in)".
- `yield_curve.py:1378-1388` prints "the inversion reflects the market pricing multiple Fed cuts… peak funds rate of ~5.3%" and "EUR front end is modestly inverted".
- `yield_curve.py:1266` prints "The inverted USD curve…".
- Measured on the shipped data: USD 2Y-10Y zero slope **+43.9 bp**, EUR **+45.1 bp**; `shape_label()` itself returns "upward-sloping" and the same printed line contradicts itself. The EUR input zeros are strictly increasing in maturity.
- Section 8 header (`1014-1017`) still describes "early 2024".
- Fix (Phase 1): delete all narrative; every sentence generated from computed quantities or removed.

**B2 — README and code comments claim things the code does not do.**
| Claim | Location | Reality |
|---|---|---|
| "12Y/15Y gap-filler pillars, ~64 bp bias fixed" | README:104-106, `yield_curve.py:196-198` | No 12Y/15Y pillar exists (USD pillars: 1M, 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y). The gap is handled by the anchored self-consistent iteration (`283-305`), which is the right mechanism. On current data the naive flat-extrapolation bias it avoids is **+18.9 bp at 20Y** (−16.8 bp at 30Y), not ~64 bp. |
| "Four standard scenarios: parallel ±100 bp, bear steepener, bull flattener" | README:71-73 | Code: +100 bp only, bear steepener, bear flattener, inversion. |
| "IRRBB / Basel standard shocks" | README:4, 44, 63, 67; `yield_curve.py:859, 864` | Not BCBS 368: four ad-hoc scenarios, piecewise-linear, same sizes for USD and EUR. The standard has six scenarios, exponential shapes, currency-specific sizes, recalibrated by BCBS d578 (July 2024, in force 1 Jan 2026). |
| "verify_bootstrap builds a CubicSpline… independent of log-linear bootstrap" | `yield_curve.py:337-347` | The function calls `_loglinear_interp` (`355-356`), the bootstrap's own interpolator. It is a final-curve consistency check (useful, necessary), not an independent repricing. |
| "USD data from FRED" | `yield_curve.py:9, 48` | Data come from home.treasury.gov (`data/metadata.json`). |
| "11 tenors" | README:13 | `usd_yields.csv` has 13. |

**B3 — The scenario engine does not apply the declared shock.**
`apply_shock` (`875-894`) adds the shock at the pillars and re-fits a natural cubic spline, so between pillars the applied shock is a spline of a piecewise-linear vector and overshoots:

| Scenario | Declared max | Applied max | Worst error | Applied min |
|---|---|---|---|---|
| USD Bear Steepener | 50 bp | **58.3 bp** | +8.3 bp at 13.8Y | −0.5 bp |
| USD Bear Flattener | 50 bp | 50.5 bp | −8.3 bp at 13.8Y | **−8.3 bp** (rates fall in a "bear" scenario) |
| USD Inversion | 75 / −25 bp | 76.0 bp | **−16.5 bp** at 13.8Y | −41.5 bp |
| EUR Bear Steepener | 50 bp | 53.3 bp | +3.3 bp at 11.9Y | −0.5 bp |
| EUR Inversion | 75 / −25 bp | 76.0 bp | −6.7 bp at 11.9Y | −31.7 bp |

The overshoot is printed in the titles of figures 02/03 ("max |shock| = 58bp" for a 50 bp scenario). Every cash flow between pillars is revalued on the wrong curve. Fix: shock the continuous curve, `z_shocked(t) = z(t) + s(t)` with `s` evaluated at `t`, never re-interpolated.

### MAJOR

**M1 — Calibration curve ≠ pricing curve.** The bootstrap uses log-linear interpolation on discount factors; all pricing, risk and scenarios use a natural cubic spline on zero rates (`1217-1218`). The calibration instruments therefore do not reprice on the curve used for pricing:

| Par instrument | 2Y | 5Y | 7Y | 10Y | 20Y | 30Y |
|---|---|---|---|---|---|---|
| Repricing error on pricing curve | −0.04 bp | −0.12 bp | −0.14 bp | −0.17 bp | **−1.68 bp** | −0.11 bp |

`test_par_swap_rate_matches_input_on_cubic` checks 10Y only, with a 1 bp tolerance, which hides this. Fix: interpolation-aware bootstrap — for each interpolation method, solve all pillar values jointly so every instrument reprices to ~1e-10 under that same method.

**M2 — Swaps priced on government curves.** USD and EUR "swaps" are discounted and projected on the Treasury par curve and the ECB AAA government curve. The comment at `754-760` states this is standard for OIS swaps; it is not: OIS swaps are discounted on the OIS curve, and the swap spread to government yields is non-zero and itself a traded quantity. A rates interviewer will open on this. No free official source for SOFR or €STR OIS term quotes has been identified, so a correct OIS curve is not buildable from public data. Recommendation: remove swaps; position the repo on government cash bonds and bond futures (consistent with the Eurex extension).

**M3 — Natural cubic spline on zeros used as the "reference method".** It is non-local and overshoots: the USD zero curve reaches **5.156 % at 23.4Y**, 5.9 bp above both bracketing pillars (20Y 5.098 %, 30Y 5.035 %). The two other methods produce discontinuous forwards on the same pillars: linear-on-zero forwards jump by up to **141 bp** at a pillar; log-linear-on-DF forwards are piecewise flat, jumping **+90 bp at 10Y** and **−83 bp at 20Y**. Monotone convex (Hagan–West 2006), the method Treasury uses for its own par curve since 6 Dec 2021, is absent.

**M4 — The EUR leg has no calibration.** ECB AAA zeros are themselves a Svensson fit; they are ingested, and a Nelson-Siegel curve is then fitted on top of that parametric output. Figure 01 labels the EUR points "Bootstrapped pillars" (`550-551`). The brief's par-repricing test cannot be run on EUR as it stands.

**M5 — The bond engine has no dates.** Cash flows sit on integer multiples of the coupon period from t = 0, valuation is on a coupon date, no accrued interest, no clean/dirty split, no day count, no settlement lag. This blocks key-rate durations on real bonds and the entire futures extension (conversion factors, accrued at delivery, carry).

**M6 — Risk is limited to yield DV01 (`D_mod · P · 1e-4`) and a parallel swap DV01.** No curve DV01, no key-rate durations, no bucketed par DV01. The scenario P&L table mixes price points (bonds) and currency amounts (swaps) in one frame (`1342-1345`).

**M7 — Tests pass but test little.** Coverage 42 %; `verify_bootstrap`, the scenario engine, the comparison table and all plotting are never executed by tests. `test_bootstrap_reprices_at_par` replays the bootstrap's own interpolator (necessary, not sufficient). `test_nelson_siegel_tau_is_fitted_not_stuck` asserts that USD τ ≠ EUR τ, a property of the data, not an invariant. Integration tests read `data/`, which `fetch_market_data.py` overwrites: no frozen regression set.

**M8 — Not reproducible by construction.** `fetch_market_data.py` defaults to "latest available", overwrites `data/`, writes `_debug_*.txt` into `data/`, keeps a FRED fallback; dependencies are lower bounds only; no checksum on data. The library has side effects at import (`os.makedirs("figures")` line 33, global `rcParams` 35-42) and prints inside calculation functions (`175, 273-274, 310, 509-514`).

**M9 — Nelson-Siegel EUR fit sits on the grid boundary.** EUR τ = **10.000** = upper bound of `np.linspace(0.5, 10, 191)` (`496`). The reported fit is a corner solution; nothing flags it.

### MINOR

- **m1** `extrapolate=False` returns NaN that propagates silently: a 0.1Y EUR cash flow or any t > 30Y prices to NaN (`438`). Must raise.
- **m2** `day_count` arguments are accepted and never used (`147`, `248`). The deposit formula uses the year fraction directly, which is close to the bond-equivalent basis Treasury uses for bills; the "Act/360" label is wrong, the number is roughly right.
- **m3** `_loglinear_interp` returns DF(first pillar) for t below the first pillar (`228-229`): flat discount factor, not flat rate.
- **m4** The 25-iteration loop (`291-305`) accepts non-convergence silently.
- **m5** Runtime validation by `assert` (stripped under `python -O`); `assert swap_mats == sorted(swap_mats)` right after sorting is a tautology (`271`).
- **m6** YTM solved by `brentq` on [1e-4, 0.5] (`672`, `688`): fails for negative yields (EUR 2019-2021). Undocumented domain restriction.
- **m7** Inconsistent tolerances: 0.01 bp (`377`), 1e-6 (test `65`), 1e-5 (test `254`), 1e-4 = 1 bp (test `72`).
- **m8** The short-end dip of the USD-EUR spread near 1Y in figure 04 is a spline artefact through five closely spaced bill pillars, not an economic signal.

### NIT

- **n1** README in French (standard requires English); "bootstrap exact to machine precision" next to a 0.01 bp threshold.
- **n2** `.vscode/` committed; UTF-8 BOM in `.gitignore` and the `.vscode` JSON files.
- **n3** One 1409-line script; the zero-rate/spline construction is re-implemented at 12 sites (`540, 845, 887, 932, 977, 1031, 1037, 1041, 1097, 1101, 1205, 1207`).

### What holds and is kept

- Bond analytics: D_mac, D_mod and convexity under one yield basis, convention/frequency mismatch raises, verified by finite difference.
- Signed DV01 on the position's contractual fixed rate.
- Anchored self-consistent iteration across pillar gaps — better than synthetic gap-fillers.
- Nelson-Siegel by τ grid + OLS on the betas (sound; extend grid, report saturation).
- Data provenance in `metadata.json`.

---

## 2. Gap against the Repo 3 brief

| Brief | Current state | Gap |
|---|---|---|
| Public curves (ECB euro area, €STR) frozen in `data/sample/` | ECB AAA zeros + Treasury CMT in mutable `data/`; no €STR | Add €STR (ECB dataset `EST`, series `EST.B.EU000A2X2A25.WT`); freeze with checksums. €STR enters as overnight financing rate (carry, implied repo comparison), not as a term curve (see M2). |
| Par repricing of calibration instruments ≈ 0 | USD: yes, under the bootstrap interpolator only; EUR: impossible | Interpolation-aware bootstrap (M1); EUR calibration decision (D3). |
| Linear on zeros / log-linear on DF / monotone convex; positivity and regularity of forwards | Linear and log-linear exist (log-linear only inside the bootstrap); monotone convex absent; no forward tests | Implement Hagan–West; property tests. |
| R1 instantaneous forwards by interpolation | Absent | New figure + table. |
| R2 DV01 and KRD of a bond portfolio | Yield DV01 only | Dated engine (M5), KRD, bucketed par DV01. |
| R3 PCA of curve changes | Single date only | Daily history panel (Treasury yearly CSV, ECB YC API). |
| R4 Bond futures: basket, CF, CTD, net basis, implied repo | Absent | Dated engine, Eurex basket and CF list, Bund prices, futures settlement, €STR. |

---

## 3. Decisions (taken 2026-10-28, autonomous session, recommended option in each case)

| # | Decision | Taken | Why |
|---|---|---|---|
| D1 | Unpushed later version? | **HEAD `247592a` is the baseline.** No later commit exists on the remote, so the audit applies in full. | The public repo is what a recruiter sees. |
| D2 | Swaps | **Removed.** | No free OIS term quotes: swaps priced on government curves were indefensible (M2). |
| D3 | EUR calibration | **(B) Exact fit to Bundesbank-priced Bunds**, with a selection rule fixed in advance; the ECB AAA curve is kept only for the EUR PCA. | Makes the EUR repricing test meaningful and gives R2 and R4 real, dated bonds. |
| D4 | Scenarios | **Kept as a secondary module** (SRP31 shapes, July 2024 sizes), applied continuously in maturity; used for the R2 P&L columns only. | Not in the brief; correct implementation fixes B3. |
| D5 | As-of date | **27 October 2026** for every snapshot (Treasury, Bundesbank, Eurex settlement, €STR); histories from January 2023. | Latest date with all sources published when the sample was frozen. |
| D6 | USD | **Kept**: Treasury par curve calibration and USD PCA. | Cleanest par-repricing demonstration. |

---

## 4. Plan — Phases 1 to 6, adapted to this repo

Estimate: 10-12 sessions, 25-35 h, i.e. 4-5 weeks at 6-8 h/week. Estimate, not a commitment; R4 carries most of the uncertainty (data access).

**Phase 1 — Structure (1 session)**
- `src/ratescurve/`: `conventions.py` (day counts, schedules, Act/Act ICMA), `instruments.py`, `curve.py` (`df`, `zero`, `fwd` on one object), `interpolation.py`, `calibration.py` (global solve), `bonds.py`, `risk.py`, `pca.py`, `futures.py`, `scenarios.py`. Plotting and commentary live in `scripts/`, never in the library.
- `scripts/fetch_data.py --asof YYYY-MM-DD` → `data/raw/` (git-ignored); `scripts/freeze_sample.py` → `data/sample/` + `MANIFEST.json` (source URL, as-of date, fetch timestamp, sha256).
- Pure functions, type hints, docstrings with formula and reference; `raise` instead of `assert` or NaN.
- `pyproject.toml` with exact pins, `pip install -e ".[dev]"`.
- Apply D2; delete narrative (B1); remove `.vscode/`, BOMs, FRED fallback.
- Done when: install works, existing invariant tests ported and green.

**Phase 2 — Tests (2 sessions)**
- Calibration: every instrument reprices under every interpolation method, |error| < 1e-8 (price per 100 or par rate in bp, fixed per instrument type).
- Interpolation properties, each asserted as documented: positive forwards for monotone convex on positive discrete forwards; continuous forwards for monotone convex, piecewise constant for log-linear DF, discontinuous for linear zero; locality (bump one pillar, measure response outside neighbours); no overshoot between pillars for monotone convex.
- Independent oracle: QuantLib (dev dependency only) for monotone convex and log-linear curves on the same inputs.
- Identities: DF(0) = 1; `z·t = −ln DF`; integral of forwards = `−ln DF`; par-rate formula.
- Risk: sum of KRD vs parallel DV01; sum of bucketed par DV01 vs parallel par DV01; analytic vs finite difference.
- Scenarios: applied shock equals declared function at every t (B3).
- PCA: eigenvalues ≥ 0, sum = trace, orthonormal loadings, fixed sign convention.
- Futures: computed CF equals the Eurex published CF for every bond in the basket at published precision; accrued checked against Bundesbank data; implied repo and net basis identities; CTD by max implied repo = CTD by min net basis.
- Regression: golden JSON of key outputs on `data/sample/`.
- Coverage ≥ 80 % on `src/`.

**Phase 3 — Quality and CI (1 short session)**
- ruff (lint + format), mypy on `src/`; GitHub Actions on push: ubuntu + windows (local environment is Windows), badge.

**Phase 4 — Results (4 sessions, one per result) — `scripts/run_results.py` regenerates everything**
- **R1 Forwards by interpolation.** Same instruments, all repricing to 0, forwards from linear zero, log-linear DF, monotone convex, natural cubic as counter-example. Table: max forward jump, min forward, value of a 10Y-into-10Y forward-starting par instrument per method. Expected message, to be confirmed by the numbers: zero curves nearly coincide, forwards differ by tens of bp.
- **R2 DV01 and KRD.** Portfolio of 5-8 real Bunds by ISIN (D3-B) plus USD par bonds; KRD at {2, 5, 10, 20, 30}Y with triangular zero bumps (Ho 1992); bucketed par DV01 by bumping each input and re-calibrating; reconciliation with parallel DV01.
- **R3 PCA.** Daily changes in bp over at least two years; Treasury par panel, ECB panel shown with caveat; variance share of PC1-PC3; loadings figure; stability across two sub-periods. Caveat stated: the ECB curve lives in a six-parameter Svensson family, so its factor concentration is partly mechanical; Treasury par yields are also smoothed (monotone convex on on-the-run inputs).
- **R4 Euro-Bund futures.** Front FGBL at the as-of date: deliverable basket (Eurex list), conversion factors (computed, checked against Eurex), gross basis, carry with €STR as general-collateral proxy, net basis, implied repo, CTD; CTD switch under parallel yield shifts. Limitations stated: delivery options not valued (net basis bundles option value and repo specialness); specific repo not observed.

**Phase 5 — README (English, imposed order)**: one-line problem + method; key results table + 1-2 figures; methodology and stated limits; 2-command reproduction; code structure; references (Hagan–West 2006 and 2008; Ho 1992; Litterman–Scheinkman 1991; Eurex CF methodology; Treasury yield-curve methodology; BCBS d368 and d578).

**Phase 6 — Delivery**: demo notebook under 2 min end to end; tag v1.0 with release notes; pinned on profile; 5-line LinkedIn draft.

---

## 5. Data sources (checked 2026-10-13, updated 2026-10-29)

| Data | Source | Status |
|---|---|---|
| USD par curve | home.treasury.gov daily par yield curve | Working. Monotone convex since 6 Dec 2021; inputs are bid-side indicative quotes on the most recently auctioned securities, around 3:30 pm. |
| EUR AAA zeros | ECB Data API, dataset `YC`, `B.U2.EUR.4F.G_N_A.SV_C_YM.*` | Working. |
| €STR | ECB Data Portal, dataset `EST`, `EST.B.EU000A2X2A25.WT` | Fetched and frozen from January 2023. |
| Bund prices | Bundesbank, "Prices and yields of listed Federal securities": daily reference prices per ISIN, monthly XLSX/PDF files | One sheet per trading day: ISIN, coupon, description, maturity, residual life, issue volume, clean price, yield, dirty price; parsed and frozen for October 2026. |
| FGBL basket and CF | Eurex, "Deliverable Bonds and Conversion Factors"; Eurex CF calculation methodology (PDF) | Exists. |
| FGBL settlement price | Eurex statistics API (`/api/v1/overallstatistics/34656`) | Daily settlement, OHLC, volume and open interest per contract; October 2026 frozen. |
| IRRBB sizes | BCBS d368 (2016); d578 recalibration (July 2024, in force 1 Jan 2026) | Confirmed against Basel Framework SRP31 (version in force 1 Jan 2026): EUR 225/350/200 bp, USD 200/300/225 bp (parallel/short/long); SRP31.92 worked example used as a unit test. |
| SOFR / €STR OIS term quotes | — | No free official source identified (assumption). |

---

## 6. Resolution (status on 2026-10-29)

Every finding was re-checked against the code after the rebuild. Commits: `6a3c208` src layout and removal of the legacy script; `700823f` core (dated bonds, four schemes, global calibration); `213f283` analytics (risk, PCA, futures, scenarios); `1d2971f` frozen data; `4e74926` tests; `7f357c3` CI; `05dbfa6` Windows checksum fix; `c0b7972` data wording; `e97e987` test hardening; `9b08525` results and README; `34a514e` results and notebook in CI.

| Finding | Status | Where |
|---|---|---|
| B1 hard-coded narrative | Fixed: no narrative in the library; every sentence of README and `docs/results.md` is generated from computed values, and data-dependent claims are checked by `ensure()` in `scripts/run_results.py`, which fails rather than print an unsupported claim | `9b08525` |
| B2 false claims in README and comments | Fixed: README rewritten; each claim is either an input (convention, rule) or a generated number; the CI checks that published numbers equal the script output | `9b08525`, `34a514e` |
| B3 scenario shock not applied as declared | Fixed: `ShiftedCurve` applies `z(t) + s(t)` pointwise, never re-interpolated; test asserts applied = declared shock on a 300-point grid for all six SRP31 scenarios | `700823f`, `4e74926` |
| M1 calibration curve ≠ pricing curve | Fixed: one global calibration per scheme; the calibrated curve is the pricing curve; repricing error below 1e-11 bp for every instrument and scheme | `700823f`, `1d2971f`, `4e74926` |
| M2 swaps on government curves | Fixed by removal (D2) | `6a3c208` |
| M3 natural spline as reference, no monotone convex | Fixed: Hagan-West monotone convex with the positivity amendment is the default, validated against tf-quant-finance and by property tests; the spline is kept as a counter-example | `700823f`, `4e74926` |
| M4 EUR leg not calibrated | Fixed: exact fit to 15 Bunds (D3) with dated cash flows; 49 out-of-sample bonds repriced | `700823f`, `1d2971f` |
| M5 no dates in the bond engine | Fixed: TARGET2 calendar, T+2, ACT/ACT ICMA, short and long first coupons inferred from published accrued interest; 1,459 Bundesbank accrued amounts and yields reproduced | `700823f`, `1d2971f` |
| M6 risk limited to yield DV01 | Fixed: parallel, key-rate (Ho) and bucketed DV01; key-rate sum and bucket additivity tested | `213f283`, `e97e987` |
| M7 tests test little | Fixed: 92 tests, coverage above 95% on `src/`, external oracles (Bundesbank, Eurex, QuantLib, tf-quant-finance), golden regression on the frozen sample | `4e74926`, `e97e987` |
| M8 not reproducible | Fixed: dated fetch into git-ignored `data/raw/`, frozen `data/sample/` with SHA-256 manifest, exact pins, no import side effects or prints in the library; CI on Linux and Windows | `6a3c208`, `1d2971f`, `7f357c3`, `05dbfa6` |
| M9 Nelson-Siegel on the grid boundary | Closed: Nelson-Siegel is no longer used; monotone convex replaces it | `6a3c208` |
| m1 silent NaN outside the curve | Fixed: out-of-domain times raise `ValueError` | `700823f` |
| m2 unused day-count arguments | Fixed: day counts are explicit functions, used where named | `700823f` |
| m3 flat discount factor below the first pillar | Fixed: flat zero rate below the first pillar (monotone convex extrapolates the forward to t = 0) | `700823f` |
| m4 silent non-convergence | Fixed: `CalibrationError` if any residual exceeds tolerance | `700823f` |
| m5 runtime `assert` | Fixed: no `assert` in `src/` | `700823f` |
| m6 YTM solver excludes negative yields | Fixed: bracket [-50%, 100%], negative yields tested | `700823f`, `4e74926` |
| m7 inconsistent tolerances | Fixed: one relative calibration tolerance (1e-10); test tolerances stated with their units | `700823f`, `4e74926` |
| m8 spread artefact in figure 04 | Closed: figure removed | `6a3c208` |
| n1 README in French | Fixed: English README in the standard order | `9b08525` |
| n2 `.vscode/`, BOMs | Fixed: removed | `6a3c208` |
| n3 single 1409-line script, duplicated curve code | Fixed: 13-module package, one curve object | `6a3c208` |

Found during the rebuild and handled:

- Windows CI failed the manifest checksum because git rewrote CSV line endings; `.gitattributes` marks the frozen data as binary (`05dbfa6`).
- The first bucketed-DV01 test used a calibration bond and could not fail; replaced by an off-pillar test (`e97e987`).
- Monotone convex is only piecewise smooth in its quotes: with 1 bp bumps its bucket DV01s overstate the joint sensitivity by about 3% on the Bund curve; published bucketed risk uses 0.01 bp bumps and a test documents the effect (`e97e987`, `9b08525`).
- An unverified "around noon" fixing time for Bundesbank prices was removed from the data manifest (`c0b7972`).
