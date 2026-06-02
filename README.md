# Yield Curve Construction and Rates Pricing — USD vs EUR

Bootstrap, interpolate, and price interest rate instruments on USD (SOFR) and EUR (ESTR) discount curves. Implements the complete quant workflow from market quote ingestion to scenario P&L — with explicit treatment of four common implementation failure modes.

---

## Repository Structure

```
yield_curve_project/
├── yield_curve.py          # Single executable script (all sections)
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
└── figures/
    ├── 01_zero_curves_interpolation.png   # Linear / Cubic Spline / Nelson-Siegel
    ├── 02_scenario_analysis_usd.png       # USD: 4 shock scenarios
    ├── 03_scenario_analysis_eur.png       # EUR: 4 shock scenarios
    └── 04_usd_eur_comparison.png          # Cross-currency zero curve + spread
```

---

## Installation and Execution

```bash
pip install -r requirements.txt
python yield_curve.py
```

All output is printed to stdout. Figures are saved to `/figures/` automatically.

---

## Mathematical Background

### 1. Deposit Bootstrap

Deposits give the only algebraically exact discount factor — no iteration required.

For an Act/360 deposit with quoted rate `r` and maturity `T` (in years):

```
P(0, T) = 1 / (1 + r × T)

r_c(T) = -ln P(0, T) / T        [continuously compounded zero rate]
```

### 2. Swap Bootstrap

Par swap pricing (fixed leg = floating leg at inception):

```
0 = K × Σ δᵢ × P(0, Tᵢ)  −  [1 − P(0, Tₙ)]
```

Solving for the unknown terminal discount factor:

```
P(0, Tₙ) = [1 − K × Σᵢ₌₁ⁿ⁻¹ δᵢ × P(0, Tᵢ)]  /  [1 + K × δₙ]
```

where `δᵢ = 0.5` (USD semi-annual) or `δᵢ = 1.0` (EUR annual).

Intermediate coupon dates with no pillar are filled via log-linear interpolation:

```
ln P(0, t) = ln P(0, t_lo) + [(t − t_lo) / (t_hi − t_lo)] × [ln P(0, t_hi) − ln P(0, t_lo)]
```

When `t` falls between the last known pillar and `Tₙ` (e.g. the 6M→2Y gap), `P(0, Tₙ)` is solved iteratively: an initial estimate anchors log-linear interpolation, then the bootstrap equation is re-evaluated to convergence (typically 3–5 iterations, converging to 1e-12 tolerance).

### 3. Nelson-Siegel Parametric Curve

```
r(t) = β₀  +  β₁ × [(1 − e^{−t/τ}) / (t/τ)]
            +  β₂ × [(1 − e^{−t/τ}) / (t/τ)  −  e^{−t/τ}]
```

Parameter interpretation:
- `β₀`: long-run rate (asymptote as t → ∞)
- `β₀ + β₁`: instantaneous short rate (t → 0)
- `β₂`: hump (positive) or trough (negative) at medium maturities
- `τ`: hump location in years. Constrained `τ ≥ 0.5`.

Fitted via L-BFGS-B on sum of squared errors to bootstrapped pillar zeros.

### 4. Duration Conventions

**Macaulay duration** (continuous discount curve, convention-free):

```
D_mac = (1 / P) × Σ tᵢ × Cᵢ × P(0, tᵢ)
```

**Modified duration** — three conventions, all implemented explicitly:

| Convention   | Formula                     | When used        |
|:------------ |:--------------------------- |:---------------- |
| Continuous   | `D_mod = D_mac`             | Cts compounding  |
| Semi-annual  | `D_mod = D_mac / (1 + y/2)` | USD bonds/swaps  |
| Annual       | `D_mod = D_mac / (1 + y)`   | EUR bonds/swaps  |

`D_mod = D_mac` as a default **never appears** in this codebase.

---

## Key Results (Reference Date: 2024-01-15)

### Bootstrapped Zero Rates

| Maturity | USD Zero (cts) | EUR Zero (cts) |
|:-------- |:-------------- |:-------------- |
| 1M       | 5.2983%        | 3.8737%        |
| 3M       | 5.2948%        | 3.8811%        |
| 6M       | 5.2310%        | 3.8330%        |
| 2Y       | 4.5597%        | 3.1949%        |
| 5Y       | 4.1353%        | 2.9982%        |
| 10Y      | 4.0367%        | 3.0754%        |
| 20Y      | 4.0821%        | 3.1649%        |
| 30Y      | 4.1752%        | 3.1876%        |

### 10Y Bond Analytics (3% coupon, face 100)

| Metric    | USD     | EUR     |
|:--------- |:------- |:------- |
| Price     | 91.0795 | 98.9779 |
| YTM       | 4.0962% | 3.1206% |
| D_mac     | 8.6460  | 8.7789  |
| D_mod     | 8.4725  | 8.5132  |
| DV01      | 0.0772  | 0.0843  |
| Convexity | 81.994  | 83.484  |

### Par Swap Rates

| Maturity | USD Par Rate | EUR Par Rate |
|:-------- |:------------ |:------------ |
| 2Y       | 4.6251%      | 3.2538%      |
| 5Y       | 4.2020%      | 3.0513%      |
| 10Y      | 4.1016%      | 3.1205%      |

### Scenario P&L — 10Y Bond (3% coupon, face 100)

Shocks are applied to the continuously compounded zero curve, not to par swap rates. On an inverted curve, a parallel zero-rate shock does not produce a parallel par-rate shock.

| Scenario        | USD P&L   | EUR P&L   |
|:--------------- |:--------- |:--------- |
| Parallel +100bp | −7.51     | −8.29     |
| Bear Steepener  | −3.60     | −3.99     |
| Bear Flattener  | −0.24     | −0.26     |
| Inversion       | +1.50     | +1.68     |

---

## Known Limitations

**1. Single-curve framework**
This implementation does not separate OIS discounting from IBOR-based forward curve construction (the "multi-curve" framework introduced post-2008). In practice, USD SOFR swaps are consistently single-curve (OIS discounting, SOFR forwards). EUR ESTR is also single-curve. The simplification is exact for SOFR/ESTR instruments but would misprice EURIBOR-linked products.

**2. No convexity adjustment for futures**
Eurodollar/SOFR futures embed a convexity adjustment relative to FRA rates due to daily margining. This codebase uses only deposit and swap quotes; no futures are included. Incorporating futures would require adjustments of order `σ² × T₁ × T₂ / 2`.

**3. Act/360 approximation vs exact business-day counting**
Coupon periods are approximated as `δ = 1 / coupon_freq` in years rather than computed from exact business-day-adjusted dates. For a 5Y semi-annual USD swap this introduces at most ~0.5bp error on the par rate. Production implementations use an explicit date schedule with holiday calendar roll conventions (Modified Following).

**4. Flat extrapolation risk for long maturities**
Without intermediate pillars between 10Y and 20Y, the bootstrap would assign all 11Y–19Y coupon dates the discount factor `P(0, 10Y)` (flat extrapolation of the zero curve). This produces a structural bias of approximately +64bp on the 20Y zero rate. This codebase explicitly includes **12Y and 15Y pillars** to eliminate this gap. The 30Y pillar is anchored by a direct market quote, so no extrapolation occurs beyond 30Y within the bootstrapped region.

**5. Convexity uses continuous compounding convention**
The convexity formula `C = (1/P) × Σ tᵢ² × Cᵢ × P(0,tᵢ)` is the continuous-compounding version. Under semi-annual compounding the standard textbook formula uses `tᵢ(tᵢ+δ)/(1+y/f)²` terms. The difference is less than 0.5% relative on a 10Y bond and does not materially affect hedging decisions, but a production system would match the convexity convention to the duration convention for internal consistency.

---

## Interview Q&A

**Q1: What is a discount factor and how do you bootstrap it from market instruments?**

A discount factor `P(0, T)` is the present value today of one unit of currency paid at time `T`. For a deposit with Act/360 rate `r`, it is algebraically exact: `P(0, T) = 1 / (1 + r × T)`. For a swap, the par swap equation gives us `P(0, T_n)` analytically, provided all shorter-maturity discount factors are already known. This means we must process instruments strictly in ascending maturity order — the bootstrap is a sequential, causal algorithm.

**Q2: Why does the order of bootstrap iteration matter?**

The swap bootstrap formula for `P(0, T_n)` requires every `P(0, T_i)` for `i < n` as inputs. If we process a 10Y swap before a 5Y swap, the 5Y coupon dates of the 10Y swap have unknown discount factors, and any interpolation from the 6M deposit pillar to 10Y would be uninformed and biased. The strictly ascending order ensures each pillar is solved using only already-exact discount factors, so no circularity occurs.

**Q3: What is the difference between Macaulay duration and Modified duration?**

Macaulay duration (`D_mac`) is the weighted average time to receive cash flows, weighted by their present value. It is a property of the bond's cash flow structure and the discount curve — it does not depend on a compounding convention. Modified duration (`D_mod`) converts `D_mac` into a price sensitivity: `ΔP / P ≈ −D_mod × Δy`. The conversion depends on the yield compounding convention: for semi-annual compounding (USD), `D_mod = D_mac / (1 + y/2)`; for annual (EUR), `D_mod = D_mac / (1 + y)`. Equating `D_mod = D_mac` is only valid under continuous compounding and is a common error in implementations that fail to specify the convention.

**Q4: Why is the EUR 10Y bond priced differently from the USD 10Y bond for the same coupon?**

Both bonds have the same 3% coupon but trade at different levels because the 10Y discount rate differs: USD 10Y zero = 4.04%, EUR 10Y zero = 3.08%. Lower EUR discount rates produce higher discount factors across all coupon dates and at the 10Y maturity, raising the EUR bond price to ~98.98 vs USD ~91.08. The USD inversion also applies higher short-end rates to near-term coupons specifically, further depressing USD bond prices relative to EUR where the curve is less inverted.

**Q5: What does a bear flattener mean for a long-duration bond position?**

A bear flattener is a scenario where short-end rates rise materially while long-end rates are anchored (in this model: +50bp at ≤2Y, 0bp at ≥10Y). For a long-duration bond (e.g. the 30Y 5% coupon bond), most of the price sensitivity comes from long-end discount rates. Under a bear flattener, the 30Y zero rate barely moves, so the price impact is small (USD P&L ≈ −0.41 per 100 face). Contrast with a parallel +100bp shock where the 30Y bond loses ~17 points. A long-duration bond is therefore relatively protected in a bear flattener compared to a parallel shock — which explains why some portfolio managers express views on curve shape rather than parallel rate direction.

**Q6: What happens if you bootstrap a 20Y swap without intermediate pillars between 10Y and 20Y?**

Without pillars at (say) 12Y and 15Y, the bootstrap must interpolate discount factors for all coupon dates between 10Y and 20Y using only `P(0, 10Y)` as the last known anchor. Log-linear interpolation from 10Y to 20Y with no intermediate data point forces the zero curve to remain essentially flat between these tenors, regardless of the actual market term structure. The 20Y swap equation then solves for `P(0, 20Y)` under this biased annuity, producing a structural error of approximately +64bp on the 20Y zero rate. This is a known failure mode of naive bootstrap implementations. The fix is straightforward: include 12Y and 15Y market quotes (or interpolated pillars) to anchor the intermediate region. This codebase includes both explicitly.

---

## Reference

All data is synthetic and calibrated to early-2024 market conditions (reference date: 2024-01-15). No live API calls are made.
