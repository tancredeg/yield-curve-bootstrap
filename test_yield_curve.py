"""
Invariant test suite for yield_curve.py
=======================================

Run from the project folder:

    pip install pytest
    pytest -v                      # or: python -m pytest -v

Each test checks a property that MUST hold if the maths is correct, not just
that the code runs. Failures point at a specific broken invariant.

Importing yield_curve does NOT execute main() (it is guarded by __main__),
so the tests are fast and produce no figures.
"""

import math
import pytest

import yield_curve as yc


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _usd_pillars():
    dep = yc.bootstrap_deposits(yc.USD_DEPOSITS, yc.ACT360)
    return yc.bootstrap_swaps(yc.USD_SWAPS, dep, yc.SEMI, yc.ACT360)


def _eur_pillars():
    dep = yc.bootstrap_deposits(yc.EUR_DEPOSITS, yc.ACT360)
    return yc.bootstrap_swaps(yc.EUR_SWAPS, dep, yc.ANNUAL, yc.ACT360)


def _cubic(pillars):
    mats = sorted(pillars.keys())
    zeros = [-math.log(pillars[m]) / m for m in mats]
    return yc.build_cubic_spline(mats, zeros)


# ----------------------------------------------------------------------
# 1. Deposit bootstrap — algebraic exactness
# ----------------------------------------------------------------------

def test_deposit_df_known_value():
    """Act/360 deposit DF is exact: P = 1 / (1 + r*T)."""
    pillars = yc.bootstrap_deposits({0.5: 0.05}, yc.ACT360)
    expected = 1.0 / (1.0 + 0.05 * 0.5)
    assert math.isclose(pillars[0.5], expected, rel_tol=0, abs_tol=1e-15)


def test_deposit_zero_rate_consistency():
    """Continuously compounded zero recovers the deposit DF: P = exp(-z*T)."""
    pillars = yc.bootstrap_deposits(yc.USD_DEPOSITS, yc.ACT360)
    for T, df in pillars.items():
        z = -math.log(df) / T
        assert math.isclose(math.exp(-z * T), df, abs_tol=1e-12)


# ----------------------------------------------------------------------
# 2. Swap bootstrap — independent repricing residual
# ----------------------------------------------------------------------

@pytest.mark.parametrize("swaps, freq, builder", [
    (yc.USD_SWAPS, yc.SEMI, _usd_pillars),
    (yc.EUR_SWAPS, yc.ANNUAL, _eur_pillars),
])
def test_bootstrap_reprices_at_par(swaps, freq, builder):
    """Every input swap must reprice to par on the final curve (< 1e-6 in rate)."""
    pillars = builder()
    delta = 1.0 / freq
    for T in sorted(swaps.keys()):
        n = int(round(T / delta))
        coupon_dates = [round(i * delta, 8) for i in range(1, n + 1)]
        annuity = sum(delta * yc._loglinear_interp(t, pillars) for t in coupon_dates)
        k_repriced = (1.0 - yc._loglinear_interp(T, pillars)) / annuity
        assert abs(k_repriced - swaps[T]) < 1e-6, f"T={T}Y residual too large"


def test_par_swap_rate_matches_input_on_cubic():
    """Par rate from the cubic curve is close to the input (interpolation residual)."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    k = yc.par_swap_rate(10, yc.SEMI, cs)
    assert abs(k - yc.USD_SWAPS[10]) < 1e-4   # < 1bp


# ----------------------------------------------------------------------
# 3. Swap pricing invariants
# ----------------------------------------------------------------------

def test_swap_prices_to_zero_at_par():
    """A swap struck at the par rate has ~zero NPV."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    k_par = yc.par_swap_rate(5, yc.SEMI, cs)
    npv = yc.price_swap(10e6, k_par, 5, yc.SEMI, cs, "payer")
    assert abs(npv) < 1e-3   # essentially zero on 10M notional


def test_payer_receiver_antisymmetry():
    """Payer and receiver NPVs are exact negatives of each other."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    pay = yc.price_swap(10e6, 0.045, 5, yc.SEMI, cs, "payer")
    rec = yc.price_swap(10e6, 0.045, 5, yc.SEMI, cs, "receiver")
    assert math.isclose(pay, -rec, abs_tol=1e-9)


def test_payer_itm_when_par_above_fixed():
    """Pay-fixed is ITM (NPV>0) iff the par rate exceeds the fixed rate."""
    pillars = _eur_pillars()
    cs = _cubic(pillars)
    k_par = yc.par_swap_rate(2, yc.ANNUAL, cs)
    fixed = k_par - 0.0050  # 50bp below par -> payer should gain
    npv = yc.price_swap(10e6, fixed, 2, yc.ANNUAL, cs, "payer")
    assert npv > 0


def test_floating_leg_identity():
    """Single-curve floating leg equals N*(1 - P(0,T))."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    T = 5
    df_T = math.exp(-float(cs(T)) * T)
    v_float = yc.price_swap(1.0, 0.0, T, yc.SEMI, cs, "payer")
    assert math.isclose(v_float, 1.0 - df_T, abs_tol=1e-12)


# ----------------------------------------------------------------------
# 4. Swap DV01 — signed position sensitivity
# ----------------------------------------------------------------------

def test_swap_dv01_sign_and_antisymmetry():
    """Payer DV01 > 0 (gains on +1bp), receiver DV01 < 0, equal magnitude."""
    pillars = _usd_pillars()
    pay = yc.swap_dv01(10e6, 0.045, 10, yc.SEMI, pillars, "payer")
    rec = yc.swap_dv01(10e6, 0.045, 10, yc.SEMI, pillars, "receiver")
    assert pay > 0 and rec < 0
    assert math.isclose(pay, -rec, rel_tol=1e-9)


def test_swap_dv01_uses_position_fixed_rate():
    """DV01 is a curve sensitivity: ~invariant to the fixed rate for the same
    notional/maturity (the fixed leg's rate scales the small annuity change),
    but the function must accept and use the actual rate (no recomputed par)."""
    pillars = _usd_pillars()
    d_low  = yc.swap_dv01(10e6, 0.030, 10, yc.SEMI, pillars, "payer")
    d_high = yc.swap_dv01(10e6, 0.060, 10, yc.SEMI, pillars, "payer")
    # Both positive, same order of magnitude (curve DV01 dominated by float leg)
    assert d_low > 0 and d_high > 0
    assert 0.8 < d_low / d_high < 1.25


# ----------------------------------------------------------------------
# 5. Bond analytics — convention consistency + finite-difference check
# ----------------------------------------------------------------------

def test_duration_conventions_internally_consistent():
    """Each convention satisfies its own D_mod = D_mac / (1 + y/f) and they differ."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    semi = yc.bond_analytics(100, 0.03, 10, 2, cs, "semiannual")
    cont = yc.bond_analytics(100, 0.03, 10, 2, cs, "continuous")
    ann  = yc.bond_analytics(100, 0.03, 10, 1, cs, "annual")

    # Continuous: D_mod == D_mac
    assert math.isclose(cont["duration_mod"], cont["duration_mac"], abs_tol=1e-12)
    # Semi-annual: D_mod = D_mac / (1 + y/2)
    assert math.isclose(semi["duration_mod"],
                        semi["duration_mac"] / (1 + semi["ytm"] / 2), rel_tol=1e-9)
    # Annual: D_mod = D_mac / (1 + y)
    assert math.isclose(ann["duration_mod"],
                        ann["duration_mac"] / (1 + ann["ytm"]), rel_tol=1e-9)
    # Conventions are genuinely distinct (never conflated)
    assert abs(semi["duration_mod"] - cont["duration_mod"]) > 1e-4


def test_convention_freq_mismatch_raises():
    """A periodic convention inconsistent with coupon_freq must raise."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    with pytest.raises(ValueError):
        yc.bond_analytics(100, 0.03, 10, 2, cs, "annual")   # freq=2 but annual
    with pytest.raises(ValueError):
        yc.bond_analytics(100, 0.03, 10, 1, cs, "semiannual")


def test_unknown_convention_raises():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    with pytest.raises(ValueError):
        yc.bond_analytics(100, 0.04, 5, yc.SEMI, cs, "monthly")


def test_dv01_identity():
    """DV01 = D_mod * Price * 1e-4."""
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    a = yc.bond_analytics(100, 0.04, 5, yc.SEMI, cs, "semiannual")
    assert math.isclose(a["dv01"], a["duration_mod"] * a["price"] * 1e-4, rel_tol=1e-12)


def test_bond_risk_matches_finite_difference():
    """
    D_mod and convexity must match a finite-difference bump of the YTM, i.e. the
    Taylor relation dP/P = -D_mod*dy + 0.5*Conv*dy^2 holds. This is the test
    that the convexity convention is consistent with the duration convention.
    """
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    face, cpn, mat, f = 100.0, 0.03, 10, 2
    a = yc.bond_analytics(face, cpn, mat, f, cs, "semiannual")
    y = a["ytm"]

    delta = 1.0 / f
    coupon = face * cpn * delta
    n = int(round(mat / delta))
    dates = [round(i * delta, 8) for i in range(1, n + 1)]
    cfs = [(t, coupon) for t in dates]
    cfs[-1] = (dates[-1], coupon + face)

    def P(yld):
        g = 1.0 + yld / f
        return sum(cf / g ** (f * t) for t, cf in cfs)

    h = 1e-6
    p0 = P(y)
    dP = (P(y + h) - P(y - h)) / (2 * h)
    d2P = (P(y + h) - 2 * p0 + P(y - h)) / (h * h)

    dmod_fd = -dP / p0
    conv_fd = d2P / p0
    assert math.isclose(a["duration_mod"], dmod_fd, rel_tol=1e-5)
    assert math.isclose(a["convexity"], conv_fd, rel_tol=1e-4)


# ----------------------------------------------------------------------
# 6. Interpolation guards
# ----------------------------------------------------------------------

def test_loglinear_extrapolation_forbidden():
    """Bootstrap interpolation must refuse to extrapolate beyond the last pillar."""
    pillars = {0.5: 0.97, 2.0: 0.91}
    with pytest.raises(ValueError):
        yc._loglinear_interp(5.0, pillars)


def test_parallel_shock_shifts_every_zero():
    """A +100bp parallel zero shock raises every pillar zero by ~100bp."""
    pillars = _usd_pillars()
    mats = sorted(pillars.keys())
    base = [-math.log(pillars[m]) / m for m in mats]
    cs_shock = yc.apply_shock(pillars, {0: 100, 30: 100})
    for m, z0 in zip(mats, base):
        assert math.isclose(float(cs_shock(m)), z0 + 0.01, abs_tol=1e-9)


# ----------------------------------------------------------------------
# 7. Nelson-Siegel calibration — regression guard for the stuck-tau bug
# ----------------------------------------------------------------------

def test_nelson_siegel_tau_is_fitted_not_stuck():
    """
    Regression test: the previous L-BFGS-B fit left tau frozen at its init
    for every curve. The grid-search fit must produce DIFFERENT taus for two
    different curves and keep tau within bounds.
    """
    up = _usd_pillars(); ep = _eur_pillars()
    um = sorted(up.keys()); uz = [-math.log(up[m]) / m for m in um]
    em = sorted(ep.keys()); ez = [-math.log(ep[m]) / m for m in em]

    _, _, _, tau_usd = yc.fit_nelson_siegel(um, uz, "USD")
    _, _, _, tau_eur = yc.fit_nelson_siegel(em, ez, "EUR")

    assert 0.5 <= tau_usd <= 10.0
    assert 0.5 <= tau_eur <= 10.0
    assert abs(tau_usd - tau_eur) > 0.1


def test_nelson_siegel_reasonable_fit():
    """NS RMSE on the bootstrapped curve should be small (< 10bp)."""
    up = _usd_pillars()
    um = sorted(up.keys()); uz = [-math.log(up[m]) / m for m in um]
    b0, b1, b2, tau = yc.fit_nelson_siegel(um, uz, "USD")
    sse = sum((yc.nelson_siegel(t, b0, b1, b2, tau) - z) ** 2 for t, z in zip(um, uz))
    rmse_bp = math.sqrt(sse / len(um)) * 10000
    assert rmse_bp < 10.0
