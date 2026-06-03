"""
Invariant test suite for yield_curve.py
=======================================

Run: pytest -v

Tests use INTERNAL FIXTURES, not external data files. This ensures
mathematical invariants are verified regardless of whether data/
CSVs exist. A separate integration test section verifies CSV loading
when data files are present.
"""

import math
import os
import pytest

import yield_curve as yc


# ── Fixtures (self-contained, no external deps) ───────────────

_USD_DEP = {0.25: 0.054, 0.5: 0.053}
_USD_PAR = {1: 0.048, 2: 0.044, 5: 0.040, 10: 0.039, 20: 0.041, 30: 0.040}

_EUR_ZEROS_ANN = {0.5: 0.020, 1: 0.022, 2: 0.025, 5: 0.032, 10: 0.038, 20: 0.040, 30: 0.041}
_EUR_PILLARS = {t: (1 + z) ** (-t) for t, z in _EUR_ZEROS_ANN.items()}


def _usd_pillars():
    dep = yc.bootstrap_deposits(_USD_DEP, yc.ACT360)
    return yc.bootstrap_swaps(_USD_PAR, dep, yc.SEMI, yc.ACT360)


def _cubic(pillars):
    mats = sorted(pillars.keys())
    zeros = [-math.log(pillars[m]) / m for m in mats]
    return yc.build_cubic_spline(mats, zeros)


# ── 1. Deposit bootstrap — algebraic exactness ───────────────

def test_deposit_df_known_value():
    pillars = yc.bootstrap_deposits({0.5: 0.05}, yc.ACT360)
    expected = 1.0 / (1.0 + 0.05 * 0.5)
    assert math.isclose(pillars[0.5], expected, abs_tol=1e-15)


def test_deposit_zero_rate_consistency():
    pillars = yc.bootstrap_deposits(_USD_DEP, yc.ACT360)
    for T, df in pillars.items():
        z = -math.log(df) / T
        assert math.isclose(math.exp(-z * T), df, abs_tol=1e-12)


# ── 2. Swap/par-yield bootstrap — independent repricing ──────

def test_bootstrap_reprices_at_par():
    pillars = _usd_pillars()
    delta = 1.0 / yc.SEMI
    for T in sorted(_USD_PAR.keys()):
        n = int(round(T / delta))
        coupon_dates = [round(i * delta, 8) for i in range(1, n + 1)]
        annuity = sum(delta * yc._loglinear_interp(t, pillars) for t in coupon_dates)
        k_repriced = (1.0 - yc._loglinear_interp(T, pillars)) / annuity
        assert abs(k_repriced - _USD_PAR[T]) < 1e-6, f"T={T}Y residual too large"


def test_par_swap_rate_matches_input_on_cubic():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    k = yc.par_swap_rate(10, yc.SEMI, cs)
    assert abs(k - _USD_PAR[10]) < 1e-4


# ── 3. Swap pricing invariants ───────────────────────────────

def test_swap_prices_to_zero_at_par():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    k_par = yc.par_swap_rate(5, yc.SEMI, cs)
    npv = yc.price_swap(10e6, k_par, 5, yc.SEMI, cs, "payer")
    assert abs(npv) < 1e-3


def test_payer_receiver_antisymmetry():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    pay = yc.price_swap(10e6, 0.045, 5, yc.SEMI, cs, "payer")
    rec = yc.price_swap(10e6, 0.045, 5, yc.SEMI, cs, "receiver")
    assert math.isclose(pay, -rec, abs_tol=1e-9)


def test_payer_itm_when_par_above_fixed():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    k_par = yc.par_swap_rate(2, yc.SEMI, cs)
    npv = yc.price_swap(10e6, k_par - 0.005, 2, yc.SEMI, cs, "payer")
    assert npv > 0


def test_floating_leg_identity():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    T = 5
    df_T = math.exp(-float(cs(T)) * T)
    v_float = yc.price_swap(1.0, 0.0, T, yc.SEMI, cs, "payer")
    assert math.isclose(v_float, 1.0 - df_T, abs_tol=1e-12)


# ── 4. Swap DV01 — signed position sensitivity ──────────────

def test_swap_dv01_sign_and_antisymmetry():
    pillars = _usd_pillars()
    pay = yc.swap_dv01(10e6, 0.04, 10, yc.SEMI, pillars, "payer")
    rec = yc.swap_dv01(10e6, 0.04, 10, yc.SEMI, pillars, "receiver")
    assert pay > 0 and rec < 0
    assert math.isclose(pay, -rec, rel_tol=1e-9)


def test_swap_dv01_uses_position_fixed_rate():
    pillars = _usd_pillars()
    d_low  = yc.swap_dv01(10e6, 0.030, 10, yc.SEMI, pillars, "payer")
    d_high = yc.swap_dv01(10e6, 0.060, 10, yc.SEMI, pillars, "payer")
    assert d_low > 0 and d_high > 0
    assert 0.8 < d_low / d_high < 1.25


# ── 5. Bond analytics — convention consistency + FD check ────

def test_duration_conventions_internally_consistent():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    semi = yc.bond_analytics(100, 0.03, 10, 2, cs, "semiannual")
    cont = yc.bond_analytics(100, 0.03, 10, 2, cs, "continuous")
    assert math.isclose(cont["duration_mod"], cont["duration_mac"], abs_tol=1e-12)
    assert math.isclose(semi["duration_mod"],
                        semi["duration_mac"] / (1 + semi["ytm"] / 2), rel_tol=1e-9)
    assert abs(semi["duration_mod"] - cont["duration_mod"]) > 1e-4


def test_convention_freq_mismatch_raises():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    with pytest.raises(ValueError):
        yc.bond_analytics(100, 0.03, 10, 2, cs, "annual")
    with pytest.raises(ValueError):
        yc.bond_analytics(100, 0.03, 10, 1, cs, "semiannual")


def test_unknown_convention_raises():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    with pytest.raises(ValueError):
        yc.bond_analytics(100, 0.04, 5, yc.SEMI, cs, "monthly")


def test_dv01_identity():
    pillars = _usd_pillars()
    cs = _cubic(pillars)
    a = yc.bond_analytics(100, 0.04, 5, yc.SEMI, cs, "semiannual")
    assert math.isclose(a["dv01"], a["duration_mod"] * a["price"] * 1e-4, rel_tol=1e-12)


def test_bond_risk_matches_finite_difference():
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
    assert math.isclose(a["duration_mod"], -dP / p0, rel_tol=1e-5)
    assert math.isclose(a["convexity"], d2P / p0, rel_tol=1e-4)


# ── 6. Interpolation guards ─────────────────────────────────

def test_loglinear_extrapolation_forbidden():
    pillars = {0.5: 0.97, 2.0: 0.91}
    with pytest.raises(ValueError):
        yc._loglinear_interp(5.0, pillars)


def test_parallel_shock_shifts_every_zero():
    pillars = _usd_pillars()
    mats = sorted(pillars.keys())
    base = [-math.log(pillars[m]) / m for m in mats]
    cs_shock = yc.apply_shock(pillars, {0: 100, 30: 100})
    for m, z0 in zip(mats, base):
        assert math.isclose(float(cs_shock(m)), z0 + 0.01, abs_tol=1e-9)


# ── 7. Nelson-Siegel — regression guard ─────────────────────

def test_nelson_siegel_tau_is_fitted_not_stuck():
    up = _usd_pillars()
    um = sorted(up.keys()); uz = [-math.log(up[m]) / m for m in um]
    em = sorted(_EUR_PILLARS.keys()); ez = [-math.log(_EUR_PILLARS[m]) / m for m in em]
    _, _, _, tau_usd = yc.fit_nelson_siegel(um, uz, "USD")
    _, _, _, tau_eur = yc.fit_nelson_siegel(em, ez, "EUR")
    assert 0.5 <= tau_usd <= 10.0
    assert 0.5 <= tau_eur <= 10.0
    assert abs(tau_usd - tau_eur) > 0.1


def test_nelson_siegel_reasonable_fit():
    up = _usd_pillars()
    um = sorted(up.keys()); uz = [-math.log(up[m]) / m for m in um]
    b0, b1, b2, tau = yc.fit_nelson_siegel(um, uz, "USD")
    sse = sum((yc.nelson_siegel(t, b0, b1, b2, tau) - z) ** 2 for t, z in zip(um, uz))
    rmse_bp = math.sqrt(sse / len(um)) * 10000
    assert rmse_bp < 10.0


# ── 8. EUR pillar ingestion ──────────────────────────────────

def test_eur_pillars_consistency():
    """ECB annually compounded zeros must produce consistent DFs."""
    for t, z_ann in _EUR_ZEROS_ANN.items():
        df = _EUR_PILLARS[t]
        assert math.isclose(df, (1 + z_ann) ** (-t), abs_tol=1e-12)
        z_cc = -math.log(df) / t
        assert math.isclose(z_cc, math.log(1 + z_ann), abs_tol=1e-12)


# ── 9. Integration: real CSV data (skipped if data/ absent) ──

_DATA_OK = os.path.exists("data/usd_yields.csv") and os.path.exists("data/eur_zeros.csv")

@pytest.mark.skipif(not _DATA_OK, reason="data/ CSVs not present")
def test_csv_loading_and_usd_bootstrap():
    """Full pipeline on real CSV data: load → bootstrap → verify."""
    dep, par = yc.load_usd_data()
    assert len(dep) >= 2 and len(par) >= 5
    pillars_dep = yc.bootstrap_deposits(dep, yc.ACT360)
    pillars = yc.bootstrap_swaps(par, pillars_dep, yc.SEMI, yc.ACT360)
    delta = 1.0 / yc.SEMI
    for T, k in par.items():
        n = int(round(T / delta))
        dates = [round(i * delta, 8) for i in range(1, n + 1)]
        annuity = sum(delta * yc._loglinear_interp(t, pillars) for t in dates)
        k_re = (1.0 - yc._loglinear_interp(T, pillars)) / annuity
        assert abs(k_re - k) < 1e-5, f"CSV T={T}Y residual too large"


@pytest.mark.skipif(not _DATA_OK, reason="data/ CSVs not present")
def test_csv_loading_eur_zeros():
    pillars, raw = yc.load_eur_data()
    assert len(pillars) >= 6
    for t, df in pillars.items():
        assert 0 < df < 1, f"EUR DF at {t}Y out of range"
