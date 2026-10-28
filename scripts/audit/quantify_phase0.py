"""Phase 0 quantification of suspected defects on public HEAD 247592a.

Run from the repo root:  python scripts/audit/quantify_phase0.py
Every number quoted in AUDIT.md comes from this script.
"""
import math
import sys
import io
import contextlib

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, ".")  # run from repo root
import yield_curve as yc  # noqa: E402


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def zeros_of(pillars):
    m = sorted(pillars)
    return m, [-math.log(pillars[t]) / t for t in m]


dep, par = yc.load_usd_data()
usd = quiet(yc.bootstrap_swaps, par, quiet(yc.bootstrap_deposits, dep), yc.SEMI)
eur, _ = yc.load_eur_data()
um, uz = zeros_of(usd)
em, ez = zeros_of(eur)
ucs = yc.build_cubic_spline(um, uz)
ecs = yc.build_cubic_spline(em, ez)

print("=== A. Par repricing of USD calibration instruments on the PRICING curve")
print("    (natural cubic spline on zeros) vs the CALIBRATION curve (log-linear DF)")
worst = 0.0
for T, K in sorted(par.items()):
    k_cs = yc.par_swap_rate(T, yc.SEMI, ucs)
    r = (k_cs - K) * 1e4
    worst = max(worst, abs(r))
    print(f"    {T:>4.0f}Y  input {K:.4%}  cubic-spline par {k_cs:.4%}  residual {r:+.3f}bp")
print(f"    max |residual| on pricing curve: {worst:.3f}bp")

print("\n=== B. Scenario shock actually applied vs shock declared (zero-rate bp)")
t = np.linspace(max(um[0], em[0]), 30, 3001)
for name, cur_m, cur_z in (("USD", um, uz), ("EUR", em, ez)):
    cs0 = CubicSpline(cur_m, cur_z, bc_type="natural")
    for sc, shock in yc.build_scenarios().items():
        sm = sorted(shock)
        sv = [shock[k] for k in sm]
        declared = np.interp(t, sm, sv)
        cs1 = yc.apply_shock(dict(zip(cur_m, [math.exp(-z * m) for m, z in zip(cur_m, cur_z)])), shock)
        applied = (cs1(t) - cs0(t)) * 1e4
        err = applied - declared
        i = int(np.nanargmax(np.abs(err)))
        print(f"    {name} {sc:<16} declared max {max(abs(v) for v in sv):>4.0f}bp | "
              f"applied max {np.nanmax(np.abs(applied)):6.1f}bp | worst error {err[i]:+6.1f}bp at {t[i]:5.2f}Y "
              f"| applied min {np.nanmin(applied):+6.1f}bp")

print("\n=== C. Instantaneous forwards f(t) = d/dt[t z(t)] by interpolation method, USD")
tf = np.linspace(um[0], 30, 30001)


def fwd_from_zero(zfun, tt, h=1e-5):
    g = lambda x: x * zfun(x)
    return (g(tt + h) - g(tt - h)) / (2 * h)


lin = lambda x: np.interp(x, um, uz)
cub = CubicSpline(um, uz, bc_type="natural")
lnP = [-z * m for m, z in zip(um, uz)]
loglin = lambda x: -np.interp(x, um, lnP) / x
tt = tf[1:-1]
for label, fn in (("linear on zero", lin), ("natural cubic on zero", cub), ("log-linear on DF", loglin)):
    f = fwd_from_zero(fn, tt) * 100
    # discontinuity: largest jump between consecutive grid points
    jump = np.max(np.abs(np.diff(f)))
    print(f"    {label:<22} f min {np.min(f):6.3f}%  f max {np.max(f):6.3f}%  "
          f"largest step on 1e-3Y grid {jump*100:6.1f}bp")
# forward jump at 20Y under log-linear
fl = fwd_from_zero(loglin, np.array([19.99, 20.01])) * 100
print(f"    log-linear DF: f(20Y-) = {fl[0]:.3f}%, f(20Y+) = {fl[1]:.3f}%  -> jump {(fl[1]-fl[0])*100:+.0f}bp")
fl = fwd_from_zero(loglin, np.array([9.99, 10.01])) * 100
print(f"    log-linear DF: f(10Y-) = {fl[0]:.3f}%, f(10Y+) = {fl[1]:.3f}%  -> jump {(fl[1]-fl[0])*100:+.0f}bp")

print("\n=== D. Cubic-spline overshoot between USD 20Y and 30Y pillars")
seg = np.linspace(20, 30, 1001)
zz = cub(seg) * 100
print(f"    pillars: z(20)={uz[um.index(20.0)]*100:.3f}%  z(30)={uz[um.index(30.0)]*100:.3f}%  "
      f"spline max {zz.max():.3f}% at {seg[zz.argmax()]:.2f}Y  "
      f"(overshoot {(zz.max()-max(uz[um.index(20.0)], uz[um.index(30.0)])*100)*100:.1f}bp)")

print("\n=== E. Domain: price a 0.1Y EUR cash flow on the pricing curve")
print(f"    ecs(0.1) = {ecs(0.1)}  (extrapolate=False -> NaN propagates silently)")
print(f"    ucs(31)  = {ucs(31.0)}")

print("\n=== F. Nelson-Siegel tau vs grid bound [0.5, 10]")
for lab, m, z in (("USD", um, uz), ("EUR", em, ez)):
    b = quiet(yc.fit_nelson_siegel, m, z)
    print(f"    {lab} tau = {b[3]:.3f}  {'<- AT UPPER BOUND' if abs(b[3]-10) < 1e-9 else ''}")

print("\n=== G. Shape label vs hard-coded narrative")
s = (-math.log(usd[10.0]) / 10 + math.log(usd[2.0]) / 2) * 1e4
print(f"    USD 2Y-10Y zero slope = {s:+.1f}bp -> shape_label = 'upward-sloping', "
      f"yet commentary prints 'the inversion reflects...' and fig 04 says 'USD inverted'")

print("\n=== H. Gap-filler pillars claimed in README (12Y/15Y): present in USD pillars?")
print(f"    USD pillar maturities: {[round(x, 4) for x in um]}")
