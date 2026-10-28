"""Check README claim: without 12Y/15Y gap fillers the 20Y bootstrap is biased by ~64bp."""
import io, contextlib, math, sys
sys.path.insert(0, ".")  # run from repo root
import yield_curve as yc

with contextlib.redirect_stdout(io.StringIO()):
    dep, par = yc.load_usd_data()
    d = yc.bootstrap_deposits(dep)
    ref = yc.bootstrap_swaps(par, d, yc.SEMI)

# Naive variant: coupons beyond the last known pillar use FLAT zero extrapolation from that pillar
def naive(par, d):
    pil = dict(d)
    for T in sorted(par):
        K = par[T]; n = int(round(T / 0.5))
        dates = [round(i * 0.5, 8) for i in range(1, n + 1)]
        last = max(pil); z_last = -math.log(pil[last]) / last
        ann = 0.0
        for t in dates[:-1]:
            ann += 0.5 * (yc._loglinear_interp(t, pil) if t <= last else math.exp(-z_last * t))
        pil[T] = (1 - K * ann) / (1 + K * 0.5)
    return pil

nv = naive(par, d)
for T in (10.0, 20.0, 30.0):
    zr = -math.log(ref[T]) / T; zn = -math.log(nv[T]) / T
    print(f"{T:>4.0f}Y zero: anchored-iteration {zr:.4%} | naive flat-extrapolation {zn:.4%} | diff {(zn-zr)*1e4:+.1f}bp")
