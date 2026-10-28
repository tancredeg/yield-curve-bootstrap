"""ratescurve: government yield curve construction, bond risk and Bund futures basis.

Modules:
    conventions   TARGET2 calendar, ACT/365F, ACT/360, ACT/ACT ICMA
    instruments   dated fixed-rate bonds, constant-maturity par instruments
    interpolation linear zero, log-linear DF, Hagan-West monotone convex, natural cubic
    curve         discount curve, shifted curve, parallel and key-rate shifts
    calibration   exact global calibration of pillar zeros to instrument prices
    bonds         curve pricing, ICMA yields, duration, convexity
    risk          parallel DV01, key-rate DV01, bucketed (par) DV01
    pca           principal components of curve changes
    futures       Eurex conversion factors, basis, carry, implied repo, CTD
    scenarios     BCBS IRRBB standardised shocks (SRP31, 2024 calibration)
    data          loaders for the frozen sample in data/sample/
"""

__version__ = "1.0.0"
