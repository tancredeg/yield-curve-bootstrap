"""PCA identities, BCBS scenario shapes, data integrity."""

import numpy as np
import pytest

from ratescurve.conventions import add_business_days
from ratescurve.curve import DiscountCurve
from ratescurve.data import (
    Observation,
    infer_first_period,
    load_bond_static,
    load_bund_quotes,
    tenor_to_years,
    verify_manifest,
)
from ratescurve.pca import loading_similarity, principal_components
from ratescurve.scenarios import SCENARIOS, SHOCK_SIZES_BP, scenario_shift, shock_bp


def test_pca_recovers_a_known_factor_structure():
    rng = np.random.default_rng(1)
    n = 10
    x = np.linspace(-1, 1, n)
    level, slope, curv = np.ones(n), x, x**2 - np.mean(x**2)
    basis = np.column_stack([v / np.linalg.norm(v) for v in (level, slope, curv)])
    scores = rng.normal(size=(5000, 3)) * np.array([5.0, 2.0, 0.7])
    data = scores @ basis.T + rng.normal(scale=0.05, size=(5000, n))
    res = principal_components(data, k=3)
    assert np.allclose(res.loadings.T @ res.loadings, np.eye(3), atol=1e-12)
    assert np.all(np.diff(res.eigenvalues) <= 0)
    assert res.explained.sum() <= 1 + 1e-12
    for k in range(3):
        assert abs(res.loadings[:, k] @ basis[:, k]) > 0.99
    assert res.loadings[:, 0].sum() > 0
    assert res.loadings[-1, 1] > res.loadings[0, 1]
    half = principal_components(data[:2500], 3)
    assert np.all(loading_similarity(res, half) > 0.98)
    with pytest.raises(ValueError):
        principal_components(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        principal_components(np.ones(5))


def test_srp31_worked_example():
    sizes = (100.0, 100.0, 100.0)
    t = np.array([3.5])
    assert shock_bp("short_up", t, sizes)[0] == pytest.approx(41.7, abs=0.05)
    assert shock_bp("steepener", t, sizes)[0] == pytest.approx(25.4, abs=0.05)
    assert shock_bp("flattener", t, sizes)[0] == pytest.approx(-1.6, abs=0.05)
    with pytest.raises(ValueError):
        shock_bp("twist", t, sizes)


def test_scenario_sizes_and_limits():
    eur = SHOCK_SIZES_BP["EUR"]
    assert eur == (225.0, 350.0, 200.0) and SHOCK_SIZES_BP["USD"] == (200.0, 300.0, 225.0)
    assert shock_bp("short_up", np.array([0.0]), eur)[0] == pytest.approx(350.0)
    assert shock_bp("parallel_down", np.array([7.0]), eur)[0] == -225.0
    assert len(SCENARIOS) == 6


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_applied_exactly_on_continuous_curve(scenario):
    curve = DiscountCurve([0.5, 2, 5, 10, 30], [0.025, 0.027, 0.03, 0.033, 0.036])
    shocked = curve.shifted(scenario_shift(scenario, "EUR"))
    grid = np.linspace(0.1, 30, 300)
    applied = (shocked.zero(grid) - curve.zero(grid)) * 1e4
    assert np.allclose(applied, shock_bp(scenario, grid, SHOCK_SIZES_BP["EUR"]), atol=1e-9)


def test_manifest_checksums():
    files = verify_manifest()
    assert {
        "ust_par_yields.csv",
        "ecb_aaa_spot.csv",
        "estr.csv",
        "bundesbank_federal_securities.csv",
        "bond_static.csv",
        "eurex_fgbl_daily.csv",
        "eurex_deliverable_bonds.csv",
    } <= set(files)


def test_first_period_inference_reproduces_static_table():
    quotes = load_bund_quotes()
    static = load_bond_static()
    for isin in ["DE000BU2Z064", "DE000BU22155", "DE000BU25075"]:
        grp = quotes[quotes["isin"] == isin]
        obs = [
            Observation(
                add_business_days(r["price_date"], 2),
                float(r["dirty"] - r["clean"]),
                float(r["dirty"]),
                float(r["yield"]),
            )
            for _, r in grp.iterrows()
        ]
        start, first = infer_first_period(isin, float(grp["coupon"].iloc[0]), grp["maturity"].iloc[0], obs)
        assert (start, first) == (static.loc[isin, "interest_start"], static.loc[isin, "first_coupon"])
    crosscheck = static[static["borsaitaliana_interest_commencement"].notna()]
    assert len(crosscheck) >= 2
    assert all(
        str(a) == str(b)
        for a, b in zip(crosscheck["interest_start"], crosscheck["borsaitaliana_interest_commencement"])
    )
    regular = quotes[quotes["isin"] == "DE000BU2Z056"]
    obs = [
        Observation(
            add_business_days(r["price_date"], 2), float(r["dirty"] - r["clean"]), float(r["dirty"]), float(r["yield"])
        )
        for _, r in regular.iterrows()
    ]
    assert infer_first_period("DE000BU2Z056", 2.6, regular["maturity"].iloc[0], obs) == (None, None)


def test_tenor_labels():
    assert tenor_to_years("1.5M") == pytest.approx(0.125)
    assert tenor_to_years("30Y") == 30.0
    with pytest.raises(ValueError):
        tenor_to_years("5D")
