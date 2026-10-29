"""Regression on the frozen sample: calibrated curves and headline analytics must not drift.

The golden values in ``tests/fixtures/golden.json`` were produced by this module on
``data/sample/``. After an intentional change, regenerate them with::

    UPDATE_GOLDEN=1 pytest tests/test_regression.py

and review the diff before committing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ratescurve.data import ASOF, bunds_on, load_deliverables, load_ecb_spot, load_estr, load_fgbl, load_ust_par
from ratescurve.futures import basis, conversion_factor, delivery_date
from ratescurve.markets import out_of_sample
from ratescurve.pca import principal_components

from .conftest import METHODS

GOLDEN = Path(__file__).parent / "fixtures" / "golden.json"


def _snapshot(usd_markets, eur_markets, bund_quotes) -> dict:
    snap: dict = {"asof": ASOF.isoformat(), "usd": {}, "eur": {}}
    for m in METHODS:
        snap["usd"][m] = {"zeros": usd_markets[m].curve.zeros.tolist()}
        oos = out_of_sample(eur_markets[m], bund_quotes)
        snap["eur"][m] = {
            "zeros": eur_markets[m].curve.zeros.tolist(),
            "oos_residual_bp": oos.set_index("isin")["residual_bp"].round(9).to_dict(),
        }
    snap["eur_selection"] = list(eur_markets["monotone_convex"].instruments["isin"])
    q = bunds_on(ASOF).set_index("isin")
    fut = float(load_fgbl().set_index(["business_date", "contract"]).loc[(ASOF, "20261208"), "settlement"])
    repo = float(load_estr().loc[str(ASOF)]) / 100
    deliv = delivery_date(2026, 12)
    dl = load_deliverables()
    snap["fgbl_dec26"] = {}
    for isin in dl[(dl["product"] == "FGBL") & (dl["contract"] == "20261208")]["isin"]:
        bond = q.loc[isin, "bond"]
        cf = round(conversion_factor(bond, deliv), 6)
        b = basis(bond, float(q.loc[isin, "clean"]), q.loc[isin, "settle"], deliv, fut, cf, repo)
        snap["fgbl_dec26"][isin] = {"net_basis": b.net_basis, "implied_repo": b.implied_repo}
    ust = load_ust_par()[["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]].dropna()
    snap["pca_explained"] = {
        "USD": principal_components((ust.diff().dropna() * 100).to_numpy()).explained.tolist(),
        "EUR": principal_components((load_ecb_spot().dropna().diff().dropna() * 100).to_numpy()).explained.tolist(),
    }
    return snap


def test_golden_snapshot(usd_markets, eur_markets, bund_quotes):
    snap = _snapshot(usd_markets, eur_markets, bund_quotes)
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("golden values rewritten")
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert snap["asof"] == gold["asof"]
    assert snap["eur_selection"] == gold["eur_selection"]
    for m in METHODS:
        np.testing.assert_allclose(snap["usd"][m]["zeros"], gold["usd"][m]["zeros"], rtol=0, atol=1e-11)
        np.testing.assert_allclose(snap["eur"][m]["zeros"], gold["eur"][m]["zeros"], rtol=0, atol=1e-11)
        mine, theirs = snap["eur"][m]["oos_residual_bp"], gold["eur"][m]["oos_residual_bp"]
        assert mine.keys() == theirs.keys()
        np.testing.assert_allclose([mine[k] for k in theirs], list(theirs.values()), rtol=0, atol=1e-6)
    for isin, ref in gold["fgbl_dec26"].items():
        assert snap["fgbl_dec26"][isin]["net_basis"] == pytest.approx(ref["net_basis"], abs=1e-10)
        assert snap["fgbl_dec26"][isin]["implied_repo"] == pytest.approx(ref["implied_repo"], abs=1e-10)
    for ccy in ("USD", "EUR"):
        np.testing.assert_allclose(snap["pca_explained"][ccy], gold["pca_explained"][ccy], rtol=1e-10)
