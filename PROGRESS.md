# PROGRESS

| Date | Phase | Output | Status |
|---|---|---|---|
| 2026-10-13 | 0 Audit | `AUDIT.md`, `scripts/audit/` | 3 BLOCKER, 9 MAJOR, 8 MINOR, 3 NIT on `247592a` |
| 2026-10-28 | Decisions | `AUDIT.md` §3 | D1-D6 taken (recommended options) |
| 2026-10-28 | 1 Structure | `src/ratescurve/` (13 modules), `pyproject.toml` with exact pins, legacy script removed | Done |
| 2026-10-28 | Data | `scripts/fetch_data.py`, `scripts/freeze_sample.py`, `data/sample/` (as of 2026-10-27, SHA-256 manifest) | Done |
| 2026-10-28 | 2 Tests | 83 tests: conventions, Bundesbank and Eurex replication, QuantLib and tf-quant-finance oracles, risk identities | Done |
| 2026-10-28 | 3 Quality | ruff, mypy, GitHub Actions on Linux (3.11-3.13) and Windows (3.12) | Done; Windows fixed on 2026-10-29 (`.gitattributes`) |
| 2026-10-29 | 2 Tests (hardening) | bucket additivity on an off-pillar bond, monotone convex non-linearity, out-of-sample residuals, golden regression | 92 tests, coverage 97% |
| 2026-10-29 | 4 Results | `scripts/run_results.py`: R1-R4, `docs/results.md`, `docs/key_results.json`, `docs/figures/`; claims checked by `ensure()` | Done; `--check` in CI |
| 2026-10-29 | 5 README | English, standard order, key-results block generated | Done |
| 2026-10-29 | 6 Delivery | `notebooks/demo.ipynb` (runs in seconds, checked in CI); release notes drafted | Notebook done; tag `v1.0` not pushed (the session's git proxy refused tag pushes) |

## Open items

1. Tag and release by hand: `git tag -a v1.0 -m "v1.0" && git push origin v1.0`, then create the GitHub release from the tag with the drafted notes.
2. Pin the repository on the GitHub profile (profile setting, done by hand).
3. Publish the LinkedIn post (draft delivered outside the repository).
4. Optional extensions, not started: Bund curve history for a Bund-only PCA (daily Bundesbank files), delivery option value in the net basis, an OIS curve if a free official source of term quotes appears.

## Working rules

Before a session: read `AUDIT.md` and this file. After: update this file. Every published number comes from `scripts/run_results.py`; run `python scripts/run_results.py --check` and `pytest` before pushing.
