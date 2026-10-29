# CLAUDE.md

## Standard "recruiter repo" (common to the portfolio repositories)

Goal: a quant or trading recruiter understands in 90 seconds what the repository does and what it proves, and can rerun everything with two commands.

1. Audit: `AUDIT.md` lists issues by severity (BLOCKER / MAJOR / MINOR / NIT) and the plan.
2. Structure: `src/<package>/`, `tests/`, `notebooks/`, `data/sample/`, `scripts/`, `docs/`; `pyproject.toml` with pinned dependencies, `pip install -e .`; pure functions, type hints, docstrings with formula and reference, no absolute paths, fixed seeds.
3. Tests (pytest): financial consistency (identities, limit cases, convergence), regression on the frozen `data/sample/`, coverage of the computational core at least 80%.
4. Quality and CI: ruff (lint and format), mypy on `src/`, GitHub Actions on every push, badge in the README.
5. Results: `scripts/run_results.py` regenerates every figure and table of the README; two to four key results, each read in one sentence.
6. README in English, in this order: one sentence (problem and method); key results (a table and one or two figures); methodology with stated limits; reproduce in two commands; code structure; references.
7. Delivery: demo notebook running end to end in under two minutes; tag and release notes.

Working rules: read `AUDIT.md` and `PROGRESS.md` before a session and update `PROGRESS.md` after it; atomic commits with explicit messages; a validated plan before any large refactor; never invent a result: every number in the README comes from an executed script.

## This repository

- Public data only, frozen in `data/sample/` with SHA-256 checksums in `MANIFEST.json`. Refresh only through `scripts/fetch_data.py` and `scripts/freeze_sample.py`; never edit sample files by hand (`.gitattributes` keeps their bytes identical on Windows).
- Never type a result into `README.md` or `docs/`: change `scripts/run_results.py`, run it, and check `python scripts/run_results.py --check`. Data-dependent sentences go through `ensure()`.
- Before pushing: `ruff check .`, `ruff format --check .`, `mypy`, `pytest`, `python scripts/run_results.py --check`.
- Curves: the calibrated curve is the pricing curve; extrapolation beyond the last pillar raises. Monotone convex is only piecewise smooth in its quotes: compute its bucketed risk with small bumps (0.01 bp).
- Golden values (`tests/fixtures/golden.json`) change only after an intentional modelling change: `UPDATE_GOLDEN=1 pytest tests/test_regression.py`, then review the diff.
