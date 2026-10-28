# Phase 0 audit scripts

These two scripts reproduce the figures quoted in `AUDIT.md`. They import the
legacy single-file pipeline (`yield_curve.py`), which was removed in Phase 1, so
they run against the audited commit:

```bash
git worktree add ../yield-curve-audit 247592a
cp scripts/audit/*.py ../yield-curve-audit/
cd ../yield-curve-audit
python quantify_phase0.py
python check_gapfill_claim.py
```
