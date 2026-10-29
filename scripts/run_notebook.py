"""Execute ``notebooks/demo.ipynb`` end to end and fail if it errors or takes two minutes or more.

Usage::

    python scripts/run_notebook.py            # execute, report the run time
    python scripts/run_notebook.py --write    # also save the executed notebook (outputs included)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "demo.ipynb"
LIMIT_S = 120.0


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="save the executed notebook in place")
    args = parser.parse_args()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    resources = {"metadata": {"path": str(ROOT)}}
    client = NotebookClient(nb, timeout=int(LIMIT_S), kernel_name="python3", resources=resources)
    start = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - start
    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)
    if args.write:
        nbformat.write(nb, NOTEBOOK)
    print(f"{NOTEBOOK.relative_to(ROOT)} executed in {elapsed:.1f} s (limit {LIMIT_S:.0f} s)")
    if elapsed >= LIMIT_S:
        sys.exit(1)


if __name__ == "__main__":
    main()
