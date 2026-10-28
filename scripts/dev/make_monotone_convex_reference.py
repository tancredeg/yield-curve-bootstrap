"""Generate tests/fixtures/monotone_convex_reference.json with an independent implementation.

Developer-only script (not part of the project dependencies). It evaluates the
Hagan-West monotone convex interpolation of tf-quant-finance
(``tf_quant_finance.rates.hagan_west.monotone_convex``, which does not apply the
positivity collar) on six pillar sets; the test suite compares
``ratescurve.interpolation.MonotoneConvex(positive=False)`` against the stored values.

Run in a separate environment::

    pip install tensorflow-cpu tf-quant-finance tf_keras
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/dev/make_monotone_convex_reference.py \
        tests/fixtures/monotone_convex_reference.json
"""

import json
import sys

import numpy as np
from tf_quant_finance.rates.hagan_west import monotone_convex as mc

rng = np.random.default_rng(0)
cases = []
times = np.array([91, 182, 365, 730, 1095, 1826, 2557, 3652, 5479, 7305, 10957]) / 365.0
for k in range(6):
    if k == 0:
        zeros = 0.03 + 0.01 * np.log1p(times) / 3 + rng.normal(0, 0.0015, times.size)
    elif k == 1:
        zeros = np.full(times.size, 0.025)
    else:
        zeros = 0.02 + 0.015 * rng.random(times.size)
    tt = np.linspace(0.0, times[-1], 1201)
    y = mc.interpolate_yields(tt, times, yields=zeros, dtype=np.float64).numpy()
    f = mc.interpolate_forward_rate(tt, times, yields=zeros, dtype=np.float64).numpy()
    cases.append(
        {"times": times.tolist(), "zeros": zeros.tolist(), "t": tt.tolist(), "zero": y.tolist(), "forward": f.tolist()}
    )
with open(sys.argv[1], "w") as fh:
    json.dump(
        {"source": "tf-quant-finance 0.0.1.dev34, tf_quant_finance.rates.hagan_west.monotone_convex", "cases": cases},
        fh,
    )
print("ok", len(cases))
