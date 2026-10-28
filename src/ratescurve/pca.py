"""Principal component analysis of daily yield curve changes.

For a ``T x N`` matrix of daily changes ``X`` (one column per tenor), the
sample covariance ``S = X_c' X_c / (T - 1)`` is decomposed as ``S = V L V'``;
column ``k`` of ``V`` is the loading of component ``k`` and ``L_k / tr(S)`` its
share of total variance. Signs are fixed so the factors read as level
(loadings sum positive), slope (long end minus short end positive) and
curvature (middle minus wings positive).

Reference: R. Litterman and J. Scheinkman (1991), "Common factors affecting
bond returns", Journal of Fixed Income 1(1), 54-61.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PCAResult:
    """Eigenvalues (descending), loadings (columns), variance shares, sample size."""

    eigenvalues: np.ndarray
    loadings: np.ndarray
    explained: np.ndarray
    n_obs: int


def _align(v: np.ndarray) -> np.ndarray:
    v = v.copy()
    n = v.shape[0]
    if v[:, 0].sum() < 0:
        v[:, 0] *= -1
    if v.shape[1] > 1 and v[-1, 1] - v[0, 1] < 0:
        v[:, 1] *= -1
    if v.shape[1] > 2:
        mid = v[n // 3 : 2 * n // 3 + 1, 2].mean()
        wings = 0.5 * (v[0, 2] + v[-1, 2])
        if mid - wings < 0:
            v[:, 2] *= -1
    return v


def principal_components(changes: ArrayLike, k: int = 3) -> PCAResult:
    """PCA of curve changes; ``changes`` is ``T x N`` with no missing values."""
    x = np.asarray(changes, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("changes must be a T x N matrix with T >= 2")
    if not np.all(np.isfinite(x)):
        raise ValueError("changes contain missing values")
    cov = np.cov(x, rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = _align(vecs[:, order])
    return PCAResult(vals[:k], vecs[:, :k], vals[:k] / np.trace(cov), x.shape[0])


def loading_similarity(a: PCAResult, b: PCAResult) -> np.ndarray:
    """|cosine| between matching loadings of two fits (1 = identical shape)."""
    num = np.abs(np.sum(a.loadings * b.loadings, axis=0))
    return np.asarray(num / (np.linalg.norm(a.loadings, axis=0) * np.linalg.norm(b.loadings, axis=0)))
