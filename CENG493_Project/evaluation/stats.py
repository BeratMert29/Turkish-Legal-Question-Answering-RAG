"""
evaluation/stats.py — bootstrap confidence intervals and paired comparisons.

Operates on per-query score arrays so stage-to-stage differences can be
reported with uncertainty instead of bare means.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_RESAMPLES = 2000


def _clean(values) -> np.ndarray:
    arr = [float(v) for v in values if v is not None and not _isnan(v)]
    return np.asarray(arr, dtype=float)


def _isnan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


def bootstrap_ci(
    values: Sequence[float | None],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI of the mean. None/NaN values are dropped."""
    n_resamples = max(int(n_resamples), 1000)
    arr = _clean(values)
    n = int(arr.size)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0,
                "n_resamples": n_resamples, "confidence": confidence}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": n, "n_resamples": n_resamples, "confidence": confidence}


def _align(a, b) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = [k for k in a if k in b]
        pairs = [(a[k], b[k]) for k in keys]
    else:
        if len(a) != len(b):
            raise ValueError(
                f"paired_bootstrap needs aligned arrays, got {len(a)} vs {len(b)}"
            )
        pairs = list(zip(a, b))
    pairs = [(x, y) for x, y in pairs
             if x is not None and y is not None and not _isnan(x) and not _isnan(y)]
    if not pairs:
        return np.zeros(0), np.zeros(0)
    return (np.asarray([p[0] for p in pairs], dtype=float),
            np.asarray([p[1] for p in pairs], dtype=float))


def paired_bootstrap(
    a: Sequence[float | None] | Mapping[str, float | None],
    b: Sequence[float | None] | Mapping[str, float | None],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of mean(b - a) over the same queries.

    ``a`` is the earlier stage, ``b`` the later one. Inputs are either
    equal-length sequences aligned by query, or dicts keyed by query_id
    (intersected). Pairs with a missing value are dropped.
    ``p_value`` is the two-sided bootstrap probability that the mean
    difference crosses zero; ``significant`` is True when the CI excludes 0.
    """
    n_resamples = max(int(n_resamples), 1000)
    xa, xb = _align(a, b)
    n = int(xa.size)
    if n == 0:
        return {"mean_diff": None, "ci_low": None, "ci_high": None,
                "p_value": None, "significant": None, "n": 0,
                "n_resamples": n_resamples, "confidence": confidence}
    diff = xb - xa
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    p_le = (np.sum(means <= 0) + 1) / (n_resamples + 1)
    p_ge = (np.sum(means >= 0) + 1) / (n_resamples + 1)
    p = float(min(1.0, 2 * min(p_le, p_ge)))
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p, "significant": bool(lo > 0 or hi < 0), "n": n,
            "n_resamples": n_resamples, "confidence": confidence}


def holm_adjust(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm-Bonferroni adjusted p-values (family = the given tests).

    None entries are left as None and do not count towards the family size.
    """
    valid = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m = len(valid)
    out: dict[str, float | None] = {k: None for k in pvalues}
    running = 0.0
    for i, (p, k) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
