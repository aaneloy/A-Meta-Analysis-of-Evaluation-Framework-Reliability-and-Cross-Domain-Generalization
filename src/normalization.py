"""
Normalization schemes for cross-framework score comparison.

Provides four normalization strategies so the paper can report a
sensitivity analysis addressing reviewer comments on Section 4.2:

- raw:    no normalization, original framework outputs
- minmax: per-metric min-max scaling to [0, 1] (paper's primary choice)
- zscore: per-metric z-score standardization
- rank:   per-metric rank transform divided by n-1, yielding [0, 1]

Only primary metric columns (one per framework) are rescaled. Identifier
columns such as id, domain, subset, and human_label are preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List


ID_COLS = ("id", "domain", "subset", "human_label")


def _metric_columns(df: pd.DataFrame, exclude: Iterable[str] = ID_COLS) -> List[str]:
    return [c for c in df.columns if c not in set(exclude)]


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` without any rescaling."""
    return df.copy()


def normalize_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric min-max scaling to [0, 1]."""
    out = df.copy()
    for col in _metric_columns(out):
        x = out[col].astype(float).values
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi - lo < 1e-12:
            out[col] = 0.0
        else:
            out[col] = (x - lo) / (hi - lo)
    return out


def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric z-score standardization (mean 0, unit variance)."""
    out = df.copy()
    for col in _metric_columns(out):
        x = out[col].astype(float).values
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        if sd < 1e-12:
            out[col] = 0.0
        else:
            out[col] = (x - mu) / sd
    return out


def normalize_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric rank transform scaled to [0, 1]."""
    out = df.copy()
    for col in _metric_columns(out):
        ranks = out[col].rank(method="average").values.astype(float)
        n = np.sum(~np.isnan(ranks))
        if n <= 1:
            out[col] = 0.0
        else:
            out[col] = (ranks - 1.0) / (n - 1.0)
    return out


SCHEMES = {
    "raw": normalize_raw,
    "minmax": normalize_minmax,
    "zscore": normalize_zscore,
    "rank": normalize_rank,
}


def apply_scheme(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Dispatch helper. Raises ``ValueError`` on unknown scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown normalization scheme: {scheme!r}. "
                         f"Expected one of {sorted(SCHEMES)}.")
    return SCHEMES[scheme](df)
