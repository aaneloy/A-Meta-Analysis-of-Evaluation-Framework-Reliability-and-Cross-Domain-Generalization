"""
Human validation utilities.

Selects a stratified subset of 50 samples for annotation, computes
per-framework correlation with the resulting human labels, estimates
intra-rater Cohen's kappa on a re-annotation subset, and compares
each cluster's human alignment.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

DOMAIN_QUOTAS = {
    "General Knowledge": 17,
    "Finance": 17,
    "Biomedicine": 16,
}

SELECTION_SEED = 20260409


def _primary_columns(scores: pd.DataFrame) -> List[str]:
    """Heuristic copy of MetricsAnalyzer._select_primary_metric_cols."""
    preferred = [
        "faithfulness", "factual_consistency", "consistency", "adherence",
        "nli_ensemble_score", "f1", "overall", "score",
    ]
    exclude = {"id", "domain", "subset", "human_label"}
    metric_cols = [c for c in scores.columns if c not in exclude]
    frameworks: List[str] = []
    seen = set()
    for col in metric_cols:
        fw = col.split("_")[0]
        if fw not in seen:
            frameworks.append(fw)
            seen.add(fw)

    selected = []
    for fw in frameworks:
        fw_cols = [c for c in metric_cols if c.startswith(f"{fw}_")]
        chosen = None
        for key in preferred:
            chosen = next((c for c in fw_cols if key in c.lower()), None)
            if chosen:
                break
        if not chosen and fw_cols:
            chosen = fw_cols[0]
        if chosen:
            selected.append(chosen)
    return selected


def select_validation_subset(
    samples: List[dict],
    scores: pd.DataFrame,
    quotas: Dict[str, int] = DOMAIN_QUOTAS,
    seed: int = SELECTION_SEED,
) -> List[dict]:
    """
    Pick a stratified set of samples that spans both low- and
    high-disagreement regions in each domain.

    The disagreement signal is the max-minus-min range across the 20
    primary framework scores on that sample. Within each domain, we
    pick samples at five evenly spaced percentiles (10, 30, 50, 70, 90)
    and fill the remainder uniformly at random.
    """
    rng = np.random.default_rng(seed)
    primary = _primary_columns(scores)

    score_by_id = scores.set_index("id")
    ranges = score_by_id[primary].max(axis=1) - score_by_id[primary].min(axis=1)

    by_domain: Dict[str, List[dict]] = {}
    for s in samples:
        by_domain.setdefault(s["domain"], []).append(s)

    selected = []
    for domain, quota in quotas.items():
        pool = by_domain.get(domain, [])
        if not pool:
            continue
        ids = [s["id"] for s in pool if s["id"] in ranges.index]
        if not ids:
            continue
        local_ranges = ranges.loc[ids].sort_values()

        # Five anchor percentiles for breadth
        n = len(local_ranges)
        anchors = []
        for pct in (10, 30, 50, 70, 90):
            idx = int(round((pct / 100.0) * (n - 1)))
            anchors.append(local_ranges.index[idx])

        anchors = list(dict.fromkeys(anchors))  # dedupe preserving order
        remaining_quota = max(0, quota - len(anchors))
        remaining_ids = [i for i in ids if i not in set(anchors)]
        rng.shuffle(remaining_ids)
        picked_ids = anchors + remaining_ids[:remaining_quota]

        pool_by_id = {s["id"]: s for s in pool}
        for pid in picked_ids:
            selected.append(pool_by_id[pid])

    return selected


def build_annotation_template(selected: List[dict]) -> pd.DataFrame:
    """Return a CSV-ready DataFrame the annotator fills in."""
    rows = []
    for s in selected:
        rows.append({
            "id": s["id"],
            "domain": s["domain"],
            "subset": s.get("subset", ""),
            "question": s.get("question", ""),
            "context": s.get("context", ""),
            "answer": s.get("answer", ""),
            "human_label": "",        # 1 = faithful, 0 = unfaithful
            "confidence": "",         # low / medium / high
            "notes": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis after annotation
# ---------------------------------------------------------------------------

def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    mask = (~pd.isna(a)) & (~pd.isna(b))
    a, b = a[mask].astype(int), b[mask].astype(int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    p1 = float((a == 1).mean())
    p2 = float((b == 1).mean())
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if 1.0 - pe < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def intra_rater_reliability(
    first_pass: pd.DataFrame, second_pass: pd.DataFrame
) -> Dict[str, float]:
    """Intra-rater Cohen's kappa between two annotation passes."""
    merged = first_pass.merge(
        second_pass, on="id", suffixes=("_pass1", "_pass2")
    )
    a = merged["human_label_pass1"].to_numpy()
    b = merged["human_label_pass2"].to_numpy()
    return {
        "n_reannotated": int(len(merged)),
        "cohens_kappa": _cohens_kappa(a, b),
        "percent_agreement": float((a == b).mean()) if len(merged) else float("nan"),
    }


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20260409,
    method: str = "pearson",
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        if method == "pearson":
            r, _ = pearsonr(xb, yb)
        else:
            r, _ = spearmanr(xb, yb)
        vals.append(r)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return float(np.mean(vals)), lo, hi


def framework_human_correlation(
    labelled_scores: pd.DataFrame,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """
    Compute per-framework Pearson and Spearman correlation with the
    human label column. ``labelled_scores`` must contain a
    ``human_label`` column with 0/1 values and one column per primary
    framework metric.
    """
    if "human_label" not in labelled_scores.columns:
        raise KeyError("labelled_scores must contain a 'human_label' column.")

    y = labelled_scores["human_label"].astype(float).to_numpy()
    primary = _primary_columns(labelled_scores)

    rows = []
    for col in primary:
        x = labelled_scores[col].astype(float).to_numpy()
        if np.std(x) < 1e-12:
            continue
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        _, lo_p, hi_p = _bootstrap_corr_ci(x, y, n_boot=n_boot, method="pearson")
        _, lo_s, hi_s = _bootstrap_corr_ci(x, y, n_boot=n_boot, method="spearman")
        rows.append({
            "framework": col.split("_")[0],
            "metric": col,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "pearson_ci_low": lo_p,
            "pearson_ci_high": hi_p,
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
            "spearman_ci_low": lo_s,
            "spearman_ci_high": hi_s,
            "n": int(len(y)),
        })
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def cluster_alignment(
    human_corr: pd.DataFrame,
    cluster_assignments: Dict[str, int],
) -> pd.DataFrame:
    """Mean framework-human Pearson correlation per cluster."""
    labels = human_corr.copy()
    labels["cluster"] = labels["framework"].map(cluster_assignments)
    grouped = (
        labels.dropna(subset=["cluster"])
              .groupby("cluster")["pearson_r"]
              .agg(["mean", "std", "count"])
              .reset_index()
              .rename(columns={"mean": "mean_pearson_r", "std": "sd_pearson_r", "count": "n_frameworks"})
    )
    return grouped
