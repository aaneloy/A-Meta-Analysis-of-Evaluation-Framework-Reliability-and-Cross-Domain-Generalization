"""
Cross-cluster consensus protocol.

Defines the decision rule the paper advocates in Section 6.3 for
practitioners who want a single actionable verdict from a set of
disagreeing frameworks. The rule is intentionally simple and reads
straight from the three-cluster structure established in Section 5.

Given:
  - the 20x200 primary score matrix,
  - the cluster assignment produced by Ward's method at k=3,
  - a per-cluster representative framework,
  - a binarization threshold tau,

we define three per-sample outcomes:

  FAITHFUL    - every cluster representative votes 1 at threshold tau.
  UNFAITHFUL  - every cluster representative votes 0 at threshold tau.
  CONTESTED   - representatives disagree. The paper reports this as an
                explicit uncertainty band rather than forcing a verdict.

The representative of each cluster is the framework whose mean
within-cluster Pearson correlation is highest, which is a cheap proxy
for "most central" and requires nothing beyond the already-computed
correlation matrix.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


FAITHFUL = "faithful"
UNFAITHFUL = "unfaithful"
CONTESTED = "contested"


def pick_cluster_representatives(
    correlation_matrix: pd.DataFrame,
    cluster_assignments: Dict[str, int],
) -> Dict[int, str]:
    """
    For each cluster, return the framework whose mean pairwise Pearson
    correlation with the other members of the same cluster is highest.
    Singleton clusters return their single member.
    """
    representatives: Dict[int, str] = {}
    clusters: Dict[int, List[str]] = {}
    for fw, c in cluster_assignments.items():
        clusters.setdefault(c, []).append(fw)

    for c, members in clusters.items():
        if len(members) == 1:
            representatives[c] = members[0]
            continue
        best_fw = members[0]
        best_score = -np.inf
        for fw in members:
            others = [m for m in members if m != fw]
            r_values = []
            for o in others:
                if fw in correlation_matrix.index and o in correlation_matrix.columns:
                    r_values.append(float(correlation_matrix.loc[fw, o]))
            score = float(np.mean(r_values)) if r_values else -np.inf
            if score > best_score:
                best_score = score
                best_fw = fw
        representatives[c] = best_fw
    return representatives


def _framework_to_column(scores: pd.DataFrame, framework: str) -> str:
    """Pick the primary column for a framework from a scores DataFrame."""
    preferred = [
        "faithfulness", "factual_consistency", "consistency", "adherence",
        "nli_ensemble_score", "f1", "overall", "score",
    ]
    fw_cols = [c for c in scores.columns if c.startswith(f"{framework}_")]
    if not fw_cols:
        raise KeyError(f"No columns found for framework {framework!r}.")
    for key in preferred:
        chosen = next((c for c in fw_cols if key in c.lower()), None)
        if chosen:
            return chosen
    return fw_cols[0]


def consensus_verdicts(
    scores: pd.DataFrame,
    representatives: Dict[int, str],
    tau: float = 0.5,
) -> pd.DataFrame:
    """
    Apply the cross-cluster consensus rule to every row of ``scores``.

    Returns a DataFrame with columns id, domain, verdict, and one binary
    vote column per cluster representative.
    """
    keep_cols = [c for c in ("id", "domain", "subset") if c in scores.columns]
    out = scores[keep_cols].copy()

    rep_cols: List[Tuple[int, str, str]] = []
    for cluster_id, fw in representatives.items():
        col = _framework_to_column(scores, fw)
        rep_cols.append((cluster_id, fw, col))

    vote_frames = []
    for cluster_id, fw, col in rep_cols:
        votes = (scores[col] >= tau).astype(int)
        name = f"vote_cluster_{cluster_id}_{fw}"
        out[name] = votes
        vote_frames.append(votes)

    votes_matrix = np.stack([v.to_numpy() for v in vote_frames], axis=1)
    all_one = np.all(votes_matrix == 1, axis=1)
    all_zero = np.all(votes_matrix == 0, axis=1)
    verdict = np.where(all_one, FAITHFUL,
                np.where(all_zero, UNFAITHFUL, CONTESTED))
    out["verdict"] = verdict
    return out


def coverage_report(verdicts: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics for the consensus verdicts."""
    n = len(verdicts)
    counts = verdicts["verdict"].value_counts()
    return {
        "n": int(n),
        "faithful_pct": float(counts.get(FAITHFUL, 0)) / n * 100.0 if n else 0.0,
        "unfaithful_pct": float(counts.get(UNFAITHFUL, 0)) / n * 100.0 if n else 0.0,
        "contested_pct": float(counts.get(CONTESTED, 0)) / n * 100.0 if n else 0.0,
    }


def evaluate_against_human(
    verdicts: pd.DataFrame,
    human_labels: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compare verdicts to human labels on the intersection of ids.

    Accuracy is computed only over non-contested verdicts, which is the
    operational interpretation of the uncertainty band. Coverage is the
    fraction of samples for which the protocol returned a non-contested
    verdict.
    """
    merged = verdicts.merge(
        human_labels[["id", "human_label"]],
        on="id",
        how="inner",
    )
    n_total = len(merged)
    if n_total == 0:
        return {"n": 0}

    decided = merged[merged["verdict"] != CONTESTED].copy()
    if len(decided) == 0:
        return {
            "n": n_total,
            "coverage": 0.0,
            "accuracy_on_decided": float("nan"),
        }
    decided["predicted"] = (decided["verdict"] == FAITHFUL).astype(int)
    acc = float((decided["predicted"] == decided["human_label"].astype(int)).mean())
    return {
        "n": n_total,
        "n_decided": int(len(decided)),
        "coverage": float(len(decided)) / n_total,
        "accuracy_on_decided": acc,
    }
