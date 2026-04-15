#!/usr/bin/env python3
"""
Script 06: Normalization Sensitivity Analysis
=============================================

Re-runs the paper's core analyses under four normalization schemes
(raw, min-max, z-score, rank) and reports how much the cluster
structure and heterogeneity statistics depend on the choice.

Outputs under ``results/<run>/normalization_sensitivity/``:

- summary.csv:             per-scheme heterogeneity and mean correlation
- cluster_assignments.csv: cluster labels per framework per scheme
- ari_matrix.csv:          adjusted Rand index between pairs of schemes
- correlation_<scheme>.csv: pairwise primary-metric correlation matrix

Usage:
    python scripts/06_normalization_sensitivity.py
    python scripts/06_normalization_sensitivity.py --results-dir results/run_20260217_105538
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import MetricsAnalyzer
from src.normalization import SCHEMES, apply_scheme


def get_results_dir(specified_dir=None):
    if specified_dir and os.path.exists(specified_dir):
        return specified_dir
    base = "results"
    if os.path.exists(base):
        runs = sorted(
            d for d in os.listdir(base)
            if d.startswith("run_")
            and os.path.isdir(os.path.join(base, d))
            and os.path.exists(os.path.join(base, d, "evaluation_scores_raw.csv"))
        )
        if runs:
            return os.path.join(base, runs[-1])
    raise FileNotFoundError("No results directory with evaluation_scores_raw.csv found.")


def analyze_scheme(raw_df: pd.DataFrame, scheme: str):
    """Apply a normalization scheme and compute the statistics we care about."""
    df = apply_scheme(raw_df, scheme)
    analyzer = MetricsAnalyzer(df)

    het = analyzer.compute_heterogeneity()
    clust = analyzer.compute_hierarchical_clustering(n_clusters=3)

    primary = analyzer._select_primary_metric_cols()
    corr = df[primary].corr(method="pearson")

    upper = corr.values[np.triu_indices_from(corr.values, k=1)]
    mean_r = float(np.nanmean(upper))

    return {
        "scheme": scheme,
        "Q": het.get("Q_statistic"),
        "Q_df": het.get("Q_df"),
        "Q_pvalue": het.get("Q_pvalue"),
        "I_squared": het.get("I_squared"),
        "mean_pairwise_r": mean_r,
        "within_cluster_r_mean": np.mean([
            v["within_cluster_mean_r"]
            for v in clust.get("clusters", {}).values()
        ]) if clust.get("clusters") else np.nan,
        "between_cluster_r": clust.get("between_cluster_mean_r"),
        "cluster_assignments": clust.get("cluster_assignments", {}),
        "primary_cols": primary,
        "correlation_matrix": corr,
    }


def main(results_dir=None):
    results_dir = get_results_dir(results_dir)
    print(f"Results directory: {results_dir}")

    raw_path = os.path.join(results_dir, "evaluation_scores_raw.csv")
    raw_df = pd.read_csv(raw_path)
    print(f"Loaded raw scores: {len(raw_df)} rows, {len(raw_df.columns)} columns")

    out_dir = os.path.join(results_dir, "normalization_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for scheme in SCHEMES:
        print(f"\n--- scheme: {scheme} ---")
        res = analyze_scheme(raw_df, scheme)
        results[scheme] = res
        print(f"Q = {res['Q']:.2f}, I^2 = {res['I_squared']:.2f}%, "
              f"mean r = {res['mean_pairwise_r']:.3f}, "
              f"between-cluster r = {res['between_cluster_r']}")

        res["correlation_matrix"].to_csv(
            os.path.join(out_dir, f"correlation_{scheme}.csv")
        )

    summary_rows = [
        {
            "scheme": r["scheme"],
            "Q": r["Q"],
            "Q_df": r["Q_df"],
            "Q_pvalue": r["Q_pvalue"],
            "I_squared_pct": r["I_squared"],
            "mean_pairwise_r": r["mean_pairwise_r"],
            "within_cluster_r_mean": r["within_cluster_r_mean"],
            "between_cluster_r": r["between_cluster_r"],
        }
        for r in results.values()
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "summary.csv"), index=False
    )

    # Cluster assignment table (rows = frameworks, columns = scheme)
    all_fws = sorted({
        fw for r in results.values() for fw in r["cluster_assignments"]
    })
    assign_rows = []
    for fw in all_fws:
        row = {"framework": fw}
        for scheme in SCHEMES:
            row[scheme] = results[scheme]["cluster_assignments"].get(fw)
        assign_rows.append(row)
    assign_df = pd.DataFrame(assign_rows)
    assign_df.to_csv(os.path.join(out_dir, "cluster_assignments.csv"), index=False)

    # Adjusted Rand index between every pair of schemes
    scheme_list = list(SCHEMES.keys())
    ari = pd.DataFrame(
        np.eye(len(scheme_list)),
        index=scheme_list,
        columns=scheme_list,
    )
    for i, s1 in enumerate(scheme_list):
        for j, s2 in enumerate(scheme_list):
            if i >= j:
                continue
            labels_1 = [results[s1]["cluster_assignments"].get(fw, -1) for fw in all_fws]
            labels_2 = [results[s2]["cluster_assignments"].get(fw, -1) for fw in all_fws]
            score = adjusted_rand_score(labels_1, labels_2)
            ari.loc[s1, s2] = score
            ari.loc[s2, s1] = score
    ari.to_csv(os.path.join(out_dir, "ari_matrix.csv"))

    print(f"\nWrote sensitivity outputs to {out_dir}")
    print("Adjusted Rand index between schemes:")
    print(ari.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    main(args.results_dir)
