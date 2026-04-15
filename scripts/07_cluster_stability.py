#!/usr/bin/env python3
"""
Script 07: Cluster Stability Validation
========================================

Runs three stability checks on the three-cluster Ward solution used in
the paper and writes the outputs to
``results/<run>/cluster_stability/``.

- coassignment_matrix.csv: bootstrap probability that each framework pair
  ends up in the same cluster (B resamples of the 200 samples).
- k_selection.csv: silhouette width and gap statistic for k in {2..8}.
- linkage_comparison.json: ARI between Ward and average/complete/single
  linkage at k=3.

Usage:
    python scripts/07_cluster_stability.py
    python scripts/07_cluster_stability.py --results-dir results/run_20260217_105538 --n-boot 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import MetricsAnalyzer
from src.cluster_stability import (
    bootstrap_coassignment,
    silhouette_and_gap,
    linkage_comparison,
)


def get_results_dir(specified_dir=None):
    if specified_dir and os.path.exists(specified_dir):
        return specified_dir
    base = "results"
    if os.path.exists(base):
        runs = sorted(
            d for d in os.listdir(base)
            if d.startswith("run_")
            and os.path.isdir(os.path.join(base, d))
            and os.path.exists(os.path.join(base, d, "evaluation_scores.csv"))
        )
        if runs:
            return os.path.join(base, runs[-1])
    raise FileNotFoundError("No results directory with evaluation_scores.csv found.")


def main(results_dir=None, n_boot=1000, n_reference=20):
    results_dir = get_results_dir(results_dir)
    print(f"Results directory: {results_dir}")

    df = pd.read_csv(os.path.join(results_dir, "evaluation_scores.csv"))
    analyzer = MetricsAnalyzer(df)
    primary = analyzer._select_primary_metric_cols()
    print(f"Primary metrics: {len(primary)}")

    out_dir = os.path.join(results_dir, "cluster_stability")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nBootstrap co-assignment ({n_boot} resamples)...")
    coassign = bootstrap_coassignment(df, primary, n_boot=n_boot)
    coassign.to_csv(os.path.join(out_dir, "coassignment_matrix.csv"))
    diag_mean = coassign.values[
        ~pd.DataFrame(coassign).isna().values
    ].mean()
    print(f"Mean co-assignment probability: {diag_mean:.3f}")

    print("\nSilhouette + gap statistic over k in {2..8}...")
    k_select = silhouette_and_gap(df, primary, n_reference=n_reference)
    k_select.to_csv(os.path.join(out_dir, "k_selection.csv"), index=False)
    print(k_select.round(3).to_string(index=False))

    print("\nLinkage comparison at k=3...")
    link = linkage_comparison(df, primary, k=3)
    with open(os.path.join(out_dir, "linkage_comparison.json"), "w") as f:
        json.dump(link, f, indent=2)
    for key, val in link.items():
        if key.startswith("ari_"):
            print(f"  {key}: {val:.3f}")

    print(f"\nWrote cluster stability outputs to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-reference", type=int, default=20)
    args = parser.parse_args()
    main(args.results_dir, n_boot=args.n_boot, n_reference=args.n_reference)
