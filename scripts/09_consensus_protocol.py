#!/usr/bin/env python3
"""
Script 09: Cross-Cluster Consensus Protocol
===========================================

Runs the decision rule defined in src/consensus.py on the full 200
samples and, when available, evaluates it against the human validation
labels produced by Script 08.

Outputs under ``results/<run>/consensus/``:

- representatives.json:   chosen representative framework per cluster
- verdicts.csv:           per-sample verdict with per-cluster votes
- coverage.json:          fraction faithful / unfaithful / contested
- human_agreement.json:   accuracy and coverage against human labels
                          (only if data/human_validation_filled.csv exists)

Usage:
    python scripts/09_consensus_protocol.py
    python scripts/09_consensus_protocol.py --tau 0.5 --results-dir results/run_20260217_105538
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.consensus import (
    consensus_verdicts,
    coverage_report,
    evaluate_against_human,
    pick_cluster_representatives,
)
from src.metrics import MetricsAnalyzer


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


def main(results_dir=None, tau=0.5, filled=None):
    results_dir = get_results_dir(results_dir)
    print(f"Results directory: {results_dir}")

    scores = pd.read_csv(os.path.join(results_dir, "evaluation_scores.csv"))

    cluster_path = os.path.join(results_dir, "clustering.json")
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(
            f"{cluster_path} not found. Run scripts/03_analyze_results.py first."
        )
    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_info = json.load(f)
    assignments = cluster_info.get("cluster_assignments", {})
    if not assignments:
        raise RuntimeError("clustering.json has no cluster_assignments.")

    analyzer = MetricsAnalyzer(scores)
    primary = analyzer._select_primary_metric_cols()
    corr = scores[primary].corr(method="pearson")
    # Rename rows/columns to the bare framework name so lookups match
    # the cluster_assignments dict.
    corr.index = [c.split("_")[0] for c in corr.index]
    corr.columns = [c.split("_")[0] for c in corr.columns]

    representatives = pick_cluster_representatives(corr, assignments)
    print(f"Cluster representatives: {representatives}")

    out_dir = os.path.join(results_dir, "consensus")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "representatives.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in representatives.items()}, f, indent=2)

    verdicts = consensus_verdicts(scores, representatives, tau=tau)
    verdicts.to_csv(os.path.join(out_dir, "verdicts.csv"), index=False)

    report = coverage_report(verdicts)
    print("Coverage report:")
    print(json.dumps(report, indent=2))
    with open(os.path.join(out_dir, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    filled = filled or "data/human_validation_filled.csv"
    if os.path.exists(filled):
        human = pd.read_csv(filled).dropna(subset=["human_label"])
        human["human_label"] = human["human_label"].astype(int)
        agreement = evaluate_against_human(verdicts, human)
        print("Human agreement:")
        print(json.dumps(agreement, indent=2))
        with open(os.path.join(out_dir, "human_agreement.json"), "w", encoding="utf-8") as f:
            json.dump(agreement, f, indent=2)
    else:
        print(f"(No human labels at {filled}; skipping human agreement.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--filled", default=None)
    args = parser.parse_args()
    main(args.results_dir, tau=args.tau, filled=args.filled)
