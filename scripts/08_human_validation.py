#!/usr/bin/env python3
"""
Script 08: Human Validation Pipeline
====================================

Two entry points:

  --select    Build a stratified 50-sample annotation template from
              data/samples_200.json and the latest evaluation run.
              Writes data/human_validation_template.csv and a manifest.

  --analyze   Read data/human_validation_filled.csv (provided by the
              annotator) and compute per-framework correlation with the
              human labels, per-cluster alignment, and intra-rater
              kappa from an optional re-annotation file.

Usage:
    python scripts/08_human_validation.py --select
    python scripts/08_human_validation.py --analyze
    python scripts/08_human_validation.py --analyze \\
        --filled data/human_validation_filled.csv \\
        --reannot data/human_validation_reannot.csv \\
        --results-dir results/run_20260217_105538
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.human_validation import (
    DOMAIN_QUOTAS,
    build_annotation_template,
    cluster_alignment,
    framework_human_correlation,
    intra_rater_reliability,
    select_validation_subset,
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


def do_select(args):
    with open("data/samples_200.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    results_dir = get_results_dir(args.results_dir)
    scores = pd.read_csv(os.path.join(results_dir, "evaluation_scores.csv"))

    selected = select_validation_subset(samples, scores, quotas=DOMAIN_QUOTAS)
    template = build_annotation_template(selected)

    out_path = args.output or "data/human_validation_template.csv"
    template.to_csv(out_path, index=False)

    manifest = {
        "results_dir": results_dir,
        "quotas": DOMAIN_QUOTAS,
        "n_selected": len(template),
        "ids": template["id"].tolist(),
    }
    with open("data/human_validation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(template)} samples to {out_path}")
    print("Domain breakdown:")
    print(template["domain"].value_counts().to_string())


def do_analyze(args):
    results_dir = get_results_dir(args.results_dir)
    scores = pd.read_csv(os.path.join(results_dir, "evaluation_scores.csv"))

    filled_path = args.filled or "data/human_validation_filled.csv"
    if not os.path.exists(filled_path):
        raise FileNotFoundError(
            f"Annotation file not found at {filled_path}. "
            "Run --select, fill in the CSV, and re-run with --analyze."
        )
    filled = pd.read_csv(filled_path)
    filled = filled.dropna(subset=["human_label"])
    filled["human_label"] = filled["human_label"].astype(int)

    labelled_scores = scores.merge(
        filled[["id", "human_label"]].rename(columns={"human_label": "human_label_new"}),
        on="id",
        how="inner",
    )
    labelled_scores = labelled_scores.drop(columns=[c for c in ["human_label"] if c in labelled_scores.columns])
    labelled_scores = labelled_scores.rename(columns={"human_label_new": "human_label"})
    print(f"Merged on {len(labelled_scores)} labelled samples.")

    out_dir = os.path.join(results_dir, "human_validation")
    os.makedirs(out_dir, exist_ok=True)

    corr_df = framework_human_correlation(labelled_scores, n_boot=args.n_boot)
    corr_df.to_csv(os.path.join(out_dir, "framework_human_correlation.csv"), index=False)
    print("\nTop 5 frameworks by Pearson r with human:")
    print(corr_df.head(5).to_string(index=False))

    cluster_path = os.path.join(results_dir, "clustering.json")
    if os.path.exists(cluster_path):
        with open(cluster_path, "r", encoding="utf-8") as f:
            cluster_info = json.load(f)
        assignments = cluster_info.get("cluster_assignments", {})
        if assignments:
            alignment = cluster_alignment(corr_df, assignments)
            alignment.to_csv(os.path.join(out_dir, "cluster_alignment.csv"), index=False)
            print("\nPer-cluster mean human correlation:")
            print(alignment.round(3).to_string(index=False))

    if args.reannot and os.path.exists(args.reannot):
        reannot = pd.read_csv(args.reannot).dropna(subset=["human_label"])
        reannot["human_label"] = reannot["human_label"].astype(int)
        iaa = intra_rater_reliability(filled, reannot)
        with open(os.path.join(out_dir, "intra_rater_reliability.json"), "w", encoding="utf-8") as f:
            json.dump(iaa, f, indent=2)
        print(f"\nIntra-rater kappa: {iaa['cohens_kappa']:.3f} "
              f"(n = {iaa['n_reannotated']})")

    print(f"\nWrote human validation outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None,
                        help="Output path for the annotation template.")
    parser.add_argument("--filled", default=None,
                        help="Path to the filled annotation CSV.")
    parser.add_argument("--reannot", default=None,
                        help="Optional path to re-annotation CSV for intra-rater kappa.")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    if args.select == args.analyze:
        parser.error("Pass exactly one of --select or --analyze.")

    if args.select:
        do_select(args)
    else:
        do_analyze(args)


if __name__ == "__main__":
    main()
