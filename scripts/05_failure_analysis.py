#!/usr/bin/env python3
"""
Script 05: Failure Analysis
===========================

Identify and analyze cases where frameworks disagree significantly.
Generates qualitative analysis for paper discussion.

Usage:
    python scripts/05_failure_analysis.py
    python scripts/05_failure_analysis.py --results-dir results/run_20240101_120000
"""

import sys
import os
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.failure_analysis import FailureAnalyzer


def get_results_dir(specified_dir=None):
    """Get the results directory to use."""
    if specified_dir and os.path.exists(specified_dir):
        return specified_dir

    # Try 'latest' symlink
    latest = 'results/latest'
    if os.path.exists(latest):
        return latest

    # Find the most recent run_* directory
    results_base = 'results'
    if os.path.exists(results_base):
        run_dirs = [d for d in os.listdir(results_base)
                   if d.startswith('run_') and os.path.isdir(os.path.join(results_base, d))]
        if run_dirs:
            # Sort by name (timestamp format ensures chronological order)
            latest_run = sorted(run_dirs)[-1]
            latest_run_path = os.path.join(results_base, latest_run)
            # Verify it has evaluation_scores.csv
            if os.path.exists(os.path.join(latest_run_path, 'evaluation_scores.csv')):
                return latest_run_path

    # Fallback to results/ if evaluation_scores.csv exists there
    if os.path.exists('results/evaluation_scores.csv'):
        return 'results'

    raise FileNotFoundError("No results found. Run 02_run_evaluation.py first.")


def main(results_dir=None):
    print("="*60)
    print("RAG Meta-Analysis: Failure Case Analysis")
    print("="*60)
    
    # Get results directory
    results_dir = get_results_dir(results_dir)
    print(f"\nUsing results from: {results_dir}")
    
    # Load results
    print("\nLoading data...")
    df = pd.read_csv(f'{results_dir}/evaluation_scores.csv')
    
    with open('data/samples_200.json', 'r') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(df)} evaluation results and {len(samples)} samples")
    
    # Initialize analyzer
    analyzer = FailureAnalyzer(df, samples)
    
    # Print summary
    analyzer.print_summary()
    
    # Export detailed results
    print("\n" + "="*60)
    print("EXPORTING FAILURE ANALYSIS")
    print("="*60)
    
    export_data = analyzer.export_failure_cases(f'{results_dir}/failure_cases.json', top_n=15)
    
    # Print some example cases for paper
    print("\n" + "="*60)
    print("EXAMPLE CASES FOR PAPER DISCUSSION")
    print("="*60)
    
    cases = analyzer.find_disagreement_cases(top_n=3)
    
    for i, case in enumerate(cases, 1):
        print(f"\n{'='*60}")
        print(f"CASE {i}: {case.sample_id}")
        print(f"{'='*60}")
        print(f"Domain: {case.domain}")
        print(f"Question: {case.question[:150]}...")
        print(f"\nAnswer preview: {case.answer[:200]}...")
        print(f"\nFramework Scores:")
        for fw, score in sorted(case.framework_scores.items(), key=lambda x: -x[1]):
            print(f"  {fw}: {score:.3f}")
        print(f"\nMax Disagreement: {case.max_disagreement:.3f}")
        print(f"Highest: {case.high_scorer} | Lowest: {case.low_scorer}")
        print(f"\nAnalysis: {case.analysis}")
    
    # Also show agreement cases for contrast
    print("\n" + "="*60)
    print("HIGH AGREEMENT CASES (for contrast)")
    print("="*60)
    
    agreement_cases = analyzer.find_agreement_cases(top_n=3)
    for case in agreement_cases:
        print(f"\n{case['id']} ({case['domain']})")
        print(f"  Disagreement: {case['disagreement']:.3f}")
        print(f"  Mean score: {case['mean_score']:.3f}")
    
    print("\n" + "="*60)
    print("Failure analysis complete!")
    print(f"Results saved to: {results_dir}/failure_cases.json")
    print("="*60)
    
    return analyzer, results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to results directory')
    args = parser.parse_args()
    
    analyzer, results_dir = main(args.results_dir)
