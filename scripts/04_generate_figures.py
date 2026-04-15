#!/usr/bin/env python3
"""
Script 04: Generate Figures
===========================

Generate all publication-quality figures including forest plots.

Usage:
    python scripts/04_generate_figures.py
    python scripts/04_generate_figures.py --results-dir results/run_20240101_120000
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import Visualizer


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
    print("RAG Meta-Analysis: Figure Generation")
    print("="*60)
    
    # Get results directory
    results_dir = get_results_dir(results_dir)
    print(f"\nUsing results from: {results_dir}")
    
    # Create figures subdirectory in results
    figures_dir = f'{results_dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load results
    print("\nLoading evaluation results...")
    df = pd.read_csv(f'{results_dir}/evaluation_scores.csv')
    print(f"Loaded {len(df)} samples")
    
    # Load bootstrap CIs if available
    bootstrap_ci = None
    bootstrap_path = f'{results_dir}/bootstrap_ci.csv'
    if os.path.exists(bootstrap_path):
        bootstrap_ci = pd.read_csv(bootstrap_path)
        print(f"Loaded bootstrap CIs for {len(bootstrap_ci)} comparisons")
    
    # Initialize visualizer
    viz = Visualizer(df, output_dir=figures_dir)
    
    # Generate all figures
    viz.generate_all_figures(bootstrap_ci_df=bootstrap_ci)
    
    # List generated figures
    print("\n" + "="*60)
    print("Generated Figures")
    print("="*60)
    
    for f in sorted(os.listdir(figures_dir)):
        if f.endswith(('.pdf', '.png')):
            size = os.path.getsize(f'{figures_dir}/{f}') / 1024
            print(f"  - {f} ({size:.1f} KB)")
    
    print("\n" + "="*60)
    print("Figure generation complete!")
    print(f"Figures saved to: {figures_dir}")
    print("="*60)
    
    return figures_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to results directory')
    args = parser.parse_args()
    
    figures_dir = main(args.results_dir)
