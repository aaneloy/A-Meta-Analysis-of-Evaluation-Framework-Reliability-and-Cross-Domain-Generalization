#!/usr/bin/env python3
"""
Script 03: Analyze Results
==========================

Perform comprehensive statistical analysis including:
- Correlation analysis with bootstrap CIs
- Heterogeneity statistics
- Agreement metrics
- Domain-specific analysis

Usage:
    python scripts/03_analyze_results.py
    python scripts/03_analyze_results.py --results-dir results/run_20240101_120000
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import MetricsAnalyzer


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
    print("RAG Meta-Analysis: Statistical Analysis")
    print("="*60)
    
    # Get results directory
    results_dir = get_results_dir(results_dir)
    print(f"\nUsing results from: {results_dir}")
    
    # Load results
    print("\nLoading evaluation results...")
    df = pd.read_csv(f'{results_dir}/evaluation_scores.csv')
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Initialize analyzer
    analyzer = MetricsAnalyzer(df)
    
    # ==================== Heterogeneity Analysis ====================
    print("\n" + "="*60)
    print("HETEROGENEITY ANALYSIS (Meta-Analysis Statistics)")
    print("="*60)
    
    het = analyzer.compute_heterogeneity()
    print(f"\nCochran's Q: {het['Q_statistic']:.3f} (df={het['Q_df']}, p={het['Q_pvalue']:.4f})")
    print(f"I² statistic: {het['I_squared']:.1f}%")
    print(f"Interpretation: {het['interpretation']}")
    
    # ==================== Bootstrap Confidence Intervals ====================
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*60)
    
    print("\nComputing bootstrap CIs (1000 iterations)...")
    bootstrap_ci = analyzer.compute_all_bootstrap_ci(n_bootstrap=1000)
    
    print("\nFaithfulness Correlations with 95% CIs:")
    for _, row in bootstrap_ci.iterrows():
        print(f"  {row['framework_1']} vs {row['framework_2']}: "
              f"r={row['correlation']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    
    # ==================== Correlation Analysis ====================
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    faith_corr = analyzer.compute_faithfulness_correlations()
    print("\n--- Faithfulness Metric Correlations (Pearson) ---")
    print(faith_corr['pearson'].round(3).to_string())
    
    pairwise = analyzer.compute_pairwise_correlations()

    print("\nTop 10 Cross-Framework Correlations:")
    top_corr = pairwise.nlargest(10, 'pearson_r')[['framework_1', 'framework_2', 'metric_1', 'pearson_r', 'pearson_p']]
    print(top_corr.to_string(index=False))

    # Faithfulness framework correlations (for paper Table 2)
    print("\n--- Framework Faithfulness Correlations (for Paper Table 2) ---")
    faith_fw_corr = analyzer.compute_framework_faithfulness_correlations()
    if len(faith_fw_corr) > 0:
        print(faith_fw_corr[['framework_1', 'framework_2', 'pearson_r', 'spearman_rho']].to_string(index=False))
        print("\nNOTE: These are the actual computed correlations. Update paper tables to match.")
    
    # ==================== Agreement Analysis ====================
    print("\n" + "="*60)
    print("AGREEMENT ANALYSIS")
    print("="*60)
    
    agreement, kappa = analyzer.compute_agreement_matrix(threshold=0.5)
    threshold_robustness = analyzer.compute_threshold_robustness([0.3, 0.4, 0.5, 0.6, 0.7])
    
    print("\n--- Percent Agreement Matrix ---")
    print((agreement * 100).round(1).to_string())
    
    print("\n--- Cohen's Kappa Matrix ---")
    print(kappa.round(3).to_string())

    print("\n--- Threshold Robustness (vs τ=0.3) ---")
    print(threshold_robustness.round(3).to_string(index=False))
    

    # ==================== Human Correlation Analysis ====================
    print("\n" + "="*60)
    print("HUMAN CORRELATION ANALYSIS")
    print("="*60)

    human_corr = analyzer.compute_human_label_correlations()
    if len(human_corr) > 0:
        print("\n--- Top correlations vs human_label ---")
        print(human_corr.head(10).round(3).to_string(index=False))
    else:
        print("No binary human labels available in this dataset run.")

    # ==================== Cluster Analysis ====================
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS (Ward's Method)")
    print("="*60)

    clustering = analyzer.compute_hierarchical_clustering(n_clusters=3)
    if 'error' not in clustering:
        for cname, cinfo in clustering['clusters'].items():
            print(f"\n  {cname}: {cinfo['frameworks']}")
            print(f"    Size: {cinfo['size']}, Within-cluster mean r: {cinfo['within_cluster_mean_r']}")
        print(f"\n  Between-cluster mean r: {clustering['between_cluster_mean_r']}")
    else:
        print(f"  Clustering error: {clustering['error']}")

    # ==================== Domain Analysis ====================
    print("\n" + "="*60)
    print("DOMAIN ANALYSIS")
    print("="*60)

    domain_stats = analyzer.compute_domain_statistics()
    faith_stats = domain_stats[domain_stats['metric'].str.contains('faithfulness', case=False)]
    
    if len(faith_stats) > 0:
        print("\n--- Faithfulness by Domain ---")
        pivot = faith_stats.pivot(index='metric', columns='domain', values='mean')
        print(pivot.round(3).to_string())
    
    print("\n--- ANOVA: Domain Effect ---")
    anova = analyzer.compute_domain_anova()
    sig_anova = anova[anova['significant']]
    print(f"Significant domain effects found for {len(sig_anova)}/{len(anova)} metrics")
    print(sig_anova[['metric', 'f_statistic', 'p_value', 'eta_squared']].round(4).to_string(index=False))

    # ==================== Mixed-Effects Model ====================
    print("\n" + "="*60)
    print("MIXED-EFFECTS MODEL ANALYSIS")
    print("="*60)
    print("\nModel: score_ij = B0 + B1(domain_i) + u_j + e_ij")
    print("  B0: Fixed intercept")
    print("  B1: Fixed effect of domain")
    print("  u_j: Random effect for sample")
    print("  e_ij: Residual error\n")

    try:
        mixed_results = analyzer.compute_all_mixed_effects()
        if len(mixed_results) > 0:
            print("--- Mixed-Effects Results (Faithfulness Metrics) ---")
            faith_mixed = mixed_results[mixed_results['metric'].str.contains('faithfulness', case=False)]
            if len(faith_mixed) > 0:
                display_cols = ['metric', 'intercept', 'random_effects_var', 'icc', 'aic', 'converged']
                available_cols = [c for c in display_cols if c in faith_mixed.columns]
                print(faith_mixed[available_cols].round(4).to_string(index=False))

                # Show domain effects
                domain_cols = [c for c in mixed_results.columns if c.startswith('domain_effect_')]
                if domain_cols:
                    print("\n--- Domain Effects (relative to baseline) ---")
                    print(faith_mixed[['metric'] + domain_cols].round(4).to_string(index=False))

                print(f"\nIntraclass Correlation (ICC) interpretation:")
                print("  ICC measures proportion of variance attributable to sample-level differences")
                avg_icc = faith_mixed['icc'].mean()
                print(f"  Average ICC across faithfulness metrics: {avg_icc:.3f}")
            else:
                print("No faithfulness metrics found for mixed-effects analysis")
        else:
            print("Mixed-effects model computation returned no results")
    except Exception as e:
        print(f"Could not compute mixed-effects model: {e}")
        print("This may be due to missing statsmodels package or data issues.")
    
    # ==================== Summary Statistics ====================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    fw_summary = analyzer.compute_framework_summary()
    print("\n--- Framework Summary ---")
    print(fw_summary.round(3).to_string(index=False))
    
    # ==================== Export Results ====================
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    analyzer.export_results(output_dir=f'{results_dir}/')
    
    # Save bootstrap CIs separately
    bootstrap_ci.to_csv(f'{results_dir}/bootstrap_ci.csv', index=False)
    print(f"[OK] Saved bootstrap CIs to {results_dir}/bootstrap_ci.csv")
    
    # Generate LaTeX tables
    latex_tables = analyzer.generate_latex_tables()
    with open(f'{results_dir}/latex_tables.tex', 'w') as f:
        for name, table in latex_tables.items():
            f.write(f"% Table: {name}\n")
            f.write(table)
            f.write("\n\n")
    print(f"[OK] LaTeX tables saved to {results_dir}/latex_tables.tex")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {results_dir}")
    print("="*60)
    
    return analyzer, bootstrap_ci, results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to results directory')
    args = parser.parse_args()
    
    analyzer, bootstrap_ci, results_dir = main(args.results_dir)
