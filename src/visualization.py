"""
Visualization Module
====================

Publication-quality figures for RAG evaluation analysis.
Includes forest plots and enhanced visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
})


class Visualizer:
    """Create publication-quality figures for RAG evaluation analysis."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: str = 'figures/'):
        self.df = results_df
        self.output_dir = output_dir
        self.metric_cols = []
        non_metric_cols = {'id', 'domain', 'subset', 'human_label'}
        for col in self.df.columns:
            if col in non_metric_cols:
                continue
            numeric_series = pd.to_numeric(self.df[col], errors='coerce')
            if numeric_series.notna().sum() == 0:
                continue
            self.df[col] = numeric_series
            self.metric_cols.append(col)
        
        # Colors for all 20 frameworks
        self.framework_colors = {
            # Traditional (2020-2024)
            'RAGAS': '#e41a1c',
            'DeepEval': '#377eb8',
            'RAGChecker': '#4daf4a',
            'TRACe': '#984ea3',
            'ARES': '#ff7f00',
            'G-Eval': '#a65628',
            'BERTScore': '#f781bf',
            'UniEval': '#999999',
            'QAFactEval': '#17becf',
            # 2025 frameworks
            'ReDeEP': '#1f77b4',
            'LettuceDetect': '#ff7f0e',
            'FaithJudge': '#2ca02c',
            'LRP4RAG': '#d62728',
            'LUMINA': '#9467bd',
            'HALT-RAG': '#8c564b',
            'MetaRAG': '#e377c2',
            'KG-RAG': '#7f7f7f',
            'GaRAGe': '#bcbd22',
            'HSAD': '#17becf',
            # 2026 frameworks
            'SIRG': '#1a9850'
        }
        
        self.domain_colors = {
            'General Knowledge': '#2ecc71',
            'Finance': '#3498db',
            'Biomedicine': '#e74c3c'
        }
        
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def _save_figure(self, fig, name: str, formats: List[str] = ['pdf', 'png']):
        """Save figure in multiple formats."""
        for fmt in formats:
            fig.savefig(f"{self.output_dir}/{name}.{fmt}", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    # ==================== Figure 1: Correlation Heatmap ====================
    
    def plot_correlation_heatmap(self, metric_filter: Optional[str] = 'faithfulness'):
        """Plot correlation heatmap for selected metrics."""
        if metric_filter:
            cols = [c for c in self.metric_cols if metric_filter in c.lower() or 
                    'consistency' in c.lower() or 'adherence' in c.lower()]
        else:
            cols = self.metric_cols
        
        if len(cols) < 2:
            print(f"Not enough columns matching filter: {metric_filter}")
            return
        
        corr = self.df[cols].corr(method='pearson')
        labels = [c.split('_')[0] for c in cols]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
                    xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title('Inter-Framework Correlation: Faithfulness Metrics', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig1_correlation_heatmap')
        print("[OK] Saved fig1_correlation_heatmap")
    
    # ==================== Figure 2: Domain Comparison ====================
    
    def plot_domain_comparison(self):
        """Plot comparison of metrics across domains."""
        faith_cols = [c for c in self.metric_cols if
                      'faithfulness' in c.lower() or 'consistency' in c.lower()]

        if not faith_cols:
            faith_cols = self.metric_cols
        
        domains = self.df['domain'].unique()
        n_domains = len(domains)
        n_metrics = len(faith_cols)
        
        data = []
        for domain in domains:
            domain_df = self.df[self.df['domain'] == domain]
            for col in faith_cols:
                framework = col.split('_')[0]
                data.append({
                    'Domain': domain,
                    'Framework': framework,
                    'Metric': col,
                    'Mean': domain_df[col].mean(),
                    'Std': domain_df[col].std(),
                    'SE': domain_df[col].std() / np.sqrt(len(domain_df))
                })
        
        plot_df = pd.DataFrame(data)

        # Adjust figure width based on number of metrics
        fig_width = max(16, n_metrics * 1.2)
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        x = np.arange(n_domains)
        width = 0.8 / n_metrics
        
        for i, col in enumerate(faith_cols):
            framework = col.split('_')[0]
            col_data = plot_df[plot_df['Metric'] == col]
            offset = (i - n_metrics/2 + 0.5) * width
            
            bars = ax.bar(x + offset, col_data['Mean'].values, width * 0.85,
                         yerr=col_data['SE'].values, label=framework,
                         color=self.framework_colors.get(framework, f'C{i}'),
                         capsize=2, alpha=0.85)
        
        ax.set_xlabel('Domain', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Faithfulness Score', fontsize=12, fontweight='bold')
        ax.set_title('Faithfulness Scores by Domain and Framework',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, fontsize=11)
        ax.set_ylim(0, 1.1)

        # Adjust legend columns based on number of metrics
        legend_cols = max(1, n_metrics // 10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=legend_cols, fontsize=8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig2_domain_comparison')
        print("[OK] Saved fig2_domain_comparison")
    
    # ==================== Figure 3: Score Distributions ====================

    def plot_score_distributions(self):
        """Plot distribution of scores for each framework as separate images."""
        frameworks = []
        cols = []

        # First pass: prefer faithfulness/consistency/f1/overall metrics
        for col in self.metric_cols:
            fw = col.split('_')[0]
            if fw not in frameworks and ('faithfulness' in col.lower() or
                                         'consistency' in col.lower() or
                                         'f1' in col.lower() or
                                         'overall' in col.lower()):
                frameworks.append(fw)
                cols.append(col)

        # Second pass: pick up any remaining frameworks using their first metric
        for col in self.metric_cols:
            fw = col.split('_')[0]
            if fw not in frameworks:
                frameworks.append(fw)
                cols.append(col)

        saved_count = 0
        skipped = []
        for i, (col, fw) in enumerate(zip(cols, frameworks)):
            valid_scores = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if valid_scores.empty:
                skipped.append(col)
                continue

            fig, ax = plt.subplots(figsize=(5, 3.5))

            ax.hist(valid_scores, bins=20, edgecolor='black', alpha=0.7,
                   color=self.framework_colors.get(fw, f'C{i}'))

            mean_val = valid_scores.mean()
            median_val = valid_scores.median()
            std_val = valid_scores.std()

            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                      label=f'Median: {median_val:.2f}')

            metric_name = col.replace(f'{fw}_', '')
            ax.set_title(f'{fw} — {metric_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9, loc='upper left')

            # Add text box with summary stats
            stats_text = f'$\\sigma$={std_val:.2f}, n={len(valid_scores)}'
            ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            safe_name = fw.replace('-', '_')
            self._save_figure(fig, f'fig3_score_dist_{safe_name}')
            saved_count += 1

        if skipped:
            print(f"[WARN] Skipped {len(skipped)} empty/non-numeric score columns")
        print(f"[OK] Saved {saved_count} separate fig3_score_dist_<framework> files")
    
    # ==================== Figure 4: Scatter Matrix ====================

    def plot_scatter_matrix(self):
        """Plot scatter matrix comparing each framework against all others (separate images)."""
        faith_cols = []
        seen_fw = set()

        for col in self.metric_cols:
            if 'faithfulness' in col.lower() or 'consistency' in col.lower():
                fw = col.split('_')[0]
                if fw not in seen_fw:
                    faith_cols.append(col)
                    seen_fw.add(fw)

        labels = [c.split('_')[0] for c in faith_cols]
        n = len(faith_cols)

        for idx in range(n):
            # For each framework, create a figure showing it vs all others
            others = [(i, faith_cols[i], labels[i]) for i in range(n) if i != idx]
            n_others = len(others)
            n_cols_grid = min(4, n_others)
            n_rows_grid = (n_others + n_cols_grid - 1) // n_cols_grid

            fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                                     figsize=(4 * n_cols_grid, 3.5 * n_rows_grid))
            axes = np.array(axes).flatten() if n_others > 1 else [axes]

            for k, (j, other_col, other_label) in enumerate(others):
                ax = axes[k]
                for domain in self.df['domain'].unique():
                    domain_df = self.df[self.df['domain'] == domain]
                    ax.scatter(domain_df[faith_cols[idx]], domain_df[other_col],
                              alpha=0.45, s=25, label=domain,
                              color=self.domain_colors.get(domain, 'gray'))

                r, p = stats.pearsonr(self.df[faith_cols[idx]], self.df[other_col])
                ax.annotate(f'r={r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=9, fontweight='bold',
                           color='green' if r > 0.7 else ('orange' if r > 0.4 else 'red'))
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel(labels[idx], fontsize=9)
                ax.set_ylabel(other_label, fontsize=9)
                ax.set_title(f'{labels[idx]} vs {other_label}', fontsize=10, fontweight='bold')

            # Hide unused subplots
            for k in range(n_others, len(axes)):
                axes[k].set_visible(False)

            # Single domain legend for the figure
            handles = [mpatches.Patch(color=self.domain_colors[d], label=d)
                       for d in self.df['domain'].unique()]
            fig.legend(handles=handles, loc='upper right', fontsize=9,
                       title='Domain', title_fontsize=10)

            fig.suptitle(f'Pairwise Comparison: {labels[idx]} vs Others',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()
            safe_name = labels[idx].replace('-', '_')
            self._save_figure(fig, f'fig4_scatter_{safe_name}')

        print(f"[OK] Saved {n} separate fig4_scatter_<framework> files")
    
    # ==================== Figure 5: Agreement Heatmap ====================

    def plot_agreement_heatmap(self, threshold: float = 0.5):
        """Plot two separate heatmaps: percent agreement and Cohen's Kappa."""
        faith_cols = [c for c in self.metric_cols if
                      'faithfulness' in c.lower() or 'consistency' in c.lower()]

        if len(faith_cols) < 2:
            print("Not enough faithfulness columns for agreement heatmap")
            return

        n = len(faith_cols)
        agreement_matrix = np.zeros((n, n))
        kappa_matrix = np.zeros((n, n))

        for i, col1 in enumerate(faith_cols):
            for j, col2 in enumerate(faith_cols):
                binary1 = (self.df[col1] >= threshold).astype(int)
                binary2 = (self.df[col2] >= threshold).astype(int)

                agreement_matrix[i, j] = (binary1 == binary2).mean()

                po = (binary1 == binary2).mean()
                pe = binary1.mean() * binary2.mean() + (1-binary1.mean()) * (1-binary2.mean())
                kappa_matrix[i, j] = (po - pe) / (1 - pe) if pe < 1 else 1

        labels = [c.split('_')[0] for c in faith_cols]
        mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)

        # Scale figure size based on number of frameworks
        fig_size = max(8, n * 0.7)
        annot_size = max(6, 10 - n * 0.2)

        # --- Figure 5a: Percent Agreement ---
        fig1, ax1 = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(agreement_matrix, mask=mask, annot=True, fmt='.0%',
                    cmap='YlGn', vmin=0.5, vmax=1, square=True, linewidths=0.5,
                    annot_kws={'size': annot_size},
                    xticklabels=labels, yticklabels=labels, ax=ax1,
                    cbar_kws={'shrink': 0.8, 'label': 'Agreement'})
        ax1.set_title(f'Percent Agreement (threshold={threshold})',
                      fontsize=14, fontweight='bold', pad=15)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.tick_params(axis='y', rotation=0, labelsize=9)
        plt.tight_layout()
        self._save_figure(fig1, 'fig5a_percent_agreement')

        # --- Figure 5b: Cohen's Kappa ---
        fig2, ax2 = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(kappa_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', vmin=0, vmax=1, square=True, linewidths=0.5,
                    annot_kws={'size': annot_size},
                    xticklabels=labels, yticklabels=labels, ax=ax2,
                    cbar_kws={'shrink': 0.8, 'label': "Cohen's Kappa"})
        ax2.set_title(f"Cohen's Kappa (threshold={threshold})",
                      fontsize=14, fontweight='bold', pad=15)
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.tick_params(axis='y', rotation=0, labelsize=9)
        plt.tight_layout()
        self._save_figure(fig2, 'fig5b_cohens_kappa')

        print("[OK] Saved fig5a_percent_agreement and fig5b_cohens_kappa")
    
    # ==================== Figure 6: Box Plots by Domain ====================
    
    def plot_domain_boxplots(self):
        """Plot box plots of scores by domain for each framework."""
        metrics = [c for c in self.metric_cols if 'faithfulness' in c.lower()]
        if len(metrics) < 3:
            metrics = self.metric_cols[:9]
        
        melt_df = pd.melt(self.df[['domain'] + metrics], 
                          id_vars=['domain'], var_name='Metric', value_name='Score')
        melt_df['Framework'] = melt_df['Metric'].apply(lambda x: x.split('_')[0])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        frameworks = melt_df['Framework'].unique()
        domains = melt_df['domain'].unique()
        
        for i, fw in enumerate(frameworks):
            fw_data = melt_df[melt_df['Framework'] == fw]
            bp_data = [fw_data[fw_data['domain'] == d]['Score'].values for d in domains]
            
            bp = ax.boxplot(bp_data, 
                           positions=[i * (len(domains) + 1) + j for j in range(len(domains))],
                           widths=0.6, patch_artist=True)
            
            for patch, domain in zip(bp['boxes'], domains):
                patch.set_facecolor(self.domain_colors.get(domain, 'gray'))
                patch.set_alpha(0.7)
        
        ax.set_xticks([i * (len(domains) + 1) + 1 for i in range(len(frameworks))])
        ax.set_xticklabels(frameworks, fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Score Distributions by Framework and Domain', 
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        handles = [mpatches.Patch(color=self.domain_colors[d], label=d, alpha=0.7) 
                   for d in domains]
        ax.legend(handles=handles, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig6_domain_boxplots')
        print("[OK] Saved fig6_domain_boxplots")
    
    # ==================== Figure 7: Forest Plot ====================

    def plot_forest_plot(self, bootstrap_ci_df: Optional[pd.DataFrame] = None):
        """
        Plot forest plot showing per-framework mean correlations with CIs.

        Summarizes each framework's average correlation with all other
        frameworks, producing a concise, paper-ready figure.
        """
        # Build pairwise data
        if bootstrap_ci_df is None:
            faith_cols = [c for c in self.metric_cols if 'faithfulness' in c.lower()]

            data = []
            for i, col1 in enumerate(faith_cols):
                for col2 in faith_cols[i+1:]:
                    r, p = stats.pearsonr(self.df[col1], self.df[col2])
                    n_obs = len(self.df)
                    se = np.sqrt((1 - r**2) / (n_obs - 2))
                    data.append({
                        'framework_1': col1.split('_')[0],
                        'framework_2': col2.split('_')[0],
                        'correlation': r,
                        'ci_lower': r - 1.96 * se,
                        'ci_upper': r + 1.96 * se
                    })

            pairwise_df = pd.DataFrame(data)
        else:
            pairwise_df = bootstrap_ci_df.copy()

        # Compute per-framework mean correlation and pooled CI
        frameworks = sorted(set(pairwise_df['framework_1']) | set(pairwise_df['framework_2']))
        summary = []
        for fw in frameworks:
            mask = (pairwise_df['framework_1'] == fw) | (pairwise_df['framework_2'] == fw)
            fw_data = pairwise_df[mask]
            mean_r = fw_data['correlation'].mean()
            mean_ci_lo = fw_data['ci_lower'].mean()
            mean_ci_hi = fw_data['ci_upper'].mean()
            summary.append({
                'framework': fw,
                'mean_correlation': mean_r,
                'ci_lower': mean_ci_lo,
                'ci_upper': mean_ci_hi,
                'n_comparisons': len(fw_data)
            })

        summary_df = pd.DataFrame(summary).sort_values('mean_correlation', ascending=True)
        n_fw = len(summary_df)

        fig, ax = plt.subplots(figsize=(8, max(6, n_fw * 0.45)))

        y_positions = range(n_fw)

        for i, (idx, row) in enumerate(summary_df.iterrows()):
            r = row['mean_correlation']
            color = '#2ca02c' if r > 0.7 else '#ff7f0e' if r > 0.4 else '#d62728'

            # CI bar
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
                   color=color, linewidth=2.5, alpha=0.6, solid_capstyle='round')
            # Point estimate
            ax.scatter(r, i, color=color, s=90, zorder=5, edgecolors='black', linewidths=0.5)
            # Value label
            ax.text(row['ci_upper'] + 0.02, i, f'{r:.2f}', va='center', fontsize=9,
                    fontweight='bold', color=color)

        # Reference lines
        ax.axvline(x=0.7, color='#2ca02c', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(x=0.4, color='#ff7f0e', linestyle='--', alpha=0.4, linewidth=1)

        # Shaded regions for interpretation
        ax.axvspan(-0.1, 0.4, alpha=0.04, color='red')
        ax.axvspan(0.4, 0.7, alpha=0.04, color='orange')
        ax.axvspan(0.7, 1.1, alpha=0.04, color='green')

        ax.text(0.2, n_fw + 0.3, 'Weak', ha='center', fontsize=8, color='#d62728', fontstyle='italic')
        ax.text(0.55, n_fw + 0.3, 'Moderate', ha='center', fontsize=8, color='#ff7f0e', fontstyle='italic')
        ax.text(0.85, n_fw + 0.3, 'Strong', ha='center', fontsize=8, color='#2ca02c', fontstyle='italic')

        ax.set_yticks(y_positions)
        ax.set_yticklabels(summary_df['framework'].values, fontsize=10)
        ax.set_xlabel('Mean Pearson Correlation with Other Frameworks (95% CI)',
                       fontsize=11, fontweight='bold')
        ax.set_title('Forest Plot: Mean Inter-Framework Correlation by Framework',
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, n_fw + 0.5)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                   markersize=8, label='Strong (r > 0.7)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                   markersize=8, label='Moderate (0.4 < r ≤ 0.7)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                   markersize=8, label='Weak (r ≤ 0.4)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
                  title='Correlation Strength', title_fontsize=10,
                  framealpha=0.9)

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        self._save_figure(fig, 'fig7_forest_plot')
        print("[OK] Saved fig7_forest_plot")
    
    # ==================== Figure 8: Significance Matrix ====================
    
    def plot_significance_matrix(self):
        """Plot matrix showing statistical significance of framework differences."""
        faith_cols = [c for c in self.metric_cols if 'faithfulness' in c.lower()]
        if len(faith_cols) < 2:
            faith_cols = self.metric_cols[:6]
        
        n = len(faith_cols)
        p_value_matrix = np.ones((n, n))
        
        for i, col1 in enumerate(faith_cols):
            for j, col2 in enumerate(faith_cols):
                if i != j:
                    t_stat, p_val = stats.ttest_rel(self.df[col1], self.df[col2])
                    p_value_matrix[i, j] = p_val
        
        labels = [c.split('_')[0] for c in faith_cols]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sig_matrix = np.zeros_like(p_value_matrix)
        sig_matrix[p_value_matrix < 0.001] = 3
        sig_matrix[(p_value_matrix >= 0.001) & (p_value_matrix < 0.01)] = 2
        sig_matrix[(p_value_matrix >= 0.01) & (p_value_matrix < 0.05)] = 1
        
        mask = np.eye(n, dtype=bool)
        
        sns.heatmap(sig_matrix, mask=mask, annot=p_value_matrix, fmt='.4f',
                    cmap='RdYlGn_r', vmin=0, vmax=3,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Significance Level', 'ticks': [0, 1, 2, 3]})
        
        ax.set_title('Statistical Significance of Framework Differences\n(Paired t-test p-values)', 
                     fontsize=13, fontweight='bold')
        
        ax.text(1.15, 0.8, '*** p < 0.001\n** p < 0.01\n* p < 0.05\nns p ≥ 0.05',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self._save_figure(fig, 'fig8_significance_matrix')
        print("[OK] Saved fig8_significance_matrix")
    
    # ==================== Figure 9: Disagreement Analysis ====================
    
    def plot_disagreement_distribution(self):
        """Plot distribution of framework disagreements."""
        faith_cols = [c for c in self.metric_cols if 'faithfulness' in c.lower()]
        
        if len(faith_cols) < 2:
            print("Not enough faithfulness columns")
            return
        
        # Compute disagreement for each sample
        disagreements = []
        for idx, row in self.df.iterrows():
            scores = [row[col] for col in faith_cols]
            disagreements.append(max(scores) - min(scores))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of disagreements
        ax1.hist(disagreements, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.mean(disagreements), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(disagreements):.2f}')
        ax1.axvline(0.3, color='orange', linestyle=':', 
                   label='Threshold: 0.3')
        ax1.set_xlabel('Max Disagreement (max - min)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Distribution of Framework Disagreement', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        
        # Disagreement by domain
        domain_disagreements = {d: [] for d in self.df['domain'].unique()}
        for idx, row in self.df.iterrows():
            scores = [row[col] for col in faith_cols]
            domain_disagreements[row['domain']].append(max(scores) - min(scores))
        
        domains = list(domain_disagreements.keys())
        # Shorten domain names if needed
        domain_labels = []
        for d in domains:
            if len(d) > 12:
                domain_labels.append(d[:10] + '..')
            else:
                domain_labels.append(d)
        
        means = [np.mean(domain_disagreements[d]) for d in domains]
        stds = [np.std(domain_disagreements[d]) for d in domains]
        
        bars = ax2.bar(range(len(domains)), means, yerr=stds, capsize=5, 
                       color=[self.domain_colors.get(d, 'gray') for d in domains],
                       alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(domains)))
        ax2.set_xticklabels(domain_labels, fontsize=10, rotation=15, ha='right')
        ax2.set_ylabel('Mean Disagreement', fontsize=11)
        ax2.set_title('Disagreement by Domain', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(means) * 1.4)
        
        fig.suptitle('Framework Disagreement Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, 'fig9_disagreement_analysis')
        print("[OK] Saved fig9_disagreement_analysis")
    
    # ==================== Generate All Figures ====================
    
    def generate_all_figures(self, bootstrap_ci_df: Optional[pd.DataFrame] = None):
        """Generate all publication figures."""
        print("\n" + "="*60)
        print("Generating Publication Figures")
        print("="*60 + "\n")
        
        self.plot_correlation_heatmap()
        self.plot_domain_comparison()
        self.plot_score_distributions()
        self.plot_scatter_matrix()
        self.plot_agreement_heatmap()
        self.plot_domain_boxplots()
        self.plot_forest_plot(bootstrap_ci_df)
        self.plot_significance_matrix()
        self.plot_disagreement_distribution()
        
        print(f"\n[OK] All figures saved to {self.output_dir}")
