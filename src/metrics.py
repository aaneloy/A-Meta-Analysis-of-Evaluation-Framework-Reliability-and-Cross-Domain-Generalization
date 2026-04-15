"""
Metrics Analysis Module
=======================

Statistical analysis tools for comparing evaluation frameworks.
Includes bootstrap confidence intervals and meta-analysis statistics.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MetricsAnalyzer:
    """
    Comprehensive statistical analysis for RAG evaluation metrics.
    
    Provides:
    - Correlation analysis (Pearson, Spearman, Kendall)
    - Bootstrap confidence intervals
    - Agreement metrics (Cohen's Kappa, Krippendorff's Alpha)
    - Domain-specific analysis
    - Statistical significance tests
    - Heterogeneity statistics (I², Q)
    """
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer with evaluation results.
        
        Args:
            results_df: DataFrame with columns for each metric and 'domain' column
        """
        self.df = results_df
        self.metric_cols = [c for c in self.df.columns if c not in ['id', 'domain', 'subset', 'human_label']]
        self.frameworks = self._extract_frameworks()
    
    def _extract_frameworks(self) -> List[str]:
        """Extract unique framework names from column names."""
        frameworks = set()
        for col in self.metric_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                frameworks.add(parts[0])
        return sorted(frameworks)
    
    def get_framework_metrics(self, framework: str) -> List[str]:
        """Get all metric columns for a specific framework."""
        return [c for c in self.metric_cols if c.startswith(f"{framework}_")]

    def _select_primary_metric_cols(self) -> List[str]:
        """Select one primary metric per framework for cross-framework analyses."""
        preferred = [
            "faithfulness", "factual_consistency", "consistency", "adherence",
            "nli_ensemble_score", "f1", "overall", "score"
        ]
        selected = []
        for framework in self.frameworks:
            fw_cols = self.get_framework_metrics(framework)
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
    
    # ==================== Bootstrap Confidence Intervals ====================
    
    def bootstrap_correlation(self, col1: str, col2: str, n_bootstrap: int = 1000, 
                              ci: float = 0.95, method: str = 'pearson') -> Dict:
        """
        Compute correlation with bootstrap confidence interval.
        
        Args:
            col1: First column name
            col2: Second column name
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval level
            method: 'pearson', 'spearman', or 'kendall'
            
        Returns:
            Dictionary with correlation, CI bounds, and standard error
        """
        x = self.df[col1].values
        y = self.df[col2].values
        n = len(x)
        
        # Compute observed correlation
        if method == 'pearson':
            corr_func = lambda a, b: pearsonr(a, b)[0]
        elif method == 'spearman':
            corr_func = lambda a, b: spearmanr(a, b)[0]
        else:
            corr_func = lambda a, b: kendalltau(a, b)[0]
        
        observed_corr = corr_func(x, y)
        
        # Bootstrap
        bootstrap_corrs = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_corr = corr_func(x[indices], y[indices])
            bootstrap_corrs.append(boot_corr)
        
        bootstrap_corrs = np.array(bootstrap_corrs)
        
        # Compute CI
        alpha = 1 - ci
        ci_lower = np.percentile(bootstrap_corrs, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_corrs, (1 - alpha/2) * 100)
        
        return {
            'correlation': observed_corr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': np.std(bootstrap_corrs),
            'method': method
        }
    
    def compute_all_bootstrap_ci(self, n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Compute bootstrap CIs for all faithfulness correlations.
        
        Returns:
            DataFrame with correlations and confidence intervals
        """
        faith_cols = [c for c in self.metric_cols if 
                      'faithfulness' in c.lower() or 'consistency' in c.lower()]
        
        results = []
        
        for i, col1 in enumerate(faith_cols):
            for col2 in faith_cols[i+1:]:
                fw1 = col1.split('_')[0]
                fw2 = col2.split('_')[0]
                
                boot_result = self.bootstrap_correlation(col1, col2, n_bootstrap)
                
                results.append({
                    'framework_1': fw1,
                    'framework_2': fw2,
                    'metric_1': col1,
                    'metric_2': col2,
                    'correlation': boot_result['correlation'],
                    'ci_lower': boot_result['ci_lower'],
                    'ci_upper': boot_result['ci_upper'],
                    'std_error': boot_result['std_error']
                })
        
        return pd.DataFrame(results)
    
    # ==================== Heterogeneity Statistics ====================
    
    def compute_heterogeneity(self) -> Dict:
        """Compute heterogeneity across primary metrics from all frameworks."""
        primary_cols = self._select_primary_metric_cols()

        if len(primary_cols) < 2:
            return {'error': 'Not enough primary metric columns'}

        grand_mean = self.df[primary_cols].values.mean()
        effect_sizes = []
        variances = []

        for col in primary_cols:
            mean_i = self.df[col].mean()
            var_i = self.df[col].var() / len(self.df)
            var_i = max(var_i, 1e-9)
            effect_sizes.append(mean_i - grand_mean)
            variances.append(var_i)

        effect_sizes = np.array(effect_sizes)
        variances = np.array(variances)
        weights = 1 / variances

        weighted_mean = np.sum(weights * effect_sizes) / np.sum(weights)
        Q = np.sum(weights * (effect_sizes - weighted_mean) ** 2)

        df = len(primary_cols) - 1
        I_squared = max(0, (Q - df) / Q * 100) if Q > 0 else 0
        p_value = 1 - stats.chi2.cdf(Q, df)

        if I_squared < 25:
            interpretation = "Low heterogeneity - frameworks largely agree"
        elif I_squared < 75:
            interpretation = "Moderate heterogeneity - some framework disagreement"
        else:
            interpretation = "High heterogeneity - substantial framework disagreement"

        return {
            'Q_statistic': Q,
            'Q_df': df,
            'Q_pvalue': p_value,
            'I_squared': I_squared,
            'framework_count': len(primary_cols),
            'framework_metrics_used': primary_cols,
            'interpretation': interpretation
        }

    # ==================== Correlation Analysis ====================
    
    def compute_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """Compute correlation matrix for all metrics."""
        return self.df[self.metric_cols].corr(method=method)
    
    def compute_faithfulness_correlations(self) -> Dict[str, pd.DataFrame]:
        """Compute correlations for faithfulness-related metrics."""
        faith_cols = [c for c in self.metric_cols if any(
            keyword in c.lower() for keyword in 
            ['faithfulness', 'consistency', 'adherence', 'factual', 'hallucination']
        )]
        
        df_faith = self.df[faith_cols].copy()
        for col in faith_cols:
            if 'hallucination' in col.lower():
                df_faith[col] = 1 - df_faith[col]
        
        return {
            'pearson': df_faith.corr(method='pearson'),
            'spearman': df_faith.corr(method='spearman'),
            'kendall': df_faith.corr(method='kendall')
        }
    
    def compute_pairwise_correlations(self) -> pd.DataFrame:
        """Compute all pairwise correlations between frameworks."""
        results = []
        
        for i, fw1 in enumerate(self.frameworks):
            for fw2 in self.frameworks[i+1:]:
                fw1_cols = self.get_framework_metrics(fw1)
                fw2_cols = self.get_framework_metrics(fw2)
                
                for col1 in fw1_cols:
                    metric1 = col1.replace(f"{fw1}_", "")
                    for col2 in fw2_cols:
                        metric2 = col2.replace(f"{fw2}_", "")
                        
                        if self._metrics_match(metric1, metric2):
                            r_pearson, p_pearson = pearsonr(self.df[col1], self.df[col2])
                            r_spearman, p_spearman = spearmanr(self.df[col1], self.df[col2])
                            tau, p_kendall = kendalltau(self.df[col1], self.df[col2])
                            
                            results.append({
                                'framework_1': fw1,
                                'framework_2': fw2,
                                'metric_1': metric1,
                                'metric_2': metric2,
                                'pearson_r': r_pearson,
                                'pearson_p': p_pearson,
                                'spearman_rho': r_spearman,
                                'spearman_p': p_spearman,
                                'kendall_tau': tau,
                                'kendall_p': p_kendall
                            })
        
        return pd.DataFrame(results)
    
    def _metrics_match(self, m1: str, m2: str) -> bool:
        """Check if two metrics measure the same thing."""
        m1, m2 = m1.lower(), m2.lower()
        
        if m1 == m2:
            return True
        
        semantic_groups = [
            {'faithfulness', 'consistency', 'adherence', 'factual_consistency'},
            {'relevance', 'relevancy', 'answer_relevance', 'answer_relevancy'},
            {'precision', 'context_precision', 'contextual_precision'},
            {'recall', 'context_recall', 'contextual_recall', 'claim_recall'},
            {'hallucination'},
            {'coherence'},
            {'fluency'},
            {'f1'}
        ]
        
        for group in semantic_groups:
            if m1 in group and m2 in group:
                return True
        
        return False
    
    # ==================== Agreement Analysis ====================
    
    def compute_cohens_kappa(self, col1: str, col2: str, threshold: float = 0.5) -> float:
        """Compute Cohen's Kappa for binarized scores."""
        binary1 = (self.df[col1] >= threshold).astype(int)
        binary2 = (self.df[col2] >= threshold).astype(int)
        
        po = (binary1 == binary2).mean()
        p1, p2 = binary1.mean(), binary2.mean()
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        
        if pe == 1:
            return 1.0
        return (po - pe) / (1 - pe)
    
    def compute_agreement_matrix(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute agreement metrics for faithfulness-related metrics."""
        faith_cols = [c for c in self.metric_cols if any(
            keyword in c.lower() for keyword in 
            ['faithfulness', 'consistency', 'adherence', 'factual']
        )]
        
        n = len(faith_cols)
        agreement_matrix = np.zeros((n, n))
        kappa_matrix = np.zeros((n, n))
        
        for i, col1 in enumerate(faith_cols):
            for j, col2 in enumerate(faith_cols):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                    kappa_matrix[i, j] = 1.0
                else:
                    binary1 = (self.df[col1] >= threshold).astype(int)
                    binary2 = (self.df[col2] >= threshold).astype(int)
                    
                    agreement_matrix[i, j] = (binary1 == binary2).mean()
                    kappa_matrix[i, j] = self.compute_cohens_kappa(col1, col2, threshold)
        
        labels = [c.replace('_faithfulness', '').replace('_consistency', '').replace('_adherence', '') 
                  for c in faith_cols]
        
        return (
            pd.DataFrame(agreement_matrix, index=labels, columns=labels),
            pd.DataFrame(kappa_matrix, index=labels, columns=labels)
        )
    
    def compute_threshold_robustness(self, thresholds: Optional[List[float]] = None) -> pd.DataFrame:
        """Evaluate agreement robustness across decision thresholds."""
        thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        rows = []
        reference = None

        for threshold in thresholds:
            agreement, kappa = self.compute_agreement_matrix(threshold=threshold)
            upper_agreement = agreement.values[np.triu_indices_from(agreement.values, k=1)]
            upper_kappa = kappa.values[np.triu_indices_from(kappa.values, k=1)]

            row = {
                'threshold': threshold,
                'mean_agreement': float(np.nanmean(upper_agreement)) if len(upper_agreement) else np.nan,
                'mean_kappa': float(np.nanmean(upper_kappa)) if len(upper_kappa) else np.nan,
            }

            if reference is None:
                reference = upper_agreement
                row['spearman_vs_0_3'] = 1.0
            else:
                rho, _ = spearmanr(reference, upper_agreement)
                row['spearman_vs_0_3'] = float(rho)

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_human_label_correlations(self) -> pd.DataFrame:
        """Compute metric correlations against binary human labels, when available."""
        if 'human_label' not in self.df.columns:
            return pd.DataFrame()

        labeled = self.df[self.df['human_label'].isin([0, 1])].copy()
        if labeled.empty:
            return pd.DataFrame()

        results = []
        y = labeled['human_label'].astype(float)

        for col in self.metric_cols:
            x = labeled[col].astype(float)
            if x.nunique(dropna=True) < 2:
                continue
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_rho, spearman_p = spearmanr(x, y)
            results.append({
                'metric': col,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_rho': spearman_rho,
                'spearman_p': spearman_p,
                'n_labeled': len(labeled)
            })

        return pd.DataFrame(results).sort_values('pearson_r', ascending=False)

    # ==================== Cluster Analysis ====================

    def compute_hierarchical_clustering(self, n_clusters: int = 3) -> Dict:
        """
        Hierarchical clustering (Ward's method) on the framework correlation matrix.

        Uses distance d_jk = sqrt(2(1 - r_jk)) and determines cluster count
        via the elbow method on within-cluster sum of squares.

        Args:
            n_clusters: Number of clusters (default 3, or auto-detected via elbow)

        Returns:
            Dictionary with cluster assignments, within-cluster means,
            between-cluster mean, and linkage information.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Build correlation matrix over primary metrics
        primary_cols = self._select_primary_metric_cols()
        if len(primary_cols) < 3:
            return {'error': 'Not enough frameworks for clustering'}

        corr_matrix = self.df[primary_cols].corr(method='pearson')
        fw_labels = [c.split('_')[0] for c in primary_cols]

        # Distance: d_jk = sqrt(2(1 - r_jk))
        dist_matrix = np.sqrt(2 * (1 - corr_matrix.values))
        np.fill_diagonal(dist_matrix, 0)

        # Ward's method linkage
        condensed_dist = squareform(dist_matrix, checks=False)
        Z = linkage(condensed_dist, method='ward')

        # Elbow method: try 2..max_k and pick elbow
        max_k = min(len(primary_cols) - 1, 8)
        wcss_values = []
        for k in range(2, max_k + 1):
            labels_k = fcluster(Z, t=k, criterion='maxclust')
            wcss = 0.0
            for c in range(1, k + 1):
                members = np.where(labels_k == c)[0]
                if len(members) > 1:
                    cluster_corrs = corr_matrix.values[np.ix_(members, members)]
                    upper = cluster_corrs[np.triu_indices_from(cluster_corrs, k=1)]
                    centroid = np.mean(upper)
                    wcss += np.sum((upper - centroid) ** 2)
            wcss_values.append({'k': k, 'wcss': wcss})

        # Use provided n_clusters (paper uses 3)
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

        # Compute within-cluster and between-cluster mean correlations
        cluster_info = {}
        all_within = []
        for c in range(1, n_clusters + 1):
            members = np.where(cluster_labels == c)[0]
            member_names = [fw_labels[i] for i in members]
            if len(members) > 1:
                cluster_corrs = corr_matrix.values[np.ix_(members, members)]
                upper = cluster_corrs[np.triu_indices_from(cluster_corrs, k=1)]
                within_mean = float(np.mean(upper))
                all_within.extend(upper.tolist())
            else:
                within_mean = 1.0
            cluster_info[f'cluster_{c}'] = {
                'frameworks': member_names,
                'size': len(members),
                'within_cluster_mean_r': round(within_mean, 2)
            }

        # Between-cluster correlations
        between_corrs = []
        for c1 in range(1, n_clusters + 1):
            for c2 in range(c1 + 1, n_clusters + 1):
                members_1 = np.where(cluster_labels == c1)[0]
                members_2 = np.where(cluster_labels == c2)[0]
                cross = corr_matrix.values[np.ix_(members_1, members_2)]
                between_corrs.extend(cross.flatten().tolist())

        between_mean = float(np.mean(between_corrs)) if between_corrs else 0.0

        return {
            'n_clusters': n_clusters,
            'cluster_assignments': {fw_labels[i]: int(cluster_labels[i]) for i in range(len(fw_labels))},
            'clusters': cluster_info,
            'between_cluster_mean_r': round(between_mean, 2),
            'wcss_elbow': wcss_values,
            'linkage_method': 'ward',
            'distance_metric': 'sqrt(2*(1-r))'
        }

    # ==================== Mixed-Effects Model ====================

    def compute_mixed_effects_model(self, metric_col: str = None) -> Dict:
        """
        Compute mixed-effects model for domain effects.

        Model: score_ij = β₀ + β₁(domain_i) + u_j + ε_ij

        Where:
        - β₀: Fixed intercept
        - β₁: Fixed effect of domain
        - u_j: Random effect for sample (accounts for sample-level variation)
        - ε_ij: Residual error

        Args:
            metric_col: Specific metric column to analyze. If None, analyzes faithfulness metrics.

        Returns:
            Dictionary with model results, fixed effects, random effects variance, and fit statistics
        """
        try:
            import statsmodels.api as sm
            from statsmodels.regression.mixed_linear_model import MixedLM
        except ImportError as e:
            return {'error': f'statsmodels not available: {e}. Run: pip install statsmodels'}
        except Exception as e:
            return {'error': f'Error importing statsmodels: {e}'}

        # Select metric columns
        if metric_col:
            metric_cols = [metric_col] if metric_col in self.metric_cols else []
        else:
            metric_cols = [c for c in self.metric_cols if 'faithfulness' in c.lower()]

        if not metric_cols:
            return {'error': 'No matching metric columns found'}

        results = {}

        for col in metric_cols:
            try:
                # Prepare data for mixed-effects model
                model_df = self.df[['domain', 'id', col]].copy()
                model_df = model_df.rename(columns={col: 'score'})
                model_df = model_df.dropna()

                # Create dummy variables for domain (fixed effect)
                model_df['domain_code'] = pd.Categorical(model_df['domain']).codes

                # Fit mixed-effects model: score ~ domain (fixed) + (1|sample)
                # Using domain as fixed effect and sample ID as random effect
                model = MixedLM(
                    endog=model_df['score'],
                    exog=sm.add_constant(pd.get_dummies(model_df['domain'], drop_first=True)),
                    groups=model_df['id']
                )

                fitted = model.fit(method='powell', maxiter=500)

                # Extract results
                results[col] = {
                    'converged': fitted.converged,
                    'fixed_effects': {
                        'intercept': float(fitted.fe_params.get('const', fitted.fe_params.iloc[0])),
                        'domain_effects': {k: float(v) for k, v in fitted.fe_params.items() if k != 'const'}
                    },
                    'fixed_effects_pvalues': {k: float(v) for k, v in fitted.pvalues.items()},
                    'random_effects_variance': float(fitted.cov_re.iloc[0, 0]) if hasattr(fitted.cov_re, 'iloc') else float(fitted.cov_re),
                    'residual_variance': float(fitted.scale),
                    'log_likelihood': float(fitted.llf),
                    'aic': float(fitted.aic),
                    'bic': float(fitted.bic),
                    'icc': self._compute_icc(fitted),
                    'n_groups': int(fitted.ngroups),
                    'n_observations': int(fitted.nobs)
                }
            except Exception as e:
                results[col] = {'error': str(e)}

        return results

    def _compute_icc(self, fitted_model) -> float:
        """
        Compute Intraclass Correlation Coefficient (ICC).

        ICC = σ²_between / (σ²_between + σ²_within)

        Measures proportion of variance attributable to group-level differences.
        """
        try:
            random_var = float(fitted_model.cov_re.iloc[0, 0]) if hasattr(fitted_model.cov_re, 'iloc') else float(fitted_model.cov_re)
            residual_var = float(fitted_model.scale)
            total_var = random_var + residual_var
            if total_var > 0:
                return random_var / total_var
            return 0.0
        except:
            return 0.0

    def compute_all_mixed_effects(self) -> pd.DataFrame:
        """
        Compute mixed-effects model for all metrics.

        Returns:
            DataFrame with mixed-effects results for each metric
        """
        all_results = []

        for col in self.metric_cols:
            result = self.compute_mixed_effects_model(col)
            if col in result and 'error' not in result[col]:
                row = {
                    'metric': col,
                    'intercept': result[col]['fixed_effects']['intercept'],
                    'random_effects_var': result[col]['random_effects_variance'],
                    'residual_var': result[col]['residual_variance'],
                    'icc': result[col]['icc'],
                    'aic': result[col]['aic'],
                    'bic': result[col]['bic'],
                    'converged': result[col]['converged']
                }
                # Add domain effects
                for domain, effect in result[col]['fixed_effects']['domain_effects'].items():
                    row[f'domain_effect_{domain}'] = effect
                # Add p-values
                for param, pval in result[col]['fixed_effects_pvalues'].items():
                    if param != 'const':
                        row[f'pvalue_{param}'] = pval

                all_results.append(row)

        return pd.DataFrame(all_results)

    # ==================== Domain Analysis ====================

    def compute_domain_statistics(self) -> pd.DataFrame:
        """Compute statistics for each domain."""
        results = []
        
        for domain in self.df['domain'].unique():
            domain_df = self.df[self.df['domain'] == domain]
            
            for col in self.metric_cols:
                results.append({
                    'domain': domain,
                    'metric': col,
                    'mean': domain_df[col].mean(),
                    'std': domain_df[col].std(),
                    'median': domain_df[col].median(),
                    'min': domain_df[col].min(),
                    'max': domain_df[col].max(),
                    'n': len(domain_df)
                })
        
        return pd.DataFrame(results)
    
    def compute_domain_anova(self) -> pd.DataFrame:
        """Compute ANOVA for domain effect on each metric."""
        results = []
        domains = self.df['domain'].unique()
        
        for col in self.metric_cols:
            groups = [self.df[self.df['domain'] == d][col].values for d in domains]
            f_stat, p_val = stats.f_oneway(*groups)
            
            ss_between = sum(len(g) * (np.mean(g) - self.df[col].mean())**2 for g in groups)
            ss_total = sum((self.df[col] - self.df[col].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            results.append({
                'metric': col,
                'f_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta_squared,
                'significant': p_val < 0.05
            })
        
        return pd.DataFrame(results)
    
    def compute_domain_pairwise_tests(self) -> pd.DataFrame:
        """Compute pairwise t-tests between domains."""
        results = []
        domains = sorted(self.df['domain'].unique())
        
        for col in self.metric_cols:
            for i, d1 in enumerate(domains):
                for d2 in domains[i+1:]:
                    g1 = self.df[self.df['domain'] == d1][col].values
                    g2 = self.df[self.df['domain'] == d2][col].values
                    
                    t_stat, p_val = stats.ttest_ind(g1, g2)
                    pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
                    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
                    
                    results.append({
                        'metric': col,
                        'domain_1': d1,
                        'domain_2': d2,
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'cohens_d': cohens_d,
                        'mean_diff': np.mean(g1) - np.mean(g2)
                    })
        
        return pd.DataFrame(results)
    
    # ==================== Summary Statistics ====================
    
    def compute_summary_statistics(self) -> pd.DataFrame:
        """Compute comprehensive summary statistics."""
        stats_dict = self.df[self.metric_cols].describe()
        stats_dict.loc['skewness'] = self.df[self.metric_cols].skew()
        stats_dict.loc['kurtosis'] = self.df[self.metric_cols].kurtosis()
        return stats_dict.T
    
    def compute_framework_summary(self) -> pd.DataFrame:
        """Compute summary statistics grouped by framework."""
        results = []
        
        for framework in self.frameworks:
            fw_cols = self.get_framework_metrics(framework)
            fw_scores = self.df[fw_cols].values.flatten()
            
            results.append({
                'framework': framework,
                'mean': np.mean(fw_scores),
                'std': np.std(fw_scores),
                'min': np.min(fw_scores),
                'max': np.max(fw_scores),
                'n_metrics': len(fw_cols)
            })
        
        return pd.DataFrame(results)
    
    # ==================== Export Methods ====================
    
    def export_results(self, output_dir: str = 'results/'):
        """Export all analysis results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Correlation matrices
        for method in ['pearson', 'spearman', 'kendall']:
            corr = self.compute_correlation_matrix(method)
            corr.to_csv(f"{output_dir}/correlation_{method}.csv")
        
        # Faithfulness correlations
        faith_corr = self.compute_faithfulness_correlations()
        faith_corr['pearson'].to_csv(f"{output_dir}/faithfulness_correlation.csv")
        
        # Bootstrap CIs
        bootstrap_ci = self.compute_all_bootstrap_ci()
        bootstrap_ci.to_csv(f"{output_dir}/bootstrap_ci.csv", index=False)
        
        # Heterogeneity
        het = self.compute_heterogeneity()
        with open(f"{output_dir}/heterogeneity.json", 'w') as f:
            json.dump(het, f, indent=2)
        
        # Pairwise correlations
        pairwise = self.compute_pairwise_correlations()
        pairwise.to_csv(f"{output_dir}/pairwise_correlations.csv", index=False)
        
        # Agreement matrices
        agreement, kappa = self.compute_agreement_matrix()
        agreement.to_csv(f"{output_dir}/agreement_matrix.csv")
        kappa.to_csv(f"{output_dir}/kappa_matrix.csv")

        threshold_robustness = self.compute_threshold_robustness()
        threshold_robustness.to_csv(f"{output_dir}/threshold_robustness.csv", index=False)

        # Hierarchical clustering
        clustering = self.compute_hierarchical_clustering()
        with open(f"{output_dir}/clustering.json", 'w') as f:
            json.dump(clustering, f, indent=2)

        human_corr = self.compute_human_label_correlations()
        if len(human_corr) > 0:
            human_corr.to_csv(f"{output_dir}/human_label_correlations.csv", index=False)
        
        # Domain statistics
        domain_stats = self.compute_domain_statistics()
        domain_stats.to_csv(f"{output_dir}/domain_statistics.csv", index=False)
        
        # ANOVA results
        anova = self.compute_domain_anova()
        anova.to_csv(f"{output_dir}/domain_anova.csv", index=False)

        # Mixed-effects model results
        try:
            mixed_effects = self.compute_all_mixed_effects()
            if len(mixed_effects) > 0:
                mixed_effects.to_csv(f"{output_dir}/mixed_effects_model.csv", index=False)

                # Also save detailed results for faithfulness metrics
                faith_mixed = self.compute_mixed_effects_model()
                with open(f"{output_dir}/mixed_effects_detailed.json", 'w') as f:
                    json.dump(faith_mixed, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not compute mixed-effects model: {e}")

        # Summary statistics
        summary = self.compute_summary_statistics()
        summary.to_csv(f"{output_dir}/summary_statistics.csv")
        
        # Framework summary
        fw_summary = self.compute_framework_summary()
        fw_summary.to_csv(f"{output_dir}/framework_summary.csv", index=False)

        # Framework faithfulness correlations (for paper Table 2)
        faith_framework_corr = self.compute_framework_faithfulness_correlations()
        if len(faith_framework_corr) > 0:
            faith_framework_corr.to_csv(f"{output_dir}/faithfulness_framework_correlations.csv", index=False)

        # Correlation summary for paper (human-readable)
        corr_summary = self.generate_correlation_summary_for_paper()
        with open(f"{output_dir}/correlation_summary_for_paper.txt", 'w') as f:
            f.write(corr_summary)

        print(f"Exported all results to {output_dir}")
    
    def compute_framework_faithfulness_correlations(self) -> pd.DataFrame:
        """
        Compute correlations between faithfulness metrics from different frameworks.

        This is the key comparison metric for inter-framework agreement.

        Returns:
            DataFrame with framework pairs and their faithfulness correlations
        """
        faith_cols = [c for c in self.metric_cols if 'faithfulness' in c.lower()]

        if len(faith_cols) < 2:
            return pd.DataFrame()

        results = []
        for i, col1 in enumerate(faith_cols):
            for col2 in faith_cols[i+1:]:
                fw1 = col1.split('_')[0]
                fw2 = col2.split('_')[0]

                r_pearson, p_pearson = pearsonr(self.df[col1], self.df[col2])
                r_spearman, p_spearman = spearmanr(self.df[col1], self.df[col2])
                tau, p_kendall = kendalltau(self.df[col1], self.df[col2])

                results.append({
                    'framework_1': fw1,
                    'framework_2': fw2,
                    'metric_1': col1,
                    'metric_2': col2,
                    'pearson_r': round(r_pearson, 4),
                    'pearson_p': p_pearson,
                    'spearman_rho': round(r_spearman, 4),
                    'spearman_p': p_spearman,
                    'kendall_tau': round(tau, 4),
                    'kendall_p': p_kendall
                })

        return pd.DataFrame(results)

    def generate_correlation_summary_for_paper(self) -> str:
        """
        Generate a formatted summary of actual correlation values for paper tables.

        This ensures paper claims match computed values.

        Returns:
            Formatted string with actual computed correlations
        """
        faith_corr = self.compute_framework_faithfulness_correlations()

        output = []
        output.append("=" * 70)
        output.append("ACTUAL COMPUTED CORRELATIONS FOR PAPER TABLES")
        output.append("=" * 70)
        output.append("\nThese are the actual correlations computed from evaluation_scores.csv")
        output.append("Use these values when updating paper tables to ensure accuracy.\n")

        output.append("--- Framework Faithfulness Correlations (Pearson) ---")
        output.append(f"{'Framework 1':<15} {'Framework 2':<15} {'Pearson r':<12} {'p-value':<12}")
        output.append("-" * 55)

        for _, row in faith_corr.iterrows():
            output.append(f"{row['framework_1']:<15} {row['framework_2']:<15} {row['pearson_r']:<12.4f} {row['pearson_p']:<12.4e}")

        output.append("\n--- For LaTeX Table (copy-paste ready) ---")
        for _, row in faith_corr.iterrows():
            output.append(f"{row['framework_1']} vs {row['framework_2']} & {row['pearson_r']:.2f} \\\\")

        return "\n".join(output)

    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for paper."""
        tables = {}
        
        # Table 1: Summary statistics
        summary = self.compute_summary_statistics()
        tables['summary'] = summary[['mean', 'std', 'min', 'max']].to_latex(
            float_format="%.3f",
            caption="Summary statistics for all evaluation metrics",
            label="tab:summary"
        )
        
        # Table 2: Faithfulness correlations with CIs
        bootstrap_ci = self.compute_all_bootstrap_ci()
        tables['correlations_ci'] = bootstrap_ci.to_latex(
            float_format="%.3f",
            index=False,
            caption="Faithfulness correlations with 95\\% bootstrap confidence intervals",
            label="tab:corr_ci"
        )
        
        # Table 3: Domain ANOVA
        anova = self.compute_domain_anova()
        anova_subset = anova[['metric', 'f_statistic', 'p_value', 'eta_squared']]
        tables['domain_anova'] = anova_subset.to_latex(
            float_format="%.3f",
            index=False,
            caption="ANOVA results for domain effect on evaluation metrics",
            label="tab:anova"
        )
        
        return tables
