"""
Failure Analysis Module
=======================

Identify and analyze cases where frameworks disagree significantly.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FailureCase:
    """Represents a case where frameworks significantly disagree."""
    sample_id: str
    domain: str
    question: str
    answer: str
    context_preview: str
    framework_scores: Dict[str, float]
    max_disagreement: float
    high_scorer: str
    low_scorer: str
    analysis: str


class FailureAnalyzer:
    """
    Analyze cases where evaluation frameworks disagree.
    
    Identifies samples with maximum disagreement and provides
    qualitative analysis for paper discussion.
    """
    
    def __init__(self, results_df: pd.DataFrame, samples: List[Dict]):
        """
        Initialize analyzer.
        
        Args:
            results_df: DataFrame with evaluation scores
            samples: List of original sample dictionaries
        """
        self.df = results_df.copy()
        self.samples = {s['id']: s for s in samples}
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

    def _framework_score_display(self, framework_scores: Dict[str, float], framework: str) -> str:
        """Return a display string for a framework score from a metric dict."""
        prefix = f"{framework}_"
        candidates = [(k, v) for k, v in framework_scores.items() if k.startswith(prefix)]
        if not candidates:
            return "N/A"

        for metric_name, score in candidates:
            if metric_name.lower() == f"{framework.lower()}_faithfulness":
                return f"{float(score):.3f}"

        for metric_name, score in candidates:
            if 'faithfulness' in metric_name.lower():
                return f"{float(score):.3f}"

        return f"{float(candidates[0][1]):.3f}"
    
    def find_disagreement_cases(self, metric_keyword: str = 'faithfulness', 
                                 top_n: int = 10) -> List[FailureCase]:
        """
        Find cases with highest framework disagreement.
        
        Args:
            metric_keyword: Metric type to analyze (e.g., 'faithfulness')
            top_n: Number of top disagreement cases to return
            
        Returns:
            List of FailureCase objects
        """
        # Get relevant columns
        relevant_cols = [c for c in self.metric_cols if metric_keyword in c.lower()]
        
        if len(relevant_cols) < 2:
            print(f"Not enough columns matching '{metric_keyword}'")
            return []
        
        # Compute disagreement for each sample
        disagreements = []
        
        for idx, row in self.df.iterrows():
            scores = {
                col: float(row[col]) for col in relevant_cols
                if pd.notna(row[col]) and np.isfinite(row[col])
            }
            if len(scores) < 2:
                continue
            
            max_score = max(scores.values())
            min_score = min(scores.values())
            disagreement = max_score - min_score
            
            high_scorer = max(scores, key=scores.get)
            low_scorer = min(scores, key=scores.get)
            
            disagreements.append({
                'idx': idx,
                'id': row['id'],
                'domain': row['domain'],
                'scores': scores,
                'disagreement': disagreement,
                'high_scorer': high_scorer.split('_')[0],
                'low_scorer': low_scorer.split('_')[0],
                'high_score': max_score,
                'low_score': min_score
            })
        
        # Sort by disagreement
        disagreements.sort(key=lambda x: x['disagreement'], reverse=True)
        
        # Create FailureCase objects for top N
        failure_cases = []
        
        for d in disagreements[:top_n]:
            sample = self.samples.get(d['id'], {})
            
            # Generate analysis
            analysis = self._generate_analysis(d, sample)
            
            failure_cases.append(FailureCase(
                sample_id=d['id'],
                domain=d['domain'],
                question=sample.get('question', '')[:200],
                answer=sample.get('answer', '')[:300],
                context_preview=str(sample.get('context', ''))[:200] + '...',
                framework_scores=d['scores'],
                max_disagreement=d['disagreement'],
                high_scorer=d['high_scorer'],
                low_scorer=d['low_scorer'],
                analysis=analysis
            ))
        
        return failure_cases
    
    def _generate_analysis(self, disagreement: Dict, sample: Dict) -> str:
        """Generate qualitative analysis for a disagreement case."""
        high = disagreement['high_scorer']
        low = disagreement['low_scorer']
        high_score = disagreement['high_score']
        low_score = disagreement['low_score']
        
        analysis_parts = []
        
        # Score difference
        analysis_parts.append(
            f"{high} scores {high_score:.2f} while {low} scores {low_score:.2f} "
            f"(difference: {disagreement['disagreement']:.2f})."
        )
        
        # Methodology difference
        methodology = {
            # Traditional (2020-2024)
            'RAGAS': 'LLM-as-judge with claim decomposition',
            'DeepEval': 'LLM-as-judge with strict rubrics',
            'RAGChecker': 'NLI-based claim verification',
            'TRACe': 'Token-level annotation',
            'ARES': 'Fine-tuned classifier',
            'G-Eval': 'LLM chain-of-thought',
            'BERTScore': 'Embedding similarity',
            'UniEval': 'Boolean QA approach',
            'QAFactEval': 'QA-based verification',
            # 2025 frameworks
            'ReDeEP': 'Mechanistic interpretability (FFN/attention analysis)',
            'LettuceDetect': 'ModernBERT token-level classification',
            'FaithJudge': 'LLM-as-judge with few-shot examples',
            'LRP4RAG': 'Layer-wise relevance propagation',
            'LUMINA': 'Context-knowledge signal analysis (MMD)',
            'HALT-RAG': 'Dual NLI ensemble with calibration',
            'MetaRAG': 'Metamorphic testing (synonym/antonym)',
            'KG-RAG': 'Knowledge graph-based verification',
            'GaRAGe': 'Grounding annotation analysis',
            'HSAD': 'Hidden state spectral analysis',
            # 2026 frameworks
            'SIRG': 'Semantic-level internal reasoning graph'
        }
        
        high_method = methodology.get(high, 'Unknown')
        low_method = methodology.get(low, 'Unknown')
        
        analysis_parts.append(
            f"Methodological difference: {high} uses {high_method}, "
            f"while {low} uses {low_method}."
        )
        
        # Potential explanation based on answer characteristics
        answer = sample.get('answer', '')
        context = str(sample.get('context', ''))
        
        if len(answer.split()) > 50:
            analysis_parts.append(
                "Long answer may cause scoring variance - "
                "claim-based methods may find more unsupported statements."
            )
        
        if len(context.split()) > 500:
            analysis_parts.append(
                "Long context may affect precision-based metrics differently."
            )
        
        return ' '.join(analysis_parts)
    
    def compute_disagreement_statistics(self, metric_keyword: str = 'faithfulness') -> Dict:
        """
        Compute overall disagreement statistics.
        
        Returns:
            Dictionary with mean, std, max disagreement and affected samples
        """
        relevant_cols = [c for c in self.metric_cols if metric_keyword in c.lower()]
        
        if len(relevant_cols) < 2:
            return {'error': f'Not enough columns matching {metric_keyword}'}
        
        disagreements = []
        for idx, row in self.df.iterrows():
            scores = [
                float(row[col]) for col in relevant_cols
                if pd.notna(row[col]) and np.isfinite(row[col])
            ]
            if len(scores) < 2:
                continue
            disagreements.append(max(scores) - min(scores))

        if not disagreements:
            return {'error': f'No valid rows with at least 2 finite scores for {metric_keyword}'}

        disagreements = np.array(disagreements)
        
        return {
            'mean_disagreement': float(np.mean(disagreements)),
            'std_disagreement': float(np.std(disagreements)),
            'max_disagreement': float(np.max(disagreements)),
            'min_disagreement': float(np.min(disagreements)),
            'samples_with_high_disagreement': int(np.sum(disagreements > 0.3)),
            'percent_high_disagreement': float(np.mean(disagreements > 0.3) * 100)
        }
    
    def find_agreement_cases(self, metric_keyword: str = 'faithfulness',
                             top_n: int = 5) -> List[Dict]:
        """
        Find cases where all frameworks agree (for contrast).
        
        Returns:
            List of dictionaries with sample info and scores
        """
        relevant_cols = [c for c in self.metric_cols if metric_keyword in c.lower()]
        
        agreements = []
        for idx, row in self.df.iterrows():
            scores = [
                float(row[col]) for col in relevant_cols
                if pd.notna(row[col]) and np.isfinite(row[col])
            ]
            if len(scores) < 2:
                continue
            disagreement = max(scores) - min(scores)
            
            agreements.append({
                'id': row['id'],
                'domain': row['domain'],
                'disagreement': disagreement,
                'mean_score': np.mean(scores),
                'scores': {
                    col: float(row[col]) for col in relevant_cols
                    if pd.notna(row[col]) and np.isfinite(row[col])
                }
            })
        
        # Sort by lowest disagreement
        agreements.sort(key=lambda x: x['disagreement'])
        
        return agreements[:top_n]
    
    def analyze_by_domain(self, metric_keyword: str = 'faithfulness') -> pd.DataFrame:
        """
        Analyze disagreement patterns by domain.
        
        Returns:
            DataFrame with disagreement statistics per domain
        """
        relevant_cols = [c for c in self.metric_cols if metric_keyword in c.lower()]
        
        results = []
        for domain in self.df['domain'].unique():
            domain_df = self.df[self.df['domain'] == domain]
            
            disagreements = []
            for idx, row in domain_df.iterrows():
                scores = [
                    float(row[col]) for col in relevant_cols
                    if pd.notna(row[col]) and np.isfinite(row[col])
                ]
                if len(scores) < 2:
                    continue
                disagreements.append(max(scores) - min(scores))

            if not disagreements:
                continue

            disagreements = np.array(disagreements)
            
            results.append({
                'domain': domain,
                'mean_disagreement': np.mean(disagreements),
                'std_disagreement': np.std(disagreements),
                'max_disagreement': np.max(disagreements),
                'n_samples': len(domain_df)
            })
        
        return pd.DataFrame(results)
    
    def export_failure_cases(self, output_path: str, top_n: int = 10):
        """Export failure cases to JSON for paper discussion."""
        cases = self.find_disagreement_cases(top_n=top_n)
        
        export_data = {
            'statistics': self.compute_disagreement_statistics(),
            'domain_analysis': self.analyze_by_domain().to_dict('records'),
            'failure_cases': [
                {
                    'sample_id': c.sample_id,
                    'domain': c.domain,
                    'question': c.question,
                    'answer': c.answer,
                    'framework_scores': c.framework_scores,
                    'max_disagreement': c.max_disagreement,
                    'high_scorer': c.high_scorer,
                    'low_scorer': c.low_scorer,
                    'analysis': c.analysis
                }
                for c in cases
            ],
            'agreement_cases': self.find_agreement_cases()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported failure analysis to {output_path}")
        
        return export_data
    
    def print_summary(self):
        """Print failure analysis summary to console."""
        stats = self.compute_disagreement_statistics()
        
        print("\n" + "="*60)
        print("FAILURE CASE ANALYSIS")
        print("="*60)
        
        print(f"\nDisagreement Statistics:")
        print(f"  Mean disagreement: {stats['mean_disagreement']:.3f}")
        print(f"  Std disagreement: {stats['std_disagreement']:.3f}")
        print(f"  Max disagreement: {stats['max_disagreement']:.3f}")
        print(f"  Samples with >0.3 disagreement: {stats['samples_with_high_disagreement']} ({stats['percent_high_disagreement']:.1f}%)")
        
        print("\nDomain Analysis:")
        domain_df = self.analyze_by_domain()
        print(domain_df.to_string(index=False))
        
        print("\nTop 5 Disagreement Cases:")
        cases = self.find_disagreement_cases(top_n=5)
        for i, case in enumerate(cases, 1):
            high_score_display = self._framework_score_display(case.framework_scores, case.high_scorer)
            low_score_display = self._framework_score_display(case.framework_scores, case.low_scorer)
            print(f"\n{i}. {case.sample_id} ({case.domain})")
            print(f"   Disagreement: {case.max_disagreement:.3f}")
            print(f"   {case.high_scorer}: {high_score_display} vs "
                  f"{case.low_scorer}: {low_score_display}")
            print(f"   Analysis: {case.analysis[:150]}...")
