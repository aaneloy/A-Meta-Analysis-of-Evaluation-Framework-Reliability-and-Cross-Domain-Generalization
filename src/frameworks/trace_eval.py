"""
TRACe Evaluator
===============

Implementation of TRACe (Token-level RAG Annotation for Comprehensive Evaluation) metrics.

Reference: Friel et al., "RAGBench: Explainable Benchmark for Retrieval-Augmented 
Generation Systems", arXiv:2407.11005
"""

from typing import Dict, List
import numpy as np
from . import HeuristicEvaluator


class TRACeEvaluator(HeuristicEvaluator):
    """
    TRACe-style evaluation framework.
    
    TRACe provides token-level annotations for RAG evaluation, making it
    more fine-grained than sentence or claim-level approaches.
    
    We use TRACe as a pseudo ground truth since RAGBench includes
    token-level annotations.
    
    Metrics:
        - utilization: How much of context is used in answer
        - relevance: How relevant is context to question
        - adherence: How closely answer follows context
        - completeness: How complete is the answer
    """
    
    def __init__(self, noise_std: float = 0.03):  # Lower noise - closer to ground truth
        super().__init__(
            name="TRACe",
            metrics=["utilization", "relevance", "adherence", "completeness"],
            noise_std=noise_std
        )
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a sample using TRACe-style token-level metrics.
        
        TRACe metrics:
        - Utilization: Token overlap between answer and context
        - Relevance: Token overlap between question and context
        - Adherence: How faithfully answer follows context
        - Completeness: Coverage of expected information
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})
        
        # Check for actual TRACe annotations
        if 'all_labels' in sample.get('metadata', {}):
            return self._evaluate_with_annotations(sample)
        
        # Otherwise compute heuristic TRACe scores
        utilization = self._compute_utilization(answer, context)
        relevance = self._compute_relevance(question, context)
        adherence = self._compute_adherence(answer, context)
        completeness = self._compute_completeness(answer, ground_truth)
        
        return {
            'utilization': self.add_noise(utilization),
            'relevance': self.add_noise(relevance),
            'adherence': self.add_noise(adherence),
            'completeness': self.add_noise(completeness)
        }
    
    def _evaluate_with_annotations(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using actual TRACe annotations if available.
        """
        all_labels = sample.get('ground_truth', {})
        
        # TRACe annotations include token-level labels
        # Extract aggregate scores
        utilization = all_labels.get('utilization', 0.5)
        relevance = all_labels.get('relevance', 0.5)
        adherence = all_labels.get('adherence', 0.5)
        completeness = all_labels.get('completeness', 0.5)
        
        return {
            'utilization': self.add_noise(float(utilization)),
            'relevance': self.add_noise(float(relevance)),
            'adherence': self.add_noise(float(adherence)),
            'completeness': self.add_noise(float(completeness))
        }
    
    def _compute_utilization(self, answer: str, context: str) -> float:
        """
        Compute utilization score.
        
        Token-level measure of how much context appears in answer.
        """
        if not answer or not context:
            return 0.0
        
        # Token overlap
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        # What fraction of answer tokens come from context
        from_context = answer_tokens & context_tokens
        utilization = len(from_context) / len(answer_tokens)
        
        return utilization
    
    def _compute_relevance(self, question: str, context: str) -> float:
        """
        Compute relevance score.
        
        How relevant is the retrieved context to the question.
        """
        if not question or not context:
            return 0.0
        
        return self.word_overlap(question, context)
    
    def _compute_adherence(self, answer: str, context: str) -> float:
        """
        Compute adherence score.
        
        How closely does the answer adhere to/follow the context.
        Stricter than utilization - checks phrase-level alignment.
        """
        if not answer or not context:
            return 0.0
        
        # Combine word and phrase overlap
        word_adherence = self.coverage_score(answer, context)
        phrase_adherence = self.ngram_overlap(answer, context, n=3)
        
        return 0.6 * word_adherence + 0.4 * phrase_adherence
    
    def _compute_completeness(self, answer: str, ground_truth: Dict) -> float:
        """
        Compute completeness score.
        
        Does the answer contain all expected information?
        """
        if not answer:
            return 0.0
        
        # Extract ground truth text
        if isinstance(ground_truth, dict):
            gt_text = ' '.join(str(v) for v in ground_truth.values() if v)
        else:
            gt_text = str(ground_truth)
        
        if not gt_text:
            # Without ground truth, estimate based on answer structure
            # Longer, more detailed answers are typically more complete
            answer_sentences = self.sentence_count(answer)
            return min(1.0, answer_sentences / 3.0)  # 3+ sentences = complete
        
        # Check coverage of ground truth
        return self.word_overlap(gt_text, answer)
