"""
GaRAGe Evaluator
================

Grounding Annotations for RAG Evaluation Benchmark.

Reference: Sorodoc et al., "GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation"
arXiv:2506.07671, ACL 2025 Findings

GaRAGe provides fine-grained evaluation of whether LLMs can identify relevant
grounding when generating RAG answers. Contains 2366 questions with 35K+
annotated passages from both private documents and the Web.

Key Findings:
- Models tend to over-summarize rather than ground strictly on relevant passages
- Best models achieve at most 60% Relevance-Aware Factuality Score
- Models struggle to deflect when no relevant grounding available (31% TPR max)

Metrics:
- relevance_aware_factuality: Factuality considering only relevant passages
- grounding_precision: Precision of grounding on relevant passages
- deflection_score: Ability to refuse when context is insufficient
- over_summarization: Penalty for including irrelevant information
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import re
from . import HeuristicEvaluator


class GaRAGeEvaluator(HeuristicEvaluator):
    """
    GaRAGe: Grounding Annotations for RAG Evaluation (ACL 2025)
    
    Evaluates fine-grained grounding ability:
    - Can the model identify ONLY relevant passages?
    - Does the model deflect when context is insufficient?
    - Does the model over-summarize irrelevant content?
    
    Paper shows state-of-the-art LLMs struggle with grounding:
    - Max Relevance-Aware Factuality: 60%
    - Max deflection TPR: 31%
    """
    
    def __init__(self, noise_std: float = 0.05):
        super().__init__(
            name="GaRAGe",
            metrics=["relevance_aware_factuality", "grounding_precision", 
                     "deflection_score", "over_summarization"],
            noise_std=noise_std
        )
        self.version = "2025-06"  # June 2025
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using GaRAGe methodology.
        
        GaRAGe annotates each passage with relevance labels,
        then evaluates if the answer uses ONLY relevant passages.
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})
        
        # Simulate passage relevance annotation
        passages, relevance_labels = self._annotate_passages(question, context)
        
        # Relevance-Aware Factuality: factuality on relevant passages only
        raf = self._compute_relevance_aware_factuality(
            answer, passages, relevance_labels
        )
        
        # Grounding Precision: using only relevant passages
        grounding_precision = self._compute_grounding_precision(
            answer, passages, relevance_labels
        )
        
        # Deflection Score: refusing when insufficient context
        deflection = self._compute_deflection_score(
            answer, passages, relevance_labels
        )
        
        # Over-summarization: including irrelevant content
        over_summarization = self._compute_over_summarization(
            answer, passages, relevance_labels
        )
        
        return {
            'relevance_aware_factuality': self.add_noise(raf),
            'grounding_precision': self.add_noise(grounding_precision),
            'deflection_score': self.add_noise(deflection),
            'over_summarization': self.add_noise(over_summarization)
        }
    
    def _annotate_passages(
        self, 
        question: str, 
        context: str
    ) -> Tuple[List[str], List[bool]]:
        """
        Simulate passage relevance annotation.
        
        In real GaRAGe, this is human-annotated. We simulate by
        computing semantic relevance to the question.
        """
        if not context:
            return [], []
        
        # Split context into passages (by sentence or paragraph)
        passages = [p.strip() for p in re.split(r'[.!?]\s+|\n\n', context) if len(p.strip()) > 20]
        
        if not passages:
            passages = [context]
        
        # Annotate relevance based on question overlap
        relevance_labels = []
        question_words = set(question.lower().split())
        
        # Remove stopwords
        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 
                     'on', 'with', 'how', 'why', 'when', 'where', 'who', 'which',
                     'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                     'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        question_keywords = question_words - stopwords
        
        for passage in passages:
            passage_words = set(passage.lower().split())
            overlap = len(question_keywords & passage_words)
            
            # Passage is relevant if it shares key terms with question
            is_relevant = overlap >= max(1, len(question_keywords) * 0.2)
            relevance_labels.append(is_relevant)
        
        return passages, relevance_labels
    
    def _compute_relevance_aware_factuality(
        self,
        answer: str,
        passages: List[str],
        relevance_labels: List[bool]
    ) -> float:
        """
        Compute factuality considering only relevant passages.
        
        GaRAGe's key metric: answer should be factual w.r.t. relevant passages.
        """
        if not answer or not passages:
            return 0.5
        
        # Get relevant passages only
        relevant_passages = [p for p, r in zip(passages, relevance_labels) if r]
        
        if not relevant_passages:
            # No relevant context - should deflect
            # If answer is substantive, penalize
            if len(answer.split()) > 10:
                return 0.3  # Should have deflected
            return 0.7  # Short/deflective response is better
        
        # Compute factuality against relevant passages
        relevant_context = ' '.join(relevant_passages)
        
        claims = self.extract_claims(answer)
        if not claims:
            return 0.5
        
        supported = 0
        for claim in claims:
            # Check if claim is supported by relevant context
            support = self.coverage_score(claim, relevant_context)
            if support > 0.25:
                supported += 1
        
        return supported / len(claims)
    
    def _compute_grounding_precision(
        self,
        answer: str,
        passages: List[str],
        relevance_labels: List[bool]
    ) -> float:
        """
        Compute precision of grounding on relevant passages.
        
        High precision = answer draws from relevant passages only.
        """
        if not answer or not passages:
            return 0.5
        
        answer_words = set(answer.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'or', 'but', 'not', 'this', 'that', 'it', 'its'}
        answer_content = answer_words - stopwords
        
        if not answer_content:
            return 0.5
        
        # Count words from relevant vs irrelevant passages
        relevant_contribution = 0
        irrelevant_contribution = 0
        
        for passage, is_relevant in zip(passages, relevance_labels):
            passage_words = set(passage.lower().split()) - stopwords
            overlap = len(answer_content & passage_words)
            
            if is_relevant:
                relevant_contribution += overlap
            else:
                irrelevant_contribution += overlap
        
        total = relevant_contribution + irrelevant_contribution
        if total == 0:
            return 0.5
        
        precision = relevant_contribution / total
        return precision
    
    def _compute_deflection_score(
        self,
        answer: str,
        passages: List[str],
        relevance_labels: List[bool]
    ) -> float:
        """
        Compute ability to deflect when context is insufficient.
        
        GaRAGe found models struggle with deflection (max 31% TPR).
        """
        relevant_count = sum(relevance_labels)
        
        if relevant_count > 0:
            # There is relevant context - deflection not needed
            # Score based on how well the model uses the context
            return 0.7 + 0.3 * (relevant_count / len(relevance_labels))
        
        # No relevant context - model SHOULD deflect
        # Check if answer looks like a deflection
        answer_lower = answer.lower()
        
        deflection_phrases = [
            'i don\'t know', 'i cannot', 'i can\'t', 'i\'m not sure',
            'there is no information', 'the context does not',
            'based on the provided', 'insufficient information',
            'cannot determine', 'not enough information',
            'unable to answer', 'no relevant', 'not mentioned',
            'the documents do not', 'i don\'t have enough'
        ]
        
        is_deflection = any(phrase in answer_lower for phrase in deflection_phrases)
        
        # Also check answer length (short = more likely deflection)
        is_short = len(answer.split()) < 15
        
        if is_deflection:
            return 0.9
        elif is_short:
            return 0.6
        else:
            return 0.2  # Failed to deflect when should have
    
    def _compute_over_summarization(
        self,
        answer: str,
        passages: List[str],
        relevance_labels: List[bool]
    ) -> float:
        """
        Compute over-summarization penalty.
        
        GaRAGe found models tend to over-summarize, including
        information from irrelevant passages.
        
        Returns: Score where 1.0 = no over-summarization (good)
        """
        if not answer or not passages:
            return 0.5
        
        # Get irrelevant passages
        irrelevant_passages = [p for p, r in zip(passages, relevance_labels) if not r]
        
        if not irrelevant_passages:
            return 1.0  # No irrelevant passages to over-summarize from
        
        irrelevant_context = ' '.join(irrelevant_passages)
        
        # Check how much of answer comes from irrelevant passages
        answer_words = set(answer.lower().split())
        irrelevant_words = set(irrelevant_context.lower().split())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'of', 'to', 'in', 'for', 
                     'on', 'with', 'and', 'or', 'but', 'this', 'that', 'it'}
        answer_content = answer_words - stopwords
        irrelevant_content = irrelevant_words - stopwords
        
        if not answer_content:
            return 0.5
        
        # Over-summarization: content from irrelevant passages
        irrelevant_overlap = len(answer_content & irrelevant_content)
        over_summarization_ratio = irrelevant_overlap / len(answer_content)
        
        # Return inverse (1 - ratio) so higher = better (less over-summarization)
        return 1.0 - min(1.0, over_summarization_ratio)
