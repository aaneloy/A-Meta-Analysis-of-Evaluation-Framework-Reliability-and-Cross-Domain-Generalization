"""
ARES Evaluator
==============

Implementation of ARES (Automated Retrieval Evaluation System) metrics.

Reference: Saad-Falcon et al., "ARES: An Automated Evaluation Framework for 
Retrieval-Augmented Generation Systems", NAACL 2024
arXiv:2311.09476
"""

from typing import Dict
import numpy as np
from . import HeuristicEvaluator


class ARESEvaluator(HeuristicEvaluator):
    """
    ARES-style evaluation framework.
    
    ARES differs from RAGAS by using fine-tuned classifier judges
    instead of prompting LLMs. It also uses Prediction-Powered Inference (PPI)
    for confidence intervals.
    
    Metrics:
        - context_relevance: Is retrieved context relevant to query?
        - answer_faithfulness: Is answer grounded in context?
        - answer_relevance: Is answer relevant to query?
    """
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="ARES",
            metrics=["context_relevance", "answer_faithfulness", "answer_relevance"],
            noise_std=noise_std
        )
        # ARES uses classifier confidence thresholds
        self.confidence_threshold = 0.5
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a sample using ARES-style classifier approach.
        
        ARES trains DeBERTa-v3 classifiers for each metric,
        then uses PPI for statistically valid estimates.
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        
        # Context Relevance (binary classification simulated as score)
        context_relevance = self._compute_context_relevance(question, context)
        
        # Answer Faithfulness (grounding check)
        answer_faithfulness = self._compute_answer_faithfulness(answer, context)
        
        # Answer Relevance (query-answer alignment)
        answer_relevance = self._compute_answer_relevance(question, answer)
        
        return {
            'context_relevance': self.add_noise(context_relevance),
            'answer_faithfulness': self.add_noise(answer_faithfulness),
            'answer_relevance': self.add_noise(answer_relevance)
        }
    
    def _compute_context_relevance(self, question: str, context: str) -> float:
        """
        Compute context relevance.
        
        ARES classifier: Given (question, context), is context relevant?
        This is a binary classification task in the original paper.
        """
        if not question or not context:
            return 0.0
        
        # Simulate classifier confidence
        # Higher overlap = higher confidence of relevance
        word_overlap = self.word_overlap(question, context)
        
        # Check for entity overlap (proper nouns, numbers)
        q_tokens = question.split()
        c_tokens = context.split()
        
        # Find potential entities (capitalized words, numbers)
        q_entities = set(t for t in q_tokens if t[0].isupper() or t.isdigit())
        c_entities = set(t for t in c_tokens if len(t) > 0 and (t[0].isupper() or t.isdigit()))
        
        entity_overlap = len(q_entities & c_entities) / max(len(q_entities), 1)
        
        # Combine for confidence score
        confidence = 0.6 * word_overlap + 0.4 * entity_overlap
        
        # ARES uses sigmoid-like transformation for classifier output
        return self._sigmoid_transform(confidence)
    
    def _compute_answer_faithfulness(self, answer: str, context: str) -> float:
        """
        Compute answer faithfulness.
        
        ARES classifier: Is answer grounded in context?
        """
        if not answer or not context:
            return 0.0
        
        # ARES checks for grounding at multiple granularities
        
        # Word-level grounding
        word_grounding = self.coverage_score(answer, context)
        
        # Phrase-level grounding
        phrase_grounding = self.ngram_overlap(answer, context, n=3)
        
        # Claim-level grounding
        claims = self.extract_claims(answer)
        if claims:
            claim_scores = []
            for claim in claims:
                score = self.coverage_score(claim, context)
                claim_scores.append(score)
            claim_grounding = np.mean(claim_scores)
        else:
            claim_grounding = word_grounding
        
        # Weighted combination
        confidence = 0.3 * word_grounding + 0.3 * phrase_grounding + 0.4 * claim_grounding
        
        return self._sigmoid_transform(confidence)
    
    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """
        Compute answer relevance.
        
        ARES classifier: Is answer relevant to query?
        """
        if not question or not answer:
            return 0.0
        
        # Word overlap
        word_overlap = self.word_overlap(question, answer)
        
        # Check if answer addresses the question type
        # (who/what/when/where/why/how questions)
        question_type_score = self._check_question_type_match(question, answer)
        
        # Combine
        confidence = 0.5 * word_overlap + 0.5 * question_type_score
        
        return self._sigmoid_transform(confidence)
    
    def _sigmoid_transform(self, x: float, steepness: float = 5) -> float:
        """
        Apply sigmoid transformation to convert raw score to confidence.
        
        This simulates the classifier's probability output.
        """
        # Shift and scale sigmoid
        return 1 / (1 + np.exp(-steepness * (x - 0.3)))
    
    def _check_question_type_match(self, question: str, answer: str) -> float:
        """
        Check if answer matches the expected question type.
        """
        q_lower = question.lower()
        a_lower = answer.lower()
        
        # Simple heuristics for question types
        if q_lower.startswith('who'):
            # Answer should contain names or entities
            has_caps = any(c.isupper() for c in answer[1:] if c.isalpha())
            return 0.7 if has_caps else 0.3
        
        elif q_lower.startswith('when'):
            # Answer should contain dates or times
            has_numbers = any(c.isdigit() for c in answer)
            time_words = ['year', 'month', 'day', 'century', 'decade', 'ago']
            has_time_word = any(w in a_lower for w in time_words)
            return 0.7 if (has_numbers or has_time_word) else 0.3
        
        elif q_lower.startswith('where'):
            # Answer should contain locations
            location_indicators = ['in', 'at', 'near', 'city', 'country', 'region']
            has_location = any(w in a_lower for w in location_indicators)
            return 0.7 if has_location else 0.4
        
        elif q_lower.startswith(('what', 'which')):
            # General - check for substantive answer
            return min(1.0, len(answer.split()) / 10)  # Longer = more substantive
        
        elif q_lower.startswith(('how', 'why')):
            # Explanatory - check for explanation markers
            explanation_markers = ['because', 'therefore', 'thus', 'due to', 'since', 'by']
            has_explanation = any(m in a_lower for m in explanation_markers)
            return 0.6 if has_explanation else 0.4
        
        # Default
        return 0.5
