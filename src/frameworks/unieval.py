"""
UniEval Evaluator
=================

Implementation of UniEval metrics.

Reference: Zhong et al., "Towards a Unified Multi-Dimensional Evaluator for Text Generation"
EMNLP 2022, arXiv:2210.07197
"""

from typing import Dict
import numpy as np
from . import HeuristicEvaluator


class UniEvalEvaluator(HeuristicEvaluator):
    """
    UniEval-style evaluation framework.
    
    UniEval unifies multiple evaluation dimensions into a single
    Boolean QA framework. Given a source and target, it asks
    "Is this [dimension]?" and uses the model's answer probability.
    
    Metrics:
        - coherence: Is the text coherent?
        - consistency: Is the text consistent with source?
        - fluency: Is the text fluent?
        - relevance: Is the text relevant?
        - overall: Aggregated quality score
    """
    
    def __init__(self, noise_std: float = 0.05):
        super().__init__(
            name="UniEval",
            metrics=["coherence", "consistency", "fluency", "relevance", "overall"],
            noise_std=noise_std
        )
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a sample using UniEval's unified QA approach.
        
        UniEval methodology:
        1. Formulate evaluation as Boolean questions
        2. Use T5-based model to answer yes/no
        3. Score = probability of "Yes"
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        
        # Compute individual dimensions
        coherence = self._evaluate_coherence(answer)
        consistency = self._evaluate_consistency(answer, context)
        fluency = self._evaluate_fluency(answer)
        relevance = self._evaluate_relevance(question, answer)
        
        # Overall: weighted average
        overall = (0.25 * coherence + 0.3 * consistency + 
                   0.2 * fluency + 0.25 * relevance)
        
        return {
            'coherence': self.add_noise(coherence),
            'consistency': self.add_noise(consistency),
            'fluency': self.add_noise(fluency),
            'relevance': self.add_noise(relevance),
            'overall': self.add_noise(overall)
        }
    
    def _evaluate_coherence(self, text: str) -> float:
        """
        Evaluate: "Is this text coherent?"
        
        UniEval coherence checks for logical flow and structure.
        """
        if not text:
            return 0.0
        
        sentences = self.extract_claims(text)
        
        if len(sentences) <= 1:
            return 0.8  # Single sentence is coherent by default
        
        # Check for discourse markers
        discourse_markers = [
            'however', 'but', 'although', 'while', 'whereas',  # Contrast
            'therefore', 'thus', 'consequently', 'hence',  # Consequence
            'moreover', 'furthermore', 'additionally', 'also',  # Addition
            'first', 'second', 'third', 'finally', 'then', 'next',  # Sequence
            'for example', 'such as', 'specifically',  # Exemplification
            'in other words', 'that is', 'namely'  # Clarification
        ]
        
        text_lower = text.lower()
        marker_count = sum(1 for m in discourse_markers if m in text_lower)
        marker_score = min(1.0, marker_count / 2)  # 2+ markers = good
        
        # Check sentence connectivity (shared vocabulary)
        connectivity_scores = []
        for i in range(len(sentences) - 1):
            overlap = self.word_overlap(sentences[i], sentences[i+1])
            connectivity_scores.append(overlap)
        
        avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 0.5
        
        return 0.4 * marker_score + 0.6 * min(1.0, avg_connectivity * 2)
    
    def _evaluate_consistency(self, text: str, source: str) -> float:
        """
        Evaluate: "Is this text consistent with the source?"
        
        UniEval consistency checks factual alignment.
        """
        if not text or not source:
            return 0.0
        
        # Word-level grounding
        word_consistency = self.coverage_score(text, source)
        
        # Phrase-level grounding
        bigram_consistency = self.ngram_overlap(text, source, n=2)
        trigram_consistency = self.ngram_overlap(text, source, n=3)
        
        # Claim-level consistency
        claims = self.extract_claims(text)
        if claims:
            claim_scores = [self.coverage_score(c, source) for c in claims]
            claim_consistency = np.mean(claim_scores)
        else:
            claim_consistency = word_consistency
        
        return (0.2 * word_consistency + 0.2 * bigram_consistency + 
                0.2 * trigram_consistency + 0.4 * claim_consistency)
    
    def _evaluate_fluency(self, text: str) -> float:
        """
        Evaluate: "Is this text fluent?"
        
        UniEval fluency checks grammaticality and readability.
        """
        if not text:
            return 0.0
        
        # Check basic fluency indicators
        
        # 1. Proper sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.3
        
        # Average sentence length (10-25 words is ideal)
        lengths = [len(s.split()) for s in sentences]
        avg_len = np.mean(lengths)
        
        if 8 <= avg_len <= 25:
            length_score = 1.0
        elif avg_len < 4:
            length_score = 0.3
        elif avg_len > 40:
            length_score = 0.4
        else:
            length_score = 0.7
        
        # 2. No word repetition
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        repetition_score = min(1.0, unique_ratio * 1.2)  # Allow some repetition
        
        # 3. Proper capitalization and punctuation
        proper_start = text[0].isupper() if text else False
        proper_end = text.rstrip()[-1] in '.!?' if text.rstrip() else False
        format_score = (0.5 * proper_start + 0.5 * proper_end)
        
        return 0.4 * length_score + 0.4 * repetition_score + 0.2 * format_score
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate: "Is this answer relevant to the question?"
        
        UniEval relevance checks topical alignment.
        """
        if not question or not answer:
            return 0.0
        
        # Extract key terms from question
        q_words = set(question.lower().split())
        stopwords = {
            'what', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on',
            'with', 'how', 'why', 'when', 'where', 'who', 'which', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their'
        }
        
        key_terms = q_words - stopwords
        
        if not key_terms:
            # Fall back to general overlap
            return self.word_overlap(question, answer)
        
        # Check coverage of key terms
        a_words = set(answer.lower().split())
        covered = key_terms & a_words
        
        direct_coverage = len(covered) / len(key_terms)
        
        # Also check word overlap for broader relevance
        general_overlap = self.word_overlap(question, answer)
        
        return 0.6 * direct_coverage + 0.4 * general_overlap
