"""
HSAD Evaluator
==============

Hidden Signal Analysis-based Detection for LLM Hallucination.

Reference: Li et al., "LLM Hallucination Detection: A Fast Fourier Transform 
Method Based on Hidden Layer Temporal Signals"
arXiv:2509.13154, September 2025

HSAD proposes a novel approach that models temporal dynamics of hidden
representations during autoregressive generation. It applies FFT to extract
frequency-domain features and achieves 10+ percentage points improvement
over prior state-of-the-art on TruthfulQA.

Key Innovation:
- Models hidden-layer signals by sampling activations across layers
- Applies FFT to obtain frequency-domain representations
- Extracts strongest non-DC frequency component as spectral features
- Identifies optimal observation points for reliable detection

Metrics:
- spectral_consistency: Frequency-domain consistency score
- temporal_stability: Stability of hidden representations
- confidence_score: Model confidence based on spectral features
- hallucination_probability: Probability of hallucination (lower = better)
"""

from typing import Dict, List, Tuple
import numpy as np
import re
from . import HeuristicEvaluator


class HSADEvaluator(HeuristicEvaluator):
    """
    HSAD: Hidden Signal Analysis-based Detection (September 2025)
    
    Novel approach to hallucination detection using:
    - Temporal dynamics of hidden representations
    - Fast Fourier Transform (FFT) analysis
    - Spectral feature extraction
    
    Achieves 10+ percentage points improvement over prior SOTA on TruthfulQA.
    """
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="HSAD",
            metrics=["spectral_consistency", "temporal_stability", 
                     "confidence_score", "hallucination_probability"],
            noise_std=noise_std
        )
        self.version = "2025-09"  # September 2025
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using HSAD methodology.
        
        HSAD analyzes the "reasoning dynamics" during generation.
        We simulate this by analyzing text patterns that correlate
        with hallucination vs. grounded responses.
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        
        # Simulate spectral analysis of generation dynamics
        spectral_consistency = self._compute_spectral_consistency(answer, context)
        
        # Temporal stability of the response
        temporal_stability = self._compute_temporal_stability(answer, context)
        
        # Confidence score based on spectral features
        confidence = self._compute_confidence_score(answer, context, question)
        
        # Hallucination probability (inverse of grounding)
        hallucination_prob = self._compute_hallucination_probability(
            spectral_consistency, temporal_stability, confidence
        )
        
        return {
            'spectral_consistency': self.add_noise(spectral_consistency),
            'temporal_stability': self.add_noise(temporal_stability),
            'confidence_score': self.add_noise(confidence),
            'hallucination_probability': self.add_noise(hallucination_prob)
        }
    
    def _compute_spectral_consistency(self, answer: str, context: str) -> float:
        """
        Compute spectral consistency score.
        
        HSAD finds that hallucinated content shows different frequency
        patterns in hidden states. We simulate by analyzing:
        - Vocabulary consistency with context
        - Semantic drift indicators
        - Unusual token patterns
        """
        if not answer or not context:
            return 0.5
        
        # Tokenize
        answer_tokens = answer.lower().split()
        context_tokens = context.lower().split()
        
        if not answer_tokens:
            return 0.5
        
        # Vocabulary consistency (simulates FFT frequency alignment)
        context_vocab = set(context_tokens)
        answer_in_context = sum(1 for t in answer_tokens if t in context_vocab)
        vocab_consistency = answer_in_context / len(answer_tokens)
        
        # N-gram consistency (simulates higher frequency components)
        answer_bigrams = set(zip(answer_tokens[:-1], answer_tokens[1:]))
        context_bigrams = set(zip(context_tokens[:-1], context_tokens[1:]))
        
        if answer_bigrams:
            bigram_overlap = len(answer_bigrams & context_bigrams) / len(answer_bigrams)
        else:
            bigram_overlap = 0.0
        
        # Spectral consistency: combination of frequency components
        spectral = 0.6 * vocab_consistency + 0.4 * bigram_overlap
        
        return min(1.0, spectral)
    
    def _compute_temporal_stability(self, answer: str, context: str) -> float:
        """
        Compute temporal stability score.
        
        HSAD observes that hallucinations often show instability in
        hidden states over time. We simulate by checking:
        - Consistency across answer segments
        - Topic drift within answer
        - Self-contradiction patterns
        """
        if not answer:
            return 0.5
        
        # Split answer into segments (simulates temporal windows)
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        if len(sentences) < 2:
            # Short answer - check against context
            return self.word_overlap(answer, context) if context else 0.5
        
        # Check consistency between segments
        stability_scores = []
        
        for i in range(len(sentences) - 1):
            s1_words = set(sentences[i].lower().split())
            s2_words = set(sentences[i + 1].lower().split())
            
            # Remove stopwords
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                         'of', 'to', 'in', 'for', 'on', 'with', 'and', 'or', 'but'}
            s1_content = s1_words - stopwords
            s2_content = s2_words - stopwords
            
            if s1_content and s2_content:
                # Semantic continuity between segments
                continuity = len(s1_content & s2_content) / max(len(s1_content), len(s2_content))
                stability_scores.append(continuity)
        
        if not stability_scores:
            return 0.5
        
        # Average stability (some variation is normal, but high variation = instability)
        avg_stability = np.mean(stability_scores)
        
        # Variance check (lower variance = more stable)
        variance = np.var(stability_scores) if len(stability_scores) > 1 else 0
        stability_bonus = max(0, 0.2 - variance)  # Bonus for low variance
        
        return min(1.0, avg_stability + stability_bonus + 0.3)
    
    def _compute_confidence_score(
        self, 
        answer: str, 
        context: str, 
        question: str
    ) -> float:
        """
        Compute confidence score based on spectral features.
        
        HSAD extracts the strongest non-DC frequency component
        as a confidence indicator. We simulate with:
        - Answer-context alignment
        - Answer-question relevance
        - Hedging language detection
        """
        if not answer:
            return 0.5
        
        # Context grounding (primary confidence signal)
        context_alignment = self.coverage_score(answer, context) if context else 0.3
        
        # Question relevance
        question_relevance = self.word_overlap(answer, question) if question else 0.3
        
        # Hedging language (indicates lower confidence)
        answer_lower = answer.lower()
        hedge_phrases = [
            'might', 'may', 'could', 'possibly', 'perhaps', 'probably',
            'i think', 'i believe', 'it seems', 'appears to', 'likely',
            'uncertain', 'not sure', 'unclear', 'approximately', 'roughly'
        ]
        
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in answer_lower)
        hedge_penalty = min(0.3, hedge_count * 0.1)
        
        # Assertive language (indicates higher confidence)
        assert_phrases = [
            'is', 'are', 'was', 'were', 'will be', 'definitely',
            'certainly', 'clearly', 'obviously', 'undoubtedly',
            'in fact', 'actually', 'indeed'
        ]
        
        assert_count = sum(1 for phrase in assert_phrases if phrase in answer_lower)
        assert_boost = min(0.2, assert_count * 0.05)
        
        # Combined confidence
        base_confidence = 0.5 * context_alignment + 0.3 * question_relevance + 0.2
        confidence = base_confidence - hedge_penalty + assert_boost
        
        return max(0.0, min(1.0, confidence))
    
    def _compute_hallucination_probability(
        self,
        spectral_consistency: float,
        temporal_stability: float,
        confidence: float
    ) -> float:
        """
        Compute hallucination probability.
        
        HSAD combines multiple spectral features to predict hallucination.
        Lower score = less likely to be hallucinated.
        """
        # Combine features (HSAD uses learned weights)
        # Higher consistency/stability/confidence = lower hallucination
        grounding_score = (
            0.4 * spectral_consistency +
            0.3 * temporal_stability +
            0.3 * confidence
        )
        
        # Hallucination probability is inverse of grounding
        hallucination_prob = 1.0 - grounding_score
        
        # HSAD applies sigmoid-like transformation
        # to get calibrated probabilities
        hallucination_prob = 1.0 / (1.0 + np.exp(-3 * (hallucination_prob - 0.5)))
        
        return hallucination_prob
