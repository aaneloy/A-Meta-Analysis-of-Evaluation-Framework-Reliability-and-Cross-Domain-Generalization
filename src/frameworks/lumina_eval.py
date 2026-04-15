"""
LUMINA Evaluator
================

Detecting Hallucinations in RAG Systems with Context-Knowledge Signals.

Reference: Yeh et al., "LUMINA: Detecting Hallucinations in RAG System with 
Context-Knowledge Signals"
arXiv:2509.21875, September 2025
NeurIPS 2025 Workshop: ResponsibleFM

LUMINA detects hallucinations by measuring external context utilization and 
internal knowledge utilization signals, achieving +13% AUROC improvement over
prior methods on HalluRAG dataset.

Key Features:
- Context-knowledge signal decomposition
- Maximum mean discrepancy for external context utilization
- Information processing rate for internal knowledge utilization
- Robust under relaxed assumptions (no perfect context, proxy LLMs)
- Layer-agnostic measurement (no hyperparameter tuning)

Metrics:
- context_utilization: External context utilization score
- knowledge_utilization: Internal knowledge utilization score  
- lumina_score: Combined hallucination detection score
- signal_confidence: Confidence in signal measurements
"""

from typing import Dict, List, Tuple
import numpy as np
import re
from . import HeuristicEvaluator


class LUMINAEvaluator(HeuristicEvaluator):
    """
    LUMINA: Context-Knowledge Signal-based Hallucination Detection (Sep 2025)
    
    Key insight: Hallucinations occur when LLMs over-rely on internal parametric
    knowledge while under-utilizing retrieved external context.
    
    LUMINA quantifies:
    - External context utilization via distributional distance (MMD)
    - Internal knowledge utilization via information processing rate
    
    Benchmark results:
    - >0.9 AUROC on HalluRAG across models
    - +13% AUROC improvement over ReDeEP
    - Robust to 30% noise in retrieved documents
    
    From NeurIPS 2025 Workshop on Responsible Foundation Models
    """
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="LUMINA",
            metrics=["context_utilization", "knowledge_utilization", "lumina_score", "signal_confidence"],
            noise_std=noise_std
        )
        self.version = "2025-09"  # September 2025
        self.lambda_weight = 0.5  # Balance between context and knowledge signals
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using LUMINA methodology.
        
        LUMINA measures:
        1. External context utilization (Eq. 1 in paper)
        2. Internal knowledge utilization via information processing rate
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        domain = sample.get('domain', 'General Knowledge')
        
        # Compute external context utilization (simulated MMD)
        context_util = self._compute_context_utilization(answer, context)
        
        # Compute internal knowledge utilization
        knowledge_util = self._compute_knowledge_utilization(answer, context, question)
        
        # Combined LUMINA score
        # High context_util + low knowledge_util = grounded (low hallucination)
        # Low context_util + high knowledge_util = hallucination risk
        lumina_score = self._compute_lumina_score(context_util, knowledge_util)
        
        # Signal confidence
        signal_confidence = self._compute_signal_confidence(context_util, knowledge_util)
        
        return {
            'context_utilization': self.add_noise(context_util),
            'knowledge_utilization': self.add_noise(knowledge_util),
            'lumina_score': self.add_noise(lumina_score),
            'signal_confidence': self.add_noise(signal_confidence)
        }
    
    def _compute_context_utilization(self, answer: str, context: str) -> float:
        """
        Compute external context utilization score.
        
        LUMINA uses Maximum Mean Discrepancy (MMD) between:
        - Next token distribution with full context
        - Next token distribution with random/no context
        
        We approximate this by measuring how much the answer relies on context.
        """
        if not answer or not context:
            return 0.5
        
        # Extract key content from answer
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove stopwords
        stopwords = self._get_stopwords()
        answer_content = answer_words - stopwords
        context_content = context_words - stopwords
        
        if not answer_content:
            return 0.5
        
        # Direct overlap (strongest signal)
        direct_overlap = len(answer_content & context_content) / len(answer_content)
        
        # N-gram overlap for phrases
        ngram_overlap = self.ngram_overlap(answer, context, n=3)
        
        # Key entity overlap
        answer_entities = self._extract_key_terms(answer)
        context_entities = self._extract_key_terms(context)
        entity_overlap = (
            len(answer_entities & context_entities) / len(answer_entities)
            if answer_entities else direct_overlap
        )
        
        # LUMINA emphasizes distributional similarity
        # Higher overlap = higher context utilization
        context_util = 0.3 * direct_overlap + 0.3 * ngram_overlap + 0.4 * entity_overlap
        
        return min(1.0, context_util)
    
    def _compute_knowledge_utilization(
        self, 
        answer: str, 
        context: str, 
        question: str
    ) -> float:
        """
        Compute internal knowledge utilization score.
        
        LUMINA introduces "information processing rate" - tracking how predicted
        tokens evolve across transformer layers to determine internal knowledge use.
        
        We approximate by detecting:
        - Information in answer not traceable to context
        - General knowledge statements
        - Elaborations beyond retrieved content
        """
        if not answer:
            return 0.5
        
        answer_lower = answer.lower()
        context_lower = context.lower() if context else ""
        
        # Find answer content not in context
        answer_sentences = self.extract_claims(answer)
        
        if not answer_sentences:
            return 0.3
        
        ungrounded_score = 0
        for sentence in answer_sentences:
            # Check if sentence is grounded in context
            grounding = self.coverage_score(sentence, context)
            
            if grounding < 0.3:
                # Low grounding = likely using internal knowledge
                ungrounded_score += 1
            elif grounding < 0.5:
                ungrounded_score += 0.5
        
        knowledge_util = ungrounded_score / len(answer_sentences)
        
        # Also check for common patterns of parametric knowledge use
        knowledge_patterns = [
            r'\b(generally|typically|usually|often|commonly)\b',
            r'\b(it is known|as we know|everyone knows)\b',
            r'\b(in general|overall|broadly speaking)\b',
            r'\b(according to|based on|research shows)\b',  # without citation in context
        ]
        
        pattern_count = 0
        for pattern in knowledge_patterns:
            if re.search(pattern, answer_lower):
                # Check if this pattern is also in context
                if not re.search(pattern, context_lower):
                    pattern_count += 1
        
        pattern_score = min(pattern_count * 0.1, 0.3)
        
        # Combine scores
        knowledge_util = min(1.0, knowledge_util * 0.7 + pattern_score + 0.1)
        
        return knowledge_util
    
    def _compute_lumina_score(
        self, 
        context_util: float, 
        knowledge_util: float
    ) -> float:
        """
        Compute combined LUMINA hallucination score.
        
        LUMINA paper uses: score = λ * context_score + (1-λ) * knowledge_score
        With λ = 0.5 as default.
        
        Higher score = less hallucination (more grounded)
        """
        # Context utilization is positive (more = better)
        # Knowledge utilization is negative (more = worse, hallucination risk)
        
        # Transform to: high = grounded, low = hallucination
        grounding_score = (
            self.lambda_weight * context_util + 
            (1 - self.lambda_weight) * (1 - knowledge_util)
        )
        
        # LUMINA normalizes to [0, 1] where higher = less hallucination
        return grounding_score
    
    def _compute_signal_confidence(
        self, 
        context_util: float, 
        knowledge_util: float
    ) -> float:
        """
        Compute confidence in signal measurements.
        
        LUMINA paper shows stable performance across different conditions.
        Higher confidence when signals are clearly differentiated.
        """
        # Confidence is higher when signals are polarized
        # (either clearly grounded or clearly hallucinating)
        
        # Distance from uncertain middle ground
        context_certainty = abs(context_util - 0.5) * 2
        knowledge_certainty = abs(knowledge_util - 0.5) * 2
        
        # Also high confidence when signals agree
        # (high context + low knowledge = grounded, or vice versa)
        signal_agreement = 1.0 - abs(context_util - (1 - knowledge_util))
        
        confidence = 0.4 * context_certainty + 0.4 * knowledge_certainty + 0.2 * signal_agreement
        
        return min(1.0, confidence)
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms (entities, numbers, technical terms)."""
        if not text:
            return set()
        
        terms = set()
        
        # Capitalized words (potential entities)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        terms.update(c.lower() for c in caps)
        
        # Numbers with units
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|k|m|b|million|billion|thousand)?\b', text.lower())
        terms.update(numbers)
        
        # Technical terms (words with specific patterns)
        tech_terms = re.findall(r'\b[a-z]+(?:tion|ment|ity|ness|ing)\b', text.lower())
        terms.update(t for t in tech_terms if len(t) > 5)
        
        return terms
    
    def _get_stopwords(self) -> set:
        """Return common stopwords."""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'not', 'only', 'same', 'so', 'than', 'too', 'very', 'and',
            'but', 'if', 'or', 'because', 'while', 'although', 'however'
        }
