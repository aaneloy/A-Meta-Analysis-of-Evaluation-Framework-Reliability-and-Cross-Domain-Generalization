"""
ReDeEP Evaluator
================

Detecting Hallucination in RAG via Mechanistic Interpretability.

Reference: Sun et al., "ReDeEP: Detecting Hallucination in Retrieval-Augmented 
Generation via Mechanistic Interpretability"
arXiv:2410.11414, ICLR 2025 Spotlight
GitHub: https://github.com/Jeryi-Sun/ReDEeP-ICLR

ReDeEP investigates internal mechanisms behind RAG hallucinations by decoupling
external context utilization from parametric knowledge contribution. It 
significantly improves hallucination detection accuracy on RAGTruth and Dolly.

Key Features:
- Mechanistic interpretability-based detection
- External Context Score via attention head analysis
- Parametric Knowledge Score via FFN contribution analysis
- AARF mitigation method (modulating FFN and Copying Head contributions)
- Causal analysis of hallucination generation

Metrics:
- external_context_score: How much model uses retrieved context
- parametric_knowledge_score: How much model uses internal knowledge
- redeep_score: Combined hallucination detection score
- copying_head_signal: Signal from attention copying behavior
"""

from typing import Dict, List, Tuple
import numpy as np
import re
from . import HeuristicEvaluator


class ReDeEPEvaluator(HeuristicEvaluator):
    """
    ReDeEP: Mechanistic Interpretability for RAG Hallucination Detection (ICLR 2025)
    
    Key finding: Hallucinations occur when:
    1. Knowledge FFNs overemphasize parametric knowledge in residual stream
    2. Copying Heads fail to retain/integrate external knowledge
    
    ReDeEP decouples these mechanisms for accurate detection.
    
    Published: ICLR 2025 Spotlight Paper (January 2025)
    
    The paper introduces:
    - External Context Score (ECS): attention-based context utilization
    - Parametric Knowledge Score (PKS): FFN-based knowledge contribution
    - Multivariate regression approach addressing confounding
    """
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="ReDeEP",
            metrics=["external_context_score", "parametric_knowledge_score", 
                     "redeep_score", "copying_head_signal"],
            noise_std=noise_std
        )
        self.version = "2025-01"  # ICLR 2025 (January)
        
        # Domain calibration based on benchmark results
        self.domain_weights = {
            'General Knowledge': {'ecs_weight': 0.6, 'pks_weight': 0.4},
            'Finance': {'ecs_weight': 0.55, 'pks_weight': 0.45},
            'Biomedicine': {'ecs_weight': 0.65, 'pks_weight': 0.35},
        }
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using ReDeEP methodology.
        
        ReDeEP uses mechanistic interpretability to:
        1. Measure External Context Score (attention-based)
        2. Measure Parametric Knowledge Score (FFN-based)
        3. Apply multivariate regression for detection
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        domain = sample.get('domain', 'General Knowledge')
        
        # External Context Score (simulating attention analysis)
        ecs = self._compute_external_context_score(answer, context)
        
        # Parametric Knowledge Score (simulating FFN analysis)
        pks = self._compute_parametric_knowledge_score(answer, context, question)
        
        # Copying Head Signal (simulating attention copying behavior)
        copying_signal = self._compute_copying_head_signal(answer, context)
        
        # Combined ReDeEP score using multivariate approach
        weights = self.domain_weights.get(domain, self.domain_weights['General Knowledge'])
        redeep_score = self._compute_redeep_score(ecs, pks, weights)
        
        return {
            'external_context_score': self.add_noise(ecs),
            'parametric_knowledge_score': self.add_noise(pks),
            'redeep_score': self.add_noise(redeep_score),
            'copying_head_signal': self.add_noise(copying_signal)
        }
    
    def _compute_external_context_score(self, answer: str, context: str) -> float:
        """
        Compute External Context Score (ECS).
        
        ReDeEP ECS formula (Eq. 2-3 in paper):
        1. Extract attended tokens from copying heads
        2. Compute cosine similarity between answer tokens and context
        3. Average across relevant attention heads
        
        We approximate by measuring how much answer content is traceable to context.
        """
        if not answer or not context:
            return 0.5
        
        # Extract claims/statements from answer
        claims = self.extract_claims(answer)
        
        if not claims:
            # Fallback to word-level analysis
            return self.coverage_score(answer, context)
        
        # Compute grounding for each claim
        grounding_scores = []
        for claim in claims:
            # Word overlap (simulating attention copying)
            word_score = self.word_overlap(claim, context)
            
            # Key term matching (simulating entity-level attention)
            claim_terms = self._extract_key_terms(claim)
            context_terms = self._extract_key_terms(context)
            
            if claim_terms:
                term_overlap = len(claim_terms & context_terms) / len(claim_terms)
            else:
                term_overlap = word_score
            
            # N-gram matching (phrase-level copying)
            ngram_score = self.ngram_overlap(claim, context, n=3)
            
            # Combined score (weighted by ReDeEP's emphasis on copying)
            claim_grounding = 0.4 * word_score + 0.35 * term_overlap + 0.25 * ngram_score
            grounding_scores.append(claim_grounding)
        
        # ECS is average grounding across claims
        ecs = np.mean(grounding_scores) if grounding_scores else 0.5
        
        return min(1.0, ecs)
    
    def _compute_parametric_knowledge_score(
        self, 
        answer: str, 
        context: str, 
        question: str
    ) -> float:
        """
        Compute Parametric Knowledge Score (PKS).
        
        ReDeEP PKS (Eq. 4-5 in paper):
        1. Analyze FFN contributions to residual stream
        2. Compare output distribution with/without FFN intervention
        3. Higher score = more reliance on parametric knowledge
        
        We approximate by detecting content that cannot be traced to context.
        """
        if not answer:
            return 0.5
        
        answer_lower = answer.lower()
        context_lower = context.lower() if context else ""
        question_lower = question.lower() if question else ""
        
        # Extract statements
        claims = self.extract_claims(answer)
        
        if not claims:
            return 0.3
        
        # Measure content not grounded in context
        ungrounded_content = 0
        total_content = 0
        
        for claim in claims:
            claim_words = set(claim.lower().split())
            context_words = set(context_lower.split())
            question_words = set(question_lower.split())
            
            # Remove common words
            stopwords = self._get_stopwords()
            claim_content = claim_words - stopwords
            
            if not claim_content:
                continue
            
            total_content += len(claim_content)
            
            # Words not in context or question = likely from parametric knowledge
            grounded = claim_content & (context_words | question_words)
            ungrounded = claim_content - grounded
            
            ungrounded_content += len(ungrounded)
        
        if total_content == 0:
            return 0.3
        
        # PKS: ratio of ungrounded content (parametric knowledge use)
        pks = ungrounded_content / total_content
        
        # Also detect hallucination patterns
        hallucination_patterns = self._detect_hallucination_patterns(answer, context)
        
        # Combine
        pks = 0.7 * pks + 0.3 * hallucination_patterns
        
        return min(1.0, pks)
    
    def _detect_hallucination_patterns(self, answer: str, context: str) -> float:
        """
        Detect patterns indicative of parametric knowledge hallucination.
        """
        answer_lower = answer.lower()
        context_lower = context.lower() if context else ""
        
        pattern_score = 0
        
        # Numbers not in context
        answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', answer_lower))
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', context_lower))
        
        if answer_numbers:
            ungrounded_numbers = answer_numbers - context_numbers
            pattern_score += 0.3 * len(ungrounded_numbers) / len(answer_numbers)
        
        # Proper nouns not in context
        answer_caps = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))
        context_caps = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context))
        
        if answer_caps:
            ungrounded_caps = {c.lower() for c in answer_caps} - {c.lower() for c in context_caps}
            pattern_score += 0.3 * len(ungrounded_caps) / len(answer_caps)
        
        # Confident assertions without evidence
        assertion_patterns = [
            r'\b(definitely|certainly|absolutely|clearly|obviously)\b',
            r'\b(always|never|all|none|every)\b',
            r'\b(must be|has to be|cannot be)\b',
        ]
        
        for pattern in assertion_patterns:
            if re.search(pattern, answer_lower) and not re.search(pattern, context_lower):
                pattern_score += 0.1
        
        return min(1.0, pattern_score)
    
    def _compute_copying_head_signal(self, answer: str, context: str) -> float:
        """
        Compute Copying Head signal.
        
        ReDeEP finds that Copying Heads (attention heads that copy from context)
        are crucial for grounding. Low signal = copying failure = hallucination.
        
        We approximate by measuring direct textual copying.
        """
        if not answer or not context:
            return 0.5
        
        # Look for directly copied phrases
        answer_words = answer.split()
        context_lower = context.lower()
        
        copied_sequences = 0
        total_sequences = 0
        
        # Check for copied 3-grams
        for i in range(len(answer_words) - 2):
            sequence = ' '.join(answer_words[i:i+3]).lower()
            total_sequences += 1
            if sequence in context_lower:
                copied_sequences += 1
        
        # Check for copied 4-grams (stronger signal)
        for i in range(len(answer_words) - 3):
            sequence = ' '.join(answer_words[i:i+4]).lower()
            if sequence in context_lower:
                copied_sequences += 0.5  # Extra weight for longer matches
        
        if total_sequences == 0:
            return 0.5
        
        # Copying head signal: proportion of copied content
        copying_signal = copied_sequences / total_sequences
        
        return min(1.0, copying_signal)
    
    def _compute_redeep_score(
        self, 
        ecs: float, 
        pks: float, 
        weights: Dict[str, float]
    ) -> float:
        """
        Compute final ReDeEP hallucination score.
        
        ReDeEP uses multivariate regression:
        - Higher ECS (context use) = lower hallucination
        - Higher PKS (parametric use) = higher hallucination
        
        Returns score where higher = better (less hallucination)
        """
        ecs_weight = weights.get('ecs_weight', 0.6)
        pks_weight = weights.get('pks_weight', 0.4)
        
        # ECS contributes positively (more context = less hallucination)
        # PKS contributes negatively (more parametric = more hallucination)
        redeep_score = ecs_weight * ecs + pks_weight * (1 - pks)
        
        return redeep_score
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms for analysis."""
        if not text:
            return set()
        
        terms = set()
        
        # Proper nouns
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        terms.update(c.lower() for c in caps if len(c) > 2)
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        terms.update(numbers)
        
        # Technical terms
        tech = re.findall(r'\b\w+(?:tion|ment|ity|ness|ical|ious|ous)\b', text.lower())
        terms.update(t for t in tech if len(t) > 5)
        
        return terms
    
    def _get_stopwords(self) -> set:
        """Return stopwords set."""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'and', 'but',
            'or', 'if', 'then', 'so', 'that', 'this', 'these', 'those',
            'it', 'its', 'they', 'their', 'them', 'he', 'she', 'his',
            'her', 'we', 'our', 'you', 'your', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'
        }
