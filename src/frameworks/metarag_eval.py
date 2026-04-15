"""
MetaRAG Evaluator
=================

Metamorphic Testing for Hallucination Detection in RAG Systems.

Reference: Sok et al., "MetaRAG: Metamorphic Testing for Hallucination Detection 
in RAG Systems"
arXiv:2509.09360, Identity-Aware AI Workshop at ECAI 2025, October 2025

MetaRAG introduces metamorphic testing methodology for RAG hallucination detection,
decomposing answers into factoids and applying linguistic transformations
(synonym/antonym) to verify against retrieved context.

Key Features:
- Reference-free, black-box hallucination detection
- Factoid decomposition of generated answers
- Metamorphic relations: synonym and antonym transformations
- Context-based verification for each factoid
- Identity-aware deployment considerations

Metrics:
- factoid_consistency: Consistency of factoids with context
- synonym_stability: Stability under synonym transformation
- antonym_detection: Detection via antonym contradiction
- metarag_score: Final aggregated hallucination score
"""

from typing import Dict, List, Tuple
import numpy as np
import re
from . import HeuristicEvaluator


class MetaRAGEvaluator(HeuristicEvaluator):
    """
    MetaRAG: Metamorphic Testing for RAG Hallucination Detection (Oct 2025)
    
    Key innovation: Uses metamorphic testing principles from software testing
    for hallucination detection. Transforms inputs/outputs and checks if
    relationships hold (metamorphic relations).
    
    Metamorphic Relations (MRs):
    - Synonym MR: Replacing words with synonyms shouldn't change verification
    - Antonym MR: Replacing words with antonyms should flip verification
    
    The method is reference-free and works in black-box settings.
    
    From: Identity-Aware AI Workshop at ECAI 2025 (October 25, 2025, Bologna)
    """
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="MetaRAG",
            metrics=["factoid_consistency", "synonym_stability", 
                     "antonym_detection", "metarag_score"],
            noise_std=noise_std
        )
        self.version = "2025-10"  # October 2025
        
        # Simple synonym/antonym pairs for transformation testing
        self.synonyms = {
            'big': 'large', 'large': 'big',
            'small': 'little', 'little': 'small',
            'fast': 'quick', 'quick': 'fast',
            'good': 'excellent', 'excellent': 'good',
            'bad': 'poor', 'poor': 'bad',
            'important': 'significant', 'significant': 'important',
            'increase': 'rise', 'rise': 'increase',
            'decrease': 'decline', 'decline': 'decrease',
            'begin': 'start', 'start': 'begin',
            'end': 'finish', 'finish': 'end',
            'create': 'make', 'make': 'create',
            'use': 'utilize', 'utilize': 'use',
            'show': 'demonstrate', 'demonstrate': 'show',
        }
        
        self.antonyms = {
            'big': 'small', 'small': 'big',
            'large': 'little', 'little': 'large',
            'fast': 'slow', 'slow': 'fast',
            'good': 'bad', 'bad': 'good',
            'high': 'low', 'low': 'high',
            'increase': 'decrease', 'decrease': 'increase',
            'rise': 'fall', 'fall': 'rise',
            'true': 'false', 'false': 'true',
            'yes': 'no', 'no': 'yes',
            'positive': 'negative', 'negative': 'positive',
            'success': 'failure', 'failure': 'success',
            'before': 'after', 'after': 'before',
            'more': 'less', 'less': 'more',
            'always': 'never', 'never': 'always',
        }
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using MetaRAG methodology.
        
        MetaRAG process:
        1. Decompose answer into factoids
        2. Verify each factoid against context
        3. Apply synonym transformation (should preserve verification)
        4. Apply antonym transformation (should flip verification)
        5. Aggregate into final score
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        
        # Decompose answer into factoids
        factoids = self._decompose_into_factoids(answer)
        
        if not factoids:
            return {
                'factoid_consistency': self.add_noise(0.5),
                'synonym_stability': self.add_noise(0.5),
                'antonym_detection': self.add_noise(0.5),
                'metarag_score': self.add_noise(0.5)
            }
        
        # Verify factoids against context
        factoid_scores = self._verify_factoids(factoids, context)
        factoid_consistency = np.mean(factoid_scores) if factoid_scores else 0.5
        
        # Test synonym stability (MR1)
        synonym_stability = self._test_synonym_stability(factoids, context, factoid_scores)
        
        # Test antonym detection (MR2)
        antonym_detection = self._test_antonym_detection(factoids, context, factoid_scores)
        
        # Aggregate MetaRAG score
        # Paper uses max hallucination among factoids for final score
        worst_factoid = 1.0 - min(factoid_scores) if factoid_scores else 0.5
        metarag_score = 1.0 - worst_factoid  # Convert to "goodness" score
        
        # Adjust based on MR results
        mr_adjustment = 0.5 * synonym_stability + 0.5 * antonym_detection
        metarag_score = 0.6 * metarag_score + 0.4 * mr_adjustment
        
        return {
            'factoid_consistency': self.add_noise(factoid_consistency),
            'synonym_stability': self.add_noise(synonym_stability),
            'antonym_detection': self.add_noise(antonym_detection),
            'metarag_score': self.add_noise(metarag_score)
        }
    
    def _decompose_into_factoids(self, answer: str) -> List[str]:
        """
        Decompose answer into atomic factoids.
        
        MetaRAG decomposes responses into minimal factual units.
        Each factoid should express a single verifiable claim.
        """
        if not answer:
            return []
        
        factoids = []
        
        # First, split into sentences
        sentences = self.extract_claims(answer)
        
        for sentence in sentences:
            # Further decompose complex sentences
            # Split on conjunctions and relative clauses
            parts = re.split(r'\s*(?:,\s*(?:and|but|which|that|while|although)|\s+and\s+|\s+but\s+)\s*', sentence)
            
            for part in parts:
                part = part.strip()
                if len(part) > 10:  # Minimum factoid length
                    factoids.append(part)
        
        # If no decomposition happened, use sentences
        if not factoids:
            factoids = sentences
        
        return factoids
    
    def _verify_factoids(self, factoids: List[str], context: str) -> List[float]:
        """
        Verify each factoid against context.
        
        Returns verification score for each factoid (0 = hallucinated, 1 = verified).
        """
        if not context:
            return [0.5] * len(factoids)
        
        scores = []
        context_lower = context.lower()
        
        for factoid in factoids:
            # Word-level verification
            word_overlap = self.word_overlap(factoid, context)
            
            # Key term verification
            factoid_terms = self._extract_verifiable_terms(factoid)
            context_terms = self._extract_verifiable_terms(context)
            
            if factoid_terms:
                term_match = len(factoid_terms & context_terms) / len(factoid_terms)
            else:
                term_match = word_overlap
            
            # N-gram verification
            ngram_match = self.ngram_overlap(factoid, context, n=2)
            
            # Combined verification score
            verification = 0.35 * word_overlap + 0.4 * term_match + 0.25 * ngram_match
            
            scores.append(verification)
        
        return scores
    
    def _extract_verifiable_terms(self, text: str) -> set:
        """Extract terms that can be verified (entities, numbers, etc.)."""
        if not text:
            return set()
        
        terms = set()
        
        # Named entities (capitalized words)
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        terms.update(c.lower() for c in caps)
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        terms.update(numbers)
        
        # Dates
        dates = re.findall(r'\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', text)
        terms.update(d.lower() for d in dates)
        
        return terms
    
    def _test_synonym_stability(
        self, 
        factoids: List[str], 
        context: str, 
        original_scores: List[float]
    ) -> float:
        """
        Test Metamorphic Relation 1: Synonym Stability.
        
        Replacing words with synonyms should not change verification outcome.
        Instability indicates unreliable verification.
        """
        if not factoids or not context:
            return 0.5
        
        stability_scores = []
        
        for i, factoid in enumerate(factoids):
            # Apply synonym transformation
            transformed = self._apply_synonyms(factoid)
            
            if transformed == factoid:
                # No transformation applied
                stability_scores.append(1.0)
                continue
            
            # Re-verify transformed factoid
            transformed_score = self._verify_factoids([transformed], context)[0]
            original_score = original_scores[i]
            
            # Stability: scores should be similar after synonym transformation
            score_diff = abs(transformed_score - original_score)
            stability = 1.0 - score_diff
            
            stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _test_antonym_detection(
        self, 
        factoids: List[str], 
        context: str,
        original_scores: List[float]
    ) -> float:
        """
        Test Metamorphic Relation 2: Antonym Detection.
        
        Replacing words with antonyms should flip verification outcome.
        If original was verified, antonym version should be unverified.
        """
        if not factoids or not context:
            return 0.5
        
        detection_scores = []
        
        for i, factoid in enumerate(factoids):
            # Apply antonym transformation
            transformed = self._apply_antonyms(factoid)
            
            if transformed == factoid:
                # No transformation possible
                detection_scores.append(0.5)
                continue
            
            # Re-verify transformed (antonym) factoid
            transformed_score = self._verify_factoids([transformed], context)[0]
            original_score = original_scores[i]
            
            # Detection: antonym score should be inverse of original
            # If original was high (verified), antonym should be low (contradicted)
            expected_antonym = 1.0 - original_score
            detection_accuracy = 1.0 - abs(transformed_score - expected_antonym)
            
            detection_scores.append(detection_accuracy)
        
        return np.mean(detection_scores) if detection_scores else 0.5
    
    def _apply_synonyms(self, text: str) -> str:
        """Apply synonym transformation to text."""
        words = text.split()
        transformed = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:()[]{}"\'-')
            
            if word_lower in self.synonyms:
                # Replace with synonym, preserving case
                synonym = self.synonyms[word_lower]
                if word[0].isupper():
                    synonym = synonym.capitalize()
                transformed.append(word.replace(word_lower, synonym))
            else:
                transformed.append(word)
        
        return ' '.join(transformed)
    
    def _apply_antonyms(self, text: str) -> str:
        """Apply antonym transformation to text."""
        words = text.split()
        transformed = []
        any_changed = False
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:()[]{}"\'-')
            
            if word_lower in self.antonyms:
                # Replace with antonym, preserving case
                antonym = self.antonyms[word_lower]
                if word[0].isupper():
                    antonym = antonym.capitalize()
                transformed.append(word.replace(word_lower, antonym))
                any_changed = True
            else:
                transformed.append(word)
        
        result = ' '.join(transformed)
        return result if any_changed else text
