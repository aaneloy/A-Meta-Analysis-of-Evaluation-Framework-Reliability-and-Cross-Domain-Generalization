"""
Base Evaluator Class
====================

Abstract base class for all evaluation frameworks.

Framework Timeline (20 total):
- 2020-2023: BERTScore, UniEval, G-Eval, QAFactEval (traditional)
- 2023-2024: RAGAS, ARES, DeepEval, RAGChecker, TRACe (RAG-specific)
- 2025 (Early): ReDeEP (ICLR 2025), LettuceDetect (Feb 2025), FaithJudge (EMNLP 2025), LRP4RAG (Jun 2025)
- 2025 (Mid-Late): LUMINA (NeurIPS 2025), HALT-RAG (Sep 2025), MetaRAG (ECAI 2025), KG-RAG (Oct 2025)
- 2025 (Ongoing): GaRAGe (ACL 2025), HSAD (Sep 2025)
- 2026: SIRG (Jan 2026)

Frameworks with References:
- ReDeEP: Mechanistic interpretability for RAG hallucination (arXiv:2410.11414, ICLR 2025 Spotlight)
- LettuceDetect: Token-level hallucination detection using ModernBERT (arXiv:2502.17125)
- FaithJudge: LLM-as-judge faithfulness evaluation (arXiv:2505.04847, EMNLP 2025)
- LRP4RAG: Layer-wise Relevance Propagation for RAG (arXiv:2408.15533v3, Jun 2025)
- LUMINA: Context-knowledge signal-based detection (arXiv:2509.21875, NeurIPS 2025 Workshop)
- HALT-RAG: Calibrated NLI ensemble with abstention (arXiv:2509.07475, Sep 2025)
- MetaRAG: Metamorphic testing for hallucination detection (arXiv:2509.09360, ECAI 2025)
- KG-RAG: Knowledge graph-based RAG evaluation (arXiv:2510.02549)
- GaRAGe: Grounding annotations for RAG evaluation (arXiv:2412.13834, ACL 2025)
- HSAD: Hidden state analysis detection (arXiv:2509.13154)
- SIRG: Semantic-level Internal Reasoning Graph (arXiv:2601.03052, Jan 2026)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class BaseEvaluator(ABC):
    """
    Abstract base class for RAG evaluation frameworks.
    
    All framework implementations should inherit from this class
    and implement the evaluate() method.
    """
    
    def __init__(self, name: str, metrics: List[str]):
        """
        Initialize the evaluator.
        
        Args:
            name: Name of the framework
            metrics: List of metric names this framework produces
        """
        self.name = name
        self.metrics = metrics
        self.noise_std = 0.0  # Standard deviation for noise (simulation)
    
    @abstractmethod
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a single sample.
        
        Args:
            sample: Dictionary containing:
                - question: The input question
                - answer: The generated answer
                - context: The retrieved context
                - ground_truth: Optional ground truth
                
        Returns:
            Dictionary mapping metric names to scores (0-1 range)
        """
        pass
    
    def evaluate_batch(self, samples: List[Dict]) -> List[Dict[str, float]]:
        """
        Evaluate a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            List of result dictionaries
        """
        results = []
        for sample in samples:
            result = self.evaluate(sample)
            results.append(result)
        return results
    
    def add_noise(self, score: float, std: Optional[float] = None) -> float:
        """
        Add Gaussian noise to a score (for simulation).
        
        Args:
            score: Original score
            std: Standard deviation (uses self.noise_std if not provided)
            
        Returns:
            Score with noise, clipped to [0, 1]
        """
        if std is None:
            std = self.noise_std
        noisy_score = score + np.random.normal(0, std)
        return max(0.0, min(1.0, noisy_score))
    
    def get_metric_names(self) -> List[str]:
        """
        Get full metric names with framework prefix.
        
        Returns:
            List of metric names like "RAGAS_faithfulness"
        """
        return [f"{self.name}_{m}" for m in self.metrics]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', metrics={self.metrics})"


class HeuristicEvaluator(BaseEvaluator):
    """
    Base class for heuristic-based evaluation.
    
    Provides common text analysis methods used across frameworks
    when LLM-based evaluation is not available.
    """
    
    def __init__(self, name: str, metrics: List[str], noise_std: float = 0.05):
        super().__init__(name, metrics)
        self.noise_std = noise_std
    
    @staticmethod
    def word_overlap(text1: str, text2: str) -> float:
        """
        Compute word overlap ratio between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap ratio (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        return len(intersection) / max(len(words1), len(words2))
    
    @staticmethod
    def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
        """
        Compute n-gram overlap ratio.
        
        Args:
            text1: First text
            text2: Second text
            n: N-gram size
            
        Returns:
            Overlap ratio (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        return len(intersection) / max(len(ngrams1), len(ngrams2))
    
    @staticmethod
    def sentence_count(text: str) -> int:
        """Count sentences in text."""
        if not text:
            return 0
        # Simple sentence splitting
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        return len(sentences)
    
    @staticmethod
    def extract_claims(text: str) -> List[str]:
        """
        Extract claims (sentences) from text.
        
        Args:
            text: Input text
            
        Returns:
            List of claim strings
        """
        if not text:
            return []
        
        # Split by sentence-ending punctuation
        import re
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return claims
    
    @staticmethod
    def coverage_score(source: str, target: str) -> float:
        """
        Compute how much of source is covered by target.
        
        Args:
            source: Source text (e.g., answer)
            target: Target text (e.g., context)
            
        Returns:
            Coverage ratio (0-1)
        """
        if not source or not target:
            return 0.0
        
        source_words = set(str(source).lower().split())
        target_words = set(str(target).lower().split())
        
        if not source_words:
            return 0.0
        
        covered = source_words & target_words
        return len(covered) / len(source_words)
