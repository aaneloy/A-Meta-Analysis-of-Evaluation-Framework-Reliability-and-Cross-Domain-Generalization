"""
LettuceDetect Evaluator
=======================

Token-Level Hallucination Detection Framework for RAG Applications.

Reference: Kovács & Recski, "LettuceDetect: A Hallucination Detection Framework 
for RAG Applications"
arXiv:2502.17125, February 2025

LettuceDetect is built on ModernBERT with extended context capabilities (up to 8k tokens)
and achieves 79.22% F1 on RAGTruth benchmark, outperforming Luna (65.4%) and GPT-4 (63.4%).

Key Features:
- Token-level classification for precise span detection
- 30-60 examples/second inference on single GPU
- 30x smaller than comparable LLM-based methods
- Extended context window (up to 8192 tokens via ModernBERT)
- Multilingual support via EuroBERT variants

Metrics:
- token_hallucination: Token-level hallucination score
- span_precision: Precision of hallucinated span detection
- example_f1: Example-level F1 score
- confidence: Detection confidence score
"""

from typing import Dict, List, Tuple
import numpy as np
import re
from . import HeuristicEvaluator


class LettuceDetectEvaluator(HeuristicEvaluator):
    """
    LettuceDetect: ModernBERT-based RAG Hallucination Detection (Feb 2025)
    
    Key innovation: Formulates hallucination detection as token-level 
    classification task, enabling precise identification of unsupported
    claims at the token/span level.
    
    Benchmark results on RAGTruth:
    - LettuceDetect-large: 79.22% F1 (SOTA for encoder-based)
    - Outperforms GPT-4 (63.4%), Luna (65.4%)
    - Competitive with fine-tuned LLaMA-3-8B (83.9%)
    
    GitHub: https://github.com/KRLabsOrg/LettuceDetect
    HuggingFace: KRLabsOrg/lettucedect-large-modernbert-en-v1
    """
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="LettuceDetect",
            metrics=["token_hallucination", "span_precision", "example_f1", "confidence"],
            noise_std=noise_std
        )
        self.version = "2025-02"  # February 2025
        
        # Domain calibration based on RAGTruth task types
        self.task_calibration = {
            'General Knowledge': 0.0,  # QA task baseline
            'Finance': 0.02,  # Data-to-text typically higher scores
            'Biomedicine': -0.02,  # Summarization can be harder
        }
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using LettuceDetect methodology.
        
        LettuceDetect processes context-question-answer triples and
        outputs token-level predictions which are aggregated into spans.
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        domain = sample.get('domain', 'General Knowledge')
        
        # Tokenize answer and check support
        tokens, support_scores = self._tokenize_and_score(answer, context)
        
        # Token-level hallucination rate
        token_hallucination = self._compute_token_hallucination(support_scores)
        
        # Span-level precision
        spans = self._identify_hallucinated_spans(tokens, support_scores)
        span_precision = self._compute_span_precision(spans, answer, context)
        
        # Example-level F1 (simplified to binary classification)
        example_f1 = self._compute_example_f1(token_hallucination, span_precision)
        
        # Confidence score
        confidence = self._compute_confidence(support_scores)
        
        # Apply domain calibration
        domain_adj = self.task_calibration.get(domain, 0.0)
        
        return {
            'token_hallucination': self.add_noise(np.clip(token_hallucination + domain_adj, 0, 1)),
            'span_precision': self.add_noise(span_precision),
            'example_f1': self.add_noise(example_f1),
            'confidence': self.add_noise(confidence)
        }
    
    def _tokenize_and_score(self, answer: str, context: str) -> Tuple[List[str], List[float]]:
        """
        Tokenize answer and compute support scores per token.
        
        LettuceDetect uses ModernBERT tokenization; we simulate with words.
        """
        if not answer:
            return [], []
        
        # Simplified tokenization (words)
        tokens = answer.split()
        context_lower = context.lower() if context else ""
        
        support_scores = []
        for token in tokens:
            token_lower = token.lower().strip('.,!?;:()[]{}"\'-')
            
            if len(token_lower) < 2:
                # Skip very short tokens
                support_scores.append(0.8)
                continue
            
            # Check if token appears in context
            if token_lower in context_lower:
                # Strong support
                support_scores.append(0.9)
            elif any(token_lower in word for word in context_lower.split()):
                # Partial support
                support_scores.append(0.7)
            else:
                # Check for synonyms/related terms (simplified)
                related = self._check_semantic_support(token_lower, context_lower)
                support_scores.append(related)
        
        return tokens, support_scores
    
    def _check_semantic_support(self, token: str, context: str) -> float:
        """
        Check semantic support for a token not directly in context.
        """
        # Common word mappings (simplified semantic similarity)
        if len(token) < 3:
            return 0.5
        
        # Check if token is a number - these need exact match
        if token.isdigit():
            return 0.2 if token not in context else 0.9
        
        # Check character overlap with context words
        context_words = context.split()
        max_overlap = 0
        for word in context_words:
            if len(word) >= 3:
                # Jaccard character overlap
                token_chars = set(token)
                word_chars = set(word.lower())
                overlap = len(token_chars & word_chars) / len(token_chars | word_chars)
                max_overlap = max(max_overlap, overlap)
        
        if max_overlap > 0.7:
            return 0.6  # Likely related
        elif max_overlap > 0.5:
            return 0.4  # Possibly related
        else:
            return 0.25  # Low support - potential hallucination
    
    def _compute_token_hallucination(self, support_scores: List[float]) -> float:
        """
        Compute overall token-level hallucination rate.
        
        Lower support scores indicate higher hallucination.
        """
        if not support_scores:
            return 0.5
        
        # Hallucination is inverse of average support
        avg_support = np.mean(support_scores)
        hallucination_rate = 1.0 - avg_support
        
        # Apply threshold similar to LettuceDetect
        # Tokens with support < 0.5 are considered hallucinated
        hallucinated_count = sum(1 for s in support_scores if s < 0.5)
        hallucination_ratio = hallucinated_count / len(support_scores)
        
        # Combine continuous and threshold-based
        return 0.6 * hallucination_rate + 0.4 * hallucination_ratio
    
    def _identify_hallucinated_spans(
        self, 
        tokens: List[str], 
        support_scores: List[float]
    ) -> List[Tuple[int, int, float]]:
        """
        Identify contiguous spans of hallucinated tokens.
        
        LettuceDetect outputs span-level predictions with start/end indices.
        Returns: List of (start_idx, end_idx, confidence) tuples
        """
        if not tokens or not support_scores:
            return []
        
        spans = []
        threshold = 0.5
        
        in_span = False
        span_start = 0
        span_scores = []
        
        for i, score in enumerate(support_scores):
            if score < threshold:
                if not in_span:
                    in_span = True
                    span_start = i
                    span_scores = [score]
                else:
                    span_scores.append(score)
            else:
                if in_span:
                    # End current span
                    avg_conf = 1.0 - np.mean(span_scores)  # Confidence in hallucination
                    spans.append((span_start, i - 1, avg_conf))
                    in_span = False
                    span_scores = []
        
        # Handle span at end
        if in_span:
            avg_conf = 1.0 - np.mean(span_scores)
            spans.append((span_start, len(tokens) - 1, avg_conf))
        
        return spans
    
    def _compute_span_precision(
        self, 
        spans: List[Tuple[int, int, float]], 
        answer: str, 
        context: str
    ) -> float:
        """
        Compute precision of hallucinated span detection.
        
        Higher precision means detected spans are truly unsupported.
        """
        if not spans:
            # No hallucinated spans detected
            # This is good if answer is well-grounded
            grounding = self.coverage_score(answer, context) if answer else 0.5
            return grounding
        
        # Check if detected spans are truly unsupported
        tokens = answer.split()
        verified_spans = 0
        
        for start, end, conf in spans:
            span_text = ' '.join(tokens[start:end+1]).lower()
            
            # Check if span content is in context
            context_lower = context.lower() if context else ""
            if span_text not in context_lower:
                # Span correctly identified as hallucination
                verified_spans += conf
            else:
                # False positive - span is actually supported
                verified_spans += (1 - conf) * 0.5
        
        precision = verified_spans / len(spans) if spans else 1.0
        return min(1.0, precision)
    
    def _compute_example_f1(
        self, 
        token_hallucination: float, 
        span_precision: float
    ) -> float:
        """
        Compute example-level F1 score.
        
        LettuceDetect reports 79.22% F1 on RAGTruth.
        We approximate based on token and span metrics.
        """
        # Example is hallucinated if token_hallucination > threshold
        is_hallucinated = token_hallucination > 0.3
        
        if is_hallucinated:
            # F1 based on span precision for hallucinated examples
            f1 = span_precision
        else:
            # F1 based on low hallucination rate for clean examples
            f1 = 1.0 - token_hallucination
        
        return f1
    
    def _compute_confidence(self, support_scores: List[float]) -> float:
        """
        Compute detection confidence.
        
        LettuceDetect outputs confidence scores per prediction.
        """
        if not support_scores:
            return 0.5
        
        # Confidence is higher when scores are more polarized (near 0 or 1)
        polarization = np.mean([abs(s - 0.5) * 2 for s in support_scores])
        
        # Also consider consistency
        std = np.std(support_scores) if len(support_scores) > 1 else 0
        consistency = 1.0 - min(std, 0.3) / 0.3  # Lower std = higher consistency
        
        confidence = 0.6 * polarization + 0.4 * consistency
        return min(1.0, confidence)
