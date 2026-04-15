"""
LRP4RAG Evaluator
=================

Based on: "LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation 
via Layer-wise Relevance Propagation"

Authors: Haichuan Hu, Yuhan Sun, Quanjun Zhang
Paper: arXiv:2408.15533
Published: August 28, 2024 (v1), June 27, 2025 (v3)
GitHub: https://github.com/Tomsawyerhu/LRP4RAG

Key Innovation:
- First application of Layer-wise Relevance Propagation (LRP) to RAG hallucination detection
- Computes relevance between RAG context and model output
- Extracts and resamples relevance matrix for classification
- Two variants: LRP4RAGClassifier (feature vectors) and LRP4RAGLLM (context pruning)

Performance (from paper):
- RAGTruth Llama-7B: Accuracy 77.2%, improving SEP baseline by 4.1%
- Dolly-15k: Accuracy 76.2%, improving SEP baseline by 5.1%
- F1 score: 70.64% on RAGTruth

Core Method:
1. Perform LRP backwards through each layer of RAG generator
2. Compute relevance matrix between context tokens and output tokens
3. Extract key aspects: R*, R_prompt, R_response
4. Input processed relevance data into classifiers (SVM, Logistic Regression)
5. Optionally prune context using relevance for consistency checking
"""

import re
import math
from typing import Dict, Any, List, Tuple
from . import HeuristicEvaluator


class LRP4RAGEvaluator(HeuristicEvaluator):
    """
    LRP4RAG: Hallucination detection via Layer-wise Relevance Propagation.
    
    Simulates the LRP-based approach that analyzes relevance between
    context and generated output for hallucination detection.
    """
    
    name = "LRP4RAG"
    version = "2025-06"  # v3 release date
    
    metrics = [
        "context_relevance",        # R_context: relevance from context
        "response_relevance",       # R_response: relevance to response
        "relevance_consistency",    # Consistency of relevance distribution
        "lrp_classifier_score"      # Final classifier output
    ]
    
    def __init__(self, noise_std: float = 0.04, resampling_length: int = 100):
        super().__init__(
            name="LRP4RAG",
            metrics=["context_relevance", "response_relevance", 
                     "relevance_consistency", "lrp_classifier_score"],
            noise_std=noise_std
        )
        self.resampling_length = resampling_length  # L_new in paper
        # Optimal threshold from paper experiments
        self.detection_threshold = 0.5
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _compute_token_overlap_matrix(self, context_tokens: List[str], 
                                       answer_tokens: List[str]) -> List[List[float]]:
        """
        Compute a simulated relevance matrix between context and answer tokens.
        
        In actual LRP4RAG, this is computed by backpropagating relevance
        through transformer layers from output logits to input tokens.
        """
        if not context_tokens or not answer_tokens:
            return [[0.0]]
        
        matrix = []
        context_set = set(context_tokens)
        
        for ans_token in answer_tokens:
            row = []
            for ctx_token in context_tokens:
                # Exact match
                if ans_token == ctx_token:
                    relevance = 0.8
                # Substring match
                elif ans_token in ctx_token or ctx_token in ans_token:
                    relevance = 0.5
                # Same first 3 characters (morphological similarity)
                elif len(ans_token) >= 3 and len(ctx_token) >= 3 and ans_token[:3] == ctx_token[:3]:
                    relevance = 0.3
                else:
                    relevance = 0.05
                row.append(relevance)
            matrix.append(row)
        
        return matrix
    
    def _extract_relevance_features(self, matrix: List[List[float]]) -> Dict[str, float]:
        """
        Extract key features from the relevance matrix.
        
        Following the paper's approach:
        - R*: aggregated relevance (mean resampling)
        - R_prompt: relevance from prompt/context region
        - R_response: relevance distribution in response
        """
        if not matrix or not matrix[0]:
            return {
                'r_star': 0.0,
                'r_prompt': 0.0,
                'r_response': 0.0,
                'r_variance': 0.0
            }
        
        # Flatten and compute statistics
        all_values = [v for row in matrix for v in row]
        
        # R*: mean of all relevance values
        r_star = sum(all_values) / len(all_values) if all_values else 0
        
        # R_prompt: sum of column-wise max (context token importance)
        col_maxes = []
        num_cols = len(matrix[0])
        for j in range(num_cols):
            col_max = max(matrix[i][j] for i in range(len(matrix)))
            col_maxes.append(col_max)
        r_prompt = sum(col_maxes) / len(col_maxes) if col_maxes else 0
        
        # R_response: row-wise statistics (answer token grounding)
        row_sums = [sum(row) / len(row) if row else 0 for row in matrix]
        r_response = sum(row_sums) / len(row_sums) if row_sums else 0
        
        # Variance of relevance (lower = more consistent = less hallucination)
        mean_val = r_star
        variance = sum((v - mean_val) ** 2 for v in all_values) / len(all_values) if all_values else 0
        
        return {
            'r_star': r_star,
            'r_prompt': r_prompt,
            'r_response': r_response,
            'r_variance': variance
        }
    
    def _resample_relevance(self, features: Dict[str, float], 
                            target_length: int) -> List[float]:
        """
        Resample relevance features to fixed length for classifier input.
        
        Paper uses L_new = 100 and computes mean of all samples.
        """
        # Create feature vector (simplified version)
        base_features = [
            features['r_star'],
            features['r_prompt'],
            features['r_response'],
            1.0 - features['r_variance']  # Invert variance (higher = better)
        ]
        
        # Pad or truncate to target length
        if len(base_features) < target_length:
            # Interpolate to target length
            resampled = []
            for i in range(target_length):
                idx = i * len(base_features) // target_length
                resampled.append(base_features[idx])
            return resampled
        else:
            return base_features[:target_length]
    
    def _simulate_classifier(self, features: Dict[str, float]) -> float:
        """
        Simulate the SVM classifier decision.
        
        Paper found that mean_resampling + SVM achieved best results.
        """
        # Weighted combination based on paper's findings
        # Higher relevance scores indicate less hallucination
        score = (
            features['r_star'] * 0.3 +
            features['r_prompt'] * 0.25 +
            features['r_response'] * 0.35 +
            (1.0 - min(1.0, features['r_variance'] * 5)) * 0.1  # Penalize high variance
        )
        
        # Apply sigmoid-like transformation for probability
        return 1.0 / (1.0 + math.exp(-5 * (score - 0.4)))
    
    def _compute_consistency_score(self, context: str, answer: str) -> float:
        """
        Compute consistency between pruned context and answer.
        
        This simulates LRP4RAGLLM's approach of pruning context using
        relevance and checking consistency.
        """
        context_tokens = set(self._tokenize(context))
        answer_tokens = set(self._tokenize(answer))
        
        if not answer_tokens:
            return 0.5
        
        # Overlap represents consistency
        overlap = len(context_tokens & answer_tokens)
        
        # Penalize tokens in answer not found in context
        novel_ratio = len(answer_tokens - context_tokens) / len(answer_tokens)
        
        consistency = (overlap / len(answer_tokens)) * (1 - novel_ratio * 0.5)
        
        return min(1.0, consistency)
    
    def evaluate(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate using LRP4RAG's relevance propagation approach.
        
        Process:
        1. Tokenize context and answer
        2. Compute simulated relevance matrix
        3. Extract relevance features (R*, R_prompt, R_response)
        4. Apply classifier for hallucination detection
        """
        context = sample.get('context', '')
        answer = sample.get('answer', '')
        question = sample.get('question', '')
        
        # Tokenize
        context_tokens = self._tokenize(context)
        answer_tokens = self._tokenize(answer)
        
        # Limit tokens for efficiency (paper uses first N tokens)
        max_context_tokens = 512
        max_answer_tokens = 128
        context_tokens = context_tokens[:max_context_tokens]
        answer_tokens = answer_tokens[:max_answer_tokens]
        
        # Compute relevance matrix
        relevance_matrix = self._compute_token_overlap_matrix(context_tokens, answer_tokens)
        
        # Extract features
        features = self._extract_relevance_features(relevance_matrix)
        
        # Compute classifier score
        classifier_score = self._simulate_classifier(features)
        
        # Compute consistency score (LRP4RAGLLM approach)
        consistency = self._compute_consistency_score(context, answer)
        
        # Final scores
        scores = {
            'context_relevance': self.add_noise(max(0, min(1, features['r_prompt']))),
            'response_relevance': self.add_noise(max(0, min(1, features['r_response']))),
            'relevance_consistency': self.add_noise(max(0, min(1, consistency))),
            'lrp_classifier_score': self.add_noise(max(0, min(1, classifier_score)))
        }
        
        return scores
