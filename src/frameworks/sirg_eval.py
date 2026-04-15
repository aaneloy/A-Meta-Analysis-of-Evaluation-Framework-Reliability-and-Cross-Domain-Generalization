"""
SIRG (Semantic-level Internal Reasoning Graph) Evaluator
=========================================================

Based on: "Detecting Hallucinations in Retrieval-Augmented Generation via 
Semantic-level Internal Reasoning Graph"

Authors: Jianpeng Hu et al.
Paper: arXiv:2601.03052
Published: January 6, 2026

Key Innovation:
- Extends Layer-wise Relevance Propagation (LRP) from token level to SEMANTIC level
- Constructs internal reasoning graph based on attribution vectors
- Distinguishes between "linking tokens" (non-substantive connectors) and 
  "substantive tokens" (semantic content utilizing context)
- Uses small pre-trained language model for training and detection
- Dynamic threshold adjustment for pass rate control

Performance (from paper):
- RAGTruth Llama-7B: Precision 85.71%, Recall 75.00%, F1 80.00%
- RAGTruth Llama-13B: Precision 88.46%, Recall 88.46%, F1 88.46%
- Dolly-15k Qwen2.5-7B: Precision 90.23%, Recall 88.24%, F1 89.17%
- Outperforms LRP4RAG (the previous state-of-the-art)

Core Method:
1. Classify tokens as "linking" (connective) or "substantive" (content-bearing)
2. Construct semantic-level reasoning graph from attribution vectors
3. Detect when LLM mistakenly treats substantive tokens as linking tokens
4. Use graph neural network features for hallucination classification
5. Apply dynamic thresholding based on semantic drift detection
"""

import re
import math
from typing import Dict, Any, List, Set, Tuple
from . import HeuristicEvaluator


class SIRGEvaluator(HeuristicEvaluator):
    """
    SIRG: Semantic-level Internal Reasoning Graph for hallucination detection.
    
    Simulates the semantic-level LRP approach that constructs reasoning graphs
    from attribution vectors to detect faithfulness hallucinations.
    """
    
    name = "SIRG"
    version = "2026-01"
    
    metrics = [
        "semantic_grounding",       # Substantive token grounding in context
        "linking_ratio",            # Ratio of linking vs substantive tokens
        "attribution_consistency",  # Consistency of attribution vectors
        "sirg_score"               # Final SIRG classification score
    ]
    
    # Common linking words (function words, connectives)
    LINKING_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'also', 'just', 'than', 'that', 'which', 'who', 'whom',
        'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'there',
        'here', 'where', 'when', 'what', 'how', 'why', 'if', 'then', 'else',
        'such', 'each', 'every', 'any', 'all', 'some', 'no', 'none', 'more',
        'most', 'less', 'least', 'very', 'much', 'many', 'few', 'little'
    }
    
    def __init__(self, noise_std: float = 0.04):
        super().__init__(
            name="SIRG",
            metrics=["semantic_grounding", "linking_ratio", 
                     "attribution_consistency", "sirg_score"],
            noise_std=noise_std
        )
        # Dynamic threshold parameters
        self.base_threshold = 0.5
        self.semantic_drift_penalty = 0.1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _classify_tokens(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Classify tokens into linking (function) and substantive (content) tokens.
        
        From paper: "linking tokens as non-substantive text in LLM-generated
        responses that serve to connect contextual information, whereas 
        substantive tokens refer to text that utilizes the contextual 
        information provided by users and reflects the semantic content"
        """
        linking = []
        substantive = []
        
        for token in tokens:
            if token in self.LINKING_WORDS or len(token) <= 2:
                linking.append(token)
            else:
                substantive.append(token)
        
        return linking, substantive
    
    def _compute_attribution_vector(self, token: str, context_tokens: Set[str]) -> Dict[str, float]:
        """
        Compute simulated attribution vector for a token.
        
        In actual SIRG, this extends LRP to semantic level using attribution
        vectors that capture dependency relationships.
        """
        # Check if token is grounded in context
        grounded = token in context_tokens
        
        # Check for partial matches (stemming approximation)
        partial_match = any(
            token[:min(4, len(token))] == ctx[:min(4, len(ctx))]
            for ctx in context_tokens if len(ctx) >= 3
        ) if len(token) >= 3 else False
        
        return {
            'grounded': 1.0 if grounded else (0.5 if partial_match else 0.0),
            'confidence': 0.9 if grounded else (0.6 if partial_match else 0.3)
        }
    
    def _build_reasoning_graph(self, substantive_tokens: List[str], 
                                context_tokens: Set[str]) -> Dict[str, Any]:
        """
        Build semantic-level reasoning graph from attribution vectors.
        
        The graph captures dependencies between substantive tokens and
        their grounding in the context.
        """
        nodes = []
        edges = []
        
        for i, token in enumerate(substantive_tokens):
            attr = self._compute_attribution_vector(token, context_tokens)
            nodes.append({
                'token': token,
                'grounded': attr['grounded'],
                'confidence': attr['confidence']
            })
            
            # Add edges to previous tokens (sequential dependency)
            if i > 0:
                edges.append((i-1, i, 0.5))
        
        # Compute graph-level features
        total_grounding = sum(n['grounded'] for n in nodes) / len(nodes) if nodes else 0
        avg_confidence = sum(n['confidence'] for n in nodes) / len(nodes) if nodes else 0
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_grounding': total_grounding,
            'avg_confidence': avg_confidence,
            'node_count': len(nodes)
        }
    
    def _detect_semantic_drift(self, graph: Dict[str, Any]) -> float:
        """
        Detect semantic drift - when substantive tokens are treated as linking.
        
        From paper: "hallucinations originate from LLM mistakenly generating 
        substantive tokens as linking, which manifests on the surface as 
        generation based on entity popularity"
        """
        if not graph['nodes']:
            return 0.5
        
        # Count tokens with low grounding (potential hallucinations)
        ungrounded_substantive = sum(
            1 for n in graph['nodes'] if n['grounded'] < 0.3
        )
        
        drift_ratio = ungrounded_substantive / len(graph['nodes'])
        
        # Higher drift = more hallucination risk
        return drift_ratio
    
    def _compute_attribution_consistency(self, graph: Dict[str, Any]) -> float:
        """
        Compute consistency of attribution scores across the reasoning graph.
        
        Inconsistent attribution patterns indicate potential hallucination.
        """
        if not graph['nodes'] or len(graph['nodes']) < 2:
            return 0.5
        
        groundings = [n['grounded'] for n in graph['nodes']]
        mean_g = sum(groundings) / len(groundings)
        
        # Compute variance
        variance = sum((g - mean_g) ** 2 for g in groundings) / len(groundings)
        
        # Lower variance = more consistent = less hallucination
        consistency = 1.0 - min(1.0, variance * 2)
        
        return consistency
    
    def evaluate(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate using SIRG's semantic-level reasoning graph approach.
        
        Process:
        1. Tokenize and classify tokens (linking vs substantive)
        2. Build semantic reasoning graph with attribution vectors
        3. Detect semantic drift patterns
        4. Compute final SIRG score
        """
        context = sample.get('context', '')
        answer = sample.get('answer', '')
        question = sample.get('question', '')
        
        # Tokenize
        context_tokens = set(self._tokenize(context))
        answer_tokens = self._tokenize(answer)
        
        # Classify answer tokens
        linking_tokens, substantive_tokens = self._classify_tokens(answer_tokens)
        
        # Compute linking ratio
        total_tokens = len(linking_tokens) + len(substantive_tokens)
        linking_ratio = len(linking_tokens) / total_tokens if total_tokens > 0 else 0.5
        
        # Build reasoning graph for substantive tokens
        graph = self._build_reasoning_graph(substantive_tokens, context_tokens)
        
        # Compute semantic grounding
        semantic_grounding = graph['total_grounding']
        
        # Detect semantic drift
        drift = self._detect_semantic_drift(graph)
        
        # Compute attribution consistency
        consistency = self._compute_attribution_consistency(graph)
        
        # Final SIRG score (higher = more faithful, less hallucination)
        # Combines grounding, anti-drift, and consistency
        sirg_score = (
            semantic_grounding * 0.4 +
            (1.0 - drift) * 0.35 +
            consistency * 0.25
        )
        
        # Apply dynamic threshold adjustment based on semantic drift
        if drift > 0.5:
            sirg_score = sirg_score * (1.0 - self.semantic_drift_penalty)
        
        # Apply noise and clamp
        scores = {
            'semantic_grounding': self.add_noise(max(0, min(1, semantic_grounding))),
            'linking_ratio': self.add_noise(max(0, min(1, 1 - linking_ratio))),  # Invert: more substantive = better
            'attribution_consistency': self.add_noise(max(0, min(1, consistency))),
            'sirg_score': self.add_noise(max(0, min(1, sirg_score)))
        }
        
        return scores
