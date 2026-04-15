"""
RAGChecker Evaluator
====================

Implementation of RAGChecker metrics.

Reference: Ru et al., "RAGChecker: A Fine-grained Framework for Diagnosing 
Retrieval-Augmented Generation", NeurIPS 2024
arXiv:2408.08067
"""

from typing import Dict, List
import numpy as np
from . import HeuristicEvaluator
from ..nli import safe_nli_score


class RAGCheckerEvaluator(HeuristicEvaluator):
    """
    RAGChecker-style evaluation framework.
    
    RAGChecker differs from RAGAS/DeepEval by using claim-level analysis
    with explicit entailment checking rather than LLM-as-judge.
    
    Metrics:
        - claim_recall: What fraction of expected claims are in the answer
        - context_utilization: How well is retrieved context used
        - faithfulness: Claim-level grounding score
        - noise_sensitivity: How much noise affects the answer
        - hallucination: Proportion of unsupported claims
    """
    
    def __init__(self, noise_std: float = 0.055):
        super().__init__(
            name="RAGChecker",
            metrics=["claim_recall", "context_utilization", "faithfulness", 
                     "noise_sensitivity", "hallucination"],
            noise_std=noise_std
        )
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a sample using RAGChecker's claim-level approach.
        
        RAGChecker:
        1. Decomposes answer into atomic claims
        2. Checks each claim against context via NLI
        3. Computes fine-grained metrics
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})
        
        # Extract claims from answer
        claims = self.extract_claims(answer)
        
        # Claim Recall
        claim_recall = self._compute_claim_recall(claims, ground_truth)
        
        # Context Utilization
        context_utilization = self._compute_context_utilization(question, context, answer)
        
        # Faithfulness (claim-level)
        faithfulness = self._compute_claim_faithfulness(claims, context)
        
        # Noise Sensitivity
        noise_sensitivity = self._compute_noise_sensitivity(faithfulness)
        
        # Hallucination
        hallucination = 1.0 - faithfulness
        
        return {
            'claim_recall': self.add_noise(claim_recall),
            'context_utilization': self.add_noise(context_utilization),
            'faithfulness': self.add_noise(faithfulness),
            'noise_sensitivity': self.add_noise(noise_sensitivity),
            'hallucination': self.add_noise(hallucination)
        }
    
    def _compute_claim_faithfulness(self, claims: List[str], context: str) -> float:
        """
        Compute faithfulness at the claim level.
        
        RAGChecker uses NLI to check if each claim is entailed by context.
        """
        if not claims:
            return 0.5  # No claims to verify
        
        if not context:
            return 0.0
        
        entailed_count = 0
        entailment_scores = []
        
        for claim in claims:
            nli = safe_nli_score(premise=context, hypothesis=claim)
            entailment_score = nli["entailment"]
            entailment_scores.append(entailment_score)

            if entailment_score >= 0.5:
                entailed_count += 1
        
        # RAGChecker uses mean entailment score
        return np.mean(entailment_scores) if entailment_scores else 0.0
    
    def _compute_claim_recall(self, claims: List[str], ground_truth: Dict) -> float:
        """
        Compute claim recall against ground truth.
        
        What fraction of expected information is present in claims?
        """
        if not claims:
            return 0.0
        
        # Extract ground truth text
        if isinstance(ground_truth, dict):
            gt_text = ' '.join(str(v) for v in ground_truth.values() if v)
        else:
            gt_text = str(ground_truth)
        
        if not gt_text:
            return 0.5  # No ground truth
        
        gt_claims = self.extract_claims(gt_text)
        if not gt_claims:
            gt_claims = [gt_text]
        
        # Check how many GT claims are covered by answer claims
        covered = 0
        for gt_claim in gt_claims:
            for claim in claims:
                if self.word_overlap(gt_claim, claim) > 0.3:
                    covered += 1
                    break
        
        return covered / len(gt_claims)
    
    def _compute_context_utilization(self, question: str, context: str, answer: str) -> float:
        """
        Compute how well the context is utilized.
        
        RAGChecker checks if relevant parts of context appear in answer.
        """
        if not context or not answer:
            return 0.0
        
        # Extract relevant context chunks
        chunks = [c.strip() for c in context.split('---') if c.strip()]
        if not chunks:
            chunks = [context]
        
        # Check which chunks are relevant to question
        relevant_chunks = []
        for chunk in chunks:
            if self.word_overlap(question, chunk) > 0.1:
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            return 0.3  # No clearly relevant chunks
        
        # Check utilization of relevant chunks in answer
        utilized = 0
        for chunk in relevant_chunks:
            utilization = self.word_overlap(chunk, answer)
            if utilization > 0.1:
                utilized += 1
        
        return utilized / len(relevant_chunks)
    
    def _compute_noise_sensitivity(self, faithfulness: float) -> float:
        """
        Compute noise sensitivity score.
        
        Higher faithfulness = lower sensitivity to noise.
        This simulates how much irrelevant context affects the answer.
        """
        # Inverse relationship with faithfulness
        # High faithfulness means the answer is well-grounded, thus less sensitive to noise
        return 1.0 - (faithfulness * 0.7)  # Scale to reasonable range
