"""
QAFactEval Evaluator
====================

Implementation of QAFactEval metrics.

Reference: Fabbri et al., "QAFactEval: Improved QA-Based Factual Consistency Evaluation 
for Summarization", NAACL 2022, arXiv:2112.08542
"""

from typing import Dict, List, Tuple
import numpy as np
from . import HeuristicEvaluator
from ..nli import safe_nli_score


class QAFactEvalEvaluator(HeuristicEvaluator):
    """
    QAFactEval-style evaluation framework.
    
    QAFactEval uses question generation and answering to evaluate
    factual consistency. It generates questions from the summary,
    answers them from both summary and source, then compares answers.
    
    Metrics:
        - factual_consistency: QA-based factual consistency score
    """
    
    def __init__(self, noise_std: float = 0.045):
        super().__init__(
            name="QAFactEval",
            metrics=["factual_consistency"],
            noise_std=noise_std
        )
        # Question words for synthetic QG
        self.question_templates = [
            ("what", "What is {}?"),
            ("who", "Who {}?"),
            ("where", "Where {}?"),
            ("when", "When {}?"),
            ("how", "How {}?"),
        ]
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a sample using QAFactEval methodology.
        
        QAFactEval methodology:
        1. Generate questions from the answer (summary)
        2. Answer questions using the answer
        3. Answer questions using the context (source)
        4. Compare answers - consistent if same
        """
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        
        # Compute factual consistency via QA
        factual_consistency = self._compute_qa_consistency(answer, context)
        
        return {
            'factual_consistency': self.add_noise(factual_consistency)
        }
    
    def _compute_qa_consistency(self, answer: str, context: str) -> float:
        """
        Compute QA-based factual consistency.
        
        Uses QA-style decomposition with NLI answer consistency checks.
        """
        if not answer or not context:
            return 0.0
        
        # Extract "facts" from answer (noun phrases, entities)
        facts = self._extract_facts(answer)
        
        if not facts:
            return 0.5  # No verifiable facts
        
        # For each fact, check if it's supported by context
        consistency_scores = []
        
        for fact in facts:
            # Generate a question about this fact
            question = self._generate_question(fact, answer)
            
            # "Answer" the question from context
            context_answer = self._extract_answer(question, context)
            
            # Compare: is the fact consistent with context answer?
            if context_answer:
                consistency = self._compare_answers(fact, context_answer)
            else:
                # Fact not found in context - potentially hallucinated
                consistency = 0.3
            
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract verifiable facts from text.
        
        In real QAFactEval, this uses NER and dependency parsing.
        We use heuristics to identify fact-like phrases.
        """
        if not text:
            return []
        
        facts = []
        words = text.split()
        
        # Extract capitalized phrases (potential entities)
        i = 0
        while i < len(words):
            if words[i] and words[i][0].isupper() and not i == 0:
                # Start of potential entity
                entity = [words[i]]
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    entity.append(words[j])
                    j += 1
                if entity:
                    facts.append(' '.join(entity))
                i = j
            else:
                i += 1
        
        # Extract numbers with context
        for i, word in enumerate(words):
            if any(c.isdigit() for c in word):
                # Get surrounding context
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                fact = ' '.join(words[start:end])
                facts.append(fact)
        
        # Extract key phrases (simple noun phrases)
        sentences = self.extract_claims(text)
        for sent in sentences:
            # First few words often contain key info
            sent_words = sent.split()
            if len(sent_words) >= 3:
                facts.append(' '.join(sent_words[:4]))
        
        # Deduplicate and limit
        unique_facts = list(set(facts))[:5]  # Max 5 facts
        return unique_facts
    
    def _generate_question(self, fact: str, context: str) -> str:
        """
        Generate a question about the fact.
        
        In real QAFactEval, this uses a QG model.
        """
        fact_lower = fact.lower()
        
        # Determine question type based on fact content
        if any(c.isdigit() for c in fact):
            # Numeric fact -> how many/when
            return f"How many or when {fact_lower}?"
        elif fact[0].isupper():
            # Entity -> who/what
            return f"Who or what is {fact_lower}?"
        else:
            # General -> what
            return f"What is {fact_lower}?"
    
    def _extract_answer(self, question: str, context: str) -> str:
        """
        Extract an answer to the question from context.
        
        In real QAFactEval, this uses a QA model.
        We use keyword matching to find relevant spans.
        """
        if not context:
            return ""
        
        # Extract key terms from question
        q_words = set(question.lower().split())
        stopwords = {'what', 'who', 'where', 'when', 'how', 'is', 'the', 'a', 'an', 
                     'or', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 'many'}
        key_terms = q_words - stopwords
        
        if not key_terms:
            return ""
        
        # Find sentences in context containing key terms
        sentences = [s.strip() for s in context.replace('---', '.').split('.') if s.strip()]
        
        best_match = ""
        best_score = 0
        
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(key_terms & sent_words)
            if overlap > best_score:
                best_score = overlap
                best_match = sent
        
        return best_match
    
    def _compare_answers(self, answer1: str, answer2: str) -> float:
        """
        Compare two answers for consistency.
        
        Uses NLI-based answer equivalence scoring with DeBERTa-v3-large-MNLI.
        """
        if not answer1 or not answer2:
            return 0.0
        
        nli_forward = safe_nli_score(premise=answer2, hypothesis=answer1)
        nli_backward = safe_nli_score(premise=answer1, hypothesis=answer2)
        entailment = (nli_forward["entailment"] + nli_backward["entailment"]) / 2
        contradiction = (nli_forward["contradiction"] + nli_backward["contradiction"]) / 2
        score = entailment - 0.5 * contradiction
        return max(0.0, min(1.0, score))
