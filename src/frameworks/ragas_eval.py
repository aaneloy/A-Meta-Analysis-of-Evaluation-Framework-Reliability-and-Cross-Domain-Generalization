"""
RAGAS Evaluator
===============

Implementation of RAGAS (Retrieval Augmented Generation Assessment) metrics.

Reference: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
arXiv:2309.15217, EACL 2024

This implementation supports both:
1. LLM-as-judge mode (production, matches paper claims)
2. Heuristic mode (for testing without API costs)
"""

from typing import Dict, Optional
import numpy as np
from . import HeuristicEvaluator


class RAGASEvaluator(HeuristicEvaluator):
    """
    RAGAS-style evaluation framework.

    Metrics:
        - faithfulness: Fraction of claims in answer supported by context
        - answer_relevancy: How relevant the answer is to the question
        - context_precision: Precision of retrieved context
        - context_recall: Recall of retrieved context

    Modes:
        - use_llm=True: Uses LLM for evaluation (production mode)
        - use_llm=False: Uses heuristic approximations (testing mode)
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        use_llm: bool = True,
        llm_client=None,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            noise_std: Standard deviation for noise (0 for LLM mode)
            use_llm: Whether to use LLM for evaluation
            llm_client: Pre-configured LLM client (optional)
            model: Model to use
            api_key: API key (uses env var if not provided)
            base_url: API base URL (for OpenAI-compatible providers)
        """
        super().__init__(
            name="RAGAS",
            metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            noise_std=noise_std if not use_llm else 0.0
        )

        self.use_llm = use_llm
        self.llm_client = llm_client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        # Initialize LLM client if using LLM mode
        if self.use_llm and self.llm_client is None:
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize the LLM client."""
        try:
            from ..llm_client import OpenAIClient
            self.llm_client = OpenAIClient(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Falling back to heuristic mode.")
            self.use_llm = False

    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate a sample using RAGAS metrics.

        Uses LLM-as-judge when use_llm=True, otherwise falls back
        to heuristic approximations.
        """
        if self.use_llm and self.llm_client is not None:
            return self._evaluate_with_llm(sample)
        else:
            return self._evaluate_heuristic(sample)

    def _evaluate_with_llm(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using LLM-as-judge.

        RAGAS methodology:
        - Faithfulness: Decompose answer into claims, check each against context via LLM
        - Answer Relevancy: Generate questions from answer, compare embeddings to original
        - Context Precision: Check if retrieved passages are useful for answering
        - Context Recall: Check if context covers ground truth information
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})

        results = {}

        # Faithfulness via LLM
        try:
            faith_result = self._evaluate_faithfulness_llm(answer, context, question)
            results['faithfulness'] = faith_result
        except Exception as e:
            print(f"Faithfulness evaluation failed: {e}")
            results['faithfulness'] = self._compute_faithfulness(answer, context)

        # Answer Relevancy via LLM
        try:
            relevancy_result = self._evaluate_relevancy_llm(answer, question)
            results['answer_relevancy'] = relevancy_result
        except Exception as e:
            print(f"Relevancy evaluation failed: {e}")
            results['answer_relevancy'] = self._compute_answer_relevancy(question, answer)

        # Context Precision via LLM
        try:
            precision_result = self._evaluate_context_precision_llm(question, context)
            results['context_precision'] = precision_result
        except Exception as e:
            print(f"Context precision evaluation failed: {e}")
            results['context_precision'] = self._compute_context_precision(question, context)

        # Context Recall via LLM
        try:
            recall_result = self._evaluate_context_recall_llm(context, ground_truth, question)
            results['context_recall'] = recall_result
        except Exception as e:
            print(f"Context recall evaluation failed: {e}")
            results['context_recall'] = self._compute_context_recall(context, ground_truth)

        return results

    def _evaluate_faithfulness_llm(self, answer: str, context: str, question: str) -> float:
        """
        Evaluate faithfulness using LLM.

        RAGAS Faithfulness:
        1. Extract claims from the answer
        2. For each claim, verify if it can be inferred from the context
        3. Score = number of supported claims / total claims
        """
        prompt = f"""You are evaluating the faithfulness of an AI-generated answer.
Faithfulness measures whether ALL information in the answer is supported by the context.

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Instructions:
1. Identify all factual claims made in the answer
2. For each claim, determine if it is DIRECTLY supported by the context
3. A claim is supported only if the context explicitly contains that information
4. Do not count claims that require inference beyond what's stated

Respond with JSON:
{{
    "claims": ["claim 1", "claim 2", ...],
    "verdicts": [true/false for each claim - true if supported],
    "faithfulness_score": <number of true verdicts / total claims>,
    "explanation": "brief reasoning"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "claims": {"type": "array", "items": {"type": "string"}},
                        "verdicts": {"type": "array", "items": {"type": "boolean"}},
                        "faithfulness_score": {"type": "number"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["faithfulness_score"]
                }
            )
            score = float(response.get('faithfulness_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"Faithfulness LLM evaluation failed: {e}")

    def _evaluate_relevancy_llm(self, answer: str, question: str) -> float:
        """
        Evaluate answer relevancy using LLM.

        RAGAS Answer Relevancy:
        1. Generate N questions that the answer could be responding to
        2. Compute embedding similarity between generated and original question
        3. Score = average similarity

        Simplified: Direct relevancy assessment via LLM.
        """
        prompt = f"""You are evaluating the relevancy of an answer to a question.
Answer relevancy measures how well the answer addresses what was asked.

Question: {question}

Answer: {answer}

Evaluate on a scale of 0 to 1:
- 1.0: Answer directly and completely addresses the question
- 0.8: Answer mostly addresses the question with minor gaps
- 0.5: Answer partially addresses the question
- 0.2: Answer barely relates to the question
- 0.0: Answer is completely irrelevant

Consider:
- Does the answer address the core of what was asked?
- Is the answer complete or missing key information?
- Is there irrelevant information that detracts from relevancy?

Respond with JSON:
{{
    "relevancy_score": <float between 0 and 1>,
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "relevancy_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["relevancy_score"]
                }
            )
            score = float(response.get('relevancy_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"Relevancy LLM evaluation failed: {e}")

    def _evaluate_context_precision_llm(self, question: str, context: str) -> float:
        """
        Evaluate context precision using LLM.

        RAGAS Context Precision:
        Measures what fraction of the retrieved context is actually useful
        for answering the question.
        """
        prompt = f"""You are evaluating the precision of retrieved context.
Context precision measures what fraction of retrieved passages are relevant to the question.

Question: {question}

Retrieved Context (passages separated by ---):
{context}

Instructions:
1. Identify each distinct passage or section in the context
2. For each passage, determine if it contains information useful for answering the question
3. Calculate precision = relevant passages / total passages

Respond with JSON:
{{
    "total_passages": <number>,
    "relevant_passages": <number>,
    "precision_score": <relevant/total>,
    "explanation": "brief reasoning"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "total_passages": {"type": "integer"},
                        "relevant_passages": {"type": "integer"},
                        "precision_score": {"type": "number"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["precision_score"]
                }
            )
            score = float(response.get('precision_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"Context precision LLM evaluation failed: {e}")

    def _evaluate_context_recall_llm(self, context: str, ground_truth: Dict, question: str) -> float:
        """
        Evaluate context recall using LLM.

        RAGAS Context Recall:
        Measures whether the context contains all information needed to
        answer the question (compared against ground truth).
        """
        # Format ground truth
        if isinstance(ground_truth, dict):
            gt_text = ' '.join(str(v) for v in ground_truth.values() if v)
        else:
            gt_text = str(ground_truth) if ground_truth else ""

        if not gt_text:
            # Without ground truth, estimate based on context coverage
            gt_text = question

        prompt = f"""You are evaluating the recall of retrieved context.
Context recall measures whether the context contains all information needed to answer correctly.

Question: {question}

Ground Truth Answer/Information:
{gt_text}

Retrieved Context:
{context}

Instructions:
1. Identify key facts/claims in the ground truth
2. Check if each fact can be found in or inferred from the context
3. Calculate recall = facts found in context / total facts in ground truth

Respond with JSON:
{{
    "ground_truth_facts": ["fact 1", "fact 2", ...],
    "facts_in_context": <number found>,
    "recall_score": <found/total>,
    "explanation": "brief reasoning"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "ground_truth_facts": {"type": "array", "items": {"type": "string"}},
                        "facts_in_context": {"type": "integer"},
                        "recall_score": {"type": "number"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["recall_score"]
                }
            )
            score = float(response.get('recall_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"Context recall LLM evaluation failed: {e}")

    def _evaluate_heuristic(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using heuristic approximations (no API calls).
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})

        faithfulness = self._compute_faithfulness(answer, context)
        answer_relevancy = self._compute_answer_relevancy(question, answer)
        context_precision = self._compute_context_precision(question, context)
        context_recall = self._compute_context_recall(context, ground_truth)

        return {
            'faithfulness': self.add_noise(faithfulness),
            'answer_relevancy': self.add_noise(answer_relevancy),
            'context_precision': self.add_noise(context_precision),
            'context_recall': self.add_noise(context_recall)
        }

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        """Compute faithfulness score using heuristics."""
        if not answer or not context:
            return 0.0

        claims = self.extract_claims(answer)
        if not claims:
            return 0.5

        supported_count = 0
        for claim in claims:
            support_score = self.coverage_score(claim, context)
            if support_score > 0.3:
                supported_count += 1

        return supported_count / len(claims)

    def _compute_answer_relevancy(self, question: str, answer: str) -> float:
        """Compute answer relevancy score using heuristics."""
        if not question or not answer:
            return 0.0

        overlap = self.word_overlap(question, answer)

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'how', 'why', 'when', 'where', 'who'}
        key_question_words = question_words - stopwords

        if key_question_words:
            key_coverage = len(key_question_words & answer_words) / len(key_question_words)
            return 0.5 * overlap + 0.5 * key_coverage

        return overlap

    def _compute_context_precision(self, question: str, context: str) -> float:
        """Compute context precision score using heuristics."""
        if not question or not context:
            return 0.0

        passages = context.split('---')
        if not passages:
            passages = [context]

        relevant_count = 0
        for passage in passages:
            relevance = self.word_overlap(question, passage)
            if relevance > 0.1:
                relevant_count += 1

        return relevant_count / len(passages) if passages else 0.0

    def _compute_context_recall(self, context: str, ground_truth: Dict) -> float:
        """Compute context recall score using heuristics."""
        if not context:
            return 0.0

        if isinstance(ground_truth, dict):
            gt_text = ' '.join(str(v) for v in ground_truth.values() if v)
        else:
            gt_text = str(ground_truth)

        if not gt_text:
            return 0.5

        return self.coverage_score(gt_text, context)
