"""
DeepEval Evaluator
==================

Implementation of DeepEval metrics for RAG evaluation.

Reference: Confident AI DeepEval Framework
https://github.com/confident-ai/deepeval

This implementation supports both:
1. LLM-as-judge mode (production, matches paper claims)
2. Heuristic mode (for testing without API costs)
"""

from typing import Dict, Optional
import numpy as np
from . import HeuristicEvaluator


class DeepEvalEvaluator(HeuristicEvaluator):
    """
    DeepEval-style evaluation framework.

    DeepEval provides enterprise-grade evaluation with stricter criteria
    than RAGAS and explicit hallucination detection.

    Metrics:
        - faithfulness: Stricter claim verification against context
        - answer_relevancy: Weighted relevancy with bigram overlap
        - contextual_precision: Precision with strictness factor
        - contextual_recall: Recall evaluation
        - hallucination: Explicit hallucination detection (inverse of faithfulness)

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
        Initialize DeepEval evaluator.

        Args:
            noise_std: Standard deviation for noise (0 for LLM mode)
            use_llm: Whether to use LLM for evaluation
            llm_client: Pre-configured LLM client (optional)
            model: Model to use
            api_key: API key (uses env var if not provided)
            base_url: API base URL (for OpenAI-compatible providers)
        """
        super().__init__(
            name="DeepEval",
            metrics=["faithfulness", "answer_relevancy", "contextual_precision",
                     "contextual_recall", "hallucination"],
            noise_std=noise_std if not use_llm else 0.0
        )

        self.use_llm = use_llm
        self.llm_client = llm_client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        # Strictness factor for DeepEval (stricter than RAGAS)
        self.strictness_factor = 0.95

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
        """Evaluate a sample using DeepEval metrics."""
        if self.use_llm and self.llm_client is not None:
            return self._evaluate_with_llm(sample)
        else:
            return self._evaluate_heuristic(sample)

    def _evaluate_with_llm(self, sample: Dict) -> Dict[str, float]:
        """Evaluate using LLM-as-judge with DeepEval methodology."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})

        results = {}

        # Faithfulness with strict verification
        try:
            results['faithfulness'] = self._evaluate_faithfulness_llm(answer, context, question)
        except Exception as e:
            print(f"DeepEval faithfulness failed: {e}")
            results['faithfulness'] = self._compute_faithfulness(answer, context)

        # Answer relevancy with detailed rubric
        try:
            results['answer_relevancy'] = self._evaluate_relevancy_llm(answer, question)
        except Exception as e:
            print(f"DeepEval relevancy failed: {e}")
            results['answer_relevancy'] = self._compute_answer_relevancy(question, answer)

        # Contextual precision
        try:
            results['contextual_precision'] = self._evaluate_precision_llm(question, context)
        except Exception as e:
            print(f"DeepEval precision failed: {e}")
            results['contextual_precision'] = self._compute_contextual_precision(question, context)

        # Contextual recall
        try:
            results['contextual_recall'] = self._evaluate_recall_llm(context, ground_truth, question)
        except Exception as e:
            print(f"DeepEval recall failed: {e}")
            results['contextual_recall'] = self._compute_contextual_recall(question, context)

        # Hallucination detection (explicit)
        try:
            results['hallucination'] = self._evaluate_hallucination_llm(answer, context, question)
        except Exception as e:
            print(f"DeepEval hallucination failed: {e}")
            results['hallucination'] = 1.0 - results['faithfulness']

        return results

    def _evaluate_faithfulness_llm(self, answer: str, context: str, question: str) -> float:
        """
        DeepEval Faithfulness: Stricter than RAGAS.

        Uses chain-of-thought reasoning and detailed rubrics.
        """
        prompt = f"""You are a strict evaluator assessing faithfulness of an AI-generated answer.
DeepEval defines faithfulness as: EVERY claim in the answer must be DIRECTLY and EXPLICITLY
supported by the provided context. No inference allowed.

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Evaluation criteria (be STRICT):
1. Extract ALL factual claims from the answer
2. For EACH claim, it must be EXPLICITLY stated in the context
3. Paraphrased information counts ONLY if meaning is preserved exactly
4. Any information not in context = unsupported (even if true)
5. Partial support = unsupported

Think step by step:
1. List each claim
2. Quote the supporting context passage (if any)
3. Verdict: supported or not

Respond with JSON:
{{
    "claims_analysis": [
        {{"claim": "...", "context_evidence": "...", "supported": true/false}}
    ],
    "total_claims": <number>,
    "supported_claims": <number>,
    "faithfulness_score": <supported/total>,
    "reasoning": "summary of evaluation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "claims_analysis": {"type": "array"},
                        "total_claims": {"type": "integer"},
                        "supported_claims": {"type": "integer"},
                        "faithfulness_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["faithfulness_score"]
                }
            )
            # Apply strictness factor
            score = float(response.get('faithfulness_score', 0.5)) * self.strictness_factor
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"DeepEval faithfulness failed: {e}")

    def _evaluate_relevancy_llm(self, answer: str, question: str) -> float:
        """DeepEval Answer Relevancy with detailed rubric."""
        prompt = f"""You are evaluating answer relevancy using DeepEval criteria.

Question: {question}

Answer: {answer}

DeepEval Relevancy Rubric:
- 1.0: Perfect - Directly answers the question completely
- 0.9: Excellent - Answers question with minor extra information
- 0.7: Good - Addresses main question, some gaps or extras
- 0.5: Partial - Addresses some aspects, misses key points
- 0.3: Poor - Mostly irrelevant with some related content
- 0.0: Irrelevant - Does not address the question at all

Consider:
1. Does it answer what was SPECIFICALLY asked?
2. Is any critical information missing?
3. Is there excessive irrelevant content?

Respond with JSON:
{{
    "rubric_level": <0.0-1.0>,
    "completeness": <0-1>,
    "focus": <0-1>,
    "answer_relevancy": <final score 0-1>,
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "rubric_level": {"type": "number"},
                        "completeness": {"type": "number"},
                        "focus": {"type": "number"},
                        "answer_relevancy": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["answer_relevancy"]
                }
            )
            score = float(response.get('answer_relevancy', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"DeepEval relevancy failed: {e}")

    def _evaluate_precision_llm(self, question: str, context: str) -> float:
        """DeepEval Contextual Precision."""
        prompt = f"""Evaluate the precision of retrieved context using DeepEval criteria.

Question: {question}

Retrieved Context:
{context}

For each passage/section:
1. Is it relevant to answering the question?
2. Does it contain useful information?
3. Rate the noise level (irrelevant content)

Respond with JSON:
{{
    "passages_evaluated": <number>,
    "relevant_passages": <number>,
    "noise_ratio": <0-1>,
    "contextual_precision": <relevant/total adjusted for noise>,
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "passages_evaluated": {"type": "integer"},
                        "relevant_passages": {"type": "integer"},
                        "noise_ratio": {"type": "number"},
                        "contextual_precision": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["contextual_precision"]
                }
            )
            score = float(response.get('contextual_precision', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"DeepEval precision failed: {e}")

    def _evaluate_recall_llm(self, context: str, ground_truth: Dict, question: str) -> float:
        """DeepEval Contextual Recall."""
        if isinstance(ground_truth, dict):
            gt_text = ' '.join(str(v) for v in ground_truth.values() if v)
        else:
            gt_text = str(ground_truth) if ground_truth else question

        prompt = f"""Evaluate contextual recall using DeepEval criteria.

Question: {question}

Expected Information (Ground Truth):
{gt_text}

Retrieved Context:
{context}

Determine:
1. What key information is needed to answer correctly?
2. How much of that information is in the context?
3. Are there critical gaps?

Respond with JSON:
{{
    "required_information": ["item1", "item2", ...],
    "found_in_context": <number>,
    "critical_missing": ["missing items"],
    "contextual_recall": <found/required>,
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "required_information": {"type": "array"},
                        "found_in_context": {"type": "integer"},
                        "critical_missing": {"type": "array"},
                        "contextual_recall": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["contextual_recall"]
                }
            )
            score = float(response.get('contextual_recall', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"DeepEval recall failed: {e}")

    def _evaluate_hallucination_llm(self, answer: str, context: str, question: str) -> float:
        """
        DeepEval Hallucination Detection.

        Explicit detection of fabricated or unsupported information.
        """
        prompt = f"""You are detecting hallucinations in an AI-generated answer.
A hallucination is ANY information in the answer that is NOT supported by the context.

Context:
{context}

Question: {question}

Answer to analyze:
{answer}

Identify hallucinations:
1. Fabricated facts not in context
2. Incorrect numbers, dates, names
3. Made-up relationships or causations
4. Exaggerations beyond what context states
5. Confident claims without basis

Respond with JSON:
{{
    "hallucinations_found": [
        {{"content": "...", "type": "fabricated/incorrect/exaggerated", "severity": "high/medium/low"}}
    ],
    "total_statements": <number>,
    "hallucinated_statements": <number>,
    "hallucination_score": <hallucinated/total>,
    "reasoning": "summary"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "hallucinations_found": {"type": "array"},
                        "total_statements": {"type": "integer"},
                        "hallucinated_statements": {"type": "integer"},
                        "hallucination_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["hallucination_score"]
                }
            )
            score = float(response.get('hallucination_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"DeepEval hallucination failed: {e}")

    def _evaluate_heuristic(self, sample: Dict) -> Dict[str, float]:
        """Evaluate using heuristic approximations."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')

        faithfulness = self._compute_faithfulness(answer, context)
        answer_relevancy = self._compute_answer_relevancy(question, answer)
        contextual_precision = self._compute_contextual_precision(question, context)
        contextual_recall = self._compute_contextual_recall(question, context)
        hallucination = 1.0 - faithfulness

        return {
            'faithfulness': self.add_noise(faithfulness),
            'answer_relevancy': self.add_noise(answer_relevancy),
            'contextual_precision': self.add_noise(contextual_precision),
            'contextual_recall': self.add_noise(contextual_recall),
            'hallucination': self.add_noise(hallucination)
        }

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        """Compute faithfulness with DeepEval's stricter criteria."""
        if not answer or not context:
            return 0.0

        statements = self.extract_claims(answer)
        if not statements:
            return 0.5

        faithful_count = 0
        for statement in statements:
            coverage = self.coverage_score(statement, context)
            bigram_overlap = self.ngram_overlap(statement, context, n=2)
            combined_score = 0.6 * coverage + 0.4 * bigram_overlap
            if combined_score > 0.25:
                faithful_count += 1

        base_score = faithful_count / len(statements)
        return base_score * self.strictness_factor

    def _compute_answer_relevancy(self, question: str, answer: str) -> float:
        """Compute answer relevancy."""
        if not question or not answer:
            return 0.0

        overlap = self.word_overlap(question, answer)
        bigram_overlap = self.ngram_overlap(question, answer, n=2)

        return 0.7 * overlap + 0.3 * bigram_overlap

    def _compute_contextual_precision(self, question: str, context: str) -> float:
        """Compute contextual precision."""
        if not question or not context:
            return 0.0

        chunks = [c.strip() for c in context.split('---') if c.strip()]
        if not chunks:
            chunks = [context]

        relevant_chunks = 0
        for chunk in chunks:
            relevance = self.word_overlap(question, chunk)
            if relevance > 0.15:
                relevant_chunks += 1

        precision = relevant_chunks / len(chunks)
        return precision * self.strictness_factor

    def _compute_contextual_recall(self, question: str, context: str) -> float:
        """Compute contextual recall."""
        if not question or not context:
            return 0.0

        question_words = set(question.lower().split())
        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with',
                     'how', 'why', 'when', 'where', 'who', 'which', 'are', 'was', 'were',
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can'}

        key_terms = question_words - stopwords
        if not key_terms:
            return 0.5

        context_words = set(context.lower().split())
        covered = key_terms & context_words

        return len(covered) / len(key_terms)
