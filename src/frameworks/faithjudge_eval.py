"""
FaithJudge Evaluator
====================

LLM-as-a-Judge Framework for RAG Faithfulness Evaluation.

Reference: Tamber et al., "Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards"
arXiv:2505.04847, EMNLP 2025 Industry Track

This implementation supports both:
1. LLM-as-judge mode (production, matches paper claims)
2. Heuristic mode (for testing without API costs)
"""

from typing import Dict, List, Optional
import numpy as np
import re
from . import HeuristicEvaluator


class FaithJudgeEvaluator(HeuristicEvaluator):
    """
    FaithJudge: LLM-as-a-Judge for RAG Faithfulness (EMNLP 2025)

    Key insight: Few-shot examples from the same source document calibrate
    LLM judges to subtle, article-specific details.

    Metrics:
        - faithfulness: Overall faithfulness score
        - hallucination_rate: Proportion of hallucinated content
        - grounding_score: How well grounded in source documents
        - subtle_error_detection: Sensitivity to subtle factual changes
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
        super().__init__(
            name="FaithJudge",
            metrics=["faithfulness", "hallucination_rate", "grounding_score", "subtle_error_detection"],
            noise_std=noise_std if not use_llm else 0.0
        )

        self.use_llm = use_llm
        self.llm_client = llm_client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.version = "2025-05"

        self.domain_calibration = {
            'General Knowledge': 0.0,
            'Finance': -0.02,
            'Biomedicine': -0.03,
        }

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
        """Evaluate using FaithJudge methodology."""
        if self.use_llm and self.llm_client is not None:
            return self._evaluate_with_llm(sample)
        else:
            return self._evaluate_heuristic(sample)

    def _evaluate_with_llm(self, sample: Dict) -> Dict[str, float]:
        """Evaluate using LLM with FaithJudge methodology."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        domain = sample.get('domain', 'General Knowledge')

        results = {}

        # Combined evaluation prompt for FaithJudge
        try:
            llm_results = self._evaluate_faithfulness_llm(answer, context, question)
            results['faithfulness'] = llm_results.get('faithfulness', 0.5)
            results['hallucination_rate'] = llm_results.get('hallucination_rate', 0.5)
            results['grounding_score'] = llm_results.get('grounding_score', 0.5)
            results['subtle_error_detection'] = llm_results.get('subtle_error_detection', 0.0)
        except Exception as e:
            print(f"FaithJudge LLM evaluation failed: {e}")
            return self._evaluate_heuristic(sample)

        # Apply domain calibration
        domain_adj = self.domain_calibration.get(domain, 0.0)
        results['faithfulness'] = np.clip(results['faithfulness'] + domain_adj, 0, 1)

        return results

    def _evaluate_faithfulness_llm(self, answer: str, context: str, question: str) -> Dict[str, float]:
        """
        FaithJudge comprehensive evaluation using LLM.

        FaithJudge uses few-shot prompting with source-specific examples
        to detect subtle hallucinations.
        """
        prompt = f"""You are FaithJudge, an expert evaluator for RAG system faithfulness.

Your task is to evaluate how faithful the generated answer is to the source context.
FaithJudge is specifically designed to detect SUBTLE errors that other evaluators miss:
- Hedge word removal: "reportedly carcinogenic" → "carcinogenic"
- Precision changes: "approximately 100" → "100"
- Qualifier removal: "some experts say" → "experts say"

Source Context:
{context}

Question: {question}

Generated Answer:
{answer}

Evaluate the answer on these dimensions:

1. FAITHFULNESS: Are all claims in the answer supported by the context?
   - Check each factual claim
   - Identify any unsupported or contradicted claims
   - Score 0-1 (1 = perfectly faithful)

2. HALLUCINATION RATE: What fraction of the answer contains hallucinated content?
   - Count unsupported claims / total claims
   - Score 0-1 (0 = no hallucination, 1 = all hallucinated)

3. GROUNDING SCORE: How well is the answer grounded in the source?
   - Can each statement be traced back to the context?
   - Score 0-1 (1 = fully grounded)

4. SUBTLE ERROR DETECTION: Are there subtle factual modifications?
   - Check for removed hedges/qualifiers
   - Check for precision changes
   - Check for exaggerations
   - Score 0-1 (1 = many subtle errors detected)

Respond with JSON:
{{
    "claim_analysis": [
        {{"claim": "...", "supported": true/false, "type": "supported/unsupported/subtle_error"}}
    ],
    "faithfulness": <0-1>,
    "hallucination_rate": <0-1>,
    "grounding_score": <0-1>,
    "subtle_error_detection": <0-1>,
    "subtle_errors_found": ["list of subtle errors"],
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "claim_analysis": {"type": "array"},
                        "faithfulness": {"type": "number"},
                        "hallucination_rate": {"type": "number"},
                        "grounding_score": {"type": "number"},
                        "subtle_error_detection": {"type": "number"},
                        "subtle_errors_found": {"type": "array"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["faithfulness", "hallucination_rate", "grounding_score"]
                }
            )
            return {
                'faithfulness': max(0.0, min(1.0, float(response.get('faithfulness', 0.5)))),
                'hallucination_rate': max(0.0, min(1.0, float(response.get('hallucination_rate', 0.5)))),
                'grounding_score': max(0.0, min(1.0, float(response.get('grounding_score', 0.5)))),
                'subtle_error_detection': max(0.0, min(1.0, float(response.get('subtle_error_detection', 0.0))))
            }
        except Exception as e:
            raise RuntimeError(f"FaithJudge evaluation failed: {e}")

    def _evaluate_heuristic(self, sample: Dict) -> Dict[str, float]:
        """Evaluate using heuristic approximations."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        domain = sample.get('domain', 'General Knowledge')

        grounding_score = self._compute_grounding(answer, context)
        subtle_errors = self._detect_subtle_errors(answer, context)
        claim_verification = self._verify_claims(answer, context)

        base_faithfulness = 0.5 * grounding_score + 0.5 * claim_verification
        subtle_penalty = subtle_errors * 0.15
        faithfulness = max(0.0, base_faithfulness - subtle_penalty)

        domain_adj = self.domain_calibration.get(domain, 0.0)
        faithfulness = np.clip(faithfulness + domain_adj, 0, 1)
        hallucination_rate = 1.0 - claim_verification

        return {
            'faithfulness': self.add_noise(faithfulness),
            'hallucination_rate': self.add_noise(hallucination_rate),
            'grounding_score': self.add_noise(grounding_score),
            'subtle_error_detection': self.add_noise(subtle_errors)
        }

    def _compute_grounding(self, answer: str, context: str) -> float:
        """Compute how well the answer is grounded in context."""
        if not answer or not context:
            return 0.0

        statements = self.extract_claims(answer)
        if not statements:
            return 0.5

        grounded_count = 0
        for statement in statements:
            grounding = self._statement_grounding(statement, context)
            if grounding > 0.3:
                grounded_count += 1

        return grounded_count / len(statements)

    def _statement_grounding(self, statement: str, context: str) -> float:
        """Check if a statement is grounded in context."""
        word_overlap = self.word_overlap(statement, context)
        statement_lower = statement.lower()
        context_lower = context.lower()

        key_terms = re.findall(r'\b[A-Za-z]{4,}\b|\b\d+(?:\.\d+)?%?\b', statement)
        if key_terms:
            matched = sum(1 for term in key_terms if term.lower() in context_lower)
            key_overlap = matched / len(key_terms)
        else:
            key_overlap = word_overlap

        return 0.4 * word_overlap + 0.6 * key_overlap

    def _detect_subtle_errors(self, answer: str, context: str) -> float:
        """Detect subtle factual errors."""
        if not answer or not context:
            return 0.0

        subtle_error_count = 0
        total_checks = 0

        answer_lower = answer.lower()
        context_lower = context.lower()

        hedge_words = [
            'reportedly', 'allegedly', 'approximately', 'about', 'around',
            'estimated', 'roughly', 'possibly', 'potentially', 'may', 'might',
            'could', 'some', 'certain', 'particular', 'likely', 'probably',
            'appears', 'seems', 'suggests', 'indicates', 'according to'
        ]

        for hedge in hedge_words:
            if hedge in context_lower:
                total_checks += 1
                if hedge not in answer_lower:
                    context_sentences = context_lower.split('.')
                    for sent in context_sentences:
                        if hedge in sent:
                            sent_words = set(sent.split()) - {hedge}
                            answer_words = set(answer_lower.split())
                            if len(sent_words & answer_words) > 3:
                                subtle_error_count += 1
                                break

        context_numbers = re.findall(r'(?:about|approximately|around|roughly)?\s*(\d+(?:\.\d+)?)', context_lower)
        answer_numbers = re.findall(r'(\d+(?:\.\d+)?)', answer_lower)

        for num in answer_numbers:
            total_checks += 1
            if num in context_lower:
                pattern = rf'(?:about|approximately|around|roughly)\s*{re.escape(num)}'
                if re.search(pattern, context_lower) and not re.search(pattern, answer_lower):
                    subtle_error_count += 1

        if total_checks == 0:
            return 0.0

        return subtle_error_count / total_checks

    def _verify_claims(self, answer: str, context: str) -> float:
        """Verify claims in the answer against context."""
        if not answer or not context:
            return 0.5

        claims = self.extract_claims(answer)
        if not claims:
            return 0.5

        verified = 0
        for claim in claims:
            support = self._claim_support_level(claim, context)
            if support in ['consistent', 'benign']:
                verified += 1
            elif support == 'questionable':
                verified += 0.5

        return verified / len(claims)

    def _claim_support_level(self, claim: str, context: str) -> str:
        """Determine the support level for a claim."""
        coverage = self.coverage_score(claim, context)
        ngram = self.ngram_overlap(claim, context, n=3)
        support_score = 0.6 * coverage + 0.4 * ngram

        if support_score > 0.5:
            return 'consistent'
        elif support_score > 0.3:
            return 'benign'
        elif support_score > 0.15:
            return 'questionable'
        else:
            return 'unwanted'
