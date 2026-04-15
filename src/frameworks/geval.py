"""
G-Eval Evaluator
================

Implementation of G-Eval metrics.

Reference: Liu et al., "G-Eval: NLG Evaluation using LLM with Better Human Alignment"
EMNLP 2023, arXiv:2303.16634

This implementation supports both:
1. LLM-as-judge mode (production, matches paper claims)
2. Heuristic mode (for testing without API costs)
"""

from typing import Dict, Optional
import numpy as np
from . import HeuristicEvaluator


class GEvalEvaluator(HeuristicEvaluator):
    """
    G-Eval-style evaluation framework.

    G-Eval uses LLM with chain-of-thought prompting and form-filling
    to evaluate NLG outputs. It shows better human alignment than
    traditional metrics.

    Metrics:
        - coherence: Logical flow and structure
        - consistency: Factual alignment with source
        - fluency: Grammatical correctness and readability
        - relevance: Topical relevance to the query

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
        Initialize G-Eval evaluator.

        Args:
            noise_std: Standard deviation for noise (0 for LLM mode)
            use_llm: Whether to use LLM for evaluation
            llm_client: Pre-configured LLM client (optional)
            model: OpenAI model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        super().__init__(
            name="G-Eval",
            metrics=["coherence", "consistency", "fluency", "relevance"],
            noise_std=noise_std if not use_llm else 0.0
        )

        self.use_llm = use_llm
        self.llm_client = llm_client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

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
        """Evaluate a sample using G-Eval metrics."""
        if self.use_llm and self.llm_client is not None:
            return self._evaluate_with_llm(sample)
        else:
            return self._evaluate_heuristic(sample)

    def _evaluate_with_llm(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using LLM with G-Eval methodology.

        G-Eval methodology:
        1. Define evaluation criteria with detailed descriptions
        2. Use chain-of-thought to analyze
        3. Output score on 1-5 scale
        4. Convert to 0-1 probability
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')

        results = {}

        # Coherence
        try:
            results['coherence'] = self._evaluate_coherence_llm(answer)
        except Exception as e:
            print(f"G-Eval coherence failed: {e}")
            results['coherence'] = self._compute_coherence(answer)

        # Consistency
        try:
            results['consistency'] = self._evaluate_consistency_llm(answer, context)
        except Exception as e:
            print(f"G-Eval consistency failed: {e}")
            results['consistency'] = self._compute_consistency(answer, context)

        # Fluency
        try:
            results['fluency'] = self._evaluate_fluency_llm(answer)
        except Exception as e:
            print(f"G-Eval fluency failed: {e}")
            results['fluency'] = self._compute_fluency(answer)

        # Relevance
        try:
            results['relevance'] = self._evaluate_relevance_llm(question, answer)
        except Exception as e:
            print(f"G-Eval relevance failed: {e}")
            results['relevance'] = self._compute_relevance(question, answer)

        return results

    def _evaluate_coherence_llm(self, answer: str) -> float:
        """
        G-Eval Coherence evaluation using LLM.

        Definition: "The collective quality of all sentences. We align this
        dimension with DUC quality question of structure and coherence."
        """
        prompt = f"""You are evaluating the COHERENCE of a text using G-Eval methodology.

Definition: Coherence measures the collective quality of all sentences in terms of
structure, logical flow, and organization. A coherent text has:
- Clear logical progression of ideas
- Smooth transitions between sentences
- Consistent topic and focus
- Well-organized structure

Text to evaluate:
{answer}

Evaluation Steps (Chain-of-Thought):
1. Read through the entire text
2. Identify the main topic and how ideas connect
3. Check for logical flow between sentences
4. Assess transitions and connective words
5. Consider overall organization

Rate on scale of 1-5:
1: Very incoherent - disorganized, no logical flow
2: Somewhat incoherent - major gaps in logic or organization
3: Moderately coherent - some logical flow but issues remain
4: Coherent - good flow with minor issues
5: Very coherent - excellent organization and logical progression

Respond with JSON:
{{
    "evaluation_steps": "your chain-of-thought analysis",
    "score_1_5": <integer 1-5>,
    "coherence_score": <score_1_5 - 1) / 4 to normalize to 0-1>,
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "evaluation_steps": {"type": "string"},
                        "score_1_5": {"type": "integer"},
                        "coherence_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["coherence_score"]
                }
            )
            score = float(response.get('coherence_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"G-Eval coherence failed: {e}")

    def _evaluate_consistency_llm(self, answer: str, context: str) -> float:
        """
        G-Eval Consistency evaluation using LLM.

        Definition: "The factual alignment between the summary and
        the summarized source document."
        """
        prompt = f"""You are evaluating the CONSISTENCY of a text using G-Eval methodology.

Definition: Consistency measures the factual alignment between the generated text
and the source document. A consistent text:
- Contains only information that can be verified from the source
- Does not contradict any facts in the source
- Does not add fabricated information

Source Document:
{context}

Generated Text to evaluate:
{answer}

Evaluation Steps (Chain-of-Thought):
1. Identify all factual claims in the generated text
2. For each claim, check if it matches the source
3. Look for any contradictions with the source
4. Identify any information not present in the source

Rate on scale of 1-5:
1: Very inconsistent - major contradictions or fabrications
2: Somewhat inconsistent - several factual errors
3: Moderately consistent - minor factual issues
4: Consistent - mostly accurate with tiny issues
5: Very consistent - perfect factual alignment

Respond with JSON:
{{
    "claims_identified": ["claim 1", "claim 2", ...],
    "contradictions_found": ["any contradictions"],
    "score_1_5": <integer 1-5>,
    "consistency_score": <normalized 0-1>,
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "claims_identified": {"type": "array"},
                        "contradictions_found": {"type": "array"},
                        "score_1_5": {"type": "integer"},
                        "consistency_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["consistency_score"]
                }
            )
            score = float(response.get('consistency_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"G-Eval consistency failed: {e}")

    def _evaluate_fluency_llm(self, answer: str) -> float:
        """
        G-Eval Fluency evaluation using LLM.

        Definition: "The quality of the summary in terms of grammar,
        spelling, punctuation, word choice, and sentence structure."
        """
        prompt = f"""You are evaluating the FLUENCY of a text using G-Eval methodology.

Definition: Fluency measures the linguistic quality of the text including:
- Grammar and syntax correctness
- Spelling and punctuation
- Word choice and vocabulary
- Sentence structure variety
- Natural reading flow

Text to evaluate:
{answer}

Evaluation Steps (Chain-of-Thought):
1. Read for grammatical errors
2. Check spelling and punctuation
3. Assess word choice appropriateness
4. Evaluate sentence structure variety
5. Consider overall readability

Rate on scale of 1-5:
1: Very disfluent - numerous grammar/spelling errors, hard to read
2: Somewhat disfluent - several errors affecting readability
3: Moderately fluent - some minor errors but readable
4: Fluent - good language quality with minor issues
5: Very fluent - excellent grammar, natural flow, professional quality

Respond with JSON:
{{
    "grammar_issues": ["any issues found"],
    "score_1_5": <integer 1-5>,
    "fluency_score": <normalized 0-1>,
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "grammar_issues": {"type": "array"},
                        "score_1_5": {"type": "integer"},
                        "fluency_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["fluency_score"]
                }
            )
            score = float(response.get('fluency_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"G-Eval fluency failed: {e}")

    def _evaluate_relevance_llm(self, question: str, answer: str) -> float:
        """
        G-Eval Relevance evaluation using LLM.

        Definition: "Selection of important content from the source."
        """
        prompt = f"""You are evaluating the RELEVANCE of an answer using G-Eval methodology.

Definition: Relevance measures how well the answer addresses the question by
selecting and presenting the most important, on-topic information.

Question: {question}

Answer: {answer}

Evaluation Steps (Chain-of-Thought):
1. Identify what the question is asking for
2. Check if the answer addresses the main question
3. Assess if important information is included
4. Check for off-topic or unnecessary content
5. Evaluate the answer's focus

Rate on scale of 1-5:
1: Not relevant - does not address the question at all
2: Slightly relevant - tangentially related
3: Moderately relevant - addresses some aspects
4: Relevant - addresses main points with minor gaps
5: Highly relevant - directly and completely addresses the question

Respond with JSON:
{{
    "question_intent": "what the question asks for",
    "answer_coverage": "how well the answer covers it",
    "score_1_5": <integer 1-5>,
    "relevance_score": <normalized 0-1>,
    "reasoning": "explanation"
}}"""

        try:
            response = self.llm_client.generate_with_schema(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "question_intent": {"type": "string"},
                        "answer_coverage": {"type": "string"},
                        "score_1_5": {"type": "integer"},
                        "relevance_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["relevance_score"]
                }
            )
            score = float(response.get('relevance_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"G-Eval relevance failed: {e}")

    def _evaluate_heuristic(self, sample: Dict) -> Dict[str, float]:
        """Evaluate using heuristic approximations."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')

        return {
            'coherence': self.add_noise(self._compute_coherence(answer)),
            'consistency': self.add_noise(self._compute_consistency(answer, context)),
            'fluency': self.add_noise(self._compute_fluency(answer)),
            'relevance': self.add_noise(self._compute_relevance(question, answer))
        }

    def _compute_coherence(self, answer: str) -> float:
        """Compute coherence score using heuristics."""
        if not answer:
            return 0.0

        sentences = self.extract_claims(answer)
        if len(sentences) <= 1:
            return 0.7

        coherence_markers = [
            'however', 'therefore', 'thus', 'moreover', 'furthermore',
            'additionally', 'consequently', 'as a result', 'in addition',
            'first', 'second', 'finally', 'also', 'then', 'next'
        ]

        answer_lower = answer.lower()
        marker_count = sum(1 for m in coherence_markers if m in answer_lower)
        marker_score = min(1.0, marker_count / 3)

        all_words = set()
        sentence_words = []
        for sent in sentences:
            words = set(sent.lower().split())
            sentence_words.append(words)
            all_words.update(words)

        consistency_score = 0
        for i in range(len(sentence_words) - 1):
            overlap = len(sentence_words[i] & sentence_words[i+1])
            consistency_score += min(1.0, overlap / 3)

        if len(sentence_words) > 1:
            consistency_score /= (len(sentence_words) - 1)
        else:
            consistency_score = 0.5

        return 0.4 * marker_score + 0.6 * consistency_score

    def _compute_consistency(self, answer: str, context: str) -> float:
        """Compute consistency score using heuristics."""
        if not answer or not context:
            return 0.0

        claims = self.extract_claims(answer)
        if not claims:
            return 0.5

        consistent_claims = 0
        for claim in claims:
            support = self.coverage_score(claim, context)
            if support > 0.2:
                consistent_claims += 1

        return consistent_claims / len(claims)

    def _compute_fluency(self, answer: str) -> float:
        """Compute fluency score using heuristics."""
        if not answer:
            return 0.0

        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if not sentences:
            return 0.3

        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 10 <= avg_length <= 25:
            length_score = 1.0
        elif avg_length < 5 or avg_length > 40:
            length_score = 0.3
        else:
            length_score = 0.7

        words = answer.lower().split()
        if len(words) > 3:
            repetitions = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
            repetition_score = 1.0 - (repetitions / len(words))
        else:
            repetition_score = 1.0

        first_char = answer[0] if answer else ''
        capitalization_score = 1.0 if first_char.isupper() else 0.5

        punctuation_score = 1.0 if answer.rstrip()[-1] in '.!?' else 0.5

        return (0.3 * length_score + 0.3 * repetition_score +
                0.2 * capitalization_score + 0.2 * punctuation_score)

    def _compute_relevance(self, question: str, answer: str) -> float:
        """Compute relevance score using heuristics."""
        if not question or not answer:
            return 0.0

        word_relevance = self.word_overlap(question, answer)

        q_words = set(question.lower().split())
        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on',
                     'with', 'how', 'why', 'when', 'where', 'who', 'which', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'can', 'could', 'would', 'should'}

        key_terms = q_words - stopwords
        if key_terms:
            a_words = set(answer.lower().split())
            term_coverage = len(key_terms & a_words) / len(key_terms)
        else:
            term_coverage = word_relevance

        return 0.4 * word_relevance + 0.6 * term_coverage
