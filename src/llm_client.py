"""
LLM Client Module
=================

Provides unified interface for LLM API calls via OpenAI-compatible endpoints.
Used by LLM-as-judge evaluation frameworks.
"""

import os
import json
import time
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_with_schema(self, prompt: str, schema: Dict, temperature: float = 0.0) -> Dict:
        """Generate a structured response following a schema."""
        pass


class OpenAIClient(BaseLLMClient):
    """
    OpenAI-compatible client for LLM-as-judge evaluation.

    Supports any OpenAI-compatible API (Groq, Gemini, OpenAI).
    Uses temperature=0 for deterministic evaluation.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        retry_delay: float = 5.0
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            model: Model to use (default: deepseek-chat via DeepSeek)
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: API base URL (for OpenAI-compatible providers like Groq, Gemini)
            temperature: Sampling temperature (default: 0 for deterministic)
            max_retries: Maximum retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self._fallback_client = None

        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Import OpenAI
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai>=1.0.0"
            )

    def _is_rate_limit(self, error: Exception) -> bool:
        """Detect rate-limit errors from any provider."""
        # Check exception type (openai SDK raises specific types)
        err_type = type(error).__name__
        if err_type in ("RateLimitError", "APIStatusError"):
            if hasattr(error, 'status_code') and error.status_code in (429, 503):
                return True
        # Check HTTP status code attribute
        if hasattr(error, 'status_code') and error.status_code in (429, 503):
            return True
        if hasattr(error, 'code') and error.code in (429, 503):
            return True
        # Check string representation as last resort
        err_str = str(error).lower()
        return any(k in err_str for k in [
            "429", "503", "rate_limit", "rate limit", "resource_exhausted",
            "quota", "too many requests", "tokens per", "requests per",
        ])

    def _handle_rate_limit(self, error: Exception, fallback_fn) -> any:
        """Try fallback provider, raise if both fail."""
        if self._fallback_client is not None:
            fb = self._fallback_client
            print(f"    [FALLBACK] {self.model} rate-limited -> {fb.model}")
            try:
                return fallback_fn(fb)
            except Exception as fb_err:
                print(f"    [FALLBACK] {fb.model} also failed -> heuristic")
        else:
            print(f"    [RATE-LIMITED] {self.model} - no fallback configured -> heuristic")
        raise RuntimeError(f"All LLM providers unavailable: {error}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                if self._is_rate_limit(e):
                    return self._handle_rate_limit(
                        e, lambda fb: fb.generate(prompt, system_prompt, temperature, max_tokens)
                    )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} attempts: {e}")

    def generate_with_schema(
        self,
        prompt: str,
        schema: Dict,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        Generate a structured JSON response following a schema.

        Args:
            prompt: User prompt
            schema: JSON schema for response structure
            system_prompt: Optional system prompt
            temperature: Sampling temperature

        Returns:
            Parsed JSON response as dictionary
        """
        temp = temperature if temperature is not None else self.temperature

        # Build system prompt with schema
        schema_instruction = f"""You must respond with valid JSON following this schema:
{json.dumps(schema, indent=2)}

Only output the JSON, no other text."""

        full_system = schema_instruction
        if system_prompt:
            full_system = f"{system_prompt}\n\n{schema_instruction}"

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to parse JSON response: {e}")
            except Exception as e:
                if self._is_rate_limit(e):
                    return self._handle_rate_limit(
                        e, lambda fb: fb.generate_with_schema(prompt, schema, system_prompt, temperature)
                    )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API call failed: {e}")

    def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
        question: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness of answer to context using LLM.

        Args:
            answer: Generated answer to evaluate
            context: Retrieved context
            question: Optional question for context

        Returns:
            Dictionary with faithfulness score and reasoning
        """
        system_prompt = """You are an expert evaluator assessing the faithfulness of AI-generated answers.
Faithfulness measures whether the answer is fully supported by the provided context.
Be strict: only claims directly supported by the context should be considered faithful."""

        prompt = f"""Evaluate the faithfulness of the following answer based on the context.

Context:
{context}

{"Question: " + question if question else ""}

Answer to evaluate:
{answer}

Analyze the answer and determine:
1. Extract all factual claims from the answer
2. For each claim, determine if it is supported by the context
3. Calculate the faithfulness score as (supported claims / total claims)

Respond with JSON containing:
- "claims": list of claims extracted
- "supported_claims": list of claims supported by context
- "unsupported_claims": list of claims not supported
- "faithfulness_score": float between 0 and 1
- "reasoning": brief explanation"""

        schema = {
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"type": "string"}},
                "supported_claims": {"type": "array", "items": {"type": "string"}},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "faithfulness_score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"}
            },
            "required": ["faithfulness_score"]
        }

        return self.generate_with_schema(prompt, schema, system_prompt)

    def evaluate_relevancy(
        self,
        answer: str,
        question: str
    ) -> Dict[str, float]:
        """
        Evaluate answer relevancy to question using LLM.

        Args:
            answer: Generated answer
            question: Original question

        Returns:
            Dictionary with relevancy score and reasoning
        """
        system_prompt = """You are an expert evaluator assessing answer relevancy.
Relevancy measures how well the answer addresses the question asked.
Consider completeness, directness, and whether the answer provides what was asked."""

        prompt = f"""Evaluate how relevant the following answer is to the question.

Question:
{question}

Answer:
{answer}

Score the relevancy from 0 to 1:
- 1.0: Perfectly relevant, directly and completely answers the question
- 0.7-0.9: Mostly relevant, answers the main question with minor gaps
- 0.4-0.6: Partially relevant, addresses some aspects but misses key points
- 0.1-0.3: Marginally relevant, tangentially related
- 0.0: Completely irrelevant

Respond with JSON containing:
- "relevancy_score": float between 0 and 1
- "reasoning": brief explanation of the score"""

        schema = {
            "type": "object",
            "properties": {
                "relevancy_score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"}
            },
            "required": ["relevancy_score"]
        }

        return self.generate_with_schema(prompt, schema, system_prompt)

    def evaluate_context_precision(
        self,
        question: str,
        context: str
    ) -> Dict[str, float]:
        """
        Evaluate context precision - how relevant the retrieved context is.

        Args:
            question: Original question
            context: Retrieved context passages

        Returns:
            Dictionary with precision score
        """
        system_prompt = """You are an expert evaluator assessing retrieval quality.
Context precision measures what fraction of retrieved passages are relevant to answering the question."""

        prompt = f"""Evaluate the precision of the following retrieved context for the question.

Question:
{question}

Retrieved Context:
{context}

For each passage/section in the context, determine if it helps answer the question.
Calculate precision as (relevant passages / total passages).

Respond with JSON containing:
- "total_passages": number of distinct passages
- "relevant_passages": number of passages relevant to the question
- "precision_score": float between 0 and 1
- "reasoning": brief explanation"""

        schema = {
            "type": "object",
            "properties": {
                "total_passages": {"type": "integer"},
                "relevant_passages": {"type": "integer"},
                "precision_score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"}
            },
            "required": ["precision_score"]
        }

        return self.generate_with_schema(prompt, schema, system_prompt)

    def evaluate_coherence(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Evaluate text coherence using LLM.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with coherence score
        """
        system_prompt = """You are an expert evaluator assessing text quality.
Coherence measures how well-structured, logical, and easy to follow the text is."""

        prompt = f"""Evaluate the coherence of the following text.

Text:
{text}

Consider:
- Logical flow of ideas
- Clear structure
- Smooth transitions
- Consistency of style

Score from 0 to 1 where 1 is perfectly coherent.

Respond with JSON:
- "coherence_score": float between 0 and 1
- "reasoning": brief explanation"""

        schema = {
            "type": "object",
            "properties": {
                "coherence_score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"}
            },
            "required": ["coherence_score"]
        }

        return self.generate_with_schema(prompt, schema, system_prompt)

    def evaluate_consistency(
        self,
        answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Evaluate consistency between answer and context.

        Args:
            answer: Generated answer
            context: Source context

        Returns:
            Dictionary with consistency score
        """
        system_prompt = """You are an expert evaluator assessing factual consistency.
Consistency measures whether the answer contains any contradictions with the source context."""

        prompt = f"""Evaluate the consistency of the answer with the context.

Context:
{context}

Answer:
{answer}

Check for any contradictions, conflicting facts, or inconsistencies.
A score of 1 means fully consistent, 0 means major contradictions.

Respond with JSON:
- "consistency_score": float between 0 and 1
- "contradictions": list of any contradictions found
- "reasoning": brief explanation"""

        schema = {
            "type": "object",
            "properties": {
                "consistency_score": {"type": "number", "minimum": 0, "maximum": 1},
                "contradictions": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"}
            },
            "required": ["consistency_score"]
        }

        return self.generate_with_schema(prompt, schema, system_prompt)

    def evaluate_fluency(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Evaluate text fluency using LLM.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with fluency score
        """
        system_prompt = """You are an expert evaluator assessing text quality.
Fluency measures grammatical correctness, natural language flow, and readability."""

        prompt = f"""Evaluate the fluency of the following text.

Text:
{text}

Consider:
- Grammar and syntax
- Natural word choice
- Readability
- Professional tone

Score from 0 to 1 where 1 is perfectly fluent.

Respond with JSON:
- "fluency_score": float between 0 and 1
- "issues": list of any fluency issues
- "reasoning": brief explanation"""

        schema = {
            "type": "object",
            "properties": {
                "fluency_score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"}
            },
            "required": ["fluency_score"]
        }

        return self.generate_with_schema(prompt, schema, system_prompt)


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing without API calls.
    Uses heuristic approximations.
    """

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.call_count += 1
        return "Mock response"

    def generate_with_schema(self, prompt: str, schema: Dict, temperature: float = 0.0) -> Dict:
        self.call_count += 1
        # Return mock scores
        return {
            "faithfulness_score": 0.75,
            "relevancy_score": 0.80,
            "coherence_score": 0.85,
            "consistency_score": 0.78,
            "fluency_score": 0.90,
            "reasoning": "Mock evaluation"
        }


def load_tokens(token_path: Optional[str] = None) -> Dict:
    """
    Load all tokens from secrets/token.txt.

    Returns:
        Dictionary of key=value pairs from the token file.
    """
    if token_path is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "secrets", "token.txt"),
            os.path.join(os.getcwd(), "secrets", "token.txt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                token_path = c
                break

    tokens = {}
    if token_path and os.path.exists(token_path):
        with open(token_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    tokens[k.strip()] = v.strip()
    return tokens


def _build_client(name, tokens, key_field, model_field, url_field, default_model, default_url):
    """Try to build one OpenAIClient from token config. Returns None on failure."""
    api_key = tokens.get(key_field)
    if not api_key:
        return None
    model = tokens.get(model_field, default_model)
    base_url = tokens.get(url_field, default_url)
    try:
        client = OpenAIClient(model=model, api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        print(f"  Warning: Could not initialize {name}: {e}")
        return None


def get_llm_client(
    tokens: Optional[Dict] = None,
    use_mock: bool = False
) -> BaseLLMClient:
    """
    Build an LLM client with automatic DeepSeek -> Groq -> Gemini fallback.

    Reads provider configs from secrets/token.txt. Creates clients in
    priority order and chains them: if the primary hits rate limits
    (429/503), calls are transparently routed to the next provider.

    Args:
        tokens: Pre-loaded token dict (loads from file if None)
        use_mock: If True, returns mock client for testing

    Returns:
        LLM client instance with fallback chain
    """
    if use_mock:
        return MockLLMClient()

    if tokens is None:
        tokens = load_tokens()

    # Build clients in fallback order (last = innermost, no fallback of its own)
    providers = [
        ("DeepSeek", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL",
         "deepseek-chat", "https://api.deepseek.com/v1"),
        ("Groq", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL",
         "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
        ("Gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_BASE_URL",
         "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ]

    clients = []
    for name, key_f, model_f, url_f, def_model, def_url in providers:
        c = _build_client(name, tokens, key_f, model_f, url_f, def_model, def_url)
        if c:
            clients.append((name, c))

    if not clients:
        raise ValueError(
            "No API keys found in secrets/token.txt. "
            "Add DEEPSEEK_API_KEY, GROQ_API_KEY, and/or GEMINI_API_KEY."
        )

    # Chain fallbacks: last client has no fallback, each earlier one points to the next
    for i in range(len(clients) - 1, 0, -1):
        clients[i - 1][1]._fallback_client = clients[i][1]

    # Print chain
    names = [n for n, _ in clients]
    print(f"  Primary provider: {names[0]} ({clients[0][1].model})")
    if len(names) > 1:
        print(f"  Fallback chain: {' -> '.join(names[1:])} ({', '.join(c.model for _, c in clients[1:])})")

    return clients[0][1]
