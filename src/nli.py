"""Utilities for DeBERTa-v3-large MNLI inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict


@dataclass
class NLIResult:
    entailment: float
    contradiction: float
    neutral: float


class DebertaMNLIScorer:
    """Thin wrapper around a DeBERTa-v3-large MNLI sequence classifier."""

    def __init__(self, model_name: str = "microsoft/deberta-v3-large-mnli"):
        self.model_name = model_name
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self._torch = torch

        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entailment_id = next(i for i, lbl in id2label.items() if "entail" in lbl)
        self.contradiction_id = next(i for i, lbl in id2label.items() if "contrad" in lbl)
        self.neutral_id = next(i for i, lbl in id2label.items() if "neutral" in lbl)

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        if not premise or not hypothesis:
            return NLIResult(entailment=0.0, contradiction=0.0, neutral=1.0)

        with self._torch.no_grad():
            encoded = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            logits = self.model(**encoded).logits[0]
            probs = self._torch.softmax(logits, dim=-1)

        return NLIResult(
            entailment=float(probs[self.entailment_id].item()),
            contradiction=float(probs[self.contradiction_id].item()),
            neutral=float(probs[self.neutral_id].item()),
        )


@lru_cache(maxsize=1)
def get_deberta_mnli_scorer() -> DebertaMNLIScorer:
    return DebertaMNLIScorer()


def safe_nli_score(premise: str, hypothesis: str) -> Dict[str, float]:
    """Run DeBERTa-v3-large MNLI; returns neutral default if unavailable."""
    try:
        scorer = get_deberta_mnli_scorer()
        result = scorer.score(premise=premise, hypothesis=hypothesis)
        return {
            "entailment": result.entailment,
            "contradiction": result.contradiction,
            "neutral": result.neutral,
            "model": scorer.model_name,
        }
    except Exception:
        return {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0,
            "model": "unavailable",
        }
