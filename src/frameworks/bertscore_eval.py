"""
BERTScore Evaluator
===================

Implementation of BERTScore metrics using the official `bert-score` library.

Reference: Zhang et al., "BERTScore: Evaluating Text Generation with BERT"
ICLR 2020, arXiv:1904.09675
"""

from typing import Dict, Optional
from . import HeuristicEvaluator


class BERTScoreEvaluator(HeuristicEvaluator):
    """BERTScore framework backed by the `bert-score` implementation."""

    def __init__(self, noise_std: float = 0.04, model_type: str = "microsoft/deberta-xlarge-mnli"):
        super().__init__(
            name="BERTScore",
            metrics=["precision", "recall", "f1"],
            noise_std=noise_std,
        )
        self.model_type = model_type
        self._scorer = None

    def evaluate(self, sample: Dict) -> Dict[str, float]:
        answer = sample.get("answer", "")
        context = sample.get("context", "")
        ground_truth = sample.get("ground_truth", {})

        if isinstance(ground_truth, dict):
            reference = " ".join(str(v) for v in ground_truth.values() if v)
        else:
            reference = str(ground_truth)

        if not reference:
            reference = context

        precision, recall, f1 = self._compute_bertscore(answer, reference)

        return {
            "precision": self.add_noise(precision),
            "recall": self.add_noise(recall),
            "f1": self.add_noise(f1),
        }


    def _get_scorer(self):
        if self._scorer is not None:
            return self._scorer
        try:
            from bert_score import BERTScorer
            self._scorer = BERTScorer(
                lang="en",
                model_type=self.model_type,
                rescale_with_baseline=True,
            )
        except Exception:
            self._scorer = None
        return self._scorer

    def _compute_bertscore(self, candidate: str, reference: str):
        if not candidate or not reference:
            return 0.0, 0.0, 0.0

        scorer = self._get_scorer()
        if scorer is not None:
            try:
                p, r, f = scorer.score([candidate], [reference])
                return float(p[0].item()), float(r[0].item()), float(f[0].item())
            except Exception:
                pass

        # Conservative fallback to lexical overlap for offline/limited envs.
        precision = self.word_overlap(candidate, reference)
        recall = self.coverage_score(reference, candidate)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1
