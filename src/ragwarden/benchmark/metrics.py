# SPDX-License-Identifier: Apache-2.0
"""Evaluation metrics for the benchmark harness.

Computed directly (no scikit-learn dependency). "Positive" = hallucinated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ClassificationMetrics", "response_level_metrics", "span_token_metrics"]


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    support_positive: int
    support_negative: int
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "support_positive": self.support_positive,
            "support_negative": self.support_negative,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def _prf(tp: int, fp: int, fn: int, tn: int) -> ClassificationMetrics:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return ClassificationMetrics(
        precision,
        recall,
        f1,
        accuracy,
        support_positive=tp + fn,
        support_negative=fp + tn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def response_level_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> ClassificationMetrics:
    """Example-level P/R/F1 where positive = 'response contains a hallucination'."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    return _prf(tp, fp, fn, tn)


def span_token_metrics(
    true_spans: Sequence[tuple[int, int]],
    pred_spans: Sequence[tuple[int, int]],
    text_length: int,
) -> ClassificationMetrics:
    """Character/token-index span metrics via per-position set overlap, matching
    the RAGTruth span-level evaluation style."""
    true_pos: set[int] = set()
    for s, e in true_spans:
        true_pos |= set(range(max(0, s), min(text_length, e)))
    pred_pos: set[int] = set()
    for s, e in pred_spans:
        pred_pos |= set(range(max(0, s), min(text_length, e)))
    all_pos = set(range(text_length))
    tp = len(true_pos & pred_pos)
    fp = len(pred_pos - true_pos)
    fn = len(true_pos - pred_pos)
    tn = len(all_pos - true_pos - pred_pos)
    return _prf(tp, fp, fn, tn)
