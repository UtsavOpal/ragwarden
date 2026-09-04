# SPDX-License-Identifier: Apache-2.0
"""Vectara HHEM Tier-1 detector (Build Spec Section 7.3, item 1).

Wraps ``vectara/hallucination_evaluation_model`` (HHEM-2.1-Open, Apache-2.0), a
small cross-encoder that outputs a single factual-consistency probability in
``[0, 1]`` per (premise, hypothesis) pair — ``1`` = hypothesis fully supported by
the premise. It does *not* distinguish "contradicted" from "unsupported", so a
low score maps to UNSUPPORTED (not CONTRADICTED).

Runs one forward pass per (chunk, claim) pair and keeps the best score.
Requires ``pip install 'ragwarden[hhem]'``.
"""

from __future__ import annotations

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.detectors.base import BaseDetector, require

__all__ = ["DEFAULT_MODEL_ID", "HHEMDetector"]

DEFAULT_MODEL_ID = "vectara/hallucination_evaluation_model"


class HHEMDetector(BaseDetector):
    tier = 1
    name = "hhem"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        support_threshold: float = 0.5,
        strong_support_threshold: float = 0.5,
    ) -> None:
        require("torch", extra="hhem")
        transformers = require("transformers", extra="hhem")
        self.model_id = model_id
        self.support_threshold = support_threshold
        self.strong_support_threshold = strong_support_threshold
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model.eval()

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        if not evidence:
            return ClaimVerdict(
                claim, ClaimStatus.UNSUPPORTED, 0.0, 1, explanation="no evidence provided"
            )
        pairs = [(chunk.text, claim) for chunk in evidence]
        scores = [float(s) for s in self._model.predict(pairs)]
        best = max(range(len(scores)), key=lambda i: scores[i])
        score = scores[best]
        src = evidence[best].source_id

        if score >= self.support_threshold:
            return ClaimVerdict(
                claim,
                ClaimStatus.SUPPORTED,
                round(score, 4),
                1,
                supporting_evidence=[src] if src else [],
                explanation=f"HHEM consistency {score:.2f}",
            )
        # Low consistency: unsupported. Confidence in "unsupported" grows as the
        # score falls; a mid-range score stays low-confidence and escalates.
        return ClaimVerdict(
            claim,
            ClaimStatus.UNSUPPORTED,
            round(1.0 - score, 4),
            1,
            explanation=f"HHEM consistency {score:.2f} (below support threshold)",
        )
