# SPDX-License-Identifier: Apache-2.0
"""Deterministic stub detectors — no ML, no downloads.

Useful for testing a RagWarden integration end-to-end without pulling a model,
and for RagWarden's own offline test suite. Not a real grounding checker.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.detectors.base import BaseDetector

__all__ = ["KeywordStubDetector", "ScriptedStubDetector"]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class KeywordStubDetector(BaseDetector):
    """Verdict by lexical overlap between the claim and the evidence.

    High overlap -> SUPPORTED, some overlap -> PARTIALLY_SUPPORTED, negation
    mismatch -> CONTRADICTED, no overlap -> UNSUPPORTED. Confidence is the
    overlap fraction. Purely a placeholder for real entailment models.
    """

    tier = 1
    name = "keyword_stub"

    def __init__(self, support_threshold: float = 0.6, partial_threshold: float = 0.3) -> None:
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        claim_tokens = _tokens(claim)
        if not claim_tokens:
            return ClaimVerdict(claim, ClaimStatus.UNSUPPORTED, 0.0, 1, explanation="empty claim")
        evidence_tokens: set[str] = set()
        best_source = ""
        for chunk in evidence:
            et = _tokens(chunk.text)
            if len(claim_tokens & et) > len(claim_tokens & evidence_tokens):
                best_source = chunk.source_id
            evidence_tokens |= et
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)

        claim_negated = bool({"not", "no", "never", "n't"} & claim_tokens)
        evidence_negated = bool({"not", "no", "never", "n't"} & evidence_tokens)
        if overlap >= self.partial_threshold and claim_negated != evidence_negated:
            return ClaimVerdict(
                claim,
                ClaimStatus.CONTRADICTED,
                overlap,
                1,
                supporting_evidence=[best_source] if best_source else [],
                explanation="negation mismatch vs evidence",
            )
        if overlap >= self.support_threshold:
            status = ClaimStatus.SUPPORTED
        elif overlap >= self.partial_threshold:
            status = ClaimStatus.PARTIALLY_SUPPORTED
        else:
            status = ClaimStatus.UNSUPPORTED
        return ClaimVerdict(
            claim,
            status,
            round(overlap, 4),
            1,
            supporting_evidence=[best_source]
            if best_source and status != ClaimStatus.UNSUPPORTED
            else [],
            explanation=f"lexical overlap {overlap:.2f}",
        )


class ScriptedStubDetector(BaseDetector):
    """Returns whatever the supplied function decides. Full control for tests."""

    tier = 1
    name = "scripted_stub"

    def __init__(self, verdict_fn: Callable[[str, list[RetrievedChunk]], ClaimVerdict]) -> None:
        self._fn = verdict_fn

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        return self._fn(claim, evidence)
