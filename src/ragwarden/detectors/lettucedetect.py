# SPDX-License-Identifier: Apache-2.0
"""LettuceDetect Tier-1 adapter (Build Spec Section 7.3, item 3).

Wraps the LettuceDetect ModernBERT span-level detector (MIT). LettuceDetect
flags spans of an answer that are not grounded in the context; RagWarden feeds
it one claim at a time as the "answer" and treats any flagged span as evidence
the claim is UNSUPPORTED (it does not distinguish contradiction).

Requires ``pip install 'ragwarden[lettucedetect]'``.
"""

from __future__ import annotations

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.detectors.base import BaseDetector, require

__all__ = ["DEFAULT_MODEL_PATH", "LettuceDetectAdapter"]

DEFAULT_MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


class LettuceDetectAdapter(BaseDetector):
    tier = 1
    name = "lettucedetect"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        method: str = "transformer",
        unsupported_char_fraction: float = 0.5,
    ) -> None:
        module = require(
            "lettucedetect.models.inference", extra="lettucedetect", package="lettucedetect"
        )
        self.model_path = model_path
        self.unsupported_char_fraction = unsupported_char_fraction
        self._detector = module.HallucinationDetector(method=method, model_path=model_path)

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        if not evidence:
            return ClaimVerdict(
                claim, ClaimStatus.UNSUPPORTED, 0.0, 1, explanation="no evidence provided"
            )
        spans = self._detector.predict(
            context=[c.text for c in evidence],
            question="",
            answer=claim,
            output_format="spans",
        )
        if not spans:
            return ClaimVerdict(
                claim,
                ClaimStatus.SUPPORTED,
                0.9,
                1,
                supporting_evidence=[c.source_id for c in evidence if c.source_id][:1],
                explanation="LettuceDetect: no unsupported span",
            )
        flagged_chars = sum(max(0, s.get("end", 0) - s.get("start", 0)) for s in spans)
        frac = flagged_chars / max(1, len(claim))
        conf = max((float(s.get("confidence", 0.0)) for s in spans), default=0.5)
        status = (
            ClaimStatus.UNSUPPORTED
            if frac >= self.unsupported_char_fraction
            else ClaimStatus.PARTIALLY_SUPPORTED
        )
        return ClaimVerdict(
            claim,
            status,
            round(conf, 4),
            1,
            explanation=f"LettuceDetect flagged {frac:.0%} of the claim as unsupported",
        )
