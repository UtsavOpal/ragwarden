# SPDX-License-Identifier: Apache-2.0
"""Phase 2 acceptance: gate() runs real claim verification (via stub detector),
a supported answer is ALLOWed, a fabricated one is not."""

from __future__ import annotations

import pytest

from ragwarden import Policy, gate
from ragwarden.contracts import ClaimStatus, GateAction
from ragwarden.detectors.stub import KeywordStubDetector
from ragwarden.models import Answer, Chunk, Context

pytestmark = pytest.mark.integration

CONTEXT = Context(
    query="Tell me about the Eiffel Tower.",
    chunks=[
        Chunk("The Eiffel Tower was completed in 1889 for the World's Fair.", 0.93, "doc-1"),
        Chunk("The Eiffel Tower stands 330 metres tall in Paris, France.", 0.80, "doc-2"),
    ],
    retrieval_method="hybrid",
)


# Stub confidence is bounded by lexical overlap; tune the escalation threshold
# so confident stub verdicts resolve at Tier 1 rather than waiting for Tier 2.
_STUB_POLICY = Policy.default()
_STUB_POLICY.tier1.confidence_threshold = 0.5


def test_supported_answer_allowed() -> None:
    answer = Answer(
        text="The Eiffel Tower was completed in 1889. The Eiffel Tower stands in Paris France."
    )
    result = gate(CONTEXT, answer, policy=_STUB_POLICY, detectors=[KeywordStubDetector()])
    assert result.action is GateAction.ALLOW
    assert result.claim_verdicts
    assert all(v.resolved_at_tier == 1 for v in result.claim_verdicts)
    assert any(v.status is ClaimStatus.SUPPORTED for v in result.claim_verdicts)
    assert "tier1" in [t.stage_name for t in result.stage_trace]


def test_fabricated_answer_not_allowed() -> None:
    answer = Answer(
        text=(
            "The Eiffel Tower was completed in 1889. "
            "The tower was designed by Leonardo da Vinci and painted bright orange."
        )
    )
    result = gate(CONTEXT, answer, policy=_STUB_POLICY, detectors=[KeywordStubDetector()])
    assert result.action in {GateAction.REDACT_CLAIMS, GateAction.RETRY, GateAction.ABSTAIN}
    assert result.reliability_score < 0.9


def test_tier1_unavailable_falls_back_to_tier0_with_warning() -> None:
    answer = Answer(text="The Eiffel Tower was completed in 1889.")
    pol = Policy.default()
    pol.tier1.detector = "lettucedetect"  # extra deliberately not installed
    with pytest.warns(UserWarning, match="falls back to Tier 0"):
        result = gate(CONTEXT, answer, policy=pol)
    assert "Tier-0-only" in result.explanation
    assert result.claim_verdicts == []
