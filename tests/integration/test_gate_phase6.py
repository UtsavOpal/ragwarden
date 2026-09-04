# SPDX-License-Identifier: Apache-2.0
"""Phase 6: full cascade — a deliberately ambiguous claim flows Tier 1 -> 2 -> 3,
budget caps hold, escalation-rate metric is emitted."""

from __future__ import annotations

import pytest

from ragwarden import gate
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context
from ragwarden.policy import Policy

pytestmark = pytest.mark.integration

CTX = Context(
    query="Tell me about the Eiffel Tower.",
    chunks=[Chunk("The Eiffel Tower was completed in 1889 in Paris.", 0.9, "d1")],
    retrieval_method="hybrid",
)


def _always_uncertain() -> ScriptedStubDetector:
    # low confidence -> Tier 1 escalates; on Tier 2 samples it also stays split
    return ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.UNSUPPORTED, 0.4, 1))


def _split_by_sample() -> ScriptedStubDetector:
    """Low-confidence at Tier 1 (escalates); on Tier 2 exactly half the samples
    'support' the claim (a genuine split -> escalates to Tier 3)."""

    def verdict(claim: str, evidence: list) -> ClaimVerdict:
        src = evidence[0].source_id if evidence else ""
        if src.startswith("tier2-sample-"):
            n = int(src.rsplit("-", 1)[1])
            status = ClaimStatus.SUPPORTED if n % 2 == 0 else ClaimStatus.UNSUPPORTED
            return ClaimVerdict(claim, status, 0.8, 1)
        return ClaimVerdict(claim, ClaimStatus.UNSUPPORTED, 0.4, 1)

    return ScriptedStubDetector(verdict)


def test_claim_flows_through_all_three_tiers() -> None:
    pol = Policy.default()
    pol.tier1.confidence_threshold = 0.85
    pol.tier2.consistency_samples = 4  # -> 2/4 support = 0.5 agreement (a split)
    pol.tier2.agreement_resolve_threshold = 0.75

    judged = {"n": 0}

    def judge(_prompt: str) -> str:
        judged["n"] += 1
        return "REASONING: supported by the passage.\nVERDICT: supported\nCONFIDENCE: 0.9"

    result = gate(
        CTX,
        Answer(text="The Eiffel Tower was completed in 1889."),
        policy=pol,
        detectors=[_split_by_sample()],
        generate_fn=lambda _q: "The Eiffel Tower was completed in 1889.",
        judge_fn=judge,
    )
    stages = [t.stage_name for t in result.stage_trace]
    assert "tier1" in stages and "tier2" in stages and "tier3" in stages
    assert judged["n"] >= 1
    assert any(v.resolved_at_tier == 3 for v in result.claim_verdicts)
    assert "escalation rate" in result.explanation


def test_tier3_budget_caps_hold_under_many_ambiguous_claims() -> None:
    pol = Policy.default()
    pol.tier2.enabled = False
    pol.tier3.max_claims_per_request = 2

    long_answer = " ".join(f"Fact number {i} about the tower is stated here." for i in range(6))
    calls = {"n": 0}

    def judge(_p: str) -> str:
        calls["n"] += 1
        return "VERDICT: unsupported\nCONFIDENCE: 0.7"

    result = gate(
        CTX,
        Answer(text=long_answer),
        policy=pol,
        detectors=[_always_uncertain()],
        judge_fn=judge,
    )
    assert calls["n"] == 2  # hard cap respected even with 6 ambiguous claims
    assert "budget exhausted" in result.explanation
    assert result.action in {GateAction.ABSTAIN, GateAction.REDACT_CLAIMS, GateAction.RETRY}


def test_tier3_skipped_without_judge_fn() -> None:
    pol = Policy.default()
    pol.tier2.enabled = False
    result = gate(
        CTX,
        Answer(text="The Eiffel Tower was completed in 1889."),
        policy=pol,
        detectors=[_always_uncertain()],
    )
    assert "Tier 3 skipped (no judge_fn)" in result.explanation
    assert all(t.stage_name != "tier3" for t in result.stage_trace)
