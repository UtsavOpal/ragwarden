# SPDX-License-Identifier: Apache-2.0
"""Phase 5: Tier 2 wired into gate() — unresolved Tier-1 claims escalate,
consistency sampling runs, cost is visible in the stage trace."""

from __future__ import annotations

import pytest

from ragwarden import gate
from ragwarden.contracts import ClaimStatus as CS
from ragwarden.contracts import ClaimVerdict, GateAction
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context
from ragwarden.policy import Policy

pytestmark = pytest.mark.integration

CTX = Context(
    query="When was the Eiffel Tower completed?",
    chunks=[Chunk("The Eiffel Tower was completed in 1889.", 0.9, "d1")],
    retrieval_method="hybrid",
)
ANSWER = Answer(text="The Eiffel Tower was completed in 1889.")


def _low_conf_detector() -> ScriptedStubDetector:
    # Always returns a low-confidence SUPPORTED -> Tier 1 escalates it.
    return ScriptedStubDetector(lambda c, e: ClaimVerdict(c, CS.SUPPORTED, 0.3, 1))


def test_tier2_runs_and_records_cost() -> None:
    pol = Policy.default()
    pol.tier2.consistency_samples = 3
    pol.tier2.cost_per_sample_usd = 0.001
    calls = {"n": 0}

    def gen(_q: str) -> str:
        calls["n"] += 1
        return "The Eiffel Tower was completed in 1889."

    result = gate(CTX, ANSWER, policy=pol, detectors=[_low_conf_detector()], generate_fn=gen)
    assert calls["n"] == 3
    tier2_trace = next(t for t in result.stage_trace if t.stage_name == "tier2")
    assert tier2_trace.cost_estimate_usd == pytest.approx(0.003)
    assert result.claim_verdicts[0].resolved_at_tier == 2
    assert result.action is GateAction.ALLOW


def test_tier2_skipped_without_generate_fn() -> None:
    pol = Policy.default()
    pol.tier2.consistency_samples = 3
    result = gate(CTX, ANSWER, policy=pol, detectors=[_low_conf_detector()])
    assert "Tier 2 skipped (no generate_fn)" in result.explanation
    assert all(t.stage_name != "tier2" for t in result.stage_trace)
