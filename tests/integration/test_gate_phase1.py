# SPDX-License-Identifier: Apache-2.0
"""Phase 1 acceptance: a hand-built context/generation produces a real GateResult,
and Tier 0 can drive ABSTAIN on empty / low-score / off-topic retrieval."""

from __future__ import annotations

import pytest

from ragwarden import Policy, gate
from ragwarden.contracts import GateAction, GateResult
from ragwarden.models import Answer, Chunk, Context

pytestmark = pytest.mark.integration

# Phase 1 exercises the Tier-0-only path; disable Tier 1 so no detector is needed.
_T0_ONLY = Policy.default()
_T0_ONLY.tier1.enabled = False


def test_gate_returns_gateresult(grounded_context, grounded_answer) -> None:
    result = gate(grounded_context, grounded_answer, policy=_T0_ONLY)
    assert isinstance(result, GateResult)
    assert 0.0 <= result.reliability_score <= 1.0
    assert [t.stage_name for t in result.stage_trace][:2] == ["decomposition", "tier0"]
    assert result.explanation


def test_clean_retrieval_allows(grounded_context, grounded_answer) -> None:
    result = gate(grounded_context, grounded_answer, policy=_T0_ONLY)
    assert result.action is GateAction.ALLOW
    assert result.output_text == grounded_answer.text


def test_empty_retrieval_abstains(empty_context) -> None:
    result = gate(empty_context, Answer(text="Atlantis' capital is Poseidonis."))
    assert result.action is GateAction.ABSTAIN
    assert "Short-circuited at Tier 0" in result.explanation
    assert result.output_text != "Atlantis' capital is Poseidonis."


def test_off_topic_retrieval_abstains_under_strict_policy() -> None:
    ctx = Context(
        query="What is the boiling point of water at sea level?",
        chunks=[Chunk(text="The mitochondria is the powerhouse of the cell.", score=0.9)],
        retrieval_method="hybrid",
    )
    pol = Policy.default()
    pol.tier0.min_overlap = 0.5
    result = gate(ctx, Answer(text="100 degrees Celsius."), policy=pol)
    assert result.action is GateAction.ABSTAIN


def test_low_score_retrieval_is_dampened() -> None:
    ctx = Context(
        query="when was the eiffel tower completed",
        chunks=[Chunk(text="the eiffel tower was completed in 1889", score=0.15)],
        retrieval_method="hybrid",
    )
    pol = Policy.default()
    pol.tier1.enabled = False
    pol.tier0.min_top_score = 0.5
    result = gate(ctx, Answer(text="1889."), policy=pol)
    # Tier-0 flag caps the score below `allow`, so a clean-looking answer no
    # longer sails through.
    assert result.action is not GateAction.ALLOW
    assert "low_top_score" in result.explanation
