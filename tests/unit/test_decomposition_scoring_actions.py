# SPDX-License-Identifier: Apache-2.0
"""Phase 1 unit coverage: decomposition, scoring, actions."""

from __future__ import annotations

import pytest

from ragwarden.actions import apply_action
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.decomposition import SentenceSplitDecomposer, get_decomposer
from ragwarden.scoring import (
    PassThroughScoring,
    SeverityWeightedScoring,
    get_scoring_strategy,
)


def _v(status: ClaimStatus, text: str = "claim", conf: float = 0.9) -> ClaimVerdict:
    return ClaimVerdict(text, status, conf, resolved_at_tier=1)


# --- decomposition -------------------------------------------------------------
def test_sentence_split_basic() -> None:
    d = SentenceSplitDecomposer()
    claims = d.decompose("The tower opened in 1889. It is in Paris. It is tall.")
    assert len(claims) == 3
    assert claims[0].startswith("The tower opened")


def test_sentence_split_empty() -> None:
    assert SentenceSplitDecomposer().decompose("   ") == []


def test_llm_decomposer_is_open_decision() -> None:
    with pytest.raises(NotImplementedError):
        get_decomposer("llm")


def test_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        get_decomposer("nope")


# --- scoring -----------------------------------------------------------------
def test_passthrough_scoring() -> None:
    s = PassThroughScoring()
    assert s.score([]) == 1.0
    assert s.score([_v(ClaimStatus.SUPPORTED)]) == 1.0
    assert s.score([_v(ClaimStatus.CONTRADICTED)]) == 0.0


def test_severity_weighted_ordering() -> None:
    s = SeverityWeightedScoring()
    all_supported = s.score([_v(ClaimStatus.SUPPORTED), _v(ClaimStatus.SUPPORTED)])
    one_unsupported = s.score([_v(ClaimStatus.SUPPORTED), _v(ClaimStatus.UNSUPPORTED)])
    one_contradicted = s.score([_v(ClaimStatus.SUPPORTED), _v(ClaimStatus.CONTRADICTED)])
    assert all_supported == pytest.approx(1.0)
    assert all_supported > one_unsupported > one_contradicted


def test_severity_weighted_weights_are_configurable() -> None:
    lenient = get_scoring_strategy(
        "severity_weighted",
        status_credit={
            ClaimStatus.SUPPORTED: 1.0,
            ClaimStatus.PARTIALLY_SUPPORTED: 0.9,
            ClaimStatus.UNSUPPORTED: 0.7,
            ClaimStatus.CONTRADICTED: 0.3,
            ClaimStatus.UNRESOLVED: 0.7,
        },
    )
    assert lenient.score([_v(ClaimStatus.UNSUPPORTED)]) == pytest.approx(0.7)


def test_unresolved_scored_as_unsupported_by_default() -> None:
    s = SeverityWeightedScoring()
    assert s.score([_v(ClaimStatus.UNRESOLVED)]) == s.score([_v(ClaimStatus.UNSUPPORTED)])


# --- actions ---------------------------------------------------------------
def test_allow_keeps_citation_trail() -> None:
    v = ClaimVerdict("c", ClaimStatus.SUPPORTED, 0.95, 1, supporting_evidence=["doc-1"])
    out = apply_action(GateAction.ALLOW, original_text="c", verdicts=[v], abstain_message="idk")
    assert out.output_text == "c"
    assert out.payload["citations"]["c"] == ["doc-1"]


def test_redact_removes_bad_claims() -> None:
    verdicts = [
        ClaimVerdict("Paris is the capital of France.", ClaimStatus.SUPPORTED, 0.95, 1),
        ClaimVerdict("It has a population of 90 million.", ClaimStatus.CONTRADICTED, 0.9, 1),
    ]
    out = apply_action(
        GateAction.REDACT_CLAIMS,
        original_text="Paris is the capital of France. It has a population of 90 million.",
        verdicts=verdicts,
        abstain_message="idk",
    )
    assert out.action is GateAction.REDACT_CLAIMS
    assert "90 million" not in out.output_text
    assert "capital of France" in out.output_text


def test_redact_everything_falls_back_to_abstain() -> None:
    verdicts = [ClaimVerdict("Wrong one.", ClaimStatus.CONTRADICTED, 0.9, 1)]
    out = apply_action(
        GateAction.REDACT_CLAIMS,
        original_text="Wrong one.",
        verdicts=verdicts,
        abstain_message="idk",
    )
    assert out.action is GateAction.ABSTAIN
    assert out.fell_back is True


def test_retry_exhausted_falls_back_to_abstain() -> None:
    out = apply_action(
        GateAction.RETRY,
        original_text="x",
        verdicts=[],
        abstain_message="idk",
        retries_used=1,
        max_retries=1,
    )
    assert out.action is GateAction.ABSTAIN
    assert out.fell_back is True


def test_escalate_produces_review_payload() -> None:
    v = _v(ClaimStatus.CONTRADICTED)
    out = apply_action(GateAction.ESCALATE, original_text="x", verdicts=[v], abstain_message="idk")
    assert out.output_text == "x"
    assert out.payload["status"] == "pending_human_review"
    assert out.payload["review_item"]["claim_verdicts"][0]["status"] == "contradicted"
