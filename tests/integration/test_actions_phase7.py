# SPDX-License-Identifier: Apache-2.0
"""Phase 7: each action type end-to-end through gate(), including edge cases."""

from __future__ import annotations

import pytest

from ragwarden import GateRequest, Policy, gate
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context

pytestmark = pytest.mark.integration

CTX = Context(
    query="Tell me about the Eiffel Tower.",
    chunks=[
        Chunk("The Eiffel Tower was completed in 1889.", 0.9, "d1", {"authority": "high"}),
        Chunk("It stands on the Champ de Mars in Paris.", 0.8, "d2"),
    ],
    retrieval_method="hybrid",
)


def _by_claim(mapping: dict[str, tuple[ClaimStatus, float]]):
    """Detector that returns a scripted (status, confidence) per claim substring."""

    def verdict(claim: str, evidence: list) -> ClaimVerdict:
        for key, (status, conf) in mapping.items():
            if key in claim:
                expl = "[authority=high]" if status is ClaimStatus.CONTRADICTED else ""
                src = [evidence[0].source_id] if evidence else []
                return ClaimVerdict(
                    claim, status, conf, 1, supporting_evidence=src, explanation=expl
                )
        return ClaimVerdict(claim, ClaimStatus.SUPPORTED, 0.95, 1)

    return ScriptedStubDetector(verdict)


def test_allow_returns_original_with_citations() -> None:
    ans = Answer(text="The Eiffel Tower was completed in 1889. It is in Paris.")
    det = _by_claim({"1889": (ClaimStatus.SUPPORTED, 0.97), "Paris": (ClaimStatus.SUPPORTED, 0.95)})
    r = gate(CTX, ans, detectors=[det])
    assert r.action is GateAction.ALLOW
    assert r.output_text == ans.text
    assert r.action_payload["citations"]  # non-empty citation trail


def test_redact_drops_the_bad_sentence_keeps_the_rest() -> None:
    ans = Answer(
        text=(
            "The Eiffel Tower was completed in 1889. "
            "The tower is fifty metres shorter than the Empire State Building's antenna. "
            "It stands in Paris."
        )
    )
    det = _by_claim(
        {
            "1889": (ClaimStatus.SUPPORTED, 0.97),
            "fifty metres shorter": (ClaimStatus.UNSUPPORTED, 0.9),
            "Paris": (ClaimStatus.SUPPORTED, 0.95),
        }
    )
    pol = Policy.default()
    pol.escalate_to_human.enabled = False
    r = gate(CTX, ans, policy=pol, detectors=[det])
    assert r.action is GateAction.REDACT_CLAIMS
    assert "fifty metres shorter" not in r.output_text
    assert "1889" in r.output_text and "Paris" in r.output_text
    assert r.action_payload["removed_claims"]


def test_redacting_every_claim_falls_back_to_abstain() -> None:
    ans = Answer(text="Fact one is wrong. Fact two is also wrong.")
    det = _by_claim({"one": (ClaimStatus.UNSUPPORTED, 0.9), "two": (ClaimStatus.UNSUPPORTED, 0.9)})
    pol = Policy.default()
    pol.escalate_to_human.enabled = False
    # Put the (low) score into the REDACT band so REDACT is chosen — then every
    # sentence is redactable, so it must fall back to ABSTAIN rather than ship "".
    pol.thresholds.redact_claims = 0.2
    pol.thresholds.retry = 0.1
    r = gate(CTX, ans, policy=pol, detectors=[det])
    assert r.action is GateAction.ABSTAIN
    assert "fell back from redact_claims" in r.explanation
    assert r.action_payload.get("reason") == "redaction left no coherent text"


def test_retry_then_abstain_never_loops() -> None:
    ans = Answer(text="The tower opened in 1889. The tower is orange.")
    det = _by_claim(
        {"1889": (ClaimStatus.SUPPORTED, 0.9), "orange": (ClaimStatus.CONTRADICTED, 0.9)}
    )
    pol = Policy.default()
    pol.escalate_to_human.enabled = False  # otherwise the contradiction escalates
    pol.thresholds.retry = 0.4
    pol.thresholds.redact_claims = 0.55

    first = gate(
        CTX, ans, policy=pol, detectors=[det], request=GateRequest("req-1", retries_used=0)
    )
    assert first.action is GateAction.RETRY
    assert first.action_payload["retries_remaining"] == 1

    second = gate(
        CTX, ans, policy=pol, detectors=[det], request=GateRequest("req-1", retries_used=1)
    )
    assert second.action is GateAction.ABSTAIN  # budget exhausted -> forced, no loop


def test_abstain_uses_configured_template() -> None:
    ans = Answer(text="Everything here is fabricated nonsense.")
    det = _by_claim({"fabricated": (ClaimStatus.CONTRADICTED, 0.95)})
    pol = Policy.default()
    pol.escalate_to_human.enabled = False
    pol.abstain_message = "Sorry — I can't verify this."
    r = gate(CTX, ans, policy=pol, detectors=[det])
    assert r.action is GateAction.ABSTAIN
    assert r.output_text == "Sorry — I can't verify this."


def test_escalate_produces_review_payload() -> None:
    ans = Answer(text="The Eiffel Tower was demolished in 1950.")
    det = _by_claim({"demolished": (ClaimStatus.CONTRADICTED, 0.95)})
    r = gate(CTX, ans, detectors=[det])  # default policy: contradicted + high authority
    assert r.action is GateAction.ESCALATE
    assert r.output_text == ans.text  # original answer, flagged
    payload = r.action_payload
    assert payload["status"] == "pending_human_review"
    assert payload["review_item"]["claim_verdicts"][0]["status"] == "contradicted"
    assert payload["reliability_score"] is not None
