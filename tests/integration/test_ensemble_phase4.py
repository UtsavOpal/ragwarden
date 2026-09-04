# SPDX-License-Identifier: Apache-2.0
"""Phase 4: multi-detector ensembling through the gate (offline stubs)."""

from __future__ import annotations

import pytest

from ragwarden import gate
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context
from ragwarden.policy import Policy

pytestmark = pytest.mark.integration

CTX = Context(
    query="q",
    chunks=[Chunk("Paris is the capital of France.", 0.9, "d1")],
    retrieval_method="hybrid",
)
ANSWER = Answer(text="Paris is the capital of France.")


def _fixed(status: ClaimStatus, conf: float) -> ScriptedStubDetector:
    return ScriptedStubDetector(lambda c, e: ClaimVerdict(c, status, conf, 1))


def test_max_confidence_ensemble_lets_the_strong_contradiction_win() -> None:
    pol = Policy.default()
    pol.tier1.ensemble_strategy = "max_confidence"
    result = gate(
        CTX,
        ANSWER,
        policy=pol,
        detectors=[_fixed(ClaimStatus.SUPPORTED, 0.6), _fixed(ClaimStatus.CONTRADICTED, 0.97)],
    )
    assert result.claim_verdicts[0].status is ClaimStatus.CONTRADICTED
    assert result.action in {GateAction.ABSTAIN, GateAction.REDACT_CLAIMS, GateAction.RETRY}


def test_majority_ensemble_agrees() -> None:
    pol = Policy.default()
    pol.tier1.ensemble_strategy = "majority"
    pol.tier1.confidence_threshold = 0.5
    result = gate(
        CTX,
        ANSWER,
        policy=pol,
        detectors=[
            _fixed(ClaimStatus.SUPPORTED, 0.8),
            _fixed(ClaimStatus.SUPPORTED, 0.9),
            _fixed(ClaimStatus.UNSUPPORTED, 0.6),
        ],
    )
    assert result.claim_verdicts[0].status is ClaimStatus.SUPPORTED
    assert result.action is GateAction.ALLOW
