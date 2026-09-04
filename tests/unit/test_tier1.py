# SPDX-License-Identifier: Apache-2.0
"""Phase 2: Tier 1 entailment wiring, ensembling, escalation seam."""

from __future__ import annotations

import pytest

from ragwarden.cascade.tier1_entailment import ensemble_verdicts, run_tier1
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors.stub import KeywordStubDetector, ScriptedStubDetector
from ragwarden.models import Chunk

EVIDENCE = [
    Chunk("The Eiffel Tower was completed in 1889.", score=0.9, source_id="doc-1"),
    Chunk("It is located in Paris, France.", score=0.7, source_id="doc-2"),
]


def _v(status: ClaimStatus, conf: float, ev: list[str] | None = None) -> ClaimVerdict:
    return ClaimVerdict("c", status, conf, 1, supporting_evidence=ev or [])


def test_keyword_stub_supported_vs_unsupported() -> None:
    d = KeywordStubDetector()
    good = d.check("The Eiffel Tower was completed in 1889.", EVIDENCE)
    bad = d.check("The Great Wall of China is visible from orbit.", EVIDENCE)
    assert good.status is ClaimStatus.SUPPORTED
    assert bad.status is ClaimStatus.UNSUPPORTED


def test_run_tier1_escalates_low_confidence() -> None:
    scripted = ScriptedStubDetector(
        lambda claim, ev: _v(ClaimStatus.SUPPORTED, 0.4)  # below default 0.85
    )
    out = run_tier1(["a claim"], EVIDENCE, detectors=[scripted])
    assert out.resolved == []
    assert len(out.unresolved) == 1
    assert out.unresolved[0].status is ClaimStatus.UNRESOLVED


def test_run_tier1_keeps_confident_verdicts() -> None:
    scripted = ScriptedStubDetector(lambda claim, ev: _v(ClaimStatus.SUPPORTED, 0.95))
    out = run_tier1(["a claim"], EVIDENCE, detectors=[scripted])
    assert len(out.resolved) == 1
    assert out.unresolved == []


def test_run_tier1_requires_a_detector() -> None:
    with pytest.raises(ValueError):
        run_tier1(["a"], EVIDENCE, detectors=[])


def test_ensemble_max_confidence() -> None:
    combined = ensemble_verdicts(
        [_v(ClaimStatus.SUPPORTED, 0.6), _v(ClaimStatus.CONTRADICTED, 0.9)],
        "max_confidence",
    )
    assert combined.status is ClaimStatus.CONTRADICTED


def test_ensemble_majority_breaks_ties_toward_severity() -> None:
    combined = ensemble_verdicts(
        [
            _v(ClaimStatus.SUPPORTED, 0.9),
            _v(ClaimStatus.CONTRADICTED, 0.8),
        ],
        "majority",
    )
    assert combined.status is ClaimStatus.CONTRADICTED


def test_ensemble_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        ensemble_verdicts([_v(ClaimStatus.SUPPORTED, 0.9), _v(ClaimStatus.SUPPORTED, 0.9)], "nope")
