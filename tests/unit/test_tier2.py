# SPDX-License-Identifier: Apache-2.0
"""Phase 5: Tier 2 uncertainty quantification (offline, scripted)."""

from __future__ import annotations

from ragwarden.cascade.tier2_uncertainty import Tier2Config, run_tier2
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors.stub import KeywordStubDetector


def _unresolved(text: str) -> ClaimVerdict:
    return ClaimVerdict(text, ClaimStatus.UNRESOLVED, 0.5, 1, explanation="escalated from Tier 1")


CLAIM = "The Eiffel Tower was completed in 1889."


def test_consistent_samples_resolve_supported() -> None:
    calls = {"n": 0}

    def gen(_query: str) -> str:
        calls["n"] += 1
        return "The Eiffel Tower was completed in 1889 for the World's Fair."

    out = run_tier2(
        [_unresolved(CLAIM)],
        "when was the eiffel tower completed",
        generate_fn=gen,
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=3, cost_per_sample_usd=0.002),
    )
    assert calls["n"] == 3
    assert out.samples_drawn == 3
    assert out.cost_estimate_usd == 0.006
    assert len(out.resolved) == 1
    assert out.resolved[0].status is ClaimStatus.SUPPORTED
    assert out.resolved[0].resolved_at_tier == 2


def test_inconsistent_samples_resolve_unsupported() -> None:
    answers = iter(
        [
            "Bananas are yellow and rich in potassium.",
            "The recipe needs two cups of flour and one egg.",
            "Photosynthesis converts sunlight into chemical energy.",
        ]
    )

    out = run_tier2(
        [_unresolved(CLAIM)],
        "q",
        generate_fn=lambda _q: next(answers),
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=3),
    )
    assert out.resolved and out.resolved[0].status is ClaimStatus.UNSUPPORTED


def test_disabled_passes_through() -> None:
    out = run_tier2(
        [_unresolved(CLAIM)],
        "q",
        generate_fn=lambda _q: "x",
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=0),
    )
    assert out.resolved == []
    assert len(out.unresolved) == 1
    assert out.samples_drawn == 0


def test_no_generate_fn_passes_through() -> None:
    out = run_tier2(
        [_unresolved(CLAIM)],
        "q",
        generate_fn=None,
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=3),
    )
    assert len(out.unresolved) == 1
