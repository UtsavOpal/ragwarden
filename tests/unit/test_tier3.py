# SPDX-License-Identifier: Apache-2.0
"""Phase 6: Tier 3 LLM-as-judge (offline, scripted judge_fn)."""

from __future__ import annotations

import time

from ragwarden.cascade.tier3_judge import (
    Tier3Config,
    build_judge_prompt,
    run_tier3,
)
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.models import Chunk

EVIDENCE = [Chunk("The Eiffel Tower was completed in 1889.", 0.9, "d1")]


def _pending(text: str) -> ClaimVerdict:
    return ClaimVerdict(text, ClaimStatus.UNRESOLVED, 0.5, 2, explanation="from Tier 2")


def test_prompt_is_structured_g_eval_style() -> None:
    p = build_judge_prompt("claim x", EVIDENCE)
    assert "step by step" in p
    assert "VERDICT:" in p and "CONFIDENCE:" in p
    assert "The Eiffel Tower was completed in 1889." in p


def test_judge_resolves_supported() -> None:
    def judge(_prompt: str) -> str:
        return "REASONING: the evidence says exactly this.\nVERDICT: supported\nCONFIDENCE: 0.92"

    out = run_tier3([_pending("c")], EVIDENCE, judge_fn=judge, config=Tier3Config())
    assert out.claims_judged == 1
    assert out.resolved[0].status is ClaimStatus.SUPPORTED
    assert out.resolved[0].resolved_at_tier == 3
    assert out.resolved[0].confidence == 0.92


def test_judge_resolves_contradicted() -> None:
    out = run_tier3(
        [_pending("c")],
        EVIDENCE,
        judge_fn=lambda _p: "VERDICT: contradicted\nCONFIDENCE: 0.8",
        config=Tier3Config(),
    )
    assert out.resolved[0].status is ClaimStatus.CONTRADICTED


def test_unparseable_output_stays_unresolved() -> None:
    out = run_tier3(
        [_pending("c")],
        EVIDENCE,
        judge_fn=lambda _p: "I think it's probably fine?",
        config=Tier3Config(),
    )
    assert out.resolved == []
    assert out.unresolved[0].status is ClaimStatus.UNRESOLVED
    assert "could not be parsed" in out.unresolved[0].explanation


def test_judge_exception_does_not_crash() -> None:
    def boom(_p: str) -> str:
        raise RuntimeError("model down")

    out = run_tier3([_pending("c")], EVIDENCE, judge_fn=boom, config=Tier3Config())
    assert out.unresolved[0].status is ClaimStatus.UNRESOLVED
    assert "judge call failed" in out.unresolved[0].explanation


def test_claim_budget_cap() -> None:
    calls = {"n": 0}

    def judge(_p: str) -> str:
        calls["n"] += 1
        return "VERDICT: supported\nCONFIDENCE: 0.9"

    pending = [_pending(f"c{i}") for i in range(5)]
    out = run_tier3(pending, EVIDENCE, judge_fn=judge, config=Tier3Config(max_claims_per_request=2))
    assert calls["n"] == 2
    assert out.claims_judged == 2
    assert out.budget_exhausted is True
    assert len(out.resolved) == 2
    assert len(out.unresolved) == 3
    assert all("budget exhausted" in v.explanation for v in out.unresolved)


def test_latency_cap_via_deadline() -> None:
    def slow_judge(_p: str) -> str:
        time.sleep(0.02)
        return "VERDICT: supported\nCONFIDENCE: 0.9"

    pending = [_pending(f"c{i}") for i in range(10)]
    out = run_tier3(
        pending,
        EVIDENCE,
        judge_fn=slow_judge,
        config=Tier3Config(max_claims_per_request=10, max_latency_ms=30),
    )
    assert out.budget_exhausted is True
    assert out.claims_judged < 10


def test_single_hanging_call_is_bounded_by_call_timeout() -> None:
    """A judge_fn that hangs past call_timeout_s must not block the whole
    request even when max_latency_ms is generous."""

    def hangs(_prompt: str) -> str:
        time.sleep(2.0)
        return "VERDICT: supported\nCONFIDENCE: 0.9"

    t0 = time.perf_counter()
    out = run_tier3(
        [_pending("c")],
        EVIDENCE,
        judge_fn=hangs,
        config=Tier3Config(max_claims_per_request=1, max_latency_ms=10_000, call_timeout_s=0.1),
    )
    assert time.perf_counter() - t0 < 1.0
    assert out.unresolved[0].status is ClaimStatus.UNRESOLVED
    assert "judge call failed" in out.unresolved[0].explanation


def test_disabled_when_max_claims_zero() -> None:
    out = run_tier3(
        [_pending("c")],
        EVIDENCE,
        judge_fn=lambda _p: "VERDICT: supported\nCONFIDENCE: 1.0",
        config=Tier3Config(max_claims_per_request=0),
    )
    assert out.claims_judged == 0
    assert len(out.unresolved) == 1
