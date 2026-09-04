# SPDX-License-Identifier: Apache-2.0
"""Production hardening: a broken generate_fn / detector must never crash Tier 2."""

from __future__ import annotations

import time

from ragwarden.cascade.tier2_uncertainty import Tier2Config, run_tier2
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors.stub import KeywordStubDetector, ScriptedStubDetector

CLAIM = ClaimVerdict("The Eiffel Tower was completed in 1889.", ClaimStatus.UNRESOLVED, 0.5, 1)


def test_generate_fn_raising_every_time_stays_unresolved_not_crashed() -> None:
    def boom(_q: str) -> str:
        raise RuntimeError("provider down")

    out = run_tier2(
        [CLAIM],
        "q",
        generate_fn=boom,
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=3),
    )
    assert out.resolved == []
    assert out.unresolved[0].status is ClaimStatus.UNRESOLVED
    assert out.samples_failed == 3
    assert out.samples_drawn == 0


def test_generate_fn_partial_failures_still_produce_a_verdict() -> None:
    calls = {"n": 0}

    def flaky(_q: str) -> str:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("transient")
        return "The Eiffel Tower was completed in 1889 for the World's Fair."

    out = run_tier2(
        [CLAIM],
        "q",
        generate_fn=flaky,
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=4, agreement_resolve_threshold=0.5),
    )
    assert out.samples_failed == 2
    assert out.samples_drawn == 2
    assert out.resolved  # the 2 successful samples were enough to resolve


def test_hanging_generate_fn_is_bounded_by_call_timeout() -> None:
    def hangs(_q: str) -> str:
        time.sleep(2.0)
        return "x"

    t0 = time.perf_counter()
    out = run_tier2(
        [CLAIM],
        "q",
        generate_fn=hangs,
        detector=KeywordStubDetector(),
        config=Tier2Config(consistency_samples=2, call_timeout_s=0.1),
    )
    assert time.perf_counter() - t0 < 1.0
    assert out.samples_drawn == 0
    assert out.unresolved


def test_detector_check_raising_excludes_that_sample_not_the_request() -> None:
    def unstable_check(claim: str, evidence: list) -> ClaimVerdict:
        if "sample-0" in (evidence[0].source_id if evidence else ""):
            raise RuntimeError("detector hiccup")
        return ClaimVerdict(claim, ClaimStatus.SUPPORTED, 0.9, 1)

    out = run_tier2(
        [CLAIM],
        "q",
        generate_fn=lambda _q: "The Eiffel Tower was completed in 1889.",
        detector=ScriptedStubDetector(unstable_check),
        config=Tier2Config(consistency_samples=3, agreement_resolve_threshold=0.5),
    )
    # one sample's check failed, the other two still resolved the claim
    assert out.resolved
    assert out.resolved[0].status is ClaimStatus.SUPPORTED
