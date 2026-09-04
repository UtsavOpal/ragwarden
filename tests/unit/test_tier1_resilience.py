# SPDX-License-Identifier: Apache-2.0
"""Production hardening: a broken Tier-1 detector must never crash the cascade."""

from __future__ import annotations

import time

from ragwarden.cascade.tier1_entailment import run_tier1
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors.base import BaseDetector
from ragwarden.detectors.stub import KeywordStubDetector, ScriptedStubDetector


def _chunk(text: str = "The Eiffel Tower was completed in 1889.") -> object:
    from ragwarden.models import Chunk

    return Chunk(text, 0.9, "d1")


class _RaisingDetector(BaseDetector):
    tier = 1
    name = "raising"

    def check_batch(self, claims: list[str], evidence: list) -> list[ClaimVerdict]:
        raise RuntimeError("model server 500")


class _HangingDetector(BaseDetector):
    tier = 1
    name = "hanging"

    def check_batch(self, claims: list[str], evidence: list) -> list[ClaimVerdict]:
        time.sleep(2.0)
        return [ClaimVerdict(c, ClaimStatus.SUPPORTED, 0.9, 1) for c in claims]


class _WrongLengthDetector(BaseDetector):
    tier = 1
    name = "wrong_length"

    def check_batch(self, claims: list[str], evidence: list) -> list[ClaimVerdict]:
        return [ClaimVerdict(claims[0], ClaimStatus.SUPPORTED, 0.9, 1)]  # always 1, regardless of N


def test_single_detector_raises_fail_closed_marks_unresolved() -> None:
    out = run_tier1(
        ["claim a", "claim b"], [_chunk()], detectors=[_RaisingDetector()], fail_mode="fail_closed"
    )
    assert out.resolved == []
    assert len(out.unresolved) == 2
    assert all(v.status is ClaimStatus.UNRESOLVED for v in out.unresolved)
    assert out.detector_errors and out.detector_errors[0][0] == "raising"


def test_single_detector_raises_fail_open_uses_survivor() -> None:
    good = ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.SUPPORTED, 0.97, 1))
    out = run_tier1(
        ["claim a"],
        [_chunk()],
        detectors=[_RaisingDetector(), good],
        fail_mode="fail_open",
    )
    assert out.detector_errors and out.detector_errors[0][0] == "raising"
    assert len(out.resolved) == 1
    assert out.resolved[0].status is ClaimStatus.SUPPORTED


def test_all_detectors_fail_stays_unresolved_regardless_of_fail_mode() -> None:
    for mode in ("fail_closed", "fail_open"):
        out = run_tier1(["c"], [_chunk()], detectors=[_RaisingDetector()], fail_mode=mode)
        assert out.resolved == []
        assert out.unresolved[0].status is ClaimStatus.UNRESOLVED


def test_hanging_detector_is_bounded_by_call_timeout() -> None:
    t0 = time.perf_counter()
    out = run_tier1(
        ["c"],
        [_chunk()],
        detectors=[_HangingDetector()],
        call_timeout_s=0.1,
        fail_mode="fail_closed",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0  # bounded by the 0.1s timeout, not the 2.0s sleep
    assert out.unresolved[0].status is ClaimStatus.UNRESOLVED
    assert out.detector_errors


def test_wrong_length_response_is_treated_as_a_failure_not_an_indexerror() -> None:
    out = run_tier1(["c1", "c2", "c3"], [_chunk()], detectors=[_WrongLengthDetector()])
    assert out.detector_errors
    assert len(out.unresolved) == 3


def test_healthy_run_is_unaffected() -> None:
    out = run_tier1(
        ["The Eiffel Tower was completed in 1889."],
        [_chunk()],
        detectors=[KeywordStubDetector()],
    )
    assert not out.detector_errors
    assert out.resolved and out.resolved[0].status is ClaimStatus.SUPPORTED
