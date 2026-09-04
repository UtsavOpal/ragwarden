# SPDX-License-Identifier: Apache-2.0
"""Production hardening: startup warmup / readiness probe."""

from __future__ import annotations

from ragwarden.policy import Policy
from ragwarden.warmup import warmup


def test_warmup_reports_failure_for_missing_extra() -> None:
    pol = Policy.default()
    pol.tier1.detector = "lettucedetect"  # extra deliberately not installed
    report = warmup(pol)
    assert not report.ok
    assert report.results[0].detector_name == "lettucedetect"
    assert report.errors and "lettucedetect" in report.errors[0]


def test_warmup_reports_failure_for_unknown_detector() -> None:
    report = warmup(detector_names=["does-not-exist"])
    assert not report.ok
    assert "does-not-exist" in report.results[0].error


def test_warmup_dedupes_detector_names() -> None:
    pol = Policy.default()
    pol.tier1.detector = "keyword_stub"
    pol.tier1.ensemble = ["keyword_stub"]
    report = warmup(pol)
    assert len(report.results) == 1


def test_warmup_succeeds_for_a_healthy_detector() -> None:
    report = warmup(detector_names=["keyword_stub"])
    assert report.ok
    assert report.results[0].ok
    assert report.results[0].latency_ms >= 0
