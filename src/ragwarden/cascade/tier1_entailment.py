# SPDX-License-Identifier: Apache-2.0
"""Tier 1 — claim-vs-evidence entailment (Build Spec Section 7.3).

Runs one or more :class:`~ragwarden.detectors.base.Detector` implementations over
the decomposed claims. The tier that does the real "is this supported by
evidence" work for the majority of claims.

Claims whose verdict confidence is below ``confidence_threshold`` are returned as
*unresolved* so the orchestrator can escalate them to Tier 2 rather than
force-resolving on a shaky signal.

**Fault isolation.** A detector can fail — OOM, a corrupted checkpoint, a remote
model-serving call that times out. A single failing detector never crashes
``gate()``; it is caught and handled per ``fail_mode`` (Spec Section 9.3):
``fail_closed`` (default) treats every claim that detector would have scored as
``UNRESOLVED`` — a failure poisons the ensemble result rather than silently
degrading it; ``fail_open`` drops the failed detector and ensembles over the
survivors. If every detector fails, claims are always ``UNRESOLVED`` regardless
of ``fail_mode`` — RagWarden never fabricates a verdict from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.detectors.base import Detector
from ragwarden.errors import CallableError, CallableTimeoutError
from ragwarden.resilience import call_with_timeout, safe_call

__all__ = ["Tier1Outcome", "ensemble_verdicts", "run_tier1"]


@dataclass
class Tier1Outcome:
    resolved: list[ClaimVerdict] = field(default_factory=list)
    unresolved: list[ClaimVerdict] = field(default_factory=list)  # low-confidence, escalate
    detector_errors: list[tuple[str, str]] = field(default_factory=list)  # (name, message)

    @property
    def all_verdicts(self) -> list[ClaimVerdict]:
        return self.resolved + self.unresolved


def _severity(status: ClaimStatus) -> int:
    order = {
        ClaimStatus.CONTRADICTED: 4,
        ClaimStatus.UNSUPPORTED: 3,
        ClaimStatus.UNRESOLVED: 2,
        ClaimStatus.PARTIALLY_SUPPORTED: 1,
        ClaimStatus.SUPPORTED: 0,
    }
    return order[status]


def ensemble_verdicts(verdicts: list[ClaimVerdict], strategy: str) -> ClaimVerdict:
    """Combine multiple detectors' verdicts for one claim."""
    if len(verdicts) == 1:
        return verdicts[0]
    if strategy == "max_confidence":
        return max(verdicts, key=lambda v: v.confidence)
    if strategy == "majority":
        tally: dict[ClaimStatus, list[ClaimVerdict]] = {}
        for v in verdicts:
            tally.setdefault(v.status, []).append(v)
        # most votes; ties broken toward the more severe status, then confidence
        best_status = max(
            tally,
            key=lambda s: (len(tally[s]), _severity(s), max(x.confidence for x in tally[s])),
        )
        group = tally[best_status]
        winner = max(group, key=lambda v: v.confidence)
        evidence = sorted({e for v in group for e in v.supporting_evidence})
        return ClaimVerdict(
            winner.claim_text,
            best_status,
            winner.confidence,
            1,
            supporting_evidence=evidence,
            explanation="; ".join(f"{v.explanation}" for v in verdicts if v.explanation),
        )
    if strategy == "most_severe":
        return max(verdicts, key=lambda v: (_severity(v.status), v.confidence))
    raise ValueError(f"unknown ensemble strategy: {strategy!r}")


def _run_one_detector(
    detector: Detector,
    claims: list[str],
    evidence: list[RetrievedChunk],
    *,
    call_timeout_s: float | None,
) -> list[ClaimVerdict]:
    """Run one detector, isolated. Raises CallableError/CallableTimeoutError —
    never the detector's own exception type — on failure."""
    name = getattr(detector, "name", type(detector).__name__)
    what = f"Tier 1 detector {name!r}"
    if call_timeout_s is not None:
        verdicts = call_with_timeout(
            detector.check_batch, claims, evidence, timeout_s=call_timeout_s, what=what
        )
    else:
        verdicts = safe_call(detector.check_batch, claims, evidence, what=what)
    if len(verdicts) != len(claims):
        raise CallableError(
            what,
            ValueError(f"returned {len(verdicts)} verdicts for {len(claims)} claims"),
        )
    return verdicts


def run_tier1(
    claims: list[str],
    evidence: list[RetrievedChunk],
    *,
    detectors: list[Detector],
    confidence_threshold: float = 0.85,
    ensemble_strategy: str = "max_confidence",
    fail_mode: str = "fail_closed",
    call_timeout_s: float | None = None,
) -> Tier1Outcome:
    if not detectors:
        raise ValueError("run_tier1 requires at least one detector")

    per_detector: list[list[ClaimVerdict]] = []
    errors: list[tuple[str, str]] = []
    for detector in detectors:
        name = getattr(detector, "name", type(detector).__name__)
        try:
            per_detector.append(
                _run_one_detector(detector, claims, evidence, call_timeout_s=call_timeout_s)
            )
        except (CallableTimeoutError, CallableError) as exc:
            errors.append((name, str(exc)))

    outcome = Tier1Outcome(detector_errors=errors)

    if not claims:
        return outcome

    if not per_detector or (fail_mode == "fail_closed" and errors):
        reason = (
            f"all {len(detectors)} Tier 1 detector(s) failed"
            if not per_detector
            else f"fail_closed: detector {errors[0][0]!r} failed ({errors[0][1]})"
        )
        outcome.unresolved = [
            ClaimVerdict(claim, ClaimStatus.UNRESOLVED, 0.0, 1, explanation=f"Tier 1: {reason}")
            for claim in claims
        ]
        return outcome

    for i, claim in enumerate(claims):
        verdict = ensemble_verdicts([pd[i] for pd in per_detector], ensemble_strategy)
        low_confidence = verdict.confidence < confidence_threshold
        if low_confidence and verdict.status is not ClaimStatus.CONTRADICTED:
            escalated = ClaimVerdict(
                claim,
                ClaimStatus.UNRESOLVED,
                verdict.confidence,
                1,
                supporting_evidence=verdict.supporting_evidence,
                explanation=(
                    f"Tier 1 below confidence {confidence_threshold:.2f} "
                    f"({verdict.status.value} p={verdict.confidence:.2f}); escalating"
                ),
            )
            outcome.unresolved.append(escalated)
        else:
            outcome.resolved.append(verdict)
    return outcome
