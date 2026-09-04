# SPDX-License-Identifier: Apache-2.0
"""Composite scoring (Build Spec Section 8).

Combine per-claim verdicts into one 0.0-1.0 reliability score. The default
strategy is severity-weighted; weights are constructor parameters (never
hardcoded) so the same class serves a lenient internal-tools deployment and a
strict regulated-industry deployment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ragwarden.contracts import ClaimStatus, ClaimVerdict

__all__ = [
    "PassThroughScoring",
    "ScoringStrategy",
    "SeverityWeightedScoring",
    "get_scoring_strategy",
]


class ScoringStrategy(Protocol):
    name: str

    def score(self, verdicts: list[ClaimVerdict]) -> float:
        """Return a 0.0-1.0 composite reliability score."""
        ...


class PassThroughScoring:
    """v0.1 Phase-1 strategy: no claim-level verdicts exist yet (Tier 0 only).

    Returns ``1.0`` when there is nothing disqualifying and ``0.0`` once any
    verdict is negative. Real nuance arrives with :class:`SeverityWeightedScoring`
    in Phase 2.
    """

    name = "pass_through"

    def score(self, verdicts: list[ClaimVerdict]) -> float:
        if not verdicts:
            return 1.0
        bad = {ClaimStatus.CONTRADICTED, ClaimStatus.UNSUPPORTED, ClaimStatus.UNRESOLVED}
        return 0.0 if any(v.status in bad for v in verdicts) else 1.0


# Per-status "reliability credit" in [0, 1]. 1.0 = fully trustworthy claim.
_DEFAULT_STATUS_CREDIT: dict[ClaimStatus, float] = {
    ClaimStatus.SUPPORTED: 1.0,
    ClaimStatus.PARTIALLY_SUPPORTED: 0.6,
    ClaimStatus.UNSUPPORTED: 0.25,
    ClaimStatus.CONTRADICTED: 0.0,
    ClaimStatus.UNRESOLVED: 0.25,  # treated as UNSUPPORTED for scoring (configurable)
}


@dataclass
class SeverityWeightedScoring:
    """Default strategy: CONTRADICTED penalizes more than UNSUPPORTED, which
    penalizes more than PARTIALLY_SUPPORTED. UNRESOLVED is treated as
    UNSUPPORTED for scoring purposes unless remapped.

    The score is a confidence-weighted mean of per-claim reliability credit,
    where a claim's weight is its verdict confidence (low-confidence verdicts
    move the score less).
    """

    name: str = "severity_weighted"
    status_credit: dict[ClaimStatus, float] = field(
        default_factory=lambda: dict(_DEFAULT_STATUS_CREDIT)
    )
    unresolved_as: ClaimStatus = ClaimStatus.UNSUPPORTED
    min_claim_weight: float = 0.15  # a near-zero-confidence verdict still counts a little

    def score(self, verdicts: list[ClaimVerdict]) -> float:
        if not verdicts:
            return 1.0
        total_weight = 0.0
        weighted_credit = 0.0
        for v in verdicts:
            status = self.unresolved_as if v.status is ClaimStatus.UNRESOLVED else v.status
            credit = self.status_credit.get(status, 0.0)
            weight = max(self.min_claim_weight, min(1.0, float(v.confidence)))
            total_weight += weight
            weighted_credit += weight * credit
        if total_weight == 0.0:
            return 1.0
        return max(0.0, min(1.0, weighted_credit / total_weight))


_REGISTRY: dict[str, Callable[..., ScoringStrategy]] = {
    PassThroughScoring.name: PassThroughScoring,
    SeverityWeightedScoring.name: SeverityWeightedScoring,
}


def get_scoring_strategy(name: str, **kwargs: object) -> ScoringStrategy:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown scoring strategy: {name!r}") from None
    return factory(**kwargs)
