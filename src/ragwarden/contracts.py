# SPDX-License-Identifier: Apache-2.0
"""Frozen core contracts for RagWarden (Build Spec Section 6).

These are the load-bearing interfaces. Host-application types satisfy them
structurally via :class:`typing.Protocol` — nothing needs to subclass a
RagWarden base class. Once v0.1 ships, changing the shapes here is a breaking
change (major version bump past v1.0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "ClaimStatus",
    "ClaimVerdict",
    "GateAction",
    "GateResult",
    "Generation",
    "RetrievalContext",
    "RetrievedChunk",
    "StageTrace",
]


@runtime_checkable
class RetrievedChunk(Protocol):
    """A single piece of retrieved evidence.

    ``metadata`` is a deliberately open dict — source authority, freshness,
    permissions, per-method retrieval scores (``bm25_score`` / ``knn_score``),
    document structure, and any other pipeline-specific field live here without
    the core contract needing to know about them.
    """

    text: str
    score: float
    source_id: str
    metadata: dict


@runtime_checkable
class RetrievalContext(Protocol):
    """The query plus the evidence a RAG pipeline retrieved for it."""

    query: str
    chunks: Sequence[RetrievedChunk]
    retrieval_method: str  # e.g. "bm25", "knn", "hybrid" — informational only


@runtime_checkable
class Generation(Protocol):
    """The answer produced by the host application's generator.

    ``token_logprobs`` is optional: most hosted APIs (and many self-hosted
    setups) do not expose it. No Tier-2 logic may assume it is present.
    """

    text: str
    token_logprobs: Sequence[float] | None


class ClaimStatus(str, Enum):
    """The verdict for one atomic claim after the cascade resolves it."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"  # exhausted cascade budget without a confident verdict


@dataclass(frozen=True)
class ClaimVerdict:
    """Per-claim result. ``resolved_at_tier`` is always set — it makes the
    cascade's cost-efficiency auditable ("N% of claims resolved at Tier 1").

    Frozen: gate internals must not be mutated by host applications."""

    claim_text: str
    status: ClaimStatus
    confidence: float  # 0.0-1.0
    resolved_at_tier: int  # 0-3, which tier produced the final verdict
    supporting_evidence: list[str] = field(default_factory=list)  # source_ids
    explanation: str = ""


class GateAction(str, Enum):
    """The decision the policy engine hands back to the host application."""

    ALLOW = "allow"
    REDACT_CLAIMS = "redact_claims"
    RETRY = "retry"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class StageTrace:
    """Latency / cost accounting for one cascade stage."""

    stage_name: str
    latency_ms: float
    claims_processed: int
    cost_estimate_usd: float | None = None


@dataclass(frozen=True)
class GateResult:
    """Everything the host application needs to act on and to explain the gate's
    decision. Frozen — treat it as an immutable record of one decision."""

    action: GateAction
    reliability_score: float  # 0.0-1.0 composite score
    claim_verdicts: list[ClaimVerdict]
    output_text: str  # original, redacted, or abstention message
    stage_trace: list[StageTrace]
    explanation: str  # human-readable summary of the decision
    # Action-specific structured data the host consumes:
    #   RETRY    -> {"retry": True, "retries_remaining": int, "reason": str}
    #   ESCALATE -> {"status": "pending_human_review", "review_item": {...}}
    #   ALLOW    -> {"citations": {claim: [source_id, ...]}}
    #   REDACT   -> {"removed_claims": [...]}
    # Optional and additive (default empty) — does not break existing consumers.
    action_payload: dict = field(default_factory=dict)
