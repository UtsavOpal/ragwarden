# SPDX-License-Identifier: Apache-2.0
"""Action execution (Build Spec Section 10).

Turns a policy decision into the final ``output_text`` plus the structured
payload the host application consumes. RagWarden never performs retrieval or
generation itself — RETRY and ESCALATE return signals for the host to act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction

__all__ = ["ActionOutcome", "apply_action"]

# Claims with these statuses are dropped by REDACT_CLAIMS.
_REDACTABLE = {ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED, ClaimStatus.UNRESOLVED}

_MIN_COHERENT_WORDS = 4
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NORMALIZE = re.compile(r"\s+")


@dataclass
class ActionOutcome:
    action: GateAction
    output_text: str
    payload: dict = field(default_factory=dict)
    fell_back: bool = False


def _norm(text: str) -> str:
    return _NORMALIZE.sub(" ", text).strip().rstrip(".!?").lower()


def _redact(original: str, verdicts: list[ClaimVerdict]) -> tuple[str, list[str]]:
    """Remove sentences of ``original`` that correspond to a redactable claim,
    preserving the order and any connective text between kept sentences."""
    bad = {_norm(v.claim_text) for v in verdicts if v.status in _REDACTABLE}
    if not bad:
        return original.strip(), []

    removed: list[str] = []
    kept_sentences: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(original.strip()):
        s = sentence.strip()
        if not s:
            continue
        norm = _norm(s)
        if norm in bad or any(b and (b in norm or norm in b) for b in bad):
            removed.append(s)
        else:
            kept_sentences.append(s)
    return " ".join(kept_sentences).strip(), removed


def apply_action(
    action: GateAction,
    *,
    original_text: str,
    verdicts: list[ClaimVerdict],
    abstain_message: str,
    reason: str = "",
    reliability_score: float | None = None,
    retries_used: int = 0,
    max_retries: int = 1,
) -> ActionOutcome:
    if action is GateAction.ALLOW:
        return ActionOutcome(
            action,
            original_text,
            payload={
                "citations": {
                    v.claim_text: v.supporting_evidence for v in verdicts if v.supporting_evidence
                }
            },
        )

    if action is GateAction.ABSTAIN:
        return ActionOutcome(action, abstain_message, payload={"reason": reason})

    if action is GateAction.REDACT_CLAIMS:
        redacted, removed = _redact(original_text, verdicts)
        # Incoherent / empty redaction -> fall back to ABSTAIN (Spec Section 10).
        if not removed or len(redacted.split()) < _MIN_COHERENT_WORDS:
            return ActionOutcome(
                GateAction.ABSTAIN,
                abstain_message,
                payload={
                    "reason": "redaction left no coherent text",
                    "original_reason": reason,
                    "removed_claims": removed,
                },
                fell_back=True,
            )
        return ActionOutcome(action, redacted, payload={"removed_claims": removed})

    if action is GateAction.RETRY:
        remaining = max_retries - retries_used
        if remaining <= 0:
            return ActionOutcome(
                GateAction.ABSTAIN,
                abstain_message,
                payload={"reason": "retry budget exhausted", "original_reason": reason},
                fell_back=True,
            )
        return ActionOutcome(
            action,
            original_text,
            payload={
                "retry": True,
                "retries_used": retries_used,
                "retries_remaining": remaining,
                "reason": reason,
                "weak_claims": [v.claim_text for v in verdicts if v.status in _REDACTABLE],
            },
        )

    if action is GateAction.ESCALATE:
        return ActionOutcome(
            action,
            original_text,
            payload={
                "status": "pending_human_review",
                "reason": reason,
                "reliability_score": reliability_score,
                "review_item": {
                    "answer": original_text,
                    "claim_verdicts": [
                        {
                            "claim": v.claim_text,
                            "status": v.status.value,
                            "confidence": v.confidence,
                            "resolved_at_tier": v.resolved_at_tier,
                            "evidence": v.supporting_evidence,
                            "explanation": v.explanation,
                        }
                        for v in verdicts
                    ],
                },
            },
        )

    raise ValueError(f"unhandled action: {action!r}")  # pragma: no cover
