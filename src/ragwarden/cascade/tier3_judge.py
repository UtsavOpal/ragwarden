# SPDX-License-Identifier: Apache-2.0
"""Tier 3 — LLM-as-judge (Build Spec Section 7.5).

For the small remainder of claims still unresolved after Tiers 1-2. Uses
structured, G-Eval-style chain-of-thought scoring (reason step by step, then emit
a structured verdict) — measurably more reliable than a bare yes/no prompt.

RagWarden owns the prompt; the host supplies ``judge_fn(prompt: str) -> str`` (a
plain text-completion call to their LLM of choice). A hard per-request budget
(``max_claims_per_request``, ``max_latency_ms``) bounds worst-case cost/latency;
claims not reached before the budget is exhausted stay ``UNRESOLVED`` and the
policy engine decides what to do with them.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.errors import CallableError, CallableTimeoutError
from ragwarden.resilience import call_with_timeout

__all__ = ["JudgeFn", "Tier3Config", "Tier3Outcome", "build_judge_prompt", "run_tier3"]

JudgeFn = Callable[[str], str]

_VERDICT_LINE_RE = re.compile(r"verdict\b[^\n]*", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(
    r"confidence\s*[:=]?\s*([01](?:\.\d+)?|\.\d+|\d{1,3}\s*%)", re.IGNORECASE
)

# order matters: check the more specific / more severe phrases first
_STATUS_KEYWORDS: tuple[tuple[str, ClaimStatus], ...] = (
    ("partially supported", ClaimStatus.PARTIALLY_SUPPORTED),
    ("partially_supported", ClaimStatus.PARTIALLY_SUPPORTED),
    ("partial", ClaimStatus.PARTIALLY_SUPPORTED),
    ("not supported", ClaimStatus.UNSUPPORTED),
    ("not_supported", ClaimStatus.UNSUPPORTED),
    ("unsupported", ClaimStatus.UNSUPPORTED),
    ("no evidence", ClaimStatus.UNSUPPORTED),
    ("no_evidence", ClaimStatus.UNSUPPORTED),
    ("neutral", ClaimStatus.UNSUPPORTED),
    ("contradicted", ClaimStatus.CONTRADICTED),
    ("contradiction", ClaimStatus.CONTRADICTED),
    ("contradict", ClaimStatus.CONTRADICTED),
    ("refuted", ClaimStatus.CONTRADICTED),
    ("supported", ClaimStatus.SUPPORTED),
    ("support", ClaimStatus.SUPPORTED),
    ("entailed", ClaimStatus.SUPPORTED),
)


def _match_status(text: str) -> ClaimStatus | None:
    low = text.lower()
    for kw, status in _STATUS_KEYWORDS:
        if kw in low:
            return status
    return None


_PROMPT_TEMPLATE = """\
You are a meticulous fact-checker. Decide whether the CLAIM is supported by the EVIDENCE only \
(ignore any outside knowledge).

EVIDENCE:
{evidence}

CLAIM:
{claim}

Reason step by step:
1. State the specific assertion(s) the claim makes.
2. For each, quote the supporting or contradicting sentence from the EVIDENCE,
   or note that none exists.
3. Decide.

Then output these three lines exactly, nothing after them:
REASONING: <one or two sentences>
VERDICT: <supported | partially_supported | contradicted | unsupported>
CONFIDENCE: <number between 0.0 and 1.0>
"""


@dataclass
class Tier3Config:
    max_claims_per_request: int = 3
    max_latency_ms: int = 800
    cost_per_call_usd: float | None = None
    max_evidence_chars: int = 4000
    call_timeout_s: float | None = 15.0  # per judge_fn() call, on top of max_latency_ms


@dataclass
class Tier3Outcome:
    resolved: list[ClaimVerdict] = field(default_factory=list)
    unresolved: list[ClaimVerdict] = field(default_factory=list)
    claims_judged: int = 0
    budget_exhausted: bool = False
    cost_estimate_usd: float | None = None

    @property
    def all_verdicts(self) -> list[ClaimVerdict]:
        return self.resolved + self.unresolved


def build_judge_prompt(claim: str, evidence: list[RetrievedChunk], *, max_chars: int = 4000) -> str:
    joined = "\n\n".join(f"[{c.source_id or i}] {c.text}" for i, c in enumerate(evidence))
    return _PROMPT_TEMPLATE.format(evidence=joined[:max_chars], claim=claim)


def _parse_judge_output(text: str) -> tuple[ClaimStatus | None, float]:
    text = text or ""
    # Prefer a status keyword on the VERDICT line; fall back to the whole output.
    vline = _VERDICT_LINE_RE.search(text)
    status = _match_status(vline.group(0)) if vline else None
    if status is None:
        status = _match_status(text)

    confidence = 0.5
    cmatch = _CONFIDENCE_RE.search(text)
    if cmatch:
        raw = cmatch.group(1).strip()
        with contextlib.suppress(ValueError):
            value = float(raw[:-1].strip()) / 100.0 if raw.endswith("%") else float(raw)
            confidence = max(0.0, min(1.0, value))
    return status, confidence


def run_tier3(
    claims: list[ClaimVerdict],
    evidence: list[RetrievedChunk],
    *,
    judge_fn: JudgeFn | None,
    config: Tier3Config,
    deadline: float | None = None,
) -> Tier3Outcome:
    """Judge the still-unresolved ``claims``. ``deadline`` is a ``time.perf_counter``
    value shared across the whole request; None means only the per-tier caps apply."""
    outcome = Tier3Outcome()
    if judge_fn is None or config.max_claims_per_request <= 0 or not claims:
        outcome.unresolved = list(claims)
        return outcome

    tier_deadline = time.perf_counter() + config.max_latency_ms / 1000.0
    if deadline is not None:
        tier_deadline = min(tier_deadline, deadline)

    for claim in claims:
        over_claim_budget = outcome.claims_judged >= config.max_claims_per_request
        over_time_budget = time.perf_counter() >= tier_deadline
        if over_claim_budget or over_time_budget:
            outcome.budget_exhausted = True
            outcome.unresolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.UNRESOLVED,
                    claim.confidence,
                    3,
                    supporting_evidence=claim.supporting_evidence,
                    explanation=(
                        "Tier 3 budget exhausted before this claim "
                        f"({'claim cap' if over_claim_budget else 'latency cap'})"
                    ),
                )
            )
            continue

        prompt = build_judge_prompt(claim.claim_text, evidence, max_chars=config.max_evidence_chars)
        # Cap this call's own budget at whatever time remains in the tier/request
        # deadline as well as config.call_timeout_s, so one slow call can't blow
        # past the tier's own latency cap.
        remaining_s = max(0.01, tier_deadline - time.perf_counter())
        call_budget = (
            min(config.call_timeout_s, remaining_s)
            if config.call_timeout_s is not None
            else remaining_s
        )
        try:
            raw = call_with_timeout(judge_fn, prompt, timeout_s=call_budget, what="Tier 3 judge_fn")
        except (CallableTimeoutError, CallableError) as exc:
            outcome.unresolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.UNRESOLVED,
                    claim.confidence,
                    3,
                    explanation=f"Tier 3 judge call failed: {exc}",
                )
            )
            outcome.claims_judged += 1
            continue

        outcome.claims_judged += 1
        status, confidence = _parse_judge_output(raw)
        if status is None:
            outcome.unresolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.UNRESOLVED,
                    claim.confidence,
                    3,
                    explanation="Tier 3 judge output could not be parsed",
                )
            )
        else:
            outcome.resolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    status,
                    round(confidence, 4),
                    3,
                    supporting_evidence=claim.supporting_evidence,
                    explanation=f"Tier 3 LLM judge: {status.value} (confidence {confidence:.2f})",
                )
            )

    if config.cost_per_call_usd is not None and outcome.claims_judged:
        outcome.cost_estimate_usd = round(config.cost_per_call_usd * outcome.claims_judged, 6)
    return outcome
