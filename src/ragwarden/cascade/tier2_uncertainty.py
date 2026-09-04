# SPDX-License-Identifier: Apache-2.0
"""Tier 2 — uncertainty quantification (Build Spec Section 7.4).

For claims Tier 1 could not confidently resolve, use consistency-based signals:

* **Black-box** (always available): the host application's generator is called
  ``N`` more times for the same query; we measure how consistently the claim is
  supported across those independent samples (SelfCheckGPT-style). Low agreement
  is evidence of hallucination risk even without proving the claim false.
* **White-box** (optional, only if ``Generation.token_logprobs`` is populated):
  low average / minimum token probability over the claim's span is an additional
  signal, *combined* with — not replacing — the black-box signal.

This tier costs ``N`` extra LLM calls. ``N`` is configurable down to 0 (disabling
Tier 2 entirely); the estimated dollar cost is surfaced in
``StageTrace.cost_estimate_usd``.

**Fault isolation.** ``generate_fn`` is a network call to an LLM provider in most
deployments — it can hang or raise. Each sample call is isolated and
individually time-budgeted (``config.call_timeout_s``); a failed sample is
simply excluded from that claim's agreement count rather than failing the
request. If every sample fails, or a claim ends up with zero usable samples, it
stays ``UNRESOLVED`` (escalates to Tier 3) rather than being scored on no data.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors.base import Detector
from ragwarden.errors import CallableError, CallableTimeoutError
from ragwarden.models import Chunk
from ragwarden.resilience import call_with_timeout, safe_call

__all__ = ["Tier2Config", "Tier2Outcome", "run_tier2"]

GenerateFn = Callable[[str], str]

_WORD = re.compile(r"[a-z0-9']+")


@dataclass
class Tier2Config:
    consistency_samples: int = 3  # 0 disables Tier 2
    agreement_resolve_threshold: float = 0.7  # >= -> SUPPORTED; <= 1-this -> UNSUPPORTED
    logprob_low_threshold: float = 0.35  # mean token prob below this drags confidence down
    cost_per_sample_usd: float | None = None
    call_timeout_s: float | None = 10.0  # per generate_fn() call
    min_effective_samples: int = 1  # fewer successful samples than this -> stays UNRESOLVED


@dataclass
class Tier2Outcome:
    resolved: list[ClaimVerdict] = field(default_factory=list)
    unresolved: list[ClaimVerdict] = field(default_factory=list)
    samples_drawn: int = 0
    samples_failed: int = 0
    cost_estimate_usd: float | None = None

    @property
    def all_verdicts(self) -> list[ClaimVerdict]:
        return self.resolved + self.unresolved


def _mean_claim_token_prob(
    claim: str, answer_text: str, token_logprobs: list[float] | None
) -> float | None:
    """Rough white-box signal: mean P(token) over the answer region matching the
    claim. Token alignment is approximate — we map by character position."""
    if not token_logprobs:
        return None
    idx = answer_text.find(claim[:40])
    if idx < 0:
        return None
    frac_start = idx / max(1, len(answer_text))
    frac_end = (idx + len(claim)) / max(1, len(answer_text))
    n = len(token_logprobs)
    lo, hi = int(frac_start * n), max(int(frac_end * n), int(frac_start * n) + 1)
    span = token_logprobs[lo:hi] or token_logprobs
    probs = [math.exp(lp) for lp in span]
    return sum(probs) / len(probs) if probs else None


def _draw_samples(
    generate_fn: GenerateFn, query: str, n: int, *, call_timeout_s: float | None
) -> tuple[list[str], int]:
    """Draw up to ``n`` samples, isolating each call. Returns (samples, n_failed)."""
    samples: list[str] = []
    failed = 0
    for _ in range(n):
        try:
            if call_timeout_s is not None:
                samples.append(
                    call_with_timeout(
                        generate_fn, query, timeout_s=call_timeout_s, what="Tier 2 generate_fn"
                    )
                )
            else:
                samples.append(safe_call(generate_fn, query, what="Tier 2 generate_fn"))
        except (CallableTimeoutError, CallableError):
            failed += 1
    return samples, failed


def run_tier2(
    claims: list[ClaimVerdict],
    query: str,
    *,
    generate_fn: GenerateFn | None,
    detector: Detector | None,
    config: Tier2Config,
    answer_text: str = "",
    token_logprobs: list[float] | None = None,
) -> Tier2Outcome:
    """Resolve (or re-escalate) the ``claims`` Tier 1 left unresolved.

    ``claims`` are the ``UNRESOLVED`` verdicts from Tier 1. ``detector`` is reused
    to judge whether each fresh sample supports the claim.
    """
    outcome = Tier2Outcome()
    n = max(0, config.consistency_samples)

    if n == 0 or generate_fn is None or detector is None:
        # Tier 2 disabled or not wired: pass everything through unchanged.
        outcome.unresolved = list(claims)
        return outcome

    samples, failed = _draw_samples(generate_fn, query, n, call_timeout_s=config.call_timeout_s)
    outcome.samples_drawn = len(samples)
    outcome.samples_failed = failed
    if config.cost_per_sample_usd is not None and outcome.samples_drawn:
        outcome.cost_estimate_usd = round(config.cost_per_sample_usd * outcome.samples_drawn, 6)

    if not samples:
        outcome.unresolved = [
            ClaimVerdict(
                c.claim_text,
                ClaimStatus.UNRESOLVED,
                c.confidence,
                2,
                explanation=f"Tier 2: all {failed} generate_fn() calls failed; escalating",
            )
            for c in claims
        ]
        return outcome

    sample_chunks = [
        Chunk(text=s, score=1.0, source_id=f"tier2-sample-{i}") for i, s in enumerate(samples)
    ]

    positive = {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
    for claim in claims:
        agree = 0
        usable = 0
        for chunk in sample_chunks:
            try:
                verdict = safe_call(
                    detector.check, claim.claim_text, [chunk], what="Tier 2 detector.check"
                )
            except CallableError:
                continue  # this sample is unusable for this claim; not fatal
            usable += 1
            if verdict.status in positive:
                agree += 1

        if usable < config.min_effective_samples:
            outcome.unresolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.UNRESOLVED,
                    claim.confidence,
                    2,
                    explanation=(
                        f"Tier 2: only {usable}/{len(sample_chunks)} samples were checkable; "
                        "escalating"
                    ),
                )
            )
            continue

        agreement = agree / usable
        mean_prob = _mean_claim_token_prob(claim.claim_text, answer_text, token_logprobs)
        white_box_note = ""
        penalty = 0.0
        if mean_prob is not None and mean_prob < config.logprob_low_threshold:
            penalty = 0.15
            white_box_note = f"; low mean token prob {mean_prob:.2f}"

        if agreement >= config.agreement_resolve_threshold and penalty == 0.0:
            outcome.resolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.SUPPORTED,
                    round(agreement, 4),
                    2,
                    explanation=f"Tier 2: {agree}/{usable} usable samples support the claim",
                )
            )
        elif agreement <= (1.0 - config.agreement_resolve_threshold):
            outcome.resolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.UNSUPPORTED,
                    round(1.0 - agreement - penalty, 4),
                    2,
                    explanation=(
                        f"Tier 2: only {agree}/{usable} usable samples support the "
                        f"claim{white_box_note}"
                    ),
                )
            )
        else:
            outcome.unresolved.append(
                ClaimVerdict(
                    claim.claim_text,
                    ClaimStatus.UNRESOLVED,
                    round(max(agreement, 1 - agreement) - penalty, 4),
                    2,
                    explanation=(
                        f"Tier 2: split sample agreement {agree}/{usable}{white_box_note}; "
                        "escalating"
                    ),
                )
            )
    return outcome
