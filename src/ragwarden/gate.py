# SPDX-License-Identifier: Apache-2.0
"""Top-level ``gate()`` orchestration (Build Spec Section 5.2).

Flow: decomposition -> Tier 0 -> Tier 1 -> Tier 2 -> Tier 3 -> composite scoring
-> policy -> action -> GateResult. Per-stage tracing; the Tier 3 budget is
tracked against a request-wide deadline, not per-tier in isolation.

**Production safety net.** Every tier already isolates its own failure modes
(see :mod:`ragwarden.resilience` and each ``cascade`` module's docstring) —
but ``gate()`` additionally never lets an *unexpected* exception (a detector
bug, a contract violation in the caller's input, anything not already
anticipated) propagate to the host. It logs the failure, emits it on the OTel
span if telemetry is on, and returns a safe ``ABSTAIN`` decision instead —
because a broken RAG gate has to fail toward "say nothing" rather than "crash
the request" or "silently allow." Set ``raise_on_error=True`` to get the real
traceback back (for local development / tests), and note that a bad
:class:`~ragwarden.policy.Policy` (``Policy.validate()``) still raises loudly
and immediately — that is a deploy-time bug, not a request-time failure, and
hiding it would be worse.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from ragwarden.actions import apply_action
from ragwarden.cascade.tier0_heuristics import run_tier0
from ragwarden.cascade.tier1_entailment import run_tier1
from ragwarden.cascade.tier2_uncertainty import Tier2Config as _Tier2CascadeConfig
from ragwarden.cascade.tier2_uncertainty import run_tier2
from ragwarden.cascade.tier3_judge import Tier3Config as _Tier3CascadeConfig
from ragwarden.cascade.tier3_judge import run_tier3
from ragwarden.contracts import (
    ClaimVerdict,
    GateAction,
    GateResult,
    Generation,
    RetrievalContext,
    StageTrace,
)
from ragwarden.decomposition import get_decomposer
from ragwarden.detectors.base import Detector, MissingExtraError
from ragwarden.policy import Policy, decide
from ragwarden.scoring import get_scoring_strategy

__all__ = ["GateRequest", "agate", "gate"]

_LOG = logging.getLogger("ragwarden.gate")

# Tier-0 low-confidence flags cap reliability even when no claim-level verdict
# disqualifies the answer.
_TIER0_FLAGGED_CEILING = 0.75


@dataclass
class GateRequest:
    """Optional per-call state for multi-turn requests (retry accounting)."""

    request_id: str | None = None
    retries_used: int = 0


class _Clock:
    def __init__(self) -> None:
        self._t = time.perf_counter()

    def lap_ms(self) -> float:
        now = time.perf_counter()
        ms = (now - self._t) * 1000.0
        self._t = now
        return ms


def _resolve_detectors(policy: Policy) -> list[Detector]:
    """Instantiate the Tier-1 detectors named in the policy. Returns [] (with a
    loud warning) if none can be loaded — the cascade then falls back to Tier 0.

    Catches broadly, not just ``MissingExtraError``: a corrupted model cache, a
    download failure on first run, or an OOM at load time must degrade the same
    way a missing extra does — never crash the request that happens to be the
    first one to touch a not-yet-loaded model.
    """
    from ragwarden.detectors import get_detector

    names = [policy.tier1.detector, *policy.tier1.ensemble]
    detectors: list[Detector] = []
    for name in names:
        try:
            detectors.append(get_detector(name))
        except MissingExtraError as exc:
            warnings.warn(
                f"Tier 1 detector {name!r} unavailable ({exc}); cascade falls back to Tier 0 only.",
                stacklevel=2,
            )
        except Exception as exc:
            _LOG.exception("Tier 1 detector %r failed to load", name)
            warnings.warn(
                f"Tier 1 detector {name!r} failed to load ({type(exc).__name__}: {exc}); "
                f"cascade falls back to Tier 0 only.",
                stacklevel=2,
            )
    return detectors


def _emit_telemetry(
    result: GateResult, *, request_id: str | None, detector_names: list[str]
) -> None:
    """Best-effort span emission — a telemetry bug must never affect the
    decision that was already computed."""
    try:
        from ragwarden.observability.otel import emit_gate_spans

        emit_gate_spans(result, request_id=request_id, detectors_ran=detector_names or None)
    except Exception:
        _LOG.debug("OTel span emission failed", exc_info=True)


def _safety_net_result(exc: BaseException, *, policy: Policy, request_id: str | None) -> GateResult:
    """What ``gate()`` returns when something unexpected breaks the cascade.
    Always ABSTAIN — a broken gate must fail toward silence, not toward
    shipping an unverified answer."""
    _LOG.exception("ragwarden.gate: unhandled error, failing closed to ABSTAIN")
    return GateResult(
        action=GateAction.ABSTAIN,
        reliability_score=0.0,
        claim_verdicts=[],
        output_text=policy.abstain_message,
        stage_trace=[],
        explanation=(
            f"abstain: internal error ({type(exc).__name__}: {exc}); failing closed. "
            "See logs for the full traceback."
        ),
        action_payload={
            "reason": "internal_error",
            "error_type": type(exc).__name__,
            "request_id": request_id,
        },
    )


def gate(
    context: RetrievalContext,
    generation: Generation,
    *,
    policy: Policy | None = None,
    request: GateRequest | None = None,
    detectors: list[Detector] | None = None,
    generate_fn: Callable[[str], str] | None = None,
    judge_fn: Callable[[str], str] | None = None,
    emit_telemetry: bool = True,
    raise_on_error: bool = False,
) -> GateResult:
    """Gate a generated answer against its retrieval context. See Spec Section 5.2.

    ``detectors`` injects already-constructed Tier-1 detectors (share a loaded
    model, or use a stub); when omitted they come from ``policy.tier1``.
    ``generate_fn`` (``query -> answer_text``) drives Tier 2 consistency sampling;
    ``judge_fn`` (``prompt -> completion``) drives Tier 3. A tier is skipped when
    its callable is absent. ``emit_telemetry`` emits the OpenTelemetry span tree
    when ``opentelemetry`` is installed (a no-op otherwise). ``raise_on_error``
    disables the production safety net (re-raises unexpected errors instead of
    returning a safe ABSTAIN) — use it in tests/local dev, not in production.
    """
    pol = policy or Policy.default()
    pol.validate()  # a bad policy is a deploy-time bug — raise loudly, don't hide it
    req = request or GateRequest()

    try:
        return _gate_inner(
            context,
            generation,
            pol,
            req,
            detectors=detectors,
            generate_fn=generate_fn,
            judge_fn=judge_fn,
            emit_telemetry=emit_telemetry,
        )
    except Exception as exc:
        if raise_on_error:
            raise
        result = _safety_net_result(exc, policy=pol, request_id=req.request_id)
        if emit_telemetry:
            _emit_telemetry(result, request_id=req.request_id, detector_names=[])
        return result


async def agate(
    context: RetrievalContext,
    generation: Generation,
    *,
    policy: Policy | None = None,
    request: GateRequest | None = None,
    detectors: list[Detector] | None = None,
    generate_fn: Callable[[str], str] | None = None,
    judge_fn: Callable[[str], str] | None = None,
    emit_telemetry: bool = True,
    raise_on_error: bool = False,
) -> GateResult:
    """Async wrapper around :func:`gate`.

    ``gate()`` is CPU/IO-bound synchronous code (model inference, blocking HTTP
    calls inside ``generate_fn``/``judge_fn``). Calling it directly from an
    ``async def`` request handler (FastAPI, aiohttp, ...) blocks the event
    loop and stalls every other in-flight request. ``agate()`` runs it in a
    worker thread via :func:`asyncio.to_thread` so it doesn't.
    """
    return await asyncio.to_thread(
        gate,
        context,
        generation,
        policy=policy,
        request=request,
        detectors=detectors,
        generate_fn=generate_fn,
        judge_fn=judge_fn,
        emit_telemetry=emit_telemetry,
        raise_on_error=raise_on_error,
    )


def _gate_inner(
    context: RetrievalContext,
    generation: Generation,
    pol: Policy,
    req: GateRequest,
    *,
    detectors: list[Detector] | None,
    generate_fn: Callable[[str], str] | None,
    judge_fn: Callable[[str], str] | None,
    emit_telemetry: bool,
) -> GateResult:
    start = time.perf_counter()
    request_deadline = (
        start + pol.max_total_latency_ms / 1000.0 if pol.max_total_latency_ms else None
    )
    clock = _Clock()
    traces: list[StageTrace] = []
    resolved: list[ClaimVerdict] = []
    unresolved: list[ClaimVerdict] = []
    notes: list[str] = []
    costs: list[float] = []
    detector_names: list[str] = []

    def _finish(result: GateResult) -> GateResult:
        if emit_telemetry:
            _emit_telemetry(result, request_id=req.request_id, detector_names=detector_names)
        return result

    # 1. Claim decomposition
    decomposer = get_decomposer(pol.decomposition.strategy, **pol.decomposition.options)
    claims = decomposer.decompose(generation.text)
    n_claims_raw = len(claims)
    if pol.max_claims_per_request is not None and n_claims_raw > pol.max_claims_per_request:
        dropped = n_claims_raw - pol.max_claims_per_request
        claims = claims[: pol.max_claims_per_request]
        notes.append(
            f"answer produced {n_claims_raw} claims; capped at {pol.max_claims_per_request} "
            f"({dropped} not checked, per policy.max_claims_per_request)"
        )
    n_claims = len(claims)
    traces.append(StageTrace("decomposition", clock.lap_ms(), claims_processed=n_claims))

    # 2. Tier 0 — retrieval heuristics (once per request, before claim-level work)
    tier0 = run_tier0(context, pol.tier0)
    traces.append(StageTrace("tier0", clock.lap_ms(), claims_processed=0))

    if tier0.short_circuit:
        sc = apply_action(
            GateAction.ABSTAIN,
            original_text=generation.text,
            verdicts=[],
            abstain_message=pol.abstain_message,
            reason=tier0.reason or "Tier 0 retrieval check failed",
        )
        return _finish(
            GateResult(
                action=sc.action,
                reliability_score=0.0,
                claim_verdicts=[],
                output_text=sc.output_text,
                stage_trace=traces,
                explanation=f"Short-circuited at Tier 0: {tier0.reason}",
                action_payload=sc.payload,
            )
        )

    # 3. Tier 1 — claim-vs-evidence entailment
    active_detectors: list[Detector] = []
    if pol.tier1.enabled and claims:
        active_detectors = detectors if detectors is not None else _resolve_detectors(pol)

    if active_detectors:
        detector_names.extend(getattr(d, "name", type(d).__name__) for d in active_detectors)
        t1 = run_tier1(
            claims,
            list(context.chunks),
            detectors=active_detectors,
            confidence_threshold=pol.tier1.confidence_threshold,
            ensemble_strategy=pol.tier1.ensemble_strategy,
            fail_mode=pol.fail_mode,
            call_timeout_s=pol.tier1_call_timeout_s,
        )
        resolved.extend(t1.resolved)
        unresolved.extend(t1.unresolved)
        if t1.detector_errors:
            notes.append(
                "Tier 1 detector failures: "
                + "; ".join(f"{n} ({m})" for n, m in t1.detector_errors)
            )
        traces.append(StageTrace("tier1", clock.lap_ms(), claims_processed=n_claims))
    elif claims and pol.tier1.enabled:
        notes.append("Tier 1 unavailable — decision is Tier-0-only")

    # 4. Tier 2 — uncertainty quantification (only the Tier-1 remainder)
    if unresolved and pol.tier2.enabled and pol.tier2.consistency_samples > 0:
        if generate_fn is None or not active_detectors:
            why = "no generate_fn" if generate_fn is None else "no detector"
            notes.append(f"{len(unresolved)} claims unresolved at Tier 1; Tier 2 skipped ({why})")
        else:
            t2 = run_tier2(
                unresolved,
                context.query,
                generate_fn=generate_fn,
                detector=active_detectors[0],
                config=_Tier2CascadeConfig(
                    consistency_samples=pol.tier2.consistency_samples,
                    agreement_resolve_threshold=pol.tier2.agreement_resolve_threshold,
                    logprob_low_threshold=pol.tier2.logprob_low_threshold,
                    cost_per_sample_usd=pol.tier2.cost_per_sample_usd,
                    call_timeout_s=pol.tier2.call_timeout_s,
                ),
                answer_text=generation.text,
                token_logprobs=list(generation.token_logprobs)
                if generation.token_logprobs is not None
                else None,
            )
            resolved.extend(t2.resolved)
            unresolved = list(t2.unresolved)
            if t2.cost_estimate_usd:
                costs.append(t2.cost_estimate_usd)
            if t2.samples_failed:
                notes.append(
                    f"Tier 2: {t2.samples_failed}/{t2.samples_failed + t2.samples_drawn} "
                    f"generate_fn() calls failed"
                )
            traces.append(
                StageTrace(
                    "tier2",
                    clock.lap_ms(),
                    claims_processed=len(t2.all_verdicts),
                    cost_estimate_usd=t2.cost_estimate_usd,
                )
            )

    # 5. Tier 3 — LLM-as-judge (only the Tier-2 remainder), budget-capped
    reached_tier3 = 0
    if unresolved and pol.tier3.enabled and judge_fn is not None:
        reached_tier3 = len(unresolved)
        t3 = run_tier3(
            unresolved,
            list(context.chunks),
            judge_fn=judge_fn,
            config=_Tier3CascadeConfig(
                max_claims_per_request=pol.tier3.max_claims_per_request,
                max_latency_ms=pol.tier3.max_latency_ms,
                cost_per_call_usd=pol.tier3.cost_per_call_usd,
                max_evidence_chars=pol.tier3.max_evidence_chars,
                call_timeout_s=pol.tier3.call_timeout_s,
            ),
            deadline=request_deadline,
        )
        resolved.extend(t3.resolved)
        unresolved = list(t3.unresolved)
        if t3.cost_estimate_usd:
            costs.append(t3.cost_estimate_usd)
        traces.append(
            StageTrace(
                "tier3",
                clock.lap_ms(),
                claims_processed=t3.claims_judged,
                cost_estimate_usd=t3.cost_estimate_usd,
            )
        )
        if t3.budget_exhausted:
            notes.append("Tier 3 budget exhausted; remaining claims left UNRESOLVED")
    elif unresolved and pol.tier3.enabled and judge_fn is None:
        notes.append(
            f"{len(unresolved)} claims unresolved after Tier 2; Tier 3 skipped (no judge_fn)"
        )

    verdicts = resolved + unresolved

    # Escalation-rate metric (Spec Section 7.5) — surfaced, not enforced.
    if n_claims and reached_tier3:
        rate = reached_tier3 / n_claims
        notes.append(f"Tier 3 escalation rate {rate:.0%} ({reached_tier3}/{n_claims})")
        if rate > pol.tier3.escalation_rate_warn_threshold:
            notes.append("high Tier 3 escalation — consider retuning Tier 1/2 thresholds")

    # 6. Composite scoring
    strategy_name = pol.claim_weighting if verdicts else "pass_through"
    score = get_scoring_strategy(strategy_name).score(verdicts)
    if tier0.flags:
        score = min(score, _TIER0_FLAGGED_CEILING)
    traces.append(StageTrace("scoring", clock.lap_ms(), claims_processed=len(verdicts)))

    # 7. Policy engine
    decision = decide(score, verdicts, pol, retries_used=req.retries_used)
    traces.append(StageTrace("policy", clock.lap_ms(), claims_processed=0))

    # 8. Action
    outcome = apply_action(
        decision.action,
        original_text=generation.text,
        verdicts=verdicts,
        abstain_message=pol.abstain_message,
        reason=decision.reason,
        reliability_score=score,
        retries_used=req.retries_used,
        max_retries=pol.max_retries,
    )

    explanation = f"{outcome.action.value}: {decision.reason}"
    if tier0.flags:
        explanation += f"; Tier 0 flags: {', '.join(tier0.flags)}"
    if outcome.fell_back:
        explanation += f" (fell back from {decision.action.value})"
    if costs:
        explanation += f"; est. cascade cost ${sum(costs):.4f}"
    if notes:
        explanation += "; " + "; ".join(notes)

    return _finish(
        GateResult(
            action=outcome.action,
            reliability_score=score,
            claim_verdicts=verdicts,
            output_text=outcome.output_text,
            stage_trace=traces,
            explanation=explanation,
            action_payload=outcome.payload,
        )
    )
