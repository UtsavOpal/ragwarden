# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry span emission (Build Spec Section 14.1).

Span tree emitted per ``gate()`` call::

    ragwarden.gate                       (root)
    |-- ragwarden.decomposition
    |-- ragwarden.tier0
    |-- ragwarden.tier1
    |   `-- ragwarden.tier1.detector.<name>   (one per detector that ran)
    |-- ragwarden.tier2
    |-- ragwarden.tier3
    |-- ragwarden.scoring
    `-- ragwarden.policy

Standard attributes on the root span (stable keys for dashboards):
``ragwarden.claims_total``, ``ragwarden.claims_resolved_tier{0,1,2,3}``,
``ragwarden.reliability_score``, ``ragwarden.action``,
``ragwarden.cost_estimate_usd``, ``ragwarden.request_id``.

When ``opentelemetry`` is not installed every function here is a no-op — RagWarden
never hard-depends on it. RagWarden does not create a TracerProvider; it uses
whatever the host already configured (Datadog, Grafana/Tempo, Langfuse, ...).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragwarden.contracts import GateResult

__all__ = ["emit_gate_spans", "otel_available"]

_ROOT = "ragwarden.gate"
_DETECTOR_STAGE = "tier1"


def otel_available() -> bool:
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError:
        return False
    return True


def _tracer() -> Any:
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - guarded by otel_available()
        return None
    return trace.get_tracer("ragwarden")


def emit_gate_spans(
    result: GateResult,
    *,
    request_id: str | None = None,
    detectors_ran: list[str] | None = None,
) -> None:
    """Emit the span tree for a completed ``gate()`` call. No-op without OpenTelemetry.

    Spans are created with explicit back-to-back timings reconstructed from
    ``result.stage_trace`` so the structure, names, parent/child links and
    attributes are correct even though the gate already returned.
    """
    tracer = _tracer()
    if tracer is None:
        return

    import time

    from opentelemetry.trace import SpanKind

    total_ms = sum(s.latency_ms for s in result.stage_trace)
    now = time.time_ns()
    start_ns = now - int(total_ms * 1_000_000)

    tier_counts: Counter[int] = Counter(v.resolved_at_tier for v in result.claim_verdicts)

    root = tracer.start_span(_ROOT, kind=SpanKind.INTERNAL, start_time=start_ns)
    try:
        root.set_attribute("ragwarden.claims_total", len(result.claim_verdicts))
        for tier in (0, 1, 2, 3):
            root.set_attribute(
                f"ragwarden.claims_resolved_tier{tier}", int(tier_counts.get(tier, 0))
            )
        root.set_attribute("ragwarden.reliability_score", round(result.reliability_score, 4))
        root.set_attribute("ragwarden.action", result.action.value)
        if request_id:
            root.set_attribute("ragwarden.request_id", request_id)

        cursor = start_ns
        from opentelemetry import trace as _trace

        ctx = _trace.set_span_in_context(root)
        for stage in result.stage_trace:
            span_start = cursor
            span_end = cursor + int(stage.latency_ms * 1_000_000)
            child = tracer.start_span(
                f"ragwarden.{stage.stage_name}", context=ctx, start_time=span_start
            )
            child.set_attribute("ragwarden.claims_processed", stage.claims_processed)
            if stage.cost_estimate_usd is not None:
                child.set_attribute("ragwarden.cost_estimate_usd", stage.cost_estimate_usd)
            if stage.stage_name == _DETECTOR_STAGE and detectors_ran:
                d_ctx = _trace.set_span_in_context(child)
                for name in detectors_ran:
                    leaf = tracer.start_span(
                        f"ragwarden.tier1.detector.{name}",
                        context=d_ctx,
                        start_time=span_start,
                    )
                    leaf.end(end_time=span_end)
            child.end(end_time=span_end)
            cursor = span_end

        total_cost = sum(s.cost_estimate_usd or 0.0 for s in result.stage_trace)
        if total_cost:
            root.set_attribute("ragwarden.cost_estimate_usd", round(total_cost, 6))
    finally:
        root.end(end_time=start_ns + int(total_ms * 1_000_000))
