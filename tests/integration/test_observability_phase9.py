# SPDX-License-Identifier: Apache-2.0
"""Phase 9: OpenTelemetry span tree + versioned structured logging."""

from __future__ import annotations

import contextlib
import json
import logging

import pytest

from ragwarden import GateRequest, Policy, gate
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context
from ragwarden.observability.logging import (
    SCHEMA_VERSION,
    gate_result_to_dict,
    gate_result_to_json,
    log_gate_result,
)

pytestmark = pytest.mark.integration

CTX = Context(
    query="Eiffel Tower?",
    chunks=[Chunk("The Eiffel Tower was completed in 1889.", 0.9, "d1")],
    retrieval_method="hybrid",
)
_SUPPORT = ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.SUPPORTED, 0.95, 1))
ANSWER = Answer(text="The Eiffel Tower was completed in 1889.")


# --- structured logging (no optional deps) --------------------------------
def test_log_schema_is_versioned_and_complete() -> None:
    result = gate(CTX, ANSWER, detectors=[_SUPPORT], emit_telemetry=False)
    d = gate_result_to_dict(result, request_id="r1")
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["request_id"] == "r1"
    assert d["action"] == "allow"
    assert d["claim_verdicts"][0]["status"] == "supported"
    assert [s["stage_name"] for s in d["stage_trace"]][:2] == ["decomposition", "tier0"]
    json.loads(gate_result_to_json(result))  # round-trips


def test_log_gate_result_emits_one_record(caplog) -> None:
    result = gate(CTX, ANSWER, detectors=[_SUPPORT], emit_telemetry=False)
    with caplog.at_level(logging.INFO, logger="ragwarden.gate"):
        log_gate_result(result, request_id="r2")
    recs = [r for r in caplog.records if r.message == "ragwarden.gate.decision"]
    assert len(recs) == 1
    assert recs[0].ragwarden["request_id"] == "r2"


# --- OpenTelemetry span tree (needs the sdk) -----------------------------
def test_otel_span_tree() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # set_tracer_provider only takes effect once per process; tolerate re-runs
    with contextlib.suppress(Exception):
        trace.set_tracer_provider(provider)
    tracer_provider = trace.get_tracer_provider()
    if tracer_provider is not provider:
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    gate(CTX, ANSWER, detectors=[_SUPPORT], request=GateRequest("req-otel"))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "ragwarden.gate" in spans
    for name in (
        "ragwarden.decomposition",
        "ragwarden.tier0",
        "ragwarden.tier1",
        "ragwarden.scoring",
        "ragwarden.policy",
    ):
        assert name in spans, name
    assert "ragwarden.tier1.detector.scripted_stub" in spans

    root = spans["ragwarden.gate"]
    assert root.parent is None
    assert spans["ragwarden.tier0"].parent.span_id == root.context.span_id
    assert spans["ragwarden.tier1.detector.scripted_stub"].parent.span_id == (
        spans["ragwarden.tier1"].context.span_id
    )
    attrs = dict(root.attributes)
    assert attrs["ragwarden.claims_total"] == 1
    assert attrs["ragwarden.claims_resolved_tier1"] == 1
    assert attrs["ragwarden.action"] == "allow"
    assert attrs["ragwarden.request_id"] == "req-otel"


def test_emit_telemetry_false_is_silent() -> None:
    # Should not raise even with no tracer provider configured for real export.
    r = gate(CTX, ANSWER, detectors=[_SUPPORT], policy=Policy.default(), emit_telemetry=False)
    assert r.action.value == "allow"
