# SPDX-License-Identifier: Apache-2.0
"""Observability (Build Spec Section 14): OpenTelemetry spans + structured logs.

RagWarden emits standard OTel spans that plug into whatever the host already
runs — it does not ship a dashboard.
"""

from ragwarden.observability.logging import (
    SCHEMA_VERSION,
    gate_result_to_dict,
    gate_result_to_json,
    log_gate_result,
)
from ragwarden.observability.otel import emit_gate_spans, otel_available

__all__ = [
    "SCHEMA_VERSION",
    "emit_gate_spans",
    "gate_result_to_dict",
    "gate_result_to_json",
    "log_gate_result",
    "otel_available",
]
