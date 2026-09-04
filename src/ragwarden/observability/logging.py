# SPDX-License-Identifier: Apache-2.0
"""Structured logging (Build Spec Section 14.2).

Every :class:`~ragwarden.contracts.GateResult` serializes to a versioned JSON
schema. ``schema_version`` is present from day one so the schema can evolve
without a painful migration later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from ragwarden.contracts import GateResult

__all__ = ["SCHEMA_VERSION", "gate_result_to_dict", "gate_result_to_json", "log_gate_result"]

SCHEMA_VERSION = "1"

_LOGGER = logging.getLogger("ragwarden.gate")


def gate_result_to_dict(result: GateResult, *, request_id: str | None = None) -> dict[str, Any]:
    """Serialize a GateResult to the versioned log schema."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "action": result.action.value,
        "reliability_score": round(result.reliability_score, 6),
        "explanation": result.explanation,
        "output_text": result.output_text,
        "action_payload": result.action_payload,
        "claim_verdicts": [{**asdict(v), "status": v.status.value} for v in result.claim_verdicts],
        "stage_trace": [asdict(s) for s in result.stage_trace],
    }
    return payload


def gate_result_to_json(result: GateResult, *, request_id: str | None = None) -> str:
    return json.dumps(gate_result_to_dict(result, request_id=request_id), default=str)


def log_gate_result(
    result: GateResult,
    *,
    request_id: str | None = None,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit the GateResult as one structured JSON log record."""
    (logger or _LOGGER).log(
        level,
        "ragwarden.gate.decision",
        extra={"ragwarden": gate_result_to_dict(result, request_id=request_id)},
    )
