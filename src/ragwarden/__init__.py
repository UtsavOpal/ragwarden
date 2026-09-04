# SPDX-License-Identifier: Apache-2.0
"""RagWarden — an inline hallucination gate for production RAG pipelines.

Public API surface is deliberately small. Everything under ``ragwarden.cascade``,
``ragwarden.detectors``, ``ragwarden.adapters`` etc. is implementation detail and
may change without a major version bump until v1.0.

    from ragwarden import gate, Policy
    from ragwarden.models import Chunk, Context, Answer

    result = gate(context, generation, policy=Policy.default())

``gate()`` never raises for a runtime cascade failure — see its docstring and
:mod:`ragwarden.resilience`. Use ``agate()`` from an ``async def`` handler.
"""

from __future__ import annotations

from ragwarden.contracts import (
    ClaimStatus,
    ClaimVerdict,
    GateAction,
    GateResult,
    Generation,
    RetrievalContext,
    RetrievedChunk,
    StageTrace,
)
from ragwarden.errors import (
    CallableError,
    CallableTimeoutError,
    ConfigError,
    DetectorError,
    RagWardenError,
)
from ragwarden.gate import GateRequest, agate, gate
from ragwarden.policy import Policy

__version__ = "0.3.1"

__all__ = [
    "CallableError",
    "CallableTimeoutError",
    "ClaimStatus",
    "ClaimVerdict",
    "ConfigError",
    "DetectorError",
    "GateAction",
    "GateRequest",
    "GateResult",
    "Generation",
    "Policy",
    "RagWardenError",
    "RetrievalContext",
    "RetrievedChunk",
    "StageTrace",
    "__version__",
    "agate",
    "gate",
]
