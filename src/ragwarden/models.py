# SPDX-License-Identifier: Apache-2.0
"""Concrete, ready-to-use implementations of the core contract protocols.

Host applications can pass their own objects (anything structurally matching
:mod:`ragwarden.contracts`), but these are convenient for adapters, tests, and
quickstarts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ragwarden.contracts import RetrievedChunk

__all__ = ["Answer", "Chunk", "Context"]


@dataclass
class Chunk:
    """A concrete :class:`~ragwarden.contracts.RetrievedChunk`."""

    text: str
    score: float = 0.0
    source_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Context:
    """A concrete :class:`~ragwarden.contracts.RetrievalContext`."""

    query: str
    chunks: Sequence[RetrievedChunk] = field(default_factory=list)
    retrieval_method: str = "unknown"


@dataclass
class Answer:
    """A concrete :class:`~ragwarden.contracts.Generation`."""

    text: str
    token_logprobs: Sequence[float] | None = None
