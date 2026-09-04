# SPDX-License-Identifier: Apache-2.0
"""Claim decomposition (Build Spec Section 7.1).

Runs before the cascade: split a generation into atomic, independently-checkable
claims. v0.1 ships sentence-level splitting — a known simplification relative to
true semantic claim extraction (FActScore / RefChecker) that is good enough to
validate the cascade design and carries zero heavy dependencies.

A model-based / LLM-prompted decomposer is an OPEN DECISION (Spec Section 17):
it is exposed behind ``decomposition.strategy`` config, never selected silently,
and is treated as a pipeline stage with its own latency / cost accounting.
"""

from __future__ import annotations

import re
from typing import Protocol

__all__ = ["ClaimDecomposer", "LLMDecomposer", "SentenceSplitDecomposer", "get_decomposer"]

_WHITESPACE = re.compile(r"\s+")


class ClaimDecomposer(Protocol):
    """Turn generated answer text into a list of atomic claim strings."""

    name: str

    def decompose(self, generation_text: str) -> list[str]:
        """Return atomic claim strings extracted from the generation."""
        ...


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


class SentenceSplitDecomposer:
    """v0.1 default: sentence-level splitting via ``pysbd`` (pure-Python, no ML).

    Falls back to a conservative regex splitter if ``pysbd`` is unavailable so
    the core package never hard-fails on decomposition.
    """

    name = "sentence_split"

    def __init__(self, language: str = "en", min_chars: int = 3) -> None:
        self.language = language
        self.min_chars = min_chars
        self._segmenter = None
        try:
            import pysbd

            self._segmenter = pysbd.Segmenter(language=language, clean=False)
        except Exception:  # pragma: no cover - exercised only without pysbd
            self._segmenter = None

    def decompose(self, generation_text: str) -> list[str]:
        text = generation_text or ""
        if not text.strip():
            return []
        if self._segmenter is not None:
            raw = self._segmenter.segment(text)
        else:
            raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
        claims = [_normalize(s) for s in raw]
        return [c for c in claims if len(c) >= self.min_chars]


class LLMDecomposer:
    """OPEN DECISION placeholder (Spec Section 7.1 v0.3+).

    Higher-quality model / LLM-prompted claim extraction. Wired in a later phase;
    constructing it today raises so the choice is never made silently.
    """

    name = "llm"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "LLMDecomposer is an OPEN DECISION (Spec Section 7.1) not yet implemented. "
            "Use SentenceSplitDecomposer, or track this in the roadmap."
        )

    def decompose(self, generation_text: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError


def get_decomposer(strategy: str = "sentence_split", **kwargs: object) -> ClaimDecomposer:
    """Registry entry point used by :func:`ragwarden.gate.gate` and config loading."""
    if strategy == "sentence_split":
        return SentenceSplitDecomposer(**kwargs)  # type: ignore[arg-type]
    if strategy == "llm":
        return LLMDecomposer(**kwargs)
    raise ValueError(f"unknown decomposition strategy: {strategy!r}")
