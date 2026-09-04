# SPDX-License-Identifier: Apache-2.0
"""Tier 0 — retrieval heuristics (Build Spec Section 7.2).

Runs once per request on the :class:`~ragwarden.contracts.RetrievalContext`
alone, before any claim-level work. Purely deterministic, no ML models,
near-zero latency. If retrieval itself looks insufficient the gate can
short-circuit straight to ABSTAIN.

Because the target stack produces both BM25 and kNN scores, the heuristics read
``chunk.metadata['bm25_score']`` / ``['knn_score']`` when present, in addition to
the normalized ``chunk.score`` field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragwarden.contracts import RetrievalContext

__all__ = ["Tier0Config", "Tier0Finding", "run_tier0"]

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "how",
    ]
)


@dataclass
class Tier0Config:
    min_top_score: float | None = None  # score floor; None disables
    score_gap_threshold: float | None = None  # top-vs-rest drop-off; None disables
    min_overlap: float = 0.0  # query-context lexical overlap floor (0 disables)
    score_field_priority: tuple[str, ...] = ("score", "knn_score", "bm25_score")


@dataclass
class Tier0Finding:
    short_circuit: bool  # True -> gate should ABSTAIN without claim-level checks
    flags: list[str] = field(default_factory=list)  # non-fatal low-confidence signals
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def _chunk_score(chunk: object, priority: tuple[str, ...]) -> float:
    meta = getattr(chunk, "metadata", {}) or {}
    for name in priority:
        val = getattr(chunk, "score", None) if name == "score" else meta.get(name)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 0.0


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _query_context_overlap(query: str, texts: list[str]) -> float:
    q = _tokens(query)
    if not q:
        return 1.0
    ctx: set[str] = set()
    for t in texts:
        ctx |= _tokens(t)
    return len(q & ctx) / len(q)


def run_tier0(context: RetrievalContext, config: Tier0Config | None = None) -> Tier0Finding:
    cfg = config or Tier0Config()
    chunks = list(context.chunks)

    # 1. Empty retrieval -> always short-circuit to ABSTAIN.
    if not chunks:
        return Tier0Finding(
            short_circuit=True,
            flags=["empty_retrieval"],
            reason="retrieval returned no chunks",
        )

    scores = sorted((_chunk_score(c, cfg.score_field_priority) for c in chunks), reverse=True)
    top = scores[0]
    rest_max = scores[1] if len(scores) > 1 else 0.0
    overlap = _query_context_overlap(context.query, [getattr(c, "text", "") for c in chunks[:5]])

    finding = Tier0Finding(
        short_circuit=False,
        metrics={
            "top_score": top,
            "second_score": rest_max,
            "score_gap": top - rest_max,
            "query_context_overlap": overlap,
        },
    )

    # 2. Score floor.
    if cfg.min_top_score is not None and top < cfg.min_top_score:
        finding.flags.append("low_top_score")

    # 3. Score gap — top result dominates or everything clusters near the floor.
    if cfg.score_gap_threshold is not None and len(scores) > 1:
        if (top - rest_max) > cfg.score_gap_threshold:
            finding.flags.append("large_score_gap")
        elif cfg.min_top_score is not None and all(s <= cfg.min_top_score for s in scores):
            finding.flags.append("scores_clustered_low")

    # 4. Query-context overlap — retrieval returned results but they're off-topic.
    if cfg.min_overlap > 0.0 and overlap < cfg.min_overlap:
        finding.flags.append("low_query_context_overlap")
        finding.short_circuit = True
        finding.reason = f"query-context overlap {overlap:.2f} below floor {cfg.min_overlap:.2f}"

    if finding.flags and not finding.reason:
        finding.reason = "low-confidence retrieval: " + ", ".join(finding.flags)
    return finding
