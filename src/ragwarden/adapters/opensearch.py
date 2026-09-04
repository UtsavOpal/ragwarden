# SPDX-License-Identifier: Apache-2.0
"""OpenSearch adapter (Build Spec Section 13.1) — priority adapter.

Builds a :class:`~ragwarden.contracts.RetrievalContext` from a hybrid BM25+kNN
OpenSearch query response, preserving *both* raw score types in chunk metadata
so Tier 0 heuristics can reason about them independently, not just a fused score.

Operates on the raw response dict — no ``opensearch-py`` import needed. The
``opensearch`` extra is only for the user's own client code.

**Score normalization.** OpenSearch hybrid queries fuse sub-scores via the
``normalization-processor`` in a search pipeline. Whether ``_score`` is the
normalized fused value or a raw single-method score changes how Tier 0's
``min_top_score`` / ``score_gap_threshold`` should be tuned — a mismatch here
silently breaks those heuristics. Pass ``scores_are_normalized`` accordingly; it
is recorded in ``chunk.metadata['scores_normalized']``.
"""

from __future__ import annotations

from typing import Any

from ragwarden.models import Chunk, Context

__all__ = ["from_opensearch_hybrid_response"]


def _dig(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return None


def from_opensearch_hybrid_response(
    query: str,
    response: dict,
    *,
    bm25_score_field: str | None = "bm25_score",
    knn_score_field: str | None = "knn_score",
    text_field: str = "text",
    source_id_field: str = "_id",
    scores_are_normalized: bool = True,
    retrieval_method: str = "hybrid",
) -> Context:
    """Build a RetrievalContext from a hybrid BM25+kNN OpenSearch response.

    ``bm25_score_field`` / ``knn_score_field`` are looked up in each hit's
    ``_source`` (or the hit itself) — set them to fields your pipeline writes, or
    ``None`` to skip.
    """
    hits = _dig(response, "hits") or {}
    hit_list = _dig(hits, "hits") or (hits if isinstance(hits, list) else [])

    chunks: list[Chunk] = []
    for i, hit in enumerate(hit_list):
        src = hit.get("_source", hit) if isinstance(hit, dict) else {}
        text = src.get(text_field) or hit.get(text_field) or hit.get("text") or ""
        metadata: dict[str, Any] = {k: v for k, v in src.items() if k != text_field}
        metadata["scores_normalized"] = scores_are_normalized
        metadata["opensearch_index"] = hit.get("_index")
        for name, field in (("bm25_score", bm25_score_field), ("knn_score", knn_score_field)):
            if field:
                val = src.get(field, hit.get(field))
                if val is not None:
                    metadata[name] = float(val)
        source_id = str(
            hit.get(source_id_field) or src.get("source_id") or src.get("id") or f"os-{i}"
        )
        fused = hit.get("_score")
        chunks.append(
            Chunk(
                text=str(text),
                score=float(fused) if fused is not None else 0.0,
                source_id=source_id,
                metadata=metadata,
            )
        )
    return Context(query=query, chunks=chunks, retrieval_method=retrieval_method)
