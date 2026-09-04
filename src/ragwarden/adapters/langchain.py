# SPDX-License-Identifier: Apache-2.0
"""LangChain adapter (Build Spec Section 13.3).

Converts LangChain ``Document`` objects (optionally paired with a relevance
score) into :class:`~ragwarden.contracts.RetrievedChunk`. Duck-typed — only
needs ``.page_content`` and ``.metadata`` — so no ``langchain-core`` import.
"""

from __future__ import annotations

from typing import Any

from ragwarden.models import Chunk, Context

__all__ = ["from_langchain_documents"]

_SOURCE_KEYS = ("source_id", "id", "_id", "source", "file_path", "doc_id")


def _source_id(metadata: dict[str, Any], index: int) -> str:
    for key in _SOURCE_KEYS:
        if metadata.get(key):
            return str(metadata[key])
    return f"lc-{index}"


def from_langchain_documents(
    query: str,
    documents: list,
    *,
    retrieval_method: str = "unknown",
    score_key: str = "score",
    default_score: float = 0.0,
) -> Context:
    """Build a RetrievalContext from LangChain ``Document`` objects.

    ``documents`` may be a list of ``Document`` or of ``(Document, score)``
    tuples (as returned by ``similarity_search_with_relevance_scores``). When a
    tuple score is absent, ``metadata[score_key]`` is used, else ``default_score``.
    """
    chunks: list[Chunk] = []
    for i, item in enumerate(documents):
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
        else:
            doc, score = item, None
        metadata = dict(getattr(doc, "metadata", {}) or {})
        if score is None:
            score = metadata.get(score_key, default_score)
        chunks.append(
            Chunk(
                text=getattr(doc, "page_content", "") or "",
                score=float(score) if score is not None else default_score,
                source_id=_source_id(metadata, i),
                metadata=metadata,
            )
        )
    return Context(query=query, chunks=chunks, retrieval_method=retrieval_method)
