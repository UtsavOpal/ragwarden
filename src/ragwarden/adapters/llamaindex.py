# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex adapter (Build Spec Section 13.3).

Converts ``NodeWithScore`` objects into
:class:`~ragwarden.contracts.RetrievedChunk`. Duck-typed — reads ``.score`` and
``.node`` (``.get_content()``/``.text``, ``.metadata``, ``.node_id``) — so no
``llama-index-core`` import.
"""

from __future__ import annotations

from ragwarden.models import Chunk, Context

__all__ = ["from_llamaindex_nodes"]


def _content(node: object) -> str:
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            return str(getter())
        except TypeError:
            pass
    return str(getattr(node, "text", "") or "")


def from_llamaindex_nodes(
    query: str,
    nodes: list,
    *,
    retrieval_method: str = "unknown",
    default_score: float = 0.0,
) -> Context:
    """Build a RetrievalContext from a list of LlamaIndex ``NodeWithScore``
    (or bare ``TextNode``) objects."""
    chunks: list[Chunk] = []
    for i, nws in enumerate(nodes):
        node = getattr(nws, "node", nws)
        score = getattr(nws, "score", None)
        metadata = dict(getattr(node, "metadata", {}) or {})
        source_id = (
            getattr(node, "node_id", None)
            or getattr(node, "id_", None)
            or metadata.get("source")
            or f"li-{i}"
        )
        chunks.append(
            Chunk(
                text=_content(node),
                score=float(score) if score is not None else default_score,
                source_id=str(source_id),
                metadata=metadata,
            )
        )
    return Context(query=query, chunks=chunks, retrieval_method=retrieval_method)
