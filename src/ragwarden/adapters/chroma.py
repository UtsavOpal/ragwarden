# SPDX-License-Identifier: Apache-2.0
"""ChromaDB adapter (extension beyond the Build Spec, requested by the user).

Converts a raw ``collection.query(...)`` result dict into a
:class:`~ragwarden.contracts.RetrievalContext`. Chroma returns *distances*
(lower = closer); this adapter converts them to a similarity ``score`` in
``[0, 1]`` for Tier 0 and keeps the raw distance in ``metadata['distance']``.

Operates on the plain result dict — no ``chromadb`` import needed.
"""

from __future__ import annotations

from ragwarden.models import Chunk, Context

__all__ = ["distance_to_similarity", "from_chroma_query_result"]


def distance_to_similarity(distance: float, metric: str) -> float:
    metric = metric.lower()
    if metric in ("cosine", "cos"):
        return max(0.0, min(1.0, 1.0 - distance / 2.0))
    if metric in ("l2", "euclidean"):
        return 1.0 / (1.0 + max(0.0, distance))
    if metric in ("ip", "inner_product", "dot"):
        # Chroma reports IP distance as 1 - <a,b>; recover the similarity.
        return max(0.0, min(1.0, 1.0 - distance))
    return max(0.0, min(1.0, 1.0 - distance))


def from_chroma_query_result(
    query: str,
    result: dict,
    *,
    query_index: int = 0,
    distance_metric: str = "cosine",
    retrieval_method: str = "knn",
) -> Context:
    """Build a RetrievalContext from a ChromaDB ``query`` result.

    ``result`` is the dict Chroma returns: ``ids``, ``documents``, ``distances``,
    ``metadatas`` — each a list-of-lists indexed by query. ``query_index`` picks
    which query's results to use.
    """

    def row(key: str) -> list:
        val: list = result.get(key) or []
        if val and isinstance(val[0], list):
            return list(val[query_index]) if query_index < len(val) else []
        return list(val)

    ids = row("ids")
    docs = row("documents")
    distances = row("distances")
    metadatas = row("metadatas")

    chunks: list[Chunk] = []
    for i, text in enumerate(docs):
        dist = float(distances[i]) if i < len(distances) and distances[i] is not None else None
        meta = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}
        if dist is not None:
            meta["distance"] = dist
            meta["distance_metric"] = distance_metric
        chunks.append(
            Chunk(
                text=str(text),
                score=distance_to_similarity(dist, distance_metric) if dist is not None else 0.0,
                source_id=str(ids[i]) if i < len(ids) else f"chroma-{i}",
                metadata=meta,
            )
        )
    return Context(query=query, chunks=chunks, retrieval_method=retrieval_method)
