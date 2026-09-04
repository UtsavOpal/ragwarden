# SPDX-License-Identifier: Apache-2.0
"""Docling adapter (Build Spec Section 13.2).

Docling is a parsing/chunking tool, not a retriever — so this adapter is about
**metadata enrichment**: it lifts Docling's structural metadata (page numbers,
section headings, document name, bounding boxes) into
``RetrievedChunk.metadata`` so chunk provenance survives into the gate's
evidence trail (``ClaimVerdict.supporting_evidence`` and downstream policy rules
like "trust chunks from a given section more").

Duck-typed against Docling's ``chunk`` / ``meta`` objects — no ``docling-core``
import required.
"""

from __future__ import annotations

from typing import Any

from ragwarden.models import Chunk

__all__ = ["docling_provenance", "enrich_chunk_from_docling"]


def docling_provenance(docling_meta: Any) -> dict[str, Any]:
    """Extract a flat provenance dict from a Docling chunk-meta object or dict."""
    if docling_meta is None:
        return {}
    get = (
        docling_meta.get
        if isinstance(docling_meta, dict)
        else lambda k, d=None: getattr(docling_meta, k, d)
    )

    prov: dict[str, Any] = {}
    headings = get("headings")
    if headings:
        prov["headings"] = list(headings)
        prov["section"] = headings[-1]
    doc_items = get("doc_items") or []
    pages: list[int] = []
    for item in doc_items:
        for p in getattr(item, "prov", None) or (
            item.get("prov", []) if isinstance(item, dict) else []
        ):
            page_no = getattr(p, "page_no", None) if not isinstance(p, dict) else p.get("page_no")
            if page_no is not None:
                pages.append(int(page_no))
    if pages:
        prov["page_no"] = pages[0]
        prov["pages"] = sorted(set(pages))
    for key in ("origin", "filename", "doc_name", "mimetype", "caption"):
        val = get(key)
        if val is not None:
            prov[key] = val
    return prov


def enrich_chunk_from_docling(chunk: Chunk, docling_meta: Any) -> Chunk:
    """Return ``chunk`` with Docling structural metadata merged into
    ``chunk.metadata`` (existing keys win)."""
    provenance = docling_provenance(docling_meta)
    merged = {**provenance, **chunk.metadata}
    return Chunk(text=chunk.text, score=chunk.score, source_id=chunk.source_id, metadata=merged)
