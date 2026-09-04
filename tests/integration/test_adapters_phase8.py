# SPDX-License-Identifier: Apache-2.0
"""Phase 8: retrieval adapters — recorded-fixture tests, each feeding gate()."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ragwarden import gate
from ragwarden.adapters.chroma import distance_to_similarity, from_chroma_query_result
from ragwarden.adapters.docling import docling_provenance, enrich_chunk_from_docling
from ragwarden.adapters.langchain import from_langchain_documents
from ragwarden.adapters.llamaindex import from_llamaindex_nodes
from ragwarden.adapters.opensearch import from_opensearch_hybrid_response
from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievalContext, RetrievedChunk
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk

pytestmark = pytest.mark.integration

_SUPPORT = ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.SUPPORTED, 0.95, 1))


# --- OpenSearch ------------------------------------------------------------
OPENSEARCH_HYBRID_RESPONSE = {
    "took": 12,
    "hits": {
        "total": {"value": 2},
        "max_score": 0.86,
        "hits": [
            {
                "_index": "docs",
                "_id": "doc-42",
                "_score": 0.86,  # normalized fused score
                "_source": {
                    "text": "The Eiffel Tower was completed in 1889.",
                    "bm25_score": 14.7,
                    "knn_score": 0.91,
                    "authority": "high",
                },
            },
            {
                "_index": "docs",
                "_id": "doc-7",
                "_score": 0.31,
                "_source": {
                    "text": "It stands on the Champ de Mars in Paris.",
                    "bm25_score": 6.2,
                    "knn_score": 0.44,
                },
            },
        ],
    },
}


def test_opensearch_adapter_preserves_both_score_types() -> None:
    ctx = from_opensearch_hybrid_response("Eiffel Tower?", OPENSEARCH_HYBRID_RESPONSE)
    assert isinstance(ctx, RetrievalContext)
    assert [c.source_id for c in ctx.chunks] == ["doc-42", "doc-7"]
    top = ctx.chunks[0]
    assert isinstance(top, RetrievedChunk)
    assert top.score == 0.86
    assert top.metadata["bm25_score"] == 14.7
    assert top.metadata["knn_score"] == 0.91
    assert top.metadata["scores_normalized"] is True
    assert top.metadata["authority"] == "high"
    result = gate(ctx, Answer("The Eiffel Tower was completed in 1889."), detectors=[_SUPPORT])
    assert result.claim_verdicts


# --- ChromaDB ------------------------------------------------------------
CHROMA_RESULT = {
    "ids": [["c1", "c2"]],
    "documents": [["The Eiffel Tower was completed in 1889.", "It is 330 metres tall."]],
    "distances": [[0.12, 0.55]],
    "metadatas": [[{"page": 3}, {"page": 8}]],
}


def test_chroma_adapter_converts_distance_to_similarity() -> None:
    ctx = from_chroma_query_result("Eiffel Tower?", CHROMA_RESULT, distance_metric="cosine")
    assert [c.source_id for c in ctx.chunks] == ["c1", "c2"]
    assert ctx.chunks[0].score > ctx.chunks[1].score  # closer distance -> higher score
    assert ctx.chunks[0].metadata["distance"] == 0.12
    assert ctx.chunks[0].metadata["page"] == 3


def test_distance_to_similarity_bounds() -> None:
    assert distance_to_similarity(0.0, "cosine") == 1.0
    assert 0.0 <= distance_to_similarity(2.0, "cosine") <= 1.0
    assert distance_to_similarity(0.0, "l2") == 1.0


# --- LangChain ---------------------------------------------------------
@dataclass
class _LCDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_langchain_adapter_plain_and_scored() -> None:
    plain = [_LCDoc("A", {"source": "s1"}), _LCDoc("B", {"id": "s2"})]
    ctx = from_langchain_documents("q", plain)
    assert [c.source_id for c in ctx.chunks] == ["s1", "s2"]

    scored = [(_LCDoc("A", {"source": "s1"}), 0.9), (_LCDoc("B", {"source": "s2"}), 0.4)]
    ctx2 = from_langchain_documents("q", scored, retrieval_method="hybrid")
    assert ctx2.retrieval_method == "hybrid"
    assert [c.score for c in ctx2.chunks] == [0.9, 0.4]


# --- LlamaIndex ------------------------------------------------------
@dataclass
class _LINode:
    text: str
    node_id: str
    metadata: dict = field(default_factory=dict)

    def get_content(self) -> str:
        return self.text


@dataclass
class _LINodeWithScore:
    node: _LINode
    score: float


def test_llamaindex_adapter() -> None:
    nodes = [
        _LINodeWithScore(_LINode("The tower opened in 1889.", "n1", {"page": 1}), 0.88),
        _LINodeWithScore(_LINode("It is in Paris.", "n2"), 0.51),
    ]
    ctx = from_llamaindex_nodes("q", nodes)
    assert [c.source_id for c in ctx.chunks] == ["n1", "n2"]
    assert ctx.chunks[0].text == "The tower opened in 1889."
    assert ctx.chunks[0].score == 0.88
    assert ctx.chunks[0].metadata["page"] == 1


# --- Docling ---------------------------------------------------------
@dataclass
class _Prov:
    page_no: int


@dataclass
class _DocItem:
    prov: list


@dataclass
class _DoclingMeta:
    headings: list
    doc_items: list
    filename: str


def test_docling_enrichment() -> None:
    meta = _DoclingMeta(
        headings=["Introduction", "History"],
        doc_items=[_DocItem(prov=[_Prov(page_no=4)])],
        filename="eiffel.pdf",
    )
    prov = docling_provenance(meta)
    assert prov["section"] == "History"
    assert prov["page_no"] == 4
    assert prov["filename"] == "eiffel.pdf"

    enriched = enrich_chunk_from_docling(Chunk("text", 0.9, "src", {"kept": 1}), meta)
    assert enriched.metadata["section"] == "History"
    assert enriched.metadata["kept"] == 1  # existing key preserved
    assert enriched.source_id == "src"


def test_docling_provenance_from_dict() -> None:
    assert docling_provenance({"headings": ["A"], "doc_name": "x"})["section"] == "A"
    assert docling_provenance(None) == {}
