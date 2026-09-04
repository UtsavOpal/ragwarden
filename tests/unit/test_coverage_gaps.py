# SPDX-License-Identifier: Apache-2.0
"""Phase 12: close small coverage gaps in pure-logic paths."""

from __future__ import annotations

import pytest

from ragwarden.adapters.chroma import distance_to_similarity, from_chroma_query_result
from ragwarden.adapters.langchain import from_langchain_documents
from ragwarden.adapters.llamaindex import from_llamaindex_nodes
from ragwarden.cascade.tier0_heuristics import Tier0Config, run_tier0
from ragwarden.cascade.tier1_entailment import Tier1Outcome, ensemble_verdicts
from ragwarden.contracts import ClaimStatus, ClaimVerdict
from ragwarden.detectors import get_detector
from ragwarden.models import Chunk, Context
from ragwarden.observability.otel import otel_available
from ragwarden.scoring import get_scoring_strategy


def test_distance_to_similarity_all_metrics() -> None:
    assert distance_to_similarity(1.0, "l2") == 0.5
    assert distance_to_similarity(0.0, "ip") == 1.0
    assert distance_to_similarity(0.3, "manhattan") == pytest.approx(0.7)  # unknown -> 1-d


def test_chroma_result_without_nested_lists() -> None:
    ctx = from_chroma_query_result(
        "q", {"ids": ["a"], "documents": ["doc a"], "distances": [0.1], "metadatas": [None]}
    )
    assert ctx.chunks[0].source_id == "a"


def test_langchain_falls_back_to_default_score() -> None:
    class _Doc:
        page_content = "x"
        metadata = {}  # noqa: RUF012 - trivial test stub

    ctx = from_langchain_documents("q", [_Doc()], default_score=0.33)
    assert ctx.chunks[0].score == 0.33
    assert ctx.chunks[0].source_id == "lc-0"


def test_llamaindex_get_content_typeerror_fallback() -> None:
    class _Node:
        text = "plain text"
        node_id = "n1"
        metadata = {}  # noqa: RUF012 - trivial test stub

        def get_content(self, extra):  # wrong signature -> TypeError -> .text fallback
            return "unused"

    ctx = from_llamaindex_nodes("q", [_Node()])
    assert ctx.chunks[0].text == "plain text"


def test_tier0_chunk_score_handles_unparseable_metadata() -> None:
    cfg = Tier0Config(min_top_score=0.5, score_field_priority=("bm25_score", "score"))
    ctx = Context(
        query="q",
        chunks=[Chunk("t", score=0.9, metadata={"bm25_score": "not-a-number"})],
    )
    finding = run_tier0(ctx, cfg)
    assert "low_top_score" not in finding.flags  # falls through to .score = 0.9


def test_ensemble_single_verdict_passthrough() -> None:
    v = ClaimVerdict("c", ClaimStatus.SUPPORTED, 0.9, 1)
    assert ensemble_verdicts([v], "majority") is v


def test_tier1outcome_all_verdicts() -> None:
    out = Tier1Outcome(
        resolved=[ClaimVerdict("a", ClaimStatus.SUPPORTED, 0.9, 1)],
        unresolved=[ClaimVerdict("b", ClaimStatus.UNRESOLVED, 0.4, 1)],
    )
    assert len(out.all_verdicts) == 2


def test_get_scoring_strategy_unknown() -> None:
    with pytest.raises(ValueError, match="unknown scoring strategy"):
        get_scoring_strategy("does-not-exist")


def test_get_detector_unknown() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        get_detector("does-not-exist")


def test_otel_available_is_bool() -> None:
    assert isinstance(otel_available(), bool)
