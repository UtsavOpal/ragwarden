# SPDX-License-Identifier: Apache-2.0
"""Phase 1 acceptance: every Tier 0 heuristic, triggering and non-triggering."""

from __future__ import annotations

from ragwarden.cascade.tier0_heuristics import Tier0Config, run_tier0
from ragwarden.models import Chunk, Context


def _ctx(chunks: list[Chunk], query: str = "when was the eiffel tower completed") -> Context:
    return Context(query=query, chunks=chunks, retrieval_method="hybrid")


def test_empty_retrieval_short_circuits() -> None:
    finding = run_tier0(_ctx([]))
    assert finding.short_circuit is True
    assert "empty_retrieval" in finding.flags


def test_non_empty_retrieval_does_not_short_circuit_by_default() -> None:
    finding = run_tier0(_ctx([Chunk("the eiffel tower was completed in 1889", score=0.9)]))
    assert finding.short_circuit is False
    assert finding.flags == []


def test_score_floor_triggers_and_clears() -> None:
    cfg = Tier0Config(min_top_score=0.5)
    low = run_tier0(_ctx([Chunk("eiffel tower 1889", score=0.2)]), cfg)
    assert "low_top_score" in low.flags
    high = run_tier0(_ctx([Chunk("eiffel tower 1889", score=0.8)]), cfg)
    assert "low_top_score" not in high.flags


def test_score_gap_triggers() -> None:
    cfg = Tier0Config(score_gap_threshold=0.4)
    finding = run_tier0(
        _ctx([Chunk("eiffel tower 1889", score=0.95), Chunk("unrelated", score=0.1)]), cfg
    )
    assert "large_score_gap" in finding.flags


def test_score_gap_does_not_trigger_when_clustered() -> None:
    cfg = Tier0Config(score_gap_threshold=0.4)
    finding = run_tier0(
        _ctx([Chunk("eiffel tower 1889", score=0.62), Chunk("eiffel tower paris", score=0.58)]), cfg
    )
    assert "large_score_gap" not in finding.flags


def test_query_context_overlap_short_circuits_when_off_topic() -> None:
    cfg = Tier0Config(min_overlap=0.5)
    finding = run_tier0(_ctx([Chunk("bananas are a good source of potassium", score=0.9)]), cfg)
    assert "low_query_context_overlap" in finding.flags
    assert finding.short_circuit is True


def test_query_context_overlap_passes_when_on_topic() -> None:
    cfg = Tier0Config(min_overlap=0.5)
    finding = run_tier0(
        _ctx([Chunk("the eiffel tower was completed in 1889 in paris", score=0.9)]), cfg
    )
    assert "low_query_context_overlap" not in finding.flags
    assert finding.short_circuit is False


def test_reads_per_method_scores_from_metadata() -> None:
    cfg = Tier0Config(min_top_score=0.5, score_field_priority=("bm25_score",))
    finding = run_tier0(
        _ctx([Chunk("eiffel tower 1889", score=0.99, metadata={"bm25_score": 0.1})]), cfg
    )
    assert "low_top_score" in finding.flags
