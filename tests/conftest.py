# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures. All fixtures here are offline — no model downloads."""

from __future__ import annotations

import pytest

from ragwarden.models import Answer, Chunk, Context


@pytest.fixture
def grounded_context() -> Context:
    return Context(
        query="When was the Eiffel Tower completed?",
        chunks=[
            Chunk(
                text="The Eiffel Tower was completed in 1889 for the World's Fair.",
                score=0.92,
                source_id="doc-eiffel",
                metadata={"bm25_score": 14.2, "knn_score": 0.88, "authority": "high"},
            ),
            Chunk(
                text="It stands 330 metres tall on the Champ de Mars in Paris.",
                score=0.71,
                source_id="doc-eiffel",
                metadata={"bm25_score": 9.1, "knn_score": 0.63},
            ),
        ],
        retrieval_method="hybrid",
    )


@pytest.fixture
def grounded_answer() -> Answer:
    return Answer(text="The Eiffel Tower was completed in 1889.")


@pytest.fixture
def empty_context() -> Context:
    return Context(query="What is the capital of Atlantis?", chunks=[], retrieval_method="hybrid")
