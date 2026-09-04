# SPDX-License-Identifier: Apache-2.0
"""Opt-in check that the real RAGTruth loader works: `pytest -m model`.

Downloads ~20 MB of parquet on first run.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.model, pytest.mark.benchmark]


def test_load_ragtruth_shape() -> None:
    pytest.importorskip("datasets")
    from ragwarden.benchmark.datasets import load_ragtruth

    rows = load_ragtruth("test", limit=25, shuffle_seed=7)
    assert len(rows) == 25
    assert all(r.query is not None and r.output for r in rows)
    assert all(r.context_chunks for r in rows)
    assert any(r.is_hallucinated for r in rows) and any(not r.is_hallucinated for r in rows)
    # span offsets, when present, land inside the output text
    for r in rows:
        for start, end in r.hallucinated_spans:
            assert 0 <= start < end <= len(r.output)


def test_task_type_filter() -> None:
    pytest.importorskip("datasets")
    from ragwarden.benchmark.datasets import load_ragtruth

    rows = load_ragtruth("test", task_types=["QA"], limit=10, shuffle_seed=1)
    assert rows and all(r.task_type == "QA" for r in rows)
