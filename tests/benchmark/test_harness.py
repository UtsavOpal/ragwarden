# SPDX-License-Identifier: Apache-2.0
"""Phase 3: benchmark + calibration harness (offline, synthetic data)."""

from __future__ import annotations

import json

import pytest

from ragwarden.benchmark.calibrate import run_calibration
from ragwarden.benchmark.datasets import RawSample, load_jsonl, split_context_into_chunks
from ragwarden.benchmark.metrics import response_level_metrics, span_token_metrics
from ragwarden.benchmark.runner import run_benchmark
from ragwarden.detectors.stub import KeywordStubDetector
from ragwarden.models import Chunk
from ragwarden.policy import Policy

pytestmark = pytest.mark.benchmark


# --- metrics -----------------------------------------------------------------
def test_response_level_metrics_perfect() -> None:
    m = response_level_metrics([True, False, True], [True, False, True])
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


def test_response_level_metrics_mixed() -> None:
    m = response_level_metrics([True, True, False, False], [True, False, True, False])
    assert m.tp == 1 and m.fn == 1 and m.fp == 1 and m.tn == 1
    assert m.precision == 0.5 and m.recall == 0.5


def test_response_level_length_mismatch() -> None:
    with pytest.raises(ValueError):
        response_level_metrics([True], [True, False])


def test_span_token_metrics() -> None:
    m = span_token_metrics([(0, 10)], [(5, 15)], text_length=20)
    assert m.tp == 5 and m.fp == 5 and m.fn == 5 and m.tn == 5


# --- context splitting -------------------------------------------------------
def test_split_context_on_passage_markers() -> None:
    ctx = "passage 1: The sky is blue.\npassage 2: Grass is green."
    chunks = split_context_into_chunks(ctx)
    assert len(chunks) == 2
    assert "sky is blue" in chunks[0].text


def test_split_context_single_blob() -> None:
    chunks = split_context_into_chunks("A single paragraph with no markers.")
    assert len(chunks) == 1


# --- jsonl loader -----------------------------------------------------------
def test_load_jsonl(tmp_path) -> None:
    p = tmp_path / "labeled.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"query": "q1", "context": ["c1", "c2"], "answer": "a1", "label": 1},
                {"query": "q2", "context": "c3", "output": "a2", "is_hallucinated": False},
            ]
        )
    )
    rows = list(load_jsonl(str(p)))
    assert len(rows) == 2
    assert rows[0].is_hallucinated is True
    assert len(rows[0].context_chunks) == 2
    assert rows[1].is_hallucinated is False


# --- runner ---------------------------------------------------------------
def _samples() -> list[RawSample]:
    ev = [Chunk("The Eiffel Tower was completed in 1889 in Paris France.", 1.0, "c0")]
    return [
        RawSample(
            "ok",
            "q",
            ev,
            "The Eiffel Tower was completed in 1889 in Paris France.",
            is_hallucinated=False,
            task_type="QA",
            source_dataset="synthetic",
        ),
        RawSample(
            "bad",
            "q",
            ev,
            "The Eiffel Tower was designed by Leonardo da Vinci in Rome.",
            is_hallucinated=True,
            task_type="QA",
            source_dataset="synthetic",
        ),
        RawSample(
            "bad2",
            "q",
            ev,
            "The tower is made entirely of solid gold bricks.",
            is_hallucinated=True,
            task_type="Summary",
            source_dataset="synthetic",
        ),
    ]


def test_run_benchmark_produces_report_and_file(tmp_path) -> None:
    pol = Policy.default()
    pol.tier1.confidence_threshold = 0.4
    report = run_benchmark(
        samples=_samples(),
        policy=pol,
        detectors=[KeywordStubDetector()],
        output_dir=tmp_path,
    )
    assert report.n_samples == 3
    assert set(report.response_level) >= {"precision", "recall", "f1", "accuracy"}
    assert "QA" in report.per_task_response_level
    assert report.cascade["total_claims_resolved"] >= 3
    # the fabricated ones should not be predicted clean
    preds = {s["id"]: s["pred_hallucinated"] for s in report.samples}
    assert preds["bad"] is True or preds["bad2"] is True
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    parsed = json.loads(files[0].read_text())
    assert parsed["ragwarden_version"]


def test_run_benchmark_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        run_benchmark(dataset="not-a-dataset")


# --- calibration --------------------------------------------------------
def test_run_calibration(tmp_path) -> None:
    pol = Policy.default()
    pol.tier1.confidence_threshold = 0.4
    report = run_calibration(
        dataset="unused",
        samples=_samples(),
        policy=pol,
        detectors=[KeywordStubDetector()],
        thresholds=[0.5, 0.7, 0.9],
        output_path=tmp_path / "calib.json",
    )
    assert len(report.curve) == 3
    assert 0.5 <= report.recommended_allow_threshold <= 0.9
    assert (tmp_path / "calib.json").exists()
