# SPDX-License-Identifier: Apache-2.0
"""Threshold calibration (Build Spec Section 15 Phase 3, Section 16).

Runs the gate once per labeled sample to capture its reliability score, then
sweeps the ``allow`` threshold and reports the hallucination-catch vs
over-abstention trade-off at each point. Calibration against the adopter's own
traffic is the product experience, not optional polish.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ragwarden.benchmark.datasets import RawSample, load_jsonl
from ragwarden.benchmark.metrics import response_level_metrics
from ragwarden.detectors.base import Detector
from ragwarden.gate import gate
from ragwarden.models import Answer, Context
from ragwarden.policy import Policy

__all__ = ["CalibrationPoint", "CalibrationReport", "run_calibration"]


@dataclass
class CalibrationPoint:
    threshold: float
    precision: float  # of "flagged" predictions, how many were truly hallucinated
    recall: float  # of truly hallucinated, how many were flagged
    f1: float
    flag_rate: float  # fraction of all answers the gate would not ALLOW
    false_abstention_rate: float  # clean answers wrongly flagged


@dataclass
class CalibrationReport:
    n_samples: int
    detector: str
    score_distribution: dict[str, float]
    curve: list[dict]
    recommended_allow_threshold: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def run_calibration(
    dataset: str,
    *,
    policy: Policy | None = None,
    detectors: list[Detector] | None = None,
    thresholds: Sequence[float] | None = None,
    samples: Sequence[RawSample] | None = None,
    target_recall: float = 0.9,
    output_path: str | Path | None = None,
) -> CalibrationReport:
    pol = policy or Policy.default()
    raw = list(samples) if samples is not None else list(load_jsonl(dataset))
    if not raw:
        raise ValueError("no samples to calibrate on")

    scored: list[tuple[float, bool]] = []
    for s in raw:
        ctx = Context(query=s.query, chunks=s.context_chunks, retrieval_method="calibration")
        res = gate(ctx, Answer(text=s.output), policy=pol, detectors=detectors)
        scored.append((res.reliability_score, s.is_hallucinated))

    grid = thresholds or [round(x / 100, 2) for x in range(50, 100, 2)]
    curve: list[CalibrationPoint] = []
    for tau in grid:
        # "flagged" = gate would not ALLOW at this allow-threshold
        y_pred = [score < tau for score, _ in scored]
        y_true = [is_hallu for _, is_hallu in scored]
        m = response_level_metrics(y_true, y_pred)
        flagged = sum(y_pred)
        clean_flagged = sum(1 for (score, is_h) in scored if score < tau and not is_h)
        n_clean = sum(1 for _, is_h in scored if not is_h)
        curve.append(
            CalibrationPoint(
                threshold=tau,
                precision=round(m.precision, 4),
                recall=round(m.recall, 4),
                f1=round(m.f1, 4),
                flag_rate=round(flagged / len(scored), 4),
                false_abstention_rate=round(clean_flagged / n_clean, 4) if n_clean else 0.0,
            )
        )

    # Recommend the lowest threshold that still hits target recall (fewest false
    # abstentions while catching enough hallucinations); fall back to best F1.
    meeting = [p for p in curve if p.recall >= target_recall]
    if meeting:
        recommended = min(meeting, key=lambda p: p.false_abstention_rate).threshold
    else:
        recommended = max(curve, key=lambda p: p.f1).threshold

    all_scores = sorted(score for score, _ in scored)
    n = len(all_scores)
    dist = {
        "min": round(all_scores[0], 4),
        "p25": round(all_scores[n // 4], 4),
        "median": round(all_scores[n // 2], 4),
        "p75": round(all_scores[3 * n // 4], 4),
        "max": round(all_scores[-1], 4),
    }

    report = CalibrationReport(
        n_samples=len(raw),
        detector=detectors[0].name if detectors else pol.tier1.detector,
        score_distribution=dist,
        curve=[asdict(p) for p in curve],
        recommended_allow_threshold=recommended,
    )
    if output_path:
        Path(output_path).write_text(report.to_json(), encoding="utf-8")
    return report
