# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner: execute the cascade over a labeled dataset and report
reproducible precision / recall / F1 plus cascade cost-efficiency stats.

The number this produces is the actual output of the actual code as it exists —
never aspirational. It is the baseline the project improves from and is honest
about in the README.
"""

from __future__ import annotations

import json
import platform
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ragwarden import __version__
from ragwarden.benchmark.datasets import RawSample, load_ragtruth
from ragwarden.benchmark.metrics import response_level_metrics
from ragwarden.contracts import ClaimStatus, GateAction
from ragwarden.detectors.base import Detector
from ragwarden.gate import gate
from ragwarden.models import Answer, Context
from ragwarden.policy import Policy

__all__ = ["BenchmarkReport", "BenchmarkSample", "run_benchmark"]

_HALLUCINATION_STATUSES = {ClaimStatus.CONTRADICTED, ClaimStatus.UNSUPPORTED}

DatasetLoader = Callable[[], Sequence[RawSample]]


@dataclass
class BenchmarkSample:
    id: str
    task_type: str
    true_hallucinated: bool
    pred_hallucinated: bool
    action: str
    reliability_score: float
    claims_total: int
    claims_by_tier: dict[int, int]
    latency_ms: float


@dataclass
class BenchmarkReport:
    dataset: str
    split: str
    ragwarden_version: str
    timestamp: str
    n_samples: int
    detector: str
    policy_summary: dict
    unresolved_counts_as_hallucination: bool
    response_level: dict
    per_task_response_level: dict
    cascade: dict
    latency_ms: dict
    environment: dict
    samples: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _predict_hallucinated(
    verdicts: list, action: GateAction, *, unresolved_is_hallucination: bool
) -> bool:
    if verdicts:
        statuses = {v.status for v in verdicts}
        return bool(statuses & _HALLUCINATION_STATUSES) or (
            unresolved_is_hallucination and ClaimStatus.UNRESOLVED in statuses
        )
    # No claim-level verdicts (Tier-0-only or short-circuit): trust the action.
    return action is not GateAction.ALLOW


def run_benchmark(
    dataset: str = "ragtruth",
    *,
    split: str = "test",
    task_types: list[str] | None = None,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    policy: Policy | None = None,
    detectors: list[Detector] | None = None,
    unresolved_counts_as_hallucination: bool = True,
    output_dir: str | Path = "benchmarks/results",
    samples: Sequence[RawSample] | None = None,
    write: bool = True,
) -> BenchmarkReport:
    """Run the cascade over ``dataset`` and write a results file.

    ``samples`` overrides dataset loading entirely (used by tests). ``detectors``
    is passed straight through to :func:`ragwarden.gate.gate`.
    """
    pol = policy or Policy.default()

    if samples is None:
        if dataset == "ragtruth":
            raw = load_ragtruth(
                split, task_types=task_types, limit=limit, shuffle_seed=shuffle_seed
            )
        else:
            raise ValueError(f"unknown dataset {dataset!r} (known: ragtruth; or pass samples=)")
    else:
        raw = list(samples)

    results: list[BenchmarkSample] = []
    tier_counter: Counter[int] = Counter()
    detector_name = (
        detectors[0].name
        if detectors
        else (pol.tier1.detector if pol.tier1.enabled else "tier0-only")
    )

    for s in raw:
        ctx = Context(query=s.query, chunks=s.context_chunks, retrieval_method="benchmark")
        ans = Answer(text=s.output)
        t0 = time.perf_counter()
        res = gate(ctx, ans, policy=pol, detectors=detectors)
        latency = (time.perf_counter() - t0) * 1000.0

        by_tier: Counter[int] = Counter(v.resolved_at_tier for v in res.claim_verdicts)
        tier_counter.update(by_tier)
        results.append(
            BenchmarkSample(
                id=s.id,
                task_type=s.task_type,
                true_hallucinated=s.is_hallucinated,
                pred_hallucinated=_predict_hallucinated(
                    res.claim_verdicts,
                    res.action,
                    unresolved_is_hallucination=unresolved_counts_as_hallucination,
                ),
                action=res.action.value,
                reliability_score=round(res.reliability_score, 4),
                claims_total=len(res.claim_verdicts),
                claims_by_tier={int(k): int(v) for k, v in by_tier.items()},
                latency_ms=round(latency, 2),
            )
        )

    y_true = [r.true_hallucinated for r in results]
    y_pred = [r.pred_hallucinated for r in results]
    overall = response_level_metrics(y_true, y_pred)

    per_task: dict[str, dict] = {}
    for task in sorted({r.task_type for r in results}):
        idx = [i for i, r in enumerate(results) if r.task_type == task]
        per_task[task] = response_level_metrics(
            [y_true[i] for i in idx], [y_pred[i] for i in idx]
        ).as_dict()

    total_claims = sum(tier_counter.values())
    cascade = {
        "total_claims_resolved": total_claims,
        "resolved_by_tier": {int(k): int(v) for k, v in sorted(tier_counter.items())},
        "pct_resolved_by_tier": {
            int(k): round(100.0 * v / total_claims, 2) for k, v in sorted(tier_counter.items())
        }
        if total_claims
        else {},
        "tier3_escalation_rate_pct": round(100.0 * tier_counter.get(3, 0) / total_claims, 2)
        if total_claims
        else 0.0,
    }
    latencies = sorted(r.latency_ms for r in results)
    latency_stats = _latency_stats(latencies)

    report = BenchmarkReport(
        dataset=raw[0].source_dataset if raw else dataset,
        split=split,
        ragwarden_version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_samples=len(results),
        detector=detector_name,
        policy_summary=_policy_summary(pol),
        unresolved_counts_as_hallucination=unresolved_counts_as_hallucination,
        response_level=overall.as_dict(),
        per_task_response_level=per_task,
        cascade=cascade,
        latency_ms=latency_stats,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        samples=[asdict(r) for r in results],
    )

    if write and results:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out / f"{report.dataset}_{split}_{detector_name}_{stamp}.json"
        path.write_text(report.to_json(), encoding="utf-8")
        report.environment["results_file"] = str(path)

    return report


def _latency_stats(sorted_latencies: list[float]) -> dict[str, float]:
    if not sorted_latencies:
        return {}
    n = len(sorted_latencies)

    def pct(p: float) -> float:
        return round(sorted_latencies[min(n - 1, int(p * n))], 2)

    return {
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "mean": round(sum(sorted_latencies) / n, 2),
        "max": round(sorted_latencies[-1], 2),
    }


def _policy_summary(pol: Policy) -> dict:
    return {
        "claim_weighting": pol.claim_weighting,
        "fail_mode": pol.fail_mode,
        "thresholds": asdict(pol.thresholds),
        "tier1": {
            "enabled": pol.tier1.enabled,
            "detector": pol.tier1.detector,
            "confidence_threshold": pol.tier1.confidence_threshold,
        },
        "tier2_enabled": pol.tier2.enabled,
        "tier3_enabled": pol.tier3.enabled,
    }
