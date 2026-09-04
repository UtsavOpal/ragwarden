# SPDX-License-Identifier: Apache-2.0
"""RagWarden command-line interface.

ragwarden benchmark --dataset ragtruth --split test --limit 500
ragwarden calibrate --dataset path/to/labeled.jsonl
ragwarden warmup --policy prod.yaml   # readiness probe: exit 0 = ready
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragwarden.detectors.base import Detector
    from ragwarden.policy import Policy

__all__ = ["main"]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", help="path to a policy.yaml (defaults to Policy.default())")
    p.add_argument(
        "--detector",
        default=None,
        help="Tier-1 detector name to use (overrides the policy); e.g. nli, hhem, keyword_stub",
    )


def _load_policy(path: str | None) -> Policy:
    from ragwarden.policy import Policy

    return Policy.from_yaml(path) if path else Policy.default()


def _detectors(name: str | None) -> list[Detector] | None:
    if not name:
        return None
    from ragwarden.detectors import get_detector

    return [get_detector(name)]


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from ragwarden.benchmark import run_benchmark

    pol = _load_policy(args.policy)
    if args.detector:
        pol.tier1.detector = args.detector
    report = run_benchmark(
        dataset=args.dataset,
        split=args.split,
        task_types=args.task_types.split(",") if args.task_types else None,
        limit=args.limit,
        shuffle_seed=args.shuffle_seed,
        policy=pol,
        detectors=_detectors(args.detector),
        output_dir=args.output_dir,
    )
    print(f"n={report.n_samples}  detector={report.detector}")
    rl = report.response_level
    print(
        f"response-level  P={rl['precision']:.3f}  R={rl['recall']:.3f}  "
        f"F1={rl['f1']:.3f}  acc={rl['accuracy']:.3f}"
    )
    print(f"cascade         {report.cascade['pct_resolved_by_tier']}")
    print(f"latency ms      {report.latency_ms}")
    print(f"results file    {report.environment.get('results_file', '(not written)')}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from ragwarden.benchmark import run_calibration

    report = run_calibration(
        dataset=args.dataset,
        policy=_load_policy(args.policy),
        detectors=_detectors(args.detector),
        output_path=args.output,
    )
    print(f"n={report.n_samples}  score distribution={report.score_distribution}")
    print(f"{'thr':>6} {'prec':>7} {'recall':>7} {'f1':>7} {'flag%':>7} {'false-abstain%':>15}")
    for p in report.curve:
        flag = p["flag_rate"] * 100
        fabs = p["false_abstention_rate"] * 100
        print(
            f"{p['threshold']:>6.2f} {p['precision']:>7.3f} {p['recall']:>7.3f} "
            f"{p['f1']:>7.3f} {flag:>6.1f}% {fabs:>14.1f}%"
        )
    print(f"\nrecommended allow threshold: {report.recommended_allow_threshold}")
    return 0


def _cmd_warmup(args: argparse.Namespace) -> int:
    from ragwarden.warmup import warmup

    pol = _load_policy(args.policy)
    names = [args.detector] if args.detector else None
    report = warmup(pol, detector_names=names)
    for r in report.results:
        status = "OK  " if r.ok else "FAIL"
        print(f"[{status}] {r.detector_name:<16} {r.latency_ms:>8.1f}ms  {r.error or ''}")
    if not report.ok:
        print(f"\nwarmup failed: {'; '.join(report.errors)}", file=sys.stderr)
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragwarden", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    b = sub.add_parser("benchmark", help="run the cascade against a public dataset")
    b.add_argument("--dataset", default="ragtruth")
    b.add_argument("--split", default="test")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--shuffle-seed", type=int, default=None, help="shuffle before applying --limit")
    b.add_argument("--task-types", default=None, help="comma-separated filter, e.g. QA,Summary")
    b.add_argument("--output-dir", default="benchmarks/results")
    _add_common(b)
    b.set_defaults(func=_cmd_benchmark)

    c = sub.add_parser("calibrate", help="threshold-vs-metric curves on your labeled data")
    c.add_argument("--dataset", required=True, help="path to a labeled .jsonl file")
    c.add_argument("--output", default=None, help="write the JSON report here")
    _add_common(c)
    c.set_defaults(func=_cmd_calibrate)

    w = sub.add_parser("warmup", help="load Tier-1 detector(s) and probe them; readiness check")
    _add_common(w)
    w.set_defaults(func=_cmd_warmup)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
