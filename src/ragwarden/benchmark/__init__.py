# SPDX-License-Identifier: Apache-2.0
"""Benchmark + calibration harness (Build Spec Section 15, Phase 3).

Internal package. The public entry points are ``ragwarden benchmark`` and
``ragwarden calibrate`` (see :mod:`ragwarden.cli`). Datasets are loaded via the
optional ``datasets`` library, which is a *benchmark-only* dependency
(``pip install 'ragwarden[bench]'``), not something end users of the gate need.
"""

from ragwarden.benchmark.calibrate import CalibrationReport, run_calibration
from ragwarden.benchmark.metrics import ClassificationMetrics, response_level_metrics
from ragwarden.benchmark.runner import BenchmarkReport, BenchmarkSample, run_benchmark

__all__ = [
    "BenchmarkReport",
    "BenchmarkSample",
    "CalibrationReport",
    "ClassificationMetrics",
    "response_level_metrics",
    "run_benchmark",
    "run_calibration",
]
