# SPDX-License-Identifier: Apache-2.0
"""`ragwarden benchmark` (Build Spec Section 15, Phase 3).

Thin re-export. Implementation lives in :mod:`ragwarden.benchmark.runner` so the
harness is importable without going through the CLI.
"""

from __future__ import annotations

from ragwarden.benchmark.runner import BenchmarkReport, run_benchmark

__all__ = ["BenchmarkReport", "run_benchmark"]
