# SPDX-License-Identifier: Apache-2.0
"""Startup warmup / readiness check.

Loading a Tier-1 model (downloading weights, allocating GPU memory, JIT-ing a
kernel) can fail for reasons that have nothing to do with any particular
request — a cold Hugging Face cache, a network blip, an OOM. If that failure
first happens on a live request, the caller pays for it. ``warmup()`` loads
every configured Tier-1 detector and runs one throwaway check *now*, so those
failures surface at process startup or in a Kubernetes readiness probe —
before the process is added to a load balancer's rotation — not mid-request.

    # at process startup, before serving traffic:
    report = warmup(policy)
    if not report.ok:
        raise SystemExit(f"ragwarden warmup failed: {report.errors}")

    # readiness probe (e.g. `ragwarden warmup --policy prod.yaml`):
    $ ragwarden warmup --policy prod.yaml   # exit 0 = ready, exit 1 = not ready
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ragwarden.contracts import RetrievedChunk
from ragwarden.detectors.base import Detector, MissingExtraError
from ragwarden.models import Chunk
from ragwarden.policy import Policy

__all__ = ["WarmupReport", "WarmupResult", "warmup"]

_PROBE_CLAIM = "The sample document was published in 2020."
_PROBE_EVIDENCE: list[RetrievedChunk] = [
    Chunk(text="The sample document was published in 2020.", score=1.0, source_id="warmup")
]


@dataclass
class WarmupResult:
    detector_name: str
    ok: bool
    latency_ms: float
    error: str | None = None


@dataclass
class WarmupReport:
    results: list[WarmupResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def errors(self) -> list[str]:
        return [f"{r.detector_name}: {r.error}" for r in self.results if not r.ok]


def _warm_one(name: str) -> WarmupResult:
    from ragwarden.detectors import get_detector

    t0 = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000

    try:
        detector: Detector = get_detector(name)
        detector.check(_PROBE_CLAIM, _PROBE_EVIDENCE)
    except MissingExtraError as exc:
        return WarmupResult(name, ok=False, latency_ms=elapsed_ms(), error=str(exc))
    except Exception as exc:
        return WarmupResult(
            name, ok=False, latency_ms=elapsed_ms(), error=f"{type(exc).__name__}: {exc}"
        )
    return WarmupResult(name, ok=True, latency_ms=elapsed_ms())


def warmup(
    policy: Policy | None = None, *, detector_names: list[str] | None = None
) -> WarmupReport:
    """Load and probe every Tier-1 detector named in ``policy`` (or
    ``detector_names`` explicitly), once, synchronously.

    Returns a report rather than raising — the caller decides whether a failed
    warmup is fatal (typically: yes, for a readiness probe; the CLI does).
    """
    pol = policy or Policy.default()
    if detector_names is not None:
        names = detector_names
    else:
        names = [pol.tier1.detector, *pol.tier1.ensemble]
    ordered = list(dict.fromkeys(names))  # dedupe, keep first-seen order
    return WarmupReport(results=[_warm_one(name) for name in ordered])
