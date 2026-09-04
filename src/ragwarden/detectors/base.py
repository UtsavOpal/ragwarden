# SPDX-License-Identifier: Apache-2.0
"""The ``Detector`` protocol (Build Spec Section 7.3).

Any tier can have multiple interchangeable detector implementations. Detectors
that need a heavy dependency (``transformers``, ``torch``, ...) must lazy-import
it inside ``__init__`` and raise a clear :class:`ImportError` naming the extra —
importing this module must never pull an ML stack.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ragwarden.contracts import ClaimVerdict, RetrievedChunk

__all__ = ["BaseDetector", "Detector", "MissingExtraError", "require"]


class MissingExtraError(ImportError):
    """Raised when a detector/adapter is used without its optional extra installed."""

    def __init__(self, extra: str, package: str) -> None:
        super().__init__(
            f"'{package}' is required for this component. "
            f"Install it with: pip install 'ragwarden[{extra}]'"
        )
        self.extra = extra
        self.package = package


def require(module: str, *, extra: str, package: str | None = None) -> Any:
    """Import ``module`` lazily or raise :class:`MissingExtraError`.

    Returns ``Any`` deliberately: callers use the returned module dynamically.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - trivial
        raise MissingExtraError(extra, package or module) from exc


@runtime_checkable
class Detector(Protocol):
    """A pluggable claim-vs-evidence detector."""

    tier: int
    name: str

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        """Return a verdict for a single claim against the retrieved evidence."""
        ...

    def check_batch(self, claims: list[str], evidence: list[RetrievedChunk]) -> list[ClaimVerdict]:
        """Batched check. Implementations should override for throughput; the
        default falls back to per-claim :meth:`check`."""
        ...


class BaseDetector:
    """Optional convenience base: provides a per-claim fallback for ``check_batch``.

    Detectors may implement the :class:`Detector` protocol directly instead.
    """

    tier: int = 1
    name: str = "base"

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        raise NotImplementedError

    def check_batch(self, claims: list[str], evidence: list[RetrievedChunk]) -> list[ClaimVerdict]:
        return [self.check(c, evidence) for c in claims]
