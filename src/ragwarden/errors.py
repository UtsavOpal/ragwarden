# SPDX-License-Identifier: Apache-2.0
"""RagWarden's exception hierarchy.

``gate()`` and ``agate()`` never let a runtime failure inside the cascade escape
as an unhandled exception — see :mod:`ragwarden.resilience`. These types exist
for the failures that legitimately *are* the caller's problem: bad config, a
malformed contract, or (when the caller explicitly asks for it) a re-raised
underlying error for debugging.
"""

from __future__ import annotations

__all__ = [
    "CallableError",
    "CallableTimeoutError",
    "ConfigError",
    "DetectorError",
    "RagWardenError",
]


class RagWardenError(Exception):
    """Base class for all RagWarden-raised exceptions."""


class ConfigError(RagWardenError, ValueError):
    """An invalid :class:`~ragwarden.policy.Policy` or config value."""


class DetectorError(RagWardenError):
    """A Tier-1 detector failed to load or to score a batch."""

    def __init__(self, detector_name: str, cause: BaseException) -> None:
        super().__init__(f"detector {detector_name!r} failed: {cause}")
        self.detector_name = detector_name
        self.__cause__ = cause


class CallableTimeoutError(RagWardenError, TimeoutError):
    """A host-supplied ``generate_fn`` / ``judge_fn`` did not return in time."""

    def __init__(self, what: str, timeout_s: float) -> None:
        super().__init__(f"{what} did not return within {timeout_s:.1f}s")
        self.timeout_s = timeout_s


class CallableError(RagWardenError):
    """A host-supplied ``generate_fn`` / ``judge_fn`` raised."""

    def __init__(self, what: str, cause: BaseException) -> None:
        super().__init__(f"{what} raised {type(cause).__name__}: {cause}")
        self.__cause__ = cause
