# SPDX-License-Identifier: Apache-2.0
"""Production hardening: low-level fault-isolation primitives."""

from __future__ import annotations

import time

import pytest

from ragwarden.errors import CallableError, CallableTimeoutError, ConfigError
from ragwarden.resilience import call_with_timeout, safe_call, shutdown_executor


@pytest.fixture(autouse=True)
def _fresh_executor():
    yield
    shutdown_executor(wait=False)


def test_call_with_timeout_returns_result() -> None:
    assert call_with_timeout(lambda x: x * 2, 21, timeout_s=1.0, what="t") == 42


def test_call_with_timeout_raises_on_timeout() -> None:
    def slow(_x: int) -> int:
        time.sleep(0.3)
        return 1

    t0 = time.perf_counter()
    with pytest.raises(CallableTimeoutError):
        call_with_timeout(slow, 1, timeout_s=0.05, what="slow thing")
    # the caller must get control back near the timeout, not after the real 0.3s
    assert time.perf_counter() - t0 < 0.25


def test_call_with_timeout_wraps_exceptions() -> None:
    def boom(_x: int) -> int:
        raise RuntimeError("kaboom")

    with pytest.raises(CallableError, match="kaboom"):
        call_with_timeout(boom, 1, timeout_s=1.0, what="boom thing")


def test_call_with_timeout_none_waits_indefinitely_but_isolates_exceptions() -> None:
    with pytest.raises(CallableError):
        call_with_timeout(lambda: 1 / 0, timeout_s=None, what="t")


def test_safe_call_returns_result() -> None:
    assert safe_call(lambda x: x + 1, 1, what="t") == 2


def test_safe_call_wraps_exceptions() -> None:
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(CallableError, match="nope"):
        safe_call(boom, what="boom")


def test_config_error_is_a_value_error() -> None:
    # existing callers doing `except ValueError` (Section 9 reference logic,
    # older tests) must keep working after this hardening pass.
    assert issubclass(ConfigError, ValueError)


def test_shared_pool_runs_calls_concurrently_not_serially() -> None:
    """Two 150ms calls through ragwarden's own shared pool should overlap
    (~150ms total), not serialize to ~300ms — proves it isn't a size-1 pool."""
    from concurrent.futures import ThreadPoolExecutor

    def slow() -> float:
        t0 = time.perf_counter()
        time.sleep(0.15)
        return t0

    with ThreadPoolExecutor(max_workers=2) as driver:
        t0 = time.perf_counter()
        f1 = driver.submit(call_with_timeout, slow, timeout_s=2.0, what="a")
        f2 = driver.submit(call_with_timeout, slow, timeout_s=2.0, what="b")
        f1.result()
        f2.result()
    assert time.perf_counter() - t0 < 0.28
