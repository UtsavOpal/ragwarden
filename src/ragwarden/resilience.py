# SPDX-License-Identifier: Apache-2.0
"""Fault isolation for host-supplied callables and detector calls.

Production RAG pipelines call out to things that fail in the usual ways —
network timeouts, model OOMs, a flaky provider. RagWarden's job is to gate an
answer, not to become a second point of failure: nothing in this module lets an
exception from a *host-supplied* callable (``generate_fn``, ``judge_fn``) or a
*detector* propagate uncontrolled into the cascade. Every call site converts a
failure into an explicit, budgeted, fail-mode-aware outcome instead.

Two isolation primitives:

* :func:`call_with_timeout` — runs a synchronous callable in a worker thread and
  bounds how long the caller waits. Python cannot forcibly kill a thread, so a
  callable that ignores the timeout keeps running in the background — this
  bounds *your* request latency, it is not a guarantee the underlying call
  stops. Well-behaved ``generate_fn``/``judge_fn`` implementations should still
  set their own client-level timeout (``httpx.Client(timeout=...)`` etc.); this
  is the safety net for when they don't.
* :func:`safe_call` — runs a synchronous callable and converts any exception
  into a :class:`~ragwarden.errors.CallableError` rather than propagating it.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout
from typing import TypeVar

from ragwarden.errors import CallableError, CallableTimeoutError

__all__ = ["call_with_timeout", "safe_call", "shutdown_executor"]

T = TypeVar("T")

# A shared, lazily-created pool. Bounded so a burst of slow/hung calls can't
# spawn unbounded threads; new work queues rather than spawning more workers.
_MAX_WORKERS = 64
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="ragwarden-io")
    return _executor


def shutdown_executor(wait: bool = False) -> None:
    """Release the shared worker pool. Mainly for tests and clean process exit;
    RagWarden re-creates the pool lazily on the next call."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait, cancel_futures=True)
        _executor = None


def call_with_timeout(fn: Callable[..., T], *args: object, timeout_s: float | None, what: str) -> T:
    """Call ``fn(*args)`` with a wall-clock budget.

    Raises :class:`~ragwarden.errors.CallableTimeoutError` if it does not return
    in time, or :class:`~ragwarden.errors.CallableError` if it raises. With
    ``timeout_s is None`` the call still runs isolated in a worker thread (so a
    crash doesn't take down the caller's stack) but waits indefinitely.
    """
    future = _get_executor().submit(fn, *args)
    try:
        return future.result(timeout=timeout_s)
    except _FutTimeout as exc:
        raise CallableTimeoutError(what, timeout_s or float("inf")) from exc
    except CallableTimeoutError:
        raise
    except Exception as exc:
        raise CallableError(what, exc) from exc


def safe_call(fn: Callable[..., T], *args: object, what: str) -> T:
    """Call ``fn(*args)`` in-process; convert any exception to CallableError."""
    try:
        return fn(*args)
    except Exception as exc:
        raise CallableError(what, exc) from exc
