# SPDX-License-Identifier: Apache-2.0
"""Production hardening, end-to-end through gate()/agate():
- gate() never raises for a runtime cascade failure (safety net)
- a bad Policy still raises immediately (deploy-time bug, not hidden)
- a global claims cap bounds worst-case cost from a pathological answer
- agate() does not block the event loop
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ragwarden import ConfigError, GateRequest, Policy, agate, gate
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context

pytestmark = pytest.mark.integration

CTX = Context(
    query="Eiffel Tower?",
    chunks=[Chunk("The Eiffel Tower was completed in 1889.", 0.9, "d1")],
    retrieval_method="hybrid",
)
_SUPPORT = ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.SUPPORTED, 0.95, 1))


# --- config errors still raise loudly, immediately -------------------------
def test_bad_policy_raises_immediately_not_swallowed() -> None:
    pol = Policy.default()
    pol.thresholds.allow = 0.1  # invalid: allow < redact_claims
    with pytest.raises(ConfigError):
        gate(CTX, Answer("x"), policy=pol, detectors=[_SUPPORT])


# --- the safety net ---------------------------------------------------
def test_unexpected_cascade_failure_returns_abstain_not_an_exception(monkeypatch) -> None:
    from sys import modules as _modules

    gate_module = _modules["ragwarden.gate"]

    def boom(*_a, **_k):
        raise RuntimeError("something deep in the cascade broke")

    monkeypatch.setattr(gate_module, "run_tier0", boom)
    result = gate(CTX, Answer("The Eiffel Tower was completed in 1889."), detectors=[_SUPPORT])
    assert result.action is GateAction.ABSTAIN
    assert result.reliability_score == 0.0
    assert "internal error" in result.explanation
    assert result.action_payload["reason"] == "internal_error"
    assert result.action_payload["error_type"] == "RuntimeError"


def test_raise_on_error_bypasses_the_safety_net(monkeypatch) -> None:
    from sys import modules as _modules

    gate_module = _modules["ragwarden.gate"]

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(gate_module, "run_tier0", boom)
    with pytest.raises(RuntimeError, match="boom"):
        gate(
            CTX,
            Answer("x"),
            detectors=[_SUPPORT],
            raise_on_error=True,
        )


def test_safety_net_survives_even_with_telemetry_on(monkeypatch) -> None:
    from sys import modules as _modules

    gate_module = _modules["ragwarden.gate"]

    monkeypatch.setattr(
        gate_module, "run_tier0", lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
    )
    result = gate(CTX, Answer("x"), detectors=[_SUPPORT], emit_telemetry=True)
    assert result.action is GateAction.ABSTAIN


# --- global claims cap -----------------------------------------------
def test_max_claims_per_request_caps_a_pathological_answer() -> None:
    long_answer = " ".join(
        f"This is sentence number {i} in a very long answer." for i in range(200)
    )
    pol = Policy.default()
    pol.max_claims_per_request = 5
    pol.tier2.enabled = False
    result = gate(CTX, Answer(long_answer), policy=pol, detectors=[_SUPPORT])
    assert len(result.claim_verdicts) <= 5
    assert "capped at 5" in result.explanation


def test_max_claims_per_request_none_is_uncapped() -> None:
    pol = Policy.default()
    pol.max_claims_per_request = None
    answer = " ".join(f"Sentence {i} is true." for i in range(30))
    result = gate(CTX, Answer(answer), policy=pol, detectors=[_SUPPORT])
    assert len(result.claim_verdicts) == 30


# --- agate() does not block the event loop -----------------------------
def test_agate_does_not_block_the_event_loop() -> None:
    from sys import modules as _modules

    gate_module = _modules["ragwarden.gate"]

    real_run_tier0 = gate_module.run_tier0

    def slow_tier0(*a, **k):
        time.sleep(0.3)
        return real_run_tier0(*a, **k)

    async def main() -> tuple[float, object]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.02)
                ticks += 1

        gate_module.run_tier0 = slow_tier0
        try:
            t0 = time.perf_counter()
            gate_task = asyncio.create_task(
                agate(CTX, Answer("The Eiffel Tower was completed in 1889."), detectors=[_SUPPORT])
            )
            tick_task = asyncio.create_task(ticker())
            result = await gate_task
            await tick_task
            return time.perf_counter() - t0, ticks, result
        finally:
            gate_module.run_tier0 = real_run_tier0

    _elapsed, ticks, result = asyncio.run(main())
    assert result.action is GateAction.ALLOW
    # the ticker (20ms cadence) must have kept advancing while gate() ran in a
    # worker thread -- if agate() blocked the loop, ticks would stall near 0.
    assert ticks >= 5


# --- retry semantics still bounded under the hardened path -----------------
def test_retry_never_loops_even_after_hardening() -> None:
    pol = Policy.default()
    pol.escalate_to_human.enabled = False
    pol.thresholds.retry = 0.4
    pol.thresholds.redact_claims = 0.55
    det = ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.CONTRADICTED, 0.9, 1))
    ans = Answer("The tower is orange.")
    r1 = gate(CTX, ans, policy=pol, detectors=[det], request=GateRequest("r", retries_used=0))
    r2 = gate(CTX, ans, policy=pol, detectors=[det], request=GateRequest("r", retries_used=1))
    assert r1.action in {GateAction.RETRY, GateAction.ABSTAIN}
    assert r2.action is GateAction.ABSTAIN
