# SPDX-License-Identifier: Apache-2.0
"""Phase 1 acceptance: policy engine decision logic + config loading."""

from __future__ import annotations

import textwrap

import pytest

from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.policy import Policy, Thresholds, decide, register_trigger


def _v(status: ClaimStatus, conf: float = 0.9, explanation: str = "") -> ClaimVerdict:
    return ClaimVerdict("c", status, conf, resolved_at_tier=1, explanation=explanation)


def test_default_policy_is_fail_closed() -> None:
    assert Policy.default().fail_mode == "fail_closed"


def test_threshold_ladder() -> None:
    pol = Policy(thresholds=Thresholds(allow=0.9, redact_claims=0.7, retry=0.5), max_retries=1)
    assert decide(0.95, [], pol).action is GateAction.ALLOW
    assert decide(0.8, [], pol).action is GateAction.REDACT_CLAIMS
    assert decide(0.6, [], pol).action is GateAction.RETRY
    assert decide(0.3, [], pol).action is GateAction.ABSTAIN


def test_retry_budget_exhaustion_forces_abstain() -> None:
    pol = Policy(max_retries=1)
    assert decide(0.6, [], pol, retries_used=0).action is GateAction.RETRY
    assert decide(0.6, [], pol, retries_used=1).action is GateAction.ABSTAIN


def test_escalation_trigger_wins_over_threshold() -> None:
    pol = Policy()
    verdicts = [_v(ClaimStatus.CONTRADICTED, explanation="conflict [authority=high]")]
    assert decide(0.99, verdicts, pol).action is GateAction.ESCALATE


def test_custom_trigger_registration() -> None:
    register_trigger("always", lambda verdicts, policy: True)
    pol = Policy()
    pol.escalate_to_human.triggers = ["always"]
    assert decide(1.0, [], pol).action is GateAction.ESCALATE


def test_invalid_thresholds_rejected() -> None:
    pol = Policy(thresholds=Thresholds(allow=0.5, redact_claims=0.7, retry=0.6))
    with pytest.raises(ValueError):
        pol.validate()


def test_from_yaml(tmp_path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        textwrap.dedent(
            """
            policy:
              fail_mode: fail_open
              thresholds:
                allow: 0.95
                redact_claims: 0.8
                retry: 0.6
              tier1:
                detector: hhem
            """
        )
    )
    pol = Policy.from_yaml(p)
    pol.validate()
    assert pol.fail_mode == "fail_open"
    assert pol.thresholds.allow == 0.95
    assert pol.tier1.detector == "hhem"
