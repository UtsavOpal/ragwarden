# SPDX-License-Identifier: Apache-2.0
"""Policy engine (Build Spec Section 9) — RagWarden's core differentiator.

Maps ``(reliability_score, per-claim verdicts, config)`` to exactly one
:class:`~ragwarden.contracts.GateAction`, with a human-readable reason. The
engine is an ordered list of rules: the first rule that fires wins. Host
applications register custom trigger rules without forking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragwarden.cascade.tier0_heuristics import Tier0Config
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction

__all__ = [
    "DecompositionConfig",
    "EscalateConfig",
    "Policy",
    "PolicyDecision",
    "Thresholds",
    "Tier0Config",
    "Tier1Config",
    "Tier2Config",
    "Tier3Config",
    "TriggerRule",
    "decide",
    "register_trigger",
]

FailMode = str  # "fail_closed" | "fail_open"


@dataclass
class Thresholds:
    allow: float = 0.90
    redact_claims: float = 0.70
    retry: float = 0.50
    # anything below `retry` falls through to abstain


@dataclass
class Tier1Config:
    enabled: bool = True
    confidence_threshold: float = 0.85
    detector: str = "nli"  # which registered Tier-1 detector to use
    ensemble: list[str] = field(default_factory=list)
    ensemble_strategy: str = "max_confidence"  # "majority" | "max_confidence"


@dataclass
class Tier2Config:
    enabled: bool = True
    consistency_samples: int = 3  # 0 disables Tier 2 entirely
    agreement_resolve_threshold: float = 0.7
    logprob_low_threshold: float = 0.35
    cost_per_sample_usd: float | None = None
    # Per-call wall-clock budget for each generate_fn() invocation. A call that
    # exceeds this is treated as a failure for that sample (fail_mode applies),
    # not a hang that blocks the request. None = no per-call timeout.
    call_timeout_s: float | None = 10.0


@dataclass
class Tier3Config:
    enabled: bool = True
    max_claims_per_request: int = 3
    max_latency_ms: int = 800
    judge_model: str | None = None  # host app supplies a callable at call time
    cost_per_call_usd: float | None = None
    max_evidence_chars: int = 4000
    # If more than this fraction of claims reach Tier 3, Tier 1/2 thresholds
    # likely need retuning (Spec Section 7.5). Surfaced, not enforced.
    escalation_rate_warn_threshold: float = 0.2
    # Per-call wall-clock budget for each judge_fn() invocation, independent of
    # (and typically tighter than) max_latency_ms. None = no per-call timeout.
    call_timeout_s: float | None = 15.0


@dataclass
class EscalateConfig:
    enabled: bool = True
    triggers: list[str] = field(default_factory=lambda: ["contradicted_high_authority_source"])


@dataclass
class DecompositionConfig:
    strategy: str = "sentence_split"  # OPEN DECISION: "sentence_split" | "llm"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Policy:
    claim_weighting: str = "severity_weighted"  # registered ScoringStrategy name
    thresholds: Thresholds = field(default_factory=Thresholds)
    max_retries: int = 1  # bounded — never loop indefinitely
    fail_mode: FailMode = "fail_closed"  # fail_closed | fail_open
    max_total_latency_ms: int | None = None  # request-wide budget; None = tier caps only
    # Hard cap on claims processed per request, applied right after decomposition —
    # bounds worst-case cost/latency from a pathologically long or adversarial
    # answer. Excess claims are recorded (not silently dropped) and treated as
    # UNRESOLVED for scoring. None = uncapped (fine for trusted, bounded inputs).
    max_claims_per_request: int | None = 50
    # Per-call wall-clock budget for a single Tier-1 detector.check_batch() call.
    # None = no timeout (detector inference is normally local and fast; set this
    # if a detector talks to a remote model-serving endpoint).
    tier1_call_timeout_s: float | None = None
    abstain_message: str = "I don't have enough reliable information to answer this confidently."
    high_authority_metadata_key: str = "authority"
    high_authority_values: list[str] = field(default_factory=lambda: ["high", "primary"])
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    tier0: Tier0Config = field(default_factory=Tier0Config)
    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)
    tier3: Tier3Config = field(default_factory=Tier3Config)
    escalate_to_human: EscalateConfig = field(default_factory=EscalateConfig)

    # ----- construction -------------------------------------------------------
    @classmethod
    def default(cls) -> Policy:
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        root = data.get("policy", data) or {}
        return cls(
            claim_weighting=root.get("claim_weighting", cls.claim_weighting),
            thresholds=Thresholds(**(root.get("thresholds") or {})),
            max_retries=root.get("max_retries", 1),
            fail_mode=root.get("fail_mode", "fail_closed"),
            max_total_latency_ms=root.get("max_total_latency_ms"),
            max_claims_per_request=root.get("max_claims_per_request", 50),
            tier1_call_timeout_s=root.get("tier1_call_timeout_s"),
            abstain_message=root.get("abstain_message", cls.abstain_message),
            high_authority_metadata_key=root.get("high_authority_metadata_key", "authority"),
            high_authority_values=list(root.get("high_authority_values", ["high", "primary"])),
            decomposition=DecompositionConfig(**(root.get("decomposition") or {})),
            tier0=Tier0Config(**(root.get("tier0") or {})),
            tier1=Tier1Config(**(root.get("tier1") or {})),
            tier2=Tier2Config(**(root.get("tier2") or {})),
            tier3=Tier3Config(**(root.get("tier3") or {})),
            escalate_to_human=EscalateConfig(**(root.get("escalate_to_human") or {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Policy:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def validate(self) -> None:
        from ragwarden.errors import ConfigError

        t = self.thresholds
        if not (0.0 <= t.retry <= t.redact_claims <= t.allow <= 1.0):
            raise ConfigError(
                "policy.thresholds must satisfy 0 <= retry <= redact_claims <= allow <= 1"
            )
        if self.fail_mode not in ("fail_closed", "fail_open"):
            raise ConfigError("policy.fail_mode must be 'fail_closed' or 'fail_open'")
        if self.max_retries < 0:
            raise ConfigError("policy.max_retries must be >= 0")
        if self.max_claims_per_request is not None and self.max_claims_per_request < 1:
            raise ConfigError("policy.max_claims_per_request must be >= 1 or None")
        for name, val in (
            ("tier1_call_timeout_s", self.tier1_call_timeout_s),
            ("tier2.call_timeout_s", self.tier2.call_timeout_s),
            ("tier3.call_timeout_s", self.tier3.call_timeout_s),
            ("max_total_latency_ms", self.max_total_latency_ms),
        ):
            if val is not None and val <= 0:
                raise ConfigError(f"policy.{name} must be > 0 or None")


@dataclass
class PolicyDecision:
    action: GateAction
    reason: str
    triggered_rule: str


# --------------------------------------------------------------------------- #
# Pluggable trigger rules
# --------------------------------------------------------------------------- #
TriggerRule = Callable[[list[ClaimVerdict], "Policy"], bool]
_TRIGGERS: dict[str, TriggerRule] = {}


def register_trigger(name: str, fn: TriggerRule) -> None:
    """Register a custom escalation trigger callable ``fn(verdicts, policy) -> bool``."""
    _TRIGGERS[name] = fn


def _is_high_authority(verdict: ClaimVerdict, policy: Policy) -> bool:
    # supporting_evidence holds source_ids; without the chunk objects here we
    # rely on the verdict explanation / evidence ids being annotated upstream.
    marker = f"[{policy.high_authority_metadata_key}="
    return any(f"{marker}{val}]" in verdict.explanation for val in policy.high_authority_values)


def _trigger_contradicted_high_authority(verdicts: list[ClaimVerdict], policy: Policy) -> bool:
    return any(
        v.status is ClaimStatus.CONTRADICTED and _is_high_authority(v, policy) for v in verdicts
    )


register_trigger("contradicted_high_authority_source", _trigger_contradicted_high_authority)


# --------------------------------------------------------------------------- #
# Decision logic (Spec Section 9.2) — ordered rules, first match wins
# --------------------------------------------------------------------------- #
def decide(
    score: float,
    verdicts: list[ClaimVerdict],
    policy: Policy,
    *,
    retries_used: int = 0,
) -> PolicyDecision:
    if policy.escalate_to_human.enabled:
        for name in policy.escalate_to_human.triggers:
            rule = _TRIGGERS.get(name)
            if rule is not None and rule(verdicts, policy):
                return PolicyDecision(
                    GateAction.ESCALATE,
                    f"escalation trigger '{name}' fired",
                    name,
                )

    t = policy.thresholds
    if score >= t.allow:
        return PolicyDecision(
            GateAction.ALLOW, f"reliability {score:.3f} >= allow {t.allow}", "threshold.allow"
        )
    if score >= t.redact_claims:
        return PolicyDecision(
            GateAction.REDACT_CLAIMS,
            f"reliability {score:.3f} in [redact, allow)",
            "threshold.redact_claims",
        )
    remaining = policy.max_retries - retries_used
    if score >= t.retry and remaining > 0:
        return PolicyDecision(
            GateAction.RETRY,
            f"reliability {score:.3f} in [retry, redact); {remaining} retr"
            f"{'y' if remaining == 1 else 'ies'} left",
            "threshold.retry",
        )
    return PolicyDecision(
        GateAction.ABSTAIN,
        f"reliability {score:.3f} below actionable thresholds"
        + ("" if retries_used < policy.max_retries else " (retry budget exhausted)"),
        "fallthrough.abstain",
    )
