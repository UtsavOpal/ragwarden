# The policy engine

The policy engine is RagWarden's core differentiator: it maps
`(reliability_score, per-claim verdicts, config)` to exactly one action, with a reason. It is an
**ordered list of rules — the first that fires wins.**

## Decision logic

```python
def decide(score, verdicts, policy):
    for trigger in policy.escalate_to_human.triggers:  # 1. custom escalation rules
        if trigger(verdicts, policy):
            return ESCALATE
    if score >= policy.thresholds.allow:  # 2. threshold ladder
        return ALLOW
    if score >= policy.thresholds.redact_claims:
        return REDACT_CLAIMS
    if score >= policy.thresholds.retry and retries_remaining:
        return RETRY
    return ABSTAIN  # 3. fall-through
```

## Composite scoring

`SeverityWeightedScoring` (the default) is a confidence-weighted mean of per-claim reliability credit:
`CONTRADICTED` (0.0) penalizes more than `UNSUPPORTED` (0.25), which penalizes more than
`PARTIALLY_SUPPORTED` (0.6). `UNRESOLVED` is scored as `UNSUPPORTED` unless remapped. Weights are
constructor parameters — the same class serves a lenient internal-tools deployment and a strict
regulated one.

## Pluggable escalation triggers

```python
from ragwarden.policy import register_trigger

register_trigger(
    "contradicted_medical_claim",
    lambda verdicts, policy: any(
        v.status.value == "contradicted" and "drug" in v.claim_text.lower() for v in verdicts
    ),
)
# then in policy.yaml:  escalate_to_human: { triggers: [contradicted_medical_claim] }
```

Host applications register triggers without forking the package.

## `fail_mode` — a real production safety decision

| Mode | Behavior on a detector error / timeout / load failure | Trade-off |
|---|---|---|
| `fail_closed` **(default)** | Affected claims → `UNRESOLVED` → drags the score down → more likely to `ABSTAIN`. | Safer; infra hiccups reduce answer availability. |
| `fail_open` | Skip the failed detector, score on what succeeded. | Higher availability; a silent detector outage degrades protection unless monitored. |

Ship `fail_closed`. Make the choice consciously.

## Configuration schema

```yaml
policy:
  claim_weighting: severity_weighted
  thresholds: { allow: 0.90, redact_claims: 0.70, retry: 0.50 }
  max_retries: 1
  fail_mode: fail_closed
  max_total_latency_ms: null          # request-wide budget; null = per-tier caps only
  decomposition: { strategy: sentence_split }
  tier0: { min_top_score: null, min_overlap: 0.0 }
  tier1: { enabled: true, detector: nli, confidence_threshold: 0.85, ensemble: [] }
  tier2: { enabled: true, consistency_samples: 3, cost_per_sample_usd: null }
  tier3: { enabled: true, max_claims_per_request: 3, max_latency_ms: 800 }
  escalate_to_human: { enabled: true, triggers: [contradicted_high_authority_source] }
```
