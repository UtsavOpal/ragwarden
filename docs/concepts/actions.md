# Actions

The gate returns exactly one action. RagWarden never performs retrieval or generation itself —
`RETRY` and `ESCALATE` hand back structured signals for the host to act on. Everything an action
produces beyond the text is in `GateResult.action_payload`.

| Action | `output_text` | `action_payload` |
|---|---|---|
| `ALLOW` | the original answer, unmodified | `{"citations": {claim: [source_id, ...]}}` — the trail is never silently dropped |
| `REDACT_CLAIMS` | the answer with unsupported/contradicted sentences removed (order + connective text preserved) | `{"removed_claims": [...]}` |
| `RETRY` | the original answer | `{"retry": True, "retries_remaining": int, "reason": str, "weak_claims": [...]}` |
| `ABSTAIN` | a configurable "I don't have enough reliable information…" message | `{"reason": str}` |
| `ESCALATE` | the original answer, flagged as pending review | `{"status": "pending_human_review", "reliability_score": float, "review_item": {...}}` |

## Edge cases handled explicitly

- **Redaction leaves nothing coherent** (every claim redactable, or < 4 words left) → falls back to
  `ABSTAIN` with `reason = "redaction left no coherent text"`, never ships a mangled answer.
- **Retry budget exhausted** — the host passes `GateRequest(request_id=..., retries_used=N)` back on
  each attempt; once `retries_used >= policy.max_retries`, `RETRY` is forced to `ABSTAIN`. The gate
  never loops.

## Consuming RETRY / ESCALATE

```python
result = gate(context, answer, policy=pol, request=GateRequest("req-1", retries_used=n))

if result.action is GateAction.RETRY:
    fresh_context, fresh_answer = my_pipeline.regenerate(query)
    result = gate(
        fresh_context, fresh_answer, policy=pol, request=GateRequest("req-1", retries_used=n + 1)
    )

elif result.action is GateAction.ESCALATE:
    review_queue.put(result.action_payload["review_item"])
```
