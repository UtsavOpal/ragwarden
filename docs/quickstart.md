# Quickstart

Goal: from `pip install` to a working inline gate call in a few minutes.

## 1. Install

```bash
pip install 'ragwarden[nli]'   # core + the default DeBERTa-v3 NLI detector
```

For a first run with **no model download**, install just the core and use the stub detector shown at
the bottom of this page.

## 2. Build the two inputs

RagWarden consumes what your pipeline already has — a query with its retrieved evidence, and the
generated answer. Any object matching the [contracts](api.md#contracts) works; `ragwarden.models`
has ready-made ones:

```python
from ragwarden.models import Chunk, Context, Answer

context = Context(
    query="When was the Eiffel Tower completed?",
    chunks=[
        Chunk(
            text="The Eiffel Tower was completed in 1889 for the World's Fair.",
            score=0.92,
            source_id="doc-1",
            metadata={"bm25_score": 14.2, "knn_score": 0.88},
        ),
        Chunk(
            text="It stands 330 metres tall on the Champ de Mars in Paris.",
            score=0.71,
            source_id="doc-1",
        ),
    ],
    retrieval_method="hybrid",
)
answer = Answer(text="The Eiffel Tower was completed in 1889.")
```

## 3. Gate it

```python
from ragwarden import gate, Policy

result = gate(context, answer, policy=Policy.default())

print(result.action)  # GateAction.ALLOW
print(result.reliability_score)  # composite 0.0–1.0
print(result.output_text)  # original, redacted, or an abstention message
print(result.explanation)  # human-readable decision trail
print(result.action_payload)  # structured data for the chosen action
```

## 4. Wire in Tiers 2 and 3 (optional)

```python
result = gate(
    context,
    answer,
    policy=Policy.default(),
    generate_fn=my_llm,  # query -> answer_text, for Tier 2 consistency sampling
    judge_fn=my_llm_complete,  # prompt -> completion, for Tier 3 LLM-as-judge
)
```

Tier 2 is skipped without `generate_fn`; Tier 3 without `judge_fn`. Both are budget-capped
(see the [policy engine](concepts/policy-engine.md)).

## Config via YAML

```python
policy = Policy.from_yaml("policy.yaml")
```

```yaml
policy:
  claim_weighting: severity_weighted
  fail_mode: fail_closed
  thresholds: { allow: 0.90, redact_claims: 0.70, retry: 0.50 }
  max_retries: 1
  tier1: { detector: nli, confidence_threshold: 0.85 }
  tier2: { enabled: true, consistency_samples: 3 }
  tier3: { enabled: true, max_claims_per_request: 3, max_latency_ms: 800 }
```

## No-download smoke test

```python
from ragwarden import gate
from ragwarden.detectors.stub import KeywordStubDetector

result = gate(context, answer, detectors=[KeywordStubDetector()])
```

## Going to production

`gate()` never raises for a runtime failure — a broken detector, a hanging `generate_fn`/`judge_fn`,
anything unanticipated — it degrades to `ABSTAIN` instead. From an `async def` handler, use `agate()`
so a slow call doesn't block your event loop; run `ragwarden warmup` before taking traffic so a broken
model fails at startup, not on your first request. See
**[Running in production](production.md)** for the full picture (timeouts, cost caps, monitoring).

```python
from ragwarden import agate

result = await agate(context, answer, policy=policy, detectors=detectors)
```
