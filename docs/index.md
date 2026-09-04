# RagWarden

> An inline hallucination gate for production RAG pipelines.

**RagWarden doesn't compete with RAGAS or UQLM** — it's the runtime enforcement layer that uses
signals like theirs (and its own fast detectors) to make a bounded-latency, explainable decision
inline in production, with a policy engine enterprises can tune to their own risk tolerance.

## What it does

An answer from *any* RAG pipeline is broken into atomic, independently-checkable claims. Each claim
runs a cost-tiered cascade, cheapest-first:

| Tier | Check | Cost |
|---|---|---|
| **0** | Retrieval heuristics (empty results, score floor, score gap, query–context overlap) | ~0, always on |
| **1** | Claim-vs-evidence entailment (NLI / HHEM / LettuceDetect / MiniCheck) | ms, encoder models |
| **2** | Uncertainty quantification (consistency sampling; token log-probs if available) | N extra LLM calls |
| **3** | LLM-as-judge (structured chain-of-thought, budget-capped) | expensive, ambiguous remainder only |

Per-claim verdicts combine into a composite reliability score. A configurable **policy engine** turns
that score into an action — `ALLOW`, `REDACT_CLAIMS`, `RETRY`, `ABSTAIN`, or `ESCALATE` — and every
decision carries a full explanation trail plus a structured `action_payload`.

## Install

```bash
pip install ragwarden                 # core: light, no ML dependencies
pip install 'ragwarden[nli]'          # + DeBERTa-v3 NLI Tier-1 detector
pip install 'ragwarden[opensearch]'   # + OpenSearch adapter
pip install 'ragwarden[all]'          # everything (OSI-licensed extras)
```

The core install pulls only a YAML parser and a pure-Python sentence splitter.

## Next

- **[Quickstart](quickstart.md)** — from `pip install` to a working gate call.
- **[The detection cascade](concepts/cascade.md)** — how the tiers work and escalate.
- **[The policy engine](concepts/policy-engine.md)** — turning a score into a decision.
- **[Honest limitations](limitations.md)** — what a gate can and cannot do.
