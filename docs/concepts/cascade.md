# The detection cascade

Checks run cheapest-first. Each tier only sees the claims the previous tier could not resolve
confidently, so the expensive tiers run on a shrinking remainder.

## Claim decomposition

Runs before the cascade. v0.1 splits the answer into sentences (`pysbd`, zero heavy deps) and treats
each as an atomic claim — a known simplification vs. true semantic claim extraction, good enough to
validate the design. A model-based decomposer is exposed behind `decomposition.strategy` (an open
decision; not selected silently).

## Tier 0 — retrieval heuristics

Deterministic, no models, once per request on the `RetrievalContext` alone:

- **Empty retrieval** → short-circuit straight to `ABSTAIN`.
- **Score floor** (`tier0.min_top_score`) — top chunk below the floor flags low-confidence retrieval.
- **Score gap** (`tier0.score_gap_threshold`) — a suspicious drop-off, or everything clustered low.
- **Query–context overlap** (`tier0.min_overlap`) — cheap lexical check that retrieval is on-topic.

Because hybrid pipelines produce both BM25 and kNN scores, Tier 0 reads
`chunk.metadata["bm25_score"]` / `["knn_score"]` as well as the fused `chunk.score`.

## Tier 1 — claim-vs-evidence entailment

Each claim is checked against the top evidence chunks with a fast encoder model. Ships as optional
extras:

| Detector | Extra | Notes |
|---|---|---|
| `NLIDetector` | `[nli]` | DeBERTa-v3 NLI (default). entailment→SUPPORTED, contradiction→CONTRADICTED, neutral→UNSUPPORTED. Batched inference. |
| `HHEMDetector` | `[hhem]` | Vectara HHEM-2.1-Open consistency probability. |
| `LettuceDetectAdapter` | `[lettucedetect]` | ModernBERT span-level detector. |
| `MiniCheckDetector` | `[minicheck]` | Install the fact-checker from git; the PyPI `minicheck` is an unrelated project. |

Install more than one and set `tier1.ensemble` + `tier1.ensemble_strategy`
(`max_confidence` / `majority` / `most_severe`).

Verdicts below `tier1.confidence_threshold` (default 0.85) are **escalated** rather than force-resolved.

If no detector extra is installed, Tier 1 is skipped with a loud warning and the decision is
Tier-0-only.

## Tier 2 — uncertainty quantification

For the Tier-1 remainder. The host's `generate_fn(query)` is called `N` more times
(`tier2.consistency_samples`, default 3, `0` disables) and a reused Tier-1 detector measures how
consistently the claim is supported across those independent samples. An optional white-box signal
(mean token probability over the claim span) is combined when `Generation.token_logprobs` is present.
The estimated dollar cost lands in `StageTrace.cost_estimate_usd`.

## Tier 3 — LLM-as-judge

For the small remainder still unresolved. RagWarden owns a G-Eval-style structured chain-of-thought
prompt; the host passes `judge_fn(prompt) -> completion`. A hard budget —
`tier3.max_claims_per_request` **and** `tier3.max_latency_ms`, the latter clamped to a request-wide
deadline (`policy.max_total_latency_ms`) — bounds worst-case cost. Claims not reached stay
`UNRESOLVED`; the policy engine decides what to do with them (scored as `UNSUPPORTED` by default).

**Watch the escalation rate.** If more than ~15–20% of claims reach Tier 3 in real traffic, retune
Tier 1/2 thresholds — RagWarden emits this ratio on every call and warns past
`tier3.escalation_rate_warn_threshold`.
