# Running in production

What RagWarden guarantees, what it doesn't, and how to configure it for real traffic.

## The core guarantee: `gate()` does not crash your request

Every external dependency `gate()` touches — a Tier-1 model, your `generate_fn`, your `judge_fn` — can
fail in the usual ways: it raises, it hangs, it returns garbage. None of that reaches your caller as
an unhandled exception.

* **Detectors** (Tier 1): a failing detector is caught per-call. `policy.fail_mode` decides the blast
  radius — `fail_closed` (default) treats every claim that detector would have scored as `UNRESOLVED`;
  `fail_open` drops the detector and ensembles over the survivors. If every detector fails, claims are
  always `UNRESOLVED` — RagWarden never fabricates a verdict from nothing.
* **`generate_fn`** (Tier 2): each consistency sample is isolated and individually time-budgeted
  (`policy.tier2.call_timeout_s`, default 10s). A failed sample is excluded from that claim's agreement
  count, not fatal to the request.
* **`judge_fn`** (Tier 3): each call is time-budgeted (`policy.tier3.call_timeout_s`, default 15s) *and*
  clamped to whatever remains of the tier's own latency cap — one slow call can't blow past
  `tier3.max_latency_ms`.
* **Everything else**: an exception anywhere in the cascade that isn't already one of the above (a bug,
  a malformed `RetrievalContext`, anything unanticipated) is caught by a top-level safety net. The
  failure is logged (`logging.getLogger("ragwarden.gate")`, full traceback) and `gate()` returns a
  normal `GateResult` with `action=ABSTAIN` instead of raising. Set `raise_on_error=True` to get the
  real exception back — do that in tests and local development, not in production.

A `Policy` that fails validation (`Policy.validate()` — bad thresholds, a negative retry count, ...) is
the one thing that still raises immediately. That's a deploy-time configuration bug, not a
request-time failure, and hiding it would be worse than crashing loudly at startup.

```python
result = gate(context, answer, policy=policy, detectors=[detector])
# result.action is always one of ALLOW/REDACT_CLAIMS/RETRY/ABSTAIN/ESCALATE.
# gate() never raises for a runtime failure.
```

## Timeouts — configure them for your actual latency

| Knob | Default | Bounds |
|---|---|---|
| `policy.tier1_call_timeout_s` | `None` (no cap) | One `detector.check_batch()` call |
| `policy.tier2.call_timeout_s` | `10.0` | One `generate_fn()` call |
| `policy.tier3.call_timeout_s` | `15.0` | One `judge_fn()` call (also clamped by `tier3.max_latency_ms`) |
| `policy.max_total_latency_ms` | `None` | The whole request, shared across Tier 3's budget |

**A timeout only bounds how long `gate()` waits — Python cannot forcibly kill a thread.** A call that
ignores its timeout keeps running in the background until it eventually returns. This is a real
limitation, not a bug: your `generate_fn`/`judge_fn` should still set its own client-level timeout
(`httpx.Client(timeout=...)`, your SDK's `timeout=` kwarg) so the underlying connection is actually
torn down. RagWarden's timeout is the safety net for when that isn't enough — it guarantees *your*
request latency is bounded even if the client's isn't.

If you're using a real Ollama/local model as `judge_fn` in a test, remember it can take several
seconds per call — the library default (`max_latency_ms=800`) is tuned for fast hosted judges;
widen it for slower judges (see `tests/integration/test_tier3_llm.py` for the pattern).

## Bounding cost from adversarial or pathological input

`policy.max_claims_per_request` (default **50**) caps how many claims a single answer can produce
*before* any tier runs. An answer that decomposes into thousands of sentences (a prompt-injection
attempt, a runaway generation loop) gets truncated, not processed in full — the cap is recorded in
`GateResult.explanation`, never silently applied.

## Warm up before taking traffic

Loading a Tier-1 model can fail for reasons that have nothing to do with any request — a cold Hugging
Face cache, a network blip, an OOM. Surface that at startup, not on your first real user:

```python
from ragwarden.warmup import warmup

report = warmup(policy)
if not report.ok:
    raise SystemExit(f"ragwarden warmup failed: {report.errors}")
```

Or as a Kubernetes readiness probe / CI smoke check:

```bash
ragwarden warmup --policy prod.yaml   # exit 0 = ready, exit 1 = not ready, one line per detector
```

## Async hosts (FastAPI, aiohttp, ...)

`gate()` is synchronous, CPU/IO-bound code (model inference, blocking calls inside your callables).
Calling it directly from an `async def` handler blocks the event loop for every other in-flight
request. Use `agate()` instead — it runs `gate()` in a worker thread via `asyncio.to_thread`:

```python
from ragwarden import agate


@app.post("/answer")
async def answer(req: Request) -> Response:
    result = await agate(context, generation, policy=policy, detectors=detectors)
    ...
```

## Concurrency and thread-safety

* `gate()`/`agate()` hold no shared mutable state between calls — safe to call concurrently from
  multiple threads or coroutines with the same `Policy`/detector instances.
* **Share detector instances across requests**; do not construct a fresh `NLIDetector()` (or any
  model-backed detector) per call — model loading is the expensive part. PyTorch models in `eval()`
  mode under `torch.no_grad()` (as RagWarden's detectors run them) are safe to call concurrently from
  multiple threads; throughput under heavy concurrent load is bounded by your CPU/GPU, not by a lock
  in RagWarden.
* Host callables (`generate_fn`, `judge_fn`) run in a shared, bounded worker pool
  (`ragwarden.resilience`, 64 threads) used for timeout isolation. If you drive very high concurrency,
  that pool — not RagWarden's own logic — is the thing to size or replace; the pool is process-wide and
  lazily created.
* For real throughput at scale, put Tier-1 inference behind a dedicated model server (Triton,
  TorchServe, or a simple batching queue) rather than loading the model in every worker process — see
  [benchmarks.md](benchmarks.md) for why per-call latency, not corpus size, is what you're scaling.

## What to monitor

Every `gate()` call emits (via [observability](observability.md)):

* `ragwarden.action`, `ragwarden.reliability_score` — decision distribution over time; a sudden shift
  toward `ABSTAIN` is your canary.
* `ragwarden.claims_resolved_tier{0,1,2,3}` — **Tier 3 escalation rate** is the headline health metric
  (Spec Section 7.5): above `tier3.escalation_rate_warn_threshold` (default 20%) means Tier 1/2
  thresholds need retuning against your real traffic, not that Tier 3 needs to get faster.
* `action_payload.reason == "internal_error"` — the safety net fired. This should be rare; alert on it
  and check the logs for the traceback (`ragwarden.gate` logger).
* Detector/sample failure counts surface in `GateResult.explanation` (e.g. "Tier 1 detector failures:
  ...", "Tier 2: N/M generate_fn() calls failed") — grep your structured logs for these to catch a
  degrading upstream dependency before it trips `fail_closed` on real traffic.

## Checklist before serving real traffic

1. `ragwarden warmup --policy prod.yaml` passes in CI and at deploy time.
2. Timeouts (`tier1_call_timeout_s`, `tier2.call_timeout_s`, `tier3.call_timeout_s`,
   `max_total_latency_ms`) are set to your actual provider latency, not left at the library defaults
   if your judge/generator is slower than a typical hosted API.
3. `max_claims_per_request` is sized for your longest legitimate answer, not just the default.
4. `fail_mode` is a conscious choice (`fail_closed` unless you have a specific availability reason not
   to — see [the policy engine](concepts/policy-engine.md)).
5. `ragwarden calibrate` has been run against your own labeled traffic, not just RAGTruth.
6. OTel export and structured logging are wired to something you actually watch.
