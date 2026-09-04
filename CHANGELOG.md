# Changelog

All notable changes to RagWarden are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0 onward.

## [0.3.0] — 2026-09-04

**Project renamed: HalluGate → RagWarden.** Package, import path, PyPI project name, CLI entry point,
docs, and every reference across the repo. `pip install hallugate` never shipped — this rename has no
migration burden for anyone; there is no `hallugate` compatibility shim.

```diff
- pip install hallugate
+ pip install ragwarden

- from hallugate import gate, Policy
+ from ragwarden import gate, Policy

- hallugate benchmark --dataset ragtruth
+ ragwarden benchmark --dataset ragtruth
```

No functional changes. `src/hallugate/` → `src/ragwarden/` via `git mv` (history preserved).
140 tests pass; ruff + mypy --strict clean; `mkdocs build --strict` clean; `python -m build` +
`twine check` produce `ragwarden-0.3.0`.

## [0.2.0] — 2026-09-04

**Production hardening.** `gate()` now never raises for a runtime cascade failure — every external
dependency (a Tier-1 detector, `generate_fn`, `judge_fn`) is fault-isolated and time-boxed.

### Added
- `ragwarden.errors`: `RagWardenError` hierarchy (`ConfigError`, `DetectorError`, `CallableError`,
  `CallableTimeoutError`) — exported from `ragwarden`.
- `ragwarden.resilience`: `call_with_timeout()` / `safe_call()` run a callable in a shared, bounded
  worker pool and convert any exception or timeout into a typed error instead of propagating.
- **Tier 1**: a detector that raises, times out, or returns a malformed batch is isolated per
  `policy.fail_mode` (`fail_closed`: poisons that claim's ensemble result → `UNRESOLVED`;
  `fail_open`: drops it, ensembles over survivors). New `policy.tier1_call_timeout_s`.
- **Tier 2**: each `generate_fn()` sample and each reused-detector check is isolated individually; a
  claim resolves on whatever samples succeeded (`Tier2Outcome.samples_failed`) rather than failing the
  request. New `policy.tier2.call_timeout_s` (default 10s), `min_effective_samples`.
- **Tier 3**: each `judge_fn()` call is now time-boxed (`policy.tier3.call_timeout_s`, default 15s),
  clamped to the tier's remaining latency budget — previously only the *aggregate* deadline was
  enforced, so a single slow call could exceed `max_latency_ms` unbounded.
- **`gate()` safety net**: any exception not already handled by a tier (a bug, a malformed input) is
  caught, logged with a full traceback, and returned as a normal `ABSTAIN` `GateResult` — never an
  unhandled exception. `raise_on_error=True` opts back into raising, for tests/local dev.
  `Policy.validate()` errors still raise immediately (a deploy-time bug, not a request-time failure).
- **`policy.max_claims_per_request`** (default 50): caps claims processed per request right after
  decomposition, bounding worst-case cost from a pathological or adversarial answer; excess is
  recorded in `GateResult.explanation`, never silently dropped.
- **`agate()`**: async wrapper around `gate()` via `asyncio.to_thread`, for use from `async def`
  handlers (FastAPI, aiohttp, ...) without blocking the event loop.
- **`ragwarden.warmup`** / `ragwarden warmup` CLI: loads and probes configured Tier-1 detector(s) once,
  synchronously — a readiness-probe / startup check so a broken model fails before serving traffic,
  not on the first real request.
- `docs/production.md`: the production runbook (timeouts, cost caps, warmup, async, concurrency /
  thread-safety, what to monitor, a pre-launch checklist).

### Changed
- `GateResult.action_payload` and log/telemetry emission unaffected; existing `Detector`,
  `generate_fn`, `judge_fn` signatures are unchanged — this release is additive (new optional
  parameters, new fields with defaults), no breaking changes to `ragwarden.contracts`.

33 new tests (failure-injection: raising/hanging/malformed detectors and callables, the safety net,
the claims cap, `agate()` non-blocking behavior) — 140 total offline, all passing; 85% coverage;
ruff + mypy --strict clean.

## [0.1.0] — 2026-09-04

First tagged release: the full Build Spec cascade end-to-end (Tiers 0–3), policy engine, five actions,
five retrieval adapters, benchmark + calibration harness, OpenTelemetry + structured logging, docs
site, and a Trusted-Publishing release pipeline. Benchmark numbers are an honest baseline, not yet
competitive (see the README).

### Added
- **Phase 12 — v1.0 hardening.** `ClaimVerdict`, `StageTrace`, `GateResult` are now `frozen=True` —
  a `GateResult` is an immutable record of one decision. Every Build-Spec OPEN DECISION resolved with
  rationale in [`docs/decisions.md`](docs/decisions.md). API stability policy documented:
  `ragwarden.contracts` is frozen and `ragwarden.__all__` is the public surface; breaking it is a
  major bump from v1.0. Coverage audit — 84% on the offline suite (model-dependent detector code is
  covered by the opt-in `-m model` suite); CI enforces `--cov-fail-under=80`. README carries the
  positioning statement and a complementary-not-competitive comparison table
  (RAGAS / DeepEval / UQLM / Guardrails AI / cloud gates).
- **Phase 11 — Packaging, security, release readiness.** `release.yml`: PyPI **Trusted Publishing**
  (OIDC, `id-token: write`, no stored token) with Sigstore attestations on by default; a
  `workflow_dispatch` path to TestPyPI first; a CycloneDX SBOM job whose artifact is attached to the
  GitHub Release alongside the wheel/sdist. `pip-audit --strict` and a `mkdocs build --strict` job
  in CI; `dependabot.yml` for pip + github-actions. `benchmark.yml`: a scheduled RAGTruth run that
  commits fresh numbers to `benchmarks/results/`. Verify the trusted-publisher setup against
  <https://docs.pypi.org/trusted-publishers/> before the first real release.
- **Phase 10 — Documentation site.** MkDocs Material site (`mkdocs.yml` + `docs/`): home, quickstart
  (`pip install` → working gate call), concept pages (the cascade, the policy engine, actions),
  one page per adapter, observability, benchmarks & calibration, honest limitations, and an
  mkdocstrings-generated API reference for the public surface. `mkdocs build --strict` is clean and
  runs in CI.
- **Phase 9 — Observability.** `ragwarden.observability.otel.emit_gate_spans` emits the full
  `ragwarden.gate` span tree per Spec 14.1 (one child per stage, `ragwarden.tier1.detector.<name>`
  leaves, correct parent/child, back-to-back timings reconstructed from the stage trace) with the
  standard attribute keys on the root (`ragwarden.claims_total`, `claims_resolved_tier{0..3}`,
  `reliability_score`, `action`, `cost_estimate_usd`, `request_id`). Wired into `gate()` via
  `emit_telemetry=True` — a no-op when `opentelemetry` is not installed (RagWarden never hard-depends
  on it and never creates a TracerProvider). `ragwarden.observability.logging`: versioned JSON schema
  (`schema_version` from day one) via `gate_result_to_dict` / `gate_result_to_json` /
  `log_gate_result`. `examples/otel_console.py` prints the span tree locally.
- **Phase 8 — Retrieval adapters.** Five adapters that translate host objects into the core
  contracts, each behind its own extra, each duck-typed so it needs no heavy import:
  `from_opensearch_hybrid_response` (preserves raw `bm25_score`/`knn_score` in metadata alongside the
  fused `_score`, records whether scores are normalized), `from_chroma_query_result` (distance →
  similarity, keeps raw distance), `from_langchain_documents` (plain or `(Document, score)` tuples),
  `from_llamaindex_nodes` (`NodeWithScore`), and Docling `enrich_chunk_from_docling` /
  `docling_provenance` (metadata enrichment — page numbers, headings, filename — since Docling is a
  parser, not a retriever). Recorded-fixture test per adapter.
- **Phase 7 — Actions fully implemented.** `GateResult` gains an optional additive `action_payload`
  dict — the structured data a host consumes per action (RETRY loop-back info, ESCALATE review item,
  ALLOW citations, REDACT removed-claims). REDACT_CLAIMS now edits the *original* answer text
  (removes the offending sentences, keeps order and connective text) and falls back to ABSTAIN with a
  reason when redaction leaves nothing coherent. RETRY carries `retries_remaining` and forces ABSTAIN
  once the per-request budget (caller-supplied `GateRequest.retries_used` vs `policy.max_retries`) is
  spent — never loops. ESCALATE returns the original answer flagged plus a full human-review payload
  (verdicts, evidence, reliability score). One dedicated integration test per action incl. the edge
  cases.
- **Phase 6 — Tier 3 LLM-as-judge + full cascade.** `ragwarden.cascade.tier3_judge`: RagWarden owns a
  G-Eval-style structured chain-of-thought prompt; the host passes `judge_fn(prompt) -> completion`
  (any LLM). Robust verdict/confidence parsing (keyword match on the VERDICT line, whole-text
  fallback, `%` confidence). Hard per-request budget — `max_claims_per_request` **and**
  `max_latency_ms`, the latter clamped to a request-wide deadline (`policy.max_total_latency_ms`);
  claims not reached stay `UNRESOLVED`. Judge exceptions and unparseable output fail closed, never
  crash the gate. `gate()` now runs the complete Tier 1→2→3 cascade — only each tier's remainder
  descends — and emits the Tier-3 escalation-rate metric (Spec 7.5) with a retune hint when it
  exceeds `tier3.escalation_rate_warn_threshold`. Verified end-to-end against a local Ollama model.
- **Phase 5 — Tier 2 uncertainty quantification.** `ragwarden.cascade.tier2_uncertainty`: black-box
  consistency sampling — the host's `generate_fn(query) -> answer` is called N times and a reused
  Tier-1 detector scores how consistently each unresolved claim is supported across the fresh
  samples; high agreement resolves SUPPORTED, near-zero agreement resolves UNSUPPORTED, a split stays
  UNRESOLVED for Tier 3. Optional white-box signal: mean token probability over the claim span when
  `Generation.token_logprobs` is present. `N` is configurable down to 0 (Tier 2 off); estimated
  dollar cost flows into `StageTrace.cost_estimate_usd`. Wired into `gate()` behind a `generate_fn=`
  argument — only the Tier-1 remainder reaches it.
- **Phase 4 — Remaining Tier-1 detectors + ensembling.** `HHEMDetector` (`[hhem]`, wraps Vectara
  HHEM-2.1-Open via `model.predict`), `LettuceDetectAdapter` (`[lettucedetect]`, span-level → per-claim
  verdict), `MiniCheckDetector` (`[minicheck]` — install the fact-checker from git; PyPI `minicheck`
  is an unrelated project). Ensembling across multiple installed detectors (`max_confidence`,
  `majority`, `most_severe`) is wired through `policy.tier1.ensemble` / `ensemble_strategy` and
  exercised end-to-end.
- **Phase 3 — Benchmark harness + calibration.** `ragwarden.benchmark` package: RAGTruth loader
  (`wandb/RAGTruth-processed` mirror, MIT, via the `[bench]` extra), a JSONL loader for a user's own
  labeled data, context-blob → chunk splitting, and no-dependency metrics (response-level P/R/F1 +
  RAGTruth-style span-token metrics). `run_benchmark` executes the cascade over a dataset and writes
  a timestamped, non-aspirational results file to `benchmarks/results/` recording the response-level
  numbers, per-task breakdown, cascade tier-resolution %, Tier-3 escalation rate, latency
  percentiles, policy summary, and environment. `run_calibration` sweeps the `allow` threshold and
  reports the hallucination-catch vs over-abstention trade-off with a recommended threshold. CLI:
  `ragwarden benchmark` / `ragwarden calibrate` (+ `python -m ragwarden.cli`).
- **Phase 2 — Tier 1 claim-vs-evidence entailment.** `Detector` protocol + `BaseDetector`
  convenience base with a per-claim `check_batch` fallback. `NLIDetector` (`[nli]` extra) wrapping a
  DeBERTa-v3-class NLI model (default `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), label
  positions resolved from model config, per-chunk aggregation. `run_tier1` with a confidence-gated
  escalation seam (`Tier1Outcome.resolved` / `.unresolved`) and configurable ensembling
  (`max_confidence`, `majority`, `most_severe`). `gate()` now runs decomposition → Tier 0 → Tier 1 →
  severity-weighted scoring → policy, with a loud warning + Tier-0-only fallback when no Tier-1
  detector is installed, and a `detectors=` injection hook. Deterministic `KeywordStubDetector` /
  `ScriptedStubDetector` for offline testing. Real-model tests gated behind `pytest -m model`.
- **Phase 0 — Foundations.** Repository scaffold (`src/` layout, Hatchling build).
  Frozen core contracts (`ragwarden.contracts`): `RetrievedChunk`, `RetrievalContext`,
  `Generation` protocols; `ClaimStatus` / `GateAction` enums; `ClaimVerdict`, `StageTrace`,
  `GateResult` dataclasses. Concrete helpers in `ragwarden.models`. Stub modules for every
  cascade tier, detector, adapter, and observability component so imports resolve. Core install
  is ML-dependency-free.
- **Phase 1 — Tier 0 + policy engine.** Deterministic Tier 0 retrieval heuristics
  (empty-retrieval short-circuit, score floor, score gap, query–context overlap). Policy engine
  with YAML config loading, ordered first-match rule evaluation, pluggable escalation triggers,
  and `fail_closed` default. Sentence-split claim decomposition (`pysbd`). Composite scoring
  strategies (`pass_through`, `severity_weighted`). Action execution for all five actions,
  including the REDACT→ABSTAIN incoherence fallback and RETRY budget guard. Top-level `gate()`
  orchestration with per-stage latency tracing.
