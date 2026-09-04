# Design decisions

Every decision the build spec (Section 17) left open, resolved with rationale. Where a decision could
reasonably go either way, it is implemented behind a config flag so it is never silently made.

| Decision | Resolution | Rationale |
|---|---|---|
| **Claim decomposition** — sentence-split vs. model-based | Ship `SentenceSplitDecomposer` (`pysbd`) as the v0.1 default; `decomposition.strategy` exposes `llm`. | Sentence splitting has zero heavy deps and is good enough to validate the cascade. Model-based extraction is a real pipeline stage with its own latency/cost and should be opt-in until it earns its place. Bounds achievable span-level precision — stated in [limitations](limitations.md). |
| **`fail_mode` default** | `fail_closed`. | A detector outage should reduce answer *availability*, not silently reduce *protection*. `fail_open` is one line of config for teams that would rather ship. |
| **Default Tier-1 detector** | `NLIDetector` → `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT). | Clearest license, best-documented checkpoint, CPU-fast, no `trust_remote_code`. The `-large` variant and HHEM/LettuceDetect are one extra away. |
| **Build backend** | Hatchling. | Well-supported, simple, the backend in the official PyPA packaging tutorial. `uv_build` to be re-evaluated once independently stable. |
| **Third-party model licenses** | No non-OSI model is ever a default or required dependency. The MiniCheck fact-checker (some checkpoints Llama-derived) installs from git and is clearly labeled; prefer its Flan-T5/RoBERTa checkpoints. | Keeps `pip install ragwarden` and every documented default fully permissive OSS. |
| **`[hhem]` transformers pin** | `transformers>=4.40,<5`. | HHEM-2.1-Open's `trust_remote_code` model code is not yet compatible with `transformers` 5.x (verified 2026-09). `[nli]` has no such cap. Re-check on the next HHEM release. |
| **`mkdocs.yml` location** | Repo root, not `docs/` (the spec's sketch). | mkdocs-material convention; lets `mkdocs build --strict` run with no extra path config. |
| **Benchmark harness package** | `ragwarden.benchmark` (internal) with thin `ragwarden.cli` shims. | The spec sketches `cli/benchmark.py`; a package keeps the harness importable and testable without going through argparse. |
| **Unhandled cascade failures** | `gate()` catches anything unexpected and returns `ABSTAIN` (never raises), except `Policy.validate()` errors which still raise immediately. `raise_on_error=True` opts back into raising. | A broken gate must fail toward "say nothing," not toward crashing the host's request or silently shipping an unverified answer. A bad *config*, unlike a runtime failure, is a deploy-time bug that should be loud, not hidden behind a graceful-looking `ABSTAIN`. See [production.md](production.md). |
| **Timeout mechanism for host callables** | A shared, bounded (`64` workers) `ThreadPoolExecutor`; `future.result(timeout=...)`. | Portable across any synchronous callable (HTTP client, local model, anything) without requiring `asyncio` in the caller. Documented limitation: Python cannot forcibly kill a thread, so a callable that ignores the timeout keeps running in the background — the wrapper bounds *RagWarden's* wait, not the callable's own execution; well-behaved callables should still set their own client-level timeout. |
| **Default per-call timeouts** | `tier2.call_timeout_s=10`, `tier3.call_timeout_s=15`, `tier1_call_timeout_s=None`. | Tier 2/3 call host LLMs (network-bound) — a generous but finite default beats hanging forever. Tier 1 detectors are normally local model inference (no network); defaulting to no cap avoids surprising a legitimately slow first (cold) call. Both are one config line to change. |
| **`max_claims_per_request` default** | `50`, applied right after decomposition, before any tier runs. | Bounds worst-case cost from a pathologically long or adversarial answer. High enough not to affect any realistic RAG answer; low enough to cap a runaway generation. |
| **Async API** | `agate()` wraps the fully-synchronous `gate()` via `asyncio.to_thread` rather than an independent async implementation. | `gate()`'s cost is model inference and blocking I/O in host callables — there is no async-native benefit to reimplementing it; the only thing that matters is not blocking the event loop, which `to_thread` solves directly. |

## API stability

The interfaces in `ragwarden.contracts` are **frozen**. `ragwarden.__all__` — `gate`, `GateRequest`,
`Policy`, and the contract types — is the public surface. Everything under `ragwarden.cascade`,
`ragwarden.detectors`, `ragwarden.adapters`, `ragwarden.benchmark`, and `ragwarden.observability` is
implementation detail and may change in a minor release pre-1.0. From v1.0, breaking `contracts`
requires a major version bump.
