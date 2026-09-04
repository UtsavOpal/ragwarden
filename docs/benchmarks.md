# Benchmarks & calibration

`pip install 'ragwarden[nli,bench]'`.

## Reproduce the published numbers

```bash
ragwarden benchmark --dataset ragtruth --split test --detector nli --shuffle-seed 42
# add --limit N for a partial run
```

RAGTruth is loaded from `wandb/RAGTruth-processed` (MIT). Each run writes a **non-aspirational**
results file to `benchmarks/results/` recording the response-level precision / recall / F1 (positive
= "response contains a hallucination"), a per-task breakdown (QA / Summary / Data2txt), cascade
tier-resolution percentages, the Tier-3 escalation rate, latency percentiles, the policy summary, and
the environment.

### Current baseline (honest, not competitive yet)

150 shuffled RAGTruth test rows, DeBERTa-v3-base NLI, default policy, Tiers 0–1:

| | Precision | Recall | F1 |
|---|---|---|---|
| Response-level (overall) | 0.42 | 0.98 | 0.59 |
| Data2txt | 0.73 | 1.00 | 0.85 |
| Summary | 0.28 | 0.93 | 0.43 |
| QA | 0.22 | 1.00 | 0.36 |

The gate over-flags (very high recall, low precision) — the expected over-abstention baseline before
Tier 2/3, calibration, and better claim decomposition. It is the number to improve from.

## Calibrate against your own traffic

```bash
ragwarden calibrate --dataset labeled.jsonl --detector nli
```

`labeled.jsonl` — one JSON object per line: `{"query": ..., "context": [...] or str,
"answer": ..., "label": 0|1}`. The command sweeps the `allow` threshold and prints the
hallucination-catch vs over-abstention trade-off with a recommended threshold.

Calibration against your own traffic is not optional polish — it is the product experience that
determines whether the gate is usable in practice.
