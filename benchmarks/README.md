# Benchmark results

Files in `results/` are the **actual output** of `ragwarden benchmark` at a point in the project's
history — never hand-edited or aspirational numbers. They are checked in (not gitignored) so the
project's progress is auditable.

## Reproduce

```bash
pip install 'ragwarden[nli,bench]'
ragwarden benchmark --dataset ragtruth --split test --detector nli --shuffle-seed 42
# add --limit N for a quick partial run
```

Each result file records the dataset + split, `n_samples`, the detector and policy used, the
response-level precision / recall / F1 (positive = "response contains a hallucination"), a per-task
breakdown (QA / Summary / Data2txt), cascade tier-resolution percentages, the Tier-3 escalation
rate, latency percentiles, and the environment.

## Protocol notes

- **Response-level** metrics compare "any claim CONTRADICTED/UNSUPPORTED (and, by default, UNRESOLVED)"
  against RAGTruth's `hallucination_labels_processed` (`evident_conflict` + `baseless_info` counts).
- RAGTruth `context` blobs are split into evidence chunks on `passage N:` markers, then blank lines.
- Sentence-split decomposition (v0.1) is a known simplification vs. true atomic-claim extraction; it
  bounds the achievable span-level precision. See the build spec Section 7.1.
- Early numbers are a **baseline to improve from**, not a competitive claim.
