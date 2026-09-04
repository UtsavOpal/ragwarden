# Honest limitations

- **A gate cannot fix bad retrieval.** If the wrong evidence was retrieved, no downstream check
  recovers the right answer — this bounds the whole system's ceiling.
- **Every detector has its own error rate.** The composite reliability score is a *calibrated
  confidence estimate*, not ground truth. Don't oversell precision.
- **Over-aggressive gating trades hallucination risk for over-abstention.** This has real UX and
  business cost. [Calibration](benchmarks.md) against your own traffic is not optional polish — it is
  the actual product experience.
- **Latency and cost budgets are real constraints**, not implementation details to hide. The entire
  tiered design exists because "run an LLM judge on everything" is correct but unusable at production
  scale and cost.
- **v0.1 claim decomposition is sentence splitting.** A known simplification vs. true atomic-claim
  extraction (FActScore / RefChecker); it bounds achievable span-level precision.
- **v1.0 gates complete answers**, not token streams, and does not verify image/table claims.
