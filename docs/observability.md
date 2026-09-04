# Observability

`pip install 'ragwarden[otel]'` for OpenTelemetry. Without it, span emission is a silent no-op —
RagWarden never hard-depends on OpenTelemetry and never creates a `TracerProvider`. It emits standard
spans into whatever the host already runs (Datadog, Grafana/Tempo, Langfuse, …).

## Span tree

Every `gate()` call emits:

```
ragwarden.gate                             (root)
├── ragwarden.decomposition
├── ragwarden.tier0
├── ragwarden.tier1
│   └── ragwarden.tier1.detector.<name>    (one per detector that ran)
├── ragwarden.tier2
├── ragwarden.tier3
├── ragwarden.scoring
└── ragwarden.policy
```

Standard attributes on the root span (stable keys for dashboards): `ragwarden.claims_total`,
`ragwarden.claims_resolved_tier0/1/2/3`, `ragwarden.reliability_score`, `ragwarden.action`,
`ragwarden.cost_estimate_usd`, `ragwarden.request_id`.

Pass `gate(..., emit_telemetry=False)` to disable.

## Local example

```bash
pip install 'ragwarden[otel]'
python examples/otel_console.py           # prints the span tree
```

For a UI, run Jaeger (`docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one`) and point an
OTLP exporter at it. The span tree is the product — RagWarden does not ship a dashboard.

## Structured logging

Every `GateResult` serializes to a versioned JSON schema (`schema_version` from day one):

```python
from ragwarden.observability.logging import gate_result_to_dict, log_gate_result

log_gate_result(result, request_id="req-1")  # one structured record on the "ragwarden.gate" logger
record = gate_result_to_dict(result)  # or build the dict yourself
```

::: ragwarden.observability.logging.gate_result_to_dict
