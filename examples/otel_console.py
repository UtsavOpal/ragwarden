# SPDX-License-Identifier: Apache-2.0
"""Minimal local OpenTelemetry example (Build Spec Section 14.1).

    pip install 'ragwarden[nli,otel]'   # or [otel] with a stub detector
    python examples/otel_console.py

Prints the RagWarden span tree to the console. Point an OTLP exporter at Jaeger
(docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one) for a UI — the
span tree is the product, not a bundled dashboard.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from ragwarden import GateRequest, gate
from ragwarden.detectors.stub import KeywordStubDetector
from ragwarden.models import Answer, Chunk, Context


def main() -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    context = Context(
        query="When was the Eiffel Tower completed?",
        chunks=[
            Chunk("The Eiffel Tower was completed in 1889 for the World's Fair.", 0.92, "doc-1"),
            Chunk("It stands 330 metres tall on the Champ de Mars in Paris.", 0.71, "doc-2"),
        ],
        retrieval_method="hybrid",
    )
    answer = Answer(text="The Eiffel Tower was completed in 1889. It is 330 metres tall.")

    result = gate(
        context,
        answer,
        detectors=[KeywordStubDetector(support_threshold=0.4)],
        request=GateRequest(request_id="example-1"),
    )
    print(f"\naction={result.action.value}  score={result.reliability_score:.3f}")
    print(result.explanation)


if __name__ == "__main__":
    main()
