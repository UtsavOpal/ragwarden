# SPDX-License-Identifier: Apache-2.0
"""Dataset loaders for the benchmark harness.

RAGTruth (Niu et al., ACL 2024) is loaded from the ``wandb/RAGTruth-processed``
mirror on the Hugging Face Hub (MIT license). Verify the mirror / access method
is still current before publishing numbers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from ragwarden.models import Chunk

__all__ = ["RawSample", "load_jsonl", "load_ragtruth", "split_context_into_chunks"]

_PASSAGE_MARKER = re.compile(r"(?im)^\s*(?:passage|document|context)\s*\d*\s*[:.\-]\s*")


@dataclass
class RawSample:
    """One labeled example, dataset-agnostic."""

    id: str
    query: str
    context_chunks: list[Chunk]
    output: str
    is_hallucinated: bool
    hallucinated_spans: list[tuple[int, int]] = field(default_factory=list)
    task_type: str = "unknown"
    source_dataset: str = "unknown"


def split_context_into_chunks(context: str) -> list[Chunk]:
    """Split a RAGTruth ``context`` blob into evidence chunks.

    RAGTruth QA contexts concatenate multiple passages with 'passage N:' markers;
    summarization / data-to-text contexts are a single document. We split on those
    markers first, then on blank lines, and fall back to the whole string.
    """
    context = (context or "").strip()
    if not context:
        return []
    parts = _PASSAGE_MARKER.split(context)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()] or [context]
    return [
        Chunk(text=p, score=1.0, source_id=f"ctx-{i}", metadata={"chunk_index": i})
        for i, p in enumerate(parts)
    ]


def _spans_from_ragtruth_labels(raw: object, output: str) -> list[tuple[int, int]]:
    """RAGTruth span labels carry the offending substring and/or offsets."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    spans: list[tuple[int, int]] = []
    for label in raw if isinstance(raw, list) else []:
        if not isinstance(label, dict):
            continue
        start, end = label.get("start"), label.get("end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append((start, end))
            continue
        text = label.get("text") or label.get("span")
        if isinstance(text, str) and text:
            idx = output.find(text)
            if idx >= 0:
                spans.append((idx, idx + len(text)))
    return spans


def load_ragtruth(
    split: str = "test",
    *,
    task_types: list[str] | None = None,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    hf_path: str = "wandb/RAGTruth-processed",
) -> list[RawSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - trivial
        raise ImportError(
            "RAGTruth loading needs the 'datasets' library: pip install 'ragwarden[bench]'"
        ) from exc

    ds = load_dataset(hf_path, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    out: list[RawSample] = []
    for row in ds:
        task = str(row.get("task_type", "unknown"))
        if task_types and task not in task_types:
            continue
        output = str(row.get("output", ""))
        processed = row.get("hallucination_labels_processed") or {}
        if isinstance(processed, dict):
            n = sum(v for v in processed.values() if isinstance(v, (int, float)))
            is_hallucinated = n > 0
        else:
            is_hallucinated = bool(row.get("hallucination_labels"))
        out.append(
            RawSample(
                id=str(row.get("id", len(out))),
                query=str(row.get("query", "")),
                context_chunks=split_context_into_chunks(str(row.get("context", ""))),
                output=output,
                is_hallucinated=is_hallucinated,
                hallucinated_spans=_spans_from_ragtruth_labels(
                    row.get("hallucination_labels"), output
                ),
                task_type=task,
                source_dataset="ragtruth",
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def load_jsonl(path: str) -> Iterator[RawSample]:
    """Load a user's own labeled set. One JSON object per line with keys:
    ``query``, ``context`` (str or list of str), ``answer``/``output``, and
    ``label`` / ``is_hallucinated`` (bool or 0/1). Optional ``id``, ``task_type``.
    """
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ctx = row.get("context", "")
            if isinstance(ctx, list):
                chunks = [
                    Chunk(text=str(c), score=1.0, source_id=f"ctx-{j}") for j, c in enumerate(ctx)
                ]
            else:
                chunks = split_context_into_chunks(str(ctx))
            label = row.get("label", row.get("is_hallucinated"))
            yield RawSample(
                id=str(row.get("id", i)),
                query=str(row.get("query", "")),
                context_chunks=chunks,
                output=str(row.get("answer", row.get("output", ""))),
                is_hallucinated=bool(label),
                task_type=str(row.get("task_type", "unknown")),
                source_dataset="jsonl",
            )
