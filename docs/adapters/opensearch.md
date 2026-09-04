# OpenSearch adapter

`pip install 'ragwarden[opensearch]'` (the extra is only for your own client code — the adapter
works on the raw response dict).

```python
from ragwarden.adapters.opensearch import from_opensearch_hybrid_response

response = client.search(index="docs", body=hybrid_query, search_pipeline="nlp-search-pipeline")
context = from_opensearch_hybrid_response("your query", response)
result = gate(context, answer)
```

## Score normalization — get this right early

OpenSearch hybrid queries fuse BM25 and kNN sub-scores via the `normalization-processor` in a search
pipeline. Whether `_score` in the hits is the **normalized fused** value or a **raw single-method**
score changes how Tier 0's `min_top_score` / `score_gap_threshold` should be tuned — a mismatch
silently breaks those heuristics.

- Pass `scores_are_normalized=True` (default) when you use the normalization processor. It is
  recorded in `chunk.metadata["scores_normalized"]`.
- Point `bm25_score_field` / `knn_score_field` at fields your pipeline writes into `_source` (or set
  them to `None`). When present they are copied to `chunk.metadata["bm25_score"]` / `["knn_score"]`
  so Tier 0 can reason per-method.

::: ragwarden.adapters.opensearch.from_opensearch_hybrid_response
