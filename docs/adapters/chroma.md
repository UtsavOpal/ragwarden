# ChromaDB adapter

`pip install 'ragwarden[chroma]'` (the adapter works on the plain result dict).

```python
from ragwarden.adapters.chroma import from_chroma_query_result

res = collection.query(query_texts=[query], n_results=5)
context = from_chroma_query_result(query, res, distance_metric="cosine")
result = gate(context, answer)
```

Chroma returns **distances** (lower = closer). The adapter converts them to a similarity `score` in
`[0, 1]` for Tier 0 and keeps the raw value in `chunk.metadata["distance"]`. Set `distance_metric`
to match your collection (`cosine`, `l2`, `ip`).

::: ragwarden.adapters.chroma.from_chroma_query_result

::: ragwarden.adapters.chroma.distance_to_similarity
