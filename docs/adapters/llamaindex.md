# LlamaIndex adapter

`pip install 'ragwarden[llamaindex]'` (the adapter is duck-typed against `NodeWithScore`).

```python
from ragwarden.adapters.llamaindex import from_llamaindex_nodes

nodes = retriever.retrieve(query)  # list[NodeWithScore]
context = from_llamaindex_nodes(query, nodes, retrieval_method="hybrid")
result = gate(context, Answer(text=str(response)))
```

`source_id` comes from `node.node_id` / `node.id_` / `metadata["source"]`. Content is read via
`node.get_content()`, falling back to `node.text`.

::: ragwarden.adapters.llamaindex.from_llamaindex_nodes
