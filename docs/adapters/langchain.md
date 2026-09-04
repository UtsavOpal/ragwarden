# LangChain adapter

`pip install 'ragwarden[langchain]'` (the adapter is duck-typed — it only needs `.page_content` and
`.metadata`).

```python
from ragwarden.adapters.langchain import from_langchain_documents

docs = retriever.invoke(query)  # list[Document]
# or: docs = vectorstore.similarity_search_with_relevance_scores(query)  # list[(Document, score)]

context = from_langchain_documents(query, docs, retrieval_method="hybrid")
result = gate(context, Answer(text=chain_output))
```

`source_id` is taken from `metadata` (`source_id` / `id` / `_id` / `source` / `file_path` /
`doc_id`, in that order). Scores come from the tuple, else `metadata[score_key]`, else
`default_score`.

::: ragwarden.adapters.langchain.from_langchain_documents
