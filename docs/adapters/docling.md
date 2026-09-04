# Docling adapter

`pip install 'ragwarden[docling]'`.

Docling is a **parser/chunker, not a retriever** — so this adapter is about *metadata enrichment*.
It lifts Docling's structural metadata (page numbers, section headings, filename, bounding boxes)
into `RetrievedChunk.metadata` so chunk provenance survives into the gate's evidence trail
(`ClaimVerdict.supporting_evidence`) and downstream policy rules like "trust chunks from a given
section more."

```python
from ragwarden.adapters.docling import enrich_chunk_from_docling
from ragwarden.models import Chunk

# after your retriever returns a chunk that came from a Docling-parsed document:
chunk = Chunk(text=hit.text, score=hit.score, source_id=hit.id)
chunk = enrich_chunk_from_docling(chunk, docling_chunk.meta)
# chunk.metadata now has: section, headings, page_no, pages, filename, ...
```

::: ragwarden.adapters.docling.enrich_chunk_from_docling

::: ragwarden.adapters.docling.docling_provenance
