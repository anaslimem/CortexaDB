# CortexaDB Documentation

**CortexaDB** is a simple, fast, and hard-durable embedded database designed specifically for AI agent memory. It provides a single-file experience (no server required) with native support for vectors, graphs, and temporal search.

Think of it as **SQLite, but with semantic and relational intelligence for your agents.**

---

## Documentation Overview

### Getting Started
- [Installation](./getting-started/installation.md) - Install CortexaDB via pip or Cargo
- [Quickstart](./getting-started/quickstart.md) - Your first database in 5 minutes

### Guides
- [Core Concepts](./guides/core-concepts.md) - Architecture and how CortexaDB works
- [Storage Engine](./guides/storage-engine.md) - WAL, segments, checkpoints, and compaction
- [Query Engine](./guides/query-engine.md) - Hybrid search with vector, graph, and temporal scoring
- [Indexing](./guides/indexing.md) - Exact search vs HNSW approximate nearest neighbor
- [Chunking](./guides/chunking.md) - Document ingestion and chunking strategies
- [Collections](./guides/collections.md) - Multi-agent memory isolation
- [Embedders](./guides/embedders.md) - Embedding providers (OpenAI, Gemini, Ollama, Hash)
- [Replay & Recording](./guides/replay.md) - Deterministic session recording and replay
- [Configuration](./guides/configuration.md) - All configuration options explained

### API Reference
- [Python API](./api/python.md) - Complete Python API reference
- [Rust API](./api/rust.md) - Rust crate API reference

### Resources
- [Benchmarks](./resources/benchmarks.md) - Performance benchmarks and methodology
- [Examples](./resources/examples.md) - Code examples for common use cases

---

## Key Features

- **Hybrid Retrieval** - Combine vector similarity, graph relations, and recency in a single query
- **Smart Chunking** - 5 strategies for document ingestion (fixed, recursive, semantic, markdown, json)
- **File Support** - Load TXT, MD, JSON, DOCX, and PDF documents directly
- **HNSW Indexing** - Ultra-fast approximate nearest neighbor search via USearch
- **Hard Durability** - Write-Ahead Log and segmented storage ensure crash safety
- **Multi-Agent Namespaces** - Isolate memories between agents within a single database file
- **Deterministic Replay** - Record and replay operations for debugging or migration
- **Automatic Capacity Management** - LRU/importance-based eviction with `max_entries` or `max_bytes`

---

## Quick Example

```python
from cortexadb import CortexaDB
from cortexadb.providers.openai import OpenAIEmbedder

db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

db.add("The user prefers dark mode.")
db.add("User works at Stripe.")

hits = db.search("What does the user like?")
for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}")
```

---

## License

CortexaDB is released under the **MIT** and **Apache-2.0** licenses.
