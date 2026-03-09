<div align="center">
  <img src="https://raw.githubusercontent.com/anaslimem/CortexaDB/main/logo.png" alt="CortexaDB Logo" width="200" />
</div>

<h1 align="center">CortexaDB</h1>
<p align="center">
  <small>SQLite for AI Agents</small>
</p>


<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg" alt="License" /></a>
  <a href="#current-status"><img src="https://img.shields.io/badge/Status-Beta-brightgreen.svg" alt="Status" /></a>
  <a href="https://github.com/anaslimem/CortexaDB/releases"><img src="https://img.shields.io/badge/Version-0.1.8-blue.svg" alt="Version" /></a>
  <a href="https://pepy.tech/projects/cortexadb"><img src="https://static.pepy.tech/personalized-badge/cortexadb?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads" alt="Downloads" /></a>
</p>


**CortexaDB** is a lightweight, high-performance embedded database built in Rust, specifically designed to serve as the long-term memory for AI agents. It provides a single-file, zero-dependency storage solution that combines the simplicity of SQLite with the semantic power of vector search, graph relationships, and temporal indexing.

---

### The Problem: Why CortexaDB?

Current AI agent frameworks often struggle with "memory" once the context window fills up. Developers usually have to choose between complex, over-engineered vector databases (that require a running server) or simple JSON files (that are slow and lose searchability at scale). 

CortexaDB exists to provide a **middle ground**: a hard-durable, embedded memory engine that runs inside your agent's process. It ensures your agent never forgets, starting instantly with zero overhead, and maintaining millisecond query latencies even as it learns thousands of new facts.

---

### Quickstart

```python
from cortexadb import CortexaDB
from cortexadb.providers.openai import OpenAIEmbedder

# 1. Open database with an embedder 
db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

# 2. Add facts 
mid1 = db.add("The user prefers dark mode.")
mid2 = db.add("User works at Stripe.")
db.connect(mid1, mid2, "relates_to")

# 3. Fluent Query Builder
hits = db.query("What are the user's preferences?") \
    .limit(5) \
    .use_graph() \
    .execute()

print(f"Top Hit: {hits[0].id}")
```

---

### Installation

CortexaDB is available on PyPI for Python and can be added via Cargo for Rust.

**Python**
```bash
pip install cortexadb
pip install cortexadb[docs,pdf]  # Optional: For PDF/Docx support
```

---

### Core Capabilities

- **100x Faster Ingestion**: New batch insertion system allows processing 5,000+ chunks/second.
- **Hybrid Retrieval**: Search by semantic similarity (Vector), structural relationship (Graph), and time-based recency in a single query.
- **Ultra-Fast Indexing**: Uses **HNSW (USearch)** for sub-millisecond approximate nearest neighbor search.
- **Fluent API**: Chainable QueryBuilder for expressive searching and collection scoping.
- **Hard Durability**: WAL-backed storage ensures zero data loss.
- **Privacy First**: Completely local. Your agent's memory stays on your machine.

---

<details>
<summary><b>Technical Architecture & Benchmarks</b></summary>

### Performance Benchmarks (v0.1.8)

CortexaDB `v0.1.8` introduced a new batching architecture. Measured on an M2 Mac with 1,000 chunks of text:

| Operation | v0.1.6 (Sync) | v0.1.8 (Batch) | Improvement |
|-----------|---------------|----------------|-------------|
| Ingestion | 12.4s         | **0.12s**      | **103x Faster** |
| Memory Add| 15ms          | 1ms            | 15x Faster |
| HNSW Search| 0.3ms        | 0.28ms         | - |

</details>

---

## License & Status
CortexaDB is currently in **Beta (v0.1.8)**. It is released under the **MIT** and **Apache-2.0** licenses.  
We are actively refining the API and welcome feedback!

---
> *CortexaDB — Because agents shouldn't have to choose between speed and a soul (memory).*
