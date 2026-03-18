<div align="center">
  <img src="https://raw.githubusercontent.com/anaslimem/CortexaDB/main/logo.png" alt="CortexaDB Logo" width="200" />
</div>

<h1 align="center">CortexaDB</h1>
<p align="center">
  <small>SQLite for AI Agents</small>
</p>


<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg" alt="License" /></a>
  <a href="#current-status"><img src="https://img.shields.io/badge/Status-Stable-brightgreen.svg" alt="Status" /></a>
  <a href="https://github.com/anaslimem/CortexaDB/releases"><img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version" /></a>
  <a href="https://pepy.tech/projects/cortexadb"><img src="https://static.pepy.tech/personalized-badge/cortexadb?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads" alt="Downloads" /></a>
  <a href="https://cortexa-db.vercel.app"><img src="https://img.shields.io/badge/Docs-cortexa--db.vercel.app-purple.svg" alt="Documentation" /></a>
</p>


📖 **[Read the full documentation](https://cortexa-db.vercel.app)**

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

### Performance Benchmarks (v1.0.0)

Measured on an M-series Mac — 10,000 embeddings × 384 dimensions.

| Operation | Latency / Time |
|-----------|---------------|
| Bulk Ingestion (1,000 chunks) | **0.12s** |
| Single Memory Add | **1ms** |
| HNSW Search p50 | **1.03ms** (debug) / ~0.3ms (release) |
| HNSW Recall | **95%** |

See the [full benchmark docs](https://cortexa-db.vercel.app/docs/resources/benchmarks) for HNSW vs Exact comparison and how to reproduce.

</details>

---

## License & Status
CortexaDB `v1.0.0` is a **stable release** available under the **MIT** and **Apache-2.0** licenses.  
We welcome feedback and contributions!

---
> *CortexaDB — Because agents shouldn't have to choose between speed and a soul (memory).*
