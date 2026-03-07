<p align="center">
  <img src="https://raw.githubusercontent.com/anaslimem/CortexaDB/main/logo.png" alt="CortexaDB Logo" width="200" />
</p>

# CortexaDB: SQLite for AI Agents

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-brightgreen.svg)](#current-status)
[![Version](https://img.shields.io/badge/Version-0.1.7-blue.svg)](https://github.com/anaslimem/CortexaDB/releases)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cortexadb?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/cortexadb)

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

# 1. Open database with an embedder for automatic text-to-vector
db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

# 2. Store facts and connect them logically
mid1 = db.remember("The user prefers dark mode.")
mid2 = db.remember("User works at Stripe.")
db.connect(mid1, mid2, "relates_to")

# 3. Query with semantic and graph intelligence
hits = db.ask("What are the user's preferences?", use_graph=True)
print(f"Top Hit: {hits[0].id} (Score: {hits[0].score})")
```

---

### Installation

CortexaDB is available on PyPI for Python and can be added via Cargo for Rust.

**Python**
```bash
pip install cortexadb
pip install cortexadb[docs,pdf]  # Optional: For PDF/Docx support
```

**Rust**
```toml
[dependencies]
cortexadb-core = { git = "https://github.com/anaslimem/CortexaDB.git" }
```

---

### Core Capabilities

- **Hybrid Retrieval**: Search by semantic similarity (Vector), structural relationship (Graph), and time-based recency in a single query.
- **Ultra-Fast Indexing**: Uses **HNSW (USearch)** for sub-millisecond approximate nearest neighbor search with 95%+ recall.
- **Hard Durability**: A Write-Ahead Log (WAL) and segmented storage ensure zero data loss, even after a crash.
- **Smart Document Ingestion**: Built-in recursive, semantic, and markdown chunking for TXT, MD, PDF, and DOCX files.
- **Privacy First**: Completely local and embedded. Your agent's data never leaves its environment unless you want it to.
- **Deterministic Replay**: Capture session operations for debugging or syncing memory across different agents.

---

<details>
<summary><b>Technical Architecture & Benchmarks</b></summary>

### Rust Architecture Overview

```
┌──────────────────────────────────────────────────┐
│              Python API (PyO3 Bindings)          │
│   CortexaDB, Namespace, Embedder, chunk(), etc.  │
└────────────────────────┬─────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────┐
│               CortexaDB Facade                   │
│        High-level API (remember, ask, etc.)      │
└────────────────────────┬─────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────┐
│              CortexaDBStore                      │
│    Concurrency coordinator & durability layer    │
│  ┌────────────────┐  ┌────────────────────────┐  │
│  │ WriteState     │  │ ReadSnapshot           │  │
│  │ (Mutex)        │  │ (ArcSwap, lock-free)   │  │
│  └────────────────┘  └────────────────────────┘  │
└───────┬──────────────────┬───────────────┬───────┘
        │                  │               │
┌───────▼─────┐  ┌───────▼───────┐  ┌────▼───────────┐
│   Engine    │  │   Segments    │  │  Index Layer    │
│   (WAL)     │  │   (Storage)   │  │                 │
│             │  │               │  │  VectorIndex    │
│  Command    │  │  MemoryEntry  │  │  HnswBackend    │
│  recording  │  │  persistence  │  │  GraphIndex     │
│             │  │               │  │  TemporalIndex  │
│  Crash      │  │  CRC32        │  │                 │
│  recovery   │  │  checksums    │  │  HybridQuery    │
└─────────────┘  └───────────────┘  └─────────────────┘
                         │
              ┌──────────▼──────────┐
              │    State Machine     │
              │   (In-memory state)  │
              │  - Memory entries    │
              │  - Graph edges       │
              │  - Temporal index    │
              └─────────────────────┘
```

### Performance Benchmarks (v0.1.7)
Measured with 10,000 embeddings (384-dimensions) on a standard SSD.

| Mode | Query (p50) | Throughput | Recall |
|------|-------------|-----------|--------|
| Exact (baseline) | 1.34ms | 690 QPS | 100% |
| HNSW | 0.29ms | 3,203 QPS | 95% |

</details>

---

## License & Status
CortexaDB is currently in **Beta (v0.1.7)**. It is released under the **MIT** and **Apache-2.0** licenses.  
We are actively refining the API and welcome feedback!

---
> *CortexaDB — Because agents shouldn't have to choose between speed and a soul (memory).*
