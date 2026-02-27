# CortexaDB: SQLite for AI Agents

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-brightgreen.svg)](#current-status)
[![Version](https://img.shields.io/badge/Version-0.1.2-blue.svg)](https://github.com/anaslimem/CortexaDB/releases)

**CortexaDB** is a simple, fast, and hard-durable embedded database designed specifically for AI agent memory. It provides a single-file-like experience (no server required) but with native support for vectors, graphs, and temporal search.

Think of it as **SQLite, but with semantic and relational intelligence for your agents.**

---

## Quickstart

### Python (Recommended)
CortexaDB is designed to be extremely easy to use from Python via high-performance Rust bindings.

```python
from cortexadb import CortexaDB
from cortexadb.providers.openai import OpenAIEmbedder

# Open database with embedder (auto-embeds text)
db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

# Store memories
db.remember("The user prefers dark mode.")
db.remember("User works at Stripe.")

# Load a file (TXT, MD, JSON, DOCX, PDF)
db.load("document.pdf", strategy="recursive")

# Ask questions (Semantic Search)
hits = db.ask("What does the user like?")
for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}")

# Connect memories (Graph Relationships)
db.connect(mid1, mid2, "relates_to")
```

---

## Installation

### Python
CortexaDB is available on PyPI and can be installed via `pip`:

```bash
# Recommended: Install from PyPI
pip install cortexadb

# With document support (DOCX, PDF)
pip install cortexadb[docs]
pip install cortexadb[pdf]

# From GitHub (Install latest release)
pip install "cortexadb @ git+https://github.com/anaslimem/CortexaDB.git#subdirectory=crates/cortexadb-py"
```

### Rust
Add CortexaDB to your `Cargo.toml`:
```toml
[dependencies]
cortexadb-core = { git = "https://github.com/anaslimem/CortexaDB.git" }
```

---

## Key Features

- **Hybrid Retrieval**: Combine vector similarity (semantic), graph relations (structural), and recency (temporal) in a single query.
- **Smart Chunking**: Multiple strategies for document ingestion - `fixed`, `recursive`, `semantic`, `markdown`, `json`.
- **File Support**: Load documents directly - TXT, MD, JSON, DOCX, PDF.
- **HNSW Indexing**: Ultra-fast approximate nearest neighbor search using USearch (95%+ recall at millisecond latency).
- **Hard Durability**: Write-Ahead Log (WAL) and Segmented logs ensure your agent never forgets, even after a crash.
- **Multi-Agent Namespaces**: Isolate memories between different agents or workspaces within a single database file.
- **Deterministic Replay**: Record operations to a log file and replay them exactly to debug agent behavior or migrate data.
- **Automatic Capacity Management**: Set `max_entries` or `max_bytes` and let CortexaDB handle LRU/Importance-based eviction automatically.
- **Crash-Safe Compaction**: Background maintenance that keeps your storage lean without risking data loss.

---

## HNSW Indexing

CortexaDB uses **USearch** for high-performance approximate nearest neighbor search. Switch between exact and HNSW modes based on your needs:

| Mode | Use Case | Recall | Speed |
|------|----------|--------|-------|
| `exact` | Small datasets (<10K) | 100% | O(n) |
| `hnsw` | Large datasets | 95%+ | O(log n) |

```python
from cortexadb import CortexaDB, HashEmbedder

# Default: exact (brute-force)
db = CortexaDB.open("db.mem", dimension=128)

# Or use HNSW for large-scale search
db = CortexaDB.open("db.mem", dimension=128, index_mode="hnsw")

# HNSW with custom parameters
db = CortexaDB.open("db.mem", dimension=128, index_mode={
    "type": "hnsw",
    "m": 16,           # connections per node
    "ef_search": 50     # query-time search width
})
```

---

## Chunking Strategies

CortexaDB provides 5 smart chunking strategies for document ingestion:

| Strategy | Use Case |
|----------|----------|
| `fixed` | Simple character-based with word-boundary snap |
| `recursive` | General purpose - splits paragraphs → sentences → words |
| `semantic` | Articles, blogs - split by paragraphs |
| `markdown` | Technical docs - preserves headers, lists, code blocks |
| `json` | Structured data - flattens to key-value pairs |

```python
from cortexadb import CortexaDB, chunk

# Use chunk() directly
chunks = chunk(text, strategy="recursive", chunk_size=512, overlap=50)

# Or use db.ingest() / db.load()
db.ingest("text...", strategy="markdown")
db.load("document.pdf", strategy="recursive")
```

---

## File Format Support

| Format | Extension | Install |
|--------|-----------|---------|
| Plain Text | `.txt` | Built-in |
| Markdown | `.md` | Built-in |
| JSON | `.json` | Built-in |
| Word | `.docx` | `pip install cortexadb[docs]` |
| PDF | `.pdf` | `pip install cortexadb[pdf]` |

---

## API Guide

### Core Operations

| Method | Description |
|--------|-------------|
| `CortexaDB.open(path, ...)` | Opens or creates a database at the specified path. |
| `.remember(text, ...)` | Stores a new memory. Auto-embeds if an embedder is configured. |
| `.ingest(text, ...)` | Ingests text with smart chunking. |
| `.load(path, ...)` | Loads and ingests a file. |
| `.ask(query, ...)` | Performs a hybrid search across vectors, graphs, and time. |
| `.connect(id1, id2, rel)` | Creates a directed edge between two memory entries. |
| `.namespace(name)` | Returns a scoped view of the database for a specific agent/context. |
| `.delete_memory(id)` | Permanently removes a memory and updates all indexes. |
| `.compact()` | Reclaims space by removing deleted entries from disk. |
| `.checkpoint()` | Truncates the WAL and snapshots the current state for fast startup. |

### Configuration Options
When calling `CortexaDB.open()`, you can tune the behavior:
- `sync`: `"strict"` (safest), `"async"` (fastest), or `"batch"` (balanced).
- `max_entries`: Limits the total number of memories (triggers auto-eviction).
- `record`: Path to a log file for capturing the entire session for replay.

---

## Technical Essentials: How it's built

<details>
<summary><b>Click to see the Rust Architecture</b></summary>

### Why Rust?
CortexaDB is written in Rust to provide **memory safety without a garbage collector**, ensuring predictable performance (sub-100ms startup) and low resource overhead—critical for "embedded" use cases where the DB runs inside your agent's process.

### The Storage Engine
CortexaDB follows a **Log-Structured** design:
1. **WAL (Write-Ahead Log)**: Every command is first appended to a durable log with CRC32 checksums.
2. **Segment Storage**: Large memory payloads are stored in append-only segments.
3. **Deterministic State Machine**: On startup, the database replays the log into an in-memory state machine. This ensures 100% consistency between the disk and your queries.

### Hybrid Query Engine
Unlike standard vector DBs, CortexaDB doesn't just look at distance. Our query planner can:
- **Vector**: Find semantic matches using Cosine Similarity.
- **Graph**: Discover related concepts by traversing edges created with `.connect()`.
- **Temporal**: Boost or filter results based on when they were "remembered".

### Smart Chunking
The chunking engine is built in Rust for performance:
- 5 strategies covering most use cases
- Word-boundary awareness to avoid splitting words
- Overlap support for context continuity
- JSON flattening for structured data

### Versioned Serialization
We use a custom versioned serialization layer (with a "magic-byte" header). This allows us to update the CortexaDB engine without breaking your existing database files—it knows how to read "legacy" data while writing new records in the latest format.

</details>

---

## License & Status
CortexaDB is currently in **Beta (v0.1.1)**. It is released under the **MIT** and **Apache-2.0** licenses.  
We are actively refining the API and welcome feedback!

---
> *CortexaDB — Because agents shouldn't have to choose between speed and a soul (memory).*
