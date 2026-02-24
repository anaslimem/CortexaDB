# Mnemos Roadmap

> **Last updated**: February 2026

---

## Vision

Mnemos aims to be:

- **Embedded-first** — no server required
- **Deterministic and crash-safe** — WAL-backed, reproducible agent runs
- **Hybrid retrieval** — vector + graph + temporal
- **Ingestion-aware** — built-in chunking + embeddings
- **Fast startup** — SQLite-like open-and-go
- **Python-friendly** — `pip install mnemos`, no server
- **Multi-agent capable** — locally, with namespaces

**Not** distributed. **Not** cloud-native. **Not** enterprise-first.

---

## Guiding Principles

| Principle | Meaning |
|---|---|
| Embedded before networked | Library-first, server-optional |
| Simplicity over configurability | Minimal API surface, sane defaults |
| Determinism as a differentiator | Reproducible from WAL replay |
| Fast startup | < 100 ms for small/medium DBs |
| Minimal default API | `open → remember → ask → connect → compact` |
| Rust core, Python-first adoption | Performance of Rust, ergonomics of Python |

---

## Phase 1 — Embedded Foundation (0–2 Months)

### 1.1 Extract Pure Rust Core

Create a clear crate separation:

| Crate | Purpose |
|---|---|
| `mnemos-core` | No gRPC, no network. Pure embedded library. |
| `mnemos-server` | Optional binary wrapping core with gRPC. |

`mnemos-core` must:

- Open from a path
- Handle WAL
- Handle segment storage
- Run hybrid retrieval
- Manage namespaces

**Target API:**

```rust
pub struct Mnemos;

impl Mnemos {
    pub fn open(path: &str) -> Result<Self>;
    pub fn remember(&self, text: &str, metadata: Option<Metadata>) -> Result<MemoryId>;
    pub fn ask(&self, query: &str, top_k: usize) -> Result<Vec<Memory>>;
    pub fn connect(&self, from: MemoryId, to: MemoryId) -> Result<()>;
    pub fn compact(&self) -> Result<()>;
}
```

No planner exposure. No internal engine exposure.

### 1.2 Snapshot + WAL Truncation

Implement:

- Periodic snapshot file
- WAL replay only from last snapshot
- WAL truncation after snapshot
- Fast startup path

**Goal:**

- Startup under 100 ms for small/medium DBs
- `O(size_of_recent_ops)` replay

This is mandatory for SQLite-like UX.

### 1.3 Fast Startup Benchmark

Add:

```bash
cargo run --bin startup_bench
```

Measure:

- Cold open time
- Snapshot load time
- WAL replay time

Define hard targets and track regressions.

---

## Phase 2 — Python Native Experience (2–4 Months)

### 2.1 PyO3 Bindings

Create:

- `mnemos-py` crate using PyO3
- Built with [maturin](https://github.com/PyO3/maturin)
- Publish wheels to PyPI

Installation:

```bash
pip install mnemos
```

No server required.

### 2.2 Clean Python API

**Target API:**

```python
from mnemos import Mnemos

db = Mnemos.open("agent.mem")

db.remember("We chose Stripe for payments")
results = db.ask("What payment provider did we choose?")
```

**Optional namespace support:**

```python
agent = db.namespace("agent_a")
shared = db.namespace("shared")
```

Keep it minimal.

### 2.3 Built-in Chunking

Add deterministic chunking:

- Fixed window
- Configurable overlap
- Stable chunk IDs

Expose:

```python
db.ingest_document(text, chunk_size=512, overlap=50)
```

Users should not need external chunkers.

### 2.4 Embedding Plugin System

Add an embedder trait:

```rust
trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
}
```

Python users can pass their own embedder:

```python
db = Mnemos.open("agent.mem", embedder=my_embedder)
```

Provide:

- Simple example embedder
- Clear docs

Embedding must be optional but supported.

---

## Phase 3 — Multi-Agent Local Model (4–6 Months)

### 3.1 Namespace Model

Support:

- Private namespaces per agent
- Shared namespace for inter-agent memory
- Cross-namespace queries (optional)

Design:

```python
agent_a = db.namespace("agent_a")
agent_b = db.namespace("agent_b")
shared  = db.namespace("shared")
```

No distributed logic. Purely local partitioning.

### 3.2 Deterministic Replay (Killer Feature)

Expose:

```python
db.export_replay("run.log")
db.replay("run.log")
```

Use cases:

- Agent debugging
- Reproducible experiments
- Deterministic evaluation

This becomes a core differentiator.

### 3.3 Hybrid Query Simplification

Internally keep planner logic. Externally expose:

```python
db.ask(query)
db.ask(query, use_graph=True)
db.ask(query, recency_bias=True)
```

Hide advanced tuning unless explicitly requested.

---

## Phase 4 — Stability & Performance (6–9 Months)

### 4.1 Concurrency Improvements

Implement:

- Snapshot reads during writes (MVCC-lite)
- Minimize global locking
- Stable latency under ingestion load

**Goal:** Smooth performance in hackathon-scale workloads.

### 4.2 Capacity Policies

Improve eviction:

- Deterministic
- Transparent
- Observable

Add:

```python
db.set_capacity(max_entries=10_000)
```

### 4.3 Compaction Maturity

Improve:

- Live entry rewrite efficiency
- Space recovery
- Compaction benchmarks

Ensure compaction is safe under crash.

---

## Phase 5 — Optional Enhancements (After Stability)

> These are optional and should not delay core UX.

### 5.1 ANN Backend (Optional)

Add HNSW-based ANN for larger datasets. Only if:

- Exact search becomes a bottleneck
- Target use case grows beyond 100k–1M entries

### 5.2 Memory Types

Add:

```rust
enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
}
```

Use for ranking bias.

### 5.3 Semantic Compaction

Future idea:

- Merge similar memories
- Summarize older entries
- Deterministic LLM-assisted compaction

This differentiates Mnemos from generic vector DBs.

---

## Release Milestones

### v0.2 — Embedded Stable Core
- Pure Rust core (`mnemos-core` crate, no network deps)
- Snapshot support (fast startup from snapshot + WAL tail)
- Clean Rust API (`open → remember → ask → connect → compact`)
- Fast startup (< 100 ms target for small/medium DBs)

### v0.3 — Python Native Release
- PyPI package (`pip install mnemos`, no server required)
- Clean Python API (`Mnemos.open` / `.remember` / `.ask`)
- Built-in chunking (`ingest_document` with configurable overlap)
- Embedder trait (pluggable embedding backend)

### v0.4 — Multi-Agent Local Model
- Namespaces (private + shared, local partitioning)
- Deterministic replay (`export_replay` / `replay`)
- Polished hybrid API (`use_graph`, `recency_bias` flags)

### v1.0 — SQLite-Level Local Stability
- Stable file format (backward-compatible snapshots + segments)
- Crash-tested durability (fuzz + fault-injection CI)
- Mature compaction (safe under crash, benchmarked)
- Production-ready embedded use

---

## Getting Started — Approach for Phase 1

Below is a concrete plan mapping the **existing codebase** to the first phase of work.

### What Already Exists

| Component | Status | Location |
|---|---|---|
| WAL (write-ahead log) | ✅ Done | `src/storage/wal.rs` |
| Segment storage | ✅ Done | `src/storage/segment.rs` |
| Compaction | ✅ Done | `src/storage/compaction.rs` |
| State machine | ✅ Done | `src/core/state_machine.rs` |
| Vector / Graph / Temporal indexes | ✅ Done | `src/index/` |
| Query planner + executor | ✅ Done | `src/query/` |
| Engine orchestrator | ✅ Done | `src/engine.rs` |
| Unified facade (`MnemosStore`) | ✅ Done | `src/store.rs` |
| gRPC service + server binary | ✅ Done | `src/service/`, `src/bin/mnemos_grpc.rs` |
| Checkpointing (periodic) | ✅ Done | `src/store.rs` (checkpoint thread) |
| Python client (gRPC-based) | ✅ Done | `python/mnemos-client/` |
| Snapshot + WAL truncation | ❌ Missing | — |
| Embedded `Mnemos` facade (no network) | ❌ Missing | — |
| PyO3 native bindings | ❌ Missing | — |
| Startup benchmark | ❌ Missing | — |

### Recommended First Steps

#### Step 1 — Convert to a Cargo workspace

Convert the single crate into a workspace with three members:

```text
Cargo.toml          (workspace root)
crates/
  mnemos-core/      (library: storage, indexes, query, engine, facade)
  mnemos-server/    (binary: gRPC layer, depends on mnemos-core)
  mnemos-py/        (future: PyO3 bindings, depends on mnemos-core)
```

This is the single most important structural change — it enables every subsequent phase.

**Action items:**
1. Create `crates/mnemos-core/` and move `src/core/`, `src/storage/`, `src/index/`, `src/query/`, `src/engine.rs`, `src/store.rs` into it.
2. Create `crates/mnemos-server/` with `src/service/`, `src/bin/`, and the proto build.
3. Make `mnemos-core` have **zero** network dependencies (no `tonic`, no `prost`).
4. Update the workspace `Cargo.toml` and ensure `cargo test` still passes.

#### Step 2 — Create the `Mnemos` embedded facade

In `mnemos-core`, Add a thin `Mnemos` struct wrapping `MnemosStore` with the simple target API:

```rust
pub struct Mnemos { inner: MnemosStore }

impl Mnemos {
    pub fn open(path: &str) -> Result<Self> { ... }
    pub fn remember(&self, text: &str, metadata: Option<Metadata>) -> Result<MemoryId> { ... }
    pub fn ask(&self, query: &str, top_k: usize) -> Result<Vec<Memory>> { ... }
    pub fn connect(&self, from: MemoryId, to: MemoryId) -> Result<()> { ... }
    pub fn compact(&self) -> Result<()> { ... }
}
```

This delegates to `MnemosStore` internally but hides planner/engine/index details.

#### Step 3 — Implement snapshot + WAL truncation

The checkpointing infra already exists in `store.rs`. Extend it to:

1. Write a binary snapshot file (serialize `StateMachine` state via bincode).
2. On open, load the latest snapshot first, then replay only WAL entries written after.
3. After a successful snapshot, truncate the WAL up to the snapshot point.

#### Step 4 — Add `startup_bench`

Create `crates/mnemos-core/src/bin/startup_bench.rs`:

- Seeds a DB with N entries
- Closes and reopens it
- Measures cold open, snapshot load, and WAL replay times
- Prints results and compares against target thresholds
