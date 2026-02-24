# Mnemos

[![Build](https://github.com/anaslimem/Mnemos/actions/workflows/rust.yml/badge.svg)](https://github.com/anaslimem/Mnemos/actions/workflows/rust.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)](#current-status)

> ** Experimental** — Mnemos is under active development (v0.1). APIs may change. See [Known Limitations](KNOWN_LIMITATIONS.md).

Mnemos is a simple, fast, and reliable vector + graph database for AI agents — like SQLite, but for agent memory.  
It provides durable memory storage, deterministic replay, and hybrid retrieval across:

- vector similarity (semantic search),
- graph relationships (connected memories),
- temporal constraints (time-aware recall).

Mnemos is built as an **embedded Rust library** — no server required.

## Why Mnemos

General databases are excellent at generic data workloads. Agent memory workloads need extra semantics:

- append-first durability for command replay,
- memory-native fields (`importance`, `created_at`, embeddings),
- hybrid retrieval combining meaning + relations + recency,
- deterministic behavior for reproducible agent runs.

Mnemos is designed specifically for that shape of problem.

## Current Capabilities

- Durable command log (WAL) with checksum validation.
- Append-only segment storage for memory payloads.
- Deterministic in-memory state machine.
- Vector index with cosine similarity (serial + Rayon parallel search).
- Graph traversal index (BFS/DFS/pathfinding).
- Temporal index for range and recency operations.
- Combined weighted retrieval.
- Capacity/eviction engine (`max_entries`, `max_bytes`) using deterministic delete commands.
- Segment compaction with atomic directory swap.

## Quickstart

```bash
git clone https://github.com/anaslimem/Mnemos.git
cd Mnemos
cargo run -p mnemos-core --bin manual_store
```

This runs a local demo that inserts memories, creates edges, queries, and prints scored results.

## Project Layout

```text
crates/
  mnemos-core/
    src/
      core/
        memory_entry.rs      # Memory model
        command.rs           # Command/event types
        state_machine.rs     # Deterministic state transitions
      storage/
        wal.rs               # Write-ahead log
        segment.rs           # Segment files + in-memory segment index
        compaction.rs        # Segment compaction
        checkpoint.rs        # State checkpoint serialization
      index/
        vector.rs            # Cosine similarity index
        graph.rs             # Graph traversal index
        temporal.rs          # Time-based index
        combined.rs          # Hybrid index composition
      query/
        planner.rs           # Query planning heuristics
        executor.rs          # Plan execution + metrics/traces
        hybrid.rs            # Text-oriented hybrid query engine
        intent.rs            # Intent-based weight tuning
      engine.rs              # Core orchestrator (WAL + segments + state)
      store.rs               # Unified facade API (MnemosStore)
      bin/
        manual_store.rs      # Local manual demo binary
        sync_bench.rs        # Sync policy benchmark
        monkey_writer.rs     # Crash-safety stress writer
        monkey_verify.rs     # Recovery verification
```

## Architecture Overview

Write path:

1. Client issues a command (`InsertMemory`, `DeleteMemory`, `AddEdge`, ...).
2. Command is appended to WAL.
3. Memory payload is written to segment storage when applicable.
4. Data is synced.
5. State machine applies command.

This ensures recovery can replay the same command sequence deterministically.

Read/query path:

1. Query planner selects execution path (`VectorOnly`, `VectorTemporal`, `VectorGraph`, `WeightedHybrid`), candidate multiplier, and serial/parallel mode.
2. Query executor runs staged retrieval.
3. Results are ranked and returned with score breakdowns.

## Storage Details

### WAL

Record format:

```text
[u32: len][u32: checksum][bytes: bincode(Command)]
```

- Detects corruption via checksum.
- Used for crash recovery and deterministic replay.

### Segment Storage

Record format:

```text
[u32: len][u32: checksum][bytes: bincode(MemoryEntry)]
```

- Append-only `.seg` files with rotation.
- In-memory index: `MemoryId -> (segment_id, offset, length)`.
- Supports logical delete and compaction eligibility analysis.

### Compaction

- Rewrites live entries into a fresh segment directory.
- Performs atomic swap (`old -> backup`, `new -> old`, delete backup).
- Rebuilds segment storage index after swap.

## Querying

### High-level Rust facade

`MnemosStore` wraps:

- `Engine`
- `IndexLayer`
- `QueryPlanner`
- `QueryExecutor`

It provides one practical API surface for ingest, retrieval, capacity, and compaction.

### Weighted ranking

Hybrid ranking combines:

- semantic similarity,
- importance,
- recency.

Weights are configurable and validated to sum to 100.

## Capacity & Eviction

`Engine::enforce_capacity(...)` supports:

- `max_entries`
- `max_bytes`

Eviction order is deterministic:

1. older first (`created_at` ascending),
2. less important first (`importance` ascending),
3. lower `MemoryId` first.

Evictions are issued as `DeleteMemory` commands, so WAL and replay stay consistent.

## Sync Policy Benchmark

Quick local throughput comparison between durability modes:

```bash
cargo run -p mnemos-core --bin sync_bench -- --mode strict --ops 20000
cargo run -p mnemos-core --bin sync_bench -- --mode batch --ops 20000 --batch-max-ops 128 --batch-max-delay-ms 20
cargo run -p mnemos-core --bin sync_bench -- --mode async --ops 20000 --async-interval-ms 20
```

Each run prints ingest/flush/total timing, ops/sec, and a recovery durability check.

Sample local result (`ops=500`, same machine, debug build):

| Mode | Config | Total (ms) | Ops/sec |
|---|---|---:|---:|
| Strict | default | 6614 | 75.59 |
| Batch | `max_ops=64`, `max_delay_ms=10` | 3198 | 156.32 |
| Async | `interval_ms=10` | 3105 | 160.98 |

These numbers are workload and hardware dependent, but show the expected pattern:
`batch`/`async` improve write throughput compared with `strict`.

## Test

Run all tests:

```bash
cargo test --workspace -- --nocapture
```

## Current Status

Mnemos is already usable for:

- single-node agent memory storage/retrieval,
- deterministic recovery.

Areas under active development (see [Roadmap](ROADMAP.md)):

- embedded `Mnemos` facade with simplified API,
- snapshot + WAL truncation for fast startup,
- PyO3 native Python bindings,
- multi-agent namespace model.

## Documentation

- [ROADMAP.md](ROADMAP.md) — project roadmap and phased development plan
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — current constraints and caveats
