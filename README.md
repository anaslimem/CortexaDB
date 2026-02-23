# Mnemos

Mnemos is a Rust-first memory engine for agent systems.  
It provides durable memory storage, deterministic replay, and hybrid retrieval across:

- vector similarity (semantic search),
- graph relationships (connected memories),
- temporal constraints (time-aware recall).

Mnemos is built as a library and now includes a gRPC service layer for multi-agent usage from Python and other languages.

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
- gRPC service endpoint for remote clients.
- Python typed client package with high-level `insert_text` / `query_text` APIs.

## Project Layout

```text
src/
  core/
    memory_entry.rs      # Memory model
    command.rs           # Command/event types
    state_machine.rs     # Deterministic state transitions
  storage/
    wal.rs               # Write-ahead log
    segment.rs           # Segment files + in-memory segment index
    compaction.rs        # Segment compaction
  index/
    vector.rs            # Cosine similarity index
    graph.rs             # Graph traversal index
    temporal.rs          # Time-based index
    combined.rs          # Hybrid index composition
  query/
    planner.rs           # Query planning heuristics
    executor.rs          # Plan execution + metrics/traces
    hybrid.rs            # Text-oriented hybrid query engine
  engine.rs              # Core orchestrator (WAL + segments + state)
  store.rs               # Unified facade API (MnemosStore)
  service/grpc.rs        # gRPC service implementation
  bin/
    mnemos_grpc.rs       # gRPC server binary
    manual_store.rs      # local manual demo binary
proto/
  mnemos.proto           # gRPC contract
python/
  mnemos-client/         # typed Python client package
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

### WAL (`src/storage/wal.rs`)

Record format:

```text
[u32: len][u32: checksum][bytes: bincode(Command)]
```

- Detects corruption via checksum.
- Used for crash recovery and deterministic replay.

### Segment Storage (`src/storage/segment.rs`)

Record format:

```text
[u32: len][u32: checksum][bytes: bincode(MemoryEntry)]
```

- Append-only `.seg` files with rotation.
- In-memory index: `MemoryId -> (segment_id, offset, length)`.
- Supports logical delete and compaction eligibility analysis.

### Compaction (`src/storage/compaction.rs`)

- Rewrites live entries into a fresh segment directory.
- Performs atomic swap (`old -> backup`, `new -> old`, delete backup).
- Rebuilds segment storage index after swap.

## Querying

### High-level Rust facade (`src/store.rs`)

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

## gRPC Service

Binary: `src/bin/mnemos_grpc.rs`

Endpoints include:

- insert/delete memory,
- add/remove edge,
- query,
- enforce capacity,
- compact segments,
- stats.

Environment variables:

- `MNEMOS_GRPC_ADDR` (default `127.0.0.1:50051`)
- `MNEMOS_STATUS_ADDR` (default `127.0.0.1:50052`)
- `MNEMOS_DATA_DIR` (default `/tmp/mnemos_grpc`)
- `MNEMOS_VECTOR_DIM` (default `3`)
- `MNEMOS_SYNC_POLICY` (`strict` | `batch` | `async`, default `strict`)
- `MNEMOS_SYNC_BATCH_MAX_OPS` (default `64`)
- `MNEMOS_SYNC_BATCH_MAX_DELAY_MS` (default `25`)
- `MNEMOS_SYNC_ASYNC_INTERVAL_MS` (default `25`)
- `MNEMOS_CHECKPOINT_ENABLED` (`true`/`false`, default `false`)
- `MNEMOS_CHECKPOINT_EVERY_OPS` (default `10000`)
- `MNEMOS_CHECKPOINT_EVERY_MS` (default `30000`)
- `MNEMOS_AUTH_MODE` (`none` | `api_key`, default `none`)
- `MNEMOS_API_KEY` (required when `MNEMOS_AUTH_MODE=api_key`)
- `MNEMOS_INTENT_ANCHOR_SEMANTIC` (default semantic anchor text)
- `MNEMOS_INTENT_ANCHOR_RECENCY` (default recency anchor text)
- `MNEMOS_INTENT_ANCHOR_GRAPH` (default graph anchor text)
- `MNEMOS_INTENT_GRAPH_HOPS_2_THRESHOLD` (default `0.55`)
- `MNEMOS_INTENT_GRAPH_HOPS_3_THRESHOLD` (default `0.80`)
- `MNEMOS_INTENT_IMPORTANCE_PCT` (default `20`)

Run server:

```bash
scripts/run_grpc.sh
```

Status check in browser:

- [http://127.0.0.1:50052](http://127.0.0.1:50052)

### Intent Tuning Guide

Mnemos uses intent anchors to auto-balance retrieval between similarity, recency, and graph expansion.
Tune this behavior with env vars depending on your goal.

Preset: Hackathon (higher recall, more relationship exploration)

```bash
MNEMOS_INTENT_GRAPH_HOPS_2_THRESHOLD=0.45
MNEMOS_INTENT_GRAPH_HOPS_3_THRESHOLD=0.70
MNEMOS_INTENT_IMPORTANCE_PCT=15
```

Preset: Production-Latency (tighter graph expansion, steadier latency)

```bash
MNEMOS_INTENT_GRAPH_HOPS_2_THRESHOLD=0.65
MNEMOS_INTENT_GRAPH_HOPS_3_THRESHOLD=0.88
MNEMOS_INTENT_IMPORTANCE_PCT=25
```

Quick guidance:

- Lower graph thresholds => more frequent 2-3 hop expansion => better recall, higher latency.
- Higher graph thresholds => graph is used more conservatively => lower tail latency.
- Lower importance percent => similarity/recency dominate ranking.
- Higher importance percent => stable “important memory” ranking boost.

## Python Client

Package: `python/mnemos-client`

High-level developer API:

- `insert_text(...)`
- `query_text(...)`

Low-level API also available:

- `insert_memory(...)`
- `query(...)`

Typical setup:

```bash
cd python/mnemos-client
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
./scripts/generate_proto.sh
python examples/basic_usage.py
```

## Local Manual Demo

Run the Rust manual example:

```bash
cargo run --bin manual_store
```

This demonstrates:

- inserts,
- edge creation,
- query and score outputs.

## Sync Policy Benchmark

Quick local throughput comparison between durability modes:

```bash
cargo run --bin sync_bench -- --mode strict --ops 20000
cargo run --bin sync_bench -- --mode batch --ops 20000 --batch-max-ops 128 --batch-max-delay-ms 20
cargo run --bin sync_bench -- --mode async --ops 20000 --async-interval-ms 20
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
cargo test -- --nocapture
```

## Current Status

Mnemos is already usable for:

- single-node agent memory storage/retrieval,
- deterministic recovery,
- remote access via gRPC.

Areas still evolving toward full production multi-agent scale:

- stronger service hardening (authn/authz, richer observability),
- concurrency/snapshot strategy for high write contention,
- optional ANN backends for larger vector scale,
- operational tooling and deployment templates.
