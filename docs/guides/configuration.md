# Configuration

This guide covers all configuration options available when opening a CortexaDB database.

## Opening a Database

```python
db = CortexaDB.open(
    path,                    # Database directory path
    dimension=None,          # Vector dimension (required if no embedder)
    embedder=None,           # Embedding provider (auto-sets dimension)
    sync="strict",           # Sync policy
    max_entries=None,        # Max memory count before eviction
    max_bytes=None,          # Max storage bytes before eviction
    index_mode="exact",      # Vector index mode
    record=None,             # Path to replay log file
)
```

---

## Sync Policies

The sync policy controls how writes are persisted to disk. This is the primary trade-off between durability and write throughput.

### Strict (Default)

```python
db = CortexaDB.open("db.mem", dimension=128, sync="strict")
```

- Calls `fsync()` after every write operation
- **Safest**: guaranteed durability for every write
- **Slowest**: fsync is an expensive system call
- **Use when**: data loss is unacceptable (financial data, critical agent state)

### Async

```python
db = CortexaDB.open("db.mem", dimension=128, sync="async")
```

- Background thread calls `fsync()` periodically (default: every 10ms)
- **Fastest**: writes return immediately
- **Risk**: up to 10ms of writes can be lost on crash
- **Use when**: maximum throughput is needed and occasional data loss is acceptable

### Batch

```python
db = CortexaDB.open("db.mem", dimension=128, sync="batch")
```

- Groups writes and fsyncs in batches (default: 64 ops or 50ms, whichever comes first)
- **Balanced**: good throughput with bounded data loss window
- **Use when**: you want a middle ground between strict and async

---

## Index Mode

Controls how vector similarity search is performed. See the [Indexing guide](./indexing.md) for details.

### Exact (Default)

```python
db = CortexaDB.open("db.mem", dimension=128, index_mode="exact")
```

Brute-force cosine similarity. 100% recall, O(n) query time.

### HNSW

```python
# Default HNSW parameters
db = CortexaDB.open("db.mem", dimension=128, index_mode="hnsw")

# Custom HNSW parameters
db = CortexaDB.open("db.mem", dimension=128, index_mode={
    "type": "hnsw",
    "m": 16,
    "ef_search": 50,
    "ef_construction": 200,
    "metric": "cos"    # or "l2"
})
```

Approximate nearest neighbor. ~95% recall, O(log n) query time.

---

## Capacity Management

Set limits on database size. When a limit is exceeded, CortexaDB automatically evicts the oldest and least important memories.

### Max Entries

```python
db = CortexaDB.open("db.mem", dimension=128, max_entries=10000)
```

Limits the total number of memory entries. When exceeded, the least important / oldest entries are evicted.

### Max Bytes

```python
db = CortexaDB.open("db.mem", dimension=128, max_bytes=100_000_000)  # 100MB
```

Limits total storage size in bytes. Eviction works the same as `max_entries`.

### Eviction Strategy

Eviction is deterministic and follows this priority:
1. Sort entries by `(importance ASC, created_at ASC)`
2. Evict from the bottom until under the limit
3. Evictions are logged to the WAL for crash recovery

---

## Recording

Enable operation recording for replay and debugging:

```python
db = CortexaDB.open("db.mem", dimension=128, record="session.log")
```

All write operations are appended to the specified log file in NDJSON format. See the [Replay guide](./replay.md) for details.

---

## Embedder

Pass an embedder to enable automatic text-to-vector conversion:

```python
from cortexadb.providers.openai import OpenAIEmbedder

db = CortexaDB.open("db.mem", embedder=OpenAIEmbedder())
```

When an embedder is set, the `dimension` parameter is automatically inferred from the embedder. See the [Embedders guide](./embedders.md) for available providers.

---

## Rust Configuration

In Rust, use the builder pattern for advanced configuration:

```rust
use cortexadb_core::{CortexaDB, CortexaDBConfig, SyncPolicy, IndexMode, HnswConfig};

let config = CortexaDBConfig {
    vector_dimension: 128,
    sync_policy: SyncPolicy::Batch {
        max_ops: 64,
        max_delay_ms: 50,
    },
    checkpoint_policy: CheckpointPolicy::Periodic {
        every_ops: 1000,
        every_ms: 30_000,
    },
    capacity_policy: CapacityPolicy {
        max_entries: Some(10_000),
        max_bytes: None,
    },
    index_mode: IndexMode::Hnsw(HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
        metric: MetricKind::Cos,
    }),
};

let db = CortexaDB::builder("/path/to/db", config).build()?;
```

### Checkpoint Policy (Rust Only)

| Policy | Description |
|--------|-------------|
| `Disabled` | No automatic checkpoints (default in Python) |
| `Periodic { every_ops, every_ms }` | Checkpoint after N ops or M milliseconds |

---

## Configuration Summary

| Option | Default | Description |
|--------|---------|-------------|
| `path` | Required | Database directory path |
| `dimension` | None | Vector dimension (required if no embedder) |
| `embedder` | None | Embedding provider |
| `sync` | `"strict"` | Sync policy: `"strict"`, `"async"`, `"batch"` |
| `max_entries` | None | Max memory count |
| `max_bytes` | None | Max storage bytes |
| `index_mode` | `"exact"` | `"exact"`, `"hnsw"`, or HNSW config dict |
| `record` | None | Path to replay log file |

---

## Next Steps

- [Indexing](./indexing.md) - HNSW parameter tuning
- [Storage Engine](./storage-engine.md) - How sync policies affect durability
- [Replay](./replay.md) - Using the record option
