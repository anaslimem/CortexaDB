# Rust API Reference

Reference for the `cortexadb-core` Rust crate.

## CortexaDB (Facade)

The high-level API for interacting with the database.

### Opening a Database

```rust
use cortexadb_core::CortexaDB;

// Simple open with default config
let db = CortexaDB::open("/path/to/db", 128)?;

// Builder pattern for advanced config
let db = CortexaDB::builder("/path/to/db", config).build()?;
```

---

### Memory Operations

#### `add(embedding, metadata) -> Result<u64>`

Stores a memory in the default collection.

```rust
let id = db.add(vec![0.1; 128], None)?;
let id = db.add(vec![0.1; 128], Some(metadata_map))?;
```

#### `add_in_collection(collection, embedding, metadata) -> Result<u64>`

Stores a memory in a specific collection.

```rust
let id = db.add_in_collection("agent_a", vec![0.1; 128], None)?;
```

#### `add_with_content(collection, content, embedding, metadata) -> Result<u64>`

Stores a memory with raw content bytes.

```rust
let id = db.add_with_content(
    "default",
    b"Hello world".to_vec(),
    vec![0.1; 128],
    None,
)?;
```

#### `search(embedding, top_k, metadata_filter) -> Result<Vec<Hit>>`

Vector similarity search in the default collection.

```rust
let hits = db.search(vec![0.1; 128], 5, None)?;
for hit in &hits {
    println!("ID: {}, Score: {:.3}", hit.id, hit.score);
}
```

#### `search_in_collection(collection, embedding, top_k, filter) -> Result<Vec<Hit>>`

Collection-scoped search.

```rust
let hits = db.search_in_collection("agent_a", vec![0.1; 128], 5, None)?;
```

#### `get_memory(id) -> Result<Memory>`

Retrieves a full memory entry by ID.

```rust
let mem = db.get_memory(42)?;
println!("{:?}", mem.metadata);
```

#### `delete(id) -> Result<()>`

Deletes a memory and updates all indexes.

```rust
db.delete(42)?;
```

---

### Graph Operations

#### `connect(from_id, to_id, relation) -> Result<()>`

Creates a directed edge between two memories.

```rust
db.connect(1, 2, "relates_to")?;
```

#### `get_neighbors(id) -> Result<Vec<(u64, String)>>`

Returns outgoing edges from a memory.

```rust
let neighbors = db.get_neighbors(1)?;
for (target_id, relation) in &neighbors {
    println!("→ {} ({})", target_id, relation);
}
```

---

### Maintenance

#### `compact() -> Result<()>`

Removes tombstoned entries from segment files.

#### `flush() -> Result<()>`

Forces fsync of all pending writes.

#### `checkpoint() -> Result<()>`

Creates a state snapshot and truncates the WAL.

#### `stats() -> Result<Stats>`

Returns database statistics.

```rust
let stats = db.stats()?;
println!("Entries: {}", stats.entries);
println!("Indexed: {}", stats.indexed_embeddings);
```

---

## Types

### `Hit`

```rust
pub struct Hit {
    pub id: u64,
    pub score: f32,
}
```

### `Memory`

```rust
pub struct Memory {
    pub id: u64,
    pub content: Vec<u8>,
    pub collection: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
    pub importance: f32,
}
```

### `MemoryEntry`

Internal representation used by the storage engine.

```rust
pub struct MemoryEntry {
    pub id: MemoryId,
    pub collection: String,
    pub content: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
    pub importance: f32,
}
```

### `MemoryId`

```rust
pub struct MemoryId(pub u64);
```

### `Stats`

```rust
pub struct Stats {
    pub entries: usize,
    pub indexed_embeddings: usize,
    pub wal_length: u64,
    pub vector_dimension: usize,
    pub storage_version: u32,
}
```

---

## Configuration

### `CortexaDBConfig`

```rust
pub struct CortexaDBConfig {
    pub vector_dimension: usize,
    pub sync_policy: SyncPolicy,
    pub checkpoint_policy: CheckpointPolicy,
    pub capacity_policy: CapacityPolicy,
    pub index_mode: IndexMode,
}
```

### `SyncPolicy`

```rust
pub enum SyncPolicy {
    Strict,
    Batch { max_ops: usize, max_delay_ms: u64 },
    Async { interval_ms: u64 },
}
```

### `CheckpointPolicy`

```rust
pub enum CheckpointPolicy {
    Disabled,
    Periodic { every_ops: usize, every_ms: u64 },
}
```

### `CapacityPolicy`

```rust
pub struct CapacityPolicy {
    pub max_entries: Option<usize>,
    pub max_bytes: Option<u64>,
}
```

### `IndexMode`

```rust
pub enum IndexMode {
    Exact,
    Hnsw(HnswConfig),
}
```

### `HnswConfig`

```rust
pub struct HnswConfig {
    pub m: usize,              // default: 16
    pub ef_construction: usize, // default: 200
    pub ef_search: usize,      // default: 50
    pub metric: MetricKind,    // default: Cos
}
```

### `MetricKind`

```rust
pub enum MetricKind {
    Cos,  // Cosine similarity
    L2,   // Euclidean distance
}
```

---

## Errors

### `CortexaDBError`

```rust
pub enum CortexaDBError {
    Store(CortexaDBStoreError),
    StateMachine(StateMachineError),
    Io(std::io::Error),
    MemoryNotFound(u64),
}
```

### `CortexaDBStoreError`

```rust
pub enum CortexaDBStoreError {
    Engine(String),
    Vector(String),
    Query(String),
    Checkpoint(String),
    Wal(String),
    InvariantViolation(String),
    MissingEmbeddingOnContentChange,
}
```

---

## Chunking

### `chunk(text, strategy, chunk_size, overlap) -> Vec<ChunkResult>`

```rust
use cortexadb_core::chunker::{chunk, ChunkStrategy};

let results = chunk("Long text...", ChunkStrategy::Recursive, 512, 50);
for c in &results {
    println!("Chunk {}: {}", c.index, &c.text[..50]);
}
```

### `ChunkStrategy`

```rust
pub enum ChunkStrategy {
    Fixed,
    Recursive,
    Semantic,
    Markdown,
    Json,
}
```

### `ChunkResult`

```rust
pub struct ChunkResult {
    pub text: String,
    pub index: usize,
    pub metadata: Option<ChunkMetadata>,
}
```
