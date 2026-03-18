//! Embedded `CortexaDB` facade — simplified API for agent memory.
//!
//! This is the recommended entry point for using CortexaDB as a library.
//! It wraps [`CortexaDBStore`] and hides planner/engine/index details behind
//! five core operations: `open`, `add`, `search`, `connect`, `compact`.

use std::{
    collections::HashMap,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    core::{
        memory_entry::{MemoryEntry, MemoryId},
        state_machine::StateMachineError,
    },
    engine::{CapacityPolicy, SyncPolicy},
    index::IndexMode,
    query::hybrid::{QueryEmbedder, QueryOptions},
    store::{CheckpointPolicy, CortexaDBStore, CortexaDBStoreError},
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Returned by [`CortexaDB::search`] — a scored memory hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: u64,
    pub score: f32,
}

/// A full memory entry retrieved by ID.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: u64,
    pub content: Vec<u8>,
    pub collection: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
    pub importance: f32,
}

/// Database statistics.
#[derive(Debug, Clone)]
pub struct Stats {
    pub entries: usize,
    pub indexed_embeddings: usize,
    pub wal_length: u64,
    pub vector_dimension: usize,
    pub storage_version: u32,
}

/// Configuration for opening a CortexaDB database.
#[derive(Debug, Clone)]
pub struct CortexaDBConfig {
    pub vector_dimension: usize,
    pub sync_policy: SyncPolicy,
    pub checkpoint_policy: CheckpointPolicy,
    pub capacity_policy: CapacityPolicy,
    pub index_mode: IndexMode,
}

// We deliberately do not implement `Default` for `CortexaDBConfig` because
// `vector_dimension` is a required parameter that should not be implicitly guessed.

/// A builder for constructing and opening a `CortexaDB` instance.
pub struct CortexaDBBuilder {
    path: PathBuf,
    config: CortexaDBConfig,
}

impl CortexaDBBuilder {
    /// Create a new builder with the required path and expected vector dimension.
    pub fn new<P: Into<PathBuf>>(path: P, vector_dimension: usize) -> Self {
        Self {
            path: path.into(),
            config: CortexaDBConfig {
                vector_dimension,
                sync_policy: SyncPolicy::Strict,
                checkpoint_policy: CheckpointPolicy::Periodic { every_ops: 1000, every_ms: 30_000 },
                capacity_policy: CapacityPolicy::new(None, None),
                index_mode: IndexMode::Exact,
            },
        }
    }

    /// Set the synchronisation policy.
    pub fn with_sync_policy(mut self, sync_policy: SyncPolicy) -> Self {
        self.config.sync_policy = sync_policy;
        self
    }

    /// Set the checkpointing policy.
    pub fn with_checkpoint_policy(mut self, checkpoint_policy: CheckpointPolicy) -> Self {
        self.config.checkpoint_policy = checkpoint_policy;
        self
    }

    /// Set the capacity policy.
    pub fn with_capacity_policy(mut self, capacity_policy: CapacityPolicy) -> Self {
        self.config.capacity_policy = capacity_policy;
        self
    }

    /// Set the index mode (e.g. Exact vs Approximate).
    pub fn with_index_mode(mut self, index_mode: IndexMode) -> Self {
        self.config.index_mode = index_mode;
        self
    }

    /// Build and open the `CortexaDB` instance.
    pub fn build(self) -> Result<CortexaDB> {
        let path_str = self.path.to_str().ok_or_else(|| {
            CortexaDBError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Database path is not valid UTF-8",
            ))
        })?;
        CortexaDB::open_with_config(path_str, self.config)
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from the CortexaDB facade.
#[derive(Debug, thiserror::Error)]
pub enum CortexaDBError {
    #[error("Store error: {0}")]
    Store(#[from] CortexaDBStoreError),
    #[error("State machine error: {0}")]
    StateMachine(#[from] StateMachineError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Memory not found: {0}")]
    MemoryNotFound(u64),
}

pub type Result<T> = std::result::Result<T, CortexaDBError>;

// ---------------------------------------------------------------------------
// Embedder adapter (used internally for `search`)
// ---------------------------------------------------------------------------

struct StaticEmbedder {
    embedding: Vec<f32>,
}

impl QueryEmbedder for StaticEmbedder {
    fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
        Ok(self.embedding.clone())
    }
}

// ---------------------------------------------------------------------------
// CortexaDB facade
// ---------------------------------------------------------------------------

/// Embedded, file-backed agent memory database.
///
/// `CortexaDB` provides a high-level API for storing and retrieving memories
/// with vector embeddings, graph relationships, and metadata filtering.
///
/// # Examples
///
/// ```rust,no_run
/// use cortexadb_core::CortexaDB;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a default DB with vector dimension 3
/// let db = CortexaDB::open("my_agent.db", 3)?;
///
/// // Or use the builder for advanced config
/// let db_advanced = CortexaDB::builder("advanced.db", 1536)
///     .with_sync_policy(cortexadb_core::engine::SyncPolicy::Async { interval_ms: 1000 })
///     .build()?;
///
/// let id = db.add(vec![1.0, 0.0, 0.0], None)?;
/// let hits = db.search(vec![1.0, 0.0, 0.0], 5, None)?;
/// # Ok(())
/// # }
/// ```
pub struct CortexaDB {
    inner: CortexaDBStore,
    next_id: std::sync::atomic::AtomicU64,
}

/// A record for batch insertion.
#[derive(Debug, Clone)]
pub struct BatchRecord {
    pub collection: String,
    pub content: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: Option<HashMap<String, String>>,
}

impl CortexaDB {
    /// Open a CortexaDB database at the given path with a required vector dimension,
    /// using standard safe defaults.
    ///
    /// For advanced configuration (e.g., sync policy, index mode), use [`CortexaDB::builder`].
    ///
    /// # Errors
    ///
    /// Returns a [`CortexaDBError`] if the directory cannot be created or the
    /// WAL/segments are corrupted and cannot be recovered.
    pub fn open(path: &str, vector_dimension: usize) -> Result<Self> {
        Self::builder(path, vector_dimension).build()
    }

    /// Create a [`CortexaDBBuilder`] to configure advanced database options.
    pub fn builder(path: &str, vector_dimension: usize) -> CortexaDBBuilder {
        CortexaDBBuilder::new(path, vector_dimension)
    }

    /// Open or create a CortexaDB database with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`CortexaDBError`] if initialization fails.
    pub fn open_with_config(path: &str, config: CortexaDBConfig) -> Result<Self> {
        let base = PathBuf::from(path);
        std::fs::create_dir_all(&base)?;

        let wal_path = base.join("cortexadb.wal");
        let segments_dir = base.join("segments");

        let store = if wal_path.exists() {
            CortexaDBStore::recover_with_policies(
                &wal_path,
                &segments_dir,
                config.vector_dimension,
                config.sync_policy,
                config.checkpoint_policy,
                config.capacity_policy,
                config.index_mode.clone(),
            )?
        } else {
            CortexaDBStore::new_with_policies(
                &wal_path,
                &segments_dir,
                config.vector_dimension,
                config.sync_policy,
                config.checkpoint_policy,
                config.capacity_policy,
                config.index_mode.clone(),
            )?
        };

        // Determine next memory ID from existing state.
        let max_id = store.state_machine().all_memories().iter().map(|e| e.id.0).max().unwrap_or(0);

        Ok(Self { inner: store, next_id: std::sync::atomic::AtomicU64::new(max_id + 1) })
    }

    /// Store a new memory with the given embedding and optional metadata.
    ///
    /// The memory is placed in the "default" collection.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cortexadb_core::CortexaDB;
    /// # use std::collections::HashMap;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = CortexaDB::open("test", 3)?;
    /// let mut meta = HashMap::new();
    /// meta.insert("type".to_string(), "thought".to_string());
    /// let id = db.add(vec![0.1, 0.2, 0.3], Some(meta))?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if the write-ahead log fails to append the entry.
    pub fn add(
        &self,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<u64> {
        self.add_in_collection("default", embedding, metadata)
    }

    /// Store a new memory in a specific collection.
    pub fn add_in_collection(
        &self,
        collection: &str,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<u64> {
        let id = MemoryId(self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let mut entry =
            MemoryEntry::new(id, collection.to_string(), Vec::new(), ts).with_embedding(embedding);
        if let Some(meta) = metadata {
            entry.metadata = meta;
        }

        self.inner.add(entry)?;
        Ok(id.0)
    }

    /// Store a memory with explicit content bytes optionally in a collection.
    pub fn add_with_content(
        &self,
        collection: &str,
        content: Vec<u8>,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<u64> {
        let id = MemoryId(self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let mut entry =
            MemoryEntry::new(id, collection.to_string(), content, ts).with_embedding(embedding);
        if let Some(meta) = metadata {
            entry.metadata = meta;
        }

        self.inner.add(entry)?;
        Ok(id.0)
    }

    /// Store a batch of memories efficiently.
    pub fn add_batch(&self, records: Vec<BatchRecord>) -> Result<Vec<u64>> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let mut entries = Vec::with_capacity(records.len());
        let mut ids = Vec::with_capacity(records.len());

        for rec in records {
            let id = MemoryId(self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            let mut entry = MemoryEntry::new(id, rec.collection, rec.content, ts);
            if let Some(emb) = rec.embedding {
                entry = entry.with_embedding(emb);
            }
            if let Some(meta) = rec.metadata {
                entry.metadata = meta;
            }
            ids.push(id.0);
            entries.push(entry);
        }

        self.inner.add_batch(entries)?;
        Ok(ids)
    }

    /// Query the database for the top-k most relevant memories.
    ///
    /// The search uses cosine similarity on the vector embeddings and can optionally
    /// filter by metadata values.
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if the query execution fails.
    pub fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        metadata_filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<Hit>> {
        let embedder = StaticEmbedder { embedding: query_embedding };
        let mut options = QueryOptions::with_top_k(top_k);
        options.metadata_filter = metadata_filter;
        let execution = self.inner.query("", options, &embedder)?;

        let mut results = Vec::with_capacity(execution.hits.len());
        for hit in execution.hits {
            results.push(Hit { id: hit.id.0, score: hit.final_score });
        }
        Ok(results)
    }

    /// Retrieve the outgoing graph connections from a specific memory.
    ///
    /// Returns a list of `(target_id, relation_label)` tuples.
    pub fn get_neighbors(&self, id: u64) -> Result<Vec<(u64, String)>> {
        let neighbors = self.inner.state_machine().get_neighbors(MemoryId(id))?;
        Ok(neighbors.into_iter().map(|(target_id, relation)| (target_id.0, relation)).collect())
    }

    /// Query the database scoped to a specific collection.
    ///
    /// Over-fetches by 4× top_k globally, then filters by collection and
    /// returns the top *top_k* results. This avoids a separate index per
    /// collection while keeping the filter inside Rust (no GIL round-trips).
    pub fn search_in_collection(
        &self,
        collection: &str,
        query_embedding: Vec<f32>,
        top_k: usize,
        metadata_filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<Hit>> {
        let embedder = StaticEmbedder { embedding: query_embedding };
        let mut options = QueryOptions::with_top_k(top_k.saturating_mul(4).max(top_k));
        options.collection = Some(collection.to_string());
        options.metadata_filter = metadata_filter;
        let execution = self.inner.query("", options, &embedder)?;

        let sm = self.inner.state_machine();
        let memories = sm.all_memories();

        // Build a lookup: MemoryId → collection
        let col_map: std::collections::HashMap<u64, &str> =
            memories.iter().map(|m| (m.id.0, m.collection.as_str())).collect();

        let results: Vec<Hit> = execution
            .hits
            .into_iter()
            .filter(|hit| col_map.get(&hit.id.0).copied() == Some(collection))
            .take(top_k)
            .map(|hit| Hit { id: hit.id.0, score: hit.final_score })
            .collect();

        Ok(results)
    }

    /// Retrieve a full memory by its identifier.
    pub fn get_memory(&self, id: u64) -> Result<Memory> {
        let snapshot = self.inner.snapshot();
        let entry = snapshot
            .state_machine()
            .get_memory(MemoryId(id))
            .map_err(|_e| CortexaDBError::MemoryNotFound(id))?;

        Ok(Memory {
            id: entry.id.0,
            content: entry.content.clone(),
            collection: entry.collection.clone(),
            embedding: entry.embedding.clone(),
            metadata: entry.metadata.clone(),
            created_at: entry.created_at,
            importance: entry.importance,
        })
    }

    /// Delete a memory by its identifier.
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if the deletion cannot be logged.
    pub fn delete(&self, id: u64) -> Result<()> {
        self.inner.delete(MemoryId(id))?;
        Ok(())
    }

    /// Create an edge (relationship) between two memories.
    ///
    /// Relationships are directed and labeled.
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if either memory ID does not exist or the
    /// write-ahead log fails.
    pub fn connect(&self, from: u64, to: u64, relation: &str) -> Result<()> {
        self.inner.connect(MemoryId(from), MemoryId(to), relation.to_string())?;
        Ok(())
    }

    /// Compact on-disk segment storage (removes tombstoned entries).
    ///
    /// This is a maintenance operation that reclaims space on disk. It is
    /// safe to run while the database is in use.
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if IO fails during compaction.
    pub fn compact(&self) -> Result<()> {
        self.inner.compact_segments()?;
        Ok(())
    }

    /// Flush all pending WAL writes to disk.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }

    /// Force a checkpoint now (snapshot state + truncate WAL).
    ///
    /// Checkpoints accelerate future startups by creating a snapshot of the
    /// current state and clearing the write-ahead log.
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if IO fails during checkpointing.
    pub fn checkpoint(&self) -> Result<()> {
        self.inner.checkpoint_now()?;
        Ok(())
    }

    /// Get database statistics.
    pub fn stats(&self) -> Result<Stats> {
        Ok(Stats {
            entries: self.inner.state_machine().len(),
            indexed_embeddings: self.inner.indexed_embeddings(),
            wal_length: self.inner.wal_len()?,
            vector_dimension: self.inner.vector_dimension(),
            storage_version: 1,
        })
    }

    /// Access the underlying `CortexaDBStore` for advanced operations.
    pub fn store(&self) -> &CortexaDBStore {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_open_add_search() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id1 = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        let id2 = db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        assert_ne!(id1, id2);

        let hits = db.search(vec![1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, id1);
    }

    #[test]
    fn test_connect_and_stats() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id1 = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        let id2 = db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        db.connect(id1, id2, "related").unwrap();

        let stats = db.stats().unwrap();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.indexed_embeddings, 2);
        assert_eq!(stats.vector_dimension, 3);
    }

    #[test]
    fn test_open_recover() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");

        {
            let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();
            db.add(vec![1.0, 0.0, 0.0], None).unwrap();
            db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        }

        // Reopen — should recover from WAL.
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();
        let stats = db.stats();
        assert_eq!(stats.unwrap().entries, 2);

        let hits = db.search(vec![1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_checkpoint_and_recover() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");

        let mut config = CortexaDBConfig {
            vector_dimension: 3,
            sync_policy: crate::engine::SyncPolicy::Strict,
            checkpoint_policy: crate::store::CheckpointPolicy::Disabled,
            capacity_policy: crate::engine::CapacityPolicy::new(None, None),
            index_mode: crate::index::IndexMode::Exact,
        };
        config.checkpoint_policy = crate::store::CheckpointPolicy::Disabled;

        {
            let db = CortexaDB::open_with_config(path.to_str().unwrap(), config.clone()).unwrap();
            db.add(vec![1.0, 0.0, 0.0], None).unwrap();
            db.add(vec![0.0, 1.0, 0.0], None).unwrap();
            db.flush().unwrap(); // ensure WAL is synced before checkpoint truncates it
            db.checkpoint().unwrap();
            // Write more after checkpoint.
            db.add(vec![0.0, 0.0, 1.0], None).unwrap();
        }

        let db = CortexaDB::open_with_config(path.to_str().unwrap(), config).unwrap();
        let stats = db.stats().unwrap();
        assert_eq!(stats.entries, 3);
    }

    #[test]
    fn test_compact() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.compact().unwrap();
    }

    #[test]
    fn test_add_with_metadata() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        let id = db.add(vec![1.0, 0.0, 0.0], Some(meta)).unwrap();

        let hits = db.search(vec![1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].id, id);

        let memory = db.get_memory(id).unwrap();
        assert_eq!(memory.metadata.get("source").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn test_collection_support() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id1 = db.add_in_collection("agent_b", vec![0.0, 1.0, 0.0], None).unwrap();
        let _id2 = db.add_in_collection("agent_c", vec![0.0, 0.0, 1.0], None).unwrap();

        let stats = db.stats();
        assert_eq!(stats.unwrap().entries, 2);

        let m1 = db.get_memory(id1).unwrap();
        assert_eq!(m1.collection, "agent_b");
    }

    #[test]
    fn test_delete_removes_from_stats() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        assert_eq!(db.stats().unwrap().entries, 1);

        db.delete(id).unwrap();
        assert_eq!(db.stats().unwrap().entries, 0, "entry count should be 0 after delete");
    }

    #[test]
    fn test_search_not_returned_in_search() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        // Keep a second entry so the index is non-empty after deletion.
        // (search() returns NoEmbeddings when the vector index is completely empty.)
        let _id_keep = db.add(vec![0.0, 1.0, 0.0], None).unwrap();

        db.delete(id).unwrap();

        let hits = db.search(vec![1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(
            hits.iter().all(|h| h.id != id),
            "deleted memory must not appear in search results"
        );
    }

    #[test]
    fn test_get_memory_after_delete_returns_error() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.delete(id).unwrap();

        let result = db.get_memory(id);
        assert!(result.is_err(), "get_memory on a deleted ID must return an error");
    }

    #[test]
    fn test_get_neighbors_returns_correct_edges() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id1 = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        let id2 = db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        let id3 = db.add(vec![0.0, 0.0, 1.0], None).unwrap();

        db.connect(id1, id2, "related").unwrap();
        db.connect(id1, id3, "follows").unwrap();

        let neighbors = db.get_neighbors(id1).unwrap();
        assert_eq!(neighbors.len(), 2, "id1 should have 2 outgoing edges");

        let target_ids: Vec<u64> = neighbors.iter().map(|(t, _)| *t).collect();
        assert!(target_ids.contains(&id2), "id2 must be a neighbor of id1");
        assert!(target_ids.contains(&id3), "id3 must be a neighbor of id1");

        let relations: Vec<&str> = neighbors.iter().map(|(_, r)| r.as_str()).collect();
        assert!(relations.contains(&"related"));
        assert!(relations.contains(&"follows"));
    }

    #[test]
    fn test_get_neighbors_no_edges_returns_empty() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        let id = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        let neighbors = db.get_neighbors(id).unwrap();
        assert!(neighbors.is_empty(), "node with no edges should return empty neighbors");
    }

    #[test]
    fn test_search_in_collection_only_returns_own_collection() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        // Same embedding direction — only collection should differentiate results.
        let id_a = db.add_in_collection("ns_a", vec![1.0, 0.0, 0.0], None).unwrap();
        let _id_b = db.add_in_collection("ns_b", vec![1.0, 0.0, 0.0], None).unwrap();

        let hits = db.search_in_collection("ns_a", vec![1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(!hits.is_empty(), "should find memories in ns_a");
        assert!(
            hits.iter().all(|h| h.id == id_a),
            "all hits must belong to ns_a, got: {:?}",
            hits.iter().map(|h| h.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_flush_completes_without_error() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();
        db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.flush().expect("flush must not fail");
    }

    #[test]
    fn test_compact_completes_without_error() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();
        db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.compact().expect("compact must not fail");
    }

    // ----- search_in_collection: sparse collection over-fetch regression -----

    #[test]
    fn test_search_in_collection_finds_entry_in_sparse_collection() {
        // Regression: before the 4× fix, search_in_collection returned empty results when the
        // target collection had far fewer entries than top_k * candidate_multiplier entries globally.
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();

        // Insert 10 entries in ns_majority to fill the global index.
        for i in 0..10u32 {
            let v = vec![i as f32 / 10.0, 1.0 - i as f32 / 10.0, 0.0];
            db.add_in_collection("ns_majority", v, None).unwrap();
        }
        // Insert 2 entries in ns_sparse.
        let id_a = db.add_in_collection("ns_sparse", vec![1.0, 0.0, 0.0], None).unwrap();
        let id_b = db.add_in_collection("ns_sparse", vec![0.9, 0.1, 0.0], None).unwrap();

        // Search for top-2 in ns_sparse — both must be returned.
        let hits = db.search_in_collection("ns_sparse", vec![1.0, 0.0, 0.0], 2, None).unwrap();
        let hit_ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        assert!(
            hit_ids.contains(&id_a),
            "id_a must appear in ns_sparse results; got {:?}",
            hit_ids
        );
        assert!(
            hit_ids.contains(&id_b),
            "id_b must appear in ns_sparse results; got {:?}",
            hit_ids
        );
    }

    // ----- Intent anchors end-to-end -----

    #[test]
    fn test_search_without_intent_anchors_unchanged() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap(), 3).unwrap();
        db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        // Default QueryOptions has intent_anchors = None; must produce same results as search().
        let hits = db.search(vec![1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!hits.is_empty());
    }
}
