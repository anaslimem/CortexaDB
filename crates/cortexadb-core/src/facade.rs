//! Embedded `CortexaDB` facade — simplified API for agent memory.
//!
//! This is the recommended entry point for using CortexaDB as a library.
//! It wraps [`CortexaDBStore`] and hides planner/engine/index details behind
//! five core operations: `open`, `remember`, `ask`, `connect`, `compact`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::memory_entry::{MemoryEntry, MemoryId};
use crate::core::state_machine::StateMachineError;
use crate::engine::{CapacityPolicy, SyncPolicy};
use crate::index::IndexMode;
use crate::query::hybrid::{QueryEmbedder, QueryOptions};
use crate::store::{CheckpointPolicy, CortexaDBStore, CortexaDBStoreError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Returned by [`CortexaDB::ask`] — a scored memory hit.
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
    pub namespace: String,
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

impl Default for CortexaDBConfig {
    fn default() -> Self {
        Self {
            vector_dimension: 3,
            sync_policy: SyncPolicy::Strict,
            checkpoint_policy: CheckpointPolicy::Periodic { every_ops: 1000, every_ms: 30_000 },
            capacity_policy: CapacityPolicy::new(None, None),
            index_mode: IndexMode::Exact,
        }
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
// Embedder adapter (used internally for `ask`)
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
/// use cortexadb_core::{CortexaDB, CortexaDBConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let db = CortexaDB::open("my_agent.db")?;
/// let id = db.remember(vec![1.0, 0.0, 0.0], None)?;
/// let hits = db.ask(vec![1.0, 0.0, 0.0], 5, None)?;
/// # Ok(())
/// # }
/// ```
pub struct CortexaDB {
    inner: CortexaDBStore,
    next_id: std::sync::atomic::AtomicU64,
}

impl CortexaDB {
    /// Open or create a CortexaDB database at the given path with default config.
    ///
    /// # Errors
    ///
    /// Returns a [`CortexaDBError`] if the directory cannot be created or the
    /// WAL/segments are corrupted and cannot be recovered.
    pub fn open(path: &str) -> Result<Self> {
        Self::open_with_config(path, CortexaDBConfig::default())
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
            )?
        } else {
            CortexaDBStore::new_with_policies(
                &wal_path,
                &segments_dir,
                config.vector_dimension,
                config.sync_policy,
                config.checkpoint_policy,
                config.capacity_policy,
            )?
        };

        // Determine next memory ID from existing state.
        let max_id = store.state_machine().all_memories().iter().map(|e| e.id.0).max().unwrap_or(0);

        Ok(Self { inner: store, next_id: std::sync::atomic::AtomicU64::new(max_id + 1) })
    }

    /// Store a new memory with the given embedding and optional metadata.
    ///
    /// The memory is placed in the "default" namespace.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cortexadb_core::CortexaDB;
    /// # use std::collections::HashMap;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = CortexaDB::open("test")?;
    /// let mut meta = HashMap::new();
    /// meta.insert("type".to_string(), "thought".to_string());
    /// let id = db.remember(vec![0.1, 0.2, 0.3], Some(meta))?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if the write-ahead log fails to append the entry.
    pub fn remember(
        &self,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<u64> {
        self.remember_in_namespace("default", embedding, metadata)
    }

    /// Store a new memory in a specific namespace.
    pub fn remember_in_namespace(
        &self,
        namespace: &str,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<u64> {
        let id = MemoryId(self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let mut entry =
            MemoryEntry::new(id, namespace.to_string(), Vec::new(), ts).with_embedding(embedding);
        if let Some(meta) = metadata {
            entry.metadata = meta;
        }

        self.inner.insert_memory(entry)?;
        Ok(id.0)
    }

    /// Store a memory with explicit content bytes optionally in a namespace.
    pub fn remember_with_content(
        &self,
        namespace: &str,
        content: Vec<u8>,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<u64> {
        let id = MemoryId(self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let mut entry =
            MemoryEntry::new(id, namespace.to_string(), content, ts).with_embedding(embedding);
        if let Some(meta) = metadata {
            entry.metadata = meta;
        }

        self.inner.insert_memory(entry)?;
        Ok(id.0)
    }

    /// Query the database for the top-k most relevant memories.
    ///
    /// The search uses cosine similarity on the vector embeddings and can optionally
    /// filter by metadata values.
    ///
    /// # Errors
    ///
    /// Returns [`CortexaDBError`] if the query execution fails.
    pub fn ask(
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

    /// Query the database scoped to a specific namespace.
    ///
    /// Over-fetches by 4× top_k globally, then filters by namespace and
    /// returns the top *top_k* results. This avoids a separate index per
    /// namespace while keeping the filter inside Rust (no GIL round-trips).
    pub fn ask_in_namespace(
        &self,
        namespace: &str,
        query_embedding: Vec<f32>,
        top_k: usize,
        metadata_filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<Hit>> {
        let embedder = StaticEmbedder { embedding: query_embedding };
        let mut options = QueryOptions::with_top_k(top_k);
        options.namespace = Some(namespace.to_string());
        options.metadata_filter = metadata_filter;
        let execution = self.inner.query("", options, &embedder)?;

        let sm = self.inner.state_machine();
        let memories = sm.all_memories();

        // Build a lookup: MemoryId → namespace
        let ns_map: std::collections::HashMap<u64, &str> =
            memories.iter().map(|m| (m.id.0, m.namespace.as_str())).collect();

        let results: Vec<Hit> = execution
            .hits
            .into_iter()
            .filter(|hit| ns_map.get(&hit.id.0).copied() == Some(namespace))
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
            namespace: entry.namespace.clone(),
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
    pub fn delete_memory(&self, id: u64) -> Result<()> {
        self.inner.delete_memory(MemoryId(id))?;
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
        self.inner.add_edge(MemoryId(from), MemoryId(to), relation.to_string())?;
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
    pub fn stats(&self) -> Stats {
        Stats {
            entries: self.inner.state_machine().len(),
            indexed_embeddings: self.inner.indexed_embeddings(),
            wal_length: self.inner.wal_len(),
            vector_dimension: self.inner.vector_dimension(),
            storage_version: 1,
        }
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
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_open_remember_ask() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();

        let id1 = db.remember(vec![1.0, 0.0, 0.0], None).unwrap();
        let id2 = db.remember(vec![0.0, 1.0, 0.0], None).unwrap();
        assert_ne!(id1, id2);

        let hits = db.ask(vec![1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, id1);
    }

    #[test]
    fn test_connect_and_stats() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();

        let id1 = db.remember(vec![1.0, 0.0, 0.0], None).unwrap();
        let id2 = db.remember(vec![0.0, 1.0, 0.0], None).unwrap();
        db.connect(id1, id2, "related").unwrap();

        let stats = db.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.indexed_embeddings, 2);
        assert_eq!(stats.vector_dimension, 3);
    }

    #[test]
    fn test_open_recover() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");

        {
            let db = CortexaDB::open(path.to_str().unwrap()).unwrap();
            db.remember(vec![1.0, 0.0, 0.0], None).unwrap();
            db.remember(vec![0.0, 1.0, 0.0], None).unwrap();
        }

        // Reopen — should recover from WAL.
        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();
        let stats = db.stats();
        assert_eq!(stats.entries, 2);

        let hits = db.ask(vec![1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_checkpoint_and_recover() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");

        {
            let db = CortexaDB::open(path.to_str().unwrap()).unwrap();
            db.remember(vec![1.0, 0.0, 0.0], None).unwrap();
            db.remember(vec![0.0, 1.0, 0.0], None).unwrap();
            db.checkpoint().unwrap();
            // Write more after checkpoint.
            db.remember(vec![0.0, 0.0, 1.0], None).unwrap();
        }

        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();
        let stats = db.stats();
        assert_eq!(stats.entries, 3);
    }

    #[test]
    fn test_compact() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();

        db.remember(vec![1.0, 0.0, 0.0], None).unwrap();
        db.compact().unwrap();
    }

    #[test]
    fn test_remember_with_metadata() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        let id = db.remember(vec![1.0, 0.0, 0.0], Some(meta)).unwrap();

        let hits = db.ask(vec![1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].id, id);

        let memory = db.get_memory(id).unwrap();
        assert_eq!(memory.metadata.get("source").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn test_namespace_support() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("testdb");
        let db = CortexaDB::open(path.to_str().unwrap()).unwrap();

        let id1 = db.remember_in_namespace("agent_b", vec![0.0, 1.0, 0.0], None).unwrap();
        let _id2 = db.remember_in_namespace("agent_c", vec![0.0, 0.0, 1.0], None).unwrap();

        let stats = db.stats();
        assert_eq!(stats.entries, 2);

        let m1 = db.get_memory(id1).unwrap();
        assert_eq!(m1.namespace, "agent_b");
    }
}
