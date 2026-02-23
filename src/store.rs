use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::core::command::Command;
use crate::core::memory_entry::{MemoryEntry, MemoryId};
use crate::core::state_machine::StateMachine;
use crate::engine::{CapacityPolicy, Engine, EvictionReport, SyncPolicy};
use crate::index::IndexLayer;
use crate::query::{
    QueryEmbedder, QueryExecution, QueryExecutor, QueryOptions, QueryPlan, QueryPlanner, StageTrace,
};
use crate::storage::compaction::CompactionReport;
use crate::storage::wal::CommandId;

#[derive(Error, Debug)]
pub enum MnemosStoreError {
    #[error("Engine error: {0}")]
    Engine(#[from] crate::engine::EngineError),
    #[error("Vector index error: {0}")]
    Vector(#[from] crate::index::vector::VectorError),
    #[error("Query error: {0}")]
    Query(#[from] crate::query::HybridQueryError),
    #[error("Invariant violation: {0}")]
    InvariantViolation(String),
    #[error("Embedding required when content changes for memory id {0:?}")]
    MissingEmbeddingOnContentChange(MemoryId),
}

pub type Result<T> = std::result::Result<T, MnemosStoreError>;

#[derive(Clone)]
pub struct ReadSnapshot {
    state_machine: StateMachine,
    indexes: IndexLayer,
}

impl ReadSnapshot {
    fn new(state_machine: StateMachine, indexes: IndexLayer) -> Self {
        Self {
            state_machine,
            indexes,
        }
    }

    pub fn state_machine(&self) -> &StateMachine {
        &self.state_machine
    }

    pub fn indexes(&self) -> &IndexLayer {
        &self.indexes
    }
}

struct WriteState {
    engine: Engine,
    indexes: IndexLayer,
}

struct SyncRuntime {
    pending_ops: usize,
    dirty_since: Option<Instant>,
    shutdown: bool,
}

/// Library-facing facade for storing and querying agent memory.
///
/// Wraps:
/// - `Engine`: durability + state machine + capacity/compaction
/// - `IndexLayer`: vector/graph/temporal retrieval indexes
/// - `QueryPlanner` + `QueryExecutor`: predictable query execution
///
/// Concurrency model:
/// - single writer (`Mutex<WriteState>`) for deterministic write ordering
/// - snapshot reads (`Arc<ReadSnapshot>`) for isolated concurrent queries
pub struct MnemosStore {
    writer: Arc<Mutex<WriteState>>,
    snapshot: Arc<RwLock<Arc<ReadSnapshot>>>,
    sync_policy: SyncPolicy,
    sync_control: Arc<(Mutex<SyncRuntime>, Condvar)>,
    sync_thread: Option<JoinHandle<()>>,
}

impl MnemosStore {
    pub fn new<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
    ) -> Result<Self> {
        Self::new_with_policy(wal_path, segments_dir, vector_dimension, SyncPolicy::Strict)
    }

    pub fn new_with_policy<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
    ) -> Result<Self> {
        let engine = Engine::new(wal_path, segments_dir)?;
        Self::from_engine(engine, vector_dimension, sync_policy)
    }

    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
    ) -> Result<Self> {
        Self::recover_with_policy(wal_path, segments_dir, vector_dimension, SyncPolicy::Strict)
    }

    pub fn recover_with_policy<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
    ) -> Result<Self> {
        let engine = Engine::recover(wal_path, segments_dir)?;
        Self::from_engine(engine, vector_dimension, sync_policy)
    }

    fn from_engine(
        engine: Engine,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
    ) -> Result<Self> {
        let indexes = Self::build_vector_index(engine.get_state_machine(), vector_dimension)?;
        Self::assert_vector_index_in_sync_inner(engine.get_state_machine(), &indexes)?;

        let snapshot = Arc::new(ReadSnapshot::new(
            engine.get_state_machine().clone(),
            indexes.clone(),
        ));

        let writer = Arc::new(Mutex::new(WriteState { engine, indexes }));
        let snapshot = Arc::new(RwLock::new(snapshot));
        let sync_control = Arc::new((
            Mutex::new(SyncRuntime {
                pending_ops: 0,
                dirty_since: None,
                shutdown: false,
            }),
            Condvar::new(),
        ));

        let sync_thread = match sync_policy {
            SyncPolicy::Strict => None,
            SyncPolicy::Batch { .. } | SyncPolicy::Async { .. } => Some(Self::spawn_sync_thread(
                Arc::clone(&writer),
                Arc::clone(&sync_control),
                sync_policy,
            )),
        };

        Ok(Self {
            writer,
            snapshot,
            sync_policy,
            sync_control,
            sync_thread,
        })
    }

    pub fn snapshot(&self) -> Arc<ReadSnapshot> {
        self.snapshot
            .read()
            .expect("snapshot lock poisoned")
            .clone()
    }

    pub fn flush(&self) -> Result<()> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        writer.engine.flush()?;
        self.clear_pending_sync_state();
        Ok(())
    }

    fn spawn_sync_thread(
        writer: Arc<Mutex<WriteState>>,
        sync_control: Arc<(Mutex<SyncRuntime>, Condvar)>,
        policy: SyncPolicy,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let (lock, cvar) = &*sync_control;
            loop {
                let mut runtime = lock.lock().expect("sync runtime lock poisoned");
                if runtime.shutdown {
                    break;
                }

                match policy {
                    SyncPolicy::Strict => break,
                    SyncPolicy::Batch {
                        max_ops,
                        max_delay_ms,
                    } => {
                        let max_ops = max_ops.max(1);
                        let max_delay = Duration::from_millis(max_delay_ms.max(1));

                        if runtime.pending_ops < max_ops {
                            if let Some(dirty_since) = runtime.dirty_since {
                                let elapsed = dirty_since.elapsed();
                                if elapsed < max_delay {
                                    let timeout = max_delay - elapsed;
                                    let (guard, _) = cvar
                                        .wait_timeout(runtime, timeout)
                                        .expect("sync runtime wait poisoned");
                                    runtime = guard;
                                }
                            } else {
                                runtime = cvar.wait(runtime).expect("sync runtime wait poisoned");
                            }
                        }
                    }
                    SyncPolicy::Async { interval_ms } => {
                        let wait = Duration::from_millis(interval_ms.max(1));
                        let (guard, _) = cvar
                            .wait_timeout(runtime, wait)
                            .expect("sync runtime wait poisoned");
                        runtime = guard;
                    }
                }

                if runtime.shutdown {
                    break;
                }

                let should_flush = runtime.pending_ops > 0;
                if should_flush {
                    runtime.pending_ops = 0;
                    runtime.dirty_since = None;
                }
                drop(runtime);

                if should_flush {
                    let mut write_state = writer.lock().expect("writer lock poisoned");
                    if let Err(err) = write_state.engine.flush() {
                        eprintln!("mnemos sync manager flush error: {err}");
                    }
                }
            }
        })
    }

    pub fn insert_memory(&self, entry: MemoryEntry) -> Result<CommandId> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");

        let mut effective = entry;
        if let Ok(prev) = writer.engine.get_state_machine().get_memory(effective.id) {
            let content_changed = prev.content != effective.content;
            if content_changed && effective.embedding.is_none() {
                return Err(MnemosStoreError::MissingEmbeddingOnContentChange(
                    effective.id,
                ));
            }

            // Preserve embedding on metadata-only updates when caller omits embedding.
            if !content_changed && effective.embedding.is_none() {
                effective.embedding = prev.embedding.clone();
            }
        }

        self.execute_write_transaction_locked(&mut writer, WriteOp::InsertMemory(effective))
    }

    pub fn delete_memory(&self, id: MemoryId) -> Result<CommandId> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        self.execute_write_transaction_locked(&mut writer, WriteOp::DeleteMemory(id))
    }

    pub fn add_edge(&self, from: MemoryId, to: MemoryId, relation: String) -> Result<CommandId> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        self.execute_write_transaction_locked(&mut writer, WriteOp::AddEdge { from, to, relation })
    }

    pub fn remove_edge(&self, from: MemoryId, to: MemoryId) -> Result<CommandId> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        self.execute_write_transaction_locked(&mut writer, WriteOp::RemoveEdge { from, to })
    }

    /// Rebuild in-memory vector index from current state machine entries.
    pub fn rebuild_vector_index(&self) -> Result<usize> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        writer.indexes = Self::build_vector_index(
            writer.engine.get_state_machine(),
            writer.indexes.vector.dimension(),
        )?;

        let indexed = writer.indexes.vector.len();
        Self::assert_vector_index_in_sync_inner(
            writer.engine.get_state_machine(),
            &writer.indexes,
        )?;
        self.publish_snapshot_from_write_state(&writer);
        Ok(indexed)
    }

    pub fn query(
        &self,
        query_text: &str,
        options: QueryOptions,
        embedder: &dyn QueryEmbedder,
    ) -> Result<QueryExecution> {
        Ok(self.query_with_snapshot(query_text, options, embedder)?.0)
    }

    pub fn query_with_snapshot(
        &self,
        query_text: &str,
        options: QueryOptions,
        embedder: &dyn QueryEmbedder,
    ) -> Result<(QueryExecution, Arc<ReadSnapshot>)> {
        let snapshot = self.snapshot();
        let plan = QueryPlanner::plan(options, snapshot.indexes().vector.len());
        let exec = QueryExecutor::execute(
            query_text,
            &plan,
            snapshot.state_machine(),
            snapshot.indexes(),
            embedder,
        )?;
        Ok((exec, snapshot))
    }

    pub fn query_with_plan(
        &self,
        query_text: &str,
        plan: &QueryPlan,
        embedder: &dyn QueryEmbedder,
    ) -> Result<QueryExecution> {
        let snapshot = self.snapshot();
        Ok(QueryExecutor::execute(
            query_text,
            plan,
            snapshot.state_machine(),
            snapshot.indexes(),
            embedder,
        )?)
    }

    pub fn query_with_trace(
        &self,
        query_text: &str,
        options: QueryOptions,
        embedder: &dyn QueryEmbedder,
        trace: &mut dyn FnMut(StageTrace),
    ) -> Result<QueryExecution> {
        let snapshot = self.snapshot();
        let plan = QueryPlanner::plan(options, snapshot.indexes().vector.len());
        Ok(QueryExecutor::execute_with_trace(
            query_text,
            &plan,
            snapshot.state_machine(),
            snapshot.indexes(),
            embedder,
            Some(trace),
        )?)
    }

    pub fn enforce_capacity(&self, policy: CapacityPolicy) -> Result<EvictionReport> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        let sync_now = matches!(self.sync_policy, SyncPolicy::Strict);
        let report = if sync_now {
            writer.engine.enforce_capacity(policy)?
        } else {
            writer.engine.enforce_capacity_unsynced(policy)?
        };
        for id in &report.evicted_ids {
            let _ = writer.indexes.vector_index_mut().remove(*id);
        }

        Self::assert_vector_index_in_sync_inner(
            writer.engine.get_state_machine(),
            &writer.indexes,
        )?;
        self.publish_snapshot_from_write_state(&writer);
        if !sync_now {
            self.mark_pending_write(report.evicted_ids.len());
        } else {
            self.clear_pending_sync_state();
        }
        Ok(report)
    }

    pub fn compact_segments(&self) -> Result<CompactionReport> {
        let mut writer = self.writer.lock().expect("writer lock poisoned");
        Ok(writer.engine.compact_segments()?)
    }

    pub fn state_machine(&self) -> StateMachine {
        self.snapshot().state_machine().clone()
    }

    pub fn vector_dimension(&self) -> usize {
        self.snapshot().indexes().vector.dimension()
    }

    pub fn indexed_embeddings(&self) -> usize {
        self.snapshot().indexes().vector.len()
    }

    pub fn wal_len(&self) -> u64 {
        self.writer
            .lock()
            .expect("writer lock poisoned")
            .engine
            .wal_len()
    }

    fn mark_pending_write(&self, ops: usize) {
        if ops == 0 {
            return;
        }
        let (lock, cvar) = &*self.sync_control;
        let mut runtime = lock.lock().expect("sync runtime lock poisoned");
        runtime.pending_ops = runtime.pending_ops.saturating_add(ops);
        if runtime.dirty_since.is_none() {
            runtime.dirty_since = Some(Instant::now());
        }
        cvar.notify_one();
    }

    fn clear_pending_sync_state(&self) {
        let (lock, _) = &*self.sync_control;
        let mut runtime = lock.lock().expect("sync runtime lock poisoned");
        runtime.pending_ops = 0;
        runtime.dirty_since = None;
    }

    fn execute_write_transaction_locked(
        &self,
        writer: &mut WriteState,
        op: WriteOp,
    ) -> Result<CommandId> {
        let sync_now = matches!(self.sync_policy, SyncPolicy::Strict);
        let cmd_id = match op {
            WriteOp::InsertMemory(entry) => {
                if let Some(embedding) = entry.embedding.as_ref() {
                    if embedding.len() != writer.indexes.vector.dimension() {
                        return Err(crate::index::vector::VectorError::DimensionMismatch {
                            expected: writer.indexes.vector.dimension(),
                            actual: embedding.len(),
                        }
                        .into());
                    }
                }
                let id = if sync_now {
                    writer
                        .engine
                        .execute_command(Command::InsertMemory(entry.clone()))?
                } else {
                    writer
                        .engine
                        .execute_command_unsynced(Command::InsertMemory(entry.clone()))?
                };
                match entry.embedding {
                    Some(embedding) => writer
                        .indexes
                        .vector_index_mut()
                        .index(entry.id, embedding)?,
                    None => {
                        let _ = writer.indexes.vector_index_mut().remove(entry.id);
                    }
                }
                id
            }
            WriteOp::DeleteMemory(id) => {
                let cmd_id = if sync_now {
                    writer.engine.execute_command(Command::DeleteMemory(id))?
                } else {
                    writer
                        .engine
                        .execute_command_unsynced(Command::DeleteMemory(id))?
                };
                let _ = writer.indexes.vector_index_mut().remove(id);
                cmd_id
            }
            WriteOp::AddEdge { from, to, relation } => {
                if sync_now {
                    writer
                        .engine
                        .execute_command(Command::AddEdge { from, to, relation })?
                } else {
                    writer.engine.execute_command_unsynced(Command::AddEdge {
                        from,
                        to,
                        relation,
                    })?
                }
            }
            WriteOp::RemoveEdge { from, to } => {
                if sync_now {
                    writer
                        .engine
                        .execute_command(Command::RemoveEdge { from, to })?
                } else {
                    writer
                        .engine
                        .execute_command_unsynced(Command::RemoveEdge { from, to })?
                }
            }
        };

        Self::assert_vector_index_in_sync_inner(
            writer.engine.get_state_machine(),
            &writer.indexes,
        )?;
        self.publish_snapshot_from_write_state(writer);
        if !sync_now {
            self.mark_pending_write(1);
        } else {
            self.clear_pending_sync_state();
        }
        Ok(cmd_id)
    }

    fn publish_snapshot_from_write_state(&self, writer: &WriteState) {
        let new_snapshot = Arc::new(ReadSnapshot::new(
            writer.engine.get_state_machine().clone(),
            writer.indexes.clone(),
        ));
        *self.snapshot.write().expect("snapshot lock poisoned") = new_snapshot;
    }

    fn build_vector_index(
        state_machine: &StateMachine,
        vector_dimension: usize,
    ) -> Result<IndexLayer> {
        let mut indexes = IndexLayer::new(vector_dimension);
        for entry in state_machine.all_memories() {
            if let Some(embedding) = entry.embedding.clone() {
                indexes.vector_index_mut().index(entry.id, embedding)?;
            }
        }
        Ok(indexes)
    }

    fn assert_vector_index_in_sync_inner(
        state_machine: &StateMachine,
        indexes: &IndexLayer,
    ) -> Result<()> {
        let state_ids: HashSet<MemoryId> = state_machine
            .all_memories()
            .into_iter()
            .filter(|e| e.embedding.is_some())
            .map(|e| e.id)
            .collect();
        let index_ids: HashSet<MemoryId> = indexes.vector.indexed_ids().into_iter().collect();

        if state_ids != index_ids {
            return Err(MnemosStoreError::InvariantViolation(format!(
                "vector index mismatch: state={} index={}",
                state_ids.len(),
                index_ids.len()
            )));
        }
        Ok(())
    }
}

impl Drop for MnemosStore {
    fn drop(&mut self) {
        let _ = self.flush();
        let (lock, cvar) = &*self.sync_control;
        {
            let mut runtime = lock.lock().expect("sync runtime lock poisoned");
            runtime.shutdown = true;
            cvar.notify_all();
        }
        if let Some(handle) = self.sync_thread.take() {
            let _ = handle.join();
        }
    }
}

enum WriteOp {
    InsertMemory(MemoryEntry),
    DeleteMemory(MemoryId),
    AddEdge {
        from: MemoryId,
        to: MemoryId,
        relation: String,
    },
    RemoveEdge {
        from: MemoryId,
        to: MemoryId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use tempfile::TempDir;

    struct TestEmbedder;
    impl QueryEmbedder for TestEmbedder {
        fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    struct SlowEmbedder {
        delay: Duration,
    }
    impl QueryEmbedder for SlowEmbedder {
        fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
            thread::sleep(self.delay);
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    #[test]
    fn test_store_insert_and_query() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let a = MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"a".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.8);
        let b = MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"b".to_vec(), 2000)
            .with_embedding(vec![0.9, 0.1, 0.0])
            .with_importance(0.2);
        store.insert_memory(a).unwrap();
        store.insert_memory(b).unwrap();

        let mut options = QueryOptions::with_top_k(2);
        options.namespace = Some("agent1".to_string());
        let out = store.query("hello", options, &TestEmbedder).unwrap();
        assert_eq!(out.hits.len(), 2);
    }

    #[test]
    fn test_store_delete_updates_vector_index() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let entry = MemoryEntry::new(MemoryId(10), "agent1".to_string(), b"x".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert_memory(entry).unwrap();
        assert_eq!(store.indexed_embeddings(), 1);

        store.delete_memory(MemoryId(10)).unwrap();
        assert_eq!(store.indexed_embeddings(), 0);
    }

    #[test]
    fn test_store_recover_auto_rebuilds_vector_index() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");

        {
            let store = MnemosStore::new(&wal, &seg, 3).unwrap();
            let entry = MemoryEntry::new(MemoryId(77), "agent1".to_string(), b"z".to_vec(), 1000)
                .with_embedding(vec![1.0, 0.0, 0.0]);
            store.insert_memory(entry).unwrap();
            assert_eq!(store.indexed_embeddings(), 1);
        }

        let recovered = MnemosStore::recover(&wal, &seg, 3).unwrap();
        assert_eq!(recovered.indexed_embeddings(), 1);

        let mut options = QueryOptions::with_top_k(1);
        options.namespace = Some("agent1".to_string());
        let out = recovered.query("hello", options, &TestEmbedder).unwrap();
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].id, MemoryId(77));
    }

    #[test]
    fn test_content_change_requires_embedding() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(90), "agent1".to_string(), b"old".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert_memory(original).unwrap();

        let changed_content =
            MemoryEntry::new(MemoryId(90), "agent1".to_string(), b"new".to_vec(), 1001);
        let err = store.insert_memory(changed_content).unwrap_err();
        assert!(matches!(
            err,
            MnemosStoreError::MissingEmbeddingOnContentChange(MemoryId(90))
        ));
    }

    #[test]
    fn test_unchanged_content_preserves_embedding_when_omitted() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(91), "agent1".to_string(), b"same".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.2);
        store.insert_memory(original).unwrap();

        let updated = MemoryEntry::new(MemoryId(91), "agent1".to_string(), b"same".to_vec(), 1001)
            .with_importance(0.9);
        store.insert_memory(updated).unwrap();

        assert_eq!(store.indexed_embeddings(), 1);
    }

    #[test]
    fn test_content_change_with_embedding_replaces_vector() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(92), "agent1".to_string(), b"old".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert_memory(original).unwrap();

        let changed = MemoryEntry::new(MemoryId(92), "agent1".to_string(), b"new".to_vec(), 1001)
            .with_embedding(vec![0.0, 1.0, 0.0]);
        store.insert_memory(changed).unwrap();

        assert_eq!(store.indexed_embeddings(), 1);
    }

    #[test]
    fn test_failed_write_keeps_snapshot_unchanged() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(99), "agent1".to_string(), b"old".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert_memory(original).unwrap();

        let before = store.snapshot();
        let err = store
            .insert_memory(MemoryEntry::new(
                MemoryId(99),
                "agent1".to_string(),
                b"new".to_vec(),
                1001,
            ))
            .unwrap_err();
        assert!(matches!(
            err,
            MnemosStoreError::MissingEmbeddingOnContentChange(MemoryId(99))
        ));

        let after = store.snapshot();
        let old_before = before
            .state_machine()
            .get_memory(MemoryId(99))
            .unwrap()
            .content
            .clone();
        let old_after = after
            .state_machine()
            .get_memory(MemoryId(99))
            .unwrap()
            .content
            .clone();
        assert_eq!(old_before, b"old".to_vec());
        assert_eq!(old_after, b"old".to_vec());
    }

    #[test]
    fn test_long_running_query_reads_single_snapshot() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = Arc::new(MnemosStore::new(&wal, &seg, 3).unwrap());

        store
            .insert_memory(
                MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"one".to_vec(), 1000)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        let snapshot = store.snapshot();
        let mut options = QueryOptions::with_top_k(10);
        options.namespace = Some("agent1".to_string());
        let plan = QueryPlanner::plan(options, snapshot.indexes().vector.len());

        let snapshot_for_query = Arc::clone(&snapshot);
        let query_thread = thread::spawn(move || {
            QueryExecutor::execute(
                "q",
                &plan,
                snapshot_for_query.state_machine(),
                snapshot_for_query.indexes(),
                &SlowEmbedder {
                    delay: Duration::from_millis(250),
                },
            )
            .unwrap()
        });

        thread::sleep(Duration::from_millis(50));
        for id in 2..=11 {
            store
                .insert_memory(
                    MemoryEntry::new(
                        MemoryId(id),
                        "agent1".to_string(),
                        format!("m{id}").into_bytes(),
                        1000 + id,
                    )
                    .with_embedding(vec![1.0, 0.0, 0.0]),
                )
                .unwrap();
        }

        let out = query_thread.join().unwrap();
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].id, MemoryId(1));

        // Latest snapshot sees post-write state.
        let latest = store.snapshot();
        assert_eq!(latest.state_machine().len(), 11);
    }

    #[test]
    fn test_batch_policy_flushes_on_threshold() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store_batch.wal");
        let seg = temp.path().join("segments_batch");
        let store = MnemosStore::new_with_policy(
            &wal,
            &seg,
            3,
            SyncPolicy::Batch {
                max_ops: 2,
                max_delay_ms: 10_000,
            },
        )
        .unwrap();

        store
            .insert_memory(
                MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"one".to_vec(), 1000)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();
        store
            .insert_memory(
                MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"two".to_vec(), 1001)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(120));

        let recovered = MnemosStore::recover(&wal, &seg, 3).unwrap();
        assert_eq!(recovered.state_machine().len(), 2);
    }

    #[test]
    fn test_async_policy_flushes_by_interval() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store_async.wal");
        let seg = temp.path().join("segments_async");
        let store =
            MnemosStore::new_with_policy(&wal, &seg, 3, SyncPolicy::Async { interval_ms: 25 })
                .unwrap();

        store
            .insert_memory(
                MemoryEntry::new(MemoryId(10), "agent1".to_string(), b"ten".to_vec(), 1010)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(120));

        let recovered = MnemosStore::recover(&wal, &seg, 3).unwrap();
        assert_eq!(recovered.state_machine().len(), 1);
    }
}
