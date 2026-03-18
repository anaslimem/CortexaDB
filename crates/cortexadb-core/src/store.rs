use std::{
    collections::HashSet,
    path::Path,
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use thiserror::Error;

use crate::{
    core::{
        command::Command,
        memory_entry::{MemoryEntry, MemoryId},
        state_machine::StateMachine,
    },
    engine::{CapacityPolicy, Engine, EvictionReport, SyncPolicy},
    index::{vector::VectorBackendMode, IndexLayer},
    query::{
        IntentAnchors, QueryEmbedder, QueryExecution, QueryExecutor, QueryOptions, QueryPlan,
        QueryPlanner, StageTrace,
    },
    storage::{
        checkpoint::{
            checkpoint_path_from_wal, load_checkpoint, save_checkpoint, LoadedCheckpoint,
        },
        compaction::CompactionReport,
        wal::{CommandId, WriteAheadLog},
    },
};

#[derive(Error, Debug)]
pub enum CortexaDBStoreError {
    #[error("Engine error: {0}")]
    Engine(#[from] crate::engine::EngineError),
    #[error("Vector index error: {0}")]
    Vector(#[from] crate::index::vector::VectorError),
    #[error("Query error: {0}")]
    Query(#[from] crate::query::HybridQueryError),
    #[error("Checkpoint error: {0}")]
    Checkpoint(#[from] crate::storage::checkpoint::CheckpointError),
    #[error("WAL error: {0}")]
    Wal(#[from] crate::storage::wal::WalError),
    #[error("Invariant violation: {0}")]
    InvariantViolation(String),
    #[error("Embedding required when content changes for memory id {0:?}")]
    MissingEmbeddingOnContentChange(MemoryId),
    #[error("Lock was poisoned during {0}")]
    LockPoisoned(&'static str),
}

pub type Result<T> = std::result::Result<T, CortexaDBStoreError>;

#[derive(Clone)]
pub struct ReadSnapshot {
    state_machine: StateMachine,
    indexes: IndexLayer,
}

impl ReadSnapshot {
    fn new(state_machine: StateMachine, indexes: IndexLayer) -> Self {
        Self { state_machine, indexes }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPolicy {
    Disabled,
    Periodic { every_ops: usize, every_ms: u64 },
}

struct CheckpointRuntime {
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
pub struct CortexaDBStore {
    writer: Arc<Mutex<WriteState>>,
    snapshot: Arc<ArcSwap<ReadSnapshot>>,
    sync_policy: SyncPolicy,
    sync_control: Arc<(Mutex<SyncRuntime>, Condvar)>,
    sync_thread: Option<JoinHandle<()>>,
    checkpoint_policy: CheckpointPolicy,
    checkpoint_path: std::path::PathBuf,
    hnsw_path: std::path::PathBuf,
    checkpoint_control: Arc<(Mutex<CheckpointRuntime>, Condvar)>,
    checkpoint_thread: Option<JoinHandle<()>>,
    capacity_policy: CapacityPolicy,
}

fn writer_lock<'a>(
    writer: &'a std::sync::Mutex<WriteState>,
) -> Result<std::sync::MutexGuard<'a, WriteState>> {
    writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))
}

impl CortexaDBStore {
    pub fn new<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
    ) -> Result<Self> {
        Self::new_with_policies(
            wal_path,
            segments_dir,
            vector_dimension,
            SyncPolicy::Strict,
            CheckpointPolicy::Disabled,
            CapacityPolicy::new(None, None),
            crate::index::hnsw::IndexMode::Exact,
        )
    }

    pub fn new_with_policy<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
    ) -> Result<Self> {
        Self::new_with_policies(
            wal_path,
            segments_dir,
            vector_dimension,
            sync_policy,
            CheckpointPolicy::Disabled,
            CapacityPolicy::new(None, None),
            crate::index::hnsw::IndexMode::Exact,
        )
    }

    pub fn new_with_policies<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
        checkpoint_policy: CheckpointPolicy,
        capacity_policy: CapacityPolicy,
        index_mode: crate::index::hnsw::IndexMode,
    ) -> Result<Self> {
        let wal_path = wal_path.as_ref().to_path_buf();
        let segments_dir = segments_dir.as_ref().to_path_buf();
        let checkpoint_path = checkpoint_path_from_wal(&wal_path);
        let engine = Engine::new(&wal_path, &segments_dir)?;
        Self::from_engine(
            engine,
            vector_dimension,
            sync_policy,
            checkpoint_policy,
            capacity_policy,
            checkpoint_path,
            index_mode,
        )
    }

    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
    ) -> Result<Self> {
        Self::recover_with_policies(
            wal_path,
            segments_dir,
            vector_dimension,
            SyncPolicy::Strict,
            CheckpointPolicy::Disabled,
            CapacityPolicy::new(None, None),
            crate::index::hnsw::IndexMode::Exact,
        )
    }

    pub fn recover_with_policy<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
    ) -> Result<Self> {
        Self::recover_with_policies(
            wal_path,
            segments_dir,
            vector_dimension,
            sync_policy,
            CheckpointPolicy::Disabled,
            CapacityPolicy::new(None, None),
            crate::index::hnsw::IndexMode::Exact,
        )
    }

    pub fn recover_with_policies<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
        checkpoint_policy: CheckpointPolicy,
        capacity_policy: CapacityPolicy,
        index_mode: crate::index::hnsw::IndexMode,
    ) -> Result<Self> {
        let wal_path = wal_path.as_ref().to_path_buf();
        let segments_dir = segments_dir.as_ref().to_path_buf();
        let checkpoint_path = checkpoint_path_from_wal(&wal_path);

        let loaded_checkpoint = load_checkpoint(&checkpoint_path)?;
        let engine = match loaded_checkpoint {
            Some(LoadedCheckpoint { last_applied_id, state_machine }) => {
                Engine::recover_from_checkpoint(
                    &wal_path,
                    &segments_dir,
                    Some((state_machine, CommandId(last_applied_id))),
                )?
            }
            None => Engine::recover(&wal_path, &segments_dir)?,
        };
        Self::from_engine(
            engine,
            vector_dimension,
            sync_policy,
            checkpoint_policy,
            capacity_policy,
            checkpoint_path,
            index_mode,
        )
    }

    fn from_engine(
        engine: Engine,
        vector_dimension: usize,
        sync_policy: SyncPolicy,
        checkpoint_policy: CheckpointPolicy,
        capacity_policy: CapacityPolicy,
        checkpoint_path: std::path::PathBuf,
        index_mode: crate::index::hnsw::IndexMode,
    ) -> Result<Self> {
        let hnsw_path = checkpoint_path.with_extension("hnsw");

        let hnsw_config = match index_mode {
            crate::index::hnsw::IndexMode::Exact => None,
            crate::index::hnsw::IndexMode::Hnsw(config) => Some(config),
        };

        let loaded_hnsw = if let Some(config) = hnsw_config.as_ref() {
            match crate::index::VectorIndex::load_hnsw(&hnsw_path, vector_dimension, config.clone())
            {
                Ok(Some(backend)) => {
                    log::info!("Loaded HNSW index from disk (fast recovery)");
                    Some(backend)
                }
                Ok(None) => {
                    log::info!("No HNSW index file found, building fresh index");
                    None
                }
                Err(e) => {
                    log::warn!("Failed to load HNSW index, rebuilding: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let indexes = Self::build_vector_index(
            engine.get_state_machine(),
            vector_dimension,
            hnsw_config.as_ref(),
            loaded_hnsw,
        )?;
        Self::assert_vector_index_in_sync_inner(engine.get_state_machine(), &indexes)?;

        let snapshot = Arc::new(ArcSwap::from_pointee(ReadSnapshot::new(
            engine.get_state_machine().clone(),
            indexes.clone(),
        )));

        // Capture wal_path before `engine` moves into WriteState — used by the
        // checkpoint thread without needing to re-acquire the writer lock each time.
        let wal_path_for_ckpt = engine.wal_path().to_path_buf();
        let writer = Arc::new(Mutex::new(WriteState { engine, indexes }));
        let sync_control = Arc::new((
            Mutex::new(SyncRuntime { pending_ops: 0, dirty_since: None, shutdown: false }),
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

        let checkpoint_control = Arc::new((
            Mutex::new(CheckpointRuntime { pending_ops: 0, dirty_since: None, shutdown: false }),
            Condvar::new(),
        ));
        let checkpoint_thread = match checkpoint_policy {
            CheckpointPolicy::Disabled => None,
            CheckpointPolicy::Periodic { .. } => Some(Self::spawn_checkpoint_thread(
                Arc::clone(&writer),
                Arc::clone(&checkpoint_control),
                checkpoint_path.clone(),
                wal_path_for_ckpt,
                checkpoint_policy,
            )),
        };

        Ok(Self {
            writer,
            snapshot,
            sync_policy,
            sync_control,
            sync_thread,
            checkpoint_policy,
            checkpoint_path,
            hnsw_path,
            checkpoint_control,
            checkpoint_thread,
            capacity_policy,
        })
    }

    pub fn snapshot(&self) -> Arc<ReadSnapshot> {
        self.snapshot.load_full()
    }

    pub fn flush(&self) -> Result<()> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        writer.engine.flush()?;
        self.clear_pending_sync_state();
        Ok(())
    }

    pub fn checkpoint_now(&self) -> Result<()> {
        let snapshot = self.snapshot();
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        let last_applied_id = writer.engine.last_applied_id().0;
        save_checkpoint(&self.checkpoint_path, snapshot.state_machine(), last_applied_id)?;

        if let Err(e) = snapshot.indexes.vector_index().save_hnsw(&self.hnsw_path) {
            log::warn!("Warning: Failed to save HNSW index: {}", e);
        }

        // Truncate WAL prefix — only keep entries written after the checkpoint.
        let wal_path = writer.engine.wal_path().to_path_buf();
        WriteAheadLog::truncate_prefix(&wal_path, CommandId(last_applied_id))?;
        writer.engine.reopen_wal()?;

        drop(writer);
        self.clear_pending_checkpoint_state();
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
                    SyncPolicy::Batch { max_ops, max_delay_ms } => {
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
                                    let timed_out = runtime
                                        .dirty_since
                                        .map(|d| d.elapsed() >= max_delay)
                                        .unwrap_or(true);
                                    if runtime.pending_ops < max_ops && !timed_out {
                                        drop(runtime);
                                        continue;
                                    }
                                }
                            } else {
                                runtime = cvar.wait(runtime).expect("sync runtime wait poisoned");
                                drop(runtime);
                                continue;
                            }
                        }
                    }
                    SyncPolicy::Async { interval_ms } => {
                        let wait = Duration::from_millis(interval_ms.max(1));
                        let (guard, _) =
                            cvar.wait_timeout(runtime, wait).expect("sync runtime wait poisoned");
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
                    let mut write_state = match writer.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            log::error!("cortexadb sync manager flush error (lock poisoned): {e}");
                            continue;
                        }
                    };
                    if let Err(err) = write_state.engine.flush_buffers() {
                        log::error!("cortexadb sync manager flush_buffers error: {err}");
                        continue;
                    }

                    // Extract cloned file handles to perform blocking fsync outside the lock
                    let handles = match write_state.engine.get_file_handles() {
                        Ok(hs) => Some(hs),
                        Err(err) => {
                            log::error!("cortexadb sync manager get_file_handles error: {err}");
                            None
                        }
                    };

                    // Drop the massive global lock so other agents can continue inserting memories!
                    drop(write_state);

                    if let Some((_wal_file, mut seg_file)) = handles {
                        // WAL file doesn't need to be mut for sync_all
                        if let Err(err) = _wal_file.sync_all() {
                            log::error!("cortexadb slow fsync error on wal: {err}");
                        }
                        if let Some(s) = seg_file.as_mut() {
                            if let Err(err) = s.sync_all() {
                                log::error!("cortexadb slow fsync error on segment: {err}");
                            }
                        }
                    }
                }
            }
        })
    }

    fn spawn_checkpoint_thread(
        writer: Arc<Mutex<WriteState>>,
        checkpoint_control: Arc<(Mutex<CheckpointRuntime>, Condvar)>,
        checkpoint_path: std::path::PathBuf,
        wal_path: std::path::PathBuf,
        checkpoint_policy: CheckpointPolicy,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let (lock, cvar) = &*checkpoint_control;
            loop {
                let mut runtime = lock.lock().expect("checkpoint runtime lock poisoned");
                if runtime.shutdown {
                    break;
                }

                match checkpoint_policy {
                    CheckpointPolicy::Disabled => break,
                    CheckpointPolicy::Periodic { every_ops, every_ms } => {
                        let every_ops = every_ops.max(1);
                        let max_delay = Duration::from_millis(every_ms.max(1));
                        if runtime.pending_ops < every_ops {
                            if let Some(dirty_since) = runtime.dirty_since {
                                let elapsed = dirty_since.elapsed();
                                if elapsed < max_delay {
                                    let timeout = max_delay - elapsed;
                                    let (guard, _) = cvar
                                        .wait_timeout(runtime, timeout)
                                        .expect("checkpoint runtime wait poisoned");
                                    runtime = guard;
                                }
                            } else {
                                runtime =
                                    cvar.wait(runtime).expect("checkpoint runtime wait poisoned");
                            }
                        }
                    }
                }

                if runtime.shutdown {
                    break;
                }

                if runtime.pending_ops == 0 {
                    continue;
                }
                runtime.pending_ops = 0;
                runtime.dirty_since = None;
                drop(runtime);

                // ── Atomic snapshot capture ───────────────────────────────────────
                // We MUST read both `state_machine` and `last_applied_id` under the
                // SAME writer lock acquisition. If we read them separately there is a
                // window where the main thread advances `last_applied_id` beyond the
                // state captured in the snapshot — the checkpoint would then record an
                // ID that is ahead of its actual state, causing entries to be silently
                // skipped on crash recovery (TOCTOU).
                let (checkpoint_state, last_applied_id) = match writer.lock() {
                    Ok(guard) => {
                        let id = guard.engine.last_applied_id().0;
                        let state = guard.engine.get_state_machine().clone();
                        (state, id)
                    }
                    Err(e) => {
                        log::error!("cortexadb checkpoint error (lock poisoned): {e}");
                        continue;
                    }
                };
                // Lock released — all I/O happens outside it.

                if let Err(err) =
                    save_checkpoint(&checkpoint_path, &checkpoint_state, last_applied_id)
                {
                    log::error!("cortexadb checkpoint write error: {err}");
                } else {
                    // Truncate WAL prefix after successful checkpoint.
                    // `wal_path` was captured at thread-spawn time — no lock needed.
                    if let Err(err) =
                        WriteAheadLog::truncate_prefix(&wal_path, CommandId(last_applied_id))
                    {
                        log::error!("cortexadb WAL truncation error: {err}");
                    } else {
                        let mut write_guard = match writer.lock() {
                            Ok(g) => g,
                            Err(e) => {
                                log::error!("cortexadb checkpoint error while reopening WAL (lock poisoned): {e}");
                                continue;
                            }
                        };
                        if let Err(err) = write_guard.engine.reopen_wal() {
                            log::error!("cortexadb WAL reopen error: {err}");
                        }
                    }
                }
            }
        })
    }

    pub fn add(&self, entry: MemoryEntry) -> Result<CommandId> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;

        let mut effective = entry;
        if let Ok(prev) = writer.engine.get_state_machine().get_memory(effective.id) {
            let content_changed = prev.content != effective.content;
            if content_changed && effective.embedding.is_none() {
                return Err(CortexaDBStoreError::MissingEmbeddingOnContentChange(effective.id));
            }

            // Preserve embedding on metadata-only updates when caller omits embedding.
            if !content_changed && effective.embedding.is_none() {
                effective.embedding = prev.embedding.clone();
            }
        }

        self.execute_write_transaction_locked(&mut writer, WriteOp::Add(effective))
    }

    pub fn add_batch(&self, entries: Vec<MemoryEntry>) -> Result<CommandId> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        let sync_now = matches!(self.sync_policy, SyncPolicy::Strict);
        let mut last_cmd_id = CommandId(0);

        for entry in entries {
            let mut effective = entry;
            // Check for previous state to handle partial updates if necessary
            if let Ok(prev) = writer.engine.get_state_machine().get_memory(effective.id) {
                let content_changed = prev.content != effective.content;
                if content_changed && effective.embedding.is_none() {
                    return Err(CortexaDBStoreError::MissingEmbeddingOnContentChange(effective.id));
                }
                if !content_changed && effective.embedding.is_none() {
                    effective.embedding = prev.embedding.clone();
                }
            }

            // Validate dimension
            if let Some(embedding) = effective.embedding.as_ref() {
                if embedding.len() != writer.indexes.vector.dimension() {
                    return Err(crate::index::vector::VectorError::DimensionMismatch {
                        expected: writer.indexes.vector.dimension(),
                        actual: embedding.len(),
                    }
                    .into());
                }
            }

            // Execute unsynced for the whole batch
            last_cmd_id =
                writer.engine.execute_command_unsynced(Command::Add(effective.clone()))?;

            // Update vector index
            match effective.embedding {
                Some(embedding) => {
                    writer.indexes.vector_index_mut().index_in_collection(
                        &effective.collection,
                        effective.id,
                        embedding,
                    )?;
                }
                None => {
                    let _ = writer.indexes.vector_index_mut().remove(effective.id);
                }
            }
        }

        // Single flush for the entire batch if in strict mode
        if sync_now {
            writer.engine.flush()?;
        }

        // Publish snapshot once after the batch
        self.publish_snapshot_from_write_state(&writer);

        Ok(last_cmd_id)
    }

    pub fn delete(&self, id: MemoryId) -> Result<CommandId> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        self.execute_write_transaction_locked(&mut writer, WriteOp::Delete(id))
    }

    pub fn connect(&self, from: MemoryId, to: MemoryId, relation: String) -> Result<CommandId> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        self.execute_write_transaction_locked(&mut writer, WriteOp::Connect { from, to, relation })
    }

    pub fn disconnect(&self, from: MemoryId, to: MemoryId) -> Result<CommandId> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        self.execute_write_transaction_locked(&mut writer, WriteOp::Disconnect { from, to })
    }

    /// Rebuild in-memory vector index from current state machine entries.
    pub fn rebuild_vector_index(&self) -> Result<usize> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        writer.indexes = Self::build_vector_index(
            writer.engine.get_state_machine(),
            writer.indexes.vector.dimension(),
            None,
            None,
        )?;

        let indexed = writer.indexes.vector.len();
        Self::assert_vector_index_in_sync_inner(
            writer.engine.get_state_machine(),
            &writer.indexes,
        )?;
        self.publish_snapshot_from_write_state(&writer);
        Ok(indexed)
    }

    pub fn set_vector_backend_mode(&self, mode: VectorBackendMode) -> Result<()> {
        let mut writer = writer_lock(&self.writer)?;
        writer.indexes.set_vector_backend_mode(mode);
        self.publish_snapshot_from_write_state(&writer);
        Ok(())
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
        mut options: QueryOptions,
        embedder: &dyn QueryEmbedder,
    ) -> Result<(QueryExecution, Arc<ReadSnapshot>)> {
        let snapshot = self.snapshot();

        if let Some(anchors) = options.intent_anchors.take() {
            Self::apply_intent_adjustments(&mut options, &anchors, query_text, embedder);
        }

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

    fn apply_intent_adjustments(
        options: &mut QueryOptions,
        anchors: &IntentAnchors,
        query_text: &str,
        embedder: &dyn QueryEmbedder,
    ) {
        if let Ok(query_emb) = embedder.embed(query_text) {
            if let Some(adj) = QueryPlanner::infer_intent_adjustments(
                &query_emb,
                &anchors.semantic,
                &anchors.recency,
                &anchors.graph,
                anchors.graph_hops_2_threshold,
                anchors.graph_hops_3_threshold,
                anchors.importance_pct,
            ) {
                options.score_weights = adj.score_weights;
                if let Some(exp) = options.graph_expansion.as_mut() {
                    exp.hops = adj.graph_hops;
                }
            }
        }
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
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        let sync_now = matches!(self.sync_policy, SyncPolicy::Strict);
        let report = Self::enforce_capacity_locked(&mut writer, policy, sync_now)?;

        self.publish_snapshot_from_write_state(&writer);
        if !sync_now {
            self.mark_pending_write(report.evicted_ids.len());
        } else {
            self.clear_pending_sync_state();
        }
        self.mark_pending_checkpoint(report.evicted_ids.len().max(1));
        Ok(report)
    }

    fn enforce_capacity_locked(
        writer: &mut WriteState,
        policy: CapacityPolicy,
        sync_now: bool,
    ) -> Result<EvictionReport> {
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
        Ok(report)
    }

    pub fn compact_segments(&self) -> Result<CompactionReport> {
        let mut writer =
            self.writer.lock().map_err(|_| CortexaDBStoreError::LockPoisoned("writer lock"))?;
        let report = writer.engine.compact_segments()?;
        writer.indexes.vector_index_mut().compact()?;
        self.publish_snapshot_from_write_state(&writer);
        Ok(report)
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

    pub fn wal_len(&self) -> Result<u64> {
        Ok(writer_lock(&self.writer)?.engine.wal_len())
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

    fn mark_pending_checkpoint(&self, ops: usize) {
        if ops == 0 || self.checkpoint_policy == CheckpointPolicy::Disabled {
            return;
        }
        let (lock, cvar) = &*self.checkpoint_control;
        let mut runtime = lock.lock().expect("checkpoint runtime lock poisoned");
        runtime.pending_ops = runtime.pending_ops.saturating_add(ops);
        if runtime.dirty_since.is_none() {
            runtime.dirty_since = Some(Instant::now());
        }
        cvar.notify_one();
    }

    fn clear_pending_checkpoint_state(&self) {
        let (lock, _) = &*self.checkpoint_control;
        let mut runtime = lock.lock().expect("checkpoint runtime lock poisoned");
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
            WriteOp::Add(entry) => {
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
                    writer.engine.execute_command(Command::Add(entry.clone()))?
                } else {
                    writer.engine.execute_command_unsynced(Command::Add(entry.clone()))?
                };
                match entry.embedding {
                    Some(embedding) => writer.indexes.vector_index_mut().index_in_collection(
                        &entry.collection,
                        entry.id,
                        embedding,
                    )?,
                    None => {
                        let _ = writer.indexes.vector_index_mut().remove(entry.id);
                    }
                }
                id
            }
            WriteOp::Delete(id) => {
                let cmd_id = if sync_now {
                    writer.engine.execute_command(Command::Delete(id))?
                } else {
                    writer.engine.execute_command_unsynced(Command::Delete(id))?
                };
                let _ = writer.indexes.vector_index_mut().remove(id);
                cmd_id
            }
            WriteOp::Connect { from, to, relation } => {
                if sync_now {
                    writer.engine.execute_command(Command::Connect { from, to, relation })?
                } else {
                    writer.engine.execute_command_unsynced(Command::Connect {
                        from,
                        to,
                        relation,
                    })?
                }
            }
            WriteOp::Disconnect { from, to } => {
                if sync_now {
                    writer.engine.execute_command(Command::Disconnect { from, to })?
                } else {
                    writer.engine.execute_command_unsynced(Command::Disconnect { from, to })?
                }
            }
        };

        let mut evicted = 0;
        if self.capacity_policy.max_entries.is_some() || self.capacity_policy.max_bytes.is_some() {
            let report = Self::enforce_capacity_locked(writer, self.capacity_policy, sync_now)?;
            evicted = report.evicted_ids.len();
        }

        Self::assert_vector_index_in_sync_inner(
            writer.engine.get_state_machine(),
            &writer.indexes,
        )?;
        self.publish_snapshot_from_write_state(writer);
        if !sync_now {
            self.mark_pending_write(1 + evicted);
        } else {
            self.clear_pending_sync_state();
        }
        self.mark_pending_checkpoint(1 + evicted);
        Ok(cmd_id)
    }

    fn publish_snapshot_from_write_state(&self, writer: &WriteState) {
        let new_snapshot = Arc::new(ReadSnapshot::new(
            writer.engine.get_state_machine().clone(),
            writer.indexes.clone(),
        ));
        self.snapshot.store(new_snapshot);
    }

    fn build_vector_index(
        state_machine: &StateMachine,
        vector_dimension: usize,
        hnsw_config: Option<&crate::index::hnsw::HnswConfig>,
        loaded_hnsw: Option<crate::index::hnsw::HnswBackend>,
    ) -> Result<IndexLayer> {
        let has_loaded_hnsw = loaded_hnsw.is_some();
        let indexes = match hnsw_config {
            Some(config) => {
                if let Some(loaded) = loaded_hnsw {
                    IndexLayer::new_with_loaded_hnsw(vector_dimension, config.clone(), Some(loaded))
                } else {
                    IndexLayer::new_with_hnsw(vector_dimension, config.clone())
                }
            }
            None => IndexLayer::new(vector_dimension),
        };
        let mut indexes = indexes;

        let existing_ids: HashSet<MemoryId> = indexes.vector.indexed_ids().into_iter().collect();

        // Add or update missing embeddings from the state machine into the loaded HNSW
        for entry in state_machine.all_memories() {
            if let Some(embedding) = entry.embedding.clone() {
                if !existing_ids.contains(&entry.id) {
                    indexes.vector_index_mut().index_in_collection(
                        &entry.collection,
                        entry.id,
                        embedding,
                    )?;
                }
            }
        }

        // Remove IDs from HNSW that are no longer in the state machine (e.g. they were deleted in the replayed WAL)
        if has_loaded_hnsw {
            let state_ids: HashSet<MemoryId> = state_machine
                .all_memories()
                .into_iter()
                .filter(|e| e.embedding.is_some())
                .map(|e| e.id)
                .collect();

            for existing_id in existing_ids {
                if !state_ids.contains(&existing_id) {
                    let _ = indexes.vector_index_mut().remove(existing_id);
                }
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
            return Err(CortexaDBStoreError::InvariantViolation(format!(
                "vector index mismatch: state={} index={}",
                state_ids.len(),
                index_ids.len()
            )));
        }
        Ok(())
    }
}

impl Drop for CortexaDBStore {
    fn drop(&mut self) {
        // Shutdown background threads FIRST to avoid races with WAL truncation.
        {
            let (lock, cvar) = &*self.checkpoint_control;
            let mut runtime = lock.lock().expect("checkpoint runtime lock poisoned");
            runtime.shutdown = true;
            cvar.notify_all();
        }
        if let Some(handle) = self.checkpoint_thread.take() {
            let _ = handle.join();
        }

        {
            let (lock, cvar) = &*self.sync_control;
            let mut runtime = lock.lock().expect("sync runtime lock poisoned");
            runtime.shutdown = true;
            cvar.notify_all();
        }
        if let Some(handle) = self.sync_thread.take() {
            let _ = handle.join();
        }

        // Now do the final flush + checkpoint with no background threads running.
        let _ = self.flush();
        if self.checkpoint_policy != CheckpointPolicy::Disabled {
            let _ = self.checkpoint_now();
        }

        // Always save HNSW index on drop if it exists (automatic persistence)
        let snapshot = self.snapshot.load_full();
        if let Err(e) = snapshot.indexes.vector_index().save_hnsw(&self.hnsw_path) {
            log::warn!("Warning: Failed to save HNSW on drop: {}", e);
        }
    }
}

enum WriteOp {
    Add(MemoryEntry),
    Delete(MemoryId),
    Connect { from: MemoryId, to: MemoryId, relation: String },
    Disconnect { from: MemoryId, to: MemoryId },
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread, time::Duration};

    use tempfile::TempDir;

    use super::*;

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
        let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();

        let a = MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"a".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.8);
        let b = MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"b".to_vec(), 2000)
            .with_embedding(vec![0.9, 0.1, 0.0])
            .with_importance(0.2);
        store.add(a).unwrap();
        store.add(b).unwrap();

        let mut options = QueryOptions::with_top_k(2);
        options.collection = Some("agent1".to_string());
        let out = store.query("hello", options, &TestEmbedder).unwrap();
        assert_eq!(out.hits.len(), 2);
    }

    #[test]
    fn test_store_delete_updates_vector_index() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();

        let entry = MemoryEntry::new(MemoryId(10), "agent1".to_string(), b"x".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.add(entry).unwrap();
        assert_eq!(store.indexed_embeddings(), 1);

        store.delete(MemoryId(10)).unwrap();
        assert_eq!(store.indexed_embeddings(), 0);
    }

    #[test]
    fn test_store_recover_auto_rebuilds_vector_index() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");

        {
            let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();
            let entry = MemoryEntry::new(MemoryId(77), "agent1".to_string(), b"z".to_vec(), 1000)
                .with_embedding(vec![1.0, 0.0, 0.0]);
            store.add(entry).unwrap();
            assert_eq!(store.indexed_embeddings(), 1);
        }

        let recovered = CortexaDBStore::recover(&wal, &seg, 3).unwrap();
        assert_eq!(recovered.indexed_embeddings(), 1);

        let mut options = QueryOptions::with_top_k(1);
        options.collection = Some("agent1".to_string());
        let out = recovered.query("hello", options, &TestEmbedder).unwrap();
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].id, MemoryId(77));
    }

    #[test]
    fn test_content_change_requires_embedding() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(90), "agent1".to_string(), b"old".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.add(original).unwrap();

        let changed_content =
            MemoryEntry::new(MemoryId(90), "agent1".to_string(), b"new".to_vec(), 1001);
        let err = store.add(changed_content).unwrap_err();
        assert!(matches!(err, CortexaDBStoreError::MissingEmbeddingOnContentChange(MemoryId(90))));
    }

    #[test]
    fn test_unchanged_content_preserves_embedding_when_omitted() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(91), "agent1".to_string(), b"same".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.2);
        store.add(original).unwrap();

        let updated = MemoryEntry::new(MemoryId(91), "agent1".to_string(), b"same".to_vec(), 1001)
            .with_importance(0.9);
        store.add(updated).unwrap();

        assert_eq!(store.indexed_embeddings(), 1);
    }

    #[test]
    fn test_content_change_with_embedding_replaces_vector() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(92), "agent1".to_string(), b"old".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.add(original).unwrap();

        let changed = MemoryEntry::new(MemoryId(92), "agent1".to_string(), b"new".to_vec(), 1001)
            .with_embedding(vec![0.0, 1.0, 0.0]);
        store.add(changed).unwrap();

        assert_eq!(store.indexed_embeddings(), 1);
    }

    #[test]
    fn test_failed_write_keeps_snapshot_unchanged() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = CortexaDBStore::new(&wal, &seg, 3).unwrap();

        let original = MemoryEntry::new(MemoryId(99), "agent1".to_string(), b"old".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.add(original).unwrap();

        let before = store.snapshot();
        let err = store
            .add(MemoryEntry::new(MemoryId(99), "agent1".to_string(), b"new".to_vec(), 1001))
            .unwrap_err();
        assert!(matches!(err, CortexaDBStoreError::MissingEmbeddingOnContentChange(MemoryId(99))));

        let after = store.snapshot();
        let old_before = before.state_machine().get_memory(MemoryId(99)).unwrap().content.clone();
        let old_after = after.state_machine().get_memory(MemoryId(99)).unwrap().content.clone();
        assert_eq!(old_before, b"old".to_vec());
        assert_eq!(old_after, b"old".to_vec());
    }

    #[test]
    fn test_long_running_query_reads_single_snapshot() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let store = Arc::new(CortexaDBStore::new(&wal, &seg, 3).unwrap());

        store
            .add(
                MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"one".to_vec(), 1000)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        let snapshot = store.snapshot();
        let mut options = QueryOptions::with_top_k(10);
        options.collection = Some("agent1".to_string());
        let plan = QueryPlanner::plan(options, snapshot.indexes().vector.len());

        let snapshot_for_query = Arc::clone(&snapshot);
        let query_thread = thread::spawn(move || {
            QueryExecutor::execute(
                "q",
                &plan,
                snapshot_for_query.state_machine(),
                snapshot_for_query.indexes(),
                &SlowEmbedder { delay: Duration::from_millis(250) },
            )
            .unwrap()
        });

        thread::sleep(Duration::from_millis(50));
        for id in 2..=11 {
            store
                .add(
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
        let store = CortexaDBStore::new_with_policy(
            &wal,
            &seg,
            3,
            SyncPolicy::Batch { max_ops: 2, max_delay_ms: 10_000 },
        )
        .unwrap();

        store
            .add(
                MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"one".to_vec(), 1000)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();
        store
            .add(
                MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"two".to_vec(), 1001)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(120));

        let recovered = CortexaDBStore::recover(&wal, &seg, 3).unwrap();
        assert_eq!(recovered.state_machine().len(), 2);
    }

    #[test]
    fn test_async_policy_flushes_by_interval() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store_async.wal");
        let seg = temp.path().join("segments_async");
        let store =
            CortexaDBStore::new_with_policy(&wal, &seg, 3, SyncPolicy::Async { interval_ms: 25 })
                .unwrap();

        store
            .add(
                MemoryEntry::new(MemoryId(10), "agent1".to_string(), b"ten".to_vec(), 1010)
                    .with_embedding(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(120));

        let recovered = CortexaDBStore::recover(&wal, &seg, 3).unwrap();
        assert_eq!(recovered.state_machine().len(), 1);
    }

    #[test]
    fn test_periodic_checkpoint_recovery() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store_ckpt.wal");
        let seg = temp.path().join("segments_ckpt");

        {
            let store = CortexaDBStore::new_with_policies(
                &wal,
                &seg,
                3,
                SyncPolicy::Strict,
                CheckpointPolicy::Periodic { every_ops: 1, every_ms: 10 },
                CapacityPolicy::new(None, None),
                crate::index::hnsw::IndexMode::Exact,
            )
            .unwrap();

            store
                .add(
                    MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"a".to_vec(), 1000)
                        .with_embedding(vec![1.0, 0.0, 0.0]),
                )
                .unwrap();
            std::thread::sleep(Duration::from_millis(40));
            store.checkpoint_now().unwrap();
        }

        let recovered = CortexaDBStore::recover_with_policies(
            &wal,
            &seg,
            3,
            SyncPolicy::Strict,
            CheckpointPolicy::Disabled,
            CapacityPolicy::new(None, None),
            crate::index::hnsw::IndexMode::Exact,
        )
        .unwrap();
        assert_eq!(recovered.state_machine().len(), 1);
        assert!(recovered.state_machine().get_memory(MemoryId(1)).is_ok());
    }

    #[test]
    fn test_store_compaction_rebuilds_hnsw() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store_compact_hnsw.wal");
        let seg = temp.path().join("segments_compact_hnsw");

        let store = CortexaDBStore::new_with_policies(
            &wal,
            &seg,
            3, // dimension
            SyncPolicy::Strict,
            CheckpointPolicy::Disabled,
            CapacityPolicy::new(None, None),
            crate::index::hnsw::IndexMode::Hnsw(crate::index::hnsw::HnswConfig::default()),
        )
        .unwrap();

        // Add 5 items
        for i in 0..5 {
            let entry =
                MemoryEntry::new(MemoryId(i), "agent_x".to_string(), b"data".to_vec(), 1000)
                    .with_embedding(vec![1.0, 0.0, 0.0]);
            store.add(entry).unwrap();
        }

        assert_eq!(store.indexed_embeddings(), 5);

        // Remove 3 items (they become tombstones in HNSW)
        for i in 2..5 {
            store.delete(MemoryId(i)).unwrap();
        }

        assert_eq!(store.indexed_embeddings(), 2);

        // Trigger compaction. This compacts the segments and completely rebuilds the HNSW index!
        store.compact_segments().unwrap();

        assert_eq!(store.indexed_embeddings(), 2);

        // Ensure the remaining elements are still perfectly searchable
        let search_results = store
            .snapshot()
            .indexes()
            .vector
            .search_scoped(&[1.0, 0.0, 0.0], 5, Some("agent_x"), false, 1)
            .unwrap();
        assert_eq!(search_results.len(), 2);

        let ids: Vec<u64> = search_results.iter().map(|s| s.0 .0).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
    }
}
