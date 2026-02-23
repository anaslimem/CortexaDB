use std::path::Path;
use thiserror::Error;

use crate::core::command::Command;
use crate::core::memory_entry::{MemoryEntry, MemoryId};
use crate::core::state_machine::StateMachine;
use crate::storage::segment::SegmentStorage;
use crate::storage::wal::{CommandId, WriteAheadLog};

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("WAL error: {0}")]
    WalError(#[from] crate::storage::wal::WalError),
    #[error("State machine error: {0}")]
    StateMachineError(#[from] crate::core::state_machine::StateMachineError),
    #[error("Segment error: {0}")]
    SegmentError(#[from] crate::storage::segment::SegmentError),
    #[error("Compaction error: {0}")]
    CompactionError(#[from] crate::storage::compaction::CompactionError),
    #[error("Checkpoint error: {0}")]
    CheckpointError(#[from] crate::storage::checkpoint::CheckpointError),
    #[error(
        "Checkpoint/WAL gap detected: checkpoint_last_applied={checkpoint_last_applied}, wal_highest={wal_highest:?}"
    )]
    CheckpointWalGap {
        checkpoint_last_applied: u64,
        wal_highest: Option<u64>,
    },
    #[error("Engine not recovered properly")]
    NotRecovered,
}

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    Strict,
    Batch { max_ops: usize, max_delay_ms: u64 },
    Async { interval_ms: u64 },
}

/// Capacity limits for automatic eviction.
#[derive(Debug, Clone, Copy)]
pub struct CapacityPolicy {
    pub max_entries: Option<usize>,
    pub max_bytes: Option<u64>,
}

impl CapacityPolicy {
    pub const fn new(max_entries: Option<usize>, max_bytes: Option<u64>) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }
}

/// Summary of a capacity enforcement pass.
#[derive(Debug, Clone)]
pub struct EvictionReport {
    pub evicted_ids: Vec<MemoryId>,
    pub entries_before: usize,
    pub entries_after: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
}

/// Main engine coordinating WAL + SegmentStorage + StateMachine
///
/// Ensures durability and efficient replication by:
/// 1. Writing command to WAL first (command durability)
/// 2. Writing entry to SegmentStorage (data durability)
/// 3. Applying command to StateMachine (in-memory state)
/// 4. Only returning success after all steps complete
pub struct Engine {
    wal: WriteAheadLog,
    segments: SegmentStorage,
    state_machine: StateMachine,
    last_applied_id: CommandId,
}

impl Engine {
    /// Create a new engine or recover from existing WAL and segments
    pub fn new<P: AsRef<Path>>(wal_path: P, segments_dir: P) -> Result<Self> {
        let wal_path = wal_path.as_ref();
        let segments_dir = segments_dir.as_ref();

        // Check if we need to recover from existing WAL
        if wal_path.exists() {
            Self::recover(wal_path, segments_dir)
        } else {
            // Fresh start
            let wal = WriteAheadLog::new(wal_path)?;
            let segments = SegmentStorage::new(segments_dir)?;
            let state_machine = StateMachine::new();

            Ok(Engine {
                wal,
                segments,
                state_machine,
                last_applied_id: CommandId(0),
            })
        }
    }

    /// Recover state from existing WAL and segments
    ///
    /// Recovery in order:
    /// 1. Load all entries from segment files (builds index)
    /// 2. Replay WAL commands to get latest state
    /// 3. This ensures consistency between disk and memory
    pub fn recover<P: AsRef<Path>>(wal_path: P, segments_dir: P) -> Result<Self> {
        Self::recover_from_checkpoint(wal_path, segments_dir, None)
    }

    pub fn recover_from_checkpoint<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        checkpoint: Option<(StateMachine, CommandId)>,
    ) -> Result<Self> {
        let wal_path = wal_path.as_ref();
        let segments_dir = segments_dir.as_ref();

        // Load segments (this rebuilds the index from segment files)
        let mut segments = SegmentStorage::new(segments_dir)?;

        // Read all commands from WAL, truncating incomplete/corrupt tail if needed.
        let wal_outcome = WriteAheadLog::read_all_tolerant(wal_path)?;
        let commands = wal_outcome.commands;

        // Create state machine base from checkpoint when available.
        let (mut state_machine, checkpoint_last_applied) =
            if let Some((base_state, last_applied)) = checkpoint {
                (base_state, Some(last_applied))
            } else {
                (StateMachine::new(), None)
            };

        if let Some(last_applied) = checkpoint_last_applied {
            let wal_highest = commands.last().map(|(id, _)| id.0);
            if wal_highest.map(|v| v < last_applied.0).unwrap_or(true) {
                return Err(EngineError::CheckpointWalGap {
                    checkpoint_last_applied: last_applied.0,
                    wal_highest,
                });
            }
            // Conservative behavior: rebuild segments from checkpoint snapshot.
            segments = Self::rebuild_segments_from_state(segments_dir, &state_machine)?;
        }
        let mut repaired_segments = false;

        // Replay all commands in order
        let mut last_id = checkpoint_last_applied.unwrap_or(CommandId(0));
        for (cmd_id, cmd) in commands {
            if let Some(checkpoint_id) = checkpoint_last_applied {
                if cmd_id.0 <= checkpoint_id.0 {
                    continue;
                }
            }
            match &cmd {
                Command::InsertMemory(entry) => {
                    // Ensure segment view converges to WAL command stream.
                    let needs_rewrite = match segments.read_entry(entry.id) {
                        Ok(existing) => existing != *entry,
                        Err(_) => true,
                    };
                    if needs_rewrite {
                        segments.write_entry(entry)?;
                        repaired_segments = true;
                    }
                }
                Command::DeleteMemory(id) => {
                    // Delete may refer to a missing segment entry in crash scenarios.
                    let _ = segments.delete_entry(*id);
                }
                Command::AddEdge { .. } | Command::RemoveEdge { .. } => {}
            }
            state_machine.apply_command(cmd)?;
            last_id = cmd_id;
        }
        if repaired_segments || wal_outcome.truncated {
            segments.fsync()?;
        }

        // Open WAL for appending new commands
        let wal = WriteAheadLog::new(wal_path)?;

        Ok(Engine {
            wal,
            segments,
            state_machine,
            last_applied_id: last_id,
        })
    }

    fn rebuild_segments_from_state<P: AsRef<Path>>(
        segments_dir: P,
        state_machine: &StateMachine,
    ) -> Result<SegmentStorage> {
        let segments_dir = segments_dir.as_ref();
        if segments_dir.exists() {
            std::fs::remove_dir_all(segments_dir)?;
        }
        let mut segments = SegmentStorage::new(segments_dir)?;
        for entry in state_machine.all_memories() {
            segments.write_entry(entry)?;
        }
        segments.fsync()?;
        Ok(segments)
    }

    /// Execute a command with durability guarantees
    ///
    /// Critical order for crash safety:
    /// 1. Write command to WAL
    /// 2. Write data to segments (if applicable)
    /// 3. Sync to disk
    /// 4. Apply to state machine
    ///
    /// This ensures WAL + segments always have the data before it's in memory
    pub fn execute_command(&mut self, cmd: Command) -> Result<CommandId> {
        let cmd_id = self.execute_command_unsynced(cmd)?;
        self.flush()?;
        Ok(cmd_id)
    }

    /// Execute a command without forcing fsync. Caller must invoke `flush` based on policy.
    pub fn execute_command_unsynced(&mut self, cmd: Command) -> Result<CommandId> {
        // 1. Write to WAL first (command logging)
        let cmd_id = self.wal.append(&cmd)?;

        // 2. Handle data persistence based on command type
        match &cmd {
            Command::InsertMemory(entry) => {
                // Write entry to segment storage
                self._write_entry_to_segments(entry)?;
            }
            Command::DeleteMemory(id) => {
                // Mark as deleted in segments
                self.segments.delete_entry(*id)?;
            }
            _ => {
                // AddEdge and RemoveEdge don't need segment writes
                // They're stored in StateMachine's graph
            }
        }

        // 3. Apply to state machine.
        // In strict mode this is followed by immediate `flush()`.
        // In relaxed modes caller flushes later via sync policy.
        self.state_machine.apply_command(cmd)?;

        // 5. Update tracking
        self.last_applied_id = cmd_id;

        Ok(cmd_id)
    }

    /// Force WAL + segment data to durable storage.
    pub fn flush(&mut self) -> Result<()> {
        self.wal.fsync()?;
        self.segments.fsync()?;
        Ok(())
    }

    /// Execute a command, then enforce capacity in the same deterministic command pipeline.
    pub fn execute_command_with_capacity(
        &mut self,
        cmd: Command,
        policy: CapacityPolicy,
    ) -> Result<(CommandId, EvictionReport)> {
        let cmd_id = self.execute_command(cmd)?;
        let report = self.enforce_capacity(policy)?;
        Ok((cmd_id, report))
    }

    /// Enforce configured capacity by deterministically evicting memories.
    ///
    /// Evictions are always executed through `Command::DeleteMemory`, which ensures:
    /// - WAL is appended for every eviction
    /// - segment tombstones are updated
    /// - deterministic replay behavior after recovery
    pub fn enforce_capacity(&mut self, policy: CapacityPolicy) -> Result<EvictionReport> {
        self.enforce_capacity_with_sync(policy, true)
    }

    /// Capacity enforcement without immediate fsync.
    /// Caller must call `flush()` according to sync policy.
    pub fn enforce_capacity_unsynced(&mut self, policy: CapacityPolicy) -> Result<EvictionReport> {
        self.enforce_capacity_with_sync(policy, false)
    }

    fn enforce_capacity_with_sync(
        &mut self,
        policy: CapacityPolicy,
        sync_immediately: bool,
    ) -> Result<EvictionReport> {
        let (entries_before, bytes_before) = self.current_usage();
        let mut evicted_ids = Vec::new();

        if !self.exceeds_capacity(entries_before, bytes_before, policy) {
            return Ok(EvictionReport {
                evicted_ids,
                entries_before,
                entries_after: entries_before,
                bytes_before,
                bytes_after: bytes_before,
            });
        }

        // Deterministic ordering:
        // 1) older first (created_at asc)
        // 2) less important first (importance asc)
        // 3) lower id first
        let mut unique_ids = std::collections::HashSet::new();
        let mut candidates: Vec<(MemoryId, u64, f32)> = self
            .state_machine
            .all_memories()
            .into_iter()
            .filter(|entry| unique_ids.insert(entry.id))
            .map(|entry| (entry.id, entry.created_at, entry.importance))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.cmp(&b.0))
        });

        for (id, _, _) in candidates {
            let (entries_now, bytes_now) = self.current_usage();
            if !self.exceeds_capacity(entries_now, bytes_now, policy) {
                break;
            }

            if sync_immediately {
                self.execute_command(Command::DeleteMemory(id))?;
            } else {
                self.execute_command_unsynced(Command::DeleteMemory(id))?;
            }
            evicted_ids.push(id);
        }

        let (entries_after, bytes_after) = self.current_usage();
        Ok(EvictionReport {
            evicted_ids,
            entries_before,
            entries_after,
            bytes_before,
            bytes_after,
        })
    }

    /// Return current live memory count and deterministic byte estimate.
    pub fn current_usage(&self) -> (usize, u64) {
        let entries = self.state_machine.len();
        let bytes = self
            .state_machine
            .all_memories()
            .into_iter()
            .map(Self::estimate_memory_bytes)
            .sum();
        (entries, bytes)
    }

    fn exceeds_capacity(&self, entries: usize, bytes: u64, policy: CapacityPolicy) -> bool {
        if let Some(max_entries) = policy.max_entries {
            if entries > max_entries {
                return true;
            }
        }
        if let Some(max_bytes) = policy.max_bytes {
            if bytes > max_bytes {
                return true;
            }
        }
        false
    }

    fn estimate_memory_bytes(entry: &MemoryEntry) -> u64 {
        let namespace_bytes = entry.namespace.len() as u64;
        let content_bytes = entry.content.len() as u64;
        let embedding_bytes = entry
            .embedding
            .as_ref()
            .map(|v| (v.len() as u64) * 4)
            .unwrap_or(0);
        let metadata_bytes: u64 = entry
            .metadata
            .iter()
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum();
        namespace_bytes + content_bytes + embedding_bytes + metadata_bytes
    }

    /// Helper: Write entry to segments
    fn _write_entry_to_segments(
        &mut self,
        entry: &crate::core::memory_entry::MemoryEntry,
    ) -> Result<()> {
        self.segments.write_entry(entry)?;
        Ok(())
    }

    /// Get reference to the state machine (read-only)
    pub fn get_state_machine(&self) -> &StateMachine {
        &self.state_machine
    }

    /// Get mutable reference to the state machine
    /// NOTE: If you modify state directly (not via execute_command),
    /// you bypass WAL durability! Use execute_command() instead.
    pub fn get_state_machine_mut(&mut self) -> &mut StateMachine {
        &mut self.state_machine
    }

    /// Get reference to segments
    pub fn get_segments(&self) -> &SegmentStorage {
        &self.segments
    }

    /// Get last applied command ID
    pub fn last_applied_id(&self) -> CommandId {
        self.last_applied_id
    }

    /// Get number of commands in WAL
    pub fn wal_len(&self) -> u64 {
        self.wal.len()
    }

    /// Check if engine is empty (no commands)
    pub fn is_empty(&self) -> bool {
        self.wal.is_empty()
    }

    /// Compact segments on disk and rebuild segment index.
    pub fn compact_segments(&mut self) -> Result<crate::storage::compaction::CompactionReport> {
        self.segments.prepare_for_maintenance()?;
        let dir = self.segments.data_dir().to_path_buf();
        let report = crate::storage::compaction::compact_segment_dir(&dir)?;
        self.segments = SegmentStorage::new(&dir)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryEntry, MemoryId};
    use tempfile::TempDir;

    #[test]
    fn test_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        let engine = Engine::new(&wal_path, &seg_dir).unwrap();
        assert!(engine.is_empty());
        assert_eq!(engine.wal_len(), 0);
    }

    #[test]
    fn test_engine_execute_command() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

        let entry = MemoryEntry::new(MemoryId(1), "test".to_string(), b"content".to_vec(), 1000);

        let cmd = Command::InsertMemory(entry);
        let cmd_id = engine.execute_command(cmd).unwrap();

        assert_eq!(cmd_id, CommandId(0));
        assert_eq!(engine.wal_len(), 1);
        assert_eq!(engine.get_state_machine().len(), 1);
    }

    #[test]
    fn test_capacity_eviction_by_max_entries() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

        // Lower importance + older entries should be evicted first.
        let entries = vec![
            MemoryEntry::new(MemoryId(1), "ns".to_string(), b"a".to_vec(), 1000)
                .with_importance(0.1),
            MemoryEntry::new(MemoryId(2), "ns".to_string(), b"b".to_vec(), 2000)
                .with_importance(0.9),
            MemoryEntry::new(MemoryId(3), "ns".to_string(), b"c".to_vec(), 3000)
                .with_importance(0.2),
        ];
        for entry in entries {
            engine
                .execute_command(Command::InsertMemory(entry))
                .unwrap();
        }
        let wal_before = engine.wal_len();

        let report = engine
            .enforce_capacity(CapacityPolicy::new(Some(2), None))
            .unwrap();

        assert_eq!(report.entries_before, 3);
        assert_eq!(report.entries_after, 2);
        assert_eq!(report.evicted_ids, vec![MemoryId(1)]);
        assert_eq!(engine.wal_len(), wal_before + 1);
        assert!(engine.get_state_machine().get_memory(MemoryId(1)).is_err());
        assert!(engine.get_state_machine().get_memory(MemoryId(2)).is_ok());
        assert!(engine.get_state_machine().get_memory(MemoryId(3)).is_ok());
    }

    #[test]
    fn test_capacity_eviction_by_max_bytes() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

        engine
            .execute_command(Command::InsertMemory(
                MemoryEntry::new(MemoryId(1), "ns".to_string(), vec![1; 40], 1000)
                    .with_importance(0.1),
            ))
            .unwrap();
        engine
            .execute_command(Command::InsertMemory(
                MemoryEntry::new(MemoryId(2), "ns".to_string(), vec![2; 40], 2000)
                    .with_importance(0.9),
            ))
            .unwrap();

        let (_, bytes_before) = engine.current_usage();
        let report = engine
            .enforce_capacity(CapacityPolicy::new(None, Some(50)))
            .unwrap();

        assert!(bytes_before > 50);
        assert!(report.bytes_after <= 50);
        assert_eq!(report.evicted_ids, vec![MemoryId(1)]);
        assert!(engine.get_state_machine().get_memory(MemoryId(1)).is_err());
        assert!(engine.get_state_machine().get_memory(MemoryId(2)).is_ok());
    }

    #[test]
    fn test_capacity_eviction_persists_via_wal_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        {
            let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();
            for i in 0..3 {
                let entry = MemoryEntry::new(
                    MemoryId(i as u64),
                    "ns".to_string(),
                    format!("payload_{}", i).into_bytes(),
                    1000 + i as u64,
                )
                .with_importance(i as f32);
                engine
                    .execute_command(Command::InsertMemory(entry))
                    .unwrap();
            }

            let report = engine
                .enforce_capacity(CapacityPolicy::new(Some(1), None))
                .unwrap();
            assert_eq!(report.entries_after, 1);
            assert_eq!(report.evicted_ids.len(), 2);
        }

        let recovered = Engine::recover(&wal_path, &seg_dir).unwrap();
        assert_eq!(recovered.get_state_machine().len(), 1);
        assert!(
            recovered
                .get_state_machine()
                .get_memory(MemoryId(2))
                .is_ok()
        );
        assert!(
            recovered
                .get_state_machine()
                .get_memory(MemoryId(0))
                .is_err()
        );
        assert!(
            recovered
                .get_state_machine()
                .get_memory(MemoryId(1))
                .is_err()
        );
    }

    #[test]
    fn test_execute_command_with_capacity() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");
        let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

        engine
            .execute_command(Command::InsertMemory(MemoryEntry::new(
                MemoryId(1),
                "ns".to_string(),
                b"a".to_vec(),
                1000,
            )))
            .unwrap();

        let (_cmd_id, report) = engine
            .execute_command_with_capacity(
                Command::InsertMemory(MemoryEntry::new(
                    MemoryId(2),
                    "ns".to_string(),
                    b"b".to_vec(),
                    2000,
                )),
                CapacityPolicy::new(Some(1), None),
            )
            .unwrap();

        assert_eq!(engine.get_state_machine().len(), 1);
        assert_eq!(report.evicted_ids, vec![MemoryId(1)]);
    }

    #[test]
    fn test_engine_compact_segments() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");
        let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

        for i in 0..10 {
            let entry = MemoryEntry::new(
                MemoryId(i as u64),
                "ns".to_string(),
                format!("content_{}", i).into_bytes(),
                1000 + i as u64,
            );
            engine
                .execute_command(Command::InsertMemory(entry))
                .unwrap();
        }
        for id in [0_u64, 1, 2, 3] {
            engine
                .execute_command(Command::DeleteMemory(MemoryId(id)))
                .unwrap();
        }

        let report = engine.compact_segments().unwrap();
        // We only assert compaction ran safely; eligibility depends on segment rotation history.
        assert_eq!(engine.get_state_machine().len(), 6);
        if report.live_entries_rewritten > 0 {
            assert!(!report.compacted_segments.is_empty());
        }
    }

    #[test]
    fn test_engine_multiple_commands() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

        // Insert 10 memories
        for i in 0..10 {
            let entry = MemoryEntry::new(
                MemoryId(i as u64),
                "test".to_string(),
                format!("content_{}", i).into_bytes(),
                1000 + i as u64,
            );
            let cmd = Command::InsertMemory(entry);
            engine.execute_command(cmd).unwrap();
        }

        assert_eq!(engine.wal_len(), 10);
        assert_eq!(engine.get_state_machine().len(), 10);

        // Add some edges
        for i in 0..9 {
            let cmd = Command::AddEdge {
                from: MemoryId(i as u64),
                to: MemoryId((i + 1) as u64),
                relation: "follows".to_string(),
            };
            engine.execute_command(cmd).unwrap();
        }

        assert_eq!(engine.wal_len(), 19);
    }

    #[test]
    fn test_recover_from_checkpoint_detects_wal_gap() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine_gap.wal");
        let seg_dir = temp_dir.path().join("segments_gap");

        {
            let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();
            let entry = MemoryEntry::new(MemoryId(1), "ns".to_string(), b"a".to_vec(), 1000);
            engine
                .execute_command(Command::InsertMemory(entry))
                .unwrap();
        }

        let state = StateMachine::new();
        let result =
            Engine::recover_from_checkpoint(&wal_path, &seg_dir, Some((state, CommandId(10))));
        assert!(matches!(result, Err(EngineError::CheckpointWalGap { .. })));
    }

    #[test]
    fn test_recovery_from_crash() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        // Execute commands
        {
            let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

            for i in 0..5 {
                let entry = MemoryEntry::new(
                    MemoryId(i as u64),
                    "namespace".to_string(),
                    format!("memory_{}", i).into_bytes(),
                    2000 + i as u64,
                );
                let cmd = Command::InsertMemory(entry);
                engine.execute_command(cmd).unwrap();
            }

            // Simulate crash: drop engine without cleanup
        }

        // Recover from WAL
        let engine = Engine::recover(&wal_path, &seg_dir).unwrap();

        assert_eq!(engine.wal_len(), 5);
        assert_eq!(engine.get_state_machine().len(), 5);

        // Verify data is intact
        let memory = engine.get_state_machine().get_memory(MemoryId(0)).unwrap();
        assert_eq!(memory.id, MemoryId(0));
        assert_eq!(memory.namespace, "namespace");
    }

    #[test]
    fn test_recovery_preserves_order() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        // Create initial state
        {
            let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

            // Insert 3 memories
            for i in 0..3 {
                let entry = MemoryEntry::new(
                    MemoryId(i as u64),
                    "ns".to_string(),
                    b"data".to_vec(),
                    1000 + i as u64,
                );
                engine
                    .execute_command(Command::InsertMemory(entry))
                    .unwrap();
            }

            // Add edges in specific order
            engine
                .execute_command(Command::AddEdge {
                    from: MemoryId(0),
                    to: MemoryId(1),
                    relation: "points_to".to_string(),
                })
                .unwrap();

            engine
                .execute_command(Command::AddEdge {
                    from: MemoryId(1),
                    to: MemoryId(2),
                    relation: "points_to".to_string(),
                })
                .unwrap();
        }

        // Recover and verify edges are in order
        let recovered = Engine::recover(&wal_path, &seg_dir).unwrap();

        let neighbors_0 = recovered
            .get_state_machine()
            .get_neighbors(MemoryId(0))
            .unwrap();
        assert_eq!(neighbors_0.len(), 1);
        assert_eq!(neighbors_0[0].0, MemoryId(1));

        let neighbors_1 = recovered
            .get_state_machine()
            .get_neighbors(MemoryId(1))
            .unwrap();
        assert_eq!(neighbors_1.len(), 1);
        assert_eq!(neighbors_1[0].0, MemoryId(2));
    }

    #[test]
    fn test_recovery_100_commands() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        // Execute 100 commands
        {
            let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

            for i in 0..100 {
                let entry = MemoryEntry::new(
                    MemoryId(i as u64),
                    "test".to_string(),
                    format!("content_{}", i).into_bytes(),
                    3000 + i as u64,
                );
                let cmd = Command::InsertMemory(entry);
                engine.execute_command(cmd).unwrap();
            }

            assert_eq!(engine.wal_len(), 100);
            assert_eq!(engine.get_state_machine().len(), 100);
        }

        // Recover - should rebuild entire state
        let recovered = Engine::recover(&wal_path, &seg_dir).unwrap();

        assert_eq!(recovered.wal_len(), 100);
        assert_eq!(recovered.get_state_machine().len(), 100);

        // Verify specific entries
        for i in [0, 25, 50, 75, 99] {
            let memory = recovered
                .get_state_machine()
                .get_memory(MemoryId(i as u64))
                .unwrap();
            assert_eq!(memory.id, MemoryId(i as u64));
        }
    }

    #[test]
    fn test_recovery_with_mixed_commands() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");

        // Create state with mixed operations
        {
            let mut engine = Engine::new(&wal_path, &seg_dir).unwrap();

            // Insert memories
            for i in 0..5 {
                let entry =
                    MemoryEntry::new(MemoryId(i as u64), "ns".to_string(), b"data".to_vec(), 1000);
                engine
                    .execute_command(Command::InsertMemory(entry))
                    .unwrap();
            }

            // Add edges
            for i in 0..4 {
                engine
                    .execute_command(Command::AddEdge {
                        from: MemoryId(i as u64),
                        to: MemoryId((i + 1) as u64),
                        relation: "next".to_string(),
                    })
                    .unwrap();
            }

            // Delete one memory
            engine
                .execute_command(Command::DeleteMemory(MemoryId(2)))
                .unwrap();

            assert_eq!(engine.get_state_machine().len(), 4); // 5 - 1
        }

        // Recover
        let recovered = Engine::recover(&wal_path, &seg_dir).unwrap();

        assert_eq!(recovered.get_state_machine().len(), 4);
        assert!(
            recovered
                .get_state_machine()
                .get_memory(MemoryId(2))
                .is_err()
        ); // Should be deleted
        assert!(
            recovered
                .get_state_machine()
                .get_memory(MemoryId(0))
                .is_ok()
        ); // Should exist
    }
}
