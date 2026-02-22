use std::path::Path;
use thiserror::Error;

use crate::core::command::Command;
use crate::core::state_machine::StateMachine;
use crate::storage::wal::{WriteAheadLog, CommandId};
use crate::storage::segment::SegmentStorage;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("WAL error: {0}")]
    WalError(#[from] crate::storage::wal::WalError),
    #[error("State machine error: {0}")]
    StateMachineError(#[from] crate::core::state_machine::StateMachineError),
    #[error("Segment error: {0}")]
    SegmentError(#[from] crate::storage::segment::SegmentError),
    #[error("Engine not recovered properly")]
    NotRecovered,
}

pub type Result<T> = std::result::Result<T, EngineError>;

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
        let wal_path = wal_path.as_ref();
        let segments_dir = segments_dir.as_ref();
        
        // Load segments (this rebuilds the index from segment files)
        let segments = SegmentStorage::new(segments_dir)?;
        
        // Read all commands from WAL
        let commands = WriteAheadLog::read_all(wal_path)?;

        // Create fresh state machine and replay commands
        let mut state_machine = StateMachine::new();

        // Replay all commands in order
        let mut last_id = CommandId(0);
        for (cmd_id, cmd) in commands {
            state_machine.apply_command(cmd)?;
            last_id = cmd_id;
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
        
        // 3. Sync to disk (durability guarantee)
        self.wal.fsync()?;
        self.segments.fsync()?;

        // 4. Apply to state machine (if crash after this, WAL + segments have it)
        self.state_machine.apply_command(cmd)?;

        // 5. Update tracking
        self.last_applied_id = cmd_id;

        Ok(cmd_id)
    }

    /// Helper: Write entry to segments
    fn _write_entry_to_segments(&mut self, entry: &crate::core::memory_entry::MemoryEntry) -> Result<()> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryId, MemoryEntry};
    use tempfile::TempDir;

    #[test]
    fn test_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("engine.wal");
        let seg_dir = temp_dir.path().join("segments");
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

        let entry = MemoryEntry::new(
            MemoryId(1),
            "test".to_string(),
            b"content".to_vec(),
            1000,
        );

        let cmd = Command::InsertMemory(entry);
        let cmd_id = engine.execute_command(cmd).unwrap();

        assert_eq!(cmd_id, CommandId(0));
        assert_eq!(engine.wal_len(), 1);
        assert_eq!(engine.get_state_machine().len(), 1);
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
        let memory = engine
            .get_state_machine()
            .get_memory(MemoryId(0))
            .unwrap();
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
                engine.execute_command(Command::InsertMemory(entry)).unwrap();
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
                let entry = MemoryEntry::new(MemoryId(i as u64), "ns".to_string(), b"data".to_vec(), 1000);
                engine.execute_command(Command::InsertMemory(entry)).unwrap();
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
        assert!(recovered
            .get_state_machine()
            .get_memory(MemoryId(2))
            .is_err()); // Should be deleted
        assert!(recovered
            .get_state_machine()
            .get_memory(MemoryId(0))
            .is_ok()); // Should exist
    }
}
