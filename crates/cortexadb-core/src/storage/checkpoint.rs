use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::state_machine::StateMachine;

#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Unsupported checkpoint version: {0}")]
    UnsupportedVersion(u32),
}

pub type Result<T> = std::result::Result<T, CheckpointError>;

#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    pub last_applied_id: u64,
    pub state_machine: StateMachine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    version: u32,
    last_applied_id: u64,
    state_machine: StateMachine,
}

const CURRENT_VERSION: u32 = 1;

pub fn checkpoint_path_from_wal<P: AsRef<Path>>(wal_path: P) -> PathBuf {
    wal_path.as_ref().with_extension("ckpt")
}

pub fn save_checkpoint<P: AsRef<Path>>(
    path: P,
    state_machine: &StateMachine,
    last_applied_id: u64,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let data = CheckpointData {
        version: CURRENT_VERSION,
        last_applied_id,
        state_machine: state_machine.clone(),
    };
    let bytes = bincode::serialize(&data)?;

    let tmp_path = path.with_extension("ckpt.tmp");
    {
        let mut file =
            OpenOptions::new().create(true).write(true).truncate(true).open(&tmp_path)?;
        file.write_all(&bytes)?;
        file.flush()?;
        file.sync_all()?;
    }

    std::fs::rename(&tmp_path, path)?;

    if let Some(parent) = path.parent() {
        if let Ok(dir) = OpenOptions::new().read(true).open(parent) {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Option<LoadedCheckpoint>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }

    let bytes = std::fs::read(path)?;
    let data: CheckpointData = bincode::deserialize(&bytes)?;
    if data.version != CURRENT_VERSION {
        return Err(CheckpointError::UnsupportedVersion(data.version));
    }

    Ok(Some(LoadedCheckpoint {
        last_applied_id: data.last_applied_id,
        state_machine: data.state_machine,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryEntry, MemoryId};
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_roundtrip() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("state.ckpt");

        let mut state = StateMachine::new();
        state
            .insert_memory(MemoryEntry::new(MemoryId(1), "ns".to_string(), b"hello".to_vec(), 1000))
            .unwrap();

        save_checkpoint(&path, &state, 42).unwrap();
        let loaded = load_checkpoint(&path).unwrap().unwrap();

        assert_eq!(loaded.last_applied_id, 42);
        assert_eq!(loaded.state_machine.len(), 1);
        assert!(loaded.state_machine.get_memory(MemoryId(1)).is_ok());
    }
}
