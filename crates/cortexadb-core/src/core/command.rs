use serde::{Deserialize, Serialize};

use super::memory_entry::MemoryEntry;
use crate::core::memory_entry::MemoryId;

/// State-mutating commands for the state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Insert or update a memory entry
    Add(MemoryEntry),
    /// Delete a memory entry by ID
    Delete(MemoryId),
    /// Add an edge between two memories with a relation type
    Connect { from: MemoryId, to: MemoryId, relation: String },
    /// Remove an edge between two memories
    Disconnect { from: MemoryId, to: MemoryId },
}

impl Command {
    pub fn add(entry: MemoryEntry) -> Self {
        Command::Add(entry)
    }

    pub fn delete(id: MemoryId) -> Self {
        Command::Delete(id)
    }

    pub fn connect(from: MemoryId, to: MemoryId, relation: String) -> Self {
        Command::Connect { from, to, relation }
    }

    pub fn disconnect(from: MemoryId, to: MemoryId) -> Self {
        Command::Disconnect { from, to }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_command() {
        let entry = MemoryEntry::new(MemoryId(1), "test".to_string(), b"data".to_vec(), 1000);
        let cmd = Command::add(entry.clone());
        match cmd {
            Command::Add(e) => assert_eq!(e.id, MemoryId(1)),
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_delete_command() {
        let cmd = Command::delete(MemoryId(1));
        match cmd {
            Command::Delete(id) => assert_eq!(id, MemoryId(1)),
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_edge_commands() {
        let add_cmd = Command::connect(MemoryId(1), MemoryId(2), "refers_to".to_string());
        let remove_cmd = Command::disconnect(MemoryId(1), MemoryId(2));

        match add_cmd {
            Command::Connect { from, to, relation } => {
                assert_eq!(from, MemoryId(1));
                assert_eq!(to, MemoryId(2));
                assert_eq!(relation, "refers_to");
            }
            _ => panic!("Expected Connect"),
        }

        match remove_cmd {
            Command::Disconnect { from, to } => {
                assert_eq!(from, MemoryId(1));
                assert_eq!(to, MemoryId(2));
            }
            _ => panic!("Expected Disconnect"),
        }
    }

    #[test]
    fn test_command_serialization() {
        let entry = MemoryEntry::new(MemoryId(42), "ns".to_string(), b"content".to_vec(), 5000);
        let cmd = Command::add(entry);

        let serialized = bincode::serialize(&cmd).expect("serialization failed");
        let deserialized: Command =
            bincode::deserialize(&serialized).expect("deserialization failed");

        match deserialized {
            Command::Add(e) => assert_eq!(e.id, MemoryId(42)),
            _ => panic!("Expected Add"),
        }
    }
}
