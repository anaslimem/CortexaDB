use super::memory_entry::MemoryEntry;
use crate::core::memory_entry::MemoryId;
use serde::{Deserialize, Serialize};

/// State-mutating commands for the state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Insert or update a memory entry
    InsertMemory(MemoryEntry),
    /// Delete a memory entry by ID
    DeleteMemory(MemoryId),
    /// Add an edge between two memories with a relation type
    AddEdge {
        from: MemoryId,
        to: MemoryId,
        relation: String,
    },
    /// Remove an edge between two memories
    RemoveEdge { from: MemoryId, to: MemoryId },
}

impl Command {
    pub fn insert_memory(entry: MemoryEntry) -> Self {
        Command::InsertMemory(entry)
    }

    pub fn delete_memory(id: MemoryId) -> Self {
        Command::DeleteMemory(id)
    }

    pub fn add_edge(from: MemoryId, to: MemoryId, relation: String) -> Self {
        Command::AddEdge { from, to, relation }
    }

    pub fn remove_edge(from: MemoryId, to: MemoryId) -> Self {
        Command::RemoveEdge { from, to }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_command() {
        let entry = MemoryEntry::new(MemoryId(1), "test".to_string(), b"data".to_vec(), 1000);
        let cmd = Command::insert_memory(entry.clone());
        match cmd {
            Command::InsertMemory(e) => assert_eq!(e.id, MemoryId(1)),
            _ => panic!("Expected InsertMemory"),
        }
    }

    #[test]
    fn test_delete_command() {
        let cmd = Command::delete_memory(MemoryId(1));
        match cmd {
            Command::DeleteMemory(id) => assert_eq!(id, MemoryId(1)),
            _ => panic!("Expected DeleteMemory"),
        }
    }

    #[test]
    fn test_edge_commands() {
        let add_cmd = Command::add_edge(MemoryId(1), MemoryId(2), "refers_to".to_string());
        let remove_cmd = Command::remove_edge(MemoryId(1), MemoryId(2));

        match add_cmd {
            Command::AddEdge { from, to, relation } => {
                assert_eq!(from, MemoryId(1));
                assert_eq!(to, MemoryId(2));
                assert_eq!(relation, "refers_to");
            }
            _ => panic!("Expected AddEdge"),
        }

        match remove_cmd {
            Command::RemoveEdge { from, to } => {
                assert_eq!(from, MemoryId(1));
                assert_eq!(to, MemoryId(2));
            }
            _ => panic!("Expected RemoveEdge"),
        }
    }

    #[test]
    fn test_command_serialization() {
        let entry = MemoryEntry::new(MemoryId(42), "ns".to_string(), b"content".to_vec(), 5000);
        let cmd = Command::insert_memory(entry);

        let serialized = bincode::serialize(&cmd).expect("serialization failed");
        let deserialized: Command =
            bincode::deserialize(&serialized).expect("deserialization failed");

        match deserialized {
            Command::InsertMemory(e) => assert_eq!(e.id, MemoryId(42)),
            _ => panic!("Expected InsertMemory"),
        }
    }
}
