use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

use super::command::Command;
use super::memory_entry::{MemoryEntry, MemoryId};

#[derive(Error, Debug)]
pub enum StateMachineError {
    #[error("Memory not found: {0:?}")]
    MemoryNotFound(MemoryId),
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

pub type Result<T> = std::result::Result<T, StateMachineError>;

/// Edge in the memory graph with associated relation type
#[derive(Debug, Clone)]
pub struct Edge {
    pub to: MemoryId,
    pub relation: String,
}

/// Deterministic state machine that applies commands and maintains memory state
pub struct StateMachine {
    /// All memory entries indexed by ID
    memories: HashMap<MemoryId, MemoryEntry>,
    /// Graph adjacency list: from_id -> [(to_id, relation)]
    graph: HashMap<MemoryId, Vec<Edge>>,
    /// Temporal index: timestamp -> list of memory IDs (sorted for determinism)
    temporal_index: BTreeMap<u64, Vec<MemoryId>>,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
            graph: HashMap::new(),
            temporal_index: BTreeMap::new(),
        }
    }

    /// Apply a command to the state machine
    pub fn apply_command(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::InsertMemory(entry) => self.insert_memory(entry),
            Command::DeleteMemory(id) => self.delete_memory(id),
            Command::AddEdge { from, to, relation } => self.add_edge(from, to, relation),
            Command::RemoveEdge { from, to } => self.remove_edge(from, to),
        }
    }

    /// Insert or update a memory entry
    pub fn insert_memory(&mut self, entry: MemoryEntry) -> Result<()> {
        let id = entry.id;
        let timestamp = entry.created_at;

        self.memories.insert(id, entry);

        // Add to temporal index
        self.temporal_index
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .push(id);

        // Keep temporal index sorted for determinism
        if let Some(ids) = self.temporal_index.get_mut(&timestamp) {
            ids.sort();
            ids.dedup();
        }

        Ok(())
    }

    /// Delete a memory entry and its edges
    pub fn delete_memory(&mut self, id: MemoryId) -> Result<()> {
        if !self.memories.contains_key(&id) {
            return Err(StateMachineError::MemoryNotFound(id));
        }

        self.memories.remove(&id);

        // Remove from graph
        self.graph.remove(&id);
        for edges in self.graph.values_mut() {
            edges.retain(|e| e.to != id);
        }

        // Remove from temporal index
        for ids in self.temporal_index.values_mut() {
            ids.retain(|&mid| mid != id);
        }

        Ok(())
    }

    /// Add an edge between two memories
    pub fn add_edge(&mut self, from: MemoryId, to: MemoryId, relation: String) -> Result<()> {
        if !self.memories.contains_key(&from) {
            return Err(StateMachineError::MemoryNotFound(from));
        }
        if !self.memories.contains_key(&to) {
            return Err(StateMachineError::MemoryNotFound(to));
        }

        let edges = self.graph.entry(from).or_insert_with(Vec::new);
        // Avoid duplicate edges
        if !edges.iter().any(|e| e.to == to && e.relation == relation) {
            edges.push(Edge { to, relation });
            // Keep edges sorted for determinism
            edges.sort_by_key(|e| (e.to, e.relation.clone()));
        }

        Ok(())
    }

    /// Remove an edge between two memories
    pub fn remove_edge(&mut self, from: MemoryId, to: MemoryId) -> Result<()> {
        if let Some(edges) = self.graph.get_mut(&from) {
            edges.retain(|e| e.to != to);
        }
        Ok(())
    }

    /// Get a memory by ID
    pub fn get_memory(&self, id: MemoryId) -> Result<&MemoryEntry> {
        self.memories
            .get(&id)
            .ok_or(StateMachineError::MemoryNotFound(id))
    }

    /// Get all memories in a namespace
    pub fn get_memories_in_namespace(&self, namespace: &str) -> Vec<&MemoryEntry> {
        let mut entries: Vec<_> = self
            .memories
            .values()
            .filter(|e| e.namespace == namespace)
            .collect();
        entries.sort_by_key(|e| e.id);
        entries
    }

    /// Get memories created in a time range (inclusive)
    pub fn get_memories_in_time_range(&self, start: u64, end: u64) -> Vec<&MemoryEntry> {
        let mut result = Vec::new();
        for (_, ids) in self.temporal_index.range(start..=end) {
            for id in ids {
                if let Some(entry) = self.memories.get(id) {
                    result.push(entry);
                }
            }
        }
        result.sort_by_key(|e| e.id);
        result
    }

    /// Get neighbors of a memory (memories it points to)
    pub fn get_neighbors(&self, id: MemoryId) -> Result<Vec<(MemoryId, String)>> {
        let edges = self.graph.get(&id).ok_or(StateMachineError::MemoryNotFound(id))?;
        let mut neighbors: Vec<_> = edges.iter().map(|e| (e.to, e.relation.clone())).collect();
        neighbors.sort_by_key(|n| (n.0, n.1.clone()));
        Ok(neighbors)
    }

    /// Get size of state
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(id: u64, namespace: &str, timestamp: u64) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId(id),
            namespace.to_string(),
            format!("content_{}", id).into_bytes(),
            timestamp,
        )
    }

    #[test]
    fn test_state_machine_creation() {
        let sm = StateMachine::new();
        assert!(sm.is_empty());
        assert_eq!(sm.len(), 0);
    }

    #[test]
    fn test_insert_and_retrieve_memory() {
        let mut sm = StateMachine::new();
        let entry = create_test_entry(1, "default", 1000);
        sm.insert_memory(entry.clone()).unwrap();

        assert_eq!(sm.len(), 1);
        let retrieved = sm.get_memory(MemoryId(1)).unwrap();
        assert_eq!(retrieved.id, MemoryId(1));
    }

    #[test]
    fn test_delete_memory() {
        let mut sm = StateMachine::new();
        let entry = create_test_entry(1, "default", 1000);
        sm.insert_memory(entry).unwrap();
        assert_eq!(sm.len(), 1);

        sm.delete_memory(MemoryId(1)).unwrap();
        assert_eq!(sm.len(), 0);
        assert!(sm.get_memory(MemoryId(1)).is_err());
    }

    #[test]
    fn test_add_and_remove_edges() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_test_entry(1, "default", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(2, "default", 1000))
            .unwrap();

        sm.add_edge(MemoryId(1), MemoryId(2), "refers_to".to_string())
            .unwrap();

        let neighbors = sm.get_neighbors(MemoryId(1)).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, MemoryId(2));
        assert_eq!(neighbors[0].1, "refers_to");

        sm.remove_edge(MemoryId(1), MemoryId(2)).unwrap();
        let neighbors = sm.get_neighbors(MemoryId(1)).unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_temporal_index() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_test_entry(1, "default", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(2, "default", 2000))
            .unwrap();
        sm.insert_memory(create_test_entry(3, "default", 1500))
            .unwrap();

        let range = sm.get_memories_in_time_range(1000, 1500);
        assert_eq!(range.len(), 2);
        // Check deterministic ordering
        assert_eq!(range[0].id, MemoryId(1));
        assert_eq!(range[1].id, MemoryId(3));
    }

    #[test]
    fn test_namespace_filtering() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_test_entry(1, "ns1", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(2, "ns2", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(3, "ns1", 1000))
            .unwrap();

        let ns1_entries = sm.get_memories_in_namespace("ns1");
        assert_eq!(ns1_entries.len(), 2);
        assert_eq!(ns1_entries[0].id, MemoryId(1));
        assert_eq!(ns1_entries[1].id, MemoryId(3));
    }

    #[test]
    fn test_edge_not_found() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_test_entry(1, "default", 1000))
            .unwrap();

        // Try to add edge to non-existent memory
        let result = sm.add_edge(MemoryId(1), MemoryId(999), "refers".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_command() {
        let mut sm = StateMachine::new();
        let entry = create_test_entry(1, "default", 1000);
        let cmd = Command::InsertMemory(entry);
        sm.apply_command(cmd).unwrap();
        assert_eq!(sm.len(), 1);
    }

    #[test]
    fn test_deterministic_edge_ordering() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_test_entry(1, "default", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(2, "default", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(3, "default", 1000))
            .unwrap();

        // Add edges in different order
        sm.add_edge(MemoryId(1), MemoryId(3), "rel".to_string())
            .unwrap();
        sm.add_edge(MemoryId(1), MemoryId(2), "rel".to_string())
            .unwrap();

        let neighbors = sm.get_neighbors(MemoryId(1)).unwrap();
        assert_eq!(neighbors[0].0, MemoryId(2)); // Deterministically ordered
        assert_eq!(neighbors[1].0, MemoryId(3));
    }

    #[test]
    fn test_delete_cleans_edges() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_test_entry(1, "default", 1000))
            .unwrap();
        sm.insert_memory(create_test_entry(2, "default", 1000))
            .unwrap();

        sm.add_edge(MemoryId(1), MemoryId(2), "refers".to_string())
            .unwrap();

        // Delete memory 2
        sm.delete_memory(MemoryId(2)).unwrap();

        // Edge should be cleaned up
        let neighbors = sm.get_neighbors(MemoryId(1)).unwrap();
        assert!(neighbors.is_empty());
    }
}
