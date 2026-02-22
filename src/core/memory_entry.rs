use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a memory entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryId(pub u64);

/// Core memory entry structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: MemoryId,
    pub namespace: String,
    pub content: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
    pub importance: f32,
}

impl MemoryEntry {
    pub fn new(id: MemoryId, namespace: String, content: Vec<u8>, created_at: u64) -> Self {
        Self {
            id,
            namespace,
            content,
            embedding: None,
            metadata: HashMap::new(),
            created_at,
            importance: 0.0,
        }
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new(
            MemoryId(1),
            "default".to_string(),
            b"test content".to_vec(),
            1000,
        );
        assert_eq!(entry.id, MemoryId(1));
        assert_eq!(entry.namespace, "default");
        assert_eq!(entry.importance, 0.0);
        assert_eq!(entry.embedding, None);
    }

    #[test]
    fn test_memory_entry_builder() {
        let entry = MemoryEntry::new(MemoryId(1), "default".to_string(), b"test".to_vec(), 1000)
            .with_importance(0.8)
            .with_embedding(vec![0.1, 0.2, 0.3]);

        assert_eq!(entry.importance, 0.8);
        assert_eq!(entry.embedding, Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_memory_id_ordering() {
        let id1 = MemoryId(1);
        let id2 = MemoryId(2);
        assert!(id1 < id2);
    }
}
