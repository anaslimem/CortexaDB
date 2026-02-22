use std::path::Path;

use thiserror::Error;

use crate::core::command::Command;
use crate::core::memory_entry::{MemoryEntry, MemoryId};
use crate::core::state_machine::StateMachine;
use crate::engine::{CapacityPolicy, Engine, EvictionReport};
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
}

pub type Result<T> = std::result::Result<T, MnemosStoreError>;

/// Library-facing facade for storing and querying agent memory.
///
/// Wraps:
/// - `Engine`: durability + state machine + capacity/compaction
/// - `IndexLayer`: vector/graph/temporal retrieval indexes
/// - `QueryPlanner` + `QueryExecutor`: predictable query execution
pub struct MnemosStore {
    engine: Engine,
    indexes: IndexLayer,
}

impl MnemosStore {
    pub fn new<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
    ) -> Result<Self> {
        let engine = Engine::new(wal_path, segments_dir)?;
        Ok(Self {
            engine,
            indexes: IndexLayer::new(vector_dimension),
        })
    }

    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        segments_dir: P,
        vector_dimension: usize,
    ) -> Result<Self> {
        let engine = Engine::recover(wal_path, segments_dir)?;
        Ok(Self {
            engine,
            indexes: IndexLayer::new(vector_dimension),
        })
    }

    pub fn insert_memory(&mut self, entry: MemoryEntry) -> Result<CommandId> {
        if let Some(embedding) = entry.embedding.as_ref() {
            if embedding.len() != self.indexes.vector.dimension() {
                return Err(crate::index::vector::VectorError::DimensionMismatch {
                    expected: self.indexes.vector.dimension(),
                    actual: embedding.len(),
                }
                .into());
            }
        }

        let cmd_id = self
            .engine
            .execute_command(Command::InsertMemory(entry.clone()))?;
        if let Some(embedding) = entry.embedding {
            self.indexes.vector_index_mut().index(entry.id, embedding)?;
        }
        Ok(cmd_id)
    }

    pub fn delete_memory(&mut self, id: MemoryId) -> Result<CommandId> {
        let cmd_id = self.engine.execute_command(Command::DeleteMemory(id))?;
        let _ = self.indexes.vector_index_mut().remove(id);
        Ok(cmd_id)
    }

    pub fn add_edge(
        &mut self,
        from: MemoryId,
        to: MemoryId,
        relation: String,
    ) -> Result<CommandId> {
        Ok(self
            .engine
            .execute_command(Command::AddEdge { from, to, relation })?)
    }

    pub fn remove_edge(&mut self, from: MemoryId, to: MemoryId) -> Result<CommandId> {
        Ok(self
            .engine
            .execute_command(Command::RemoveEdge { from, to })?)
    }

    /// Rebuild in-memory vector index from current state machine entries.
    pub fn rebuild_vector_index(&mut self) -> Result<usize> {
        self.indexes = IndexLayer::new(self.indexes.vector.dimension());
        let mut indexed = 0usize;
        for entry in self
            .engine
            .get_state_machine()
            .get_memories_in_time_range(0, u64::MAX)
        {
            if let Some(embedding) = entry.embedding.clone() {
                self.indexes.vector_index_mut().index(entry.id, embedding)?;
                indexed += 1;
            }
        }
        Ok(indexed)
    }

    pub fn query(
        &self,
        query_text: &str,
        options: QueryOptions,
        embedder: &dyn QueryEmbedder,
    ) -> Result<QueryExecution> {
        let plan = QueryPlanner::plan(options, self.indexes.vector.len());
        Ok(QueryExecutor::execute(
            query_text,
            &plan,
            self.engine.get_state_machine(),
            &self.indexes,
            embedder,
        )?)
    }

    pub fn query_with_plan(
        &self,
        query_text: &str,
        plan: &QueryPlan,
        embedder: &dyn QueryEmbedder,
    ) -> Result<QueryExecution> {
        Ok(QueryExecutor::execute(
            query_text,
            plan,
            self.engine.get_state_machine(),
            &self.indexes,
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
        let plan = QueryPlanner::plan(options, self.indexes.vector.len());
        Ok(QueryExecutor::execute_with_trace(
            query_text,
            &plan,
            self.engine.get_state_machine(),
            &self.indexes,
            embedder,
            Some(trace),
        )?)
    }

    pub fn enforce_capacity(&mut self, policy: CapacityPolicy) -> Result<EvictionReport> {
        let report = self.engine.enforce_capacity(policy)?;
        for id in &report.evicted_ids {
            let _ = self.indexes.vector_index_mut().remove(*id);
        }
        Ok(report)
    }

    pub fn compact_segments(&mut self) -> Result<CompactionReport> {
        Ok(self.engine.compact_segments()?)
    }

    pub fn state_machine(&self) -> &StateMachine {
        self.engine.get_state_machine()
    }

    pub fn vector_dimension(&self) -> usize {
        self.indexes.vector.dimension()
    }

    pub fn indexed_embeddings(&self) -> usize {
        self.indexes.vector.len()
    }

    pub fn wal_len(&self) -> u64 {
        self.engine.wal_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    struct TestEmbedder;
    impl QueryEmbedder for TestEmbedder {
        fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    #[test]
    fn test_store_insert_and_query() {
        let temp = TempDir::new().unwrap();
        let wal = temp.path().join("store.wal");
        let seg = temp.path().join("segments");
        let mut store = MnemosStore::new(&wal, &seg, 3).unwrap();

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
        let mut store = MnemosStore::new(&wal, &seg, 3).unwrap();

        let entry = MemoryEntry::new(MemoryId(10), "agent1".to_string(), b"x".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert_memory(entry).unwrap();
        assert_eq!(store.indexed_embeddings(), 1);

        store.delete_memory(MemoryId(10)).unwrap();
        assert_eq!(store.indexed_embeddings(), 0);
    }
}
