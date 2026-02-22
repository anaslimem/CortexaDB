use std::collections::{HashMap, HashSet};
use thiserror::Error;

use crate::core::memory_entry::MemoryId;
use crate::core::state_machine::StateMachine;
use crate::index::graph::GraphIndex;
use crate::index::temporal::TemporalIndex;
use crate::index::vector::VectorIndex;

#[derive(Error, Debug)]
pub enum CombinedError {
    #[error("Vector error: {0}")]
    VectorError(#[from] crate::index::vector::VectorError),
    #[error("Graph error: {0}")]
    GraphError(#[from] crate::index::graph::GraphError),
    #[error("Temporal error: {0}")]
    TemporalError(#[from] crate::index::temporal::TemporalError),
    #[error(
        "Invalid ranking weights: vector={vector_pct}, recency={recency_pct}, graph={graph_pct}"
    )]
    InvalidWeights {
        vector_pct: u8,
        recency_pct: u8,
        graph_pct: u8,
    },
}

pub type Result<T> = std::result::Result<T, CombinedError>;

/// Percent-based weights for weighted combined ranking.
///
/// Must sum to 100.
#[derive(Debug, Clone, Copy)]
pub struct RankingWeights {
    pub vector_pct: u8,
    pub recency_pct: u8,
    pub graph_pct: u8,
}

impl RankingWeights {
    pub const fn new(vector_pct: u8, recency_pct: u8, graph_pct: u8) -> Self {
        Self {
            vector_pct,
            recency_pct,
            graph_pct,
        }
    }

    fn normalized(self) -> Result<(f32, f32, f32)> {
        let total = self.vector_pct as u16 + self.recency_pct as u16 + self.graph_pct as u16;
        if total != 100 {
            return Err(CombinedError::InvalidWeights {
                vector_pct: self.vector_pct,
                recency_pct: self.recency_pct,
                graph_pct: self.graph_pct,
            });
        }

        Ok((
            self.vector_pct as f32 / 100.0,
            self.recency_pct as f32 / 100.0,
            self.graph_pct as f32 / 100.0,
        ))
    }
}

impl Default for RankingWeights {
    fn default() -> Self {
        // Default blend: semantic relevance is primary, then recency, then graph distance.
        Self::new(70, 20, 10)
    }
}

/// Combined index layer for multi-criteria queries
///
/// Combines Vector + Graph + Temporal indexes for rich contextual search
pub struct IndexLayer {
    pub vector: VectorIndex,
}

impl IndexLayer {
    /// Create new index layer
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            vector: VectorIndex::new(vector_dimension),
        }
    }

    /// Search similar embeddings within a time range
    ///
    /// Returns: [(MemoryId, similarity_score)]
    pub fn search_similar_in_range(
        &self,
        state_machine: &StateMachine,
        query: &[f32],
        time_start: u64,
        time_end: u64,
        top_k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // Step 1: Get memories in time range
        let temporal_results = TemporalIndex::get_range(state_machine, time_start, time_end)?;
        let candidates: HashSet<MemoryId> = temporal_results.into_iter().collect();

        // Step 2: Search similarity only among candidates
        self.vector
            .search_in_ids(query, &candidates, top_k)
            .map_err(Into::into)
    }

    /// Search similar embeddings connected to a specific memory
    ///
    /// Returns: [(MemoryId, similarity_score)]
    pub fn search_similar_connected_to(
        &self,
        state_machine: &StateMachine,
        query: &[f32],
        origin: MemoryId,
        max_hops: usize,
        top_k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // Step 1: Get all reachable memories
        let graph_results = GraphIndex::get_reachable(state_machine, origin, max_hops)?;
        let candidates: HashSet<MemoryId> = graph_results.into_iter().collect();

        // Step 2: Search similarity only among connected memories
        self.vector
            .search_in_ids(query, &candidates, top_k)
            .map_err(Into::into)
    }

    /// Search similar embeddings within time range AND connected to a memory
    ///
    /// Three-way intersection: Vector + Graph + Temporal
    /// Returns: [(MemoryId, similarity_score)]
    pub fn search_similar_in_range_connected_to(
        &self,
        state_machine: &StateMachine,
        query: &[f32],
        time_start: u64,
        time_end: u64,
        origin: MemoryId,
        max_hops: usize,
        top_k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // Step 1: Get memories in time range
        let temporal_results = TemporalIndex::get_range(state_machine, time_start, time_end)?;

        // Step 2: Get all reachable memories
        let graph_results = GraphIndex::get_reachable(state_machine, origin, max_hops)?;

        // Step 3: Find intersection of temporal AND graph
        let temporal_set: HashSet<MemoryId> = temporal_results.into_iter().collect();
        let graph_set: HashSet<MemoryId> = graph_results.into_iter().collect();
        let combined: HashSet<MemoryId> = temporal_set.intersection(&graph_set).copied().collect();

        // Step 4: Search similarity in intersection only
        self.vector
            .search_in_ids(query, &combined, top_k)
            .map_err(Into::into)
    }

    /// Weighted combined ranking over Vector + Temporal + Graph.
    ///
    /// Score is computed as:
    /// `vector_w * vector_score + recency_w * recency_score + graph_w * proximity_score`
    /// with default weights: 70% vector, 20% recency, 10% graph proximity.
    pub fn search_weighted_in_range_connected_to(
        &self,
        state_machine: &StateMachine,
        query: &[f32],
        time_start: u64,
        time_end: u64,
        origin: MemoryId,
        max_hops: usize,
        top_k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        self.search_weighted_in_range_connected_to_with_weights(
            state_machine,
            query,
            time_start,
            time_end,
            origin,
            max_hops,
            top_k,
            RankingWeights::default(),
        )
    }

    /// Same as `search_weighted_in_range_connected_to` but with caller-provided percentages.
    pub fn search_weighted_in_range_connected_to_with_weights(
        &self,
        state_machine: &StateMachine,
        query: &[f32],
        time_start: u64,
        time_end: u64,
        origin: MemoryId,
        max_hops: usize,
        top_k: usize,
        weights: RankingWeights,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let (vector_w, recency_w, graph_w) = weights.normalized()?;

        // Step 1: candidate intersection from temporal + graph.
        let temporal_results = TemporalIndex::get_range(state_machine, time_start, time_end)?;
        let temporal_set: HashSet<MemoryId> = temporal_results.into_iter().collect();
        let graph_distances = GraphIndex::bfs(state_machine, origin, max_hops)?;
        let graph_set: HashSet<MemoryId> = graph_distances.keys().copied().collect();
        let candidates: HashSet<MemoryId> =
            temporal_set.intersection(&graph_set).copied().collect();
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: vector similarity for all candidates.
        let vector_results = self
            .vector
            .search_in_ids(query, &candidates, candidates.len())?;
        if vector_results.is_empty() {
            return Ok(Vec::new());
        }
        let vector_scores: HashMap<MemoryId, f32> = vector_results.into_iter().collect();

        // Step 3: collect timestamps for normalization.
        let mut ts_min = u64::MAX;
        let mut ts_max = 0u64;
        let mut timestamps = HashMap::new();
        for id in vector_scores.keys().copied() {
            let ts = match state_machine.get_memory(id) {
                Ok(entry) => entry.created_at,
                Err(_) => continue,
            };
            ts_min = ts_min.min(ts);
            ts_max = ts_max.max(ts);
            timestamps.insert(id, ts);
        }
        if timestamps.is_empty() {
            return Ok(Vec::new());
        }

        // Step 4: weighted score composition.
        let hop_denominator = max_hops.max(1) as f32;
        let mut scored: Vec<(MemoryId, f32)> = timestamps
            .iter()
            .filter_map(|(id, ts)| {
                let raw_vector = *vector_scores.get(id)?;
                let distance = *graph_distances.get(id)? as f32;

                // Map cosine range [-1, 1] -> [0, 1] for consistent blending.
                let vector_score = ((raw_vector + 1.0) * 0.5).clamp(0.0, 1.0);
                let recency_score = if ts_max == ts_min {
                    1.0
                } else {
                    (*ts - ts_min) as f32 / (ts_max - ts_min) as f32
                };
                let proximity_score = (1.0 - (distance / hop_denominator)).clamp(0.0, 1.0);

                let final_score =
                    vector_w * vector_score + recency_w * recency_score + graph_w * proximity_score;
                Some((*id, final_score))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Get vector index
    pub fn vector_index(&self) -> &VectorIndex {
        &self.vector
    }

    /// Get mutable vector index
    pub fn vector_index_mut(&mut self) -> &mut VectorIndex {
        &mut self.vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::MemoryEntry;

    fn create_entry(id: u64, timestamp: u64) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId(id),
            "test".to_string(),
            format!("content_{}", id).into_bytes(),
            timestamp,
        )
        .with_embedding(vec![
            ((id as f32) * 0.1).sin(),
            ((id as f32) * 0.2).cos(),
            ((id as f32) * 0.3).sin(),
        ])
    }

    fn setup_combined() -> (StateMachine, IndexLayer) {
        let mut sm = StateMachine::new();
        let mut layer = IndexLayer::new(3);

        // Create memories with different timestamps
        for i in 0..5 {
            let timestamp = 1000 + (i as u64) * 1000;
            let entry = create_entry(i as u64, timestamp);

            // Index the embedding
            layer
                .vector_index_mut()
                .index(MemoryId(i as u64), entry.embedding.clone().unwrap())
                .unwrap();

            sm.insert_memory(entry).unwrap();
        }

        // Create edges: 0→1, 0→2, 1→3, 2→3, 3→4
        sm.add_edge(MemoryId(0), MemoryId(1), "points".to_string())
            .unwrap();
        sm.add_edge(MemoryId(0), MemoryId(2), "refers".to_string())
            .unwrap();
        sm.add_edge(MemoryId(1), MemoryId(3), "links".to_string())
            .unwrap();
        sm.add_edge(MemoryId(2), MemoryId(3), "connects".to_string())
            .unwrap();
        sm.add_edge(MemoryId(3), MemoryId(4), "leads".to_string())
            .unwrap();

        (sm, layer)
    }

    #[test]
    fn test_search_similar_in_range() {
        let (sm, layer) = setup_combined();

        let query = vec![0.1, 0.2, 0.3]; // Non-zero vector
        let results = layer
            .search_similar_in_range(&sm, &query, 1000, 3000, 2)
            .unwrap();

        // Should find memories within time range
        assert!(results.len() <= 2);
        for (id, _score) in results {
            // Should be in range [1000, 3000]
            assert!(id.0 <= 2);
        }
    }

    #[test]
    fn test_search_similar_connected_to() {
        let (sm, layer) = setup_combined();

        let query = vec![0.1, 0.2, 0.3]; // Non-zero vector
        let results = layer
            .search_similar_connected_to(&sm, &query, MemoryId(0), 2, 3)
            .unwrap();

        // Should find connected memories
        for (id, _score) in results {
            // Should be reachable from 0 within 2 hops
            assert!(id.0 <= 4);
        }
    }

    #[test]
    fn test_search_combined_three_way() {
        let (sm, layer) = setup_combined();

        let query = vec![0.1, 0.2, 0.3]; // Non-zero vector
        let results = layer
            .search_similar_in_range_connected_to(&sm, &query, 1000, 4000, MemoryId(0), 2, 3)
            .unwrap();

        // Should satisfy:
        // 1. In time range [1000, 4000]
        // 2. Connected to 0 within 2 hops
        // 3. Most similar to query (top 3)
        for (id, _score) in results {
            // Should be in range [0, 3]
            assert!(id.0 <= 3);
        }
    }

    #[test]
    fn test_index_layer_creation() {
        let layer = IndexLayer::new(768);
        assert_eq!(layer.vector_index().dimension(), 768);
    }

    #[test]
    fn test_search_similar_in_range_does_not_miss_valid_candidates() {
        let mut sm = StateMachine::new();
        let mut layer = IndexLayer::new(2);

        // In-range candidate (should be returned even if globally lower-ranked)
        let in_range = MemoryEntry::new(MemoryId(1), "test".to_string(), b"a".to_vec(), 1000)
            .with_embedding(vec![0.5, 0.5]);
        layer
            .vector_index_mut()
            .index(MemoryId(1), in_range.embedding.clone().unwrap())
            .unwrap();
        sm.insert_memory(in_range).unwrap();

        // Out-of-range but highly similar vectors
        let out_1 = MemoryEntry::new(MemoryId(2), "test".to_string(), b"b".to_vec(), 9000)
            .with_embedding(vec![1.0, 0.0]);
        layer
            .vector_index_mut()
            .index(MemoryId(2), out_1.embedding.clone().unwrap())
            .unwrap();
        sm.insert_memory(out_1).unwrap();

        let out_2 = MemoryEntry::new(MemoryId(3), "test".to_string(), b"c".to_vec(), 9000)
            .with_embedding(vec![0.99, 0.01]);
        layer
            .vector_index_mut()
            .index(MemoryId(3), out_2.embedding.clone().unwrap())
            .unwrap();
        sm.insert_memory(out_2).unwrap();

        let results = layer
            .search_similar_in_range(&sm, &[1.0, 0.0], 1000, 1000, 1)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, MemoryId(1));
    }

    #[test]
    fn test_weighted_search_default_prefers_newer_when_vector_ties() {
        let mut sm = StateMachine::new();
        let mut layer = IndexLayer::new(2);

        let origin = MemoryEntry::new(MemoryId(0), "test".to_string(), b"o".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0]);
        let close_old = MemoryEntry::new(MemoryId(1), "test".to_string(), b"c1".to_vec(), 5000)
            .with_embedding(vec![1.0, 0.0]);
        let mid = MemoryEntry::new(MemoryId(3), "test".to_string(), b"m".to_vec(), 3000)
            .with_embedding(vec![0.2, 0.8]);
        let far_new = MemoryEntry::new(MemoryId(2), "test".to_string(), b"c2".to_vec(), 9000)
            .with_embedding(vec![1.0, 0.0]);

        for entry in [&origin, &close_old, &mid, &far_new] {
            layer
                .vector_index_mut()
                .index(entry.id, entry.embedding.clone().unwrap())
                .unwrap();
        }
        sm.insert_memory(origin).unwrap();
        sm.insert_memory(close_old).unwrap();
        sm.insert_memory(mid).unwrap();
        sm.insert_memory(far_new).unwrap();

        sm.add_edge(MemoryId(0), MemoryId(1), "to".to_string())
            .unwrap();
        sm.add_edge(MemoryId(0), MemoryId(3), "to".to_string())
            .unwrap();
        sm.add_edge(MemoryId(3), MemoryId(2), "to".to_string())
            .unwrap();

        let results = layer
            .search_weighted_in_range_connected_to(&sm, &[1.0, 0.0], 2000, 10000, MemoryId(0), 2, 1)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, MemoryId(2));
    }

    #[test]
    fn test_weighted_search_rejects_invalid_percentages() {
        let (sm, layer) = setup_combined();
        let result = layer.search_weighted_in_range_connected_to_with_weights(
            &sm,
            &[0.1, 0.2, 0.3],
            1000,
            4000,
            MemoryId(0),
            2,
            3,
            RankingWeights::new(80, 30, 10),
        );
        assert!(result.is_err());
    }
}
