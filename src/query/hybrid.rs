use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::core::memory_entry::MemoryId;
use crate::core::state_machine::StateMachine;
use crate::index::combined::IndexLayer;
use crate::index::graph::GraphIndex;

#[derive(Error, Debug)]
pub enum HybridQueryError {
    #[error("Embedder error: {0}")]
    Embedder(String),
    #[error("Vector error: {0}")]
    Vector(#[from] crate::index::vector::VectorError),
    #[error("Graph error: {0}")]
    Graph(#[from] crate::index::graph::GraphError),
    #[error("Invalid top_k: {0}")]
    InvalidTopK(usize),
    #[error("Invalid candidate_multiplier: {0}")]
    InvalidCandidateMultiplier(usize),
    #[error(
        "Invalid score weights: similarity={similarity_pct}, importance={importance_pct}, recency={recency_pct}"
    )]
    InvalidScoreWeights {
        similarity_pct: u8,
        importance_pct: u8,
        recency_pct: u8,
    },
    #[error("Invalid time range: start={start}, end={end}")]
    InvalidTimeRange { start: u64, end: u64 },
}

pub type Result<T> = std::result::Result<T, HybridQueryError>;

/// A pluggable embedding interface so agent developers can integrate any model backend.
pub trait QueryEmbedder {
    fn embed(&self, query: &str) -> std::result::Result<Vec<f32>, String>;
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreWeights {
    pub similarity_pct: u8,
    pub importance_pct: u8,
    pub recency_pct: u8,
}

impl ScoreWeights {
    pub const fn new(similarity_pct: u8, importance_pct: u8, recency_pct: u8) -> Self {
        Self {
            similarity_pct,
            importance_pct,
            recency_pct,
        }
    }

    fn normalized(self) -> Result<(f32, f32, f32)> {
        let total =
            self.similarity_pct as u16 + self.importance_pct as u16 + self.recency_pct as u16;
        if total != 100 {
            return Err(HybridQueryError::InvalidScoreWeights {
                similarity_pct: self.similarity_pct,
                importance_pct: self.importance_pct,
                recency_pct: self.recency_pct,
            });
        }
        Ok((
            self.similarity_pct as f32 / 100.0,
            self.importance_pct as f32 / 100.0,
            self.recency_pct as f32 / 100.0,
        ))
    }
}

impl Default for ScoreWeights {
    fn default() -> Self {
        // Strong semantic signal with metadata balancing.
        Self::new(70, 20, 10)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GraphExpansionOptions {
    pub hops: usize,
}

impl GraphExpansionOptions {
    pub const fn new(hops: usize) -> Self {
        Self { hops }
    }
}

#[derive(Debug, Clone)]
pub struct QueryOptions {
    pub top_k: usize,
    pub namespace: Option<String>,
    pub time_range: Option<(u64, u64)>,
    pub graph_expansion: Option<GraphExpansionOptions>,
    pub candidate_multiplier: usize,
    pub score_weights: ScoreWeights,
}

impl QueryOptions {
    pub fn with_top_k(top_k: usize) -> Self {
        Self {
            top_k,
            ..Self::default()
        }
    }
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            namespace: None,
            time_range: None,
            graph_expansion: None,
            candidate_multiplier: 5,
            score_weights: ScoreWeights::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryHit {
    pub id: MemoryId,
    pub final_score: f32,
    pub similarity_score: f32,
    pub importance_score: f32,
    pub recency_score: f32,
}

pub struct HybridQueryEngine<'a> {
    state_machine: &'a StateMachine,
    index_layer: &'a IndexLayer,
    embedder: &'a dyn QueryEmbedder,
}

impl<'a> HybridQueryEngine<'a> {
    pub fn new(
        state_machine: &'a StateMachine,
        index_layer: &'a IndexLayer,
        embedder: &'a dyn QueryEmbedder,
    ) -> Self {
        Self {
            state_machine,
            index_layer,
            embedder,
        }
    }

    /// Convenience API
    pub fn query(
        &self,
        query_text: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<QueryHit>> {
        let mut options = QueryOptions::with_top_k(top_k);
        options.namespace = namespace.map(|ns| ns.to_string());
        self.query_with_options(query_text, options)
    }

    pub fn query_with_options(
        &self,
        query_text: &str,
        options: QueryOptions,
    ) -> Result<Vec<QueryHit>> {
        if options.top_k == 0 {
            return Err(HybridQueryError::InvalidTopK(options.top_k));
        }
        if options.candidate_multiplier == 0 {
            return Err(HybridQueryError::InvalidCandidateMultiplier(
                options.candidate_multiplier,
            ));
        }
        let (sim_w, imp_w, rec_w) = options.score_weights.normalized()?;
        if let Some((start, end)) = options.time_range {
            if start > end {
                return Err(HybridQueryError::InvalidTimeRange { start, end });
            }
        }

        let query_embedding = self
            .embedder
            .embed(query_text)
            .map_err(HybridQueryError::Embedder)?;

        let candidate_k = options.top_k.saturating_mul(options.candidate_multiplier);
        let ann_multiplier = options.candidate_multiplier.max(7);
        let vector_results = self.index_layer.vector.search_scoped(
            &query_embedding,
            candidate_k,
            options.namespace.as_deref(),
            false,
            ann_multiplier,
        )?;

        let mut candidate_scores = HashMap::new();
        for (id, cosine_similarity) in vector_results {
            if self.matches_filters(id, None, options.time_range) {
                candidate_scores.insert(id, cosine_similarity);
            }
        }

        if let Some(expansion) = options.graph_expansion {
            if expansion.hops > 0 {
                let mut expanded_ids = HashSet::new();
                let base_ids: Vec<MemoryId> = candidate_scores.keys().copied().collect();
                for id in base_ids {
                    let reachable = GraphIndex::bfs(self.state_machine, id, expansion.hops)?;
                    for reachable_id in reachable.keys().copied() {
                        if self.matches_filters(reachable_id, options.namespace.as_deref(), None) {
                            expanded_ids.insert(reachable_id);
                        }
                    }
                }

                if !expanded_ids.is_empty() {
                    let rescored = self.index_layer.vector.search_in_ids(
                        &query_embedding,
                        &expanded_ids,
                        expanded_ids.len(),
                    )?;
                    candidate_scores = rescored.into_iter().collect();
                }
            }
        }

        if candidate_scores.is_empty() {
            return Ok(Vec::new());
        }

        let mut timestamps = HashMap::new();
        let mut ts_min = u64::MAX;
        let mut ts_max = 0u64;
        for id in candidate_scores.keys().copied() {
            if let Ok(entry) = self.state_machine.get_memory(id) {
                timestamps.insert(id, entry.created_at);
                ts_min = ts_min.min(entry.created_at);
                ts_max = ts_max.max(entry.created_at);
            }
        }

        let mut results = Vec::new();
        for (id, raw_similarity) in candidate_scores {
            let entry = match self.state_machine.get_memory(id) {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let created_at = *timestamps.get(&id).unwrap_or(&entry.created_at);

            let similarity_score = ((raw_similarity + 1.0) * 0.5).clamp(0.0, 1.0);
            let importance_score = entry.importance.clamp(0.0, 1.0);
            let recency_score = if ts_min == ts_max {
                1.0
            } else {
                (created_at - ts_min) as f32 / (ts_max - ts_min) as f32
            };

            let final_score =
                sim_w * similarity_score + imp_w * importance_score + rec_w * recency_score;

            results.push(QueryHit {
                id,
                final_score,
                similarity_score,
                importance_score,
                recency_score,
            });
        }

        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.similarity_score
                        .partial_cmp(&a.similarity_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(options.top_k);
        Ok(results)
    }

    fn matches_filters(
        &self,
        id: MemoryId,
        namespace: Option<&str>,
        time_range: Option<(u64, u64)>,
    ) -> bool {
        let entry = match self.state_machine.get_memory(id) {
            Ok(entry) => entry,
            Err(_) => return false,
        };

        if let Some(ns) = namespace {
            if entry.namespace != ns {
                return false;
            }
        }

        if let Some((start, end)) = time_range {
            if entry.created_at < start || entry.created_at > end {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::MemoryEntry;

    struct TestEmbedder;

    impl QueryEmbedder for TestEmbedder {
        fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    fn build_engine() -> (StateMachine, IndexLayer, TestEmbedder) {
        let mut sm = StateMachine::new();
        let mut layer = IndexLayer::new(3);
        let embedder = TestEmbedder;

        let a = MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"a".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.9);
        let b = MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"b".to_vec(), 2000)
            .with_embedding(vec![0.9, 0.1, 0.0])
            .with_importance(0.2);
        let c = MemoryEntry::new(MemoryId(3), "agent2".to_string(), b"c".to_vec(), 3000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.5);

        for entry in [&a, &b, &c] {
            layer
                .vector_index_mut()
                .index_in_namespace(&entry.namespace, entry.id, entry.embedding.clone().unwrap())
                .unwrap();
        }
        sm.insert_memory(a).unwrap();
        sm.insert_memory(b).unwrap();
        sm.insert_memory(c).unwrap();

        sm.add_edge(MemoryId(1), MemoryId(2), "linked".to_string())
            .unwrap();
        sm.add_edge(MemoryId(2), MemoryId(3), "linked".to_string())
            .unwrap();

        (sm, layer, embedder)
    }

    #[test]
    fn test_query_with_namespace_filter() {
        let (sm, layer, embedder) = build_engine();
        let engine = HybridQueryEngine::new(&sm, &layer, &embedder);

        let hits = engine.query("hello", 10, Some("agent1")).unwrap();
        assert!(
            hits.iter()
                .all(|h| h.id == MemoryId(1) || h.id == MemoryId(2))
        );
    }

    #[test]
    fn test_query_with_time_filter() {
        let (sm, layer, embedder) = build_engine();
        let engine = HybridQueryEngine::new(&sm, &layer, &embedder);

        let mut options = QueryOptions::with_top_k(10);
        options.time_range = Some((1500, 3500));
        let hits = engine.query_with_options("hello", options).unwrap();

        assert!(
            hits.iter()
                .all(|h| h.id == MemoryId(2) || h.id == MemoryId(3))
        );
    }

    #[test]
    fn test_query_with_graph_expansion() {
        let (sm, layer, embedder) = build_engine();
        let engine = HybridQueryEngine::new(&sm, &layer, &embedder);

        let mut options = QueryOptions::with_top_k(10);
        options.namespace = Some("agent1".to_string());
        options.time_range = Some((1000, 1000)); // only id=1 from vector base filter
        options.graph_expansion = Some(GraphExpansionOptions::new(1)); // expands to id=2

        let hits = engine.query_with_options("hello", options).unwrap();
        assert!(hits.iter().any(|h| h.id == MemoryId(1)));
        assert!(hits.iter().any(|h| h.id == MemoryId(2)));
    }

    #[test]
    fn test_invalid_weight_percentages() {
        let (sm, layer, embedder) = build_engine();
        let engine = HybridQueryEngine::new(&sm, &layer, &embedder);

        let mut options = QueryOptions::with_top_k(10);
        options.score_weights = ScoreWeights::new(80, 15, 15);
        let result = engine.query_with_options("hello", options);
        assert!(result.is_err());
    }
}
