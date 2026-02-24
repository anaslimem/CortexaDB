use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::core::memory_entry::MemoryId;
use crate::core::state_machine::StateMachine;
use crate::index::combined::IndexLayer;
use crate::index::graph::GraphIndex;
use crate::query::hybrid::{HybridQueryError, QueryEmbedder, QueryHit};
use crate::query::intent::get_intent_policy;
use crate::query::planner::{ExecutionPath, QueryPlan, QueryPlanner};

pub type Result<T> = std::result::Result<T, HybridQueryError>;

#[derive(Debug, Clone)]
pub enum StageTrace {
    Embedded,
    VectorScored { candidates: usize },
    Filtered { candidates: usize },
    GraphExpanded { candidates: usize },
    Ranked { results: usize },
}

#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub plan_path: ExecutionPath,
    pub used_parallel: bool,
    pub vector_candidates: usize,
    pub filtered_candidates: usize,
    pub final_results: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone)]
pub struct QueryExecution {
    pub hits: Vec<QueryHit>,
    pub metrics: ExecutionMetrics,
}

pub struct QueryExecutor;

#[derive(Clone)]
struct IntentAnchors {
    semantic: Vec<f32>,
    recency: Vec<f32>,
    graph: Vec<f32>,
}

fn intent_anchor_cache() -> &'static Mutex<HashMap<usize, IntentAnchors>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, IntentAnchors>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn clear_intent_anchor_cache() {
    let cache = intent_anchor_cache();
    let mut guard = cache.lock().expect("intent anchor cache lock poisoned");
    guard.clear();
}

fn load_or_build_intent_anchors(
    embedder: &dyn QueryEmbedder,
    dim: usize,
) -> std::result::Result<IntentAnchors, String> {
    let cache = intent_anchor_cache();
    {
        let guard = cache.lock().expect("intent anchor cache lock poisoned");
        if let Some(found) = guard.get(&dim) {
            return Ok(found.clone());
        }
    }

    let policy = get_intent_policy();
    let semantic = embedder.embed(&policy.semantic_anchor_text)?;
    let recency = embedder.embed(&policy.recency_anchor_text)?;
    let graph = embedder.embed(&policy.graph_anchor_text)?;
    if semantic.len() != dim || recency.len() != dim || graph.len() != dim {
        return Err("intent anchor embedding dimension mismatch".to_string());
    }
    let anchors = IntentAnchors {
        semantic,
        recency,
        graph,
    };
    let mut guard = cache.lock().expect("intent anchor cache lock poisoned");
    guard.insert(dim, anchors.clone());
    Ok(anchors)
}

impl QueryExecutor {
    pub fn execute(
        query_text: &str,
        plan: &QueryPlan,
        state_machine: &StateMachine,
        index_layer: &IndexLayer,
        embedder: &dyn QueryEmbedder,
    ) -> Result<QueryExecution> {
        Self::execute_with_trace(query_text, plan, state_machine, index_layer, embedder, None)
    }

    pub fn execute_with_trace(
        query_text: &str,
        plan: &QueryPlan,
        state_machine: &StateMachine,
        index_layer: &IndexLayer,
        embedder: &dyn QueryEmbedder,
        mut trace: Option<&mut dyn FnMut(StageTrace)>,
    ) -> Result<QueryExecution> {
        let start = Instant::now();
        let mut options = plan.options.clone();

        if options.top_k == 0 {
            return Err(HybridQueryError::InvalidTopK(options.top_k));
        }
        if options.candidate_multiplier == 0 {
            return Err(HybridQueryError::InvalidCandidateMultiplier(
                options.candidate_multiplier,
            ));
        }

        if let Some((start_ts, end_ts)) = options.time_range {
            if start_ts > end_ts {
                return Err(HybridQueryError::InvalidTimeRange {
                    start: start_ts,
                    end: end_ts,
                });
            }
        }

        let query_embedding = embedder
            .embed(query_text)
            .map_err(HybridQueryError::Embedder)?;
        if let Some(cb) = &mut trace {
            cb(StageTrace::Embedded);
        }

        if let Ok(anchors) = load_or_build_intent_anchors(embedder, query_embedding.len()) {
            let policy = get_intent_policy();
            if let Some(adj) = QueryPlanner::infer_intent_adjustments(
                &query_embedding,
                &anchors.semantic,
                &anchors.recency,
                &anchors.graph,
                policy.graph_hops_2_threshold,
                policy.graph_hops_3_threshold,
                policy.importance_pct,
            ) {
                // Respect explicit user weights: only auto-adjust when still default.
                if options.score_weights.similarity_pct == 70
                    && options.score_weights.importance_pct == 20
                    && options.score_weights.recency_pct == 10
                {
                    options.score_weights = adj.score_weights;
                }
                if options.graph_expansion.is_some() {
                    options.graph_expansion =
                        Some(crate::query::GraphExpansionOptions::new(adj.graph_hops));
                }
            }
        }

        let (sim_w, imp_w, rec_w) = {
            let total = options.score_weights.similarity_pct as u16
                + options.score_weights.importance_pct as u16
                + options.score_weights.recency_pct as u16;
            if total != 100 {
                return Err(HybridQueryError::InvalidScoreWeights {
                    similarity_pct: options.score_weights.similarity_pct,
                    importance_pct: options.score_weights.importance_pct,
                    recency_pct: options.score_weights.recency_pct,
                });
            }
            (
                options.score_weights.similarity_pct as f32 / 100.0,
                options.score_weights.importance_pct as f32 / 100.0,
                options.score_weights.recency_pct as f32 / 100.0,
            )
        };

        let candidate_k = options.top_k.saturating_mul(plan.candidate_multiplier);
        let vector_results = index_layer.vector.search_scoped(
            &query_embedding,
            candidate_k,
            options.namespace.as_deref(),
            plan.use_parallel,
            plan.ann_candidate_multiplier,
        )?;
        if let Some(cb) = &mut trace {
            cb(StageTrace::VectorScored {
                candidates: vector_results.len(),
            });
        }

        let mut candidate_scores: HashMap<MemoryId, f32> = vector_results
            .into_iter()
            .filter(|(id, _)| matches_filters(state_machine, *id, None, options.time_range))
            .collect();
        if let Some(cb) = &mut trace {
            cb(StageTrace::Filtered {
                candidates: candidate_scores.len(),
            });
        }

        if matches!(
            plan.path,
            ExecutionPath::VectorGraph | ExecutionPath::WeightedHybrid
        ) {
            if let Some(expansion) = options.graph_expansion {
                if expansion.hops > 0 {
                    let mut expanded_ids = HashSet::new();
                    for id in candidate_scores.keys().copied() {
                        let reachable = if let Some(ns) = options.namespace.as_deref() {
                            GraphIndex::bfs_in_namespace(state_machine, id, expansion.hops, ns)?
                        } else {
                            GraphIndex::bfs(state_machine, id, expansion.hops)?
                        };
                        for neighbor in reachable.keys().copied() {
                            if matches_filters(
                                state_machine,
                                neighbor,
                                options.namespace.as_deref(),
                                None,
                            ) {
                                expanded_ids.insert(neighbor);
                            }
                        }
                    }
                    let rescored = index_layer.vector.search_in_ids(
                        &query_embedding,
                        &expanded_ids,
                        expanded_ids.len(),
                    )?;
                    candidate_scores = rescored.into_iter().collect();
                    if let Some(cb) = &mut trace {
                        cb(StageTrace::GraphExpanded {
                            candidates: candidate_scores.len(),
                        });
                    }
                }
            }
        }

        let hits = build_ranked_hits(
            state_machine,
            candidate_scores,
            options.top_k,
            matches!(plan.path, ExecutionPath::WeightedHybrid),
            sim_w,
            imp_w,
            rec_w,
        );
        if let Some(cb) = &mut trace {
            cb(StageTrace::Ranked {
                results: hits.len(),
            });
        }

        let metrics = ExecutionMetrics {
            plan_path: plan.path,
            used_parallel: plan.use_parallel,
            vector_candidates: candidate_k,
            filtered_candidates: hits.len(),
            final_results: hits.len(),
            elapsed_ms: start.elapsed().as_millis(),
        };

        Ok(QueryExecution { hits, metrics })
    }
}

fn matches_filters(
    state_machine: &StateMachine,
    id: MemoryId,
    namespace: Option<&str>,
    time_range: Option<(u64, u64)>,
) -> bool {
    let entry = match state_machine.get_memory(id) {
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

fn build_ranked_hits(
    state_machine: &StateMachine,
    candidates: HashMap<MemoryId, f32>,
    top_k: usize,
    weighted: bool,
    sim_w: f32,
    imp_w: f32,
    rec_w: f32,
) -> Vec<QueryHit> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut ts_min = u64::MAX;
    let mut ts_max = 0u64;
    let mut timestamps = HashMap::new();
    for id in candidates.keys().copied() {
        if let Ok(entry) = state_machine.get_memory(id) {
            ts_min = ts_min.min(entry.created_at);
            ts_max = ts_max.max(entry.created_at);
            timestamps.insert(id, entry.created_at);
        }
    }

    let mut hits = Vec::new();
    for (id, raw_similarity) in candidates {
        let entry = match state_machine.get_memory(id) {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let similarity_score = ((raw_similarity + 1.0) * 0.5).clamp(0.0, 1.0);
        let importance_score = entry.importance.clamp(0.0, 1.0);
        let recency_score = if ts_min == ts_max {
            1.0
        } else {
            let created_at = *timestamps.get(&id).unwrap_or(&entry.created_at);
            (created_at - ts_min) as f32 / (ts_max - ts_min) as f32
        };

        let final_score = if weighted {
            sim_w * similarity_score + imp_w * importance_score + rec_w * recency_score
        } else {
            similarity_score
        };

        hits.push(QueryHit {
            id,
            final_score,
            similarity_score,
            importance_score,
            recency_score,
        });
    }

    hits.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    hits.truncate(top_k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryEntry, MemoryId};
    use crate::query::hybrid::{GraphExpansionOptions, QueryOptions};
    use crate::query::planner::{ExecutionPath, QueryPlanner};

    struct TestEmbedder;
    impl QueryEmbedder for TestEmbedder {
        fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    fn setup() -> (StateMachine, IndexLayer, TestEmbedder) {
        let mut sm = StateMachine::new();
        let mut layer = IndexLayer::new(3);

        let a = MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"a".to_vec(), 1000)
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_importance(0.1);
        let b = MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"b".to_vec(), 2000)
            .with_embedding(vec![0.9, 0.1, 0.0])
            .with_importance(0.9);
        for entry in [&a, &b] {
            layer
                .vector_index_mut()
                .index(entry.id, entry.embedding.clone().unwrap())
                .unwrap();
        }
        sm.insert_memory(a).unwrap();
        sm.insert_memory(b).unwrap();
        sm.add_edge(MemoryId(1), MemoryId(2), "next".to_string())
            .unwrap();

        (sm, layer, TestEmbedder)
    }

    #[test]
    fn test_execute_vector_temporal_plan() {
        let (sm, layer, embedder) = setup();
        let mut opts = QueryOptions::with_top_k(5);
        opts.time_range = Some((1500, 2500));
        opts.graph_expansion = None;
        let plan = QueryPlanner::plan(opts, layer.vector.len());
        assert_eq!(plan.path, ExecutionPath::VectorTemporal);

        let out = QueryExecutor::execute("q", &plan, &sm, &layer, &embedder).unwrap();
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].id, MemoryId(2));
    }

    #[test]
    fn test_execute_weighted_hybrid_plan() {
        let (sm, layer, embedder) = setup();
        let mut opts = QueryOptions::with_top_k(5);
        opts.time_range = Some((500, 3000));
        opts.graph_expansion = Some(GraphExpansionOptions::new(1));
        let plan = QueryPlanner::plan(opts, layer.vector.len());
        assert_eq!(plan.path, ExecutionPath::WeightedHybrid);

        let out = QueryExecutor::execute("q", &plan, &sm, &layer, &embedder).unwrap();
        assert!(!out.hits.is_empty());
    }
}
