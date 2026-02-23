use crate::query::hybrid::{QueryOptions, ScoreWeights};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPath {
    VectorOnly,
    VectorTemporal,
    VectorGraph,
    WeightedHybrid,
}

#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub path: ExecutionPath,
    pub options: QueryOptions,
    pub candidate_multiplier: usize,
    pub ann_candidate_multiplier: usize,
    pub use_parallel: bool,
}

pub struct QueryPlanner;

#[derive(Debug, Clone, Copy)]
pub struct IntentAdjustments {
    pub score_weights: ScoreWeights,
    pub graph_hops: usize,
}

impl QueryPlanner {
    /// Build an execution plan from query options and current index size.
    ///
    /// Heuristics:
    /// - Execution path:
    ///   - VectorOnly: no temporal and no graph expansion
    ///   - VectorTemporal: temporal only
    ///   - VectorGraph: graph expansion only
    ///   - WeightedHybrid: temporal + graph expansion together
    /// - Candidate multiplier:
    ///   - increase when extra filters/expansion are present to preserve recall
    /// - Parallel search:
    ///   - enabled for larger indices and candidate sets
    pub fn plan(mut options: QueryOptions, indexed_embeddings: usize) -> QueryPlan {
        let has_temporal = options.time_range.is_some();
        let has_graph = options.graph_expansion.map(|g| g.hops > 0).unwrap_or(false);

        let path = match (has_temporal, has_graph) {
            (false, false) => ExecutionPath::VectorOnly,
            (true, false) => ExecutionPath::VectorTemporal,
            (false, true) => ExecutionPath::VectorGraph,
            (true, true) => ExecutionPath::WeightedHybrid,
        };

        let mut multiplier = 3usize;
        if has_temporal {
            multiplier += 1;
        }
        if has_graph {
            multiplier += 2;
        }
        if options.top_k > 50 {
            multiplier += 1;
        }
        multiplier = multiplier.max(1);

        let estimated_candidates = options.top_k.saturating_mul(multiplier);
        let use_parallel = indexed_embeddings >= 10_000 || estimated_candidates >= 1_000;
        let ann_candidate_multiplier = (multiplier * 2).max(7);

        options.candidate_multiplier = multiplier;

        // Weighted path should have meaningful metadata weights.
        if path == ExecutionPath::WeightedHybrid
            && options.score_weights.similarity_pct == 100
            && options.score_weights.importance_pct == 0
            && options.score_weights.recency_pct == 0
        {
            options.score_weights = ScoreWeights::default();
        }

        QueryPlan {
            path,
            options,
            candidate_multiplier: multiplier,
            ann_candidate_multiplier,
            use_parallel,
        }
    }

    /// Infer dynamic score mix + graph depth from query proximity to intent anchors.
    ///
    /// Anchors are expected to be embedded in the same vector space as `query_embedding`.
    /// Returns `None` when dimensions mismatch or any vector is degenerate.
    pub fn infer_intent_adjustments(
        query_embedding: &[f32],
        semantic_anchor: &[f32],
        recency_anchor: &[f32],
        graph_anchor: &[f32],
    ) -> Option<IntentAdjustments> {
        let semantic_sim = cosine_similarity(query_embedding, semantic_anchor)?;
        let recency_sim = cosine_similarity(query_embedding, recency_anchor)?;
        let graph_sim = cosine_similarity(query_embedding, graph_anchor)?;

        let semantic = ((semantic_sim + 1.0) * 0.5).clamp(0.0, 1.0);
        let recency = ((recency_sim + 1.0) * 0.5).clamp(0.0, 1.0);
        let graph = ((graph_sim + 1.0) * 0.5).clamp(0.0, 1.0);

        let score_weights = normalize_similarity_recency(semantic, recency);
        let graph_hops = if graph >= 0.80 {
            3
        } else if graph >= 0.55 {
            2
        } else {
            1
        };

        Some(IntentAdjustments {
            score_weights,
            graph_hops,
        })
    }
}

fn normalize_similarity_recency(semantic: f32, recency: f32) -> ScoreWeights {
    let importance_pct = 20u8;
    let budget = 100u8.saturating_sub(importance_pct);
    let denom = (semantic + recency).max(1e-6);
    let similarity_pct = ((semantic / denom) * f32::from(budget)).round() as u8;
    let recency_pct = budget.saturating_sub(similarity_pct);
    ScoreWeights::new(similarity_pct, importance_pct, recency_pct)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::hybrid::GraphExpansionOptions;

    #[test]
    fn test_plan_vector_only() {
        let mut options = QueryOptions::with_top_k(10);
        options.graph_expansion = None;
        let plan = QueryPlanner::plan(options, 100);
        assert_eq!(plan.path, ExecutionPath::VectorOnly);
        assert!(!plan.use_parallel);
    }

    #[test]
    fn test_plan_weighted_hybrid() {
        let mut options = QueryOptions::with_top_k(10);
        options.time_range = Some((1, 10));
        options.graph_expansion = Some(GraphExpansionOptions::new(2));
        let plan = QueryPlanner::plan(options, 15_000);

        assert_eq!(plan.path, ExecutionPath::WeightedHybrid);
        assert!(plan.candidate_multiplier >= 5);
        assert!(plan.use_parallel);
    }

    #[test]
    fn test_infer_intent_adjustments_prefers_graph_and_recency() {
        let query = vec![0.0, 0.8, 0.6];
        let semantic = vec![1.0, 0.0, 0.0];
        let recency = vec![0.0, 1.0, 0.0];
        let graph = vec![0.0, 0.0, 1.0];
        let adj =
            QueryPlanner::infer_intent_adjustments(&query, &semantic, &recency, &graph).unwrap();

        assert!(adj.graph_hops >= 2);
        assert!(adj.score_weights.recency_pct >= adj.score_weights.similarity_pct);
        assert_eq!(
            adj.score_weights.similarity_pct + adj.score_weights.importance_pct + adj.score_weights.recency_pct,
            100
        );
    }
}
