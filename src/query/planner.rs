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
    pub use_parallel: bool,
}

pub struct QueryPlanner;

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
        let has_graph = options.graph_expansion.is_some();

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
            use_parallel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::hybrid::GraphExpansionOptions;

    #[test]
    fn test_plan_vector_only() {
        let plan = QueryPlanner::plan(QueryOptions::with_top_k(10), 100);
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
}
