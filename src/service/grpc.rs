use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use crate::core::memory_entry::{MemoryEntry, MemoryId};
use crate::engine::CapacityPolicy;
use crate::query::{GraphExpansionOptions, QueryOptions, ScoreWeights};
use crate::store::MnemosStore;

pub mod proto {
    tonic::include_proto!("mnemos");
}

pub use proto::mnemos_service_server::MnemosServiceServer;

#[derive(Clone)]
pub struct MnemosGrpcService {
    store: Arc<Mutex<MnemosStore>>,
}

impl MnemosGrpcService {
    pub fn new(store: MnemosStore) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
        }
    }
}

struct StaticEmbedder {
    embedding: Vec<f32>,
}

impl crate::query::QueryEmbedder for StaticEmbedder {
    fn embed(&self, _query: &str) -> std::result::Result<Vec<f32>, String> {
        Ok(self.embedding.clone())
    }
}

#[tonic::async_trait]
impl proto::mnemos_service_server::MnemosService for MnemosGrpcService {
    async fn insert_memory(
        &self,
        request: Request<proto::InsertMemoryRequest>,
    ) -> Result<Response<proto::InsertMemoryResponse>, Status> {
        let req = request.into_inner();
        let mem = req
            .memory
            .ok_or_else(|| Status::invalid_argument("memory is required"))?;

        let metadata: HashMap<String, String> = mem
            .metadata
            .into_iter()
            .map(|kv| (kv.key, kv.value))
            .collect();

        let mut entry =
            MemoryEntry::new(MemoryId(mem.id), mem.namespace, mem.content, mem.created_at)
                .with_importance(mem.importance);
        if !mem.embedding.is_empty() {
            entry = entry.with_embedding(mem.embedding);
        }
        entry.metadata = metadata;

        let mut store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let cmd_id = store
            .insert_memory(entry)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::InsertMemoryResponse {
            command_id: cmd_id.0,
        }))
    }

    async fn delete_memory(
        &self,
        request: Request<proto::DeleteMemoryRequest>,
    ) -> Result<Response<proto::DeleteMemoryResponse>, Status> {
        let req = request.into_inner();
        let mut store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let cmd_id = store
            .delete_memory(MemoryId(req.id))
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::DeleteMemoryResponse {
            command_id: cmd_id.0,
        }))
    }

    async fn add_edge(
        &self,
        request: Request<proto::AddEdgeRequest>,
    ) -> Result<Response<proto::AddEdgeResponse>, Status> {
        let req = request.into_inner();
        let mut store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let cmd_id = store
            .add_edge(MemoryId(req.from), MemoryId(req.to), req.relation)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::AddEdgeResponse {
            command_id: cmd_id.0,
        }))
    }

    async fn remove_edge(
        &self,
        request: Request<proto::RemoveEdgeRequest>,
    ) -> Result<Response<proto::RemoveEdgeResponse>, Status> {
        let req = request.into_inner();
        let mut store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let cmd_id = store
            .remove_edge(MemoryId(req.from), MemoryId(req.to))
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::RemoveEdgeResponse {
            command_id: cmd_id.0,
        }))
    }

    async fn query(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<proto::QueryResponse>, Status> {
        let req = request.into_inner();
        if req.query_embedding.is_empty() {
            return Err(Status::invalid_argument("query_embedding is required"));
        }

        let top_k =
            usize::try_from(req.top_k).map_err(|_| Status::invalid_argument("invalid top_k"))?;
        let mut options = QueryOptions::with_top_k(top_k);
        options.namespace = req.namespace;
        if let (Some(start), Some(end)) = (req.time_start, req.time_end) {
            options.time_range = Some((start, end));
        }
        if let Some(hops) = req.graph_hops {
            let hops = usize::try_from(hops)
                .map_err(|_| Status::invalid_argument("invalid graph_hops"))?;
            options.graph_expansion = Some(GraphExpansionOptions::new(hops));
        }
        if req.candidate_multiplier > 0 {
            options.candidate_multiplier = usize::try_from(req.candidate_multiplier)
                .map_err(|_| Status::invalid_argument("invalid candidate_multiplier"))?;
        }
        if req.similarity_pct > 0 || req.importance_pct > 0 || req.recency_pct > 0 {
            options.score_weights = ScoreWeights::new(
                req.similarity_pct as u8,
                req.importance_pct as u8,
                req.recency_pct as u8,
            );
        }

        let embedder = StaticEmbedder {
            embedding: req.query_embedding,
        };

        let store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let out = store
            .query("grpc_query", options, &embedder)
            .map_err(|e| Status::internal(e.to_string()))?;

        let hits = out
            .hits
            .into_iter()
            .map(|h| proto::QueryHit {
                id: h.id.0,
                final_score: h.final_score,
                similarity_score: h.similarity_score,
                importance_score: h.importance_score,
                recency_score: h.recency_score,
            })
            .collect();

        Ok(Response::new(proto::QueryResponse {
            hits,
            execution_path: format!("{:?}", out.metrics.plan_path),
            used_parallel: out.metrics.used_parallel,
            vector_candidates: u32::try_from(out.metrics.vector_candidates).unwrap_or(u32::MAX),
            filtered_candidates: u32::try_from(out.metrics.filtered_candidates).unwrap_or(u32::MAX),
            final_results: u32::try_from(out.metrics.final_results).unwrap_or(u32::MAX),
            elapsed_ms: u64::try_from(out.metrics.elapsed_ms).unwrap_or(u64::MAX),
        }))
    }

    async fn enforce_capacity(
        &self,
        request: Request<proto::EnforceCapacityRequest>,
    ) -> Result<Response<proto::EnforceCapacityResponse>, Status> {
        let req = request.into_inner();
        let policy = CapacityPolicy::new(
            req.max_entries.and_then(|v| usize::try_from(v).ok()),
            req.max_bytes,
        );

        let mut store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let report = store
            .enforce_capacity(policy)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::EnforceCapacityResponse {
            evicted_ids: report.evicted_ids.into_iter().map(|id| id.0).collect(),
            entries_before: u64::try_from(report.entries_before).unwrap_or(u64::MAX),
            entries_after: u64::try_from(report.entries_after).unwrap_or(u64::MAX),
            bytes_before: report.bytes_before,
            bytes_after: report.bytes_after,
        }))
    }

    async fn compact_segments(
        &self,
        _request: Request<proto::CompactSegmentsRequest>,
    ) -> Result<Response<proto::CompactSegmentsResponse>, Status> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        let report = store
            .compact_segments()
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::CompactSegmentsResponse {
            compacted_segments: report.compacted_segments,
            live_entries_rewritten: u64::try_from(report.live_entries_rewritten)
                .unwrap_or(u64::MAX),
        }))
    }

    async fn stats(
        &self,
        _request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let store = self
            .store
            .lock()
            .map_err(|_| Status::internal("store lock poisoned"))?;
        Ok(Response::new(proto::StatsResponse {
            wal_len: store.wal_len(),
            state_entries: u64::try_from(store.state_machine().len()).unwrap_or(u64::MAX),
            indexed_embeddings: u64::try_from(store.indexed_embeddings()).unwrap_or(u64::MAX),
        }))
    }
}
