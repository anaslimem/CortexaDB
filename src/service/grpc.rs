use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

use crate::core::memory_entry::{MemoryEntry, MemoryId};
use crate::engine::CapacityPolicy;
use crate::index::vector::VectorError;
use crate::query::{GraphExpansionOptions, QueryOptions, ScoreWeights};
use crate::store::{MnemosStore, MnemosStoreError};

pub mod proto {
    tonic::include_proto!("mnemos");
}

pub use proto::mnemos_service_server::MnemosServiceServer;

pub trait AuthProvider: Send + Sync + 'static {
    fn authorize(&self, metadata: &tonic::metadata::MetadataMap) -> Result<(), Status>;
}

pub struct AllowAllAuthProvider;
impl AuthProvider for AllowAllAuthProvider {
    fn authorize(&self, _metadata: &tonic::metadata::MetadataMap) -> Result<(), Status> {
        Ok(())
    }
}

pub struct ApiKeyAuthProvider {
    api_key: String,
}

impl ApiKeyAuthProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
    }
}

impl AuthProvider for ApiKeyAuthProvider {
    fn authorize(&self, metadata: &tonic::metadata::MetadataMap) -> Result<(), Status> {
        // Accept either `x-api-key: <key>` or `authorization: Bearer <key>`.
        if let Some(v) = metadata.get("x-api-key").and_then(|v| v.to_str().ok()) {
            if v == self.api_key {
                return Ok(());
            }
        }
        if let Some(v) = metadata.get("authorization").and_then(|v| v.to_str().ok()) {
            if let Some(token) = v.strip_prefix("Bearer ") {
                if token == self.api_key {
                    return Ok(());
                }
            }
        }
        Err(Status::unauthenticated("missing/invalid API key"))
    }
}

#[derive(Default)]
struct ServiceMetrics {
    rpc_total: AtomicU64,
    writes_deduplicated: AtomicU64,
    writes_committed: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum WriteKind {
    InsertMemory,
    DeleteMemory,
    AddEdge,
    RemoveEdge,
    EnforceCapacity,
    CompactSegments,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IdempotencyKey {
    kind: WriteKind,
    request_id: String,
}

#[derive(Debug, Clone)]
struct IdempotencyRecord {
    payload_hash: u64,
    result: WriteResult,
}

#[derive(Clone)]
pub struct MnemosGrpcService {
    store: Arc<MnemosStore>,
    writer_tx: mpsc::Sender<WriteRequest>,
    auth: Arc<dyn AuthProvider>,
    metrics: Arc<ServiceMetrics>,
    idempotency: Arc<std::sync::Mutex<HashMap<IdempotencyKey, IdempotencyRecord>>>,
}

impl MnemosGrpcService {
    pub fn new(store: MnemosStore) -> Self {
        Self::new_with_auth(store, Arc::new(AllowAllAuthProvider))
    }

    pub fn new_with_auth(store: MnemosStore, auth: Arc<dyn AuthProvider>) -> Self {
        let store = Arc::new(store);
        let (writer_tx, mut writer_rx) = mpsc::channel::<WriteRequest>(1024);

        let writer_store = Arc::clone(&store);
        tokio::spawn(async move {
            while let Some(req) = writer_rx.recv().await {
                let result: Result<WriteResult, MnemosStoreError> = match req.command {
                    WriteCommand::InsertMemory(entry) => writer_store
                        .insert_memory(entry)
                        .map(|id| WriteResult::CommandId(id.0))
                        .map_err(|e| e),
                    WriteCommand::DeleteMemory(id) => writer_store
                        .delete_memory(id)
                        .map(|id| WriteResult::CommandId(id.0))
                        .map_err(|e| e),
                    WriteCommand::AddEdge { from, to, relation } => writer_store
                        .add_edge(from, to, relation)
                        .map(|id| WriteResult::CommandId(id.0))
                        .map_err(|e| e),
                    WriteCommand::RemoveEdge { from, to } => writer_store
                        .remove_edge(from, to)
                        .map(|id| WriteResult::CommandId(id.0))
                        .map_err(|e| e),
                    WriteCommand::EnforceCapacity(policy) => writer_store
                        .enforce_capacity(policy)
                        .map(WriteResult::Capacity)
                        .map_err(|e| e),
                    WriteCommand::CompactSegments => writer_store
                        .compact_segments()
                        .map(WriteResult::Compact)
                        .map_err(|e| e),
                };

                let _ = req.reply.send(result);
            }
        });

        Self {
            store,
            writer_tx,
            auth,
            metrics: Arc::new(ServiceMetrics::default()),
            idempotency: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    async fn send_write(&self, command: WriteCommand) -> Result<WriteResult, Status> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.writer_tx
            .send(WriteRequest {
                command,
                reply: reply_tx,
            })
            .await
            .map_err(|_| Status::unavailable("writer queue is closed"))?;

        let out = reply_rx
            .await
            .map_err(|_| Status::internal("writer task dropped reply"))?;
        out.map_err(map_store_error)
    }

    fn authorize_request<T>(&self, req: &Request<T>) -> Result<(), Status> {
        self.auth.authorize(req.metadata())
    }

    async fn execute_write_idempotent(
        &self,
        request_id: Option<String>,
        kind: WriteKind,
        payload_hash: u64,
        command: WriteCommand,
    ) -> Result<WriteResult, Status> {
        if let Some(req_id) = request_id {
            let key = IdempotencyKey {
                kind,
                request_id: req_id,
            };
            {
                let cache = self.idempotency.lock().expect("idempotency lock poisoned");
                if let Some(record) = cache.get(&key) {
                    if record.payload_hash != payload_hash {
                        return Err(Status::invalid_argument(
                            "request_id was reused with different payload",
                        ));
                    }
                    self.metrics
                        .writes_deduplicated
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(record.result.clone());
                }
            }

            let result = self.send_write(command).await?;
            {
                let mut cache = self.idempotency.lock().expect("idempotency lock poisoned");
                cache.insert(
                    key,
                    IdempotencyRecord {
                        payload_hash,
                        result: result.clone(),
                    },
                );
            }
            self.metrics
                .writes_committed
                .fetch_add(1, Ordering::Relaxed);
            Ok(result)
        } else {
            let result = self.send_write(command).await?;
            self.metrics
                .writes_committed
                .fetch_add(1, Ordering::Relaxed);
            Ok(result)
        }
    }
}

struct WriteRequest {
    command: WriteCommand,
    reply: oneshot::Sender<Result<WriteResult, MnemosStoreError>>,
}

enum WriteCommand {
    InsertMemory(MemoryEntry),
    DeleteMemory(MemoryId),
    AddEdge {
        from: MemoryId,
        to: MemoryId,
        relation: String,
    },
    RemoveEdge {
        from: MemoryId,
        to: MemoryId,
    },
    EnforceCapacity(CapacityPolicy),
    CompactSegments,
}

#[derive(Debug, Clone)]
enum WriteResult {
    CommandId(u64),
    Capacity(crate::engine::EvictionReport),
    Compact(crate::storage::compaction::CompactionReport),
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
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        let started = Instant::now();
        let req_id = extract_request_id(&request);
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

        let payload_hash = hash_insert_payload(&entry);
        let cmd_id = match self
            .execute_write_idempotent(
                req_id,
                WriteKind::InsertMemory,
                payload_hash,
                WriteCommand::InsertMemory(entry),
            )
            .await?
        {
            WriteResult::CommandId(id) => id,
            _ => return Err(Status::internal("unexpected writer response for insert")),
        };
        eprintln!("insert_memory completed in {}ms", started.elapsed().as_millis());
        Ok(Response::new(proto::InsertMemoryResponse {
            command_id: cmd_id,
        }))
    }

    async fn delete_memory(
        &self,
        request: Request<proto::DeleteMemoryRequest>,
    ) -> Result<Response<proto::DeleteMemoryResponse>, Status> {
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        let req_id = extract_request_id(&request);
        let req = request.into_inner();
        let cmd_id = match self
            .execute_write_idempotent(
                req_id,
                WriteKind::DeleteMemory,
                hash_delete_payload(req.id),
                WriteCommand::DeleteMemory(MemoryId(req.id)),
            )
            .await?
        {
            WriteResult::CommandId(id) => id,
            _ => return Err(Status::internal("unexpected writer response for delete")),
        };
        Ok(Response::new(proto::DeleteMemoryResponse {
            command_id: cmd_id,
        }))
    }

    async fn add_edge(
        &self,
        request: Request<proto::AddEdgeRequest>,
    ) -> Result<Response<proto::AddEdgeResponse>, Status> {
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        let req_id = extract_request_id(&request);
        let req = request.into_inner();
        let cmd_id = match self
            .execute_write_idempotent(
                req_id,
                WriteKind::AddEdge,
                hash_add_edge_payload(req.from, req.to, &req.relation),
                WriteCommand::AddEdge {
                    from: MemoryId(req.from),
                    to: MemoryId(req.to),
                    relation: req.relation,
                },
            )
            .await?
        {
            WriteResult::CommandId(id) => id,
            _ => return Err(Status::internal("unexpected writer response for add_edge")),
        };
        Ok(Response::new(proto::AddEdgeResponse { command_id: cmd_id }))
    }

    async fn remove_edge(
        &self,
        request: Request<proto::RemoveEdgeRequest>,
    ) -> Result<Response<proto::RemoveEdgeResponse>, Status> {
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        let req_id = extract_request_id(&request);
        let req = request.into_inner();
        let cmd_id = match self
            .execute_write_idempotent(
                req_id,
                WriteKind::RemoveEdge,
                hash_remove_edge_payload(req.from, req.to),
                WriteCommand::RemoveEdge {
                    from: MemoryId(req.from),
                    to: MemoryId(req.to),
                },
            )
            .await?
        {
            WriteResult::CommandId(id) => id,
            _ => {
                return Err(Status::internal(
                    "unexpected writer response for remove_edge",
                ));
            }
        };
        Ok(Response::new(proto::RemoveEdgeResponse {
            command_id: cmd_id,
        }))
    }

    async fn query(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<proto::QueryResponse>, Status> {
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
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
            options.graph_expansion = if hops == 0 {
                None
            } else {
                Some(GraphExpansionOptions::new(hops))
            };
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

        let (out, snapshot) = self
            .store
            .query_with_snapshot("grpc_query", options, &embedder)
            .map_err(map_store_error)?;

        let hits = out
            .hits
            .into_iter()
            .map(|h| {
                let memory =
                    snapshot
                        .state_machine()
                        .get_memory(h.id)
                        .ok()
                        .map(|m| proto::Memory {
                            id: m.id.0,
                            namespace: m.namespace.clone(),
                            content: m.content.clone(),
                            embedding: m.embedding.clone().unwrap_or_default(),
                            created_at: m.created_at,
                            importance: m.importance,
                            metadata: m
                                .metadata
                                .iter()
                                .map(|(k, v)| proto::MetadataPair {
                                    key: k.clone(),
                                    value: v.clone(),
                                })
                                .collect(),
                        });
                proto::QueryHit {
                    id: h.id.0,
                    final_score: h.final_score,
                    similarity_score: h.similarity_score,
                    importance_score: h.importance_score,
                    recency_score: h.recency_score,
                    memory,
                }
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
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        let req_id = extract_request_id(&request);
        let req = request.into_inner();
        let policy = CapacityPolicy::new(
            req.max_entries.and_then(|v| usize::try_from(v).ok()),
            req.max_bytes,
        );

        let report = match self
            .execute_write_idempotent(
                req_id,
                WriteKind::EnforceCapacity,
                hash_capacity_payload(req.max_entries, req.max_bytes),
                WriteCommand::EnforceCapacity(policy),
            )
            .await?
        {
            WriteResult::Capacity(report) => report,
            _ => {
                return Err(Status::internal(
                    "unexpected writer response for enforce_capacity",
                ));
            }
        };

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
        request: Request<proto::CompactSegmentsRequest>,
    ) -> Result<Response<proto::CompactSegmentsResponse>, Status> {
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        let req_id = extract_request_id(&request);
        let report = match self
            .execute_write_idempotent(
                req_id,
                WriteKind::CompactSegments,
                0,
                WriteCommand::CompactSegments,
            )
            .await?
        {
            WriteResult::Compact(report) => report,
            _ => {
                return Err(Status::internal(
                    "unexpected writer response for compact_segments",
                ));
            }
        };
        Ok(Response::new(proto::CompactSegmentsResponse {
            compacted_segments: report.compacted_segments,
            live_entries_rewritten: u64::try_from(report.live_entries_rewritten)
                .unwrap_or(u64::MAX),
        }))
    }

    async fn stats(
        &self,
        request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        self.metrics.rpc_total.fetch_add(1, Ordering::Relaxed);
        self.authorize_request(&request)?;
        Ok(Response::new(proto::StatsResponse {
            wal_len: self.store.wal_len(),
            state_entries: u64::try_from(self.store.state_machine().len()).unwrap_or(u64::MAX),
            indexed_embeddings: u64::try_from(self.store.indexed_embeddings()).unwrap_or(u64::MAX),
        }))
    }
}

fn extract_request_id<T>(request: &Request<T>) -> Option<String> {
    request
        .metadata()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn hash_insert_payload(entry: &MemoryEntry) -> u64 {
    let mut h = DefaultHasher::new();
    entry.id.hash(&mut h);
    entry.namespace.hash(&mut h);
    entry.content.hash(&mut h);
    if let Some(embedding) = &entry.embedding {
        embedding.len().hash(&mut h);
        for v in embedding {
            v.to_bits().hash(&mut h);
        }
    } else {
        0usize.hash(&mut h);
    }
    entry.created_at.hash(&mut h);
    entry.importance.to_bits().hash(&mut h);
    let mut meta: Vec<_> = entry.metadata.iter().collect();
    meta.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));
    for (k, v) in meta {
        k.hash(&mut h);
        v.hash(&mut h);
    }
    h.finish()
}

fn hash_delete_payload(id: u64) -> u64 {
    let mut h = DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}

fn hash_add_edge_payload(from: u64, to: u64, relation: &str) -> u64 {
    let mut h = DefaultHasher::new();
    from.hash(&mut h);
    to.hash(&mut h);
    relation.hash(&mut h);
    h.finish()
}

fn hash_remove_edge_payload(from: u64, to: u64) -> u64 {
    let mut h = DefaultHasher::new();
    from.hash(&mut h);
    to.hash(&mut h);
    h.finish()
}

fn hash_capacity_payload(max_entries: Option<u64>, max_bytes: Option<u64>) -> u64 {
    let mut h = DefaultHasher::new();
    max_entries.hash(&mut h);
    max_bytes.hash(&mut h);
    h.finish()
}

fn map_store_error(err: MnemosStoreError) -> Status {
    match err {
        MnemosStoreError::MissingEmbeddingOnContentChange(_) => {
            Status::invalid_argument(err.to_string())
        }
        MnemosStoreError::Vector(VectorError::DimensionMismatch { .. })
        | MnemosStoreError::Vector(VectorError::InvalidTopK(_))
        | MnemosStoreError::Vector(VectorError::EmptyQuery)
        | MnemosStoreError::Vector(VectorError::ZeroVector) => {
            Status::invalid_argument(err.to_string())
        }
        MnemosStoreError::Engine(crate::engine::EngineError::StateMachineError(
            crate::core::state_machine::StateMachineError::MemoryNotFound(_),
        )) => Status::not_found(err.to_string()),
        MnemosStoreError::Engine(crate::engine::EngineError::StateMachineError(
            crate::core::state_machine::StateMachineError::CrossNamespaceEdge { .. },
        ))
        | MnemosStoreError::Engine(crate::engine::EngineError::CheckpointWalGap { .. })
        | MnemosStoreError::InvariantViolation(_) => Status::failed_precondition(err.to_string()),
        MnemosStoreError::Query(crate::query::HybridQueryError::InvalidTopK(_))
        | MnemosStoreError::Query(crate::query::HybridQueryError::InvalidCandidateMultiplier(_))
        | MnemosStoreError::Query(crate::query::HybridQueryError::InvalidScoreWeights { .. })
        | MnemosStoreError::Query(crate::query::HybridQueryError::InvalidTimeRange { .. }) => {
            Status::invalid_argument(err.to_string())
        }
        _ => Status::internal(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;
    use tonic::metadata::MetadataValue;
    use tonic::{Code, Request};

    use super::proto::mnemos_service_server::MnemosService;
    use super::*;

    fn test_store() -> MnemosStore {
        let temp = TempDir::new().expect("tempdir");
        let wal = temp.path().join("service.wal");
        let seg = temp.path().join("segments");
        std::fs::create_dir_all(&seg).expect("segments dir");

        // Keep tempdir alive by moving owned paths into the store constructor.
        MnemosStore::new(wal, seg, 3).expect("create store")
    }

    fn test_memory(id: u64, namespace: &str, content: &[u8]) -> proto::Memory {
        proto::Memory {
            id,
            namespace: namespace.to_string(),
            content: content.to_vec(),
            embedding: vec![0.1, 0.2, 0.3],
            created_at: 1_700_000_000,
            importance: 0.6,
            metadata: vec![],
        }
    }

    fn with_request_id<T>(mut req: Request<T>, request_id: &str) -> Request<T> {
        req.metadata_mut().insert(
            "x-request-id",
            MetadataValue::try_from(request_id).expect("valid metadata"),
        );
        req
    }

    struct BearerTokenAuth;
    impl AuthProvider for BearerTokenAuth {
        fn authorize(&self, metadata: &tonic::metadata::MetadataMap) -> Result<(), Status> {
            match metadata.get("authorization").and_then(|v| v.to_str().ok()) {
                Some(v) if v == "Bearer test-token" => Ok(()),
                _ => Err(Status::unauthenticated("missing/invalid authorization")),
            }
        }
    }

    #[tokio::test]
    async fn insert_idempotency_reuses_result_for_same_request_id() {
        let service = MnemosGrpcService::new(test_store());

        let req = proto::InsertMemoryRequest {
            memory: Some(test_memory(1, "agent1", b"hello")),
        };

        let first = MnemosService::insert_memory(
            &service,
            with_request_id(Request::new(req.clone()), "req-1"),
        )
        .await
        .expect("first insert")
        .into_inner();

        let second = MnemosService::insert_memory(
            &service,
            with_request_id(Request::new(req), "req-1"),
        )
        .await
        .expect("second insert")
        .into_inner();

        assert_eq!(first.command_id, second.command_id);

        let stats = MnemosService::stats(&service, Request::new(proto::StatsRequest {}))
            .await
            .expect("stats")
            .into_inner();
        assert_eq!(stats.state_entries, 1);
    }

    #[tokio::test]
    async fn insert_idempotency_rejects_payload_mismatch() {
        let service = MnemosGrpcService::new(test_store());
        let req_a = proto::InsertMemoryRequest {
            memory: Some(test_memory(10, "agent1", b"first")),
        };
        let req_b = proto::InsertMemoryRequest {
            memory: Some(test_memory(11, "agent1", b"second")),
        };

        MnemosService::insert_memory(
            &service,
            with_request_id(Request::new(req_a), "req-shared"),
        )
        .await
        .expect("first insert");

        let err = MnemosService::insert_memory(
            &service,
            with_request_id(Request::new(req_b), "req-shared"),
        )
        .await
        .expect_err("payload mismatch must fail");
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[tokio::test]
    async fn auth_provider_is_enforced() {
        let service = MnemosGrpcService::new_with_auth(test_store(), Arc::new(BearerTokenAuth));

        let err = MnemosService::stats(&service, Request::new(proto::StatsRequest {}))
            .await
            .expect_err("missing token must fail");
        assert_eq!(err.code(), Code::Unauthenticated);

        let mut req = Request::new(proto::StatsRequest {});
        req.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from("Bearer test-token").expect("valid metadata"),
        );
        let ok = MnemosService::stats(&service, req).await;
        assert!(ok.is_ok());
    }

    #[tokio::test]
    async fn add_edge_cross_namespace_maps_to_failed_precondition() {
        let service = MnemosGrpcService::new(test_store());

        MnemosService::insert_memory(
            &service,
            Request::new(proto::InsertMemoryRequest {
                memory: Some(test_memory(1, "ns-a", b"a")),
            }),
        )
        .await
        .expect("insert a");

        MnemosService::insert_memory(
            &service,
            Request::new(proto::InsertMemoryRequest {
                memory: Some(test_memory(2, "ns-b", b"b")),
            }),
        )
        .await
        .expect("insert b");

        let err = MnemosService::add_edge(
            &service,
            Request::new(proto::AddEdgeRequest {
                from: 1,
                to: 2,
                relation: "link".to_string(),
            }),
        )
        .await
        .expect_err("cross-namespace edge must fail");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[test]
    fn api_key_auth_provider_accepts_expected_headers() {
        let auth = ApiKeyAuthProvider::new("secret-key");
        let mut meta = tonic::metadata::MetadataMap::new();

        meta.insert(
            "x-api-key",
            MetadataValue::try_from("secret-key").expect("valid metadata"),
        );
        assert!(auth.authorize(&meta).is_ok());

        let mut bearer = tonic::metadata::MetadataMap::new();
        bearer.insert(
            "authorization",
            MetadataValue::try_from("Bearer secret-key").expect("valid metadata"),
        );
        assert!(auth.authorize(&bearer).is_ok());

        let mut bad = tonic::metadata::MetadataMap::new();
        bad.insert(
            "x-api-key",
            MetadataValue::try_from("wrong").expect("valid metadata"),
        );
        assert_eq!(
            auth.authorize(&bad).expect_err("must fail").code(),
            Code::Unauthenticated
        );
    }
}
