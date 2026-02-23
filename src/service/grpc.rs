use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tonic::{Code, Request, Response, Status};

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
pub struct ServiceMetrics {
    rpc_total: AtomicUsize,
    rpc_error_total: AtomicUsize,
    writes_deduplicated: AtomicUsize,
    writes_committed: AtomicUsize,
    method_calls: std::sync::Mutex<HashMap<&'static str, usize>>,
    method_errors: std::sync::Mutex<HashMap<&'static str, usize>>,
    method_latency_ms_sum: std::sync::Mutex<HashMap<&'static str, u128>>,
    method_latency_buckets: std::sync::Mutex<HashMap<&'static str, [usize; 6]>>,
    query_vector_candidates_sum: AtomicUsize,
    query_final_results_sum: AtomicUsize,
}

impl ServiceMetrics {
    fn record_rpc(&self, method: &'static str, status: Code, elapsed: Duration) {
        self.rpc_total.fetch_add(1, Ordering::Relaxed);
        {
            let mut calls = self.method_calls.lock().expect("metrics lock poisoned");
            *calls.entry(method).or_insert(0) += 1;
        }

        let elapsed_ms = elapsed.as_millis();
        {
            let mut sum = self
                .method_latency_ms_sum
                .lock()
                .expect("metrics lock poisoned");
            *sum.entry(method).or_insert(0) += elapsed_ms;
        }
        {
            let mut buckets = self
                .method_latency_buckets
                .lock()
                .expect("metrics lock poisoned");
            let entry = buckets.entry(method).or_insert([0; 6]);
            let idx = match elapsed_ms {
                0..=5 => 0,
                6..=10 => 1,
                11..=25 => 2,
                26..=50 => 3,
                51..=100 => 4,
                _ => 5,
            };
            entry[idx] += 1;
        }

        if status != Code::Ok {
            self.rpc_error_total.fetch_add(1, Ordering::Relaxed);
            let mut errors = self.method_errors.lock().expect("metrics lock poisoned");
            *errors.entry(method).or_insert(0) += 1;
        }
    }

    fn observe_query_result(&self, vector_candidates: usize, final_results: usize) {
        self.query_vector_candidates_sum
            .fetch_add(vector_candidates, Ordering::Relaxed);
        self.query_final_results_sum
            .fetch_add(final_results, Ordering::Relaxed);
    }

    fn render_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP mnemos_rpc_total Total gRPC requests\n");
        out.push_str("# TYPE mnemos_rpc_total counter\n");
        out.push_str(&format!(
            "mnemos_rpc_total {}\n",
            self.rpc_total.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP mnemos_rpc_errors_total Total gRPC failed requests\n");
        out.push_str("# TYPE mnemos_rpc_errors_total counter\n");
        out.push_str(&format!(
            "mnemos_rpc_errors_total {}\n",
            self.rpc_error_total.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP mnemos_writes_committed_total Total committed writes\n");
        out.push_str("# TYPE mnemos_writes_committed_total counter\n");
        out.push_str(&format!(
            "mnemos_writes_committed_total {}\n",
            self.writes_committed.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP mnemos_writes_deduplicated_total Total deduplicated idempotent writes\n");
        out.push_str("# TYPE mnemos_writes_deduplicated_total counter\n");
        out.push_str(&format!(
            "mnemos_writes_deduplicated_total {}\n",
            self.writes_deduplicated.load(Ordering::Relaxed)
        ));

        let calls = self.method_calls.lock().expect("metrics lock poisoned").clone();
        let errors = self
            .method_errors
            .lock()
            .expect("metrics lock poisoned")
            .clone();
        let sums = self
            .method_latency_ms_sum
            .lock()
            .expect("metrics lock poisoned")
            .clone();
        let buckets = self
            .method_latency_buckets
            .lock()
            .expect("metrics lock poisoned")
            .clone();

        out.push_str("# HELP mnemos_rpc_method_calls_total Calls per method\n");
        out.push_str("# TYPE mnemos_rpc_method_calls_total counter\n");
        for (method, value) in calls {
            out.push_str(&format!(
                "mnemos_rpc_method_calls_total{{method=\"{}\"}} {}\n",
                method, value
            ));
        }
        out.push_str("# HELP mnemos_rpc_method_errors_total Errors per method\n");
        out.push_str("# TYPE mnemos_rpc_method_errors_total counter\n");
        for (method, value) in errors {
            out.push_str(&format!(
                "mnemos_rpc_method_errors_total{{method=\"{}\"}} {}\n",
                method, value
            ));
        }
        out.push_str("# HELP mnemos_rpc_method_latency_ms_sum Summed method latencies in ms\n");
        out.push_str("# TYPE mnemos_rpc_method_latency_ms_sum counter\n");
        for (method, value) in sums {
            out.push_str(&format!(
                "mnemos_rpc_method_latency_ms_sum{{method=\"{}\"}} {}\n",
                method, value
            ));
        }
        out.push_str("# HELP mnemos_rpc_method_latency_bucket Latency bucket counts\n");
        out.push_str("# TYPE mnemos_rpc_method_latency_bucket counter\n");
        let bounds = ["5", "10", "25", "50", "100", "+Inf"];
        for (method, vals) in buckets {
            for (i, bound) in bounds.iter().enumerate() {
                out.push_str(&format!(
                    "mnemos_rpc_method_latency_bucket{{method=\"{}\",le=\"{}\"}} {}\n",
                    method, bound, vals[i]
                ));
            }
        }

        out.push_str("# HELP mnemos_query_vector_candidates_sum Sum of query vector candidates\n");
        out.push_str("# TYPE mnemos_query_vector_candidates_sum counter\n");
        out.push_str(&format!(
            "mnemos_query_vector_candidates_sum {}\n",
            self.query_vector_candidates_sum.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP mnemos_query_final_results_sum Sum of query final results\n");
        out.push_str("# TYPE mnemos_query_final_results_sum counter\n");
        out.push_str(&format!(
            "mnemos_query_final_results_sum {}\n",
            self.query_final_results_sum.load(Ordering::Relaxed)
        ));

        out
    }
}

#[derive(Debug, Clone)]
pub struct RbacPolicy {
    pub admin_principals: HashSet<String>,
    pub read_namespace_allow: HashMap<String, HashSet<String>>,
    pub write_namespace_allow: HashMap<String, HashSet<String>>,
}

impl RbacPolicy {
    pub fn allow_all() -> Self {
        Self {
            admin_principals: HashSet::new(),
            read_namespace_allow: HashMap::new(),
            write_namespace_allow: HashMap::new(),
        }
    }

    fn is_allowed(&self, principal: &str, namespace: &str, write: bool) -> bool {
        if self.admin_principals.contains(principal) {
            return true;
        }
        let set = if write {
            self.write_namespace_allow.get(namespace)
        } else {
            self.read_namespace_allow.get(namespace)
        };
        match set {
            Some(principals) => principals.contains(principal),
            None => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuotaPolicy {
    pub max_requests_per_minute: usize,
    pub max_top_k: usize,
    pub max_graph_hops: usize,
}

impl Default for QuotaPolicy {
    fn default() -> Self {
        Self {
            max_requests_per_minute: 0,
            max_top_k: 100,
            max_graph_hops: 4,
        }
    }
}

#[derive(Debug, Clone)]
struct PrincipalWindow {
    window_started_at: Instant,
    count: usize,
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
    rbac: Arc<RbacPolicy>,
    quota: Arc<QuotaPolicy>,
    quota_windows: Arc<std::sync::Mutex<HashMap<String, PrincipalWindow>>>,
    idempotency: Arc<std::sync::Mutex<HashMap<IdempotencyKey, IdempotencyRecord>>>,
}

impl MnemosGrpcService {
    pub fn new(store: MnemosStore) -> Self {
        Self::new_with_config(
            store,
            Arc::new(AllowAllAuthProvider),
            RbacPolicy::allow_all(),
            QuotaPolicy::default(),
        )
    }

    pub fn new_with_auth(store: MnemosStore, auth: Arc<dyn AuthProvider>) -> Self {
        Self::new_with_config(store, auth, RbacPolicy::allow_all(), QuotaPolicy::default())
    }

    pub fn new_with_config(
        store: MnemosStore,
        auth: Arc<dyn AuthProvider>,
        rbac: RbacPolicy,
        quota: QuotaPolicy,
    ) -> Self {
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
            rbac: Arc::new(rbac),
            quota: Arc::new(quota),
            quota_windows: Arc::new(std::sync::Mutex::new(HashMap::new())),
            idempotency: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub fn metrics(&self) -> Arc<ServiceMetrics> {
        Arc::clone(&self.metrics)
    }

    pub fn render_prometheus_metrics(&self) -> String {
        self.metrics.render_prometheus()
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
        self.auth.authorize(req.metadata())?;
        self.check_quota(req.metadata())
    }

    fn principal_from_metadata(metadata: &tonic::metadata::MetadataMap) -> String {
        metadata
            .get("x-principal-id")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "anonymous".to_string())
    }

    fn check_quota(&self, metadata: &tonic::metadata::MetadataMap) -> Result<(), Status> {
        if self.quota.max_requests_per_minute == 0 {
            return Ok(());
        }
        let principal = Self::principal_from_metadata(metadata);
        let mut windows = self.quota_windows.lock().expect("quota lock poisoned");
        let entry = windows.entry(principal).or_insert(PrincipalWindow {
            window_started_at: Instant::now(),
            count: 0,
        });
        if entry.window_started_at.elapsed() >= Duration::from_secs(60) {
            entry.window_started_at = Instant::now();
            entry.count = 0;
        }
        if entry.count >= self.quota.max_requests_per_minute {
            return Err(Status::resource_exhausted("request quota exceeded"));
        }
        entry.count += 1;
        Ok(())
    }

    fn check_namespace_access(
        &self,
        metadata: &tonic::metadata::MetadataMap,
        namespace: &str,
        write: bool,
    ) -> Result<(), Status> {
        let principal = Self::principal_from_metadata(metadata);
        if self.rbac.is_allowed(&principal, namespace, write) {
            Ok(())
        } else {
            Err(Status::permission_denied(format!(
                "principal {} not allowed for namespace {}",
                principal, namespace
            )))
        }
    }

    fn record_rpc_result<T>(
        &self,
        method: &'static str,
        started: Instant,
        result: &Result<Response<T>, Status>,
    ) {
        let status = result.as_ref().map(|_| Code::Ok).unwrap_or_else(|e| e.code());
        self.metrics.record_rpc(method, status, started.elapsed());
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
        let started = Instant::now();
        let result = self.insert_memory_impl(request).await;
        self.record_rpc_result("insert_memory", started, &result);
        result
    }

    async fn delete_memory(
        &self,
        request: Request<proto::DeleteMemoryRequest>,
    ) -> Result<Response<proto::DeleteMemoryResponse>, Status> {
        let started = Instant::now();
        let result = self.delete_memory_impl(request).await;
        self.record_rpc_result("delete_memory", started, &result);
        result
    }

    async fn add_edge(
        &self,
        request: Request<proto::AddEdgeRequest>,
    ) -> Result<Response<proto::AddEdgeResponse>, Status> {
        let started = Instant::now();
        let result = self.add_edge_impl(request).await;
        self.record_rpc_result("add_edge", started, &result);
        result
    }

    async fn remove_edge(
        &self,
        request: Request<proto::RemoveEdgeRequest>,
    ) -> Result<Response<proto::RemoveEdgeResponse>, Status> {
        let started = Instant::now();
        let result = self.remove_edge_impl(request).await;
        self.record_rpc_result("remove_edge", started, &result);
        result
    }

    async fn query(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<proto::QueryResponse>, Status> {
        let started = Instant::now();
        let result = self.query_impl(request).await;
        self.record_rpc_result("query", started, &result);
        result
    }

    async fn enforce_capacity(
        &self,
        request: Request<proto::EnforceCapacityRequest>,
    ) -> Result<Response<proto::EnforceCapacityResponse>, Status> {
        let started = Instant::now();
        let result = self.enforce_capacity_impl(request).await;
        self.record_rpc_result("enforce_capacity", started, &result);
        result
    }

    async fn compact_segments(
        &self,
        request: Request<proto::CompactSegmentsRequest>,
    ) -> Result<Response<proto::CompactSegmentsResponse>, Status> {
        let started = Instant::now();
        let result = self.compact_segments_impl(request).await;
        self.record_rpc_result("compact_segments", started, &result);
        result
    }

    async fn stats(
        &self,
        request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let started = Instant::now();
        let result = self.stats_impl(request).await;
        self.record_rpc_result("stats", started, &result);
        result
    }
}

impl MnemosGrpcService {
    async fn insert_memory_impl(
        &self,
        request: Request<proto::InsertMemoryRequest>,
    ) -> Result<Response<proto::InsertMemoryResponse>, Status> {
        self.authorize_request(&request)?;
        let metadata = request.metadata().clone();
        let req_id = extract_request_id(&request);
        let req = request.into_inner();
        let mem = req
            .memory
            .ok_or_else(|| Status::invalid_argument("memory is required"))?;
        self.check_namespace_access(&metadata, &mem.namespace, true)?;

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
        Ok(Response::new(proto::InsertMemoryResponse {
            command_id: cmd_id,
        }))
    }

    async fn delete_memory_impl(
        &self,
        request: Request<proto::DeleteMemoryRequest>,
    ) -> Result<Response<proto::DeleteMemoryResponse>, Status> {
        self.authorize_request(&request)?;
        let id = request.get_ref().id;
        if let Ok(existing) = self.store.state_machine().get_memory(MemoryId(id)) {
            self.check_namespace_access(request.metadata(), &existing.namespace, true)?;
        }
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

    async fn add_edge_impl(
        &self,
        request: Request<proto::AddEdgeRequest>,
    ) -> Result<Response<proto::AddEdgeResponse>, Status> {
        self.authorize_request(&request)?;
        let from = request.get_ref().from;
        let to = request.get_ref().to;
        if let Ok(existing_from) = self.store.state_machine().get_memory(MemoryId(from)) {
            self.check_namespace_access(request.metadata(), &existing_from.namespace, true)?;
        }
        if let Ok(existing_to) = self.store.state_machine().get_memory(MemoryId(to)) {
            self.check_namespace_access(request.metadata(), &existing_to.namespace, true)?;
        }
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

    async fn remove_edge_impl(
        &self,
        request: Request<proto::RemoveEdgeRequest>,
    ) -> Result<Response<proto::RemoveEdgeResponse>, Status> {
        self.authorize_request(&request)?;
        let from = request.get_ref().from;
        let to = request.get_ref().to;
        if let Ok(existing_from) = self.store.state_machine().get_memory(MemoryId(from)) {
            self.check_namespace_access(request.metadata(), &existing_from.namespace, true)?;
        }
        if let Ok(existing_to) = self.store.state_machine().get_memory(MemoryId(to)) {
            self.check_namespace_access(request.metadata(), &existing_to.namespace, true)?;
        }
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

    async fn query_impl(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<proto::QueryResponse>, Status> {
        self.authorize_request(&request)?;
        let metadata = request.metadata().clone();
        let req = request.into_inner();
        if req.query_embedding.is_empty() {
            return Err(Status::invalid_argument("query_embedding is required"));
        }

        let top_k =
            usize::try_from(req.top_k).map_err(|_| Status::invalid_argument("invalid top_k"))?;
        if self.quota.max_top_k > 0 && top_k > self.quota.max_top_k {
            return Err(Status::resource_exhausted(format!(
                "top_k exceeds configured limit {}",
                self.quota.max_top_k
            )));
        }
        let mut options = QueryOptions::with_top_k(top_k);
        options.namespace = req.namespace;
        if let Some(ns) = options.namespace.as_deref() {
            self.check_namespace_access(&metadata, ns, false)?;
        }
        if let (Some(start), Some(end)) = (req.time_start, req.time_end) {
            options.time_range = Some((start, end));
        }
        if let Some(hops) = req.graph_hops {
            let hops = usize::try_from(hops)
                .map_err(|_| Status::invalid_argument("invalid graph_hops"))?;
            if self.quota.max_graph_hops > 0 && hops > self.quota.max_graph_hops {
                return Err(Status::resource_exhausted(format!(
                    "graph_hops exceeds configured limit {}",
                    self.quota.max_graph_hops
                )));
            }
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
        self.metrics
            .observe_query_result(out.metrics.vector_candidates, out.metrics.final_results);

        let hits = out
            .hits
            .into_iter()
            .map(|h| {
                let memory = snapshot.state_machine().get_memory(h.id).ok().map(|m| proto::Memory {
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

    async fn enforce_capacity_impl(
        &self,
        request: Request<proto::EnforceCapacityRequest>,
    ) -> Result<Response<proto::EnforceCapacityResponse>, Status> {
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

    async fn compact_segments_impl(
        &self,
        request: Request<proto::CompactSegmentsRequest>,
    ) -> Result<Response<proto::CompactSegmentsResponse>, Status> {
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

    async fn stats_impl(
        &self,
        request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
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
    use std::time::Duration;

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

    #[tokio::test]
    async fn query_respects_quota_limits() {
        let mut quota = QuotaPolicy::default();
        quota.max_top_k = 1;
        let service = MnemosGrpcService::new_with_config(
            test_store(),
            Arc::new(AllowAllAuthProvider),
            RbacPolicy::allow_all(),
            quota,
        );

        let err = MnemosService::query(
            &service,
            Request::new(proto::QueryRequest {
                query_embedding: vec![0.1, 0.2, 0.3],
                top_k: 10,
                namespace: Some("agent1".to_string()),
                time_start: None,
                time_end: None,
                graph_hops: None,
                candidate_multiplier: 0,
                similarity_pct: 0,
                importance_pct: 0,
                recency_pct: 0,
            }),
        )
        .await
        .expect_err("top_k should be limited");
        assert_eq!(err.code(), Code::ResourceExhausted);
    }

    #[test]
    fn metrics_render_contains_core_counters() {
        let metrics = ServiceMetrics::default();
        metrics.record_rpc("stats", Code::Ok, Duration::from_millis(5));
        let body = metrics.render_prometheus();
        assert!(body.contains("mnemos_rpc_total"));
        assert!(body.contains("mnemos_rpc_method_calls_total"));
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
