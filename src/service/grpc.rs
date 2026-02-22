use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{RwLock, mpsc, oneshot};
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
    store: Arc<RwLock<MnemosStore>>,
    writer_tx: mpsc::Sender<WriteRequest>,
}

impl MnemosGrpcService {
    pub fn new(store: MnemosStore) -> Self {
        let store = Arc::new(RwLock::new(store));
        let (writer_tx, mut writer_rx) = mpsc::channel::<WriteRequest>(1024);

        let writer_store = Arc::clone(&store);
        tokio::spawn(async move {
            while let Some(req) = writer_rx.recv().await {
                let result: Result<WriteResult, String> = {
                    let mut store = writer_store.write().await;
                    match req.command {
                        WriteCommand::InsertMemory(entry) => store
                            .insert_memory(entry)
                            .map(|id| WriteResult::CommandId(id.0))
                            .map_err(|e| e.to_string()),
                        WriteCommand::DeleteMemory(id) => store
                            .delete_memory(id)
                            .map(|id| WriteResult::CommandId(id.0))
                            .map_err(|e| e.to_string()),
                        WriteCommand::AddEdge { from, to, relation } => store
                            .add_edge(from, to, relation)
                            .map(|id| WriteResult::CommandId(id.0))
                            .map_err(|e| e.to_string()),
                        WriteCommand::RemoveEdge { from, to } => store
                            .remove_edge(from, to)
                            .map(|id| WriteResult::CommandId(id.0))
                            .map_err(|e| e.to_string()),
                        WriteCommand::EnforceCapacity(policy) => store
                            .enforce_capacity(policy)
                            .map(WriteResult::Capacity)
                            .map_err(|e| e.to_string()),
                        WriteCommand::CompactSegments => store
                            .compact_segments()
                            .map(WriteResult::Compact)
                            .map_err(|e| e.to_string()),
                    }
                };

                let _ = req.reply.send(result);
            }
        });

        Self { store, writer_tx }
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
        out.map_err(Status::internal)
    }
}

struct WriteRequest {
    command: WriteCommand,
    reply: oneshot::Sender<Result<WriteResult, String>>,
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

        let cmd_id = match self.send_write(WriteCommand::InsertMemory(entry)).await? {
            WriteResult::CommandId(id) => id,
            _ => return Err(Status::internal("unexpected writer response for insert")),
        };
        Ok(Response::new(proto::InsertMemoryResponse {
            command_id: cmd_id,
        }))
    }

    async fn delete_memory(
        &self,
        request: Request<proto::DeleteMemoryRequest>,
    ) -> Result<Response<proto::DeleteMemoryResponse>, Status> {
        let req = request.into_inner();
        let cmd_id = match self
            .send_write(WriteCommand::DeleteMemory(MemoryId(req.id)))
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
        let req = request.into_inner();
        let cmd_id = match self
            .send_write(WriteCommand::AddEdge {
                from: MemoryId(req.from),
                to: MemoryId(req.to),
                relation: req.relation,
            })
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
        let req = request.into_inner();
        let cmd_id = match self
            .send_write(WriteCommand::RemoveEdge {
                from: MemoryId(req.from),
                to: MemoryId(req.to),
            })
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

        let store = self.store.read().await;
        let out = store
            .query("grpc_query", options, &embedder)
            .map_err(|e| Status::internal(e.to_string()))?;

        let hits = out
            .hits
            .into_iter()
            .map(|h| {
                let memory = store
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
        let req = request.into_inner();
        let policy = CapacityPolicy::new(
            req.max_entries.and_then(|v| usize::try_from(v).ok()),
            req.max_bytes,
        );

        let report = match self
            .send_write(WriteCommand::EnforceCapacity(policy))
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
        _request: Request<proto::CompactSegmentsRequest>,
    ) -> Result<Response<proto::CompactSegmentsResponse>, Status> {
        let report = match self.send_write(WriteCommand::CompactSegments).await? {
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
        _request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let store = self.store.read().await;
        Ok(Response::new(proto::StatsResponse {
            wal_len: store.wal_len(),
            state_entries: u64::try_from(store.state_machine().len()).unwrap_or(u64::MAX),
            indexed_embeddings: u64::try_from(store.indexed_embeddings()).unwrap_or(u64::MAX),
        }))
    }
}
