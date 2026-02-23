from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import AbstractContextManager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import grpc

from .types import (
    CapacityReport,
    CompactReport,
    Memory,
    QueryHit,
    QueryRequest,
    QueryResult,
    Stats,
)

try:
    from .proto import mnemos_pb2, mnemos_pb2_grpc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gRPC stubs are missing. Run ./scripts/generate_proto.sh in python/mnemos-client first."
    ) from exc


class MnemosClient(AbstractContextManager["MnemosClient"]):
    """Typed Python client for the Mnemos gRPC service."""

    def __init__(
        self,
        address: str,
        *,
        timeout_seconds: Optional[float] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        default_namespace: Optional[str] = None,
        api_key: Optional[str] = None,
        principal_id: Optional[str] = None,
    ) -> None:
        self._address = address
        self._timeout = timeout_seconds
        self._channel = grpc.insecure_channel(address)
        self._stub = mnemos_pb2_grpc.MnemosServiceStub(self._channel)
        self._embedder = embedder
        self._default_namespace = default_namespace
        self._api_key = api_key
        self._principal_id = principal_id

    def close(self) -> None:
        self._channel.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def set_embedder(self, embedder: Callable[[str], List[float]]) -> None:
        """Set a text->embedding function used by high-level text APIs."""
        self._embedder = embedder

    def set_default_namespace(self, namespace: str) -> None:
        self._default_namespace = namespace

    def set_identity(
        self, *, api_key: Optional[str] = None, principal_id: Optional[str] = None
    ) -> None:
        if api_key is not None:
            self._api_key = api_key
        if principal_id is not None:
            self._principal_id = principal_id

    # Simpler UX aliases for agentic workflows.
    def remember(
        self,
        text: str,
        *,
        namespace: Optional[str] = None,
        importance: float = 0.0,
        metadata: Optional[Dict[str, str]] = None,
        memory_id: Optional[int] = None,
    ) -> int:
        """
        Store text memory and return the memory ID (not the command ID).
        """
        ns = self._resolve_namespace(namespace)
        if memory_id is None:
            memory_id = _new_memory_id()
        self.insert_text(
            namespace=ns,
            text=text,
            importance=importance,
            metadata=metadata,
            memory_id=memory_id,
        )
        return memory_id

    def recall(
        self,
        text: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 10,
        graph_hops: Optional[int] = None,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> List[QueryHit]:
        ns = self._resolve_namespace(namespace)
        return self.query_text(
            text,
            top_k=top_k,
            namespace=ns,
            graph_hops=graph_hops,
            time_start=time_start,
            time_end=time_end,
        ).hits

    def forget(self, memory_id: int) -> int:
        return self.delete_memory(memory_id)

    def link(self, source_id: int, target_id: int, relation: str = "related_to") -> int:
        return self.add_edge(source_id, target_id, relation)

    def unlink(self, source_id: int, target_id: int) -> int:
        return self.remove_edge(source_id, target_id)

    def insert_text(
        self,
        *,
        namespace: str,
        text: str,
        memory_id: Optional[int] = None,
        created_at: Optional[int] = None,
        importance: float = 0.0,
        metadata: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ) -> int:
        """High-level insert API: takes plain text and embeds automatically."""
        embedding = self._embed_text(text)
        if memory_id is None:
            memory_id = _new_memory_id()
        if created_at is None:
            created_at = int(time.time())
        return self.insert_memory(
            Memory(
                id=memory_id,
                namespace=namespace,
                content=text.encode("utf-8"),
                embedding=embedding,
                created_at=created_at,
                importance=importance,
                metadata=metadata or {},
            ),
            request_id=request_id,
        )

    def query_text(
        self,
        text: str,
        *,
        top_k: int = 10,
        namespace: Optional[str] = None,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
        graph_hops: Optional[int] = None,
        candidate_multiplier: int = 0,
        similarity_pct: int = 0,
        importance_pct: int = 0,
        recency_pct: int = 0,
    ) -> QueryResult:
        """High-level query API: takes plain text and embeds automatically."""
        query_embedding = self._embed_text(text)
        return self.query(
            QueryRequest(
                query_embedding=query_embedding,
                top_k=top_k,
                namespace=namespace,
                time_start=time_start,
                time_end=time_end,
                graph_hops=graph_hops,
                candidate_multiplier=candidate_multiplier,
                similarity_pct=similarity_pct,
                importance_pct=importance_pct,
                recency_pct=recency_pct,
            )
        )

    def insert_memory(self, memory: Memory, *, request_id: Optional[str] = None) -> int:
        req = mnemos_pb2.InsertMemoryRequest(
            memory=mnemos_pb2.Memory(
                id=memory.id,
                namespace=memory.namespace,
                content=memory.content,
                embedding=memory.embedding,
                created_at=memory.created_at,
                importance=memory.importance,
                metadata=[
                    mnemos_pb2.MetadataPair(key=k, value=v)
                    for k, v in memory.metadata.items()
                ],
            )
        )
        resp = self._stub.InsertMemory(
            req,
            timeout=self._timeout,
            metadata=self._rpc_metadata(
                request_id=request_id
                or f"insert-{memory.id}-{memory.created_at}-{hashlib.sha1(memory.content).hexdigest()[:12]}"
            ),
        )
        return int(resp.command_id)

    def delete_memory(self, memory_id: int) -> int:
        req = mnemos_pb2.DeleteMemoryRequest(id=memory_id)
        resp = self._stub.DeleteMemory(
            req,
            timeout=self._timeout,
            metadata=self._rpc_metadata(request_id=f"delete-{memory_id}-{time.time_ns()}"),
        )
        return int(resp.command_id)

    def add_edge(self, from_id: int, to_id: int, relation: str) -> int:
        req = mnemos_pb2.AddEdgeRequest(**{"from": from_id, "to": to_id, "relation": relation})
        resp = self._stub.AddEdge(
            req,
            timeout=self._timeout,
            metadata=self._rpc_metadata(
                request_id=f"add-edge-{from_id}-{to_id}-{relation}-{time.time_ns()}"
            ),
        )
        return int(resp.command_id)

    def remove_edge(self, from_id: int, to_id: int) -> int:
        req = mnemos_pb2.RemoveEdgeRequest(**{"from": from_id, "to": to_id})
        resp = self._stub.RemoveEdge(
            req,
            timeout=self._timeout,
            metadata=self._rpc_metadata(request_id=f"remove-edge-{from_id}-{to_id}-{time.time_ns()}"),
        )
        return int(resp.command_id)

    def query(self, request: QueryRequest) -> QueryResult:
        req = mnemos_pb2.QueryRequest(
            query_embedding=request.query_embedding,
            top_k=request.top_k,
            namespace=request.namespace,
            time_start=request.time_start,
            time_end=request.time_end,
            graph_hops=request.graph_hops,
            candidate_multiplier=request.candidate_multiplier,
            similarity_pct=request.similarity_pct,
            importance_pct=request.importance_pct,
            recency_pct=request.recency_pct,
        )
        resp = self._stub.Query(
            req,
            timeout=self._timeout,
            metadata=self._rpc_metadata(),
        )
        hits = [
            QueryHit(
                id=int(h.id),
                final_score=float(h.final_score),
                similarity_score=float(h.similarity_score),
                importance_score=float(h.importance_score),
                recency_score=float(h.recency_score),
                memory=(
                    Memory(
                        id=int(h.memory.id),
                        namespace=h.memory.namespace,
                        content=bytes(h.memory.content),
                        embedding=[float(v) for v in h.memory.embedding],
                        created_at=int(h.memory.created_at),
                        importance=float(h.memory.importance),
                        metadata={kv.key: kv.value for kv in h.memory.metadata},
                    )
                    if h.HasField("memory")
                    else None
                ),
            )
            for h in resp.hits
        ]
        return QueryResult(
            hits=hits,
            execution_path=resp.execution_path,
            used_parallel=bool(resp.used_parallel),
            vector_candidates=int(resp.vector_candidates),
            filtered_candidates=int(resp.filtered_candidates),
            final_results=int(resp.final_results),
            elapsed_ms=int(resp.elapsed_ms),
        )

    def stats(self) -> Stats:
        resp = self._stub.Stats(
            mnemos_pb2.StatsRequest(),
            timeout=self._timeout,
            metadata=self._rpc_metadata(),
        )
        return Stats(
            wal_len=int(resp.wal_len),
            state_entries=int(resp.state_entries),
            indexed_embeddings=int(resp.indexed_embeddings),
        )

    def enforce_capacity(
        self,
        *,
        max_entries: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> CapacityReport:
        req = mnemos_pb2.EnforceCapacityRequest(
            max_entries=max_entries,
            max_bytes=max_bytes,
        )
        resp = self._stub.EnforceCapacity(
            req,
            timeout=self._timeout,
            metadata=self._rpc_metadata(request_id=f"capacity-{time.time_ns()}"),
        )
        return CapacityReport(
            evicted_ids=[int(v) for v in resp.evicted_ids],
            entries_before=int(resp.entries_before),
            entries_after=int(resp.entries_after),
            bytes_before=int(resp.bytes_before),
            bytes_after=int(resp.bytes_after),
        )

    def compact_segments(self) -> CompactReport:
        resp = self._stub.CompactSegments(
            mnemos_pb2.CompactSegmentsRequest(),
            timeout=self._timeout,
            metadata=self._rpc_metadata(request_id=f"compact-{time.time_ns()}"),
        )
        return CompactReport(
            compacted_segments=[int(v) for v in resp.compacted_segments],
            live_entries_rewritten=int(resp.live_entries_rewritten),
        )

    def _embed_text(self, text: str) -> List[float]:
        if self._embedder is not None:
            vec = self._embedder(text)
            if not vec:
                raise ValueError("embedder returned empty vector")
            return vec
        return default_hash_embedder(text)

    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        if namespace:
            return namespace
        if self._default_namespace:
            return self._default_namespace
        raise ValueError(
            "namespace is required. Pass namespace=... or set default_namespace in MnemosClient."
        )

    def _rpc_metadata(self, *, request_id: Optional[str] = None) -> Sequence[Tuple[str, str]]:
        metadata: List[Tuple[str, str]] = []
        if request_id:
            metadata.append(("x-request-id", request_id))
        if self._api_key:
            metadata.append(("x-api-key", self._api_key))
            metadata.append(("authorization", f"Bearer {self._api_key}"))
        if self._principal_id:
            metadata.append(("x-principal-id", self._principal_id))
        return metadata


def default_hash_embedder(text: str, dim: int = 3) -> List[float]:
    """
    Deterministic fallback embedder for local development.
    For production, set `client.set_embedder(...)` with your model embedding function.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals: List[float] = []
    for i in range(dim):
        chunk = digest[i * 4 : (i + 1) * 4]
        n = int.from_bytes(chunk, byteorder="little", signed=False)
        vals.append((n % 10_000) / 10_000.0)
    return vals


def _new_memory_id() -> int:
    # 64-bit ID from UUID4 to avoid collisions in typical client usage.
    return uuid.uuid4().int & ((1 << 64) - 1)
