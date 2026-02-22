from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class Memory:
    id: int
    namespace: str
    content: bytes
    embedding: List[float] = field(default_factory=list)
    created_at: int = 0
    importance: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


@dataclass(slots=True)
class QueryRequest:
    query_embedding: List[float]
    top_k: int = 10
    namespace: Optional[str] = None
    time_start: Optional[int] = None
    time_end: Optional[int] = None
    graph_hops: Optional[int] = None
    candidate_multiplier: int = 0
    similarity_pct: int = 0
    importance_pct: int = 0
    recency_pct: int = 0


@dataclass(slots=True)
class QueryHit:
    id: int
    final_score: float
    similarity_score: float
    importance_score: float
    recency_score: float
    memory: Optional[Memory] = None

    @property
    def text(self) -> str:
        if self.memory is None:
            return ""
        return self.memory.text


@dataclass(slots=True)
class QueryResult:
    hits: List[QueryHit]
    execution_path: str
    used_parallel: bool
    vector_candidates: int
    filtered_candidates: int
    final_results: int
    elapsed_ms: int


@dataclass(slots=True)
class Stats:
    wal_len: int
    state_entries: int
    indexed_embeddings: int


@dataclass(slots=True)
class CapacityReport:
    evicted_ids: List[int]
    entries_before: int
    entries_after: int
    bytes_before: int
    bytes_after: int


@dataclass(slots=True)
class CompactReport:
    compacted_segments: List[int]
    live_entries_rewritten: int
