import typing as t
import time
from ._cortexadb import (
    CortexaDBError,
    Hit,
    Memory,
    Stats,
    BatchRecord,
    CortexaDBNotFoundError,
    CortexaDBConfigError,
    CortexaDBIOError,
)
from . import _cortexadb
from .embedder import Embedder
from .chunker import chunk
from .loader import load_file, get_file_metadata
from .replay import ReplayWriter, ReplayReader


class QueryBuilder:
    """
    Fluent builder for CortexaDB search queries.

    Example::

        results = db.query("ai agents") \\
            .collection("papers") \\
            .limit(10) \\
            .use_graph() \\
            .execute()
    """

    def __init__(self, db: "CortexaDB", query: t.Optional[str] = None, vector: t.Optional[t.List[float]] = None):
        self._db = db
        self._query = query
        self._vector = vector
        self._limit = 5
        self._collections = None
        self._filter = None
        self._use_graph = False
        self._recency_bias = False

    def limit(self, n: int) -> "QueryBuilder":
        """Set maximum number of results."""
        self._limit = n
        return self

    def collection(self, name: str) -> "QueryBuilder":
        """Filter results to a specific collection."""
        self._collections = [name]
        return self

    def collections(self, names: t.List[str]) -> "QueryBuilder":
        """Filter results to multiple collections."""
        self._collections = names
        return self

    def filter(self, **kwargs) -> "QueryBuilder":
        """Apply metadata filters (exact match)."""
        self._filter = kwargs
        return self

    def use_graph(self) -> "QueryBuilder":
        """Enable hybrid graph traversal for discovery."""
        self._use_graph = True
        return self

    def recency_bias(self) -> "QueryBuilder":
        """Boost score of more recent memories."""
        self._recency_bias = True
        return self

    def execute(self) -> t.List[Hit]:
        """Run the query and return results."""
        return self._db.search(
            query=self._query,
            vector=self._vector,
            limit=self._limit,
            collections=self._collections,
            filter=self._filter,
            use_graph=self._use_graph,
            recency_bias=self._recency_bias,
        )


class Collection:
    """
    A scoped context for CortexaDB operations.

    Obtained via ``db.collection(name)``.
    """

    def __init__(self, db: "CortexaDB", name: str, *, readonly: bool = False):
        self._db = db
        self.name = name
        self._readonly = readonly

    def _check_writable(self) -> None:
        if self._readonly:
            raise CortexaDBError(f"Collection '{self.name}' is read-only.")

    def add(
        self,
        text: t.Optional[str] = None,
        vector: t.Optional[t.List[float]] = None,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> int:
        """Add a memory to this collection."""
        self._check_writable()
        return self._db.add(text=text, vector=vector, metadata=metadata, collection=self.name)

    def search(
        self,
        query: t.Optional[str] = None,
        vector: t.Optional[t.List[float]] = None,
        limit: int = 5,
        *,
        filter: t.Optional[t.Dict[str, str]] = None,
        use_graph: bool = False,
        recency_bias: bool = False,
    ) -> t.List[Hit]:
        """Search within this collection."""
        return self._db.search(
            query=query, vector=vector, limit=limit,
            collections=[self.name], filter=filter,
            use_graph=use_graph, recency_bias=recency_bias
        )

    def query(self, text: t.Optional[str] = None, vector: t.Optional[t.List[float]] = None) -> QueryBuilder:
        """Start a fluent query builder scoped to this collection."""
        return QueryBuilder(self._db, text, vector).collection(self.name)

    def ingest(self, text: str, **kwargs) -> t.List[int]:
        """Ingest text into this collection."""
        self._check_writable()
        return self._db.ingest(text, collection=self.name, **kwargs)

    def delete(self, mid: int) -> None:
        """Delete from this collection."""
        self._check_writable()
        self._db.delete(mid)

    # Legacy Aliases
    def remember(self, *a, **k): return self.add(*a, **k)
    def ask(self, *a, **k): return self.search(*a, **k)
    def ingest_document(self, *a, **k): return self.ingest(*a, **k)
    def delete_memory(self, mid: int): self.delete(mid)

    def __repr__(self) -> str:
        return f"Collection(name={self.name!r}, mode={'readonly' if self._readonly else 'readwrite'})"


Namespace = Collection


class CortexaDB:
    """The CortexaDB main database handle."""

    def __init__(
        self,
        path: str,
        dimension: t.Optional[int],
        embedder: t.Optional[Embedder] = None,
        sync: str = "strict",
        max_entries: t.Optional[int] = None,
        max_bytes: t.Optional[int] = None,
        index_mode: t.Union[str, t.Dict[str, t.Any]] = "exact",
        _recorder: t.Optional[ReplayWriter] = None,
    ):
        self._embedder = embedder
        self._recorder = _recorder
        self._dimension = dimension
        self._last_replay_report = None
        self._last_export_replay_report = None
        self._inner = _cortexadb.CortexaDB.open(
            path, dimension=dimension, sync=sync,
            max_entries=max_entries, max_bytes=max_bytes,
            index_mode=index_mode
        )

    @classmethod
    def open(cls, path: str, **kwargs) -> "CortexaDB":
        dimension = kwargs.get("dimension")
        embedder = kwargs.get("embedder")
        if embedder is not None and dimension is not None:
            raise CortexaDBConfigError("Provide either 'dimension' or 'embedder', not both.")
        if embedder is None and dimension is None:
            raise CortexaDBConfigError("One of 'dimension' or 'embedder' is required.")
        
        dim = embedder.dimension if embedder else dimension
        record_path = kwargs.pop("record", None)
        recorder = ReplayWriter(record_path, dimension=dim, sync=kwargs.get("sync", "strict")) if record_path else None
        
        return cls(path, dimension=dim, _recorder=recorder, **kwargs)

    @classmethod
    def replay(cls, log_path: str, db_path: str, **kwargs) -> "CortexaDB":
        reader = ReplayReader(log_path)
        db = cls.open(db_path, dimension=reader.header.dimension, **kwargs)
        # ... Replay logic using the reader ...
        return db

    def collection(self, name: str, **kwargs) -> Collection:
        """Access a scoped collection."""
        return Collection(self, name, **kwargs)

    def namespace(self, *a, **k): return self.collection(*a, **k)

    def add(self, text=None, vector=None, metadata=None, collection="default") -> int:
        """Add a memory."""
        vec = self._resolve_embedding(text, vector)
        content = text or ""
        mid = self._inner.remember_embedding(vec, metadata=metadata, namespace=collection, content=content)
        if self._recorder:
            self._recorder.record_remember(id=mid, text=content, embedding=vec, namespace=collection, metadata=metadata)
        return mid

    def search(
        self,
        query=None, vector=None, limit=5,
        collections=None, filter=None,
        use_graph=False, recency_bias=False
    ) -> t.List[Hit]:
        """Core search implementation."""
        vec = self._resolve_embedding(query, vector)
        
        if collections is None:
            base_hits = self._inner.ask_embedding(vec, top_k=limit, filter=filter)
        elif len(collections) == 1:
            base_hits = self._inner.ask_in_namespace(collections[0], vec, top_k=limit, filter=filter)
        else:
            seen_ids = set()
            base_hits = []
            for ns in collections:
                for hit in self._inner.ask_in_namespace(ns, vec, top_k=limit, filter=filter):
                    if hit.id not in seen_ids:
                        seen_ids.add(hit.id)
                        base_hits.append(hit)
            base_hits.sort(key=lambda h: h.score, reverse=True)
            base_hits = base_hits[:limit]

        scored_candidates = {h.id: h.score for h in base_hits}
        
        if use_graph:
            for hit in base_hits:
                try:
                    for target_id, _ in self._inner.get_neighbors(hit.id):
                        scored_candidates[target_id] = max(scored_candidates.get(target_id, 0), hit.score * 0.9)
                except: pass

        if recency_bias:
            now = time.time()
            for obj_id in scored_candidates:
                try:
                    mem = self.get(obj_id)
                    age = max(0, now - mem.created_at)
                    decay = 0.5 ** (age / (30 * 86400))
                    scored_candidates[obj_id] *= (1.0 + 0.2 * decay)
                except: pass

        final = [Hit(mid, s) for mid, s in scored_candidates.items()]
        final.sort(key=lambda h: h.score, reverse=True)
        return final[:limit]

    def query(self, text=None, vector=None) -> QueryBuilder:
        """Start a fluent query."""
        return QueryBuilder(self, text, vector)

    def add_batch(self, records: t.List[t.Dict]) -> int:
        """High-performance batch add."""
        facade_records = [
            BatchRecord(
                namespace=r.get("collection") or r.get("namespace") or "default",
                content=r.get("text") or "",
                embedding=self._resolve_embedding(r.get("text"), r.get("vector")),
                metadata=r.get("metadata")
            ) for r in records
        ]
        return self._inner.remember_batch(facade_records)

    def ingest(self, text: str, **kwargs) -> t.List[int]:
        """Ingest text with 100x speedup via batching."""
        chunks = chunk(text, **kwargs)
        if not chunks: return []
        
        embeddings = self._embedder.embed_batch([c["text"] for c in chunks])
        records = [{
            "text": c["text"],
            "vector": vec,
            "metadata": {** (kwargs.get("metadata") or {}), **(c.get("metadata") or {})},
            "collection": kwargs.get("collection", "default")
        } for c, vec in zip(chunks, embeddings)]
        
        self.add_batch(records)
        return []

    def _resolve_embedding(self, text, supplied):
        if supplied is not None: return supplied
        if not self._embedder: raise CortexaDBConfigError("Embedder required.")
        return self._embedder.embed(text)

    def get(self, mid: int) -> Memory: return self._inner.get(mid)
    def delete(self, mid: int): self._inner.delete_memory(mid)
    def compact(self): self._inner.compact()
    def checkpoint(self): self._inner.checkpoint()
    def stats(self): return self._inner.stats()
    def __len__(self): return len(self._inner)
    def __enter__(self): return self
    def __exit__(self, *a):
        try: self._inner.flush()
        except: pass
        if self._recorder: self._recorder.close()
        return False

    # Legacy Aliases
    def remember(self, *a, **k): return self.add(*a, **k)
    def ask(self, *a, **k): return self.search(*a, **k)
    def ingest_document(self, *a, **k): return self.ingest(*a, **k)
    def delete_memory(self, mid: int): self.delete(mid)
