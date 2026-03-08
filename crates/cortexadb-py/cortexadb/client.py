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
        **kwargs
    ) -> int:
        """Add a memory to this collection."""
        self._check_writable()
        return self._db.add(text=text, vector=vector, metadata=metadata, collection=self.name, **kwargs)

    def search(self, query=None, vector=None, limit=None, **kwargs) -> t.List[Hit]:
        """Search in this collection."""
        limit = limit or kwargs.get("top_k", 5)
        return self._db.search(query=query, vector=vector, limit=limit, collections=[self.name], **kwargs)

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
        **kwargs  # Swallow extra args from .open() / .replay()
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
        dimension = kwargs.pop("dimension", None)
        embedder = kwargs.pop("embedder", None)
        if embedder is not None and dimension is not None:
            raise CortexaDBConfigError("Provide either 'dimension' or 'embedder', not both.")
        if embedder is None and dimension is None:
            raise CortexaDBConfigError("One of 'dimension' or 'embedder' is required.")
        
        dim = embedder.dimension if embedder else dimension
        record_path = kwargs.pop("record", None)
        recorder = ReplayWriter(record_path, dimension=dim, sync=kwargs.get("sync", "strict")) if record_path else None
        
        return cls(path, dimension=dim, embedder=embedder, _recorder=recorder, **kwargs)

    @classmethod
    def replay(cls, log_path: str, db_path: str, **kwargs) -> "CortexaDB":
        from .replay import ReplayReader
        try:
            reader = ReplayReader(log_path)
        except FileNotFoundError as e:
            raise CortexaDBError(str(e))
        strict = kwargs.get("strict", True)
        
        db = cls.open(db_path, dimension=reader.header.dimension, **kwargs)
        report = {"checked": 0, "exported": 0, "skipped": 0, "failed": 0, "op_counts": {}}
        
        id_map = {} # Map log IDs to new DB IDs
        
        for op in reader.operations():
            op_type = op.get("op", "unknown")
            report["checked"] += 1
            report["op_counts"][op_type] = report["op_counts"].get(op_type, 0) + 1
            
            try:
                if op_type == "remember":
                    new_id = db.add(
                        text=op.get("text"),
                        vector=op.get("embedding"),
                        metadata=op.get("metadata"),
                        collection=op.get("namespace", "default")
                    )
                    id_map[op.get("id")] = new_id
                    report["exported"] += 1
                elif op_type == "connect":
                    src = id_map.get(op.get("id1") or op.get("from_id"))
                    dst = id_map.get(op.get("id2") or op.get("to_id"))
                    if src and dst:
                        db.connect(src, dst, op.get("relation"))
                        report["exported"] += 1
                    else:
                        report["skipped"] += 1
                else:
                    report["op_counts"]["unknown"] = report["op_counts"].get("unknown", 0) + 1
                    if strict: raise CortexaDBError(f"unknown replay op: {op_type}")
                    report["skipped"] += 1
            except Exception as e:
                if strict: raise
                report["skipped"] += 1
                report["failed"] += 1
        
        db._last_replay_report = report
        return db

    def collection(self, name: str, **kwargs) -> Collection:
        """Access a scoped collection."""
        return Collection(self, name, **kwargs)

    def namespace(self, *a, **k): return self.collection(*a, **k)

    def add(self, text=None, vector=None, metadata=None, collection=None, **kwargs) -> int:
        """Add a memory."""
        collection = collection or kwargs.get("collection") or kwargs.get("namespace", "default")
        vector = vector or kwargs.get("vector") or kwargs.get("embedding")
        vec = self._resolve_embedding(text, vector)
        content = text or ""
        mid = self._inner.remember_embedding(vec, metadata=metadata, namespace=collection, content=content)
        if self._recorder:
            self._recorder.record_remember(id=mid, text=content, embedding=vec, namespace=collection, metadata=metadata)
        return mid

    def search(
        self,
        query=None, vector=None, limit=None,
        collections=None, filter=None,
        use_graph=False, recency_bias=False,
        **kwargs
    ) -> t.List[Hit]:
        """Core search implementation."""
        limit = limit or kwargs.get("limit") or kwargs.get("top_k", 5)
        vector = vector or kwargs.get("vector") or kwargs.get("embedding") or kwargs.get("query_vector")
        collections = collections or kwargs.get("collections") or kwargs.get("namespaces")
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
                        if collections:
                            # Only add neighbor if it's in requested collections
                            if self.get(target_id).namespace not in collections:
                                continue
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

    def connect(self, mid1: int, mid2: int, relation: str):
        """Connect two memories with a labeled edge."""
        self._inner.connect(mid1, mid2, relation)
        if self._recorder:
            self._recorder.record_connect(mid1, mid2, relation)

    def export_replay(self, path: str):
        """Export all memories to a replay log."""
        from .replay import ReplayWriter
        writer = ReplayWriter(path, dimension=self._dimension)
        report = {"checked": 0, "exported": 0, "skipped_missing_embedding": 0}
        
        # This is a bit slow as we iterate all IDs
        stats = self.stats()
        for i in range(1, stats.entries + 1):
            report["checked"] += 1
            try:
                mem = self.get(i)
                if mem.embedding:
                    writer.record_remember(
                        id=mem.id,
                        text=bytes(mem.content).decode("utf-8") if mem.content else "",
                        embedding=mem.embedding,
                        namespace=mem.namespace,
                        metadata=mem.metadata
                    )
                    report["exported"] += 1
                else:
                    report["skipped_missing_embedding"] += 1
            except:
                pass
        
        writer.close()
        self._last_export_replay_report = report

    @property
    def last_replay_report(self): return self._last_replay_report

    @property
    def last_export_replay_report(self): return self._last_export_replay_report

    def add_batch(self, records: t.List[t.Dict]) -> t.List[int]:
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
        if not self._embedder:
            raise CortexaDBConfigError("ingest_document requires an embedder.")
        chunks = chunk(text, **kwargs)
        if not chunks: return []
        
        embeddings = self._embedder.embed_batch([c["text"] for c in chunks])
        records = [{
            "text": c["text"],
            "vector": vec,
            "metadata": {** (kwargs.get("metadata") or {}), **(c.get("metadata") or {})},
            "collection": kwargs.get("collection") or kwargs.get("namespace", "default")
        } for c, vec in zip(chunks, embeddings)]
        
        return self.add_batch(records)

    def _resolve_embedding(self, text, supplied):
        if supplied is not None: return supplied
        if not self._embedder: raise CortexaDBConfigError("No embedder provided. Embedder required.")
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
