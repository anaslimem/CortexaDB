import typing as t
import copy

from ._cortexadb import (
    CortexaDBError,
    Hit,
    Memory,
    Stats,
    CortexaDBNotFoundError,
    CortexaDBConfigError,
    CortexaDBIOError,
)
from . import _cortexadb
from .embedder import Embedder
from .chunker import chunk_text, chunk
from .loader import load_file, get_file_metadata
from .replay import ReplayWriter, ReplayReader
import time


class Namespace:
    """
    A scoped context for CortexaDB operations.

    Obtained via ``db.namespace(name)``.  All store and query operations
    automatically apply this namespace.

    Args:
        db:       Parent :class:`CortexaDB` instance.
        name:     Namespace identifier string.
        readonly: When *True*, ``remember()`` and ``ingest_document()`` raise
                  :class:`CortexaDBError` — useful for shared read-only views.
    """

    def __init__(self, db: "CortexaDB", name: str, *, readonly: bool = False):
        self._db = db
        self.name = name
        self._readonly = readonly

    def _check_writable(self) -> None:
        if self._readonly:
            raise CortexaDBError(
                f"Namespace '{self.name}' is read-only. "
                "Open it without readonly=True to write."
            )

    def remember(
        self,
        text: str,
        embedding: t.Optional[t.List[float]] = None,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> int:
        """
        Store a new memory in this namespace.

        If the database was opened with an embedder, *embedding* is optional.
        """
        self._check_writable()
        return self._db._remember_inner(
            text=text,
            embedding=embedding,
            metadata=metadata,
            namespace=self.name,
        )

    def ask(
        self,
        query: str,
        embedding: t.Optional[t.List[float]] = None,
        top_k: int = 5,
        *,
        use_graph: bool = False,
        recency_bias: bool = False,
    ) -> t.List[Hit]:
        """
        Query for memories scoped to this namespace.

        Uses the fast Rust-side namespace filter (``ask_in_namespace``).
        """
        vec = self._db._resolve_embedding(query, embedding)
        # Note: We use the global ask() so hybrid query logic isn't duplicated.
        return self._db.ask(
            query=query,
            embedding=vec,
            top_k=top_k,
            namespaces=[self.name],
            use_graph=use_graph,
            recency_bias=recency_bias,
        )

    def delete_memory(self, mid: int) -> None:
        """Delete a memory by ID."""
        self._check_writable()
        self._db.delete_memory(mid)

    def ingest_document(
        self,
        text: str,
        *,
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[int]:
        """
        Split *text* into chunks and store each one in this namespace.
        Requires an embedder to be set on the database.
        """
        self._check_writable()
        return self._db.ingest_document(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            namespace=self.name,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        mode = "readonly" if self._readonly else "readwrite"
        return f"Namespace(name={self.name!r}, mode={mode})"


class CortexaDB:
    """
    Pythonic interface to the CortexaDB embedded vector + graph database.

    Open with a fixed dimension (manual embedding)::

        db = CortexaDB.open("agent.mem", dimension=128)
        mid = db.remember("hello", embedding=[0.1] * 128)

    Open with an embedder for automatic embedding::

        from cortexadb.providers.openai import OpenAIEmbedder
        db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder(api_key="sk-..."))
        mid = db.remember("We chose Stripe for payments")

    Record a session for deterministic replay::

        db = CortexaDB.open("agent.mem", dimension=128, record="session.log")
        db.remember("fact A", embedding=[...])   # stored + logged
        # later ...
        db2 = CortexaDB.replay("session.log", "replay.mem")

    Multi-agent namespace model::

        agent_a = db.namespace("agent_a")
        shared  = db.namespace("shared", readonly=True)
    """

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
        self._last_replay_report: t.Optional[t.Dict[str, t.Any]] = None
        self._last_export_replay_report: t.Optional[t.Dict[str, t.Any]] = None
        try:
            try:
                self._inner = _cortexadb.CortexaDB.open(
                    path,
                    dimension=dimension,
                    sync=sync,
                    max_entries=max_entries,
                    max_bytes=max_bytes,
                    index_mode=index_mode,
                )
            except TypeError as type_err:
                # Backward compatibility for environments with an older compiled
                # extension that does not yet expose max_bytes.
                if "max_bytes" not in str(type_err):
                    raise
                if max_bytes is not None:
                    raise CortexaDBConfigError(
                        "This installed cortexadb extension does not support "
                        "'max_bytes' yet. Rebuild or reinstall cortexadb."
                    )
                self._inner = _cortexadb.CortexaDB.open(
                    path,
                    dimension=dimension,
                    sync=sync,
                    max_entries=max_entries,
                    index_mode=index_mode,
                )
        except Exception as e:
            if isinstance(e, CortexaDBError):
                raise
            raise CortexaDBError(str(e))

    @classmethod
    def open(
        cls,
        path: str,
        *,
        dimension: t.Optional[int] = None,
        embedder: t.Optional[Embedder] = None,
        sync: str = "strict",
        max_entries: t.Optional[int] = None,
        max_bytes: t.Optional[int] = None,
        index_mode: t.Union[str, t.Dict[str, t.Any]] = "exact",
        record: t.Optional[str] = None,
    ) -> "CortexaDB":
        """
        Open or create a CortexaDB database.

        Exactly one of *dimension* or *embedder* must be provided.

        Args:
            path:      Directory path for the database files.
            dimension: Vector embedding dimension. Use when supplying your own
                       pre-computed embeddings.
            embedder:  An :class:`~cortexadb.Embedder` instance. The dimension is
                       inferred from ``embedder.dimension`` automatically.
            sync:      Write durability policy: ``"strict"`` (default),
                       ``"async"``, or ``"batch"``.
            max_entries: Optional entry-count limit for automatic eviction.
            max_bytes: Optional byte-size limit for automatic eviction.
            index_mode: Search index mode: ``"exact"`` (default) or ``"hnsw"``.
                         Can also be a dict with HNSW parameters.
            record:    Optional path to a replay log file. When set, every write
                       operation (remember, connect, compact) is appended to this
                       NDJSON file so the session can be replayed later.

        Raises:
            CortexaDBError: If neither or both of *dimension* and *embedder* are
                         provided, or if the database cannot be opened.
        """
        if embedder is not None and dimension is not None:
            raise CortexaDBConfigError(
                "Provide either 'dimension' or 'embedder', not both."
            )
        if embedder is None and dimension is None:
            raise CortexaDBConfigError("One of 'dimension' or 'embedder' is required.")

        dim = embedder.dimension if embedder is not None else dimension

        recorder: t.Optional[ReplayWriter] = None
        if record is not None:
            recorder = ReplayWriter(record, dimension=dim, sync=sync)

        return cls(
            path,
            dimension=dim,
            embedder=embedder,
            sync=sync,
            max_entries=max_entries,
            max_bytes=max_bytes,
            index_mode=index_mode,
            _recorder=recorder,
        )

    @classmethod
    def replay(
        cls,
        log_path: str,
        db_path: str,
        *,
        sync: str = "strict",
        strict: bool = False,
    ) -> "CortexaDB":
        """
        Replay a log file into a fresh database, returning the populated instance.

        The log must have been produced by ``CortexaDB.open(..., record=log_path)``
        or ``db.export_replay(log_path)``.

        Args:
            log_path:  Path to the NDJSON replay log file.
            db_path:   Directory path for the new database.
            sync:      Sync policy for the replayed database.
            strict:    If True, fail fast on malformed/failed operations.
                       If False (default), skip invalid operations and continue.

        Returns:
            A :class:`CortexaDB` instance with all recorded operations applied.

        Raises:
            CortexaDBError:      If the log is invalid or replay fails.
            FileNotFoundError: If *log_path* does not exist.

        Example::

            db = CortexaDB.replay("session.log", "/tmp/replayed.mem")
            hits = db.ask("payment provider?", embedding=[...])
        """
        try:
            reader = ReplayReader(log_path)
        except FileNotFoundError as e:
            raise CortexaDBIOError(str(e))
        except ValueError as e:
            raise CortexaDBConfigError(str(e))

        hdr = reader.header
        db = cls(db_path, dimension=hdr.dimension, sync=sync)

        # old_id → new_id mapping (connect ops use original IDs)
        id_map: t.Dict[int, int] = {}
        report: t.Dict[str, t.Any] = {
            "strict": strict,
            "total_ops": 0,
            "applied": 0,
            "skipped": 0,
            "failed": 0,
            "op_counts": {
                "remember": 0,
                "connect": 0,
                "delete": 0,
                "checkpoint": 0,
                "compact": 0,
                "unknown": 0,
            },
            "failures": [],
        }

        def add_failure(
            *,
            index: int,
            op: str,
            reason: str,
            record: t.Dict[str, t.Any],
            counts_as_failed: bool = False,
        ) -> None:
            if counts_as_failed:
                report["failed"] += 1
            else:
                report["skipped"] += 1
            if len(report["failures"]) < 50:
                report["failures"].append(
                    {
                        "index": index,
                        "op": op,
                        "reason": reason,
                        "record": record,
                    }
                )

        def validate_record(op: str, record: t.Dict[str, t.Any]) -> t.Optional[str]:
            if op == "remember":
                if "embedding" not in record:
                    return "remember record missing 'embedding'"
                if not isinstance(record["embedding"], list):
                    return "remember record 'embedding' must be a list"
            elif op == "connect":
                for field in ("from_id", "to_id", "relation"):
                    if field not in record:
                        return f"connect record missing '{field}'"
            elif op == "delete":
                if "id" not in record:
                    return "delete record missing 'id'"
            elif op in ("checkpoint", "compact"):
                return None
            else:
                return f"unknown replay op '{op}'"
            return None

        for index, record in enumerate(reader.operations(), start=1):
            report["total_ops"] += 1
            op = record.get("op")
            if not isinstance(op, str):
                if strict:
                    raise CortexaDBConfigError(
                        f"Replay op #{index} has invalid/missing 'op': {record!r}"
                    )
                report["op_counts"]["unknown"] += 1
                add_failure(index=index, op="unknown", reason="missing/invalid 'op'", record=record)
                continue

            if op in report["op_counts"]:
                report["op_counts"][op] += 1
            else:
                report["op_counts"]["unknown"] += 1

            validation_error = validate_record(op, record)
            if validation_error is not None:
                if strict:
                    raise CortexaDBConfigError(
                        f"Replay op #{index} ({op}) invalid: {validation_error}"
                    )
                add_failure(index=index, op=op, reason=validation_error, record=record)
                continue

            if op == "remember":
                try:
                    embedding: t.List[float] = record["embedding"]
                    new_id = db._inner.remember_embedding(
                        embedding=embedding,
                        metadata=record.get("metadata"),
                        namespace=record.get("namespace", "default"),
                        content=record.get("text", ""),
                    )
                    old_id = record.get("id")
                    if old_id is not None:
                        id_map[old_id] = new_id
                    report["applied"] += 1
                except Exception as e:
                    if strict:
                        raise CortexaDBError(
                            f"Replay op #{index} (remember) failed: {e}"
                        )
                    add_failure(
                        index=index,
                        op=op,
                        reason=f"remember failed: {e}",
                        record=record,
                        counts_as_failed=True,
                    )

            elif op == "connect":
                try:
                    old_from = record["from_id"]
                    old_to = record["to_id"]
                    new_from = id_map.get(old_from, old_from)
                    new_to = id_map.get(old_to, old_to)
                    db._inner.connect(new_from, new_to, record["relation"])
                    report["applied"] += 1
                except Exception:
                    if strict:
                        raise CortexaDBError(
                            f"Replay op #{index} (connect) failed for ids "
                            f"{record.get('from_id')}->{record.get('to_id')}"
                        )
                    add_failure(
                        index=index,
                        op=op,
                        reason="connect failed (possibly unresolved IDs)",
                        record=record,
                        counts_as_failed=True,
                    )

            elif op == "delete":
                try:
                    old_id = record.get("id")
                    new_id = id_map.get(old_id, old_id)
                    db._inner.delete_memory(new_id)
                    report["applied"] += 1
                except Exception:
                    if strict:
                        raise CortexaDBError(
                            f"Replay op #{index} (delete) failed for id {record.get('id')}"
                        )
                    add_failure(
                        index=index,
                        op=op,
                        reason="delete failed (possibly missing ID)",
                        record=record,
                        counts_as_failed=True,
                    )

            elif op == "checkpoint":
                try:
                    db._inner.checkpoint()
                    report["applied"] += 1
                except Exception:
                    if strict:
                        raise CortexaDBError(f"Replay op #{index} (checkpoint) failed")
                    add_failure(
                        index=index,
                        op=op,
                        reason="checkpoint failed",
                        record=record,
                        counts_as_failed=True,
                    )
            elif op == "compact":
                try:
                    db._inner.compact()
                    report["applied"] += 1
                except Exception:
                    if strict:
                        raise CortexaDBError(f"Replay op #{index} (compact) failed")
                    add_failure(
                        index=index,
                        op=op,
                        reason="compact failed",
                        record=record,
                        counts_as_failed=True,
                    )

        db._last_replay_report = report
        return db

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_embedding(
        self,
        text: str,
        supplied: t.Optional[t.List[float]],
    ) -> t.List[float]:
        """Return *supplied* or auto-embed *text* via the configured embedder."""
        if supplied is not None:
            return supplied
        if self._embedder is None:
            raise CortexaDBConfigError(
                "No embedder configured. Either pass 'embedding=' explicitly "
                "or open the database with 'embedder=...'."
            )
        return self._embedder.embed(text)

    def _remember_inner(
        self,
        text: str,
        embedding: t.Optional[t.List[float]],
        metadata: t.Optional[t.Dict[str, str]],
        namespace: str,
    ) -> int:
        vec = self._resolve_embedding(text, embedding)
        mid = self._inner.remember_embedding(
            embedding=vec,
            metadata=metadata,
            namespace=namespace,
            content=text,
        )
        if self._recorder is not None:
            self._recorder.record_remember(
                id=mid,
                text=text,
                embedding=vec,
                namespace=namespace,
                metadata=metadata,
            )
        return mid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def namespace(self, name: str, *, readonly: bool = False) -> "Namespace":
        """
        Return a scoped :class:`Namespace` for partitioned memory access.

        Args:
            name:     Namespace identifier (e.g. ``"agent_a"``, ``"shared"``).
            readonly: If *True*, writes to this namespace raise
                      :class:`CortexaDBError`.
        """
        return Namespace(self, name, readonly=readonly)

    def remember(
        self,
        text: str,
        embedding: t.Optional[t.List[float]] = None,
        metadata: t.Optional[t.Dict[str, str]] = None,
        namespace: str = "default",
    ) -> int:
        """
        Store a new memory.

        If an embedder is configured, *embedding* is optional.
        If recording is enabled, the operation is also appended to the log.
        """
        return self._remember_inner(
            text=text,
            embedding=embedding,
            metadata=metadata,
            namespace=namespace,
        )

    def ask(
        self,
        query: str,
        embedding: t.Optional[t.List[float]] = None,
        top_k: int = 5,
        namespaces: t.Optional[t.List[str]] = None,
        *,
        filter: t.Optional[t.Dict[str, str]] = None,
        use_graph: bool = False,
        recency_bias: bool = False,
    ) -> t.List[Hit]:
        """
        Query by vector similarity, with optional hybrid capabilities.

        Args:
            query:        Query string (auto-embedded if embedder is configured).
            embedding:    Pre-computed query vector (overrides auto-embed).
            top_k:        Maximum hits to return (default 5).
            namespaces:   Restrict search to these namespaces. ``None`` → global.
            filter:       Optional metadata filter dict (e.g. {"type": "note"}).
            use_graph:    If *True*, augments vector results with graph neighbors.
            recency_bias: If *True*, boosts scores of recently created memories.
        """
        vec = self._resolve_embedding(query, embedding)

        # 1. Base vector search
        if namespaces is None:
            base_hits = self._inner.ask_embedding(
                embedding=vec, top_k=top_k, filter=filter
            )
        elif len(namespaces) == 1:
            base_hits = self._inner.ask_in_namespace(
                namespace=namespaces[0], embedding=vec, top_k=top_k, filter=filter
            )
        else:
            seen_ids: t.Set[int] = set()
            base_hits = []
            for ns in namespaces:
                for hit in self._inner.ask_in_namespace(
                    namespace=ns, embedding=vec, top_k=top_k, filter=filter
                ):
                    if hit.id not in seen_ids:
                        seen_ids.add(hit.id)
                        base_hits.append(hit)
            base_hits.sort(key=lambda h: h.score, reverse=True)
            base_hits = base_hits[:top_k]

        scored_candidates: t.Dict[int, float] = {h.id: h.score for h in base_hits}
        allowed_namespaces: t.Optional[t.Set[str]] = (
            set(namespaces) if namespaces is not None else None
        )
        namespace_cache: t.Dict[int, str] = {}

        def is_namespace_allowed(mid: int) -> bool:
            if allowed_namespaces is None:
                return True
            if mid in namespace_cache:
                return namespace_cache[mid] in allowed_namespaces
            try:
                ns = self.get(mid).namespace
            except Exception:
                return False
            namespace_cache[mid] = ns
            return ns in allowed_namespaces

        # 2. Graph Traversal (Phase 3.3)
        if use_graph:
            # For each hit, pull its neighbors and score them slightly lower than the source
            neighbor_candidates_scores = {}
            for hit in base_hits:
                try:
                    neighbors = self._inner.get_neighbors(hit.id)
                    for target_id, relation in neighbors:
                        if not is_namespace_allowed(target_id):
                            continue
                        # Edge weight factor (e.g. 0.9 penalty for 1 hop)
                        neighbor_score = hit.score * 0.9

                        # Take the best score among multiple paths to the same neighbor
                        current_best = neighbor_candidates_scores.get(target_id, 0.0)
                        if neighbor_score > current_best:
                            neighbor_candidates_scores[target_id] = neighbor_score
                except Exception:
                    pass  # missing ID handle gracefully

            # Mix neighbors in; if already found by vector search, take the max score
            for target_id, score in neighbor_candidates_scores.items():
                scored_candidates[target_id] = max(
                    scored_candidates.get(target_id, 0.0), score
                )

        # 3. Recency Bias (Phase 3.3)
        if recency_bias:
            now = time.time()
            for obj_id in scored_candidates:
                try:
                    mem = self.get(obj_id)
                    age_seconds = max(0, now - mem.created_at)
                    # 30-day half-life decay
                    decay_factor = 0.5 ** (age_seconds / (30 * 86400))
                    # Boost final score by up to 20%
                    recency_boost = 1.0 + (0.2 * decay_factor)
                    scored_candidates[obj_id] *= recency_boost
                except Exception:
                    pass

        # 4. Final Re-ranking and Truncation
        # Convert dictionary back to Hit objects (for neighbors we don't have the original Hit, so we recreate it)
        final_hits = [Hit(id=mid, score=s) for mid, s in scored_candidates.items()]
        final_hits.sort(key=lambda h: h.score, reverse=True)
        return final_hits[:top_k]

    def connect(self, from_id: int, to_id: int, relation: str) -> None:
        """
        Create a directional edge between two memories.

        If recording is enabled, the operation is appended to the log.
        """
        self._inner.connect(from_id, to_id, relation)
        if self._recorder is not None:
            self._recorder.record_connect(
                from_id=from_id, to_id=to_id, relation=relation
            )

    def ingest(
        self,
        text: str,
        *,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 50,
        namespace: str = "default",
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[int]:
        """
        Ingest text with smart chunking and store in database.

        This is the simplified API for ingesting text content.

        Args:
            text: Text content to ingest.
            strategy: Chunking strategy - "fixed", "recursive", "semantic", "markdown", "json".
                      Default: "recursive"
            chunk_size: Target size of each chunk (for fixed/recursive). Default: 512
            overlap: Number of words to overlap between chunks. Default: 50
            namespace: Namespace to store in. Default: "default"
            metadata: Optional metadata dict.

        Returns:
            List of memory IDs.

        Requires:
            An embedder must be configured (via embedder=... when opening).
        """
        if self._embedder is None:
            raise CortexaDBConfigError(
                "ingest() requires an embedder. Open the database with 'embedder=...'"
            )

        chunks = chunk(text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []

        chunk_texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed_batch(chunk_texts)

        ids: t.List[int] = []
        for chunk_result, vec in zip(chunks, embeddings):
            meta: t.Dict[str, str] = {}
            if metadata:
                meta = {k: str(v) for k, v in metadata.items()}
            if chunk_result.get("metadata"):
                for k, v in chunk_result["metadata"].items():
                    meta[k] = str(v)

            mid = self._remember_inner(
                text=chunk_result["text"],
                embedding=vec,
                metadata=meta if meta else None,
                namespace=namespace,
            )
            ids.append(mid)
        return ids

    def load(
        self,
        path: str,
        *,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 50,
        namespace: str = "default",
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[int]:
        """
        Load a file and ingest its content.

        Automatically detects file format (.txt, .md, .json, .docx, .pdf).

        Args:
            path: Path to the file.
            strategy: Chunking strategy. Default: "recursive"
            chunk_size: Target chunk size. Default: 512
            overlap: Chunk overlap. Default: 50
            namespace: Namespace to store in. Default: "default"
            metadata: Optional metadata to merge with file metadata.

        Returns:
            List of memory IDs.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format not supported.
        """
        content = load_file(path)
        file_metadata = get_file_metadata(path)

        meta = dict(file_metadata)
        if metadata:
            meta.update(metadata)

        return self.ingest(
            content,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            namespace=namespace,
            metadata=meta,
        )

    def ingest_document(
        self,
        text: str,
        *,
        chunk_size: int = 512,
        overlap: int = 50,
        namespace: str = "default",
        metadata: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[int]:
        """
        Split *text* into chunks and store each one as a separate memory.
        Requires an embedder to be configured.
        """
        if self._embedder is None:
            raise CortexaDBConfigError(
                "ingest_document() requires an embedder. "
                "Open the database with 'embedder=...'."
            )

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []

        embeddings = self._embedder.embed_batch(chunks)
        ids: t.List[int] = []
        for chunk_str, vec in zip(chunks, embeddings):
            # Uses _remember_inner so each chunk is also logged when recording.
            mid = self._remember_inner(
                text=chunk_str,
                embedding=vec,
                metadata=metadata,
                namespace=namespace,
            )
            ids.append(mid)
        return ids

    def export_replay(self, log_path: str) -> None:
        """
        Export the current database state to a replay log file.

        Unlike the ``record=`` mode (which captures operations as they happen),
        this method produces a *snapshot* of all existing memories. The export
        does not preserve the original insertion order beyond what is stored.

        Args:
            log_path: Path to write the NDJSON replay log.

        Example::

            db = CortexaDB.open("agent.mem", dimension=128)
            # ... lots of work ...
            db.export_replay("snapshot.log")

            # Later on any machine:
            db2 = CortexaDB.replay("snapshot.log", "restored.mem")
        """
        stats = self._inner.stats()
        dim = stats.vector_dimension

        import os

        # Truncate any existing file so we start fresh.
        if os.path.exists(log_path):
            os.remove(log_path)

        report: t.Dict[str, t.Any] = {
            "checked": 0,
            "exported": 0,
            "skipped_missing_id": 0,
            "skipped_missing_embedding": 0,
            "errors": 0,
        }

        with ReplayWriter(log_path, dimension=dim, sync="strict") as writer:
            # Iterate memories by scanning IDs 1..entries range.
            # We use a generous upper bound and skip gaps.
            checked = 0
            found = 0
            candidate = 1
            target = stats.entries

            while found < target and checked < target * 4:
                try:
                    mem = self._inner.get(candidate)
                    embedding = getattr(mem, "embedding", None)
                    if not embedding:
                        report["skipped_missing_embedding"] += 1
                        candidate += 1
                        checked += 1
                        continue
                    content = getattr(mem, "content", b"")
                    if isinstance(content, bytes):
                        text = content.decode("utf-8", errors="replace")
                    else:
                        text = str(content)
                    metadata = dict(mem.metadata) if hasattr(mem, "metadata") else None
                    writer.record_remember(
                        id=mem.id,
                        text=text,
                        embedding=embedding,
                        namespace=mem.namespace,
                        metadata=metadata,
                    )
                    found += 1
                    report["exported"] += 1
                except CortexaDBNotFoundError:
                    report["skipped_missing_id"] += 1
                except Exception:
                    report["errors"] += 1
                candidate += 1
                checked += 1
            report["checked"] = checked
        self._last_export_replay_report = report

    def get(self, mid: int) -> Memory:
        """Retrieve a full memory by ID."""
        return self._inner.get(mid)

    def delete_memory(self, mid: int) -> None:
        """
        Delete a memory by ID.

        If recording is enabled, the operation is appended to the log.
        """
        self._inner.delete_memory(mid)
        if self._recorder is not None:
            self._recorder.record_delete(mid)

    def compact(self) -> None:
        """Compact on-disk segment storage (removes tombstoned entries)."""
        self._inner.compact()
        if self._recorder is not None:
            self._recorder.record_compact()

    def checkpoint(self) -> None:
        """Force a checkpoint (snapshot state + truncate WAL)."""
        self._inner.checkpoint()
        if self._recorder is not None:
            self._recorder.record_checkpoint()

    def stats(self) -> Stats:
        """Get database statistics."""
        return self._inner.stats()

    @property
    def last_replay_report(self) -> t.Optional[t.Dict[str, t.Any]]:
        """Diagnostic report from the most recent replay() call."""
        if self._last_replay_report is None:
            return None
        return copy.deepcopy(self._last_replay_report)

    @property
    def last_export_replay_report(self) -> t.Optional[t.Dict[str, t.Any]]:
        """Diagnostic report from the most recent export_replay() call."""
        if self._last_export_replay_report is None:
            return None
        return copy.deepcopy(self._last_export_replay_report)

    def __repr__(self) -> str:
        s = self._inner.stats()
        embedder_name = type(self._embedder).__name__ if self._embedder else "none"
        recording = f", recording={self._recorder._path}" if self._recorder else ""
        return (
            f"CortexaDB(entries={s.entries}, dimension={s.vector_dimension}, "
            f"indexed={s.indexed_embeddings}, embedder={embedder_name}{recording})"
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __enter__(self) -> "CortexaDB":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        # Force-flush the WAL to disk before the handle is dropped.
        try:
            self._inner.flush()
        except Exception:
            pass
        if self._recorder is not None:
            self._recorder.close()
        return False
