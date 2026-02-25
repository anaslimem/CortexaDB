import typing as t

from ._mnemos import MnemosError, Hit, Memory, Stats
from . import _mnemos
from .embedder import Embedder
from .chunker import chunk_text
from .replay import ReplayWriter, ReplayReader


class Namespace:
    """
    A scoped context for Mnemos operations.

    Obtained via ``db.namespace(name)``.  All store and query operations
    automatically apply this namespace.

    Args:
        db:       Parent :class:`Mnemos` instance.
        name:     Namespace identifier string.
        readonly: When *True*, ``remember()`` and ``ingest_document()`` raise
                  :class:`MnemosError` — useful for shared read-only views.
    """

    def __init__(self, db: "Mnemos", name: str, *, readonly: bool = False):
        self._db = db
        self.name = name
        self._readonly = readonly

    def _check_writable(self) -> None:
        if self._readonly:
            raise MnemosError(
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
    ) -> t.List[Hit]:
        """
        Query for memories scoped to this namespace.

        Uses the fast Rust-side namespace filter (``ask_in_namespace``).
        """
        vec = self._db._resolve_embedding(query, embedding)
        return self._db._inner.ask_in_namespace(
            namespace=self.name,
            embedding=vec,
            top_k=top_k,
        )

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


class Mnemos:
    """
    Pythonic interface to the Mnemos embedded vector + graph database.

    Open with a fixed dimension (manual embedding)::

        db = Mnemos.open("agent.mem", dimension=128)
        mid = db.remember("hello", embedding=[0.1] * 128)

    Open with an embedder for automatic embedding::

        from mnemos.providers.openai import OpenAIEmbedder
        db = Mnemos.open("agent.mem", embedder=OpenAIEmbedder(api_key="sk-..."))
        mid = db.remember("We chose Stripe for payments")

    Record a session for deterministic replay::

        db = Mnemos.open("agent.mem", dimension=128, record="session.log")
        db.remember("fact A", embedding=[...])   # stored + logged
        # later ...
        db2 = Mnemos.replay("session.log", "replay.mem")

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
        _recorder: t.Optional[ReplayWriter] = None,
    ):
        self._embedder = embedder
        self._recorder = _recorder
        try:
            self._inner = _mnemos.Mnemos.open(path, dimension=dimension, sync=sync)
        except Exception as e:
            raise MnemosError(str(e))

    @classmethod
    def open(
        cls,
        path: str,
        *,
        dimension: t.Optional[int] = None,
        embedder: t.Optional[Embedder] = None,
        sync: str = "strict",
        record: t.Optional[str] = None,
    ) -> "Mnemos":
        """
        Open or create a Mnemos database.

        Exactly one of *dimension* or *embedder* must be provided.

        Args:
            path:      Directory path for the database files.
            dimension: Vector embedding dimension. Use when supplying your own
                       pre-computed embeddings.
            embedder:  An :class:`~mnemos.Embedder` instance. The dimension is
                       inferred from ``embedder.dimension`` automatically.
            sync:      Write durability policy: ``"strict"`` (default),
                       ``"async"``, or ``"batch"``.
            record:    Optional path to a replay log file. When set, every write
                       operation (remember, connect, compact) is appended to this
                       NDJSON file so the session can be replayed later.

        Raises:
            MnemosError: If neither or both of *dimension* and *embedder* are
                         provided, or if the database cannot be opened.
        """
        if embedder is not None and dimension is not None:
            raise MnemosError(
                "Provide either 'dimension' or 'embedder', not both."
            )
        if embedder is None and dimension is None:
            raise MnemosError(
                "One of 'dimension' or 'embedder' is required."
            )

        dim = embedder.dimension if embedder is not None else dimension

        recorder: t.Optional[ReplayWriter] = None
        if record is not None:
            recorder = ReplayWriter(record, dimension=dim, sync=sync)

        return cls(path, dimension=dim, embedder=embedder, sync=sync, _recorder=recorder)

    @classmethod
    def replay(
        cls,
        log_path: str,
        db_path: str,
        *,
        sync: str = "strict",
    ) -> "Mnemos":
        """
        Replay a log file into a fresh database, returning the populated instance.

        The log must have been produced by ``Mnemos.open(..., record=log_path)``
        or ``db.export_replay(log_path)``.

        Args:
            log_path:  Path to the NDJSON replay log file.
            db_path:   Directory path for the new database.
            sync:      Sync policy for the replayed database.

        Returns:
            A :class:`Mnemos` instance with all recorded operations applied.

        Raises:
            MnemosError:      If the log is invalid or replay fails.
            FileNotFoundError: If *log_path* does not exist.

        Example::

            db = Mnemos.replay("session.log", "/tmp/replayed.mem")
            hits = db.ask("payment provider?", embedding=[...])
        """
        try:
            reader = ReplayReader(log_path)
        except (FileNotFoundError, ValueError) as e:
            raise MnemosError(str(e))

        hdr = reader.header
        db = cls(db_path, dimension=hdr.dimension, sync=sync)

        # old_id → new_id mapping (connect ops use original IDs)
        id_map: t.Dict[int, int] = {}

        for record in reader.operations():
            op = record.get("op")

            if op == "remember":
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

            elif op == "connect":
                old_from = record["from_id"]
                old_to   = record["to_id"]
                new_from = id_map.get(old_from, old_from)
                new_to   = id_map.get(old_to, old_to)
                try:
                    db._inner.connect(new_from, new_to, record["relation"])
                except Exception:
                    pass  # non-fatal: IDs may not exist if log is partial

            elif op == "compact":
                db._inner.compact()

            elif op == "checkpoint":
                try:
                    db._inner.checkpoint()
                except Exception:
                    pass  # non-fatal on fresh DB

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
            raise MnemosError(
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
                      :class:`MnemosError`.
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
    ) -> t.List[Hit]:
        """
        Query by vector similarity.

        Args:
            query:      Query string (auto-embedded if embedder is configured).
            embedding:  Pre-computed query vector (overrides auto-embed).
            top_k:      Maximum hits to return (default 5).
            namespaces: Restrict search to these namespaces. ``None`` → global.
        """
        vec = self._resolve_embedding(query, embedding)

        if namespaces is None:
            return self._inner.ask_embedding(embedding=vec, top_k=top_k)

        if len(namespaces) == 1:
            return self._inner.ask_in_namespace(
                namespace=namespaces[0], embedding=vec, top_k=top_k,
            )

        seen_ids: t.Set[int] = set()
        merged: t.List[Hit] = []
        for ns in namespaces:
            for hit in self._inner.ask_in_namespace(namespace=ns, embedding=vec, top_k=top_k):
                if hit.id not in seen_ids:
                    seen_ids.add(hit.id)
                    merged.append(hit)

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:top_k]

    def connect(self, from_id: int, to_id: int, relation: str) -> None:
        """
        Create a directional edge between two memories.

        If recording is enabled, the operation is appended to the log.
        """
        self._inner.connect(from_id, to_id, relation)
        if self._recorder is not None:
            self._recorder.record_connect(from_id=from_id, to_id=to_id, relation=relation)

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
            raise MnemosError(
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

            db = Mnemos.open("agent.mem", dimension=128)
            # ... lots of work ...
            db.export_replay("snapshot.log")

            # Later on any machine:
            db2 = Mnemos.replay("snapshot.log", "restored.mem")
        """
        stats = self._inner.stats()
        dim = stats.vector_dimension

        import os
        # Truncate any existing file so we start fresh.
        if os.path.exists(log_path):
            os.remove(log_path)

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
                    # Retrieve the stored embedding via ask with dimension probe
                    hits = self._inner.ask_in_namespace(
                        namespace=mem.namespace,
                        embedding=mem.embedding if hasattr(mem, "embedding") else [0.0] * dim,
                        top_k=1,
                    )
                    writer.record_remember(
                        id=mem.id,
                        text=mem.content if hasattr(mem, "content") else "",
                        embedding=mem.embedding if hasattr(mem, "embedding") else [],
                        namespace=mem.namespace,
                        metadata=None,
                    )
                    found += 1
                except Exception:
                    pass
                candidate += 1
                checked += 1

    def get(self, mid: int) -> Memory:
        """Retrieve a full memory by ID."""
        return self._inner.get(mid)

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

    def __repr__(self) -> str:
        s = self._inner.stats()
        embedder_name = type(self._embedder).__name__ if self._embedder else "none"
        recording = f", recording={self._recorder._path}" if self._recorder else ""
        return (
            f"Mnemos(entries={s.entries}, dimension={s.vector_dimension}, "
            f"indexed={s.indexed_embeddings}, embedder={embedder_name}{recording})"
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __enter__(self) -> "Mnemos":
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
