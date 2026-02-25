import typing as t

from ._mnemos import MnemosError, Hit, Memory, Stats
from . import _mnemos
from .embedder import Embedder
from .chunker import chunk_text


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
        results = self._db._inner.ask_in_namespace(
            namespace=self.name,
            embedding=vec,
            top_k=top_k,
        )
        return results

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

    Or with an embedder for automatic embedding::

        from mnemos.providers.openai import OpenAIEmbedder
        db = Mnemos.open("agent.mem", embedder=OpenAIEmbedder(api_key="sk-..."))
        mid = db.remember("We chose Stripe for payments")
        hits = db.ask("payment provider?")

    Multi-agent namespace model::

        agent_a = db.namespace("agent_a")
        agent_b = db.namespace("agent_b")
        shared  = db.namespace("shared")

        agent_a.remember("My private memory")
        shared.remember("Shared team knowledge")

        # Read shared memory without being able to write to it:
        shared_ro = db.namespace("shared", readonly=True)
        shared_ro.ask("team knowledge?")   # works
        shared_ro.remember("...")          # raises MnemosError
    """

    def __init__(
        self,
        path: str,
        dimension: t.Optional[int],
        embedder: t.Optional[Embedder] = None,
        sync: str = "strict",
    ):
        self._embedder = embedder
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
            sync:      Write durability policy. One of:

                       * ``"strict"`` *(default)* — every write is fsynced immediately.
                         Safe against crashes, slightly lower throughput.
                       * ``"async"`` — writes are fsynced every 10 ms in the
                         background. Higher throughput but may lose the last few
                         writes on unclean shutdown.
                       * ``"batch"`` — fsync every 64 writes or 50 ms, whichever
                         comes first. A middle ground.

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
        return cls(path, dimension=dim, embedder=embedder, sync=sync)

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
        return self._inner.remember_embedding(
            embedding=vec,
            metadata=metadata,
            namespace=namespace,
            content=text,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def namespace(self, name: str, *, readonly: bool = False) -> Namespace:
        """
        Return a scoped :class:`Namespace` for partitioned memory access.

        Args:
            name:     Namespace identifier (e.g. ``"agent_a"``, ``"shared"``).
            readonly: If *True*, writes to this namespace raise
                      :class:`MnemosError` — useful for a shared read-only view.

        Example::

            agent_a = db.namespace("agent_a")
            shared  = db.namespace("shared", readonly=True)
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

        If an embedder is configured, *embedding* is optional — the embedder
        will be called automatically on *text*.
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
        Query the database by vector similarity.

        Args:
            query:      Query string (auto-embedded if embedder is configured).
            embedding:  Pre-computed query vector (overrides auto-embed).
            top_k:      Maximum number of hits to return (default 5).
            namespaces: If given, restrict search to these namespaces.

                        * ``None`` → global search (all namespaces, default)
                        * ``["agent_a"]`` → single namespace (fast Rust path)
                        * ``["agent_a", "shared"]`` → cross-namespace fan-out
                          (merges + re-ranks by score across multiple searches)

        Returns:
            List of :class:`Hit` objects ranked by descending score.
        """
        vec = self._resolve_embedding(query, embedding)

        if namespaces is None:
            # Global search — no namespace filter.
            return self._inner.ask_embedding(embedding=vec, top_k=top_k)

        if len(namespaces) == 1:
            # Single namespace — use the fast Rust path.
            return self._inner.ask_in_namespace(
                namespace=namespaces[0],
                embedding=vec,
                top_k=top_k,
            )

        # Cross-namespace fan-out: query each namespace, merge, re-rank.
        seen_ids: t.Set[int] = set()
        merged: t.List[Hit] = []
        for ns in namespaces:
            ns_hits = self._inner.ask_in_namespace(
                namespace=ns,
                embedding=vec,
                top_k=top_k,
            )
            for hit in ns_hits:
                if hit.id not in seen_ids:
                    seen_ids.add(hit.id)
                    merged.append(hit)

        # Sort by descending score and return top_k.
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:top_k]

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

        Args:
            text:       The document text to ingest.
            chunk_size: Target chunk size in characters (default 512).
            overlap:    Character overlap between consecutive chunks (default 50).
            namespace:  Namespace to store chunks in (default ``"default"``).
            metadata:   Optional metadata dict applied to every chunk.

        Returns:
            List of assigned memory IDs (one per chunk).

        Raises:
            MnemosError: If no embedder is configured.
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
            mid = self._inner.remember_embedding(
                embedding=vec,
                metadata=metadata,
                namespace=namespace,
                content=chunk_str,
            )
            ids.append(mid)

        return ids

    def get(self, mid: int) -> Memory:
        """Retrieve a full memory by ID."""
        return self._inner.get(mid)

    def connect(self, from_id: int, to_id: int, relation: str) -> None:
        """Create an edge between two memories."""
        self._inner.connect(from_id, to_id, relation)

    def compact(self) -> None:
        """Compact on-disk segment storage (removes tombstoned entries)."""
        self._inner.compact()

    def checkpoint(self) -> None:
        """Force a checkpoint (snapshot state + truncate WAL)."""
        self._inner.checkpoint()

    def stats(self) -> Stats:
        """Get database statistics."""
        return self._inner.stats()

    def __repr__(self) -> str:
        s = self._inner.stats()
        embedder_name = type(self._embedder).__name__ if self._embedder else "none"
        return (
            f"Mnemos(entries={s.entries}, dimension={s.vector_dimension}, "
            f"indexed={s.indexed_embeddings}, embedder={embedder_name})"
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __enter__(self) -> "Mnemos":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False
