import typing as t

from ._mnemos import MnemosError, Hit, Memory, Stats
from . import _mnemos
from .embedder import Embedder
from .chunker import chunk_text


class Namespace:
    """
    A scoped context for Mnemos operations.
    Automatically applies the namespace name to all store/query operations.
    """
    def __init__(self, db: "Mnemos", name: str):
        self._db = db
        self.name = name

    def remember(
        self,
        text: str,
        embedding: t.Union[t.List[float], None] = None,
        metadata: t.Union[t.Dict[str, str], None] = None,
    ) -> int:
        """
        Store a new memory in this namespace.

        If the database was opened with an embedder, *embedding* is optional
        and will be generated automatically from *text*.
        """
        return self._db._remember_inner(
            text=text,
            embedding=embedding,
            metadata=metadata,
            namespace=self.name,
        )

    def ask(
        self,
        query: str,
        embedding: t.Union[t.List[float], None] = None,
        top_k: int = 5,
    ) -> t.List[Hit]:
        """
        Query for memories in this namespace.

        If the database was opened with an embedder, *embedding* is optional
        and will be generated automatically from *query*.
        """
        vec = self._db._resolve_embedding(query, embedding)
        # Ask globally then filter by namespace
        all_hits = self._db._inner.ask_embedding(embedding=vec, top_k=top_k * 4)
        filtered: t.List[Hit] = []
        for hit in all_hits:
            m = self._db._inner.get(hit.id)
            if m.namespace == self.name:
                filtered.append(hit)
                if len(filtered) == top_k:
                    break
        return filtered

    def ingest_document(
        self,
        text: str,
        *,
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: t.Union[t.Dict[str, str], None] = None,
    ) -> t.List[int]:
        """
        Split *text* into chunks and store each one in this namespace.
        Requires an embedder to be set on the database.
        """
        return self._db.ingest_document(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            namespace=self.name,
            metadata=metadata,
        )


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
    """

    def __init__(
        self,
        path: str,
        dimension: t.Optional[int],
        embedder: t.Optional[Embedder] = None,
    ):
        self._embedder = embedder
        try:
            self._inner = _mnemos.Mnemos.open(path, dimension=dimension)
        except Exception as e:
            raise MnemosError(str(e))

    @classmethod
    def open(
        cls,
        path: str,
        *,
        dimension: t.Optional[int] = None,
        embedder: t.Optional[Embedder] = None,
    ) -> "Mnemos":
        """
        Open or create a Mnemos database.

        Exactly one of *dimension* or *embedder* must be provided.

        Args:
            path:      Directory path for the database files.
            dimension: Vector embedding dimension.  Use when you supply your
                       own pre-computed embeddings.
            embedder:  An :class:`~mnemos.Embedder` instance.  The dimension
                       is inferred automatically from ``embedder.dimension``.

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
        return cls(path, dimension=dim, embedder=embedder)

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

    def namespace(self, name: str) -> Namespace:
        """Return a scoped :class:`Namespace` context (scoped remember/ask/ingest)."""
        return Namespace(self, name)

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
    ) -> t.List[Hit]:
        """
        Query the database by vector similarity.

        If an embedder is configured, *embedding* is optional — the embedder
        will be called automatically on *query*.
        """
        vec = self._resolve_embedding(query, embedding)
        return self._inner.ask_embedding(embedding=vec, top_k=top_k)

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

        Requires an embedder to be configured. Uses
        :func:`~mnemos.chunker.chunk_text` internally.

        Args:
            text:       The document text to ingest.
            chunk_size: Target size of each chunk in characters (default 512).
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

        # Use embed_batch for efficiency (providers may parallelise this).
        embeddings = self._embedder.embed_batch(chunks)

        ids: t.List[int] = []
        for chunk_text_str, vec in zip(chunks, embeddings):
            mid = self._inner.remember_embedding(
                embedding=vec,
                metadata=metadata,
                namespace=namespace,
                content=chunk_text_str,
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
