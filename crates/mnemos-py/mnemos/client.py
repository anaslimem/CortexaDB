import typing as t

from ._mnemos import MnemosError, Hit, Memory, Stats
from . import _mnemos


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
        metadata: t.Union[t.Dict[str, str], None] = None
    ) -> int:
        """
        Store a new memory in this namespace.
        Currently requires an embedding.
        """
        if embedding is None:
            raise MnemosError(f"Embedding is currently required natively.")
            
        return self._db._inner.remember_embedding(
            embedding=embedding,
            metadata=metadata,
            namespace=self.name,
            content=text,
        )

    def ask(self, query: str, embedding: t.Union[t.List[float], None] = None, top_k: int = 5) -> t.List[Hit]:
        """
        Query for memories in this namespace using an embedding.
        Currently requires an embedding.
        Note: The underlying C API currently searches globally. We will refine this.
        """
        if embedding is None:
            raise MnemosError(f"Embedding is currently required natively.")
            
        # In the future, ask will be constrained by namespace. 
        # For now, it delegates to the backend.
        hits = self._db._inner.ask_embedding(embedding=embedding, top_k=top_k)
        # Manually filter by namespace since core `ask` doesn't enforce it yet
        filtered_hits = []
        for hit in hits:
            m = self._db.get(hit.id)
            if m.namespace == self.name:
                filtered_hits.append(hit)
                if len(filtered_hits) == top_k:
                    break
        return filtered_hits


class Mnemos:
    """
    Pythonic interface to the Mnemos embedded vector + graph database.
    """
    def __init__(self, path: str, dimension: int):
        try:
            self._inner = _mnemos.Mnemos.open(path, dimension=dimension)
        except Exception as e:
            raise MnemosError(str(e))

    @classmethod
    def open(cls, path: str, dimension: int) -> "Mnemos":
        """
        Open or create a Mnemos database.
        
        Args:
            path: Directory path for the database files.
            dimension: Vector embedding dimension (required).
        """
        return cls(path, dimension)

    def namespace(self, name: str) -> Namespace:
        """
        Return a scoped Namespace context.
        """
        return Namespace(self, name)

    def remember(
        self,
        text: str,
        embedding: t.Union[t.List[float], None] = None,
        metadata: t.Union[t.Dict[str, str], None] = None,
        namespace: str = "default",
    ) -> int:
        """
        Store a new memory.
        """
        if embedding is None:
             raise MnemosError(f"Embedding is currently required natively.")
        return self._inner.remember_embedding(
            embedding=embedding,
            metadata=metadata,
            namespace=namespace,
            content=text,
        )

    def ask(self, query: str, embedding: t.Union[t.List[float], None] = None, top_k: int = 5) -> t.List[Hit]:
        """
        Query the database by embedding vector similarity.
        """
        if embedding is None:
            raise MnemosError(f"Embedding is currently required natively.")
        return self._inner.ask_embedding(embedding=embedding, top_k=top_k)

    def get(self, mid: int) -> Memory:
        """Retrieve a full memory by ID."""
        return self._inner.get(mid)

    def connect(self, from_id: int, to_id: int, relation: str):
        """Create an edge between two memories."""
        self._inner.connect(from_id, to_id, relation)

    def compact(self):
        """Compact on-disk segment storage."""
        self._inner.compact()

    def checkpoint(self):
        """Force a checkpoint (snapshot states + truncate WAL)."""
        self._inner.checkpoint()

    def stats(self) -> Stats:
        """Get database statistics."""
        return self._inner.stats()

    def __repr__(self) -> str:
        return repr(self._inner).replace("Mnemos(", "MnemosClient(")

    def __len__(self) -> int:
        return len(self._inner)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
