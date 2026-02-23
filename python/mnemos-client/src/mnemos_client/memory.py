from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from .client import MnemosClient
from .embedders import make_embedder_from_env
from .types import QueryHit


@dataclass(slots=True)
class StoreItem:
    text: str
    importance: float = 0.0
    metadata: Optional[Dict[str, str]] = None
    namespace: Optional[str] = None
    memory_id: Optional[int] = None


class MnemosMemory:
    """
    High-level helper API for agent developers.

    Primary UX:
    - store(text, ...)
    - ask(query, ...) -> list[str]
    """

    def __init__(self, client: MnemosClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "MnemosMemory":
        import os

        addr = os.getenv("MNEMOS_ADDR", "127.0.0.1:50051")
        namespace = os.getenv("MNEMOS_NAMESPACE") or "default"
        api_key = os.getenv("MNEMOS_API_KEY") or None
        principal_id = os.getenv("MNEMOS_PRINCIPAL_ID") or None
        provider = os.getenv("MNEMOS_EMBEDDER_PROVIDER", "auto")
        model = os.getenv("MNEMOS_EMBEDDER_MODEL")

        embedder = make_embedder_from_env(provider=provider, model=model)
        client = MnemosClient(
            addr,
            default_namespace=namespace,
            embedder=embedder,
            api_key=api_key,
            principal_id=principal_id,
        )
        return cls(client)

    @property
    def client(self) -> MnemosClient:
        return self._client

    def close(self) -> None:
        self._client.close()

    def store(
        self,
        text: str,
        *,
        namespace: Optional[str] = None,
        importance: float = 0.0,
        metadata: Optional[Dict[str, str]] = None,
        memory_id: Optional[int] = None,
    ) -> int:
        return self._client.remember(
            text,
            namespace=namespace,
            importance=importance,
            metadata=metadata,
            memory_id=memory_id,
        )

    def store_many(self, items: Iterable[StoreItem]) -> List[int]:
        memory_ids: List[int] = []
        for item in items:
            memory_ids.append(
                self.store(
                    item.text,
                    namespace=item.namespace,
                    importance=item.importance,
                    metadata=item.metadata,
                    memory_id=item.memory_id,
                )
            )
        return memory_ids

    def ask(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        graph_hops: Optional[int] = None,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> List[str]:
        hits = self._client.recall(
            query,
            namespace=namespace,
            top_k=top_k,
            graph_hops=graph_hops,
            time_start=time_start,
            time_end=time_end,
        )
        return [h.text for h in hits if h.text]

    def ask_raw(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        graph_hops: Optional[int] = None,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> List[QueryHit]:
        return self._client.recall(
            query,
            namespace=namespace,
            top_k=top_k,
            graph_hops=graph_hops,
            time_start=time_start,
            time_end=time_end,
        )

    def ask_with_context(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 8,
        max_chars: int = 2000,
        separator: str = "\n\n",
    ) -> str:
        texts = self.ask(query, namespace=namespace, top_k=top_k)
        if not texts:
            return ""
        chunks: List[str] = []
        total = 0
        for t in texts:
            if total + len(t) > max_chars:
                break
            chunks.append(t)
            total += len(t) + len(separator)
        return separator.join(chunks)
