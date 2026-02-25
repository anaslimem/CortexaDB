"""
cortexadb.embedder — Embedder ABC and HashEmbedder (for testing).

The Embedder protocol is the integration point between CortexaDB and any embedding
model. Implement `Embedder` and pass it to `CortexaDB.open(embedder=...)` to enable
automatic embedding.

    from cortexadb import CortexaDB
    from cortexadb.providers.openai import OpenAIEmbedder

    db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder(api_key="sk-..."))
    db.remember("We chose Stripe for payments")   # embeds automatically
    hits = db.ask("payment provider?")            # embeds query automatically
"""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """
    Abstract base class for embedding models.

    Subclass this to integrate any embedding provider (OpenAI, Gemini, Ollama,
    or your own local model) with CortexaDB.

    The two required members are:

    * ``dimension`` — the fixed output size of the embedding vectors.
    * ``embed(text)`` — produce a single embedding vector from a string.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors produced."""
        ...

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Embed *text* into a vector of length ``self.dimension``.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats of length ``self.dimension``.
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.

        The default implementation calls ``embed()`` in a loop. Override this
        method to use native batch APIs for better throughput.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input string.
        """
        return [self.embed(t) for t in texts]


class HashEmbedder(Embedder):
    """
    Deterministic hash-based embedder — **for testing only**.

    Produces vectors by hashing the input text with SHA-256, then deriving
    ``dimension`` float values from successive 4-byte blocks of the hash.
    Output is L2-normalised. Results are stable (same text → same vector) but
    are **not semantically meaningful**.

    This embedder requires no API keys or external dependencies.

    Args:
        dimension: Size of the output vector (default 64).

    Example::

        from cortexadb import CortexaDB
        from cortexadb import HashEmbedder

        db = CortexaDB.open("/tmp/test.mem", embedder=HashEmbedder(dimension=64))
        db.remember("hello world")
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be > 0, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        seed = text.encode("utf-8")
        vec: List[float] = []

        # Generate enough bytes by hashing text + chunk counter until we have
        # enough floats.
        i = 0
        while len(vec) < self._dimension:
            digest = hashlib.sha256(seed + str(i).encode()).digest()
            for j in range(0, len(digest) - 3, 4):
                if len(vec) >= self._dimension:
                    break
                (val,) = struct.unpack_from(">I", digest, j)
                vec.append(float(val))
            i += 1

        # L2 normalise.
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec[: self._dimension]
