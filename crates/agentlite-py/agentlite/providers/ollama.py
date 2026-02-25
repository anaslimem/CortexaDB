"""
agentlite.providers.ollama — Ollama local embedding wrapper.

Requires:
    pip install ollama
    (and a running Ollama server: https://ollama.ai)

Usage::

    from agentlite import AgentLite
    from agentlite.providers.ollama import OllamaEmbedder

    db = AgentLite.open(
        "agent.mem",
        embedder=OllamaEmbedder(model="nomic-embed-text"),
    )
    db.remember("We chose Stripe for payments")
    hits = db.ask("payment provider?")
"""

from __future__ import annotations

from typing import List, Optional

from ..embedder import Embedder


class OllamaEmbedder(Embedder):
    """
    Embedder backed by a local Ollama server.

    No API key required — Ollama runs entirely on your machine.

    Args:
        model:    Ollama model name (e.g. ``"nomic-embed-text"``, ``"mxbai-embed-large"``).
        host:     Ollama server URL (default ``"http://localhost:11434"``).

    Raises:
        ImportError:  If the ``ollama`` package is not installed.
        RuntimeError: If the Ollama server is unreachable on first call.

    Example::

        # Pull the model first (one-time):
        #   $ ollama pull nomic-embed-text
        from agentlite.providers.ollama import OllamaEmbedder
        embedder = OllamaEmbedder(model="nomic-embed-text")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ) -> None:
        try:
            import ollama  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required for OllamaEmbedder. "
                "Install it with: pip install ollama"
            )

        import ollama as _ollama

        self._ollama = _ollama
        self._model = model
        self._host = host
        self._client = _ollama.Client(host=host)

        # Probe to discover dimension (and fail early if server is down).
        probe = self._call_api(" ")
        self._dimension = len(probe)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _call_api(self, text: str) -> List[float]:
        response = self._client.embeddings(model=self._model, prompt=text)
        return response["embedding"]

    def embed(self, text: str) -> List[float]:
        """Embed *text* using the local Ollama server."""
        return self._call_api(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, one call per text (Ollama has no batch API)."""
        return [self.embed(t) for t in texts]
