"""
mnemos.providers.openai — OpenAI embedding wrapper.

Requires:
    pip install openai

Usage::

    from mnemos import Mnemos
    from mnemos.providers.openai import OpenAIEmbedder

    db = Mnemos.open(
        "agent.mem",
        embedder=OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small"),
    )
    db.remember("We chose Stripe for payments")
    hits = db.ask("payment provider?")
"""

from __future__ import annotations

from typing import List, Optional

from ..embedder import Embedder

# Dimension map for well-known OpenAI embedding models.
_KNOWN_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedder(Embedder):
    """
    Embedder backed by the OpenAI Embeddings API.

    Args:
        api_key:    Your OpenAI API key (``sk-...``). If *None*, the SDK
                    will fall back to the ``OPENAI_API_KEY`` environment variable.
        model:      Embedding model name (default ``"text-embedding-3-small"``).
        dimension:  Override the output dimension. Must be supported by the
                    chosen model. If *None* the known default for the model is
                    used (falls back to a probe call for unknown models).
        base_url:   Override the OpenAI-compatible API base URL (useful for
                    Azure OpenAI or local proxies).

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbedder. "
                "Install it with: pip install openai"
            )

        import openai as _openai

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = _openai.OpenAI(**kwargs)
        self._model = model

        if dimension is not None:
            self._dimension = dimension
        elif model in _KNOWN_DIMENSIONS:
            self._dimension = _KNOWN_DIMENSIONS[model]
        else:
            # Probe: embed a single space to discover the dimension.
            probe = self._call_api(" ")
            self._dimension = len(probe)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _call_api(self, text: str, **kwargs) -> List[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
            **kwargs,
        )
        return response.data[0].embedding

    def embed(self, text: str) -> List[float]:
        """Embed *text* using the OpenAI API."""
        return self._call_api(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in a single API call."""
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # Results are sorted by index, but let's be safe.
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
