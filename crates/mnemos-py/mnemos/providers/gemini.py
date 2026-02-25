"""
mnemos.providers.gemini — Google Gemini embedding wrapper.

Requires:
    pip install google-generativeai

Usage::

    from mnemos import Mnemos
    from mnemos.providers.gemini import GeminiEmbedder

    db = Mnemos.open(
        "agent.mem",
        embedder=GeminiEmbedder(api_key="AIza...", model="models/text-embedding-004"),
    )
    db.remember("We chose Stripe for payments")
    hits = db.ask("payment provider?")
"""

from __future__ import annotations

from typing import List, Optional

from ..embedder import Embedder

# Known output dimensions for Gemini embedding models.
_KNOWN_DIMENSIONS = {
    "models/text-embedding-004": 768,
    "models/embedding-001": 768,
}


class GeminiEmbedder(Embedder):
    """
    Embedder backed by the Google Gemini Embeddings API.

    Args:
        api_key:    Your Google AI API key (``AIza...``). If *None*, the SDK
                    will fall back to the ``GOOGLE_API_KEY`` environment variable.
        model:      Embedding model name
                    (default ``"models/text-embedding-004"``).
        dimension:  Override the output dimension. If *None*, the known default
                    for the model is used.
        task_type:  Gemini task type hint, e.g. ``"RETRIEVAL_DOCUMENT"`` or
                    ``"RETRIEVAL_QUERY"``. Defaults to ``"RETRIEVAL_DOCUMENT"``
                    for storing, ``"RETRIEVAL_QUERY"`` for querying — the
                    wrapper uses ``"RETRIEVAL_DOCUMENT"`` by default.

    Raises:
        ImportError: If the ``google-generativeai`` package is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/text-embedding-004",
        dimension: Optional[int] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiEmbedder. "
                "Install it with: pip install google-generativeai"
            )

        import google.generativeai as genai

        if api_key:
            genai.configure(api_key=api_key)

        self._genai = genai
        self._model = model
        self._task_type = task_type

        if dimension is not None:
            self._dimension = dimension
        elif model in _KNOWN_DIMENSIONS:
            self._dimension = _KNOWN_DIMENSIONS[model]
        else:
            probe = self._call_api(" ", task_type=task_type)
            self._dimension = len(probe)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _call_api(self, text: str, task_type: Optional[str] = None) -> List[float]:
        result = self._genai.embed_content(
            model=self._model,
            content=text,
            task_type=task_type or self._task_type,
        )
        return result["embedding"]

    def embed(self, text: str) -> List[float]:
        """Embed *text* using the Gemini API (RETRIEVAL_DOCUMENT task)."""
        return self._call_api(text)

    def embed_query(self, text: str) -> List[float]:
        """Embed *text* as a retrieval query (RETRIEVAL_QUERY task)."""
        return self._call_api(text, task_type="RETRIEVAL_QUERY")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, one API call per text (Gemini has no batch endpoint)."""
        return [self.embed(t) for t in texts]
