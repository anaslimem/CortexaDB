from __future__ import annotations

from typing import Callable, List


def make_embedder_from_env(
    *,
    provider: str = "auto",
    model: str | None = None,
    dim_fallback: int = 3,
) -> Callable[[str], List[float]]:
    provider = provider.lower().strip()
    if provider in ("auto", "gemini"):
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if key:
            return gemini_embedder(api_key=key, model=model or "text-embedding-004")
        if provider == "gemini":
            raise RuntimeError("Gemini embedder requested but GEMINI_API_KEY/GOOGLE_API_KEY is missing")

    if provider in ("auto", "openai"):
        key = _env("OPENAI_API_KEY")
        if key:
            return openai_embedder(api_key=key, model=model or "text-embedding-3-small")
        if provider == "openai":
            raise RuntimeError("OpenAI embedder requested but OPENAI_API_KEY is missing")

    from .client import default_hash_embedder

    return lambda text: default_hash_embedder(text, dim=dim_fallback)


def gemini_embedder(*, api_key: str, model: str = "text-embedding-004") -> Callable[[str], List[float]]:
    try:
        from google import genai
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-genai is not installed. Install: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)

    def _embed(text: str) -> List[float]:
        out = client.models.embed_content(model=model, contents=text)
        emb = getattr(out, "embeddings", None)
        if emb and len(emb) > 0:
            values = getattr(emb[0], "values", None)
            if values:
                return [float(v) for v in values]
        raise RuntimeError("Gemini embedding response did not contain values")

    return _embed


def openai_embedder(*, api_key: str, model: str = "text-embedding-3-small") -> Callable[[str], List[float]]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed. Install: pip install openai") from exc

    client = OpenAI(api_key=api_key)

    def _embed(text: str) -> List[float]:
        out = client.embeddings.create(model=model, input=text)
        if not out.data:
            raise RuntimeError("OpenAI embedding response did not contain data")
        return [float(v) for v in out.data[0].embedding]

    return _embed


def _env(name: str) -> str | None:
    import os

    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
