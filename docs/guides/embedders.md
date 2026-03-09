# Embedders

CortexaDB supports pluggable embedding providers. When an embedder is configured, text is automatically converted to vectors for storage and search.

## Overview

Without an embedder, you must provide raw embedding vectors manually. With an embedder, CortexaDB handles embedding automatically:

```python
# Without embedder - manual vectors
db = CortexaDB.open("db.mem", dimension=128)
db.add("text", embedding=[0.1, 0.2, ...])  # must provide embedding

# With embedder - automatic
db = CortexaDB.open("db.mem", embedder=OpenAIEmbedder())
db.add("text")  # auto-embedded
db.search("query")      # auto-embedded
```

---

## Built-in Providers

### OpenAI

Uses the OpenAI Embeddings API.

```python
from cortexadb.providers.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="sk-...",                      # or set OPENAI_API_KEY env var
    model="text-embedding-3-small"         # default model
)

db = CortexaDB.open("db.mem", embedder=embedder)
```

**Available Models:**

| Model | Dimension | Notes |
|-------|-----------|-------|
| `text-embedding-3-small` | 1536 | Default, good balance of cost and quality |
| `text-embedding-3-large` | 3072 | Higher quality, more expensive |

Supports batch embedding via `embed_batch()`.

### Gemini

Uses Google's embedding API.

```python
from cortexadb.providers.gemini import GeminiEmbedder

embedder = GeminiEmbedder(api_key="...")
db = CortexaDB.open("db.mem", embedder=embedder)
```

### Ollama

Uses a local Ollama instance for self-hosted embeddings.

```python
from cortexadb.providers.ollama import OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")
db = CortexaDB.open("db.mem", embedder=embedder)
```

**Requires:** Ollama running locally (`ollama serve`).

### HashEmbedder (Testing)

A deterministic hash-based embedder for testing. Produces consistent embeddings from text using SHA-256 — **not suitable for semantic search**.

```python
from cortexadb import HashEmbedder

embedder = HashEmbedder(dimension=64)
db = CortexaDB.open("db.mem", embedder=embedder)
```

---

## Custom Embedders

You can create your own embedder by subclassing the `Embedder` base class:

```python
from cortexadb.embedder import Embedder

class MyEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        # Your embedding logic here
        return my_model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Optional: override for efficient batch processing
        return [self.embed(t) for t in texts]
```

### Embedder Interface

| Method/Property | Required | Description |
|----------------|----------|-------------|
| `dimension` | Yes | Returns the embedding vector dimension |
| `embed(text)` | Yes | Embeds a single text string, returns `list[float]` |
| `embed_batch(texts)` | No | Embeds multiple texts, defaults to calling `embed()` in a loop |

---

## How Auto-Embedding Works

When an embedder is configured:

1. **`add(text)`** - Text is embedded via `embedder.embed(text)`, then stored with the embedding
2. **`search(query)`** - Query is embedded via `embedder.embed(query)`, then used for vector search
3. **`ingest(text)`** - Each chunk is embedded individually after chunking
4. **`load(file)`** - File is read, chunked, and each chunk is embedded

You can always override auto-embedding by providing an explicit `embedding` parameter:

```python
db.add("text", embedding=[0.1, 0.2, ...])  # uses provided embedding
```

---

## Choosing an Embedder

| Provider | Pros | Cons |
|----------|------|------|
| **OpenAI** | High quality, easy setup | Requires API key, costs money |
| **Gemini** | Good quality, Google ecosystem | Requires API key |
| **Ollama** | Free, private, no API key | Requires local setup, slower |
| **Hash** | Zero setup, deterministic | No semantic meaning, testing only |

For production use, **OpenAI** or **Ollama** are recommended depending on whether you prefer cloud or self-hosted.

---

## Next Steps

- [Quickstart](../getting-started/quickstart.md) - Get started with auto-embedding
- [Chunking](./chunking.md) - How documents are chunked before embedding
- [Python API](../api/python.md) - Full API reference
