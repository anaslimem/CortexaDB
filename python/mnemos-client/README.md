# mnemos-client

Python client for Mnemos with a simple developer UX:

- store text memories,
- retrieve plain text answers,
- keep advanced typed APIs when needed.

## Install

```bash
pip install -e .
```

Optional embedding providers:

```bash
pip install -e ".[embeddings]"
```

## Generate gRPC stubs

From this directory:

```bash
./scripts/generate_proto.sh
```

## Quickstart (Recommended)

```python
from mnemos_client import MnemosMemory

memory = MnemosMemory.from_env()
memory.store("Alice prefers concise release notes.", importance=0.8)
memory.store("Release deadline is Friday at 5 PM.", importance=0.9)

answers = memory.ask("When is the release deadline?", top_k=5)
print(answers[0])
memory.close()
```

## 0.1 API Contract (Stable Surface)

For this version line, treat the following `MnemosMemory` methods as the stable developer contract:

- `store`
- `store_many`
- `ask`
- `ask_raw`
- `ask_with_context`
- `close`

Everything else is advanced/typed API.

## Environment variables

- `MNEMOS_ADDR` default: `127.0.0.1:50051`
- `MNEMOS_NAMESPACE` default namespace used by the facade
- `MNEMOS_API_KEY` optional service API key
- `MNEMOS_PRINCIPAL_ID` optional principal id for RBAC/quota
- `MNEMOS_EMBEDDER_PROVIDER` one of: `auto`, `gemini`, `openai`
- `MNEMOS_EMBEDDER_MODEL` optional embedding model override
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` used by Gemini embedder
- `OPENAI_API_KEY` used by OpenAI embedder

## API reference

### `MnemosMemory` (high-level facade)

Create:

```python
from mnemos_client import MnemosMemory
memory = MnemosMemory.from_env()
```

Methods:

- `store(text, namespace=None, importance=0.0, metadata=None, memory_id=None) -> int`
- What it does: stores one text memory and returns `memory_id`.

- `store_many(items: Iterable[StoreItem]) -> List[int]`
- What it does: bulk helper for repeated `store`.

- `ask(query, namespace=None, top_k=5, graph_hops=None, time_start=None, time_end=None) -> List[str]`
- What it does: retrieves and returns plain text answers only.

- `ask_raw(...) -> List[QueryHit]`
- What it does: same retrieval but returns full score objects + memory payloads.

- `ask_with_context(query, ..., max_chars=2000) -> str`
- What it does: returns concatenated retrieved text for LLM prompt context.

- `close()`

### `MnemosClient` (advanced typed API)

Use when you need explicit typed objects and full control.

Create:

```python
from mnemos_client import MnemosClient
client = MnemosClient(
    "127.0.0.1:50051",
    default_namespace="agent1",
    api_key="...",
    principal_id="agent-runtime-1",
)
```

Convenience shortcuts (deprecated as primary UX):

- `remember(...)`
- `recall(...)`
- `link(...)`
- `unlink(...)`
- `forget(...)`

These still work for compatibility, but the recommended path is `MnemosMemory` for app-level usage.

Typed/low-level methods:

- `insert_text(...)`
- `query_text(...)`
- `insert_memory(Memory, request_id=None)`
- `query(QueryRequest)`
- `delete_memory(id)`
- `add_edge(from_id, to_id, relation)`
- `remove_edge(from_id, to_id)`
- `stats()`
- `enforce_capacity(...)`
- `compact_segments()`

## Embedding adapters

Available helpers:

```python
from mnemos_client import make_embedder_from_env, gemini_embedder, openai_embedder
```

Examples:

```python
client.set_embedder(make_embedder_from_env(provider="auto"))
```

```python
client.set_embedder(gemini_embedder(api_key="...", model="text-embedding-004"))
```

```python
client.set_embedder(openai_embedder(api_key="...", model="text-embedding-3-small"))
```

## Examples

- `examples/simple_memory.py` simple `store` + `ask` flow
- `examples/basic_usage.py` typed client usage
