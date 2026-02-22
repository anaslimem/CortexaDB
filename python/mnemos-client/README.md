# mnemos-client

Typed Python client for the Mnemos gRPC server.

## Install

```bash
pip install -e .
```

## Generate gRPC stubs

From this directory:

```bash
./scripts/generate_proto.sh
```

This compiles `../../proto/mnemos.proto` into `src/mnemos_client/proto/`.

## Usage

```python
from mnemos_client import MnemosClient

client = MnemosClient("127.0.0.1:50051")

client.insert_text(
    namespace="agent1",
    text="hello from developer API",
    importance=0.8,
    metadata={"source": "demo"},
)

result = client.query_text("hello", top_k=5, namespace="agent1")
for hit in result.hits:
    print(hit.text)
    print(hit.memory.metadata if hit.memory else {})
```

By default, `insert_text` / `query_text` use a deterministic local fallback embedder.
For production, configure your own embedding model once:

```python
client.set_embedder(lambda text: your_embedding_model.embed(text))
```
