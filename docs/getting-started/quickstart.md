# Quickstart

This guide gets you from zero to a working CortexaDB instance in under 5 minutes.

## Python Quickstart

### 1. Install

```bash
pip install cortexadb
```

### 2. Open a Database

```python
from cortexadb import CortexaDB
from cortexadb.providers.openai import OpenAIEmbedder

# With an embedder (recommended) - auto-embeds text
db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

# Or with manual embeddings
db = CortexaDB.open("agent.mem", dimension=128)
```

### 3. Store Memories

```python
# Auto-embedding (requires embedder)
mid1 = db.add("The user prefers dark mode.")
mid2 = db.add("User works at Stripe.")

# With metadata
mid3 = db.add("User's name is Alice.", metadata={"source": "onboarding"})
```

### 4. Query Memories

```python
# Semantic search
hits = db.search("What does the user like?")
for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score:.3f}")

# Retrieve full memory
mem = db.get_memory(hits[0].id)
print(mem.content)  # b"The user prefers dark mode."
```

### 5. Connect Memories (Graph)

```python
db.connect(mid1, mid2, "relates_to")
neighbors = db.get_neighbors(mid1)
# [Edge(to=mid2, relation="relates_to")]
```

### 6. Ingest Documents

```python
# Chunk and store a document
db.load("document.pdf", strategy="recursive")

# Or ingest raw text
db.ingest("Long article text here...", strategy="markdown")
```

### 7. Use Namespaces

```python
agent_a = db.collection("agent_a")
agent_a.add("Agent A's private memory")
hits = agent_a.search("query only agent A's memories")
```

---

## Rust Quickstart

### 1. Add Dependency

```toml
[dependencies]
cortexadb-core = { git = "https://github.com/anaslimem/CortexaDB.git" }
```

### 2. Basic Usage

```rust
use cortexadb_core::CortexaDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = CortexaDB::open("/tmp/agent.mem", 128)?;

    // Store a memory with an embedding
    let embedding = vec![0.1; 128];
    let id = db.add(embedding.clone(), None)?;

    // Query
    let hits = db.search(embedding, 5, None)?;
    for hit in &hits {
        println!("ID: {}, Score: {:.3}", hit.id, hit.score);
    }

    // Connect memories
    let id2 = db.add(vec![0.2; 128], None)?;
    db.connect(id, id2, "related_to")?;

    // Checkpoint for fast recovery
    db.checkpoint()?;

    Ok(())
}
```

---

## Next Steps

- [Core Concepts](../guides/core-concepts.md) - Understand the architecture
- [Configuration](../guides/configuration.md) - Tune sync, indexing, and capacity
- [Python API Reference](../api/python.md) - Full API documentation
- [Examples](../resources/examples.md) - More code examples
