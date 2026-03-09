# Query Engine

CortexaDB's hybrid query engine combines three retrieval signals — vector similarity, graph relations, and temporal recency — into a single scored result set.

## Overview

Unlike traditional vector databases that only consider embedding distance, CortexaDB's query planner can combine multiple signals:

| Signal | What It Measures | Default Weight |
|--------|-----------------|----------------|
| **Vector** | Cosine similarity between query and stored embeddings | 70% |
| **Importance** | User-defined importance score per memory | 20% |
| **Recency** | How recently the memory was created | 10% |

---

## Query Flow

```
Query Embedding
      │
      ▼
┌─────────────────┐
│  Vector Search   │  ← Exact (brute-force) or HNSW (approximate)
│  top candidates  │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Graph?  │──── Yes ──→ BFS expansion from top results
    └────┬────┘              merge expanded candidates
         │
    ┌────▼────┐
    │ Temporal?│──── Yes ──→ Filter by time range
    └────┬────┘
         │
    ┌────▼─────────────┐
    │  Score Combination │  ← Weighted sum of signals
    │  Sort & Return     │
    └──────────────────┘
```

---

## Execution Paths

The query planner selects the optimal execution path based on the options provided:

| Path | When Used |
|------|-----------|
| `VectorOnly` | No graph or temporal options specified |
| `VectorTemporal` | Time range filter provided |
| `VectorGraph` | Graph expansion enabled |
| `WeightedHybrid` | Multiple signals active |

---

## Vector Search

### Exact Mode (Default)

Brute-force cosine similarity scan over all embeddings in the target collection.

- **Complexity**: O(n)
- **Recall**: 100%
- **Best for**: Datasets under 10,000 entries

### HNSW Mode

Approximate nearest neighbor search using USearch.

- **Complexity**: O(log n)
- **Recall**: ~95%
- **Best for**: Datasets over 10,000 entries

See the [Indexing guide](./indexing.md) for HNSW configuration.

### Candidate Multiplier

To improve result quality with filtering, the vector search fetches more candidates than `top_k`:

```
candidate_k = top_k * candidate_multiplier
```

The default multiplier is 4 for collection-scoped queries (to account for filtering overhead).

---

## Graph Expansion

When `use_graph=True`, the query engine expands results using BFS traversal:

1. Run initial vector search to find top candidates
2. From each candidate, traverse outgoing edges up to N hops
3. Merge discovered memories into the candidate set
4. Re-score the expanded set
5. Return final top-k

```python
hits = db.search("query", use_graph=True)
```

Graph expansion only follows edges within the same collection.

### Hop Depth

The default expansion depth is 1 hop. With intent anchors (advanced), the depth can automatically increase to 2 or 3 hops based on query characteristics.

---

## Temporal Filtering

Time-based filtering uses the `created_at` timestamp stored with each memory.

### Recency Scoring

When recency is part of the scoring weights:
- Recently created memories receive a score of ~1.0
- Score decays with age
- Combined with other signals via weighted sum

```python
hits = db.search("query", recency_bias=True)
```

---

## Scoring

The final score for each candidate is a weighted sum:

```
score = (similarity_weight * similarity)
      + (importance_weight * importance)
      + (recency_weight * recency)
```

### Default Weights

| Signal | Weight |
|--------|--------|
| Similarity | 70% |
| Importance | 20% |
| Recency | 10% |

### Metadata Filtering

Results can be filtered by metadata key-value pairs:

```python
# Only return memories with source="onboarding"
hits = db.search("query", metadata_filter={"source": "onboarding"})
```

Metadata filtering is applied after vector search but before final scoring.

---

## Collection Scoping

Queries can be scoped to a specific collection:

```python
col = db.collection("agent_a")
hits = col.query("query").execute()
```

When querying within a collection, the engine over-fetches candidates globally (4x `top_k`), filters to the target collection, then returns the final `top_k`.

---

## Intent Anchors (Advanced)

Intent anchors are an advanced feature for automatic query tuning. Three anchor embeddings (`semantic`, `recency`, `graph`) define reference points in the embedding space.

The query embedding's proximity to each anchor determines:
- Score weight reallocation (shift emphasis toward the closest anchor)
- Graph expansion depth (automatically deeper if the query is "graph-like")

This is primarily used for advanced multi-signal retrieval pipelines.

---

## Python API

```python
# Basic vector search
hits = db.search("What does the user prefer?")

# With graph expansion
hits = db.search("query", use_graph=True)

# With recency bias
hits = db.search("query", recency_bias=True)

# Custom top_k
hits = db.search("query", top_k=10)

# Combined
hits = db.search("query", top_k=10, use_graph=True, recency_bias=True)
```

---

## Next Steps

- [Indexing](./indexing.md) - Configure exact vs HNSW search
- [Collections](./collections.md) - Multi-agent memory isolation
- [Python API](../api/python.md) - Full API reference
