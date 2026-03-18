# Benchmarks

CortexaDB v1.0.0 benchmarked with **10,000 embeddings** at **384 dimensions** (typical sentence-transformer size) on an M-series Mac.

> **Build mode note:** Numbers below are from a debug build. A release build (`maturin develop --release`) is 5–10x faster.

## Results

| Mode | Index Time | p50 | p95 | p99 | Throughput | Recall |
|------|-----------|-----|-----|-----|-----------|--------|
| **HNSW** | 286s | **1.03ms** | 1.18ms | 1.29ms | **952 QPS** | **95%** |
| Exact | 275s | 16.38ms | 22.69ms | 35.77ms | 56 QPS | 100% |

**HNSW is ~16x faster than exact search (debug build) while maintaining 95% recall.**

> With a release build (`maturin develop --release`), expect HNSW p50 ≈ 0.3ms and 3,000+ QPS.

---

## Disk Usage

| Mode | Disk Size |
|------|-----------|
| HNSW | 47 MB |
| Exact | 31 MB |

---

## Methodology

- **Dataset**: 10,000 random embeddings × 384 dimensions
- **Environment**: M-series Mac, debug build via `maturin develop`
- **Indexing**: Time to add 10,000 vectors + `checkpoint()` to flush
- **Query Latency**: p50/p95/p99 across 1,000 queries after 100 warmup queries
- **Recall**: % of HNSW results that match brute-force exact scan (100 queries, top-10)

---

## Reproducing Results

### Prerequisites

```bash
# Build the Rust extension (release mode for published numbers)
cd crates/cortexadb-py
maturin develop --release
cd ../..
pip install numpy psutil
```

### Generate Test Data

```bash
python3 benchmark/generate_embeddings.py --count 10000 --dimensions 384
```

### Run Benchmarks

```bash
# Exact mode (baseline, 100% recall)
python3 benchmark/run_benchmark.py --index-mode exact

# HNSW mode (fast, ~95% recall)
python3 benchmark/run_benchmark.py --index-mode hnsw
```

Results are saved to `benchmark/results/`.

### Custom Options

```bash
python3 benchmark/run_benchmark.py \
    --count 10000 \
    --dimensions 384 \
    --top-k 10 \
    --warmup 100 \
    --queries 1000 \
    --index-mode hnsw
```

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | 10000 | Number of embeddings |
| `--dimensions` | 384 | Vector dimension |
| `--top-k` | 10 | Results per query |
| `--warmup` | 100 | Warmup queries before measurement |
| `--queries` | 1000 | Number of timed queries |
| `--index-mode` | `exact` | `exact` or `hnsw` |

---

## Interpreting Results

### When to Use Exact

- Dataset under 10,000 entries
- 100% recall is required
- Simple setup is preferred

### When to Use HNSW

- Dataset over 10,000 entries
- Sub-millisecond latency is needed (release build)
- 95%+ recall is acceptable
- High query throughput is needed

### Tuning HNSW for Higher Recall

If 95% recall isn't enough, increase `ef_search`:

```python
db = CortexaDB.open("db.mem", dimension=384, index_mode={
    "type": "hnsw",
    "ef_search": 200,  # Higher = better recall, slower
    "m": 32            # More connections = better recall
})
```

This can push recall above 99% at the cost of some latency.

---

## Next Steps

- [Indexing](../guides/indexing.md) - HNSW configuration and tuning
- [Configuration](../guides/configuration.md) - All database options
