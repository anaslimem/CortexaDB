# CortexaDB Benchmark Suite

Quick benchmark runner for CortexaDB performance testing.

## Prerequisites

Before running benchmarks, you need to build the Rust extension:

```bash
cd crates/cortexadb-py
maturin develop --release
cd ../..
```

## Quick Start

```bash
# Install dependencies
pip install numpy psutil

# Generate test embeddings
python benchmark/generate_embeddings.py --count 10000 --dimensions 384

# Run benchmarks
python benchmark/run_benchmark.py --index-mode exact   # baseline
python benchmark/run_benchmark.py --index-mode hnsw    # fast mode
```

## Current Results

| Mode | Indexing | Query (p50) | Throughput | Recall |
|------|----------|-------------|-----------|--------|
| Exact | 138s | 1.34ms | 690 QPS | 100% |
| HNSW | 151s | 0.29ms | 3,203 QPS | 95% |

→ HNSW is **~5x faster** than exact with 95% recall

See the [main README](../README.md#benchmarks) for full documentation.

## Options

```bash
python benchmark/run_benchmark.py \
    --count 10000 \
    --dimensions 384 \
    --top-k 10 \
    --warmup 100 \
    --queries 1000 \
    --index-mode hnsw
```

## Output

Results saved to `results/cortexadb_<mode>_<timestamp>.json`:

```json
{
  "results": {
    "indexing_time_ms": 151520,
    "query_latency_p50_ms": 0.29,
    "query_latency_p95_ms": 0.43,
    "throughput_qps": 3203,
    "recall": 0.95,
    "disk_size_mb": 46.99
  }
}
```
