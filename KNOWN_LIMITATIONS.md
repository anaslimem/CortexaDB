# Known Limitations

AgentLite is in active early development (v0.1). This document lists current constraints so you can make informed decisions about when and how to use it.

---

### 1. Single-Node Only

AgentLite runs as a single process. There is no replication, clustering, or distributed consensus. If the node goes down, the service is unavailable until it restarts (data is safe on disk thanks to WAL durability).

**Implication**: suitable for single-machine agent deployments and dev/staging environments. Not yet suitable for HA production setups.

### 2. ANN Backend Is Experimental

The default vector backend is `exact` (brute-force cosine similarity over all entries). An `ann` backend exists (`MNEMOS_VECTOR_BACKEND=ann`) but is early-stage and not yet extensively benchmarked.

**Implication**: for datasets beyond ~100k vectors, exact search latency will grow linearly. Switch to `ann` at your own risk and benchmark for your workload.

### 3. Ingest Throughput Depends on Sync Policy

The default sync policy is `strict` (fsync after every write). This is the safest option but limits write throughput.

| Policy | Durability | Throughput |
|--------|-----------|------------|
| `strict` | Every op durable | ~75–150 ops/s (hardware dependent) |
| `batch` | Durable per batch | ~150–300 ops/s |
| `async` | Periodic flush | ~150–300 ops/s |

**Implication**: if your agent workload is write-heavy, consider `batch` mode. Understand that `async` mode can lose the last few milliseconds of writes on crash.

### 4. Default `dim=3` Is for Demos Only

The server defaults to `MNEMOS_VECTOR_DIM=3` so that quickstart examples work out of the box with hand-crafted vectors. Real embedding models produce higher-dimensional vectors (e.g., 768 or 1536).

**Implication**: always set `MNEMOS_VECTOR_DIM` explicitly in any non-demo deployment to match your embedding model's output dimension.
