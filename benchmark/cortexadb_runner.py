#!/usr/bin/env python3
"""
CortexaDB benchmark runner.
Measures indexing speed, query latency, throughput, recall, memory, and disk usage.
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def exact_search(
    embeddings: list[list[float]], query: list[float], k: int
) -> list[int]:
    """Brute-force exact search for recall comparison."""
    similarities = [
        (i + 1, cosine_similarity(query, emb)) for i, emb in enumerate(embeddings)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in similarities[:k]]


def run_benchmark(
    embeddings: list[list[float]],
    queries: list[list[float]],
    top_k: int = 10,
    warmup_queries: int = 100,
    benchmark_queries: int = 1000,
    index_mode: str = "hnsw",
) -> dict[str, Any]:
    """
    Run comprehensive benchmark on CortexaDB.

    Returns a dictionary with all metrics.
    """
    # Add parent directory to path for cortexadb
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "crates", "cortexadb-py")
    )

    from cortexadb import CortexaDB

    results = {}

    # Create temp directory for database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "cortexadb_benchmark")

    try:
        # Get process for memory tracking
        process = psutil.Process()

        # === INDEXING ===
        print(f"Building index with {index_mode} mode...")
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.perf_counter()

        # Create database with HNSW mode
        if index_mode == "hnsw":
            db = CortexaDB.open(
                db_path, dimension=len(embeddings[0]), index_mode="hnsw"
            )
        else:
            db = CortexaDB.open(db_path, dimension=len(embeddings[0]))

        for i, emb in enumerate(embeddings):
            db.remember(f"memory_{i}", embedding=emb)

        # Force checkpoint to flush
        db.checkpoint()

        indexing_time = time.perf_counter() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        results["indexing_time_ms"] = indexing_time * 1000
        results["memory_used_mb"] = round(mem_after - mem_before, 2)
        results["indexed_count"] = len(embeddings)

        # === DISK USAGE ===
        # Get directory size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(db_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        results["disk_size_mb"] = round(total_size / 1024 / 1024, 2)

        # === WARMUP ===
        print(f"Warming up with {warmup_queries} queries...")
        for i in range(warmup_queries):
            _ = db._inner.ask_embedding(
                embedding=queries[i % len(queries)], top_k=top_k
            )

        # === QUERY LATENCY ===
        print(f"Running {benchmark_queries} benchmark queries...")
        latencies = []

        for i in range(benchmark_queries):
            query = queries[i % len(queries)]

            start = time.perf_counter()
            hits = db._inner.ask_embedding(embedding=query, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            latencies.append(elapsed)

        latencies.sort()

        results["query_latency_p50_ms"] = round(np.percentile(latencies, 50), 3)
        results["query_latency_p95_ms"] = round(np.percentile(latencies, 95), 3)
        results["query_latency_p99_ms"] = round(np.percentile(latencies, 99), 3)
        results["query_latency_avg_ms"] = round(np.mean(latencies), 3)
        results["query_latency_min_ms"] = round(min(latencies), 3)
        results["query_latency_max_ms"] = round(max(latencies), 3)

        # === THROUGHPUT ===
        total_time = sum(latencies) / 1000  # Convert back to seconds
        results["throughput_qps"] = round(benchmark_queries / total_time, 2)

        # === RECALL ===
        print("Computing recall...")
        correct = 0
        total = 0

        # Sample queries for recall
        recall_queries = queries[: min(100, len(queries))]

        for query in recall_queries:
            # Get exact results
            exact_ids = exact_search(embeddings, query, top_k)

            # Get HNSW results
            hits = db._inner.ask_embedding(embedding=query, top_k=top_k)
            hnsw_ids = [hit.id for hit in hits]

            # Calculate recall
            correct += len(set(exact_ids) & set(hnsw_ids))
            total += top_k

        results["recall"] = round(correct / total, 4)

        # === CLEANUP ===
        del db

        print("Benchmark complete!")
        return results

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Test the benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CortexaDB benchmark")
    parser.add_argument(
        "--embeddings", type=str, required=True, help="Path to embeddings file"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to return"
    )
    parser.add_argument(
        "--warmup", type=int, default=100, help="Number of warmup queries"
    )
    parser.add_argument(
        "--queries", type=int, default=1000, help="Number of benchmark queries"
    )
    parser.add_argument(
        "--index-mode",
        type=str,
        default="hnsw",
        choices=["exact", "hnsw"],
        help="Index mode: exact or hnsw",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings)
    embeddings = data.tolist()
    dimensions = len(embeddings[0])
    print(f"Loaded {len(embeddings)} embeddings with {dimensions} dimensions")

    # Use first 1000 embeddings as queries
    queries = embeddings[:1000]

    # Run benchmark
    results = run_benchmark(
        embeddings=embeddings,
        queries=queries,
        top_k=args.top_k,
        warmup_queries=args.warmup,
        benchmark_queries=args.queries,
        index_mode=args.index_mode,
    )

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

    # Save to file
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
