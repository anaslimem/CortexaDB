#!/usr/bin/env python3
"""
Main benchmark script to test CortexaDB performance.
Runs CortexaDB with different configurations and prints results.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np


def load_embeddings(path: str) -> tuple[list[list[float]], int]:
    """Load embeddings from numpy file."""
    print(f"Loading embeddings from {path}")
    data = np.load(path)
    embeddings = data.tolist()
    dimensions = len(embeddings[0])
    print(f"Loaded {len(embeddings)} embeddings with {dimensions} dimensions")
    return embeddings, dimensions


def run_cortexadb(embeddings: list[list[float]], dimensions: int, args) -> dict:
    """Run CortexaDB benchmark."""
    print("\n" + "=" * 50)
    print(f"RUNNING CORTEXADB BENCHMARK ({args.index_mode.upper()} MODE)")
    print("=" * 50)

    # Import and run
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "crates", "cortexadb-py")
    )
    from cortexadb_runner import run_benchmark

    queries = embeddings[:1000]

    return run_benchmark(
        embeddings=embeddings,
        queries=queries,
        top_k=args.top_k,
        warmup_queries=args.warmup,
        benchmark_queries=args.queries,
        index_mode=args.index_mode,
    )


def print_results(results: dict, index_mode: str):
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS - CORTEXADB ({index_mode.upper()} MODE)")
    print("=" * 70)

    metrics = [
        ("Indexing Time (ms)", "indexing_time_ms"),
        ("Query Latency p50 (ms)", "query_latency_p50_ms"),
        ("Query Latency p95 (ms)", "query_latency_p95_ms"),
        ("Query Latency p99 (ms)", "query_latency_p99_ms"),
        ("Query Latency Avg (ms)", "query_latency_avg_ms"),
        ("Throughput (QPS)", "throughput_qps"),
        ("Recall", "recall"),
        ("Memory Used (MB)", "memory_used_mb"),
        ("Disk Size (MB)", "disk_size_mb"),
    ]

    for name, key in metrics:
        val = results.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.2f}"
        print(f"{name:<30}: {val}")

    print("=" * 70)


def save_results(results: dict, args):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "timestamp": timestamp,
        "config": {
            "embeddings": args.embeddings,
            "dimensions": args.dimensions,
            "count": args.count,
            "top_k": args.top_k,
            "warmup": args.warmup,
            "queries": args.queries,
            "index_mode": args.index_mode,
        },
        "results": results,
    }

    output_file = output_dir / f"cortexadb_{args.index_mode}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="CortexaDB benchmark")

    # Embeddings
    parser.add_argument(
        "--embeddings", type=str, default=None, help="Path to embeddings file"
    )
    parser.add_argument("--count", type=int, default=10000, help="Number of embeddings")
    parser.add_argument(
        "--dimensions", type=int, default=384, help="Embedding dimensions"
    )

    # Benchmark options
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup queries")
    parser.add_argument("--queries", type=int, default=1000, help="Benchmark queries")
    parser.add_argument(
        "--index-mode",
        type=str,
        default="hnsw",
        choices=["exact", "hnsw"],
        help="Index mode: exact or hnsw",
    )

    args = parser.parse_args()

    # Find embeddings file
    if args.embeddings is None:
        args.embeddings = f"results/embeddings_{args.count}_{args.dimensions}.npy"

    embeddings_file = Path(__file__).parent / args.embeddings

    if not embeddings_file.exists():
        print(f"ERROR: Embeddings file not found: {embeddings_file}")
        print("\nPlease generate embeddings first:")
        print(
            f"  python generate_embeddings.py --count {args.count} --dimensions {args.dimensions}"
        )
        sys.exit(1)

    # Load embeddings
    embeddings, dimensions = load_embeddings(str(embeddings_file))
    args.dimensions = dimensions
    args.count = len(embeddings)

    # Run benchmark
    results = run_cortexadb(embeddings, dimensions, args)

    # Print results
    print_results(results, args.index_mode)

    # Save results
    save_results(results, args)


if __name__ == "__main__":
    main()
