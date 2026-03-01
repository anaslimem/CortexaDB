#!/usr/bin/env python3
"""
Generate test embeddings using random vectors.
Uses random embeddings for benchmarking - this is better for performance testing
since we're testing database speed, not embedding quality.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np


def generate_random_embeddings(
    count: int, dimensions: int, seed: int = 42
) -> list[list[float]]:
    """Generate random embeddings for benchmarking."""
    print(f"Generating {count} random embeddings with {dimensions} dimensions...")

    # Use fixed seed for reproducibility
    np.random.seed(seed)

    # Generate random vectors and normalize them (like real embeddings)
    embeddings = np.random.randn(count, dimensions).astype(np.float32)

    # Normalize to unit length (like cosine similarity embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    print(f"Generated {len(embeddings)} embeddings with {dimensions} dimensions")

    return embeddings.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Generate test embeddings for benchmarking"
    )
    parser.add_argument(
        "--count", type=int, default=10000, help="Number of embeddings to generate"
    )
    parser.add_argument(
        "--dimensions", type=int, default=384, help="Embedding dimensions"
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Default output path - use absolute path from script location
    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = str(
            script_dir / "results" / f"embeddings_{args.count}_{args.dimensions}.npy"
        )

    # Check if cache file exists
    if os.path.exists(args.output):
        print(f"Loading cached embeddings from {args.output}")
        embeddings = np.load(args.output)
        print(f"Loaded {len(embeddings)} embeddings")
        return

    # Generate random embeddings
    embeddings = generate_random_embeddings(args.count, args.dimensions, args.seed)

    # Convert to numpy and save
    embeddings = np.array(embeddings, dtype=np.float32)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved {len(embeddings)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
