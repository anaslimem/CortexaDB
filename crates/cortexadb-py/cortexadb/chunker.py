"""
cortexadb.chunker — text chunking utilities.

Wraps the Rust chunker for multiple strategies:
- fixed: Character-based with word-boundary snapping
- recursive: Paragraph → sentence → word (general purpose)
- semantic: Split by paragraphs
- markdown: Preserve headers, lists, code blocks
- json: Flatten JSON to key-value pairs
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import typing as t

from . import _cortexadb


def chunk(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split text using the specified chunking strategy.

    Args:
        text: The input text to chunk.
        strategy: Chunking strategy - "fixed", "recursive", "semantic", "markdown", or "json".
                  Default: "recursive"
        chunk_size: Target size of each chunk in characters (for fixed/recursive).
                    Default: 512
        overlap: Number of words to overlap between consecutive chunks.
                 Default: 50

    Returns:
        List of ChunkResult dictionaries with keys:
        - text: The chunk text
        - index: Chunk index
        - metadata: Optional dict with additional info (e.g., key/value for json strategy)

    Raises:
        CortexaDBError: If strategy is invalid.
    """
    results = _cortexadb.chunk(text, strategy, chunk_size=chunk_size, overlap=overlap)
    return [
        {
            "text": r.text,
            "index": r.index,
            "metadata": dict(r.metadata) if r.metadata else None,
        }
        for r in results
    ]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Legacy function for backward compatibility.
    Use chunk() for more strategies.
    """
    results = chunk(text, strategy="fixed", chunk_size=chunk_size, overlap=overlap)
    return [r["text"] for r in results]
