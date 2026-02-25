"""
cortexadb.chunker — deterministic fixed-window text splitter.

No external dependencies. Splits on character boundaries with word-boundary snapping
so that chunks never cut mid-word.
"""

from __future__ import annotations
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping fixed-size windows.

    The splitter works on characters (model-agnostic). It snaps to the nearest
    whitespace so chunks never split a word in two.

    Args:
        text:       The input text to chunk.
        chunk_size: Target size of each chunk in characters (default 512).
        overlap:    Number of characters shared between consecutive chunks
                    (default 50). Must be < chunk_size.

    Returns:
        A list of non-empty string chunks.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
        )

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Snap backwards to the nearest whitespace to avoid mid-word cuts.
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += step

    return chunks
