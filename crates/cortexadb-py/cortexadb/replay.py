"""
cortexadb.replay — deterministic replay log (NDJSON format).

A replay log is a newline-delimited JSON file that records every write
operation on a CortexaDB database.  It can be used for:

* **Agent debugging** — replay an agent session step-by-step.
* **Reproducible experiments** — recreate exact database states.
* **Deterministic evaluation** — run the same sequence on different hardware.

File format
-----------
Line 1 — header (JSON object):

    {"cortexadb_replay": "1.0", "dimension": 3, "sync": "strict", "recorded_at": "2026-02-25T03:00:00Z"}

Lines 2..N — operation records (one JSON object per line):

    {"op": "remember", "id": 1, "text": "...", "embedding": [...], "namespace": "default", "metadata": null}
    {"op": "connect",  "from_id": 1, "to_id": 2, "relation": "caused_by"}
    {"op": "compact"}

The ``id`` field in ``remember`` records is the *original* memory ID assigned
during recording.  :class:`ReplayReader` builds an old→new ID mapping when
replaying so that ``connect`` operations translate correctly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


REPLAY_FORMAT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ReplayWriter:
    """
    Appends NDJSON operation records to a replay log file.

    The file is opened in **append mode** so multiple recording sessions can
    be concatenated (though the header is only written once, when the file
    does not yet exist).

    Args:
        path:      Path to the ``.log`` file.
        dimension: Embedding dimension of the database.
        sync:      Sync policy string (``"strict"``, ``"async"``, ``"batch"``).

    Example::

        writer = ReplayWriter("session.log", dimension=128, sync="strict")
        writer.record_remember(id=1, text="hello", embedding=[...], namespace="default")
        writer.close()
    """

    def __init__(self, path: str, dimension: int, sync: str = "strict") -> None:
        self._path = Path(path)
        self._file = self._path.open("a", encoding="utf-8", buffering=1)

        # Only write the header when creating a new file.
        if self._path.stat().st_size == 0:
            header = {
                "cortexadb_replay": REPLAY_FORMAT_VERSION,
                "dimension": dimension,
                "sync": sync,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            self._file.write(json.dumps(header) + "\n")

    # ------------------------------------------------------------------
    # Op recorders
    # ------------------------------------------------------------------

    def record_remember(
        self,
        *,
        id: int,
        text: str,
        embedding: List[float],
        namespace: str,
        metadata: Optional[Dict[str, str]],
    ) -> None:
        """Append a ``remember`` operation."""
        self._write({
            "op": "remember",
            "id": id,
            "text": text,
            "embedding": embedding,
            "namespace": namespace,
            "metadata": metadata,
        })

    def record_connect(self, *, from_id: int, to_id: int, relation: str) -> None:
        """Append a ``connect`` operation."""
        self._write({"op": "connect", "from_id": from_id, "to_id": to_id, "relation": relation})

    def record_compact(self) -> None:
        """Append a ``compact`` operation."""
        self._write({"op": "compact"})

    def record_checkpoint(self) -> None:
        """Append a ``checkpoint`` operation."""
        self._write({"op": "checkpoint"})

    def record_delete(self, id: int) -> None:
        """Append a ``delete`` operation."""
        self._write({"op": "delete", "id": id})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")

    def flush(self) -> None:
        """Flush the write buffer to disk."""
        self._file.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.flush()
        self._file.close()

    def __enter__(self) -> "ReplayWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class ReplayHeader:
    """Parsed replay log header."""

    def __init__(self, raw: dict) -> None:
        version = raw.get("cortexadb_replay")
        if version != REPLAY_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported replay log version: {version!r}. "
                f"Expected {REPLAY_FORMAT_VERSION!r}."
            )
        self.version: str   = version
        self.dimension: int = int(raw["dimension"])
        self.sync: str      = raw.get("sync", "strict")
        self.recorded_at: str = raw.get("recorded_at", "")

    def __repr__(self) -> str:
        return (
            f"ReplayHeader(version={self.version!r}, dimension={self.dimension}, "
            f"sync={self.sync!r}, recorded_at={self.recorded_at!r})"
        )


class ReplayReader:
    """
    Reads a replay log and supplies ``(header, operations)`` to the caller.

    Usage::

        reader = ReplayReader("session.log")
        print(reader.header)  # ReplayHeader(...)
        for op in reader.operations():
            print(op)         # {"op": "remember", ...}
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Replay log not found: {self._path}")

        with self._path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if not first_line:
            raise ValueError(f"Replay log is empty: {self._path}")

        self.header = ReplayHeader(json.loads(first_line))

    def operations(self) -> Iterator[dict]:
        """Yield each operation record (skips the header line)."""
        with self._path.open("r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
