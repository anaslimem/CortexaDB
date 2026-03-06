# Storage Engine

CortexaDB uses a **log-structured storage engine** designed for crash safety and fast recovery. This guide explains each component in detail.

## Overview

The storage layer consists of four components:

| Component | File(s) | Purpose |
|-----------|---------|---------|
| WAL | `cortexadb.wal` | Write-ahead log for durability |
| Segments | `000000.seg`, `000001.seg`, ... | Append-only payload storage |
| Checkpoint | `cortexadb.ckpt` | Binary snapshot for fast recovery |
| HNSW Index | `cortexadb.hnsw` | Persisted HNSW index (optional) |

---

## Write-Ahead Log (WAL)

The WAL is the primary durability mechanism. Every write operation is first appended to the WAL before the in-memory state is updated.

### Record Format

```
┌──────────┬──────────┬─────────────────────────┐
│ len (u32)│ crc (u32)│ payload (bincode Command)│
└──────────┴──────────┴─────────────────────────┘
```

Each WAL entry contains:
- **len** - Byte length of the payload
- **crc** - CRC32 checksum of the payload
- **payload** - Bincode-serialized command

### Command Types

| Command | Description |
|---------|-------------|
| `InsertMemory(MemoryEntry)` | Store or update a memory entry |
| `DeleteMemory(MemoryId)` | Mark a memory as deleted (tombstone) |
| `AddEdge { from, to, relation }` | Create a directed graph edge |
| `RemoveEdge { from, to }` | Remove a graph edge |

### Recovery

On startup, the WAL is replayed from beginning to end. If a corrupt or incomplete record is encountered, replay stops at that point — only fully-written, checksum-valid entries are applied.

### Truncation

After a successful checkpoint, the WAL is truncated to remove entries already captured in the snapshot. Only post-checkpoint entries are retained.

---

## Segment Storage

Memory payloads (content, embeddings, metadata) are stored in append-only segment files.

### File Layout

```
data/
  000000.seg    # First segment
  000001.seg    # Second segment (after rotation)
  ...
```

### Record Format

```
┌──────────┬──────────┬──────────────────────────────┐
│ len (u32)│ crc (u32)│ payload (bincode MemoryEntry) │
└──────────┴──────────┴──────────────────────────────┘
```

### Segment Rotation

A new segment is created when the current segment exceeds **10MB**. Segments are numbered sequentially (`000000`, `000001`, ...).

### In-Memory Index

The segment layer maintains a `HashMap<MemoryId, (segment_id, offset, length)>` for O(1) lookups by memory ID.

### Tombstones

When a memory is deleted, it is marked with a tombstone flag. The physical data remains until compaction reclaims the space.

### Self-Healing

On startup, if a segment file has a corrupted tail (e.g., from an interrupted write), the tail is truncated and valid entries before the corruption point are preserved.

---

## Checkpoints

Checkpoints are binary snapshots of the state machine that enable fast startup.

### Files

| File | Contents |
|------|----------|
| `cortexadb.ckpt` | Serialized state machine + last applied command ID |
| `cortexadb.hnsw` | HNSW index binary (if using HNSW mode) |

### Checkpoint Process

1. Serialize the current state machine to binary
2. Write to `cortexadb.ckpt.tmp` (atomic write)
3. Rename `cortexadb.ckpt.tmp` to `cortexadb.ckpt`
4. Truncate the WAL (keep only post-checkpoint entries)
5. Save HNSW index to `cortexadb.hnsw` (if enabled)

The atomic rename ensures that a crash during checkpointing never leaves a corrupt checkpoint file.

### Triggering Checkpoints

```python
# Manual checkpoint
db.checkpoint()

# Automatic (Rust only - via CheckpointPolicy)
# CheckpointPolicy::Periodic { every_ops: 1000, every_ms: 30_000 }
```

---

## Compaction

Over time, deleted entries leave "holes" in segment files. Compaction reclaims this space.

### Process

1. Identify segments with tombstoned entries
2. Create backup of current segments
3. Rewrite segments without tombstoned entries
4. Atomically swap old segments for new ones
5. Clean up backups

### Safety

- Backups are created before any compaction
- If a crash occurs during compaction, the backup is used on recovery
- Temporary directories are cleaned up on startup

### Usage

```python
db.compact()
```

---

## Startup & Recovery Sequence

```
1. Does WAL exist?
   ├── No  → Fresh database
   └── Yes → Recovery mode
              │
              ├── Load checkpoint (if exists)
              │   └── Deserialize state machine
              │
              ├── Replay WAL entries after checkpoint
              │   └── Apply each valid command
              │
              ├── Rebuild segment index
              │   └── Scan all .seg files
              │
              └── Repair mismatches
                  └── Sync missing vectors to HNSW
```

### Recovery Guarantees

- Every committed write (fsync'd to WAL) is recovered
- Incomplete writes (no fsync) are safely discarded
- Checkpoint + WAL replay produces identical state to the original
- Segment corruption is handled by truncating at the corruption point

---

## File Layout Example

A typical CortexaDB directory:

```
agent.mem/
  cortexadb.wal       # Write-ahead log
  cortexadb.ckpt      # State machine snapshot
  cortexadb.hnsw      # HNSW index (if using hnsw mode)
  000000.seg          # Segment file
  000001.seg          # Segment file (after rotation)
```

---

## Next Steps

- [Configuration](./configuration.md) - Tune sync policy and checkpoint behavior
- [Query Engine](./query-engine.md) - How retrieval works on top of storage
