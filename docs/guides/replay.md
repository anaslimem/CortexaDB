# Replay & Recording

CortexaDB can record every operation to a log file and replay it to recreate an identical database. This is useful for debugging agent behavior, migrating data, or creating reproducible test scenarios.

## Overview

Recording captures all write operations (add, connect, delete, compact, checkpoint) as NDJSON (newline-delimited JSON). Replay reads the log and re-applies each operation to build a new database.

---

## Recording

Enable recording by passing a `record` path when opening the database:

```python
db = CortexaDB.open("agent.mem", dimension=128, record="session.log")

# All operations are now logged
mid1 = db.add("User likes dark mode", embedding=[...])
mid2 = db.add("User works at Stripe", embedding=[...])
db.connect(mid1, mid2, "relates_to")
db.delete(mid1)
db.compact()
db.checkpoint()
```

### Log Format

The log file is NDJSON with a header line followed by operation lines:

**Header (line 1):**
```json
{
  "cortexadb_replay": "1.0",
  "dimension": 128,
  "sync": "strict",
  "recorded_at": "2026-02-25T03:00:00Z"
}
```

**Operations (lines 2+):**
```json
{"op": "add", "id": 1, "text": "User likes dark mode", "embedding": [...], "collection": "default", "metadata": null}
{"op": "add", "id": 2, "text": "User works at Stripe", "embedding": [...], "collection": "default", "metadata": null}
{"op": "connect", "from_id": 1, "to_id": 2, "relation": "relates_to"}
{"op": "delete", "id": 1}
{"op": "compact"}
{"op": "checkpoint"}
```

---

## Replaying

Replay reads a log file and rebuilds the database from scratch:

```python
db = CortexaDB.replay("session.log", "restored.mem")
```

### Strict Mode

| Mode | Behavior |
|------|----------|
| `strict=False` (default) | Skips malformed/failed operations and continues |
| `strict=True` | Raises `CortexaDBError` on the first bad operation |

```python
# Lenient replay - skip errors
db = CortexaDB.replay("session.log", "restored.mem", strict=False)

# Strict replay - fail on any error
db = CortexaDB.replay("session.log", "restored.mem", strict=True)
```

### ID Mapping

During replay, memory IDs may differ from the original session. The replay reader maintains an old-to-new ID mapping so that `connect` operations reference the correct memories in the replayed database.

---

## Replay Reports

After a replay, a diagnostic report is available:

```python
db = CortexaDB.replay("session.log", "restored.mem", strict=False)
report = db.last_replay_report

print(report["total_ops"])   # Total operations in the log
print(report["applied"])     # Successfully applied
print(report["skipped"])     # Skipped (malformed but non-fatal)
print(report["failed"])      # Failed (execution error, non-fatal)
print(report["op_counts"])   # Per-type counts: {"add": 5, "connect": 2, ...}
print(report["failures"])    # List of up to 50 failure details
```

---

## Export Replay

You can export the current database state as a replay log. This creates a snapshot-style log that, when replayed, produces an equivalent database:

```python
db.export_replay("snapshot.log")
```

### Export Report

```python
db.export_replay("snapshot.log")
report = db.last_export_replay_report

print(report["exported"])                  # Memories written to log
print(report["skipped_missing_embedding"]) # Entries without vectors
print(report["skipped_missing_id"])        # Gaps in ID space
print(report["errors"])                    # Unexpected errors
```

---

## Use Cases

### Debugging Agent Behavior

Record a session, then replay it step by step to understand what the agent stored and when:

```python
# Production
db = CortexaDB.open("agent.mem", embedder=embedder, record="debug.log")
# ... agent runs ...

# Later, replay for analysis
db2 = CortexaDB.replay("debug.log", "/tmp/debug.mem")
```

### Data Migration

Export from one database, replay into a new one (potentially with different settings):

```python
# Export current state
db.export_replay("migration.log")

# Replay into new database with different config
db2 = CortexaDB.replay("migration.log", "new.mem", sync="async")
```

### Reproducible Tests

Record a known-good session and replay it to set up test fixtures:

```python
def setup_test_db():
    return CortexaDB.replay("fixtures/test_session.log", "/tmp/test.mem")
```

---

## Next Steps

- [Configuration](./configuration.md) - Sync and recording options
- [Python API](../api/python.md) - Full replay API reference
