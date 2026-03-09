# Collections

Collections allow you to isolate memories between different agents, workspaces, or contexts within a single CortexaDB database file.

## Overview

Every memory in CortexaDB belongs to a collection. The default collection is `"default"`. Collections provide:

- **Isolation** - Queries only return results from the target collection
- **Organization** - Group memories by agent, user, or topic
- **Access Control** - Read-only collections for shared knowledge

---

## Basic Usage

### Accessing a Collection

```python
# Get a collection handle
agent_a = db.collection("agent_a")
agent_b = db.collection("agent_b")
```

### Writing to a Collection

```python
agent_a.add("Agent A's private memory")
agent_b.add("Agent B's private memory")
```

### Querying a Collection

```python
# Only searches within agent_a's memories
hits = agent_a.search("What do I know?")

# Only searches within agent_b's memories
hits = agent_b.search("What do I know?")
```

### Deleting from a Collection

```python
agent_a.delete(memory_id)
```

### Ingesting Documents

```python
agent_a.ingest("Long document...", chunk_size=512)
```

---

## Default Collection

When you use the top-level `db.add()` and `db.search()`, memories are stored in and queried from the `"default"` collection.

```python
# These are equivalent:
db.add("text")
db.collection("default").add("text")
```

---

## Read-only Collections

You can create read-only collection handles for shared knowledge that shouldn't be modified:

```python
shared = db.collection("shared_knowledge", readonly=True)

# Reading works fine
hits = shared.search("query")

# Writing raises CortexaDBError
shared.add("text")  # Error!
```

This is useful for multi-agent systems where some agents should only read from a shared knowledge base.

---

## Graph Edge Rules

Graph edges are collection-scoped. You **cannot** create edges between memories in different collections:

```python
agent_a = db.collection("agent_a")
agent_b = db.collection("agent_b")

mid1 = agent_a.add("Memory in A")
mid2 = agent_b.add("Memory in B")

# This will raise an error - cross-collection edges are forbidden
db.connect(mid1, mid2, "relates_to")
```

Graph traversal during queries also respects collection boundaries — BFS will not cross into other collections.

---

## Common Patterns

### Multi-Agent System

```python
db = CortexaDB.open("agents.mem", dimension=128)

# Each agent has its own collection
planner = db.collection("planner")
researcher = db.collection("researcher")
writer = db.collection("writer")

# Agents store memories independently
planner.add("Task: Write a blog post about AI")
researcher.add("Found 3 relevant papers on AI agents")
writer.add("Draft: AI agents are transforming...")

# Each agent queries only its own memories
planner_context = planner.search("What tsearchs are pending?")
```

### Shared Knowledge Base

```python
# Admin writes to shared collection
shared = db.collection("shared")
shared.add("Company policy: All code must be reviewed")

# Agents read from shared collection (read-only)
agent = db.collection("shared", readonly=True)
hits = agent.search("What are the code review rules?")
```

### Per-User Memory

```python
def get_user_memory(db, user_id):
    return db.collection(f"user_{user_id}")

alice = get_user_memory(db, "alice")
alice.add("Alice prefers dark mode")

bob = get_user_memory(db, "bob")
bob.add("Bob prefers light mode")
```

---

## Next Steps

- [Core Concepts](./core-concepts.md) - How collections fit into the architecture
- [Query Engine](./query-engine.md) - How collection scoping affects queries
- [Python API](../api/python.md) - Collection API reference
