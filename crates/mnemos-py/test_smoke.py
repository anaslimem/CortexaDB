import pytest
import os
import shutil
from mnemos import Mnemos, MnemosError, HashEmbedder
from mnemos.chunker import chunk_text

DB_PATH = "/tmp/mnemos_test_py"

@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    yield
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

def test_mnemos_basic_flow():
    # 1. Open with dimension 3
    db = Mnemos.open(DB_PATH, dimension=3)
    
    # 2. Store memory
    mid = db.remember("Hello world", embedding=[1.0, 0.0, 0.0])
    
    # 3. Ask
    hits = db.ask("world", embedding=[1.0, 0.0, 0.0])
    assert len(hits) == 1
    assert hits[0].id == mid
    
    # 4. Get full memory
    mem = db.get(mid)
    assert mem.namespace == "default"
    assert mem.id == mid
    assert bytes(mem.content).decode("utf-8") == "Hello world"

    # 5. Connect
    mid2 = db.remember("Goodbye", embedding=[0.0, 1.0, 0.0])
    db.connect(mid, mid2, "related")

    # 6. Stats & Len
    stats = db.stats()
    assert stats.vector_dimension == 3
    assert stats.entries == 2
    assert len(db) == 2

    # 7. Compact and Checkpoint
    db.compact()
    db.checkpoint()


def test_mnemos_namespaces():
    db = Mnemos.open(DB_PATH, dimension=3)
    
    agent_a = db.namespace("agent_a")
    agent_b = db.namespace("agent_b")

    id_a = agent_a.remember("I am Agent A", embedding=[1.0, 0.0, 0.0])
    agent_b.remember("I am Agent B", embedding=[0.0, 1.0, 0.0])

    assert db.get(id_a).namespace == "agent_a"
    
    # Test ask filters by namespace using the wrapper
    hits_a = agent_a.ask("Agent A", embedding=[1.0, 0.0, 0.0])
    assert len(hits_a) == 1
    assert hits_a[0].id == id_a

    # Context manager test
    with Mnemos.open(DB_PATH, dimension=3) as db_ctx:
        assert len(db_ctx) == 2


def test_mnemos_error_handling():
    db = Mnemos.open(DB_PATH, dimension=3)

    # Wrong dimension map
    with pytest.raises(MnemosError, match="embedding dimension mismatch"):
        db.remember("Wrong dim", embedding=[1.0, 0.0])
        
    # Missing embedding required
    with pytest.raises(MnemosError, match="No embedder"):
        db.remember("No embedding")

    # Wrong dimension on open — the mismatch check uses in-memory stats (entries > 0)
    # so no checkpoint is required.
    mid = db.remember("Seed", embedding=[1.0, 0.0, 0.0])
    with pytest.raises(MnemosError, match="(?i)dimension mismatch"):
        Mnemos.open(DB_PATH, dimension=4)


# ---------------------------------------------------------------------------
# Phase 2.3 — Chunker
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    text = "Hello world foo bar baz " * 40   # ~1000 chars
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200

def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []

def test_chunk_text_single_chunk():
    """Short text fits in one chunk."""
    chunks = chunk_text("Short sentence.", chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "Short sentence."


# ---------------------------------------------------------------------------
# Phase 2.4 — HashEmbedder + auto-embed
# ---------------------------------------------------------------------------

def test_hash_embedder_basic():
    emb = HashEmbedder(dimension=16)
    assert emb.dimension == 16
    vec = emb.embed("hello")
    assert len(vec) == 16
    # L2 norm should be ≈ 1
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-5

def test_hash_embedder_deterministic():
    emb = HashEmbedder(dimension=32)
    assert emb.embed("same") == emb.embed("same")
    assert emb.embed("a") != emb.embed("b")

def test_open_with_embedder():
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)
    # remember without explicit embedding
    mid = db.remember("Auto-embedded text")
    assert mid > 0
    hits = db.ask("Auto-embedded text")
    assert len(hits) >= 1

def test_open_requires_one_of_dimension_or_embedder():
    with pytest.raises(MnemosError, match="required"):
        Mnemos.open(DB_PATH)  # neither

    with pytest.raises(MnemosError, match="not both"):
        Mnemos.open(DB_PATH, dimension=16, embedder=HashEmbedder(16))

def test_remember_without_embedder_requires_embedding():
    db = Mnemos.open(DB_PATH, dimension=3)
    with pytest.raises(MnemosError, match="No embedder"):
        db.remember("No embedding provided")

def test_ingest_document():
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)
    long_text = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    ids = db.ingest_document(long_text, chunk_size=100, overlap=20)
    assert len(ids) > 1
    assert len(set(ids)) == len(ids)   # all IDs unique
    assert db.stats().entries == len(ids)

def test_ingest_document_requires_embedder():
    db = Mnemos.open(DB_PATH, dimension=16)
    with pytest.raises(MnemosError, match="ingest_document"):
        db.ingest_document("some text")

def test_namespace_auto_embed():
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)
    ns = db.namespace("agent_a")
    mid = ns.remember("I am agent A")
    assert db.get(mid).namespace == "agent_a"
    hits = ns.ask("agent A")
    assert any(h.id == mid for h in hits)


# ---------------------------------------------------------------------------
# Phase 3.1 — Namespace Model
# ---------------------------------------------------------------------------

def test_namespace_isolation():
    """Memories in namespace A should not appear in namespace B results."""
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)

    agent_a = db.namespace("agent_a")
    agent_b = db.namespace("agent_b")

    mid_a = agent_a.remember("I am agent A, secret info")
    mid_b = agent_b.remember("I am agent B, different info")

    hits_a = agent_a.ask("agent A", top_k=10)
    hits_b = agent_b.ask("agent B", top_k=10)

    a_ids = {h.id for h in hits_a}
    b_ids = {h.id for h in hits_b}

    assert mid_a in a_ids,  "Agent A memory not found in agent_a namespace"
    assert mid_b not in a_ids, "Agent B memory leaked into agent_a namespace"
    assert mid_b in b_ids,  "Agent B memory not found in agent_b namespace"
    assert mid_a not in b_ids, "Agent A memory leaked into agent_b namespace"


def test_namespaced_ask_param():
    """db.ask(query, namespaces=[...]) should scope results correctly."""
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)

    mid_a = db.remember("Agent A private", namespace="agent_a")
    mid_b = db.remember("Agent B private", namespace="agent_b")
    mid_s = db.remember("Shared knowledge", namespace="shared")

    # Single namespace via namespaces= param
    hits = db.ask("knowledge", namespaces=["shared"])
    ids = {h.id for h in hits}
    assert mid_s in ids
    assert mid_a not in ids
    assert mid_b not in ids


def test_cross_namespace_fan_out():
    """namespaces=[a, b] should return merged re-ranked results from both."""
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)

    mid_a = db.remember("Agent A knowledge", namespace="agent_a")
    mid_s = db.remember("Shared knowledge",  namespace="shared")
    db.remember("Agent B only",              namespace="agent_b")

    hits = db.ask("knowledge", namespaces=["agent_a", "shared"], top_k=10)
    ids = {h.id for h in hits}

    # Both agent_a and shared results must be present.
    assert mid_a in ids
    assert mid_s in ids


def test_global_ask_returns_all_namespaces():
    """db.ask(query) with no namespaces= should search globally."""
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)

    mid_a = db.remember("Agent A fact", namespace="agent_a")
    mid_b = db.remember("Agent B fact", namespace="agent_b")
    mid_s = db.remember("Shared fact",  namespace="shared")

    hits = db.ask("fact", top_k=10)
    ids = {h.id for h in hits}
    assert mid_a in ids
    assert mid_b in ids
    assert mid_s in ids


def test_readonly_namespace():
    """A readonly namespace should allow ask() but reject remember()."""
    emb = HashEmbedder(dimension=32)
    db = Mnemos.open(DB_PATH, embedder=emb)

    # Write to shared normally.
    mid = db.namespace("shared").remember("Public knowledge")

    # Read from a readonly view.
    ro = db.namespace("shared", readonly=True)
    hits = ro.ask("Public knowledge")
    assert any(h.id == mid for h in hits)

    # Writes must be rejected.
    with pytest.raises(MnemosError, match="read-only"):
        ro.remember("Trying to write")

    with pytest.raises(MnemosError, match="read-only"):
        ro.ingest_document("Document text")
