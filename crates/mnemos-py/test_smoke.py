import pytest
import os
import shutil
from mnemos import Mnemos, MnemosError

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
    with pytest.raises(MnemosError, match="Embedding is currently required natively"):
        db.remember("No embedding")

    # Wrong dimension on open — must first write something with dim=3
    mid = db.remember("Seed", embedding=[1.0, 0.0, 0.0])
    db.checkpoint()  # flush so the mismatch check sees entries > 0
    with pytest.raises(MnemosError, match="(?i)dimension mismatch"):
        Mnemos.open(DB_PATH, dimension=4)
