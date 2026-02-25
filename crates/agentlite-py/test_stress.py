import pytest
import os
import shutil
import time
from agentlite import AgentLite

@pytest.fixture
def clean_db_path(request):
    db_path = f"/tmp/agentlite_stress_{request.node.name}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    yield db_path
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

def test_replay_safety(clean_db_path):
    """All entries must survive close + reopen."""
    print("\n--- Test 1: Replay Safety (5000 inserts) ---")

    with AgentLite.open(clean_db_path, dimension=2, sync="strict") as db:
        start_time = time.time()
        for i in range(5000):
            db.remember(f"Entry {i}", embedding=[0.5, 0.5])

        print(f"Inserted 5,000 memories in {time.time() - start_time:.2f}s")
        assert len(db) == 5000

    with AgentLite.open(clean_db_path, dimension=2, sync="strict") as db2:
        assert len(db2) == 5000, f"Expected 5000 entries after reopen, got {len(db2)}"

    print("Test 1 PASS")

def test_compaction_integrity(clean_db_path):
    """All entries must survive compact + reopen."""
    print("\n--- Test 3: WAL Compaction Integrity ---")

    with AgentLite.open(clean_db_path, dimension=2, sync="strict") as db:
        for _ in range(100):
            db.remember("Stress entry", embedding=[0.1, 0.9])

        assert len(db) == 100

        print("Compacting...")
        db.compact()

    with AgentLite.open(clean_db_path, dimension=2, sync="strict") as db2:
        assert len(db2) == 100, f"Expected 100 entries after compact + reopen, got {len(db2)}"

    print("Test 3 PASS")

def test_concurrent_compaction(clean_db_path):
    """Ensure parallel read operations don't fail when the storage is compacted."""
    import threading

    print("\n--- Test 4: Concurrent Compaction and Reads ---")
    with AgentLite.open(clean_db_path, dimension=2, sync="strict") as db:
        # We need small segments or many entries to trigger rotation, 
        # but compact_segments also filters by deletion ratio.
        # Let's insert enough to have some churn.
        for i in range(1000):
            db.remember(f"Entry {i}", embedding=[0.5, 0.5])

        assert len(db) == 1000

        # Delete 300 entries to exceed the 20% threshold in segment 0 (if rotation happened)
        # Actually, let's just make sure we have enough deleted entries.
        for i in range(300):
            db.delete_memory(i + 1) # IDs start at 1

        assert len(db) == 700

        stop_threads = False
        read_errors = []
        read_count = 0

        def reader():
            nonlocal read_count
            while not stop_threads:
                try:
                    stats = db.stats()
                    # Stats are from snapshot, should be consistent
                    assert stats.entries == 700
                    read_count += 1
                except Exception as e:
                    read_errors.append(e)
                time.sleep(0.005)

        t = threading.Thread(target=reader)
        t.start()

        print("Compacting while reading...")
        time.sleep(0.05)
        # Note: compact() in facade doesn't return report, but we can verify it doesn't crash
        db.compact()
        time.sleep(0.05)

        stop_threads = True
        t.join()

        assert not read_errors, f"Errors during concurrent reads: {read_errors}"
        assert read_count > 0, "No reads executed during the test"
        print(f"Test 4 PASS: {read_count} successful reads during compaction")
