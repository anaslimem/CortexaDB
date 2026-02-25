import pytest
import os
import shutil
import time
from mnemos import Mnemos

@pytest.fixture
def clean_db_path(request):
    db_path = f"/tmp/mnemos_stress_{request.node.name}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    yield db_path
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

def test_replay_safety(clean_db_path):
    print("\n--- Test 1: Replay Safety (5000 inserts) ---")

    # Use strict sync so every write is fsynced — guarantees full replay on reopen.
    with Mnemos.open(clean_db_path, dimension=2, sync="strict") as db:
        start_time = time.time()
        for i in range(5000):
            db.remember(f"Entry {i}", embedding=[0.5, 0.5])

        print(f"Inserted 5,000 memories in {time.time() - start_time:.2f}s")
        assert len(db) == 5000

    # Simulate closing then reopen — strict policy guarantees no data loss.
    print("Reopening...")
    with Mnemos.open(clean_db_path, dimension=2, sync="strict") as db2:
        assert len(db2) == 5000, f"Expected 5000 entries after reopen, got {len(db2)}"

    print("Test 1 PASS")

def test_compaction_integrity(clean_db_path):
    print("\n--- Test 3: WAL Compaction Integrity ---")

    # Use strict sync so compact sees all 100 entries in the WAL.
    with Mnemos.open(clean_db_path, dimension=2, sync="strict") as db:
        for _ in range(100):
            db.remember("Stress entry", embedding=[0.1, 0.9])

        assert len(db) == 100

        print("Compacting...")
        db.compact()

    print("Reopening...")
    with Mnemos.open(clean_db_path, dimension=2, sync="strict") as db2:
        assert len(db2) == 100, f"Expected 100 entries after compact + reopen, got {len(db2)}"

    print("Test 3 PASS")
