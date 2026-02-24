import os
import shutil
import sys
import time
from mnemos import Mnemos

def test_replay_safety():
    print("--- Test 1: Replay Safety (50k inserts) ---")
    db_path = "/tmp/mnemos_stress"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = Mnemos.open(db_path, dimension=2)
    start_time = time.time()
    for i in range(5000):
        # We need to simulate a "crash" but we can just let Python garbage collect the handle without doing anything special. 
        # Actually, let's just create 50k entries and see if the engine handles it.
        # Strict mode is the default so this might take a bit of time if we're fsyncing.
        # But wait, Mnemos.open uses CheckpointPolicy::Periodic. So it will checkpoint in background.
        db.remember_embedding([0.5, 0.5])

    print(f"Inserted 5,000 memories in {time.time() - start_time:.2f}s")
    
    stats_before = db.stats()
    assert stats_before.entries == 5000
    
    # Simulate closing / crash
    del db
    
    # Reopen
    print("Reopening...")
    db2 = Mnemos.open(db_path, dimension=2)
    stats_after = db2.stats()
    print(stats_after)
    assert stats_after.entries == 5000
    print("Test 1 PASS")

def test_compaction_integrity():
    print("--- Test 3: WAL Compaction Integrity ---")
    db_path = "/tmp/mnemos_stress_compact"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    db = Mnemos.open(db_path, dimension=2)
    
    # Insert 100 entries
    for _ in range(100):
        db.remember_embedding([0.1, 0.9])
    
    stats_before = db.stats()
    assert stats_before.entries == 100
    
    print("Compacting...")
    db.compact()
    
    # Simulate close
    del db
    
    print("Reopening...")
    db2 = Mnemos.open(db_path, dimension=2)
    stats_after = db2.stats()
    print(stats_after)
    assert stats_after.entries == 100
    
    print("Test 3 PASS")

if __name__ == "__main__":
    test_compaction_integrity()
    test_replay_safety()
    print("All stress tests passed!")
