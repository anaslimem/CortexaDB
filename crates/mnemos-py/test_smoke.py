from mnemos import Mnemos

def test_mnemos():
    print("Testing Mnemos PyO3 bindings...")
    
    # 1. Open with dimension 3
    db = Mnemos.open("/tmp/mnemos_test_py", dimension=3)
    
    # 2. Store memory
    mid = db.remember_embedding([1.0, 0.0, 0.0])
    print(f"Stored memory: {mid}")
    
    # 3. Ask
    hits = db.ask_embedding([1.0, 0.0, 0.0])
    print(f"Found {len(hits)} hits, top score={hits[0].score:.3f}")
    assert hits[0].id == mid
    
    # 4. Get full memory
    mem = db.get(mid)
    print(f"Memory: {mem}")
    assert mem.namespace == "default"
    assert mem.id == mid

    # 5. Connect
    mid2 = db.remember_embedding([0.0, 1.0, 0.0])
    db.connect(mid, mid2, "related")
    print(f"Connected {mid} -> {mid2}")

    # 6. Stats
    stats = db.stats()
    print(f"Stats: {stats}")
    assert stats.vector_dimension == 3
    assert stats.entries == 2

    # 7. Stress - error handling
    try:
        db.remember_embedding([1.0, 0.0]) # Wrong dimension
        assert False, "Should have raised an error"
    except Exception as e:
        print(f"Caught expected error: {e}")

    try:
        db2 = Mnemos.open("/tmp/mnemos_test_py", dimension=4) # Dimension mismatch on open
        assert False, "Should have raised an error"
    except Exception as e:
        print(f"Caught expected error: {e}")

    # 8. Compact
    db.compact()
    print("Compacted successfully")

    # 9. Checkpoint
    db.checkpoint()
    print("Checkpointed successfully")

    print("All tests passed!")

if __name__ == "__main__":
    import shutil
    import os
    if os.path.exists("/tmp/mnemos_test_py"):
        shutil.rmtree("/tmp/mnemos_test_py")
    test_mnemos()
