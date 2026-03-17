//! Integration tests for CortexaDB.
//!
//! These tests exercise the full stack: open → add → search → checkpoint → recover.
//! Unlike the unit tests in `src/`, these tests run against actual disk files (via tempdir).

use cortexadb_core::{CortexaDB, CortexaDBConfig};
use serial_test::serial;
use tempfile::TempDir;

fn open_db(path: &std::path::Path) -> CortexaDB {
    let config = CortexaDBConfig {
        vector_dimension: 3,
        sync_policy: cortexadb_core::engine::SyncPolicy::Strict,
        checkpoint_policy: cortexadb_core::store::CheckpointPolicy::Disabled,
        capacity_policy: cortexadb_core::engine::CapacityPolicy::new(None, None),
        index_mode: cortexadb_core::index::IndexMode::Exact,
    };
    CortexaDB::open_with_config(path.to_str().unwrap(), config).unwrap()
}

fn open_db_with_config(dir: &TempDir, config: CortexaDBConfig) -> CortexaDB {
    let path = dir.path().join("db");
    CortexaDB::open_with_config(path.to_str().unwrap(), config).unwrap()
}

// ---------------------------------------------------------------------------
// Basic open → add → search → recover
// ---------------------------------------------------------------------------

#[test]
fn test_full_open_add_search() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let db = open_db(&path);

    let id1 = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
    let id2 = db.add(vec![0.0, 1.0, 0.0], None).unwrap();

    let hits = db.search(vec![1.0, 0.0, 0.0], 5, None).unwrap();
    assert!(!hits.is_empty(), "search should return results");
    assert_eq!(hits[0].id, id1, "top hit should be id1 (exact match)");

    let hits2 = db.search(vec![0.0, 1.0, 0.0], 5, None).unwrap();
    assert_eq!(hits2[0].id, id2, "top hit for second query should be id2");
}

#[test]
fn test_recover_after_drop_restores_entries() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let expected_ids: Vec<u64>;
    {
        let db = open_db(&path);
        let id1 = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        let id2 = db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        let id3 = db.add(vec![0.0, 0.0, 1.0], None).unwrap();
        expected_ids = vec![id1, id2, id3];
        // db dropped here (simulates process exit without explicit flush)
    }

    // Reopen: should recover from WAL
    let db = open_db(&path);
    assert_eq!(db.stats().unwrap().entries, 3, "all entries must survive reopen");
    assert_eq!(db.stats().unwrap().indexed_embeddings, 3);

    for id in &expected_ids {
        db.get_memory(*id).unwrap_or_else(|_| panic!("memory {} must survive recovery", id));
    }
}

#[test]
fn test_recover_search_returns_correct_top_hit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let id_target: u64;
    {
        let db = open_db(&path);
        db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        id_target = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.add(vec![0.0, 0.0, 1.0], None).unwrap();
    }

    let db = open_db(&path);
    let hits = db.search(vec![1.0, 0.0, 0.0], 1, None).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id_target, "top hit after recovery must be the matching entry");
}

// ---------------------------------------------------------------------------
// Checkpoint + recovery
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn test_checkpoint_recovery_preserves_all_entries() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let mut all_ids: Vec<u64> = Vec::new();
    {
        let db = open_db(&path);
        all_ids.push(db.add(vec![1.0, 0.0, 0.0], None).unwrap());
        all_ids.push(db.add(vec![0.0, 1.0, 0.0], None).unwrap());
        db.flush().unwrap(); // ensure WAL is synced before checkpoint
        db.checkpoint().unwrap();
        // Write one more entry AFTER the checkpoint.
        all_ids.push(db.add(vec![0.0, 0.0, 1.0], None).unwrap());
    }

    let db = open_db(&path);
    assert_eq!(db.stats().unwrap().entries, 3, "all 3 entries must survive checkpoint+recovery");
    for id in &all_ids {
        db.get_memory(*id)
            .unwrap_or_else(|_| panic!("memory {} missing after checkpoint recovery", id));
    }
}

#[test]
#[serial]
fn test_double_checkpoint_recovery() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    {
        let db = open_db(&path);
        db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.flush().unwrap();
        db.checkpoint().unwrap();
        db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        db.flush().unwrap();
        db.checkpoint().unwrap(); // second checkpoint
        db.add(vec![0.0, 0.0, 1.0], None).unwrap();
    }

    let db = open_db(&path);
    assert_eq!(db.stats().unwrap().entries, 3, "all entries must survive double checkpoint");
}

// ---------------------------------------------------------------------------
// Delete persistence
// ---------------------------------------------------------------------------

#[test]
fn test_delete_persists_across_recovery() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let deleted_id: u64;
    let kept_id: u64;
    {
        let db = open_db(&path);
        deleted_id = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        kept_id = db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        db.delete(deleted_id).unwrap();
        assert_eq!(db.stats().unwrap().entries, 1);
    }

    let db = open_db(&path);
    assert_eq!(db.stats().unwrap().entries, 1, "deletion must persist across recovery");
    assert!(db.get_memory(deleted_id).is_err(), "deleted entry must not be recoverable");
    assert!(db.get_memory(kept_id).is_ok(), "non-deleted entry must survive");
}

#[test]
#[serial]
fn test_delete_then_checkpoint_recovery() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let deleted_id: u64;
    {
        let db = open_db(&path);
        deleted_id = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        db.delete(deleted_id).unwrap();
        db.flush().unwrap(); // ensure WAL is synced before checkpoint
        db.checkpoint().unwrap();
    }

    let db = open_db(&path);
    assert_eq!(db.stats().unwrap().entries, 1);
    assert!(db.get_memory(deleted_id).is_err(), "deleted entry must not survive checkpoint");
}

// ---------------------------------------------------------------------------
// Graph relationship persistence
// ---------------------------------------------------------------------------

#[test]
fn test_graph_edges_persist_across_recovery() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let (id1, id2): (u64, u64);
    {
        let db = open_db(&path);
        id1 = db.add(vec![1.0, 0.0, 0.0], None).unwrap();
        id2 = db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        db.connect(id1, id2, "relates_to").unwrap();
    }

    let db = open_db(&path);
    let neighbors = db.get_neighbors(id1).unwrap();
    assert_eq!(neighbors.len(), 1, "edge must persist across recovery");
    assert_eq!(neighbors[0].0, id2);
    assert_eq!(neighbors[0].1, "relates_to");
}

// ---------------------------------------------------------------------------
// Collection isolation
// ---------------------------------------------------------------------------

#[test]
fn test_collection_isolation_persists() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let id_a: u64;
    let id_b: u64;
    {
        let db = open_db(&path);
        id_a = db.add_in_collection("agent_a", vec![1.0, 0.0, 0.0], None).unwrap();
        id_b = db.add_in_collection("agent_b", vec![1.0, 0.0, 0.0], None).unwrap();
    }

    let db = open_db(&path);
    assert_eq!(db.get_memory(id_a).unwrap().collection, "agent_a");
    assert_eq!(db.get_memory(id_b).unwrap().collection, "agent_b");
}

// ---------------------------------------------------------------------------
// Metadata persistence
// ---------------------------------------------------------------------------

#[test]
fn test_metadata_persists_across_recovery() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");

    let id: u64;
    {
        let db = open_db(&path);
        let mut meta = std::collections::HashMap::new();
        meta.insert("source".to_string(), "unit_test".to_string());
        meta.insert("priority".to_string(), "high".to_string());
        id = db.add(vec![1.0, 0.0, 0.0], Some(meta)).unwrap();
    }

    let db = open_db(&path);
    let memory = db.get_memory(id).unwrap();
    assert_eq!(memory.metadata.get("source").map(|s| s.as_str()), Some("unit_test"));
    assert_eq!(memory.metadata.get("priority").map(|s| s.as_str()), Some("high"));
}

// ---------------------------------------------------------------------------
// Capacity / eviction
// ---------------------------------------------------------------------------

#[test]
fn test_capacity_eviction_keeps_max_entries() {
    use cortexadb_core::{
        engine::{CapacityPolicy, SyncPolicy},
        index::IndexMode,
        store::CheckpointPolicy,
    };

    let dir = TempDir::new().unwrap();
    let config = CortexaDBConfig {
        vector_dimension: 3,
        sync_policy: SyncPolicy::Strict,
        checkpoint_policy: CheckpointPolicy::Disabled,
        capacity_policy: CapacityPolicy::new(Some(2), None),
        index_mode: IndexMode::Exact,
    };
    let db = open_db_with_config(&dir, config);

    db.add(vec![1.0, 0.0, 0.0], None).unwrap();
    db.add(vec![0.0, 1.0, 0.0], None).unwrap();
    db.add(vec![0.0, 0.0, 1.0], None).unwrap();

    // After inserting 3 entries with max_entries=2, one should have been evicted.
    assert_eq!(db.stats().unwrap().entries, 2, "max_entries=2 must evict oldest entry");
}

// ---------------------------------------------------------------------------
// HNSW Recovery sync tests
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_recovery_sync() {
    use cortexadb_core::{
        engine::SyncPolicy,
        index::{
            hnsw::{HnswConfig, MetricKind},
            IndexMode,
        },
        store::CheckpointPolicy,
    };

    let dir = TempDir::new().unwrap();
    let config = CortexaDBConfig {
        vector_dimension: 3,
        sync_policy: SyncPolicy::Strict,
        checkpoint_policy: CheckpointPolicy::Disabled,
        capacity_policy: cortexadb_core::engine::CapacityPolicy::new(None, None),
        index_mode: IndexMode::Hnsw(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: MetricKind::Cos,
        }),
    };

    let id_target: u64;
    let id_deleted: u64;
    {
        let db = open_db_with_config(&dir, config.clone());
        db.add(vec![0.0, 1.0, 0.0], None).unwrap();
        id_deleted = db.add(vec![0.0, 0.0, 1.0], None).unwrap();

        // Checkpoint saves the HNSW index to disk right now (with these 2 items).
        db.checkpoint().unwrap();

        // Insert a new item AFTER the checkpoint but BEFORE the crash
        id_target = db.add(vec![1.0, 0.0, 0.0], None).unwrap();

        // Delete an item that WAS saved in the HNSW on disk, AFTER the checkpoint
        db.delete(id_deleted).unwrap();

        // The process crashes/drops here. HNSW index on disk is STALE.
    }

    // Recover from the directory.
    let db = open_db_with_config(&dir, config);

    // Assert the deleted memory is truly gone
    assert!(db.get_memory(id_deleted).is_err(), "deleted entry shouldn't be recovered");

    // Assert the uncheckpointed insertion survived
    assert!(db.get_memory(id_target).is_ok(), "uncheckpointed entry must survive");

    // Perform an HNSW search to ensure the vector index was properly synced during recovery
    let hits = db.search(vec![1.0, 0.0, 0.0], 5, None).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].id, id_target, "top hit should be the post-checkpoint entry");
}
