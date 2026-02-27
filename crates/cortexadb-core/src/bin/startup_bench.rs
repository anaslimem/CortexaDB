//! Startup benchmark for CortexaDB.
//!
//! Measures cold open time, snapshot load time, and WAL replay time
//! against the <100ms target for small/medium databases.

use cortexadb_core::IndexMode;
use cortexadb_core::engine::{CapacityPolicy, SyncPolicy};
use cortexadb_core::facade::{CortexaDB, CortexaDBConfig};
use cortexadb_core::store::CheckpointPolicy;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let entry_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let vector_dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    let base_dir = std::env::temp_dir().join("cortexadb_startup_bench");
    if base_dir.exists() {
        std::fs::remove_dir_all(&base_dir)?;
    }

    let db_path = base_dir.join("db");
    let db_path_str = db_path.to_str().unwrap();

    println!("=== CortexaDB Startup Benchmark ===");
    println!("entries: {entry_count}");
    println!("vector_dim: {vector_dim}");
    println!();

    // -----------------------------------------------------------------------
    // Phase 1: Seed database
    // -----------------------------------------------------------------------
    println!("Seeding database...");
    let seed_start = Instant::now();
    {
        let config = CortexaDBConfig {
            vector_dimension: vector_dim,
            sync_policy: SyncPolicy::Batch { max_ops: 512, max_delay_ms: 50 },
            checkpoint_policy: CheckpointPolicy::Disabled,
            capacity_policy: CapacityPolicy::new(None, None),
            index_mode: IndexMode::Exact,
        };
        let db = CortexaDB::open_with_config(db_path_str, config)?;

        for i in 0..entry_count {
            let embedding: Vec<f32> =
                (0..vector_dim).map(|d| ((i * 7 + d * 13) % 100) as f32 / 100.0).collect();
            db.remember(embedding, None)?;
        }
    }
    let seed_elapsed = seed_start.elapsed();
    println!("  Seeded {entry_count} entries in {:.1}ms", seed_elapsed.as_secs_f64() * 1000.0);
    println!();

    // -----------------------------------------------------------------------
    // Phase 2: Cold open (full WAL replay, no checkpoint)
    // -----------------------------------------------------------------------
    let cold_start = Instant::now();
    {
        let config = CortexaDBConfig {
            vector_dimension: vector_dim,
            sync_policy: SyncPolicy::Strict,
            checkpoint_policy: CheckpointPolicy::Disabled,
            capacity_policy: CapacityPolicy::new(None, None),
            index_mode: IndexMode::Exact,
        };
        let db = CortexaDB::open_with_config(db_path_str, config)?;
        assert_eq!(db.stats().entries, entry_count);
    }
    let cold_elapsed = cold_start.elapsed();
    let cold_ms = cold_elapsed.as_secs_f64() * 1000.0;

    // -----------------------------------------------------------------------
    // Phase 3: Create checkpoint
    // -----------------------------------------------------------------------
    println!("Creating checkpoint...");
    let ckpt_start = Instant::now();
    {
        let config = CortexaDBConfig {
            vector_dimension: vector_dim,
            sync_policy: SyncPolicy::Strict,
            checkpoint_policy: CheckpointPolicy::Disabled,
            capacity_policy: CapacityPolicy::new(None, None),
            index_mode: IndexMode::Exact,
        };
        let db = CortexaDB::open_with_config(db_path_str, config)?;
        db.checkpoint()?;
    }
    let ckpt_elapsed = ckpt_start.elapsed();
    println!("  Checkpoint created in {:.1}ms", ckpt_elapsed.as_secs_f64() * 1000.0);

    // -----------------------------------------------------------------------
    // Phase 4: Add a few entries after checkpoint (simulates ongoing usage)
    // -----------------------------------------------------------------------
    let tail_count = (entry_count / 20).max(5);
    {
        let config = CortexaDBConfig {
            vector_dimension: vector_dim,
            sync_policy: SyncPolicy::Strict,
            checkpoint_policy: CheckpointPolicy::Disabled,
            capacity_policy: CapacityPolicy::new(None, None),
            index_mode: IndexMode::Exact,
        };
        let db = CortexaDB::open_with_config(db_path_str, config)?;
        for i in 0..tail_count {
            let embedding: Vec<f32> =
                (0..vector_dim).map(|d| ((i * 11 + d * 3) % 100) as f32 / 100.0).collect();
            db.remember(embedding, None)?;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 5: Fast open (checkpoint + WAL tail)
    // -----------------------------------------------------------------------
    let fast_start = Instant::now();
    {
        let config = CortexaDBConfig {
            vector_dimension: vector_dim,
            sync_policy: SyncPolicy::Strict,
            checkpoint_policy: CheckpointPolicy::Disabled,
            capacity_policy: CapacityPolicy::new(None, None),
            index_mode: IndexMode::Exact,
        };
        let db = CortexaDB::open_with_config(db_path_str, config)?;
        assert_eq!(db.stats().entries, entry_count + tail_count);
    }
    let fast_elapsed = fast_start.elapsed();
    let fast_ms = fast_elapsed.as_secs_f64() * 1000.0;

    // -----------------------------------------------------------------------
    // Results
    // -----------------------------------------------------------------------
    println!();
    println!("┌──────────────────────────────────┬──────────┬────────┐");
    println!("│ Metric                           │ Time(ms) │ Status │");
    println!("├──────────────────────────────────┼──────────┼────────┤");
    println!(
        "│ Cold open (full WAL replay)      │ {:>8.1} │ {:>6} │",
        cold_ms,
        if cold_ms < 100.0 { "✅ PASS" } else { "⚠️  SLOW" }
    );
    println!(
        "│ Fast open (checkpoint + {} tail) │ {:>8.1} │ {:>6} │",
        tail_count,
        fast_ms,
        if fast_ms < 100.0 { "✅ PASS" } else { "⚠️  SLOW" }
    );
    println!("└──────────────────────────────────┴──────────┴────────┘");
    let speedup = if fast_ms > 0.0 { cold_ms / fast_ms } else { f64::INFINITY };
    println!();
    println!("Speedup from checkpoint: {speedup:.1}x");
    println!("Target: <100ms for small/medium DBs");

    // Cleanup
    let _ = std::fs::remove_dir_all(&base_dir);

    Ok(())
}
