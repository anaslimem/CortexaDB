use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cortexadb_core::core::memory_entry::{MemoryEntry, MemoryId};
use cortexadb_core::engine::SyncPolicy;
use cortexadb_core::store::CortexaDBStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = BenchConfig::from_args(std::env::args().skip(1).collect())?;

    let (wal, seg) = build_paths(&cfg);
    std::fs::create_dir_all(seg.parent().unwrap_or_else(|| std::path::Path::new(".")))?;

    if wal.exists() {
        std::fs::remove_file(&wal)?;
    }
    if seg.exists() {
        std::fs::remove_dir_all(&seg)?;
    }

    let store = CortexaDBStore::new_with_policy(&wal, &seg, cfg.vector_dim, cfg.policy)?;

    let start = Instant::now();
    for i in 0..cfg.ops {
        let entry = MemoryEntry::new(
            MemoryId(i),
            cfg.namespace.clone(),
            format!("bench_mem_{}", i).into_bytes(),
            1_700_000_000 + i,
        )
        .with_embedding(vec![1.0, 0.0, 0.0])
        .with_importance(0.5);

        store.insert_memory(entry)?;
    }
    let ingest_elapsed = start.elapsed();

    let flush_start = Instant::now();
    store.flush()?;
    let flush_elapsed = flush_start.elapsed();

    let total_elapsed = start.elapsed();
    let ops_per_sec = cfg.ops as f64 / total_elapsed.as_secs_f64().max(1e-9);

    drop(store);

    let recovered =
        CortexaDBStore::recover_with_policy(&wal, &seg, cfg.vector_dim, SyncPolicy::Strict)?;
    let recovered_len = recovered.state_machine().len() as u64;

    println!("=== CortexaDB Sync Benchmark ===");
    println!("mode: {:?}", cfg.policy);
    println!("ops: {}", cfg.ops);
    println!("data_dir: {}", cfg.data_dir.display());
    println!("ingest_ms: {}", ingest_elapsed.as_millis());
    println!("flush_ms: {}", flush_elapsed.as_millis());
    println!("total_ms: {}", total_elapsed.as_millis());
    println!("ops_per_sec: {:.2}", ops_per_sec);
    println!("wal_len: {}", recovered.wal_len());
    println!("recovered_entries: {}", recovered_len);

    if recovered_len != cfg.ops {
        return Err(format!(
            "durability check failed: expected {} recovered {}, wal_len {}",
            cfg.ops,
            recovered_len,
            recovered.wal_len()
        )
        .into());
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct BenchConfig {
    ops: u64,
    vector_dim: usize,
    namespace: String,
    data_dir: PathBuf,
    policy: SyncPolicy,
}

impl BenchConfig {
    fn from_args(args: Vec<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut ops: u64 = 20_000;
        let mut vector_dim: usize = 3;
        let mut namespace = "bench".to_string();
        let mut data_dir = std::env::temp_dir().join("cortexadb_sync_bench");

        let mut mode = "strict".to_string();
        let mut batch_max_ops: usize = 64;
        let mut batch_max_delay_ms: u64 = 25;
        let mut async_interval_ms: u64 = 25;

        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--ops" => {
                    i += 1;
                    ops = args.get(i).ok_or("missing value for --ops")?.parse()?;
                }
                "--vector-dim" => {
                    i += 1;
                    vector_dim = args
                        .get(i)
                        .ok_or("missing value for --vector-dim")?
                        .parse()?;
                }
                "--namespace" => {
                    i += 1;
                    namespace = args.get(i).ok_or("missing value for --namespace")?.clone();
                }
                "--data-dir" => {
                    i += 1;
                    data_dir = PathBuf::from(args.get(i).ok_or("missing value for --data-dir")?);
                }
                "--mode" => {
                    i += 1;
                    mode = args.get(i).ok_or("missing value for --mode")?.clone();
                }
                "--batch-max-ops" => {
                    i += 1;
                    batch_max_ops = args
                        .get(i)
                        .ok_or("missing value for --batch-max-ops")?
                        .parse()?;
                }
                "--batch-max-delay-ms" => {
                    i += 1;
                    batch_max_delay_ms = args
                        .get(i)
                        .ok_or("missing value for --batch-max-delay-ms")?
                        .parse()?;
                }
                "--async-interval-ms" => {
                    i += 1;
                    async_interval_ms = args
                        .get(i)
                        .ok_or("missing value for --async-interval-ms")?
                        .parse()?;
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(format!("unknown arg: {}", other).into());
                }
            }
            i += 1;
        }

        let policy = match mode.to_ascii_lowercase().as_str() {
            "strict" => SyncPolicy::Strict,
            "batch" => SyncPolicy::Batch {
                max_ops: batch_max_ops,
                max_delay_ms: batch_max_delay_ms,
            },
            "async" => SyncPolicy::Async {
                interval_ms: async_interval_ms,
            },
            _ => return Err(format!("invalid mode: {} (use strict|batch|async)", mode).into()),
        };

        Ok(Self {
            ops,
            vector_dim,
            namespace,
            data_dir,
            policy,
        })
    }
}

fn build_paths(cfg: &BenchConfig) -> (PathBuf, PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let run_dir = cfg.data_dir.join(format!("run_{}", nonce));
    let wal = run_dir.join("bench.wal");
    let seg = run_dir.join("segments");
    (wal, seg)
}

fn print_help() {
    println!(
        "Usage: cargo run --bin sync_bench -- [options]\n\
         --mode strict|batch|async\n\
         --ops <u64> (default: 20000)\n\
         --vector-dim <usize> (default: 3)\n\
         --namespace <string> (default: bench)\n\
         --data-dir <path> (default: /tmp/cortexadb_sync_bench)\n\
         --batch-max-ops <usize> (default: 64)\n\
         --batch-max-delay-ms <u64> (default: 25)\n\
         --async-interval-ms <u64> (default: 25)"
    );
}
