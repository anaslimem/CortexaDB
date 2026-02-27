use cortexadb_core::engine::{CapacityPolicy, SyncPolicy};
use cortexadb_core::store::CheckpointPolicy;
use cortexadb_core::{CortexaDB, CortexaDBConfig, IndexMode};
use criterion::{Criterion, criterion_group, criterion_main};
use tempfile::tempdir;

fn bench_ingestion(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("bench.mem");

    let config = CortexaDBConfig {
        vector_dimension: 1536,
        sync_policy: SyncPolicy::Async { interval_ms: 100 },
        checkpoint_policy: CheckpointPolicy::Disabled,
        capacity_policy: CapacityPolicy::new(None, None),
        index_mode: IndexMode::Exact,
    };

    let db = CortexaDB::open_with_config(db_path.to_str().unwrap(), config).unwrap();

    let embedding: Vec<f32> = vec![0.1; 1536];

    c.bench_function("ingest_single_memory", |b| {
        b.iter(|| {
            db.remember(embedding.clone(), None).unwrap();
        })
    });
}

criterion_group!(benches, bench_ingestion);
criterion_main!(benches);
