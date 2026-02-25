use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cortexadb_core::{CortexaDB, CortexaDBConfig};
use tempfile::tempdir;

fn bench_ingestion(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("bench.mem");
    
    let config = CortexaDBConfig {
        dimension: 1536,
        sync: "async".to_string(), // Fastest for benchmarking
        ..Default::default()
    };
    
    let db = CortexaDB::open(db_path.to_str().unwrap(), config).unwrap();
    
    c.bench_function("ingest_single_memory", |b| {
        b.iter(|| {
            db.remember(
                black_box("The quick brown fox jumps over the lazy dog."),
                None,
                None
            ).unwrap();
        })
    });
}

criterion_group!(benches, bench_ingestion);
criterion_main!(benches);
