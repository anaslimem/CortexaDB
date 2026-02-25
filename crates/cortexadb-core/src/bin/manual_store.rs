use cortexadb_core::core::memory_entry::{MemoryEntry, MemoryId};
use cortexadb_core::query::{QueryEmbedder, QueryOptions};
use cortexadb_core::store::CortexaDBStore;

struct DemoEmbedder;

impl QueryEmbedder for DemoEmbedder {
    fn embed(&self, query: &str) -> std::result::Result<Vec<f32>, String> {
        match query {
            "rust" => Ok(vec![1.0, 0.0, 0.0]),
            "graph" => Ok(vec![0.0, 1.0, 0.0]),
            _ => Ok(vec![0.5, 0.5, 0.0]),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::temp_dir().join("cortexadb_manual_demo");
    std::fs::create_dir_all(&base)?;
    let wal = base.join("demo.wal");
    let seg = base.join("segments");

    if wal.exists() {
        std::fs::remove_file(&wal)?;
    }
    if seg.exists() {
        std::fs::remove_dir_all(&seg)?;
    }

    let store = CortexaDBStore::new(&wal, &seg, 3)?;

    store.insert_memory(
        MemoryEntry::new(
            MemoryId(1),
            "agent1".to_string(),
            b"Rust WAL design".to_vec(),
            1000,
        )
        .with_embedding(vec![1.0, 0.0, 0.0])
        .with_importance(0.8),
    )?;

    store.insert_memory(
        MemoryEntry::new(
            MemoryId(2),
            "agent1".to_string(),
            b"Graph traversal notes".to_vec(),
            2000,
        )
        .with_embedding(vec![0.0, 1.0, 0.0])
        .with_importance(0.4),
    )?;

    store.add_edge(MemoryId(1), MemoryId(2), "relates_to".to_string())?;

    let mut options = QueryOptions::with_top_k(5);
    options.namespace = Some("agent1".to_string());

    let out = store.query("rust", options, &DemoEmbedder)?;

    println!("--- CortexaDB Manual Demo ---");
    println!("WAL length: {}", store.wal_len());
    println!("Indexed embeddings: {}", store.indexed_embeddings());
    println!("Hits: {}", out.hits.len());
    for hit in out.hits {
        println!(
            "id={:?} final={:.3} sim={:.3} imp={:.3} rec={:.3}",
            hit.id, hit.final_score, hit.similarity_score, hit.importance_score, hit.recency_score
        );
    }

    Ok(())
}
