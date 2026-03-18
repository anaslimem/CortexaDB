//! CortexaDB Rust Example - Basic Usage
//!
//! Demonstrates core features:
//! - Opening a database
//! - Storing memories
//! - Using chunking strategies
//! - Hybrid search
//! - Graph relationships
//! - Collection-scoped operations

use cortexadb_core::{chunk, ChunkingStrategy, CortexaDB};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

fn embed_text(text: &str, dimension: usize) -> Vec<f32> {
    let mut vec = vec![0.0_f32; dimension];

    for (i, token) in text.split_whitespace().enumerate() {
        let mut hasher = DefaultHasher::new();
        token.to_lowercase().hash(&mut hasher);
        let hash = hasher.finish() as usize;
        let idx = hash % dimension;
        let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign;
        vec[(idx + i) % dimension] += 0.25 * sign;
    }

    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vec {
            *value /= norm;
        }
    }

    vec
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = "example_rust.db";
    let dimension = 64;

    // Cleanup old database directory
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    println!("=== CortexaDB Rust Example ===\n");

    // -----------------------------------------------------------
    // 1. Open Database (current API)
    // -----------------------------------------------------------
    let db = CortexaDB::open(db_path, dimension)?;
    println!("Opened database: {} entries", db.stats()?.entries);

    // -----------------------------------------------------------
    // 2. Chunking Strategies (5 available)
    // -----------------------------------------------------------
    println!("\n[1] Using chunking strategies...");

    let text = r#"First paragraph with some content here.
    
Second paragraph with more details.
    
Third paragraph to complete the example."#;

    // Recursive (default for RAG)
    let chunks = chunk(text, ChunkingStrategy::Recursive { chunk_size: 50, overlap: 5 });
    println!("   Recursive: {} chunks", chunks.len());

    // Semantic (split by paragraphs)
    let chunks = chunk(text, ChunkingStrategy::Semantic { overlap: 2 });
    println!("   Semantic: {} chunks", chunks.len());

    // Fixed (character-based)
    let chunks = chunk(text, ChunkingStrategy::Fixed { chunk_size: 30, overlap: 3 });
    println!("   Fixed: {} chunks", chunks.len());

    // -----------------------------------------------------------
    // 3. Markdown Chunking
    // -----------------------------------------------------------
    println!("\n[2] Markdown chunking...");

    let md_text = r#"# Heading 1

Content under heading 1.

## Heading 2

Content under heading 2.

### Heading 3

Content under heading 3.
"#;

    let chunks = chunk(md_text, ChunkingStrategy::Markdown { preserve_headers: true, overlap: 1 });
    println!("   Markdown: {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().take(2).enumerate() {
        println!("      Chunk {}: {}...", i, &chunk.text[..chunk.text.len().min(30)]);
    }

    // -----------------------------------------------------------
    // 4. JSON Chunking
    // -----------------------------------------------------------
    println!("\n[3] JSON chunking...");

    let json_text = r#"{"user": {"name": "John", "age": 30}, "city": "Paris"}"#;

    let chunks = chunk(json_text, ChunkingStrategy::Json { overlap: 0 });
    println!("   JSON: {} key-value pairs", chunks.len());
    for chunk in &chunks {
        if let Some(ref meta) = chunk.metadata {
            println!("      {} = {}", meta.key.as_ref().unwrap(), meta.value.as_ref().unwrap());
        }
    }

    // -----------------------------------------------------------
    // 5. High-Performance Batch Storage
    // -----------------------------------------------------------
    println!("\n[4] Storing memories (Batch mode)...");

    let text1 = "The user lives in Paris and loves programming.";
    let text2 = "CortexaDB is a vector database for AI agents.";
    let text3 = "Rust is a systems programming language.";

    use cortexadb_core::BatchRecord;

    let records = vec![
        BatchRecord {
            collection: "default".to_string(),
            content: text1.as_bytes().to_vec(),
            embedding: Some(embed_text(text1, dimension)),
            metadata: None,
        },
        BatchRecord {
            collection: "default".to_string(),
            content: text2.as_bytes().to_vec(),
            embedding: Some(embed_text(text2, dimension)),
            metadata: None,
        },
        BatchRecord {
            collection: "default".to_string(),
            content: text3.as_bytes().to_vec(),
            embedding: Some(embed_text(text3, dimension)),
            metadata: None,
        },
    ];

    // Bulk insert with 100x speedup
    let last_id = db.add_batch(records)?;
    println!("   Batch finished. Last inserted ID: {}", last_id.last().unwrap());
    
    // For manual IDs in the example, we'll use 1, 2, 3 assuming clean start
    let id1 = 1; let id2 = 2; let id3 = 3;

    // -----------------------------------------------------------
    // 6. Graph Relationships
    // -----------------------------------------------------------
    println!("\n[5] Creating graph connections...");

    db.connect(id1, id2, "uses")?;
    db.connect(id2, id3, "written_in")?;

    println!("   Connected: {} -> {} -> {}", id1, id2, id3);

    // -----------------------------------------------------------
    // 7. Search (query embedding -> top-k results)
    // -----------------------------------------------------------
    println!("\n[6] Querying memories...");
    let query = "Where does the user live?";
    let hits = db.search(embed_text(query, dimension), 3, None)?;
    for hit in hits {
        let mem = db.get_memory(hit.id)?;
        let content = String::from_utf8_lossy(&mem.content);
        println!("   - ID: {}, Score: {:.4}, Content: {}", hit.id, hit.score, content);
    }

    // -----------------------------------------------------------
    // 8. Collection-scoped retrieval
    // -----------------------------------------------------------
    println!("\n[7] Collections...");
    let travel_text = "Flight to Tokyo booked for June.";
    let col_id = db.add_with_content(
        "travel_agent",
        travel_text.as_bytes().to_vec(),
        embed_text(travel_text, dimension),
        None,
    )?;
    println!("   Stored in collection 'travel_agent': ID {}", col_id);
    let col_hits = db.search_in_collection(
        "travel_agent",
        embed_text("Tokyo travel plans", dimension),
        5,
        None,
    )?;
    println!("   travel_agent hits: {}", col_hits.len());

    // -----------------------------------------------------------
    // 9. Stats
    // -----------------------------------------------------------
    println!("\n[8] Database stats...");
    let stats = db.stats()?;
    println!("   Entries: {}", stats.entries);
    println!("   Indexed embeddings: {}", stats.indexed_embeddings);

    println!("\n=== Example Complete! ===");

    // Cleanup
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    Ok(())
}
