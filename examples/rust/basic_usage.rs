//! CortexaDB Rust Example - Basic Usage
//!
//! Demonstrates core features:
//! - Opening a database
//! - Storing memories
//! - Using chunking strategies
//! - Hybrid search
//! - Graph relationships

use cortexadb_core::{ChunkingStrategy, CortexaDB, CortexaDBConfig, chunk};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = "example_rust.mem";

    // Cleanup old database
    for ext in ["", ".wal", ".checkpoint"] {
        let path = format!("{}{}", db_path, ext);
        if Path::new(&path).exists() {
            std::fs::remove_file(&path)?;
        }
    }

    println!("=== CortexaDB Rust Example ===\n");

    // -----------------------------------------------------------
    // 1. Configure and Open Database
    // -----------------------------------------------------------
    let config = CortexaDBConfig { vector_dimension: 128, ..Default::default() };

    let db = CortexaDB::open(db_path, config)?;
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
    // 5. Store Memories
    // -----------------------------------------------------------
    println!("\n[4] Storing memories...");

    let id1 = db.remember("The user lives in Paris and loves programming.", None, None)?;

    let id2 = db.remember("CortexaDB is a vector database for AI agents.", None, None)?;

    let id3 = db.remember("Rust is a systems programming language.", None, None)?;

    println!("   Stored 3 memories: IDs {}, {}, {}", id1, id2, id3);

    // -----------------------------------------------------------
    // 6. Graph Relationships
    // -----------------------------------------------------------
    println!("\n[5] Creating graph connections...");

    db.connect(id1, id2, "uses", None)?;
    db.connect(id2, id3, "written_in", None)?;

    println!("   Connected: {} -> {} -> {}", id1, id2, id3);

    // -----------------------------------------------------------
    // 7. Stats
    // -----------------------------------------------------------
    println!("\n[6] Database stats...");
    let stats = db.stats()?;
    println!("   Entries: {}", stats.entries);
    println!("   Indexed embeddings: {}", stats.indexed_embeddings);

    println!("\n=== Example Complete! ===");

    // Cleanup
    for ext in ["", ".wal", ".checkpoint"] {
        let path = format!("{}{}", db_path, ext);
        if Path::new(&path).exists() {
            std::fs::remove_file(&path)?;
        }
    }

    Ok(())
}
