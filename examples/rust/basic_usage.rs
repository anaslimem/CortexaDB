use cortexadb_core::{CortexaDB, CortexaDBConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = "example_rust.mem";
    
    // Cleanup
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }
    
    println!("--- Initializing CortexaDB Rust Core ---");
    
    // 1. Configure and Open
    let config = CortexaDBConfig {
        dimension: 1536,
        ..Default::default()
    };
    
    let db = CortexaDB::open(db_path, config)?;
    
    // 2. Remember
    println!("\n[1] Storing memories...");
    let id1 = db.remember(
        "Rust is a systems programming language focused on safety.",
        None, // Metadata
        None  // Namespace
    )?;
    
    let id2 = db.remember(
        "CortexaDB is written in Rust for high performance.",
        None,
        None
    )?;
    
    // 3. Connect
    println!("\n[2] Connecting nodes in the graph...");
    db.connect(id1, id2, "powers", None)?;
    
    // 4. Query
    println!("\n[3] Querying state...");
    let stats = db.stats()?;
    println!("Database size: {} entries", stats.entries);
    
    println!("\n--- Rust Example Complete ---");
    
    Ok(())
}
