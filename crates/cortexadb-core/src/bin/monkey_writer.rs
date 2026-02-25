use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use cortexadb_core::core::memory_entry::{MemoryEntry, MemoryId};
use cortexadb_core::store::CortexaDBStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cortexadb_monkey"));
    let total: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let sleep_ms: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(2);

    std::fs::create_dir_all(&data_dir)?;
    let wal = data_dir.join("monkey.wal");
    let seg = data_dir.join("segments");

    let store = if wal.exists() {
        CortexaDBStore::recover(&wal, &seg, 3)?
    } else {
        CortexaDBStore::new(&wal, &seg, 3)?
    };

    for i in 0..total {
        let entry = MemoryEntry::new(
            MemoryId(i),
            "agent1".to_string(),
            format!("memory_{i}").into_bytes(),
            1_700_000_000 + i,
        )
        .with_embedding(vec![1.0, 0.0, 0.0]);

        let _ = store.insert_memory(entry)?;

        if i % 100 == 0 {
            eprintln!("wrote {i}/{total}");
        }

        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    eprintln!("writer completed");
    Ok(())
}
