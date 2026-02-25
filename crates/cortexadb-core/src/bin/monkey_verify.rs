use std::path::PathBuf;

use cortexadb_core::core::memory_entry::MemoryId;
use cortexadb_core::store::CortexaDBStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cortexadb_monkey"));

    let wal = data_dir.join("monkey.wal");
    let seg = data_dir.join("segments");

    let store = CortexaDBStore::recover(&wal, &seg, 3)?;
    let state = store.state_machine();

    let entries = state.len() as u64;
    let wal_len = store.wal_len();

    // With insert-only workload, recovered state size must match surviving WAL commands.
    if entries != wal_len {
        return Err(format!(
            "recovery mismatch: state_entries={} wal_len={}",
            entries, wal_len
        )
        .into());
    }

    // IDs should be contiguous from 0..entries-1 for this controlled writer.
    for id in 0..entries {
        state
            .get_memory(MemoryId(id))
            .map_err(|_| format!("missing recovered id {id}"))?;
    }

    println!(
        "Monkey recovery OK: recovered {} valid records (WAL len {}).",
        entries, wal_len
    );

    Ok(())
}
