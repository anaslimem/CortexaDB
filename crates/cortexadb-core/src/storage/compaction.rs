use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use super::segment::SegmentStorage;

#[derive(Error, Debug)]
pub enum CompactionError {
    #[error("Segment error: {0}")]
    Segment(#[from] crate::storage::segment::SegmentError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CompactionError>;

#[derive(Debug, Clone)]
pub struct CompactionReport {
    pub compacted_segments: Vec<u32>,
    pub live_entries_rewritten: usize,
}

/// Compact segment files via atomic directory swap:
/// 1) read live entries from existing segment dir
/// 2) rewrite them into a fresh temp dir
/// 3) rename old dir -> backup, temp dir -> original, delete backup
pub fn compact_segment_dir<P: AsRef<Path>>(data_dir: P) -> Result<CompactionReport> {
    let data_dir = data_dir.as_ref();
    let storage = SegmentStorage::new(data_dir)?;
    let compacted_segments = storage.get_compactable_segments();
    let mut live_entries = storage.get_all_live_entries()?;
    drop(storage);

    if compacted_segments.is_empty() {
        return Ok(CompactionReport { compacted_segments, live_entries_rewritten: 0 });
    }

    // Deterministic rewrite order.
    live_entries.sort_by_key(|(id, _)| *id);

    let parent = data_dir.parent().unwrap_or_else(|| Path::new("."));
    let data_name = data_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "segments".to_string());
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);

    let tmp_dir = parent.join(format!("{}.compact.{}", data_name, nonce));
    let backup_dir = parent.join(format!("{}.backup.{}", data_name, nonce));

    let mut new_storage = SegmentStorage::new(&tmp_dir)?;
    for (_, entry) in live_entries.iter() {
        new_storage.write_entry(entry)?;
    }
    new_storage.fsync()?;
    drop(new_storage);

    // Atomic swap sequence.
    std::fs::rename(data_dir, &backup_dir)?;
    std::fs::rename(&tmp_dir, data_dir)?;
    std::fs::remove_dir_all(&backup_dir)?;

    Ok(CompactionReport { compacted_segments, live_entries_rewritten: live_entries.len() })
}

pub fn temp_compaction_paths(base: &Path, suffix: &str) -> (PathBuf, PathBuf) {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let data_name = base
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "segments".to_string());
    (
        parent.join(format!("{}.compact.{}", data_name, suffix)),
        parent.join(format!("{}.backup.{}", data_name, suffix)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryEntry, MemoryId};
    use tempfile::TempDir;

    #[test]
    fn test_compact_segment_dir_noop_when_not_compactable() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().join("segments");
        let mut storage = SegmentStorage::new(&dir).unwrap();
        for i in 0..3 {
            storage
                .write_entry(&MemoryEntry::new(
                    MemoryId(i),
                    "ns".to_string(),
                    b"x".to_vec(),
                    1000 + i,
                ))
                .unwrap();
        }
        storage.fsync().unwrap();
        drop(storage);

        let report = compact_segment_dir(&dir).unwrap();
        assert_eq!(report.live_entries_rewritten, 0);
    }
}
