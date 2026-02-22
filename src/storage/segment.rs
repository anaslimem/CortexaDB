use crc::Crc;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::core::memory_entry::{MemoryEntry, MemoryId};

#[derive(Error, Debug)]
pub enum SegmentError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    #[error("Entry not found: {0:?}")]
    EntryNotFound(MemoryId),
    #[error("Checksum mismatch at offset {offset}: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch {
        offset: u64,
        expected: u32,
        actual: u32,
    },
    #[error("Segment {id} not found")]
    SegmentNotFound { id: u32 },
    #[error("Index error: {0}")]
    IndexError(String),
}

pub type Result<T> = std::result::Result<T, SegmentError>;

/// Location of an entry in segment storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentLocation {
    pub segment_id: u32,
    pub offset: u64,
    pub length: u32,
}

/// Metadata for a segment entry (for tracking deletes)
#[derive(Debug, Clone)]
struct SegmentEntryMeta {
    location: SegmentLocation,
    deleted: bool,
}

/// Segment storage manager
///
/// Stores MemoryEntry objects in append-only segment files
/// Maintains in-memory index for O(1) lookups
///
/// File format:
/// [u32: entry_len][u32: checksum][N bytes: bincode(MemoryEntry)]
pub struct SegmentStorage {
    data_dir: PathBuf,
    current_segment_id: u32,
    current_segment_size: u64,
    max_segment_size: u64, // 10MB by default

    // In-memory index: MemoryId → SegmentLocation
    index: HashMap<MemoryId, SegmentEntryMeta>,

    // Current segment file handle
    current_file: Option<BufWriter<File>>,
}

impl SegmentStorage {
    /// Create new segment storage or recover existing
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();

        // Create directory if not exists
        std::fs::create_dir_all(&data_dir)?;

        let mut storage = Self {
            data_dir,
            current_segment_id: 0,
            current_segment_size: 0,
            max_segment_size: 10 * 1024 * 1024, // 10MB
            index: HashMap::new(),
            current_file: None,
        };

        // Recover existing segments and rebuild index
        storage.rebuild_index()?;

        // Open current segment for writing
        storage.open_current_segment()?;

        Ok(storage)
    }

    /// Rebuild index from existing segment files
    fn rebuild_index(&mut self) -> Result<()> {
        let dir = self.data_dir.clone();
        let mut entries = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "seg")
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        // Sort by segment ID to process in order
        entries.sort_by_key(|e| e.file_name().to_string_lossy().parse::<u32>().unwrap_or(0));

        for entry in entries {
            let path = entry.path();
            let segment_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);

            self.read_segment_into_index(&path, segment_id)?;
            self.current_segment_id = self.current_segment_id.max(segment_id + 1);
        }

        Ok(())
    }

    /// Read entire segment and rebuild index from it
    fn read_segment_into_index(&mut self, path: &Path, segment_id: u32) -> Result<()> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mut reader = BufReader::new(file);
        let mut offset = 0u64;
        let crc = Crc::<u32>::new(&crc::CRC_32_CKSUM);

        loop {
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(()) => {
                    let len = u32::from_le_bytes(len_bytes) as usize;

                    // Read checksum
                    let mut checksum_bytes = [0u8; 4];
                    reader.read_exact(&mut checksum_bytes)?;
                    let expected_checksum = u32::from_le_bytes(checksum_bytes);

                    // Read entry bytes
                    let mut entry_bytes = vec![0u8; len];
                    reader.read_exact(&mut entry_bytes)?;

                    // Verify checksum
                    let actual_checksum = crc.checksum(&entry_bytes);
                    if actual_checksum != expected_checksum {
                        return Err(SegmentError::ChecksumMismatch {
                            offset,
                            expected: expected_checksum,
                            actual: actual_checksum,
                        });
                    }

                    // Deserialize to get memory ID
                    let entry: MemoryEntry = bincode::deserialize(&entry_bytes)?;

                    // Update index
                    self.index.insert(
                        entry.id,
                        SegmentEntryMeta {
                            location: SegmentLocation {
                                segment_id,
                                offset,
                                length: len as u32,
                            },
                            deleted: false,
                        },
                    );

                    offset += 4 + 4 + len as u64; // len + checksum + data
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
        }

        Ok(())
    }

    /// Open current segment file for appending
    fn open_current_segment(&mut self) -> Result<()> {
        let path = self.segment_path(self.current_segment_id);

        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Get current file size
        self.current_segment_size = file.metadata()?.len();

        self.current_file = Some(BufWriter::new(file));

        Ok(())
    }

    /// Get path for segment file
    fn segment_path(&self, segment_id: u32) -> PathBuf {
        self.data_dir.join(format!("{:06}.seg", segment_id))
    }

    /// Check if we need to rotate to new segment
    fn should_rotate(&self) -> bool {
        self.current_segment_size >= self.max_segment_size
    }

    /// Rotate to new segment
    fn rotate_segment(&mut self) -> Result<()> {
        // Flush and close current segment
        if let Some(mut file) = self.current_file.take() {
            file.flush()?;
            file.get_mut().sync_all()?;
        }

        // Move to next segment
        self.current_segment_id += 1;
        self.current_segment_size = 0;
        self.open_current_segment()?;

        Ok(())
    }

    /// Write entry to segment storage
    pub fn write_entry(&mut self, entry: &MemoryEntry) -> Result<SegmentLocation> {
        // Serialize entry
        let entry_bytes = bincode::serialize(entry)?;
        let len = entry_bytes.len() as u32;

        // Calculate checksum
        let crc = Crc::<u32>::new(&crc::CRC_32_CKSUM);
        let checksum = crc.checksum(&entry_bytes);

        // Check if we need to rotate
        if self.should_rotate() {
            self.rotate_segment()?;
        }

        // Get current offset
        let offset = self.current_segment_size;
        let segment_id = self.current_segment_id;

        // Write to current segment
        if let Some(ref mut file) = self.current_file {
            file.write_all(&len.to_le_bytes())?;
            file.write_all(&checksum.to_le_bytes())?;
            file.write_all(&entry_bytes)?;
            file.flush()?;
        }

        // Update size tracker
        self.current_segment_size += 4 + 4 + len as u64;

        // Create location
        let location = SegmentLocation {
            segment_id,
            offset,
            length: len,
        };

        // Update index
        self.index.insert(
            entry.id,
            SegmentEntryMeta {
                location,
                deleted: false,
            },
        );

        Ok(location)
    }

    /// Read entry from segment
    pub fn read_entry(&self, id: MemoryId) -> Result<MemoryEntry> {
        let meta = self.index.get(&id).ok_or(SegmentError::EntryNotFound(id))?;

        if meta.deleted {
            return Err(SegmentError::EntryNotFound(id));
        }

        let location = meta.location;
        let path = self.segment_path(location.segment_id);

        // Open segment file
        let mut file = OpenOptions::new().read(true).open(&path)?;

        // Seek to offset
        file.seek(SeekFrom::Start(location.offset))?;

        // Read length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        let _len = u32::from_le_bytes(len_bytes);

        // Read checksum
        let mut checksum_bytes = [0u8; 4];
        file.read_exact(&mut checksum_bytes)?;
        let expected_checksum = u32::from_le_bytes(checksum_bytes);

        // Read entry bytes
        let mut entry_bytes = vec![0u8; location.length as usize];
        file.read_exact(&mut entry_bytes)?;

        // Verify checksum
        let crc = Crc::<u32>::new(&crc::CRC_32_CKSUM);
        let actual_checksum = crc.checksum(&entry_bytes);
        if actual_checksum != expected_checksum {
            return Err(SegmentError::ChecksumMismatch {
                offset: location.offset,
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }

        // Deserialize
        Ok(bincode::deserialize(&entry_bytes)?)
    }

    /// Mark entry as deleted (tombstone)
    pub fn delete_entry(&mut self, id: MemoryId) -> Result<()> {
        if let Some(meta) = self.index.get_mut(&id) {
            meta.deleted = true;
            Ok(())
        } else {
            Err(SegmentError::EntryNotFound(id))
        }
    }

    /// Get location of entry
    pub fn get_location(&self, id: MemoryId) -> Option<SegmentLocation> {
        self.index.get(&id).map(|m| m.location)
    }

    /// Check if entry exists and is not deleted
    pub fn exists(&self, id: MemoryId) -> bool {
        self.index.get(&id).map(|m| !m.deleted).unwrap_or(false)
    }

    /// Get number of live entries (not deleted)
    pub fn len(&self) -> usize {
        self.index.values().filter(|m| !m.deleted).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sync current segment to disk
    pub fn fsync(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.current_file {
            file.flush()?;
            file.get_mut().sync_all()?;
        }
        Ok(())
    }

    /// Get segments that need compaction (high deletion ratio)
    pub fn get_compactable_segments(&self) -> Vec<u32> {
        let mut segments_state: HashMap<u32, (usize, usize)> = HashMap::new();

        // Count live and deleted entries per segment
        for meta in self.index.values() {
            let entry = segments_state
                .entry(meta.location.segment_id)
                .or_insert((0, 0));
            if meta.deleted {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        }

        // Return segments with >20% deletion ratio (excluding current)
        segments_state
            .iter()
            .filter(|(seg_id, (live, deleted))| {
                **seg_id < self.current_segment_id && {
                    let total = (live + deleted) as f64;
                    *deleted as f64 / total > 0.2
                }
            })
            .map(|(seg_id, _)| *seg_id)
            .collect()
    }

    /// Get all entries (for compaction)
    pub fn get_all_live_entries(&self) -> Result<Vec<(MemoryId, MemoryEntry)>> {
        let mut entries = Vec::new();

        for (id, meta) in &self.index {
            if !meta.deleted {
                let entry = self.read_entry(*id)?;
                entries.push((*id, entry));
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_entry(id: u64) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId(id),
            "test".to_string(),
            format!("content_{}", id).into_bytes(),
            1000 + id,
        )
    }

    #[test]
    fn test_segment_write_and_read() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        let entry = create_test_entry(1);
        let location = storage.write_entry(&entry).unwrap();

        assert_eq!(location.segment_id, 0);
        assert_eq!(location.offset, 0);

        let read_entry = storage.read_entry(MemoryId(1)).unwrap();
        assert_eq!(read_entry.id, MemoryId(1));
        assert_eq!(read_entry.namespace, "test");
    }

    #[test]
    fn test_segment_multiple_writes() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        for i in 0..10 {
            let entry = create_test_entry(i);
            storage.write_entry(&entry).unwrap();
        }

        assert_eq!(storage.len(), 10);

        // Verify all can be read back
        for i in 0..10 {
            let entry = storage.read_entry(MemoryId(i)).unwrap();
            assert_eq!(entry.id, MemoryId(i));
        }
    }

    #[test]
    fn test_segment_delete_entry() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        let entry = create_test_entry(1);
        storage.write_entry(&entry).unwrap();
        assert_eq!(storage.len(), 1);

        storage.delete_entry(MemoryId(1)).unwrap();
        assert_eq!(storage.len(), 0);

        let result = storage.read_entry(MemoryId(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_segment_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path();

        // Write entries
        {
            let mut storage = SegmentStorage::new(path).unwrap();
            for i in 0..5 {
                let entry = create_test_entry(i);
                storage.write_entry(&entry).unwrap();
            }
            storage.fsync().unwrap();
        }

        // Recover and verify
        {
            let storage = SegmentStorage::new(path).unwrap();
            assert_eq!(storage.len(), 5);
            for i in 0..5 {
                let entry = storage.read_entry(MemoryId(i)).unwrap();
                assert_eq!(entry.id, MemoryId(i));
            }
        }
    }

    #[test]
    fn test_segment_checksum_validation() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path();

        // Write entry
        {
            let mut storage = SegmentStorage::new(path).unwrap();
            let entry = create_test_entry(1);
            storage.write_entry(&entry).unwrap();
            storage.fsync().unwrap();
        }

        // Corrupt the segment file
        {
            let seg_path = path.join("000000.seg");
            let mut file = OpenOptions::new().write(true).open(&seg_path).unwrap();
            file.seek(SeekFrom::Start(4)).unwrap(); // Skip length
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap(); // Bad checksum
        }

        // Try to create new storage - should fail on recovery because checksum is bad
        let result = SegmentStorage::new(path);
        assert!(result.is_err()); // Should fail on recovery due to checksum mismatch
    }

    #[test]
    fn test_segment_get_location() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        let entry = create_test_entry(1);
        let written_location = storage.write_entry(&entry).unwrap();

        let retrieved_location = storage.get_location(MemoryId(1)).unwrap();
        assert_eq!(written_location.segment_id, retrieved_location.segment_id);
        assert_eq!(written_location.offset, retrieved_location.offset);
        assert_eq!(written_location.length, retrieved_location.length);
    }

    #[test]
    fn test_segment_exists() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        let entry = create_test_entry(1);
        storage.write_entry(&entry).unwrap();

        assert!(storage.exists(MemoryId(1)));
        assert!(!storage.exists(MemoryId(999)));

        storage.delete_entry(MemoryId(1)).unwrap();
        assert!(!storage.exists(MemoryId(1)));
    }

    #[test]
    fn test_segment_get_all_live_entries() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        for i in 0..5 {
            let entry = create_test_entry(i);
            storage.write_entry(&entry).unwrap();
        }

        storage.delete_entry(MemoryId(2)).unwrap();

        let live_entries = storage.get_all_live_entries().unwrap();
        assert_eq!(live_entries.len(), 4);

        let ids: Vec<_> = live_entries.iter().map(|(id, _)| id.0).collect();
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_segment_compactable() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SegmentStorage::new(temp_dir.path()).unwrap();

        // Write entries
        for i in 0..5 {
            let entry = create_test_entry(i);
            storage.write_entry(&entry).unwrap();
        }

        // Delete some (create deletion ratio)
        storage.delete_entry(MemoryId(0)).unwrap();
        storage.delete_entry(MemoryId(1)).unwrap();

        // Check compactable (should be none since we have one segment)
        let compactable = storage.get_compactable_segments();
        assert_eq!(compactable.len(), 0); // Current segment not compacted
    }
}
