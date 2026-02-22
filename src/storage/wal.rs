use std::fs::{File, OpenOptions};
use std::io::{Read, Write, BufWriter, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;
use crc::Crc;

use crate::core::command::Command;

#[derive(Error, Debug)]
pub enum WalError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    #[error("Checksum mismatch at entry {entry_id}: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch {
        entry_id: u64,
        expected: u32,
        actual: u32,
    },
    #[error("Invalid entry format at position {position}")]
    InvalidFormat { position: u64 },
}

pub type Result<T> = std::result::Result<T, WalError>;

/// Unique identifier for a command in the WAL
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommandId(pub u64);

impl CommandId {
    pub fn next(&self) -> Self {
        CommandId(self.0 + 1)
    }
}

/// Write-Ahead Log for durability and recovery
/// 
/// Format:
/// [u32: command_len][u32: crc32_checksum][N bytes: bincode(Command)]
/// [u32: command_len][u32: crc32_checksum][N bytes: bincode(Command)]
/// ...
pub struct WriteAheadLog {
    path: PathBuf,
    file: BufWriter<File>,
    entries_count: u64,
}

impl WriteAheadLog {
    /// Create a new WAL or open existing one
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open file in append mode (create if not exists)
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        // Count existing entries
        let entries_count = Self::count_entries(&path)?;

        Ok(Self {
            path,
            file: BufWriter::new(file),
            entries_count,
        })
    }

    /// Count existing entries in WAL without fully parsing
    fn count_entries<P: AsRef<Path>>(path: P) -> Result<u64> {
        let file = match OpenOptions::new().read(true).open(path.as_ref()) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(e.into()),
        };

        let mut reader = BufReader::new(file);
        let mut count = 0u64;

        loop {
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(()) => {
                    let len = u32::from_le_bytes(len_bytes) as usize;
                    
                    // Skip checksum
                    let mut checksum_bytes = [0u8; 4];
                    reader.read_exact(&mut checksum_bytes)?;

                    // Skip command bytes
                    let mut buffer = vec![0u8; len];
                    reader.read_exact(&mut buffer)?;

                    count += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
        }

        Ok(count)
    }

    /// Append a command to the WAL
    /// Returns the CommandId assigned to this command
    pub fn append(&mut self, cmd: &Command) -> Result<CommandId> {
        // Serialize command
        let command_bytes = bincode::serialize(cmd)?;
        let len = command_bytes.len() as u32;

        // Calculate CRC32 checksum
        let crc = Crc::<u32>::new(&crc::CRC_32_CKSUM);
        let checksum = crc.checksum(&command_bytes);

        // Write: [len][checksum][command_bytes]
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&checksum.to_le_bytes())?;
        self.file.write_all(&command_bytes)?;

        let command_id = CommandId(self.entries_count);
        self.entries_count += 1;

        Ok(command_id)
    }

    /// Force all pending writes to disk (critical for durability)
    pub fn fsync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.get_mut().sync_all()?;
        Ok(())
    }

    /// Read all commands from WAL (used for recovery)
    pub fn read_all<P: AsRef<Path>>(path: P) -> Result<Vec<(CommandId, Command)>> {
        let file = match OpenOptions::new().read(true).open(path.as_ref()) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let mut reader = BufReader::new(file);
        let mut commands = Vec::new();
        let mut entry_id = 0u64;
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

                    // Read command bytes
                    let mut command_bytes = vec![0u8; len];
                    reader.read_exact(&mut command_bytes)?;

                    // Verify checksum
                    let actual_checksum = crc.checksum(&command_bytes);
                    if actual_checksum != expected_checksum {
                        return Err(WalError::ChecksumMismatch {
                            entry_id,
                            expected: expected_checksum,
                            actual: actual_checksum,
                        });
                    }

                    // Deserialize command
                    let cmd = bincode::deserialize(&command_bytes)?;
                    commands.push((CommandId(entry_id), cmd));
                    entry_id += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
        }

        Ok(commands)
    }

    /// Get number of entries in WAL
    pub fn len(&self) -> u64 {
        self.entries_count
    }

    pub fn is_empty(&self) -> bool {
        self.entries_count == 0
    }

    /// Get path of the WAL file
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::MemoryEntry;
    use std::io::{Seek, SeekFrom};
    use tempfile::TempDir;

    #[test]
    fn test_wal_append_and_read() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Write 5 commands
        let mut wal = WriteAheadLog::new(&wal_path).unwrap();
        let mut cmd_ids = Vec::new();

        for i in 0..5 {
            let entry = MemoryEntry::new(
                crate::core::memory_entry::MemoryId(i as u64),
                "test".to_string(),
                format!("content_{}", i).into_bytes(),
                1000 + i as u64,
            );
            let cmd = Command::InsertMemory(entry);
            let id = wal.append(&cmd).unwrap();
            cmd_ids.push(id);
        }
        wal.fsync().unwrap();

        // Read them back
        let recovered = WriteAheadLog::read_all(&wal_path).unwrap();
        assert_eq!(recovered.len(), 5);

        // Verify IDs match
        for (i, (id, _)) in recovered.iter().enumerate() {
            assert_eq!(*id, cmd_ids[i]);
        }
    }

    #[test]
    fn test_wal_entry_count() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        let mut wal = WriteAheadLog::new(&wal_path).unwrap();
        assert_eq!(wal.len(), 0);

        for i in 0..10 {
            let entry = MemoryEntry::new(
                crate::core::memory_entry::MemoryId(i as u64),
                "test".to_string(),
                b"data".to_vec(),
                1000,
            );
            wal.append(&Command::InsertMemory(entry)).unwrap();
        }
        wal.fsync().unwrap();

        assert_eq!(wal.len(), 10);

        // Open existing WAL
        let wal2 = WriteAheadLog::new(&wal_path).unwrap();
        assert_eq!(wal2.len(), 10);
    }

    #[test]
    fn test_wal_checksum_validation() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Write a command
        let mut wal = WriteAheadLog::new(&wal_path).unwrap();
        let entry = MemoryEntry::new(
            crate::core::memory_entry::MemoryId(1),
            "test".to_string(),
            b"content".to_vec(),
            1000,
        );
        wal.append(&Command::InsertMemory(entry)).unwrap();
        wal.fsync().unwrap();
        drop(wal);

        // Corrupt the checksum in the file
        let mut file = OpenOptions::new().write(true).open(&wal_path).unwrap();
        file.seek(SeekFrom::Start(4)).unwrap(); // Skip length bytes
        file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap(); // Bad checksum
        drop(file);

        // Try to read - should fail with checksum error
        let result = WriteAheadLog::read_all(&wal_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            WalError::ChecksumMismatch { .. } => {}, // Expected
            e => panic!("Expected ChecksumMismatch, got {:?}", e),
        }
    }

    #[test]
    fn test_wal_persistence_across_opens() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Write commands with first WAL instance
        {
            let mut wal = WriteAheadLog::new(&wal_path).unwrap();
            for i in 0..3 {
                let entry = MemoryEntry::new(
                    crate::core::memory_entry::MemoryId(i as u64),
                    "test".to_string(),
                    b"data".to_vec(),
                    1000,
                );
                wal.append(&Command::InsertMemory(entry)).unwrap();
            }
            wal.fsync().unwrap();
        }

        // Open again and add more
        {
            let mut wal = WriteAheadLog::new(&wal_path).unwrap();
            assert_eq!(wal.len(), 3);

            for i in 3..5 {
                let entry = MemoryEntry::new(
                    crate::core::memory_entry::MemoryId(i as u64),
                    "test".to_string(),
                    b"data".to_vec(),
                    1000,
                );
                wal.append(&Command::InsertMemory(entry)).unwrap();
            }
            wal.fsync().unwrap();
        }

        // Read all - should have 5
        let recovered = WriteAheadLog::read_all(&wal_path).unwrap();
        assert_eq!(recovered.len(), 5);
    }

    #[test]
    fn test_wal_command_serialization_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Create various commands
        let mut wal = WriteAheadLog::new(&wal_path).unwrap();

        let entry = MemoryEntry::new(
            crate::core::memory_entry::MemoryId(1),
            "ns".to_string(),
            b"content".to_vec(),
            1000,
        )
        .with_importance(0.9);

        let cmd1 = Command::InsertMemory(entry);
        let cmd2 = Command::AddEdge {
            from: crate::core::memory_entry::MemoryId(1),
            to: crate::core::memory_entry::MemoryId(2),
            relation: "refers_to".to_string(),
        };
        let cmd3 = Command::DeleteMemory(crate::core::memory_entry::MemoryId(1));

        wal.append(&cmd1).unwrap();
        wal.append(&cmd2).unwrap();
        wal.append(&cmd3).unwrap();
        wal.fsync().unwrap();

        // Read back and verify
        let recovered = WriteAheadLog::read_all(&wal_path).unwrap();
        assert_eq!(recovered.len(), 3);

        // Verify each command roundtripped correctly
        match &recovered[0].1 {
            Command::InsertMemory(e) => assert_eq!(e.id, crate::core::memory_entry::MemoryId(1)),
            _ => panic!("Wrong command type"),
        }

        match &recovered[1].1 {
            Command::AddEdge { from, to, relation } => {
                assert_eq!(*from, crate::core::memory_entry::MemoryId(1));
                assert_eq!(*to, crate::core::memory_entry::MemoryId(2));
                assert_eq!(relation, "refers_to");
            }
            _ => panic!("Wrong command type"),
        }

        match &recovered[2].1 {
            Command::DeleteMemory(id) => assert_eq!(*id, crate::core::memory_entry::MemoryId(1)),
            _ => panic!("Wrong command type"),
        }
    }
}
