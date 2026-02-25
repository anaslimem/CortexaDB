use crc::Crc;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::core::command::Command;
use crate::storage::serialization::{deserialize_versioned, serialize_versioned};

#[derive(Error, Debug)]
pub enum WalError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    #[error("Checksum mismatch at entry {entry_id}: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch { entry_id: u64, expected: u32, actual: u32 },
    #[error("Invalid entry format at position {position}")]
    InvalidFormat { position: u64 },
}

pub type Result<T> = std::result::Result<T, WalError>;

#[derive(Debug, Clone)]
pub struct WalReadOutcome {
    pub commands: Vec<(CommandId, Command)>,
    pub valid_bytes: u64,
    pub truncated: bool,
}

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
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Count existing entries
        let entries_count = Self::count_entries(&path)?;

        Ok(Self { path, file: BufWriter::new(file), entries_count })
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
        // Serialize command (versioned)
        let command_bytes = serialize_versioned(cmd)?;
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
        Ok(Self::read_all_tolerant(path)?.commands)
    }

    /// Read all valid commands and stop at the first invalid/incomplete trailing record.
    ///
    /// Returns parsed commands plus the last known-good byte offset. Callers can truncate
    /// the file to `valid_bytes` when `truncated == true` to heal corrupted tails.
    pub fn read_all_tolerant<P: AsRef<Path>>(path: P) -> Result<WalReadOutcome> {
        let file = match OpenOptions::new().read(true).open(path.as_ref()) {
            Ok(f) => (f, path.as_ref().to_path_buf()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(WalReadOutcome {
                    commands: Vec::new(),
                    valid_bytes: 0,
                    truncated: false,
                });
            }
            Err(e) => return Err(e.into()),
        };
        let (file, wal_path) = file;
        let file_len = file.metadata()?.len();

        let mut reader = BufReader::new(file);
        let mut commands = Vec::new();
        let mut entry_id = 0u64;
        let mut valid_bytes = 0u64;
        let mut truncated = false;
        let crc = Crc::<u32>::new(&crc::CRC_32_CKSUM);

        loop {
            let record_start = valid_bytes;
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(()) => {
                    let len = u32::from_le_bytes(len_bytes) as usize;
                    let record_size = 8u64.saturating_add(len as u64);
                    if record_start.saturating_add(record_size) > file_len {
                        truncated = true;
                        break;
                    }

                    // Read checksum
                    let mut checksum_bytes = [0u8; 4];
                    if reader.read_exact(&mut checksum_bytes).is_err() {
                        truncated = true;
                        break;
                    }
                    let expected_checksum = u32::from_le_bytes(checksum_bytes);

                    // Read command bytes
                    let mut command_bytes = vec![0u8; len];
                    if reader.read_exact(&mut command_bytes).is_err() {
                        truncated = true;
                        break;
                    }

                    // Verify checksum
                    let actual_checksum = crc.checksum(&command_bytes);
                    if actual_checksum != expected_checksum {
                        // Treat as tail corruption and stop at last known-good command.
                        truncated = true;
                        break;
                    }

                    // Deserialize command (with version fallback)
                    let cmd = match deserialize_versioned(&command_bytes) {
                        Ok(cmd) => cmd,
                        Err(_) => {
                            truncated = true;
                            break;
                        }
                    };
                    commands.push((CommandId(entry_id), cmd));
                    entry_id += 1;
                    valid_bytes = record_start + record_size;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
        }

        if truncated {
            Self::truncate_to(&wal_path, valid_bytes)?;
        }

        Ok(WalReadOutcome { commands, valid_bytes, truncated })
    }

    pub fn truncate_to<P: AsRef<Path>>(path: P, valid_bytes: u64) -> Result<()> {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(valid_bytes)?;
        file.sync_all()?;
        Ok(())
    }

    /// Rewrite the WAL keeping only entries with `CommandId > keep_after`.
    ///
    /// This is the key operation for fast startup: after a checkpoint captures
    /// all state up to `keep_after`, the prefix of the WAL is no longer needed.
    /// The rewrite uses an atomic tmp-file + rename for crash safety.
    pub fn truncate_prefix<P: AsRef<Path>>(path: P, keep_after: CommandId) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(());
        }

        let outcome = Self::read_all_tolerant(path)?;
        let tail: Vec<_> =
            outcome.commands.into_iter().filter(|(id, _)| id.0 > keep_after.0).collect();

        let tmp_path = path.with_extension("wal.trunc.tmp");
        {
            let file =
                OpenOptions::new().create(true).write(true).truncate(true).open(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            let crc = Crc::<u32>::new(&crc::CRC_32_CKSUM);

            for (_id, cmd) in &tail {
                let bytes = serialize_versioned(cmd)?;
                let len = bytes.len() as u32;
                let checksum = crc.checksum(&bytes);
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(&checksum.to_le_bytes())?;
                writer.write_all(&bytes)?;
            }
            writer.flush()?;
            writer.get_mut().sync_all()?;
        }

        std::fs::rename(&tmp_path, path)?;
        // fsync parent directory for rename durability.
        if let Some(parent) = path.parent() {
            if let Ok(dir) = OpenOptions::new().read(true).open(parent) {
                let _ = dir.sync_all();
            }
        }

        Ok(())
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

        // Tail corruption should be truncated and recovered to last valid record.
        let outcome = WriteAheadLog::read_all_tolerant(&wal_path).unwrap();
        assert!(outcome.truncated);
        assert_eq!(outcome.commands.len(), 0);
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);
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
    fn test_wal_partial_tail_is_truncated() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        let mut wal = WriteAheadLog::new(&wal_path).unwrap();
        for i in 0..3 {
            let entry = MemoryEntry::new(
                crate::core::memory_entry::MemoryId(i as u64),
                "test".to_string(),
                format!("data_{i}").into_bytes(),
                1000 + i as u64,
            );
            wal.append(&Command::InsertMemory(entry)).unwrap();
        }
        wal.fsync().unwrap();
        drop(wal);

        let len = std::fs::metadata(&wal_path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&wal_path).unwrap();
        file.set_len(len - 5).unwrap();
        file.sync_all().unwrap();

        let outcome = WriteAheadLog::read_all_tolerant(&wal_path).unwrap();
        assert!(outcome.truncated);
        assert_eq!(outcome.commands.len(), 2);
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), outcome.valid_bytes);
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
