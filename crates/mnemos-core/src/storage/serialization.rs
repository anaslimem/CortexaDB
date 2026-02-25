use serde::{Deserialize, Serialize};

/// MAGIC_BYTE is used to distinguish versioned records from legacy records.
/// Legacy records (bincode(Command) or bincode(MemoryEntry)) start with 
/// small integers (LE), which typically have [0-9, 0, 0, 0] in the first 4 bytes.
/// 0xFF is a safe magic byte for future-proofing.
const MAGIC_BYTE: u8 = 0xFF;

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageWrapper<T> {
    V1(T),
    // Future: V2(NewT)
}

pub fn serialize_versioned<T: Serialize>(data: &T) -> bincode::Result<Vec<u8>> {
    let mut bytes = vec![MAGIC_BYTE];
    let wrapper = StorageWrapper::V1(data);
    bytes.extend(bincode::serialize(&wrapper)?);
    Ok(bytes)
}

pub fn deserialize_versioned<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> bincode::Result<T> {
    if bytes.first() == Some(&MAGIC_BYTE) {
        let wrapper: StorageWrapper<T> = bincode::deserialize(&bytes[1..])?;
        match wrapper {
            StorageWrapper::V1(data) => Ok(data),
        }
    } else {
        // Fallback to legacy bincode (unversioned)
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryEntry, MemoryId};

    #[test]
    fn test_versioned_roundtrip() {
        let entry = MemoryEntry::new(MemoryId(42), "ns".to_string(), b"data".to_vec(), 1000);
        let serialized = serialize_versioned(&entry).unwrap();
        assert_eq!(serialized[0], MAGIC_BYTE);
        
        let deserialized: MemoryEntry = deserialize_versioned(&serialized).unwrap();
        assert_eq!(deserialized.id, MemoryId(42));
    }

    #[test]
    fn test_legacy_fallback() {
        let entry = MemoryEntry::new(MemoryId(123), "old".to_string(), b"legacy".to_vec(), 2000);
        let legacy_serialized = bincode::serialize(&entry).unwrap();
        // Ensure legacy doesn't start with MAGIC_BYTE (it starts with ID=123 LE)
        assert_ne!(legacy_serialized[0], MAGIC_BYTE);

        let deserialized: MemoryEntry = deserialize_versioned(&legacy_serialized).unwrap();
        assert_eq!(deserialized.id, MemoryId(123));
        assert_eq!(deserialized.namespace, "old");
    }
}
