use thiserror::Error;

use crate::core::memory_entry::MemoryId;
use crate::core::state_machine::StateMachine;

#[derive(Error, Debug)]
pub enum TemporalError {
    #[error("Invalid time range: start={start}, end={end}")]
    InvalidTimeRange { start: u64, end: u64 },
    #[error("No memories in time range")]
    NoMemoriesInRange,
}

pub type Result<T> = std::result::Result<T, TemporalError>;

/// Temporal index for time-based queries and eviction
/// 
/// Provides range queries and memory cleanup operations
pub struct TemporalIndex;

impl TemporalIndex {
    /// Get all memories created within time range (inclusive)
    pub fn get_range(
        state_machine: &StateMachine,
        start: u64,
        end: u64,
    ) -> Result<Vec<MemoryId>> {
        if start > end {
            return Err(TemporalError::InvalidTimeRange { start, end });
        }

        let entries = state_machine.get_memories_in_time_range(start, end);
        
        if entries.is_empty() {
            return Err(TemporalError::NoMemoriesInRange);
        }

        let mut ids: Vec<MemoryId> = entries.iter().map(|e| e.id).collect();
        ids.sort();
        Ok(ids)
    }

    /// Get memories created within time range, with count
    pub fn get_range_with_count(
        state_machine: &StateMachine,
        start: u64,
        end: u64,
    ) -> Result<(Vec<MemoryId>, usize)> {
        if start > end {
            return Err(TemporalError::InvalidTimeRange { start, end });
        }

        let entries = state_machine.get_memories_in_time_range(start, end);
        let count = entries.len();

        if count == 0 {
            return Err(TemporalError::NoMemoriesInRange);
        }

        let mut ids: Vec<MemoryId> = entries.iter().map(|e| e.id).collect();
        ids.sort();
        Ok((ids, count))
    }

    /// Count memories in time range without returning them
    pub fn count_in_range(state_machine: &StateMachine, start: u64, end: u64) -> Result<usize> {
        if start > end {
            return Err(TemporalError::InvalidTimeRange { start, end });
        }

        let count = state_machine.get_memories_in_time_range(start, end).len();

        if count == 0 {
            return Err(TemporalError::NoMemoriesInRange);
        }

        Ok(count)
    }

    /// Get earliest timestamp with memories
    pub fn get_earliest_timestamp(state_machine: &StateMachine) -> Option<u64> {
        // Find first memory in the system (brute force for now)
        let mut earliest = u64::MAX;
        for entry in state_machine.get_memories_in_time_range(0, u64::MAX) {
            if entry.created_at < earliest {
                earliest = entry.created_at;
            }
        }

        if earliest == u64::MAX {
            None
        } else {
            Some(earliest)
        }
    }

    /// Get latest timestamp with memories
    pub fn get_latest_timestamp(state_machine: &StateMachine) -> Option<u64> {
        // Find last memory in the system
        let mut latest = 0u64;
        for entry in state_machine.get_memories_in_time_range(0, u64::MAX) {
            if entry.created_at > latest {
                latest = entry.created_at;
            }
        }

        if latest == 0 {
            None
        } else {
            Some(latest)
        }
    }

    /// Mark memories for eviction: all older than timestamp (for later deletion)
    /// 
    /// Returns list of MemoryIds marked for eviction
    /// Caller is responsible for actually deleting them
    pub fn mark_evict_before(
        state_machine: &StateMachine,
        timestamp: u64,
    ) -> Result<Vec<MemoryId>> {
        let entries = state_machine.get_memories_in_time_range(0, timestamp.saturating_sub(1));

        if entries.is_empty() {
            return Err(TemporalError::NoMemoriesInRange);
        }

        let mut ids: Vec<MemoryId> = entries.iter().map(|e| e.id).collect();
        ids.sort();
        Ok(ids)
    }

    /// Get memories to keep: all newer than or equal to timestamp
    pub fn get_recent(
        state_machine: &StateMachine,
        keep_after: u64,
    ) -> Result<Vec<MemoryId>> {
        let entries = state_machine.get_memories_in_time_range(keep_after, u64::MAX);

        if entries.is_empty() {
            return Err(TemporalError::NoMemoriesInRange);
        }

        let mut ids: Vec<MemoryId> = entries.iter().map(|e| e.id).collect();
        ids.sort();
        Ok(ids)
    }

    /// Get oldest N memories
    pub fn get_oldest(state_machine: &StateMachine, count: usize) -> Result<Vec<MemoryId>> {
        // Get all memories and sort by timestamp
        let mut all = Vec::new();
        for entry in state_machine.get_memories_in_time_range(0, u64::MAX) {
            all.push((entry.id, entry.created_at));
        }

        if all.is_empty() {
            return Err(TemporalError::NoMemoriesInRange);
        }

        // Sort by timestamp ascending (oldest first)
        all.sort_by_key(|(_id, timestamp)| *timestamp);

        // Take first `count` entries
        let ids: Vec<MemoryId> = all.iter().take(count).map(|(id, _)| *id).collect();

        Ok(ids)
    }

    /// Get newest N memories
    pub fn get_newest(state_machine: &StateMachine, count: usize) -> Result<Vec<MemoryId>> {
        // Get all memories and sort by timestamp
        let mut all = Vec::new();
        for entry in state_machine.get_memories_in_time_range(0, u64::MAX) {
            all.push((entry.id, entry.created_at));
        }

        if all.is_empty() {
            return Err(TemporalError::NoMemoriesInRange);
        }

        // Sort by timestamp descending (newest first)
        all.sort_by_key(|(_id, timestamp)| std::cmp::Reverse(*timestamp));

        // Take first `count` entries
        let ids: Vec<MemoryId> = all.iter().take(count).map(|(id, _)| *id).collect();

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::MemoryEntry;

    fn create_entry(id: u64, timestamp: u64) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId(id),
            "test".to_string(),
            format!("content_{}", id).into_bytes(),
            timestamp,
        )
    }

    fn setup_temporal() -> StateMachine {
        let mut sm = StateMachine::new();

        // Create memories with different timestamps
        sm.insert_memory(create_entry(0, 1000)).unwrap();
        sm.insert_memory(create_entry(1, 2000)).unwrap();
        sm.insert_memory(create_entry(2, 3000)).unwrap();
        sm.insert_memory(create_entry(3, 4000)).unwrap();
        sm.insert_memory(create_entry(4, 5000)).unwrap();

        sm
    }

    #[test]
    fn test_get_range() {
        let sm = setup_temporal();
        let ids = TemporalIndex::get_range(&sm, 1000, 3000).unwrap();

        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&MemoryId(0)));
        assert!(ids.contains(&MemoryId(1)));
        assert!(ids.contains(&MemoryId(2)));
    }

    #[test]
    fn test_get_range_single() {
        let sm = setup_temporal();
        let ids = TemporalIndex::get_range(&sm, 2000, 2000).unwrap();

        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], MemoryId(1));
    }

    #[test]
    fn test_get_range_invalid() {
        let sm = setup_temporal();
        let result = TemporalIndex::get_range(&sm, 3000, 1000);

        assert!(result.is_err());
    }

    #[test]
    fn test_get_range_empty() {
        let sm = setup_temporal();
        let result = TemporalIndex::get_range(&sm, 6000, 7000);

        assert!(result.is_err());
    }

    #[test]
    fn test_count_in_range() {
        let sm = setup_temporal();
        let count = TemporalIndex::count_in_range(&sm, 1000, 4000).unwrap();

        assert_eq!(count, 4); // 1000, 2000, 3000, 4000
    }

    #[test]
    fn test_get_earliest_timestamp() {
        let sm = setup_temporal();
        let earliest = TemporalIndex::get_earliest_timestamp(&sm);

        assert_eq!(earliest, Some(1000));
    }

    #[test]
    fn test_get_latest_timestamp() {
        let sm = setup_temporal();
        let latest = TemporalIndex::get_latest_timestamp(&sm);

        assert_eq!(latest, Some(5000));
    }

    #[test]
    fn test_mark_evict_before() {
        let sm = setup_temporal();
        let to_evict = TemporalIndex::mark_evict_before(&sm, 3000).unwrap();

        // Should evict: 0 (1000), 1 (2000)
        assert_eq!(to_evict.len(), 2);
        assert!(to_evict.contains(&MemoryId(0)));
        assert!(to_evict.contains(&MemoryId(1)));
        assert!(!to_evict.contains(&MemoryId(2))); // 3000 is not included
    }

    #[test]
    fn test_get_recent() {
        let sm = setup_temporal();
        let recent = TemporalIndex::get_recent(&sm, 3000).unwrap();

        // Should keep: 2 (3000), 3 (4000), 4 (5000)
        assert_eq!(recent.len(), 3);
        assert!(recent.contains(&MemoryId(2)));
        assert!(recent.contains(&MemoryId(3)));
        assert!(recent.contains(&MemoryId(4)));
    }

    #[test]
    fn test_get_oldest() {
        let sm = setup_temporal();
        let oldest = TemporalIndex::get_oldest(&sm, 2).unwrap();

        assert_eq!(oldest.len(), 2);
        assert_eq!(oldest[0], MemoryId(0)); // timestamp 1000
        assert_eq!(oldest[1], MemoryId(1)); // timestamp 2000
    }

    #[test]
    fn test_get_newest() {
        let sm = setup_temporal();
        let newest = TemporalIndex::get_newest(&sm, 2).unwrap();

        assert_eq!(newest.len(), 2);
        assert_eq!(newest[0], MemoryId(4)); // timestamp 5000
        assert_eq!(newest[1], MemoryId(3)); // timestamp 4000
    }

    #[test]
    fn test_get_oldest_more_than_available() {
        let sm = setup_temporal();
        let oldest = TemporalIndex::get_oldest(&sm, 100).unwrap();

        assert_eq!(oldest.len(), 5); // Only 5 memories available
    }

    #[test]
    fn test_get_newest_more_than_available() {
        let sm = setup_temporal();
        let newest = TemporalIndex::get_newest(&sm, 100).unwrap();

        assert_eq!(newest.len(), 5); // Only 5 memories available
    }

    #[test]
    fn test_empty_state_machine() {
        let sm = StateMachine::new();

        let result = TemporalIndex::get_range(&sm, 0, 1000);
        assert!(result.is_err());

        let earliest = TemporalIndex::get_earliest_timestamp(&sm);
        assert!(earliest.is_none());

        let latest = TemporalIndex::get_latest_timestamp(&sm);
        assert!(latest.is_none());
    }
}
