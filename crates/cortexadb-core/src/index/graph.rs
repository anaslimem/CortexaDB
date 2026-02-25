use std::collections::{HashMap, VecDeque};
use thiserror::Error;

use crate::core::memory_entry::MemoryId;
use crate::core::state_machine::StateMachine;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Memory not found: {0:?}")]
    MemoryNotFound(MemoryId),
    #[error("Invalid max_hops: {0}")]
    InvalidMaxHops(usize),
    #[error("Invalid max_depth: {0}")]
    InvalidMaxDepth(usize),
}

pub type Result<T> = std::result::Result<T, GraphError>;

/// Graph index for navigating memory relationships
///
/// Provides traversal methods: BFS, DFS, pathfinding
pub struct GraphIndex;

impl GraphIndex {
    /// Breadth-first search: find all memories reachable within N hops
    ///
    /// Returns HashMap of MemoryId → distance (hop count)
    pub fn bfs(
        state_machine: &StateMachine,
        start: MemoryId,
        max_hops: usize,
    ) -> Result<HashMap<MemoryId, usize>> {
        Self::bfs_internal(state_machine, start, max_hops, None)
    }

    pub fn bfs_in_namespace(
        state_machine: &StateMachine,
        start: MemoryId,
        max_hops: usize,
        namespace: &str,
    ) -> Result<HashMap<MemoryId, usize>> {
        Self::bfs_internal(state_machine, start, max_hops, Some(namespace))
    }

    fn bfs_internal(
        state_machine: &StateMachine,
        start: MemoryId,
        max_hops: usize,
        namespace: Option<&str>,
    ) -> Result<HashMap<MemoryId, usize>> {
        if max_hops == 0 {
            return Err(GraphError::InvalidMaxHops(max_hops));
        }

        // Verify start memory exists
        let start_entry =
            state_machine.get_memory(start).map_err(|_| GraphError::MemoryNotFound(start))?;
        if let Some(ns) = namespace {
            if start_entry.namespace != ns {
                return Ok(HashMap::new());
            }
        }

        let mut visited = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back((start, 0));
        visited.insert(start, 0);

        while let Some((current, distance)) = queue.pop_front() {
            // Don't explore beyond max_hops
            if distance >= max_hops {
                continue;
            }

            // Get neighbors of current memory
            if let Ok(neighbors) = state_machine.get_neighbors(current) {
                for (neighbor_id, _relation) in neighbors {
                    if let Some(ns) = namespace {
                        let in_namespace = state_machine
                            .namespace_of(neighbor_id)
                            .map(|v| v == ns)
                            .unwrap_or(false);
                        if !in_namespace {
                            continue;
                        }
                    }
                    if !visited.contains_key(&neighbor_id) {
                        let new_distance = distance + 1;
                        visited.insert(neighbor_id, new_distance);
                        queue.push_back((neighbor_id, new_distance));
                    }
                }
            }
        }

        Ok(visited)
    }

    /// Depth-first search: find all paths from start
    ///
    /// Returns list of paths (each path is Vec<MemoryId>)
    pub fn dfs(
        state_machine: &StateMachine,
        start: MemoryId,
        max_depth: usize,
    ) -> Result<Vec<Vec<MemoryId>>> {
        if max_depth == 0 {
            return Err(GraphError::InvalidMaxDepth(max_depth));
        }

        // Verify start memory exists
        state_machine.get_memory(start).map_err(|_| GraphError::MemoryNotFound(start))?;

        let mut paths = Vec::new();
        let mut visited = std::collections::HashSet::new();
        visited.insert(start);

        Self::dfs_recursive(
            state_machine,
            start,
            0,
            max_depth,
            &mut vec![start],
            &mut visited,
            &mut paths,
        );

        Ok(paths)
    }

    /// Helper for DFS recursion
    fn dfs_recursive(
        state_machine: &StateMachine,
        current: MemoryId,
        depth: usize,
        max_depth: usize,
        path: &mut Vec<MemoryId>,
        visited: &mut std::collections::HashSet<MemoryId>,
        paths: &mut Vec<Vec<MemoryId>>,
    ) {
        // Save current path
        paths.push(path.clone());

        // Stop if max depth reached
        if depth >= max_depth {
            return;
        }

        // Explore neighbors
        if let Ok(neighbors) = state_machine.get_neighbors(current) {
            for (neighbor_id, _relation) in neighbors {
                // Only explore if not in current path (prevent cycles)
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    path.push(neighbor_id);

                    Self::dfs_recursive(
                        state_machine,
                        neighbor_id,
                        depth + 1,
                        max_depth,
                        path,
                        visited,
                        paths,
                    );

                    path.pop();
                    visited.remove(&neighbor_id);
                }
            }
        }
    }

    /// Find shortest path between two memories using BFS
    ///
    /// Returns path as Vec<MemoryId> or None if unreachable
    pub fn find_path(
        state_machine: &StateMachine,
        from: MemoryId,
        to: MemoryId,
    ) -> Result<Option<Vec<MemoryId>>> {
        // Verify both memories exist
        state_machine.get_memory(from).map_err(|_| GraphError::MemoryNotFound(from))?;
        state_machine.get_memory(to).map_err(|_| GraphError::MemoryNotFound(to))?;

        if from == to {
            return Ok(Some(vec![from]));
        }

        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<MemoryId, MemoryId> = HashMap::new();

        queue.push_back(from);
        visited.insert(from);

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = vec![to];
                let mut current = to;

                while current != from {
                    if let Some(&prev) = parent.get(&current) {
                        path.push(prev);
                        current = prev;
                    } else {
                        break;
                    }
                }

                path.reverse();
                return Ok(Some(path));
            }

            // Explore neighbors
            if let Ok(neighbors) = state_machine.get_neighbors(current) {
                for (neighbor_id, _relation) in neighbors {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);
                        parent.insert(neighbor_id, current);
                        queue.push_back(neighbor_id);
                    }
                }
            }
        }

        Ok(None) // No path found
    }

    /// Get all neighbors (one hop away)
    pub fn get_neighbors(state_machine: &StateMachine, id: MemoryId) -> Result<Vec<MemoryId>> {
        let neighbors =
            state_machine.get_neighbors(id).map_err(|_| GraphError::MemoryNotFound(id))?;

        let ids: Vec<MemoryId> = neighbors.iter().map(|(id, _)| *id).collect();
        Ok(ids)
    }

    /// Get all memories reachable within N hops
    pub fn get_reachable(
        state_machine: &StateMachine,
        start: MemoryId,
        max_hops: usize,
    ) -> Result<Vec<MemoryId>> {
        let visited = Self::bfs(state_machine, start, max_hops)?;
        let mut ids: Vec<MemoryId> = visited.keys().copied().collect();
        ids.sort();
        Ok(ids)
    }

    pub fn get_reachable_in_namespace(
        state_machine: &StateMachine,
        start: MemoryId,
        max_hops: usize,
        namespace: &str,
    ) -> Result<Vec<MemoryId>> {
        let visited = Self::bfs_in_namespace(state_machine, start, max_hops, namespace)?;
        let mut ids: Vec<MemoryId> = visited.keys().copied().collect();
        ids.sort();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_entry::{MemoryEntry, MemoryId};

    fn create_entry(id: u64) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId(id),
            "test".to_string(),
            format!("content_{}", id).into_bytes(),
            1000 + id,
        )
    }

    fn setup_graph() -> StateMachine {
        let mut sm = StateMachine::new();

        // Create memories
        for i in 0..5 {
            sm.insert_memory(create_entry(i)).unwrap();
        }

        // Create edges: 0→1, 0→2, 1→3, 2→3, 3→4
        sm.add_edge(MemoryId(0), MemoryId(1), "points".to_string()).unwrap();
        sm.add_edge(MemoryId(0), MemoryId(2), "refers".to_string()).unwrap();
        sm.add_edge(MemoryId(1), MemoryId(3), "links".to_string()).unwrap();
        sm.add_edge(MemoryId(2), MemoryId(3), "connects".to_string()).unwrap();
        sm.add_edge(MemoryId(3), MemoryId(4), "leads".to_string()).unwrap();

        sm
    }

    #[test]
    fn test_bfs_single_hop() {
        let sm = setup_graph();
        let result = GraphIndex::bfs(&sm, MemoryId(0), 1).unwrap();

        // From 0, 1 hop reaches: 1, 2
        assert_eq!(result.len(), 3); // Including start
        assert_eq!(result.get(&MemoryId(0)), Some(&0));
        assert_eq!(result.get(&MemoryId(1)), Some(&1));
        assert_eq!(result.get(&MemoryId(2)), Some(&1));
        assert_eq!(result.get(&MemoryId(3)), None);
    }

    #[test]
    fn test_bfs_two_hops() {
        let sm = setup_graph();
        let result = GraphIndex::bfs(&sm, MemoryId(0), 2).unwrap();

        // From 0, 2 hops reaches: 1, 2, 3
        assert!(result.contains_key(&MemoryId(0))); // start
        assert!(result.contains_key(&MemoryId(1))); // 1 hop
        assert!(result.contains_key(&MemoryId(2))); // 1 hop
        assert!(result.contains_key(&MemoryId(3))); // 2 hops
        assert!(!result.contains_key(&MemoryId(4))); // 3 hops
    }

    #[test]
    fn test_bfs_distance_tracking() {
        let sm = setup_graph();
        let result = GraphIndex::bfs(&sm, MemoryId(0), 3).unwrap();

        assert_eq!(result[&MemoryId(0)], 0);
        assert_eq!(result[&MemoryId(1)], 1);
        assert_eq!(result[&MemoryId(2)], 1);
        assert_eq!(result[&MemoryId(3)], 2);
        assert_eq!(result[&MemoryId(4)], 3);
    }

    #[test]
    fn test_bfs_nonexistent_start() {
        let sm = setup_graph();
        let result = GraphIndex::bfs(&sm, MemoryId(999), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_path_direct() {
        let sm = setup_graph();
        let path = GraphIndex::find_path(&sm, MemoryId(0), MemoryId(1)).unwrap();

        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![MemoryId(0), MemoryId(1)]);
    }

    #[test]
    fn test_find_path_indirect() {
        let sm = setup_graph();
        let path = GraphIndex::find_path(&sm, MemoryId(0), MemoryId(4)).unwrap();

        assert!(path.is_some());
        let path_vec = path.unwrap();
        // Multiple paths possible: 0→1→3→4 or 0→2→3→4
        assert_eq!(path_vec[0], MemoryId(0));
        assert_eq!(path_vec[path_vec.len() - 1], MemoryId(4));
    }

    #[test]
    fn test_find_path_same_node() {
        let sm = setup_graph();
        let path = GraphIndex::find_path(&sm, MemoryId(0), MemoryId(0)).unwrap();

        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![MemoryId(0)]);
    }

    #[test]
    fn test_find_path_unreachable() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_entry(0)).unwrap();
        sm.insert_memory(create_entry(1)).unwrap();
        // No edge between them

        let path = GraphIndex::find_path(&sm, MemoryId(0), MemoryId(1)).unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_get_neighbors() {
        let sm = setup_graph();
        let neighbors = GraphIndex::get_neighbors(&sm, MemoryId(0)).unwrap();

        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&MemoryId(1)));
        assert!(neighbors.contains(&MemoryId(2)));
    }

    #[test]
    fn test_get_reachable() {
        let sm = setup_graph();
        let reachable = GraphIndex::get_reachable(&sm, MemoryId(0), 3).unwrap();

        assert_eq!(reachable.len(), 5); // All memories reachable
        assert!(reachable.contains(&MemoryId(0)));
        assert!(reachable.contains(&MemoryId(1)));
        assert!(reachable.contains(&MemoryId(2)));
        assert!(reachable.contains(&MemoryId(3)));
        assert!(reachable.contains(&MemoryId(4)));
    }

    #[test]
    fn test_dfs_paths() {
        let sm = setup_graph();
        let paths = GraphIndex::dfs(&sm, MemoryId(0), 2).unwrap();

        // Should have multiple paths
        assert!(paths.len() > 0);

        // All paths should start with 0
        for path in &paths {
            assert_eq!(path[0], MemoryId(0));
        }
    }

    #[test]
    fn test_invalid_max_hops() {
        let sm = setup_graph();
        let result = GraphIndex::bfs(&sm, MemoryId(0), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_depth() {
        let sm = setup_graph();
        let result = GraphIndex::dfs(&sm, MemoryId(0), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dfs_does_not_revisit_start_in_cycle() {
        let mut sm = StateMachine::new();
        sm.insert_memory(create_entry(0)).unwrap();
        sm.insert_memory(create_entry(1)).unwrap();
        sm.add_edge(MemoryId(0), MemoryId(1), "to".to_string()).unwrap();
        sm.add_edge(MemoryId(1), MemoryId(0), "back".to_string()).unwrap();

        let paths = GraphIndex::dfs(&sm, MemoryId(0), 3).unwrap();
        assert!(paths.iter().all(|path| path.iter().filter(|&&id| id == MemoryId(0)).count() == 1));
    }

    #[test]
    fn test_bfs_in_namespace_filters_neighbors() {
        let mut sm = StateMachine::new();
        sm.insert_memory(MemoryEntry::new(MemoryId(1), "agent1".to_string(), b"a".to_vec(), 1000))
            .unwrap();
        sm.insert_memory(MemoryEntry::new(MemoryId(2), "agent1".to_string(), b"b".to_vec(), 1001))
            .unwrap();
        sm.insert_memory(MemoryEntry::new(MemoryId(3), "agent2".to_string(), b"c".to_vec(), 1002))
            .unwrap();

        sm.add_edge(MemoryId(1), MemoryId(2), "ok".to_string()).unwrap();
        // Cross namespace should be rejected by state machine, so no leakage via edges.
        assert!(sm.add_edge(MemoryId(2), MemoryId(3), "bad".to_string()).is_err());

        let scoped = GraphIndex::bfs_in_namespace(&sm, MemoryId(1), 3, "agent1").unwrap();
        assert!(scoped.contains_key(&MemoryId(1)));
        assert!(scoped.contains_key(&MemoryId(2)));
        assert!(!scoped.contains_key(&MemoryId(3)));
    }
}
