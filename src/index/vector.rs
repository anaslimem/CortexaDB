use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use crate::core::memory_entry::MemoryId;

#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Empty query vector")]
    EmptyQuery,
    #[error("Zero vector provided (magnitude is 0)")]
    ZeroVector,
    #[error("No embeddings indexed")]
    NoEmbeddings,
    #[error("Invalid top_k: {0}")]
    InvalidTopK(usize),
}

pub type Result<T> = std::result::Result<T, VectorError>;

/// Vector index for semantic search via embeddings
///
/// Stores embeddings (vectors) and enables fast similarity search
/// using cosine similarity with parallel computation via Rayon
#[derive(Debug, Clone)]
pub struct VectorIndex {
    /// MemoryId → embedding vector
    embeddings: HashMap<MemoryId, Vec<f32>>,
    /// Dimension of embeddings (typically 384, 768, 1536)
    vector_dimension: usize,
}

impl VectorIndex {
    /// Create a new vector index with specified dimension
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            vector_dimension,
        }
    }

    /// Add or update embedding for a memory
    pub fn index(&mut self, id: MemoryId, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.vector_dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.vector_dimension,
                actual: embedding.len(),
            });
        }

        self.embeddings.insert(id, embedding);
        Ok(())
    }

    /// Remove embedding for a memory
    pub fn remove(&mut self, id: MemoryId) -> Result<()> {
        self.embeddings.remove(&id);
        Ok(())
    }

    /// Check if memory has embedding
    pub fn has(&self, id: MemoryId) -> bool {
        self.embeddings.contains_key(&id)
    }

    /// Get number of indexed embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Serial search: find top K similar embeddings
    ///
    /// Returns list of (MemoryId, cosine_similarity_score) sorted by score descending
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(MemoryId, f32)>> {
        if query.is_empty() {
            return Err(VectorError::EmptyQuery);
        }

        if query.len() != self.vector_dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.vector_dimension,
                actual: query.len(),
            });
        }

        if self.embeddings.is_empty() {
            return Err(VectorError::NoEmbeddings);
        }

        if top_k == 0 {
            return Err(VectorError::InvalidTopK(top_k));
        }

        let query_magnitude = magnitude(query)?;

        // Compute similarity with all embeddings
        let mut results: Vec<(MemoryId, f32)> = self
            .embeddings
            .iter()
            .map(|(id, embedding)| {
                let similarity = cosine_similarity(query, embedding, query_magnitude);
                (*id, similarity)
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top K
        results.truncate(top_k);
        Ok(results)
    }

    /// Parallel search: find top K similar embeddings using Rayon
    ///
    /// Same as search() but uses thread pool for parallelization
    /// Faster for large datasets (>10k embeddings)
    pub fn search_parallel(&self, query: &[f32], top_k: usize) -> Result<Vec<(MemoryId, f32)>> {
        if query.is_empty() {
            return Err(VectorError::EmptyQuery);
        }

        if query.len() != self.vector_dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.vector_dimension,
                actual: query.len(),
            });
        }

        if self.embeddings.is_empty() {
            return Err(VectorError::NoEmbeddings);
        }

        if top_k == 0 {
            return Err(VectorError::InvalidTopK(top_k));
        }

        let query_magnitude = magnitude(query)?;

        // Parallel computation using Rayon
        let mut results: Vec<(MemoryId, f32)> = self
            .embeddings
            .par_iter() // ← Rayon parallel iterator
            .map(|(id, embedding)| {
                let similarity = cosine_similarity(query, embedding, query_magnitude);
                (*id, similarity)
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top K
        results.truncate(top_k);
        Ok(results)
    }

    /// Search similarity only within a restricted set of memory IDs.
    pub fn search_in_ids(
        &self,
        query: &[f32],
        candidate_ids: &HashSet<MemoryId>,
        top_k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if query.is_empty() {
            return Err(VectorError::EmptyQuery);
        }

        if query.len() != self.vector_dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.vector_dimension,
                actual: query.len(),
            });
        }

        if self.embeddings.is_empty() {
            return Err(VectorError::NoEmbeddings);
        }

        if top_k == 0 {
            return Err(VectorError::InvalidTopK(top_k));
        }

        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let query_magnitude = magnitude(query)?;

        let mut results: Vec<(MemoryId, f32)> = candidate_ids
            .iter()
            .filter_map(|id| {
                self.embeddings.get(id).map(|embedding| {
                    let similarity = cosine_similarity(query, embedding, query_magnitude);
                    (*id, similarity)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    /// Get dimension of embeddings
    pub fn dimension(&self) -> usize {
        self.vector_dimension
    }

    /// Get all indexed memory IDs
    pub fn indexed_ids(&self) -> Vec<MemoryId> {
        self.embeddings.keys().copied().collect()
    }
}

/// Calculate cosine similarity between two vectors
///
/// Formula: (a · b) / (|a| * |b|)
/// where · is dot product and | | is magnitude
///
/// Returns value in range [-1, 1], typically [0, 1] for embeddings
fn cosine_similarity(a: &[f32], b: &[f32], a_magnitude: f32) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    // Compute dot product
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // Compute magnitude of b
    let b_magnitude = magnitude(b).unwrap_or(0.0);

    // Avoid division by zero
    if a_magnitude == 0.0 || b_magnitude == 0.0 {
        return 0.0;
    }

    dot_product / (a_magnitude * b_magnitude)
}

/// Calculate vector magnitude (L2 norm)
///
/// Formula: sqrt(sum of squares)
fn magnitude(vec: &[f32]) -> Result<f32> {
    if vec.is_empty() {
        return Err(VectorError::ZeroVector);
    }

    let sum_of_squares: f32 = vec.iter().map(|x| x * x).sum();

    if sum_of_squares == 0.0 {
        return Err(VectorError::ZeroVector);
    }

    Ok(sum_of_squares.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_embedding(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[test]
    fn test_vector_index_new() {
        let index = VectorIndex::new(768);
        assert_eq!(index.dimension(), 768);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_index_insert_and_has() {
        let mut index = VectorIndex::new(3);
        let embedding = create_embedding(&[0.1, 0.2, 0.3]);

        index.index(MemoryId(1), embedding).unwrap();

        assert!(index.has(MemoryId(1)));
        assert!(!index.has(MemoryId(2)));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_dimension_validation() {
        let mut index = VectorIndex::new(3);
        let embedding = create_embedding(&[0.1, 0.2]); // Wrong dimension

        let result = index.index(MemoryId(1), embedding);
        assert!(result.is_err());
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_cosine_similarity_identical() {
        let v = vec![0.1, 0.2, 0.3];
        let mag = magnitude(&v).unwrap();
        let similarity = cosine_similarity(&v, &v, mag);

        // Identical vectors should have similarity of 1.0
        assert!((similarity - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vector_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let mag_a = magnitude(&a).unwrap();

        let similarity = cosine_similarity(&a, &b, mag_a);

        // Orthogonal vectors should have similarity of 0.0
        assert!(similarity.abs() < 0.0001);
    }

    #[test]
    fn test_vector_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let mag_a = magnitude(&a).unwrap();

        let similarity = cosine_similarity(&a, &b, mag_a);

        // Opposite vectors should have similarity of -1.0
        assert!((similarity - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_vector_search_single_match() {
        let mut index = VectorIndex::new(3);
        index
            .index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3]))
            .unwrap();
        index
            .index(MemoryId(2), create_embedding(&[0.5, 0.6, 0.7]))
            .unwrap();

        let results = index.search(&[0.1, 0.2, 0.3], 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, MemoryId(1));
        assert!((results[0].1 - 1.0).abs() < 0.0001); // Should match perfectly
    }

    #[test]
    fn test_vector_search_top_k() {
        let mut index = VectorIndex::new(2);
        index
            .index(MemoryId(1), create_embedding(&[1.0, 0.0]))
            .unwrap();
        index
            .index(MemoryId(2), create_embedding(&[0.9, 0.1]))
            .unwrap();
        index
            .index(MemoryId(3), create_embedding(&[0.0, 1.0]))
            .unwrap();

        let results = index.search(&[1.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, MemoryId(1)); // Perfect match
        assert_eq!(results[1].0, MemoryId(2)); // Close match
        assert!(results[0].1 > results[1].1); // First has higher score
    }

    #[test]
    fn test_vector_search_sorted_by_similarity() {
        let mut index = VectorIndex::new(2);
        index
            .index(MemoryId(1), create_embedding(&[0.0, 1.0]))
            .unwrap();
        index
            .index(MemoryId(2), create_embedding(&[0.5, 0.5]))
            .unwrap();
        index
            .index(MemoryId(3), create_embedding(&[1.0, 0.0]))
            .unwrap();

        let results = index.search(&[1.0, 0.0], 3).unwrap();

        // Should be sorted by similarity descending
        assert_eq!(results[0].0, MemoryId(3));
        assert_eq!(results[1].0, MemoryId(2));
        assert_eq!(results[2].0, MemoryId(1));

        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_vector_search_parallel_matches_serial() {
        let mut index = VectorIndex::new(10);
        for i in 0..100 {
            let embedding: Vec<f32> = (0..10).map(|j| ((i + j) as f32) / 100.0).collect();
            index.index(MemoryId(i as u64), embedding).unwrap();
        }

        let query: Vec<f32> = (0..10).map(|i| (i as f32) / 10.0).collect();

        let serial = index.search(&query, 5).unwrap();
        let parallel = index.search_parallel(&query, 5).unwrap();

        // Both should return same results in same order
        assert_eq!(serial.len(), parallel.len());
        for i in 0..serial.len() {
            assert_eq!(serial[i].0, parallel[i].0);
            assert!((serial[i].1 - parallel[i].1).abs() < 0.0001);
        }
    }

    #[test]
    fn test_vector_remove() {
        let mut index = VectorIndex::new(3);
        index
            .index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3]))
            .unwrap();
        assert_eq!(index.len(), 1);

        index.remove(MemoryId(1)).unwrap();
        assert_eq!(index.len(), 0);
        assert!(!index.has(MemoryId(1)));
    }

    #[test]
    fn test_vector_search_empty_query() {
        let index = VectorIndex::new(3);
        let result = index.search(&[], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_search_no_embeddings() {
        let index = VectorIndex::new(3);
        let result = index.search(&[0.1, 0.2, 0.3], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_search_invalid_top_k() {
        let mut index = VectorIndex::new(3);
        index
            .index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3]))
            .unwrap();

        let result = index.search(&[0.1, 0.2, 0.3], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_search_top_k_larger_than_embeddings() {
        let mut index = VectorIndex::new(3);
        index
            .index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3]))
            .unwrap();
        index
            .index(MemoryId(2), create_embedding(&[0.4, 0.5, 0.6]))
            .unwrap();

        // Request top 10 but only 2 embeddings
        let results = index.search(&[0.1, 0.2, 0.3], 10).unwrap();
        assert_eq!(results.len(), 2); // Should return only 2
    }

    #[test]
    fn test_vector_indexed_ids() {
        let mut index = VectorIndex::new(3);
        index
            .index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3]))
            .unwrap();
        index
            .index(MemoryId(5), create_embedding(&[0.4, 0.5, 0.6]))
            .unwrap();
        index
            .index(MemoryId(3), create_embedding(&[0.7, 0.8, 0.9]))
            .unwrap();

        let ids = index.indexed_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&MemoryId(1)));
        assert!(ids.contains(&MemoryId(5)));
        assert!(ids.contains(&MemoryId(3)));
    }

    #[test]
    fn test_magnitude_calculation() {
        let v = vec![3.0, 4.0]; // 3-4-5 right triangle
        let mag = magnitude(&v).unwrap();
        assert!((mag - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_magnitude_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let result = magnitude(&v);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_parallel_with_large_dataset() {
        let mut index = VectorIndex::new(100);

        // Create 1000 embeddings
        for i in 0..1000 {
            let embedding: Vec<f32> = (0..100)
                .map(|j| ((i * 17 + j * 23) as f32).sin().abs())
                .collect();
            index.index(MemoryId(i as u64), embedding).unwrap();
        }

        let query: Vec<f32> = (0..100).map(|i| (i as f32).sin().abs()).collect();

        let results = index.search_parallel(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Verify results are sorted
        for i in 0..9 {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    #[test]
    fn test_vector_search_in_ids() {
        let mut index = VectorIndex::new(3);
        index.index(MemoryId(1), vec![1.0, 0.0, 0.0]).unwrap();
        index.index(MemoryId(2), vec![0.0, 1.0, 0.0]).unwrap();
        index.index(MemoryId(3), vec![0.0, 0.0, 1.0]).unwrap();

        let candidates: HashSet<MemoryId> = [MemoryId(1), MemoryId(3)].into_iter().collect();
        let results = index
            .search_in_ids(&[1.0, 0.0, 0.0], &candidates, 5)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, MemoryId(1));
        assert!(results.iter().all(|(id, _)| *id != MemoryId(2)));
    }
}
