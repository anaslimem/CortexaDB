use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

use crate::core::memory_entry::MemoryId;
use crate::index::hnsw::{HnswBackend, HnswConfig};

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

const DEFAULT_NAMESPACE: &str = "__global__";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorBackendMode {
    Exact,
    /// HNSW-like mode:
    /// 1) fetch larger approximate candidate pool (`ann_k`)
    /// 2) exact cosine rerank on those candidates
    /// 3) return top-k final results
    Ann {
        ann_search_multiplier: usize,
    },
}

impl Default for VectorBackendMode {
    fn default() -> Self {
        Self::Exact
    }
}

pub trait VectorSearchBackend: Send + Sync + std::fmt::Debug {
    fn mode(&self) -> VectorBackendMode;
    fn ann_multiplier_hint(&self) -> usize;
}

trait AnnCandidateProvider: Send + Sync + std::fmt::Debug {
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
    fn candidates(
        &self,
        query: &[f32],
        ann_k: usize,
        namespace: Option<&str>,
        partitions: &HashMap<String, NamespacePartition>,
    ) -> Result<Vec<MemoryId>>;
}

#[derive(Debug)]
struct PrefixAnnCandidateProvider;

impl AnnCandidateProvider for PrefixAnnCandidateProvider {
    fn name(&self) -> &'static str {
        "prefix"
    }

    fn candidates(
        &self,
        query: &[f32],
        ann_k: usize,
        namespace: Option<&str>,
        partitions: &HashMap<String, NamespacePartition>,
    ) -> Result<Vec<MemoryId>> {
        let approx_dims = query.len().min(8).max(1);
        let query_prefix = &query[..approx_dims];
        let query_mag = magnitude(query_prefix)?;
        let mut approx_scored = Vec::new();

        let iter: Box<dyn Iterator<Item = (&String, &NamespacePartition)>> = match namespace {
            Some(ns) => match partitions.get_key_value(ns) {
                Some(one) => Box::new(std::iter::once(one)),
                None => Box::new(std::iter::empty()),
            },
            None => Box::new(partitions.iter()),
        };

        for (_ns, partition) in iter {
            for (id, embedding) in &partition.embeddings {
                if partition.tombstones.contains(id) {
                    continue;
                }
                let score = cosine_similarity(query_prefix, &embedding[..approx_dims], query_mag);
                approx_scored.push((*id, score));
            }
        }

        if approx_scored.is_empty() {
            if namespace.is_some() {
                return Ok(Vec::new());
            }
            return Err(VectorError::NoEmbeddings);
        }

        approx_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        approx_scored.truncate(ann_k);
        Ok(approx_scored.into_iter().map(|(id, _)| id).collect())
    }
}

/// Placeholder provider with an HNSW-ready name/shape. Today it falls back to prefix ANN
/// while keeping a stable extension point for external HNSW integrations.
#[derive(Debug)]
#[allow(dead_code)]
struct HnswReadyAnnCandidateProvider;

impl AnnCandidateProvider for HnswReadyAnnCandidateProvider {
    fn name(&self) -> &'static str {
        "hnsw-ready"
    }

    fn candidates(
        &self,
        query: &[f32],
        ann_k: usize,
        namespace: Option<&str>,
        partitions: &HashMap<String, NamespacePartition>,
    ) -> Result<Vec<MemoryId>> {
        PrefixAnnCandidateProvider.candidates(query, ann_k, namespace, partitions)
    }
}

#[derive(Debug)]
struct ExactBackend;

impl VectorSearchBackend for ExactBackend {
    fn mode(&self) -> VectorBackendMode {
        VectorBackendMode::Exact
    }
    fn ann_multiplier_hint(&self) -> usize {
        1
    }
}

#[derive(Debug)]
struct AnnBackend {
    ann_search_multiplier: usize,
}

impl VectorSearchBackend for AnnBackend {
    fn mode(&self) -> VectorBackendMode {
        VectorBackendMode::Ann { ann_search_multiplier: self.ann_search_multiplier }
    }
    fn ann_multiplier_hint(&self) -> usize {
        self.ann_search_multiplier
    }
}

#[derive(Debug, Clone, Default)]
struct NamespacePartition {
    embeddings: HashMap<MemoryId, Vec<f32>>,
    tombstones: HashSet<MemoryId>,
}

/// Vector index for semantic search via embeddings
///
/// Stores embeddings (vectors) and enables fast similarity search
/// using cosine similarity with parallel computation via Rayon.
/// Supports both exact (brute-force) and HNSW approximate search.
#[derive(Clone)]
pub struct VectorIndex {
    /// namespace -> partition
    partitions: HashMap<String, NamespacePartition>,
    /// Global lookup for ID -> namespace
    id_to_namespace: HashMap<MemoryId, String>,
    /// Dimension of embeddings (typically 384, 768, 1536)
    vector_dimension: usize,
    /// Search backend mode
    backend_mode: VectorBackendMode,
    /// Pluggable backend strategy
    backend: Arc<dyn VectorSearchBackend>,
    /// ANN candidate strategy (prefix by default; swappable for HNSW/FAISS adapters)
    ann_provider: Arc<dyn AnnCandidateProvider>,
    /// HNSW backend for approximate nearest neighbor search
    hnsw_backend: Option<Arc<HnswBackend>>,
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("vector_dimension", &self.vector_dimension)
            .field("partitions", &self.partitions.len())
            .field("backend_mode", &self.backend_mode)
            .field("hnsw_enabled", &self.hnsw_backend.is_some())
            .finish()
    }
}

impl VectorIndex {
    /// Create a new vector index with specified dimension
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            partitions: HashMap::new(),
            id_to_namespace: HashMap::new(),
            vector_dimension,
            backend_mode: VectorBackendMode::Exact,
            backend: Arc::new(ExactBackend),
            ann_provider: Arc::new(PrefixAnnCandidateProvider),
            hnsw_backend: None,
        }
    }

    /// Create a new vector index with HNSW enabled
    pub fn new_with_hnsw(vector_dimension: usize, config: HnswConfig) -> Result<Self> {
        let hnsw_backend =
            HnswBackend::new(vector_dimension, config).map_err(|_e| VectorError::NoEmbeddings)?;
        Ok(Self {
            partitions: HashMap::new(),
            id_to_namespace: HashMap::new(),
            vector_dimension,
            backend_mode: VectorBackendMode::Exact,
            backend: Arc::new(ExactBackend),
            ann_provider: Arc::new(PrefixAnnCandidateProvider),
            hnsw_backend: Some(Arc::new(hnsw_backend)),
        })
    }

    #[allow(dead_code)]
    fn set_ann_provider(&mut self, provider: Arc<dyn AnnCandidateProvider>) {
        self.ann_provider = provider;
    }

    pub fn set_backend_mode(&mut self, mode: VectorBackendMode) {
        self.backend_mode = mode;
        self.backend = match mode {
            VectorBackendMode::Exact => Arc::new(ExactBackend),
            VectorBackendMode::Ann { ann_search_multiplier } => {
                Arc::new(AnnBackend { ann_search_multiplier: ann_search_multiplier.max(1) })
            }
        };
    }

    /// Enable HNSW indexing for approximate search
    pub fn enable_hnsw(&mut self, config: HnswConfig) -> Result<()> {
        let backend = HnswBackend::new(self.vector_dimension, config)
            .map_err(|_| VectorError::NoEmbeddings)?;
        self.hnsw_backend = Some(Arc::new(backend));
        Ok(())
    }

    /// Check if HNSW is enabled
    pub fn is_hnsw_enabled(&self) -> bool {
        self.hnsw_backend.is_some()
    }

    pub fn backend_mode(&self) -> VectorBackendMode {
        self.backend_mode
    }

    /// Add or update embedding for a memory
    pub fn index(&mut self, id: MemoryId, embedding: Vec<f32>) -> Result<()> {
        self.index_in_namespace(DEFAULT_NAMESPACE, id, embedding)
    }

    pub fn index_in_namespace<S: AsRef<str>>(
        &mut self,
        namespace: S,
        id: MemoryId,
        embedding: Vec<f32>,
    ) -> Result<()> {
        if embedding.len() != self.vector_dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.vector_dimension,
                actual: embedding.len(),
            });
        }

        let namespace = namespace.as_ref().to_string();
        if let Some(previous_ns) = self.id_to_namespace.get(&id).cloned() {
            if previous_ns != namespace {
                if let Some(partition) = self.partitions.get_mut(&previous_ns) {
                    partition.embeddings.remove(&id);
                    partition.tombstones.remove(&id);
                }
            }
        }

        let partition = self.partitions.entry(namespace.clone()).or_default();
        partition.tombstones.remove(&id);
        partition.embeddings.insert(id, embedding.clone());
        self.id_to_namespace.insert(id, namespace);

        // Also add to HNSW backend if enabled
        if let Some(ref hnsw_arc) = self.hnsw_backend {
            if let Some(hnsw) = Arc::get_mut(&mut Arc::clone(hnsw_arc)) {
                let _ = hnsw.add(id, &embedding);
            }
        }

        Ok(())
    }

        Ok(())
    }

    /// Remove embedding for a memory
    pub fn remove(&mut self, id: MemoryId) -> Result<()> {
        if let Some(namespace) = self.id_to_namespace.get(&id).cloned() {
            let mode = self.backend_mode;
            if let Some(partition) = self.partitions.get_mut(&namespace) {
                match mode {
                    VectorBackendMode::Exact => {
                        partition.embeddings.remove(&id);
                        partition.tombstones.remove(&id);
                    }
                    VectorBackendMode::Ann { .. } => {
                        if partition.embeddings.contains_key(&id) {
                            partition.tombstones.insert(id);
                        }
                        // Compaction trigger: rebuild if tombstones > 20%.
                        let total = partition.embeddings.len();
                        let tombstones = partition.tombstones.len();
                        if total > 0 && (tombstones as f64 / total as f64) > 0.2 {
                            Self::compact_partition(partition);
                        }
                    }
                }
                if partition.embeddings.is_empty() {
                    self.partitions.remove(&namespace);
                }
            }
            self.id_to_namespace.remove(&id);

            // Also remove from HNSW backend if enabled
            if let Some(ref hnsw_arc) = self.hnsw_backend {
                let hnsw_arc_clone = Arc::clone(hnsw_arc);
                let _ = hnsw_arc_clone.remove(id);
            }
        }
        Ok(())
    }

    /// Check if memory has embedding
    pub fn has(&self, id: MemoryId) -> bool {
        let Some(namespace) = self.id_to_namespace.get(&id) else {
            return false;
        };
        let Some(partition) = self.partitions.get(namespace) else {
            return false;
        };
        partition.embeddings.contains_key(&id) && !partition.tombstones.contains(&id)
    }

    /// Get number of indexed embeddings
    pub fn len(&self) -> usize {
        self.partitions
            .values()
            .map(|p| p.embeddings.len().saturating_sub(p.tombstones.len()))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serial search: find top K similar embeddings
    ///
    /// Returns list of (MemoryId, cosine_similarity_score) sorted by score descending
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(MemoryId, f32)>> {
        self.search_scoped(query, top_k, None, false, 1)
    }

    /// Parallel search: find top K similar embeddings using Rayon
    ///
    /// Same as search() but uses thread pool for parallelization
    /// Faster for large datasets (>10k embeddings)
    pub fn search_parallel(&self, query: &[f32], top_k: usize) -> Result<Vec<(MemoryId, f32)>> {
        self.search_scoped(query, top_k, None, true, 1)
    }

    pub fn search_scoped(
        &self,
        query: &[f32],
        top_k: usize,
        namespace: Option<&str>,
        use_parallel: bool,
        ann_candidate_multiplier: usize,
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

        if self.is_empty() {
            return Err(VectorError::NoEmbeddings);
        }

        if top_k == 0 {
            return Err(VectorError::InvalidTopK(top_k));
        }

        // Use HNSW if available (approximate search)
        if let Some(ref hnsw_arc) = self.hnsw_backend {
            let hnsw_arc_clone = Arc::clone(hnsw_arc);
            match hnsw_arc_clone.search(query, top_k, None) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    // Fall back to exact search if HNSW fails
                    eprintln!("HNSW search failed, falling back to exact: {:?}", e);
                }
            }
        }

        // Default: exact search
        match self.backend.mode() {
            VectorBackendMode::Exact => {
                self.search_exact_scoped(query, top_k, namespace, use_parallel)
            }
            VectorBackendMode::Ann { .. } => {
                let ann_multiplier =
                    ann_candidate_multiplier.max(self.backend.ann_multiplier_hint()).max(1);
                let ann_k = top_k.saturating_mul(ann_multiplier);
                let approx = self.search_approx_candidates(query, ann_k, namespace)?;
                if approx.is_empty() {
                    return Ok(Vec::new());
                }
                self.rerank_exact(query, &approx, top_k)
            }
        }
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

        if self.is_empty() {
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
                let namespace = self.id_to_namespace.get(id)?;
                let partition = self.partitions.get(namespace)?;
                if partition.tombstones.contains(id) {
                    return None;
                }
                let embedding = partition.embeddings.get(id)?;
                let similarity = cosine_similarity(query, embedding, query_magnitude);
                Some((*id, similarity))
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
        self.id_to_namespace.keys().copied().filter(|id| self.has(*id)).collect()
    }

    fn compact_partition(partition: &mut NamespacePartition) {
        if partition.tombstones.is_empty() {
            return;
        }
        let tombstones = std::mem::take(&mut partition.tombstones);
        for id in tombstones {
            partition.embeddings.remove(&id);
        }
    }

    fn partition_iter<'a>(
        &'a self,
        namespace: Option<&str>,
    ) -> Box<dyn Iterator<Item = (&'a String, &'a NamespacePartition)> + 'a> {
        match namespace {
            Some(ns) => {
                if let Some(partition) = self.partitions.get_key_value(ns) {
                    Box::new(std::iter::once(partition))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            None => Box::new(self.partitions.iter()),
        }
    }

    fn search_exact_scoped(
        &self,
        query: &[f32],
        top_k: usize,
        namespace: Option<&str>,
        use_parallel: bool,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let query_magnitude = magnitude(query)?;
        let mut results: Vec<(MemoryId, f32)> = Vec::new();

        for (_ns, partition) in self.partition_iter(namespace) {
            let iter_results: Vec<(MemoryId, f32)> = if use_parallel {
                partition
                    .embeddings
                    .par_iter()
                    .filter_map(|(id, embedding)| {
                        if partition.tombstones.contains(id) {
                            return None;
                        }
                        Some((*id, cosine_similarity(query, embedding, query_magnitude)))
                    })
                    .collect()
            } else {
                partition
                    .embeddings
                    .iter()
                    .filter_map(|(id, embedding)| {
                        if partition.tombstones.contains(id) {
                            return None;
                        }
                        Some((*id, cosine_similarity(query, embedding, query_magnitude)))
                    })
                    .collect()
            };
            results.extend(iter_results);
        }

        if results.is_empty() {
            if namespace.is_some() {
                return Ok(Vec::new());
            }
            return Err(VectorError::NoEmbeddings);
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    /// Cheap approximate pass for ANN mode: uses first 8 dimensions (or fewer),
    /// then caller performs full exact rerank on these candidates.
    fn search_approx_candidates(
        &self,
        query: &[f32],
        ann_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<MemoryId>> {
        self.ann_provider.candidates(query, ann_k, namespace, &self.partitions)
    }

    fn rerank_exact(
        &self,
        query: &[f32],
        candidate_ids: &[MemoryId],
        top_k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let query_mag = magnitude(query)?;
        let mut out = Vec::new();
        for id in candidate_ids {
            let Some(ns) = self.id_to_namespace.get(id) else {
                continue;
            };
            let Some(partition) = self.partitions.get(ns) else {
                continue;
            };
            if partition.tombstones.contains(id) {
                continue;
            }
            let Some(embedding) = partition.embeddings.get(id) else {
                continue;
            };
            out.push((*id, cosine_similarity(query, embedding, query_mag)));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top_k);
        Ok(out)
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
        index.index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3])).unwrap();
        index.index(MemoryId(2), create_embedding(&[0.5, 0.6, 0.7])).unwrap();

        let results = index.search(&[0.1, 0.2, 0.3], 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, MemoryId(1));
        assert!((results[0].1 - 1.0).abs() < 0.0001); // Should match perfectly
    }

    #[test]
    fn test_vector_search_top_k() {
        let mut index = VectorIndex::new(2);
        index.index(MemoryId(1), create_embedding(&[1.0, 0.0])).unwrap();
        index.index(MemoryId(2), create_embedding(&[0.9, 0.1])).unwrap();
        index.index(MemoryId(3), create_embedding(&[0.0, 1.0])).unwrap();

        let results = index.search(&[1.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, MemoryId(1)); // Perfect match
        assert_eq!(results[1].0, MemoryId(2)); // Close match
        assert!(results[0].1 > results[1].1); // First has higher score
    }

    #[test]
    fn test_vector_search_sorted_by_similarity() {
        let mut index = VectorIndex::new(2);
        index.index(MemoryId(1), create_embedding(&[0.0, 1.0])).unwrap();
        index.index(MemoryId(2), create_embedding(&[0.5, 0.5])).unwrap();
        index.index(MemoryId(3), create_embedding(&[1.0, 0.0])).unwrap();

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
        index.index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3])).unwrap();
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
        index.index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3])).unwrap();

        let result = index.search(&[0.1, 0.2, 0.3], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_search_top_k_larger_than_embeddings() {
        let mut index = VectorIndex::new(3);
        index.index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3])).unwrap();
        index.index(MemoryId(2), create_embedding(&[0.4, 0.5, 0.6])).unwrap();

        // Request top 10 but only 2 embeddings
        let results = index.search(&[0.1, 0.2, 0.3], 10).unwrap();
        assert_eq!(results.len(), 2); // Should return only 2
    }

    #[test]
    fn test_vector_indexed_ids() {
        let mut index = VectorIndex::new(3);
        index.index(MemoryId(1), create_embedding(&[0.1, 0.2, 0.3])).unwrap();
        index.index(MemoryId(5), create_embedding(&[0.4, 0.5, 0.6])).unwrap();
        index.index(MemoryId(3), create_embedding(&[0.7, 0.8, 0.9])).unwrap();

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
            let embedding: Vec<f32> =
                (0..100).map(|j| ((i * 17 + j * 23) as f32).sin().abs()).collect();
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
        let results = index.search_in_ids(&[1.0, 0.0, 0.0], &candidates, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, MemoryId(1));
        assert!(results.iter().all(|(id, _)| *id != MemoryId(2)));
    }

    #[test]
    fn test_namespace_partition_search_scope() {
        let mut index = VectorIndex::new(3);
        index.index_in_namespace("agent1", MemoryId(1), vec![1.0, 0.0, 0.0]).unwrap();
        index.index_in_namespace("agent2", MemoryId(2), vec![1.0, 0.0, 0.0]).unwrap();

        let scoped = index.search_scoped(&[1.0, 0.0, 0.0], 10, Some("agent1"), false, 1).unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].0, MemoryId(1));
    }

    #[test]
    fn test_ann_mode_uses_candidate_expansion_and_exact_rerank() {
        let mut index = VectorIndex::new(3);
        index.set_backend_mode(VectorBackendMode::Ann { ann_search_multiplier: 7 });
        for i in 0..30u64 {
            let emb = if i == 29 { vec![1.0, 0.0, 0.0] } else { vec![0.6, 0.8, 0.0] };
            index.index_in_namespace("agent1", MemoryId(i), emb).unwrap();
        }

        let results = index.search_scoped(&[1.0, 0.0, 0.0], 3, Some("agent1"), false, 7).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, MemoryId(29));
    }

    #[test]
    fn test_ann_tombstone_compaction_trigger_over_20pct() {
        let mut index = VectorIndex::new(3);
        index.set_backend_mode(VectorBackendMode::Ann { ann_search_multiplier: 7 });

        for i in 0..10u64 {
            index.index_in_namespace("agent1", MemoryId(i), vec![1.0, 0.0, 0.0]).unwrap();
        }
        assert_eq!(index.len(), 10);

        // 3/10 deletions exceed the 20% tombstone threshold, triggering compaction.
        index.remove(MemoryId(0)).unwrap();
        index.remove(MemoryId(1)).unwrap();
        index.remove(MemoryId(2)).unwrap();

        assert_eq!(index.len(), 7);
        let results = index.search_scoped(&[1.0, 0.0, 0.0], 20, Some("agent1"), false, 7).unwrap();
        assert!(results.iter().all(|(id, _)| *id >= MemoryId(3)));
    }
}
