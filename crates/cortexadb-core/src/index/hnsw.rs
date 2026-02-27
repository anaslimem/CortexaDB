use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::core::memory_entry::MemoryId;

#[derive(Error, Debug)]
pub enum HnswError {
    #[error("USearch error: {0}")]
    UsearchError(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("No vectors indexed")]
    NoVectors,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Lock error")]
    LockError,
}

pub type Result<T> = std::result::Result<T, HnswError>;

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, ef_search: 50 }
    }
}

#[derive(Debug, Clone)]
pub enum IndexMode {
    Exact,
    Hnsw(HnswConfig),
}

impl Default for IndexMode {
    fn default() -> Self {
        Self::Exact
    }
}

#[derive(Clone)]
pub struct HnswBackend {
    index: Arc<Mutex<usearch::Index>>,
    dimension: usize,
    config: HnswConfig,
}

impl std::fmt::Debug for HnswBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswBackend")
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish()
    }
}

impl HnswBackend {
    pub fn new(dimension: usize, config: HnswConfig) -> Result<Self> {
        let options = usearch::IndexOptions {
            dimensions: dimension,
            metric: usearch::MetricKind::Cos,
            quantization: usearch::ScalarKind::F32,
            connectivity: config.m,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            ..Default::default()
        };

        let index =
            usearch::new_index(&options).map_err(|e| HnswError::UsearchError(e.to_string()))?;

        Ok(Self { index: Arc::new(Mutex::new(index)), dimension, config })
    }

    pub fn add(&self, id: MemoryId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let index = self.index.lock().map_err(|_| HnswError::LockError)?;
        index
            .add(id.0 as usearch::Key, vector)
            .map_err(|e| HnswError::UsearchError(e.to_string()))?;

        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _ef_search: Option<usize>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if query.len() != self.dimension {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let index = self.index.lock().map_err(|_| HnswError::LockError)?;

        if index.capacity() == 0 {
            return Err(HnswError::NoVectors);
        }

        let results =
            index.search(query, top_k).map_err(|e| HnswError::UsearchError(e.to_string()))?;

        let mut output = Vec::with_capacity(top_k);
        for i in 0..results.keys.len() {
            let key = results.keys.get(i);
            let distance = results.distances.get(i);
            if let (Some(key), Some(distance)) = (key, distance) {
                let score = 1.0 - distance;
                output.push((MemoryId(*key as u64), score));
            }
        }

        Ok(output)
    }

    pub fn remove(&self, id: MemoryId) -> Result<()> {
        let index = self.index.lock().map_err(|_| HnswError::LockError)?;
        index.remove(id.0 as usearch::Key).map_err(|e| HnswError::UsearchError(e.to_string()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.index.lock().map(|i| i.size()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
