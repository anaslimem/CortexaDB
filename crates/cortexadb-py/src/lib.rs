//! PyO3 bindings for CortexaDB — embedded vector + graph memory for AI agents.
//!
//! Exposes the Rust `facade::CortexaDB` as a native Python module.

use std::collections::HashMap;

use cortexadb_core::{
    chunker,
    engine::{CapacityPolicy, SyncPolicy},
    facade,
    store::CheckpointPolicy,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

// ---------------------------------------------------------------------------
// Custom exception
// ---------------------------------------------------------------------------

create_exception!(cortexadb, CortexaDBError, PyException);
create_exception!(cortexadb, CortexaDBNotFoundError, CortexaDBError);
create_exception!(cortexadb, CortexaDBConfigError, CortexaDBError);
create_exception!(cortexadb, CortexaDBIOError, CortexaDBError);

/// Map core CortexaDBError to specific Python exceptions.
fn map_cortexadb_err(e: facade::CortexaDBError) -> PyErr {
    match e {
        facade::CortexaDBError::MemoryNotFound(id) => {
            CortexaDBNotFoundError::new_err(format!("Memory ID {} not found", id))
        }
        facade::CortexaDBError::Io(io_err) => CortexaDBIOError::new_err(io_err.to_string()),
        facade::CortexaDBError::Store(store_err) => match store_err {
            cortexadb_core::store::CortexaDBStoreError::Vector(
                cortexadb_core::index::vector::VectorError::DimensionMismatch { expected, actual },
            ) => CortexaDBConfigError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                expected, actual
            )),
            _ => CortexaDBError::new_err(store_err.to_string()),
        },
        _ => CortexaDBError::new_err(e.to_string()),
    }
}

/// Parse index_mode from Python - accepts string or dict
/// String: "exact", "hnsw"
/// Dict: {"type": "hnsw", "m": 16, "ef_search": 50, "ef_construction": 200, "metric": "cos"}
fn parse_index_mode(index_mode: Bound<'_, PyAny>) -> PyResult<cortexadb_core::IndexMode> {
    // If string: "exact" or "hnsw"
    if let Ok(s) = index_mode.extract::<String>() {
        return match s.to_lowercase().as_str() {
            "exact" => Ok(cortexadb_core::IndexMode::Exact),
            "hnsw" => Ok(cortexadb_core::IndexMode::Hnsw(cortexadb_core::HnswConfig::default())),
            other => Err(CortexaDBConfigError::new_err(format!(
                "unknown index_mode '{}'. Valid: 'exact', 'hnsw'",
                other
            ))),
        };
    }

    // If dict: {"type": "hnsw", "m": 16, "ef_search": 50, "ef_construction": 200, "metric": "cos"}
    if let Ok(dict) = index_mode.extract::<HashMap<String, Py<PyAny>>>() {
        let mode_type = dict
            .get("type")
            .and_then(|v| v.extract::<String>(index_mode.py()).ok())
            .unwrap_or_else(|| "hnsw".to_string());

        if mode_type.to_lowercase() == "hnsw" {
            let m =
                dict.get("m").and_then(|v| v.extract::<usize>(index_mode.py()).ok()).unwrap_or(16);
            let ef_search = dict
                .get("ef_search")
                .and_then(|v| v.extract::<usize>(index_mode.py()).ok())
                .unwrap_or(50);
            let ef_construction = dict
                .get("ef_construction")
                .and_then(|v| v.extract::<usize>(index_mode.py()).ok())
                .unwrap_or(200);

            let metric_str = dict
                .get("metric")
                .and_then(|v| v.extract::<String>(index_mode.py()).ok())
                .unwrap_or_else(|| "cos".to_string());

            let metric = match metric_str.to_lowercase().as_str() {
                "l2" | "l2sq" => cortexadb_core::MetricKind::L2,
                _ => cortexadb_core::MetricKind::Cos,
            };

            return Ok(cortexadb_core::IndexMode::Hnsw(cortexadb_core::HnswConfig {
                m,
                ef_search,
                ef_construction,
                metric,
            }));
        }

        if mode_type.to_lowercase() == "exact" {
            return Ok(cortexadb_core::IndexMode::Exact);
        }
    }

    Err(CortexaDBConfigError::new_err(
        "index_mode must be a string ('exact'/'hnsw') or dict with 'type' key".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// BatchRecord — for bulk insertion
// ---------------------------------------------------------------------------

/// A record for batch insertion.
///
/// Attributes:
///     collection (str): Collection to store in.
///     content (str/bytes): Content to store.
///     embedding (list[float] | None): Embedding vector.
///     metadata (dict[str, str] | None): Metadata.
#[pyclass(name = "BatchRecord")]
#[derive(Clone)]
struct PyBatchRecord {
    pub collection: String,
    pub content: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl PyBatchRecord {
    #[new]
    #[pyo3(signature = (collection, content, *, embedding=None, metadata=None))]
    fn new(
        collection: String,
        content: Bound<'_, PyAny>,
        embedding: Option<Vec<f32>>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let content_bytes = if let Ok(s) = content.extract::<String>() {
            s.into_bytes()
        } else if let Ok(b) = content.extract::<Vec<u8>>() {
            b
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "content must be str or bytes",
            ));
        };

        Ok(Self { collection, content: content_bytes, embedding, metadata })
    }
}

// ---------------------------------------------------------------------------
// Hit — lightweight query result
// ---------------------------------------------------------------------------

/// A scored query hit. Returned by `CortexaDB.search_embedding()`.
///
/// Attributes:
///     id (int): Memory identifier.
///     score (float): Relevance score (higher is better).
#[pyclass(frozen, name = "Hit")]
#[derive(Clone)]
struct PyHit {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    score: f32,
}

impl From<facade::Hit> for PyHit {
    fn from(h: facade::Hit) -> Self {
        Self { id: h.id, score: h.score }
    }
}

#[pymethods]
impl PyHit {
    #[new]
    #[pyo3(signature = (id, score))]
    fn new(id: u64, score: f32) -> Self {
        Self { id, score }
    }

    fn __repr__(&self) -> String {
        format!("Hit(id={}, score={:.4})", self.id, self.score)
    }
}

// ---------------------------------------------------------------------------
// Memory — full retrieval object
// ---------------------------------------------------------------------------

/// A full memory entry. Returned by `CortexaDB.get()`.
///
/// Attributes:
///     id (int): Memory identifier.
///     collection (str): Collection this memory belongs to.
///     embedding (list[float] | None): Stored embedding vector.
///     metadata (dict[str, str]): Key-value metadata.
///     created_at (int): Unix timestamp when the memory was created.
///     importance (float): Importance score.
///     content (bytes): Raw content bytes.
#[pyclass(frozen, name = "Memory")]
#[derive(Clone)]
struct PyMemory {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    collection: String,
    #[pyo3(get)]
    created_at: u64,
    #[pyo3(get)]
    importance: f32,
    #[pyo3(get)]
    content: Vec<u8>,
    #[pyo3(get)]
    embedding: Option<Vec<f32>>,
    metadata_inner: HashMap<String, String>,
}

#[pymethods]
impl PyMemory {
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.metadata_inner {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "Memory(id={}, collection='{}', created_at={})",
            self.id, self.collection, self.created_at
        )
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Database statistics. Returned by `CortexaDB.stats()`.
///
/// Attributes:
///     entries (int): Total number of stored memories.
///     indexed_embeddings (int): Number of indexed vector embeddings.
///     wal_length (int): Number of entries in the write-ahead log.
///     vector_dimension (int): Configured vector dimension.
///     storage_version (int): Storage format version (for forward compatibility).
#[pyclass(frozen, name = "Stats")]
#[derive(Clone)]
struct PyStats {
    #[pyo3(get)]
    entries: usize,
    #[pyo3(get)]
    indexed_embeddings: usize,
    #[pyo3(get)]
    wal_length: u64,
    #[pyo3(get)]
    vector_dimension: usize,
    #[pyo3(get)]
    storage_version: u32,
}

#[pymethods]
impl PyStats {
    fn __repr__(&self) -> String {
        format!(
            "Stats(entries={}, indexed_embeddings={}, wal_length={}, vector_dimension={}, storage_version={})",
            self.entries,
            self.indexed_embeddings,
            self.wal_length,
            self.vector_dimension,
            self.storage_version,
        )
    }
}

// ---------------------------------------------------------------------------
// CortexaDB — main database handle
// ---------------------------------------------------------------------------

/// Embedded vector + graph memory database for AI agents.
///
/// Example:
///     >>> db = CortexaDB.open("/tmp/agent.mem", dimension=128)
///     >>> mid = db.add_embedding([0.1] * 128)
///     >>> hits = db.search_embedding([0.1] * 128, top_k=5)
///     >>> print(hits[0].score)
#[pyclass(name = "CortexaDB")]
struct PyCortexaDB {
    inner: facade::CortexaDB,
    dimension: usize,
}

#[pymethods]
impl PyCortexaDB {
    /// Open or create a CortexaDB database.
    ///
    /// Args:
    ///     path: Directory path for the database files.
    ///     dimension: Vector embedding dimension (required).
    ///
    /// Returns:
    ///     CortexaDB: A database handle.
    ///
    /// Raises:
    ///     CortexaDBError: If the database cannot be opened or the dimension
    ///         mismatches an existing database.
    #[staticmethod]
    #[pyo3(
        text_signature = "(path, *, dimension, sync='strict', index_mode='exact', max_entries=None, max_bytes=None)"
    )]
    fn open(
        path: &str,
        dimension: usize,
        sync: String,
        index_mode: Bound<'_, PyAny>,
        max_entries: Option<usize>,
        max_bytes: Option<u64>,
    ) -> PyResult<Self> {
        if dimension == 0 {
            return Err(CortexaDBConfigError::new_err("dimension must be > 0"));
        }

        let sync_policy = match sync.to_lowercase().as_str() {
            "strict" => SyncPolicy::Strict,
            "async" => SyncPolicy::Async { interval_ms: 10 },
            "batch" => SyncPolicy::Batch { max_ops: 64, max_delay_ms: 50 },
            other => {
                return Err(CortexaDBConfigError::new_err(format!(
                    "unknown sync policy '{}'. Valid values: 'strict', 'async', 'batch'",
                    other,
                )));
            }
        };

        let index_mode = parse_index_mode(index_mode)?;

        let config = facade::CortexaDBConfig {
            vector_dimension: dimension,
            sync_policy,
            // Disabled: the Drop impl's checkpoint_now() truncates the WAL
            // based on a potentially-stale snapshot, which can lose the last
            // few entries. Disabling checkpoint avoids WAL truncation on Drop;
            // the user can still call checkpoint() explicitly when safe.
            checkpoint_policy: CheckpointPolicy::Disabled,
            capacity_policy: CapacityPolicy::new(max_entries, max_bytes),
            index_mode,
        };

        let db = facade::CortexaDB::open_with_config(path, config).map_err(map_cortexadb_err)?;

        // Validate dimension matches existing data.
        let stats = db.stats().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if stats.entries > 0 && stats.vector_dimension != dimension {
            return Err(PyValueError::new_err(format!(
                "Database initialized with dimension {} but opened with dimension {}",
                stats.vector_dimension, dimension,
            )));
        }

        Ok(PyCortexaDB { inner: db, dimension })
    }

    /// Store a new memory with the given embedding vector.
    ///
    /// Args:
    ///     embedding: List of floats (must match configured dimension).
    ///     metadata: Optional dict of string key-value pairs.
    ///     collection: Collection to store in (default: "default").
    ///
    /// Returns:
    ///     int: The assigned memory ID.
    ///
    /// Raises:
    ///     CortexaDBError: If the embedding dimension is wrong.
    #[pyo3(
        text_signature = "(self, embedding, *, metadata=None, collection='default', content='')",
        signature = (embedding, *, metadata=None, collection="default".to_string(), content="".to_string())
    )]
    fn add_embedding(
        &self,
        py: Python<'_>,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        collection: String,
        content: String,
    ) -> PyResult<u64> {
        if embedding.len() != self.dimension {
            return Err(CortexaDBError::new_err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len(),
            )));
        }

        let id = py
            .allow_threads(|| {
                if content.is_empty() {
                    self.inner.add_in_collection(&collection, embedding, metadata)
                } else {
                    self.inner.add_with_content(
                        &collection,
                        content.into_bytes(),
                        embedding,
                        metadata,
                    )
                }
            })
            .map_err(map_cortexadb_err)?;
        Ok(id)
    }

    /// Store a batch of memories efficiently.
    ///
    /// Args:
    ///     records: List of BatchRecord objects.
    ///
    /// Returns:
    ///     int: The ID of the last command executed (for flushing/waiting).
    #[pyo3(text_signature = "(self, records)")]
    fn add_batch(&self, py: Python<'_>, records: Vec<PyBatchRecord>) -> PyResult<Vec<u64>> {
        for rec in &records {
            if let Some(emb) = &rec.embedding {
                if emb.len() != self.dimension {
                    return Err(CortexaDBError::new_err(format!(
                        "embedding dimension mismatch in batch: expected {}, got {}",
                        self.dimension,
                        emb.len(),
                    )));
                }
            }
        }

        let facade_records: Vec<facade::BatchRecord> = records
            .into_iter()
            .map(|r| facade::BatchRecord {
                collection: r.collection,
                content: r.content,
                embedding: r.embedding,
                metadata: r.metadata,
            })
            .collect();

        let ids = py
            .allow_threads(|| self.inner.add_batch(facade_records))
            .map_err(|e| CortexaDBError::new_err(e.to_string()))?;

        Ok(ids)
    }

    /// Query the database by embedding vector similarity.
    ///
    /// Args:
    ///     embedding: Query vector (must match configured dimension).
    ///     top_k: Number of results to return (default: 5).
    ///
    /// Returns:
    ///     list[Hit]: Scored results sorted by descending relevance.
    ///
    /// Raises:
    ///     CortexaDBError: If the embedding dimension is wrong.
    #[pyo3(
        text_signature = "(self, embedding, *, top_k=5, filter=None)",
        signature = (embedding, *, top_k=5, filter=None)
    )]
    fn search_embedding(
        &self,
        py: Python<'_>,
        embedding: Vec<f32>,
        top_k: usize,
        filter: Option<HashMap<String, String>>,
    ) -> PyResult<Vec<PyHit>> {
        if embedding.len() != self.dimension {
            return Err(CortexaDBError::new_err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len(),
            )));
        }

        let results = py
            .allow_threads(|| self.inner.search(embedding, top_k, filter))
            .map_err(map_cortexadb_err)?;
        Ok(results.into_iter().map(|m| PyHit { id: m.id, score: m.score }).collect())
    }

    /// Search within a single collection, filtering in Rust before returning results.
    ///
    /// Args:
    ///     collection: Collection string to filter by.
    ///     embedding: Query vector (must match configured dimension).
    ///     top_k:     Maximum number of hits to return (default 5).
    ///
    /// Returns:
    ///     List of Hit objects ranked by score, scoped to the collection.
    ///
    /// Raises:
    ///     CortexaDBError: If the embedding dimension is wrong.
    #[pyo3(
        text_signature = "(self, collection, embedding, *, top_k=5, filter=None)",
        signature = (collection, embedding, *, top_k=5, filter=None)
    )]
    fn search_in_collection(
        &self,
        py: Python<'_>,
        collection: &str,
        embedding: Vec<f32>,
        top_k: usize,
        filter: Option<HashMap<String, String>>,
    ) -> PyResult<Vec<PyHit>> {
        if embedding.len() != self.dimension {
            return Err(CortexaDBError::new_err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len(),
            )));
        }

        let results = py
            .allow_threads(|| self.inner.search_in_collection(collection, embedding, top_k, filter))
            .map_err(map_cortexadb_err)?;

        Ok(results.into_iter().map(|m| m.into()).collect::<Vec<PyHit>>())
    }

    /// Retrieve a full memory by ID.
    ///
    /// Args:
    ///     mid: Memory identifier.
    ///
    /// Returns:
    ///     Memory: The full memory entry.
    ///
    /// Raises:
    ///     CortexaDBError: If the memory ID does not exist.
    #[pyo3(text_signature = "(self, mid)")]
    fn get(&self, mid: u64) -> PyResult<PyMemory> {
        let entry = self.inner.get_memory(mid).map_err(map_cortexadb_err)?;

        Ok(PyMemory {
            id: entry.id,
            collection: entry.collection.clone(),
            created_at: entry.created_at,
            importance: entry.importance,
            content: entry.content.clone(),
            embedding: entry.embedding.clone(),
            metadata_inner: entry.metadata.clone(),
        })
    }

    /// Delete a memory by ID.
    ///
    /// Args:
    ///     mid: Memory identifier.
    ///
    /// Raises:
    ///     CortexaDBError: If the memory ID does not exist or deletion fails.
    #[pyo3(text_signature = "(self, mid)")]
    fn delete(&self, py: Python<'_>, mid: u64) -> PyResult<()> {
        py.allow_threads(|| self.inner.delete(mid)).map_err(map_cortexadb_err)
    }

    /// Create an edge between two memories.
    ///
    /// Args:
    ///     from_id: Source memory ID.
    ///     to_id: Target memory ID.
    ///     relation: Relation label for the edge.
    ///
    /// Raises:
    ///     CortexaDBError: If either memory ID does not exist.
    #[pyo3(text_signature = "(self, from_id, to_id, relation)")]
    fn connect(&self, from_id: u64, to_id: u64, relation: &str) -> PyResult<()> {
        self.inner.connect(from_id, to_id, relation).map_err(map_cortexadb_err)
    }

    /// Retrieve the outgoing graph connections from a specific memory.
    ///
    /// Args:
    ///     id: Source memory ID.
    ///
    /// Returns:
    ///     List of ``(target_id, relation_label)`` tuples.
    ///
    /// Raises:
    ///     CortexaDBError: If the memory ID does not exist.
    #[pyo3(text_signature = "(self, id)")]
    fn get_neighbors(&self, id: u64) -> PyResult<Vec<(u64, String)>> {
        self.inner.get_neighbors(id).map_err(map_cortexadb_err)
    }

    /// Compact on-disk segment storage (removes tombstoned entries).
    ///
    /// Raises:
    ///     CortexaDBError: If compaction fails.
    #[pyo3(text_signature = "(self)")]
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.compact()).map_err(map_cortexadb_err)
    }

    /// Flush all pending WAL writes to disk.
    /// Raises:
    ///     CortexaDBError: If the flush fails.
    #[pyo3(text_signature = "(self)")]
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.flush()).map_err(map_cortexadb_err)
    }

    /// Force a checkpoint (snapshot state + truncate WAL).
    ///
    /// Raises:
    ///     CortexaDBError: If the checkpoint fails.
    #[pyo3(text_signature = "(self)")]
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.checkpoint()).map_err(map_cortexadb_err)
    }

    /// Get database statistics.
    ///
    /// Returns:
    ///     Stats: Current database statistics.
    #[pyo3(text_signature = "(self)")]
    fn stats(&self) -> PyResult<PyStats> {
        let s = self.inner.stats().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyStats {
            entries: s.entries,
            indexed_embeddings: s.indexed_embeddings,
            wal_length: s.wal_length,
            vector_dimension: s.vector_dimension,
            storage_version: s.storage_version,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let s = self.inner.stats().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(format!(
            "CortexaDB(entries={}, dim={}, indexed={})",
            s.entries, self.dimension, s.indexed_embeddings,
        ))
    }

    fn __len__(&self) -> usize {
        self.inner.stats().map(|s| s.entries).unwrap_or(0)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_value: Option<&Bound<'_, pyo3::types::PyAny>>,
        _traceback: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// ChunkResult — chunking result
// ---------------------------------------------------------------------------

#[pyclass(frozen, name = "ChunkResult")]
#[derive(Clone)]
struct PyChunkResult {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl PyChunkResult {
    fn __repr__(&self) -> String {
        format!(
            "ChunkResult(index={}, text='{}...')",
            self.index,
            &self.text[..self.text.len().min(50)]
        )
    }
}

// ---------------------------------------------------------------------------
// Chunking function
// ---------------------------------------------------------------------------

/// Chunk text using the specified strategy.
///
/// Args:
///     text: The input text to chunk.
///     strategy: Chunking strategy - "fixed", "recursive", "semantic", "markdown", or "json".
///     chunk_size: Target size of each chunk (for fixed/recursive).
///     overlap: Number of words to overlap between chunks.
///
/// Returns:
///     List of ChunkResult objects.
#[pyfunction]
#[pyo3(text_signature = "(text, strategy='recursive', *, chunk_size=512, overlap=50)")]
fn chunk(
    text: &str,
    strategy: &str,
    chunk_size: usize,
    overlap: usize,
) -> PyResult<Vec<PyChunkResult>> {
    let chunk_strategy = match strategy.to_lowercase().as_str() {
        "fixed" => chunker::ChunkingStrategy::Fixed { chunk_size, overlap },
        "recursive" => chunker::ChunkingStrategy::Recursive { chunk_size, overlap },
        "semantic" => chunker::ChunkingStrategy::Semantic { overlap },
        "markdown" => chunker::ChunkingStrategy::Markdown { preserve_headers: true, overlap },
        "json" => chunker::ChunkingStrategy::Json { overlap },
        _ => {
            return Err(CortexaDBError::new_err(format!(
                "Unknown chunking strategy '{}'. Valid values: 'fixed', 'recursive', 'semantic', 'markdown', 'json'",
                strategy
            )));
        }
    };

    let results = chunker::chunk(text, chunk_strategy);

    Ok(results
        .into_iter()
        .map(|c| {
            let metadata = c.metadata.map(|m| {
                let mut map = HashMap::new();
                if let Some(key) = m.key {
                    map.insert("key".to_string(), key);
                }
                if let Some(value) = m.value {
                    map.insert("value".to_string(), value);
                }
                map
            });
            PyChunkResult { text: c.text, index: c.index, metadata }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// CortexaDB — embedded vector + graph memory for AI agents.
#[pymodule]
fn _cortexadb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCortexaDB>()?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyBatchRecord>()?;
    m.add_class::<PyStats>()?;
    m.add_class::<PyChunkResult>()?;
    m.add_function(wrap_pyfunction!(chunk, m)?)?;
    m.add("CortexaDBError", m.py().get_type::<CortexaDBError>())?;
    m.add("CortexaDBNotFoundError", m.py().get_type::<CortexaDBNotFoundError>())?;
    m.add("CortexaDBConfigError", m.py().get_type::<CortexaDBConfigError>())?;
    m.add("CortexaDBIOError", m.py().get_type::<CortexaDBIOError>())?;
    Ok(())
}
