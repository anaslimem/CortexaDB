//! PyO3 bindings for Mnemos — embedded vector + graph memory for AI agents.
//!
//! Exposes the Rust `facade::Mnemos` as a native Python module.

use std::collections::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mnemos_core::engine::SyncPolicy;
use mnemos_core::facade;
use mnemos_core::store::CheckpointPolicy;

// ---------------------------------------------------------------------------
// Custom exception
// ---------------------------------------------------------------------------

create_exception!(mnemos, MnemosError, PyException);

/// Convert any mnemos-core error into our custom Python exception.
fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    MnemosError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Hit — lightweight query result
// ---------------------------------------------------------------------------

/// A scored query hit. Returned by `Mnemos.ask_embedding()`.
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

#[pymethods]
impl PyHit {
    fn __repr__(&self) -> String {
        format!("Hit(id={}, score={:.4})", self.id, self.score)
    }
}

// ---------------------------------------------------------------------------
// Memory — full retrieval object
// ---------------------------------------------------------------------------

/// A full memory entry. Returned by `Mnemos.get()`.
///
/// Attributes:
///     id (int): Memory identifier.
///     namespace (str): Namespace this memory belongs to.
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
    namespace: String,
    #[pyo3(get)]
    created_at: u64,
    #[pyo3(get)]
    importance: f32,
    #[pyo3(get)]
    content: Vec<u8>,
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
            "Memory(id={}, namespace='{}', created_at={})",
            self.id, self.namespace, self.created_at
        )
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Database statistics. Returned by `Mnemos.stats()`.
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
            self.entries, self.indexed_embeddings, self.wal_length,
            self.vector_dimension, self.storage_version,
        )
    }
}

// ---------------------------------------------------------------------------
// Mnemos — main database handle
// ---------------------------------------------------------------------------

/// Embedded vector + graph memory database for AI agents.
///
/// Example:
///     >>> db = Mnemos.open("/tmp/agent.mem", dimension=128)
///     >>> mid = db.remember_embedding([0.1] * 128)
///     >>> hits = db.ask_embedding([0.1] * 128, top_k=5)
///     >>> print(hits[0].score)
#[pyclass(name = "Mnemos")]
struct PyMnemos {
    inner: facade::Mnemos,
    dimension: usize,
}

#[pymethods]
impl PyMnemos {
    /// Open or create a Mnemos database.
    ///
    /// Args:
    ///     path: Directory path for the database files.
    ///     dimension: Vector embedding dimension (required).
    ///
    /// Returns:
    ///     Mnemos: A database handle.
    ///
    /// Raises:
    ///     MnemosError: If the database cannot be opened or the dimension
    ///         mismatches an existing database.
    #[staticmethod]
    #[pyo3(text_signature = "(path, *, dimension)")]
    fn open(path: &str, dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(MnemosError::new_err("dimension must be > 0"));
        }

        let config = facade::MnemosConfig {
            vector_dimension: dimension,
            sync_policy: SyncPolicy::Async {
                interval_ms: 10,
            },
            checkpoint_policy: CheckpointPolicy::Periodic {
                every_ops: 1000,
                every_ms: 30_000,
            },
        };

        let db = facade::Mnemos::open_with_config(path, config).map_err(to_py_err)?;

        // Validate dimension matches existing data.
        let stats = db.stats();
        if stats.entries > 0 && stats.vector_dimension != dimension {
            return Err(MnemosError::new_err(format!(
                "dimension mismatch: database has dimension={}, but open() was called with dimension={}",
                stats.vector_dimension, dimension,
            )));
        }

        Ok(PyMnemos {
            inner: db,
            dimension,
        })
    }

    /// Store a new memory with the given embedding vector.
    ///
    /// Args:
    ///     embedding: List of floats (must match configured dimension).
    ///     metadata: Optional dict of string key-value pairs.
    ///     namespace: Namespace to store in (default: "default").
    ///
    /// Returns:
    ///     int: The assigned memory ID.
    ///
    /// Raises:
    ///     MnemosError: If the embedding dimension is wrong.
    #[pyo3(
        text_signature = "(self, embedding, *, metadata=None, namespace='default', content='')",
        signature = (embedding, *, metadata=None, namespace="default".to_string(), content="".to_string())
    )]
    fn remember_embedding(
        &self,
        py: Python<'_>,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        namespace: String,
        content: String,
    ) -> PyResult<u64> {
        if embedding.len() != self.dimension {
            return Err(MnemosError::new_err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len(),
            )));
        }

        let id = py.allow_threads(|| {
            if content.is_empty() {
                 self.inner.remember_in_namespace(&namespace, embedding, metadata)
            } else {
                 self.inner.remember_with_content(&namespace, content.into_bytes(), embedding, metadata)
            }
        }).map_err(to_py_err)?;
        Ok(id)
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
    ///     MnemosError: If the embedding dimension is wrong.
    #[pyo3(
        text_signature = "(self, embedding, *, top_k=5)",
        signature = (embedding, *, top_k=5)
    )]
    fn ask_embedding(&self, py: Python<'_>, embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<PyHit>> {
        if embedding.len() != self.dimension {
            return Err(MnemosError::new_err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len(),
            )));
        }

        let results = py.allow_threads(|| self.inner.ask(embedding, top_k)).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|m| PyHit {
                id: m.id,
                score: m.score,
            })
            .collect())
    }

    /// Search within a single namespace, filtering in Rust before returning results.
    ///
    /// Args:
    ///     namespace: Namespace string to filter by.
    ///     embedding: Query vector (must match configured dimension).
    ///     top_k:     Maximum number of hits to return (default 5).
    ///
    /// Returns:
    ///     List of Hit objects ranked by score, scoped to the namespace.
    ///
    /// Raises:
    ///     MnemosError: If the embedding dimension is wrong.
    #[pyo3(
        text_signature = "(self, namespace, embedding, *, top_k=5)",
        signature = (namespace, embedding, *, top_k=5)
    )]
    fn ask_in_namespace(
        &self,
        py: Python<'_>,
        namespace: &str,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> PyResult<Vec<PyHit>> {
        if embedding.len() != self.dimension {
            return Err(MnemosError::new_err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len(),
            )));
        }

        let ns = namespace.to_string();
        let results = py
            .allow_threads(|| self.inner.ask_in_namespace(&ns, embedding, top_k))
            .map_err(to_py_err)?;

        Ok(results
            .into_iter()
            .map(|m| PyHit {
                id: m.id,
                score: m.score,
            })
            .collect())
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
    ///     MnemosError: If the memory ID does not exist.
    #[pyo3(text_signature = "(self, mid)")]
    fn get(&self, mid: u64) -> PyResult<PyMemory> {
        let entry = self.inner.get_memory(mid).map_err(to_py_err)?;

        Ok(PyMemory {
            id: entry.id,
            namespace: entry.namespace.clone(),
            created_at: entry.created_at,
            importance: entry.importance,
            content: entry.content.clone(),
            metadata_inner: entry.metadata.clone(),
        })
    }

    /// Create an edge between two memories.
    ///
    /// Args:
    ///     from_id: Source memory ID.
    ///     to_id: Target memory ID.
    ///     relation: Relation label for the edge.
    ///
    /// Raises:
    ///     MnemosError: If either memory ID does not exist.
    #[pyo3(text_signature = "(self, from_id, to_id, relation)")]
    fn connect(&self, from_id: u64, to_id: u64, relation: &str) -> PyResult<()> {
        self.inner
            .connect(from_id, to_id, relation)
            .map_err(to_py_err)
    }

    /// Compact on-disk segment storage (removes tombstoned entries).
    ///
    /// Raises:
    ///     MnemosError: If compaction fails.
    #[pyo3(text_signature = "(self)")]
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.compact()).map_err(to_py_err)
    }

    /// Force a checkpoint (snapshot state + truncate WAL).
    ///
    /// Raises:
    ///     MnemosError: If the checkpoint fails.
    #[pyo3(text_signature = "(self)")]
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.checkpoint()).map_err(to_py_err)
    }

    /// Get database statistics.
    ///
    /// Returns:
    ///     Stats: Current database statistics.
    #[pyo3(text_signature = "(self)")]
    fn stats(&self) -> PyStats {
        let s = self.inner.stats();
        PyStats {
            entries: s.entries,
            indexed_embeddings: s.indexed_embeddings,
            wal_length: s.wal_length,
            vector_dimension: s.vector_dimension,
            storage_version: s.storage_version,
        }
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "Mnemos(entries={}, dimension={}, indexed={})",
            s.entries, self.dimension, s.indexed_embeddings,
        )
    }

    fn __len__(&self) -> usize {
        self.inner.stats().entries
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
// Module
// ---------------------------------------------------------------------------

/// Mnemos — embedded vector + graph memory for AI agents.
#[pymodule]
fn _mnemos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMnemos>()?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyStats>()?;
    m.add("MnemosError", m.py().get_type::<MnemosError>())?;
    Ok(())
}
