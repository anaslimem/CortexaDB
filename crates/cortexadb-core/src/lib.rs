pub mod chunker;
pub mod core;
pub mod engine;
pub mod facade;
pub mod index;
pub mod query;
pub mod storage;
pub mod store;

// Re-export the primary facade types for convenience.
pub use chunker::{ChunkMetadata, ChunkResult, ChunkingStrategy, chunk};
pub use facade::{CortexaDB, CortexaDBConfig, CortexaDBError, Memory, Stats};
pub use index::{HnswBackend, HnswConfig, HnswError, IndexMode, MetricKind};
