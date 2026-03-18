pub mod chunker;
pub mod core;
pub mod engine;
pub mod facade;
pub mod index;
pub mod query;
pub mod storage;
pub mod store;

// Re-export the primary facade types for convenience.
pub use chunker::{chunk, ChunkMetadata, ChunkResult, ChunkingStrategy};
pub use facade::{BatchRecord, CortexaDB, CortexaDBBuilder, CortexaDBConfig, CortexaDBError, Hit, Memory, Stats};
pub use index::{HnswBackend, HnswConfig, HnswError, IndexMode, MetricKind};
