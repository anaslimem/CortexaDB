pub mod core;
pub mod engine;
pub mod facade;
pub mod index;
pub mod query;
pub mod storage;
pub mod store;

// Re-export the primary facade types for convenience.
pub use facade::{Memory, CortexaDB, CortexaDBConfig, CortexaDBError, Stats};
