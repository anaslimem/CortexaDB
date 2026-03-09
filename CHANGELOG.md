# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - 2026-03-08

### Added
- **100x Faster Ingestion**: Implemented Rust-level batching in `remember_batch` and Python `ingest()`.
- **Modernized API**: New `Collection` abstraction and unified `add`/`search` methods.
- **Fluent Query Builder**: Chainable interface for complex discovery queries.

### Fixed
- **CI Regression Fixes**: Resolved keyword argument collisions and restored missing core methods.
- **Graph Isolation**: Ensured graph-based discovery respects collection boundaries.

## [0.1.7] - 2026-03-07

### Added
- Initial batching support (pre-release).

## [0.1.0] - 2026-02-25

### Added
- Initial release of **CortexaDB**.
- Core Rust engine with Log-Structured Merge patterns.
- Write-Ahead Log (WAL) for crash safety.
- Python bindings via PyO3 and maturain.
- Vector semantic search and Graph relationship support.
- Temporal query boosting.
- Multi-agent collections.

### Fixed
- Python 3.14 build compatibility in CI.
