# Contributing to CortexaDB

First off, thank you for considering contributing to CortexaDB! It's people like you that make the open-source community such an amazing place.

## Development Setup

### Rust
CortexaDB is primarily written in Rust. You'll need the latest stable version of Rust installed.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Python Bindings
For the Python bindings, we use `maturin`. It's recommended to use a virtual environment.

```bash
cd crates/cortexadb-py
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop
```

## How to Contribute

### 1. Find an Issue
Check out the [Issues](https://github.com/anaslimem/CortexaDB/issues) page. Look for "good first issue" labels if you're new.

### 2. Fork & Clone
Fork the repository and clone it locally.

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Code & Test
Ensure your code follows the project standards:
- Run `cargo fmt` to format your code.
- Run `cargo clippy` to check for lints.
- Run `cargo test --workspace` to ensure all tests pass.

### 5. Submit a PR
Submit your pull request against the `main` branch. Provide a clear description of your changes and why they are needed.

## Code of Conduct
We expect all contributors to follow our Code of Conduct (coming soon). Be respectful and collaborative.

## Questions?
If you have any questions, feel free to open a "Question" issue or reach out via email.
