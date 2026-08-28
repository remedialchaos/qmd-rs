# qmd

[![CI][ci-badge]][ci-url]
[![License][license-badge]][license-url]
[![Rust][rust-badge]][rust-url]

[ci-badge]: https://github.com/qntx/qmd/actions/workflows/rust.yml/badge.svg
[ci-url]: https://github.com/qntx/qmd/actions/workflows/rust.yml
[license-badge]: https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg
[license-url]: LICENSE-MIT
[rust-badge]: https://img.shields.io/badge/rust-edition%202024-orange.svg
[rust-url]: https://doc.rust-lang.org/edition-guide/

**Lightweight SOTA local search engine for AI agents in Rust — BM25 full-text search, vector semantic search, hybrid search with query expansion and reranking, plus an MCP server for AI tool integration.**

> This fork tracks the Rust implementation of [qntx-labs/qmd](https://github.com/qntx-labs/qmd), originally authored by pyroth.sol. Its purpose is to keep this implementation live and updated.

## Crates

| Crate | | Description |
| --- | --- | --- |
| **[`qmd`](qmd/)** | [![crates.io][qmd-crate]][qmd-crate-url] [![docs.rs][qmd-doc]][qmd-doc-url] | Core library — indexing, BM25, vector search, hybrid search, embeddings |
| **[`qmd-cli`](qmd-cli/)** | [![crates.io][cli-crate]][cli-crate-url] | CLI tool — collection management, search, RAG question answering |
| **[`qmd-mcp`](qmd-mcp/)** | [![crates.io][mcp-crate]][mcp-crate-url] | MCP server — expose qmd as an AI agent tool |

[qmd-crate]: https://img.shields.io/crates/v/qmd.svg
[qmd-crate-url]: https://crates.io/crates/qmd
[cli-crate]: https://img.shields.io/crates/v/qmd-cli.svg
[cli-crate-url]: https://crates.io/crates/qmd-cli
[mcp-crate]: https://img.shields.io/crates/v/qmd-mcp.svg
[mcp-crate-url]: https://crates.io/crates/qmd-mcp
[qmd-doc]: https://img.shields.io/docsrs/qmd.svg
[qmd-doc-url]: https://docs.rs/qmd

## Quick Start

### Install the CLI

**Shell** (macOS / Linux):

```sh
curl -fsSL https://sh.qntx.fun/qmd | sh
```

**PowerShell** (Windows):

```powershell
irm https://sh.qntx.fun/qmd/ps | iex
```

Or via Cargo:

```bash
cargo install qmd-cli
```

### CLI Usage

```bash
# Add a collection of markdown files
qmd collection add ./docs --name my-docs --mask "**/*.md"

# List collections
qmd ls

# BM25 full-text search
qmd search "query expansion" -n 5

# Vector semantic search (requires embedding model)
qmd models pull          # download default models
qmd embed                # generate embeddings
qmd vsearch "how does reranking work" -n 5

# Hybrid search (BM25 + vector + query expansion + reranking)
qmd qsearch "local search engine for AI"

# Ask a question (RAG)
qmd ask "What search algorithms does qmd support?"

# Get a specific document
qmd get qmd://my-docs/README.md

# Re-index all collections
qmd update
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.
