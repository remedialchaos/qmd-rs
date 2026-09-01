# qmd

[![CI][ci-badge]][ci-url]
[![License][license-badge]][license-url]
[![Rust][rust-badge]][rust-url]

[ci-badge]: https://github.com/remedialchaos/qmd-rs/actions/workflows/rust.yml/badge.svg
[ci-url]: https://github.com/remedialchaos/qmd-rs/actions/workflows/rust.yml
[license-badge]: https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg
[license-url]: LICENSE-MIT
[rust-badge]: https://img.shields.io/badge/rust-edition%202024-orange.svg
[rust-url]: https://doc.rust-lang.org/edition-guide/

**qmd is a fast local search engine for markdown documents, written in Rust — BM25 full-text search, vector semantic search, and hybrid search with reranking, all indexed in a single local SQLite database.**

> qmd is a fork of [qntx-labs/qmd](https://github.com/qntx-labs/qmd), originally authored by pyroth.sol. This repository tracks the upstream implementation to keep it live and updated.

## Crates

| Crate | | Description |
| --- | --- | --- |
| **[`qmd`](qmd/)** | [![crates.io][qmd-crate]][qmd-crate-url] [![docs.rs][qmd-doc]][qmd-doc-url] | Core library — indexing, BM25, vector search, hybrid search, embeddings |
| **[`qmd-cli`](qmd-cli/)** | [![crates.io][cli-crate]][cli-crate-url] | CLI tool — collection management, indexing, and search |

[qmd-crate]: https://img.shields.io/crates/v/qmd.svg
[qmd-crate-url]: https://crates.io/crates/qmd
[cli-crate]: https://img.shields.io/crates/v/qmd-cli.svg
[cli-crate-url]: https://crates.io/crates/qmd-cli
[qmd-doc]: https://img.shields.io/docsrs/qmd.svg
[qmd-doc-url]: https://docs.rs/qmd

## Quick Start

### Install the CLI

**Shell** (macOS / Linux):

```sh
curl -fsSL https://raw.githubusercontent.com/remedialchaos/qmd-rs/main/install.sh | sh
```

**PowerShell** (Windows):

```powershell
irm https://raw.githubusercontent.com/remedialchaos/qmd-rs/main/install.ps1 | iex
```

Or via Cargo:

```bash
cargo install qmd-cli
```

### CLI Usage

```bash
# Register a collection of markdown files
qmd collection add ~/notes --name my-docs --pattern "**/*.md"

# List registered collections
qmd collection list

# Re-index all (or specific) collections
qmd update
qmd update -c my-docs

# Generate vector embeddings for unembedded documents
# (the first run downloads the default embedding model)
qmd embed
qmd embed --batch 500     # cap documents per run
qmd embed --force         # clear existing embeddings and rebuild

# BM25 full-text search
qmd fts "query expansion" -n 5

# Hybrid search (BM25 + vector + rerank)
qmd search "local search engine for AI"

# Page through results, or emit JSON
qmd search "indexing" --offset 10
qmd fts "bm25" --json
qmd search "indexing" --collection my-docs

# Get a document by collection/path or by #docid
qmd get my-docs/meeting-notes.md
qmd get "#a1b2c3"

# Show index status
qmd status

# Run read-only index diagnostics (also available as JSON)
qmd doctor
qmd doctor --json

# Attach context text to a collection path or globally
qmd context add my-docs /api "Internal API documentation"
qmd context list
qmd context rm my-docs /api
qmd context global "Team wiki conventions"

# Clean up inactive documents and orphaned data
qmd cleanup

# Vacuum the database to reclaim space
qmd vacuum
```

All commands operate on a single SQLite index database. Use `--index <PATH>` to
work against a specific index file instead of the default location:

```bash
qmd --index ~/indexes/docs.db status
```

## Retrieval-quality regression gate

The repository includes a small checked-in corpus and relevance judgments. The
gate exercises real FTS indexing, computes MRR, Recall@3, and nDCG@3, and checks
deterministic fusion without downloading a model:

```bash
cargo test -p qmd --test retrieval_quality
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.
