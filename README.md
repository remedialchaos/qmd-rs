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
| **[`qmd-rs`](qmd-rs/)** | [![crates.io][qmd-rs-crate]][qmd-rs-crate-url] [![docs.rs][qmd-rs-doc]][qmd-rs-doc-url] | Core library — indexing, BM25, vector search, hybrid search, embeddings |
| **[`qmd-rs-cli`](qmd-rs-cli/)** | [![crates.io][qmd-rs-cli-crate]][qmd-rs-cli-crate-url] | CLI tool — collection management, indexing, and search |

[qmd-rs-crate]: https://img.shields.io/crates/v/qmd-rs.svg
[qmd-rs-crate-url]: https://crates.io/crates/qmd-rs
[qmd-rs-cli-crate]: https://img.shields.io/crates/v/qmd-rs-cli.svg
[qmd-rs-cli-crate-url]: https://crates.io/crates/qmd-rs-cli
[qmd-rs-doc]: https://img.shields.io/docsrs/qmd-rs.svg
[qmd-rs-doc-url]: https://docs.rs/qmd-rs

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
cargo install qmd-rs-cli
```

### CLI Usage

```bash
# Register a collection of markdown files
qmd collection add ~/notes --name my-docs --pattern "**/*.md"
# Alias: --mask is also supported
qmd collection add ~/work/docs --name docs --mask "**/*.md"

# List registered collections
qmd collection list

# Re-index all (or specific) collections
qmd update
qmd update -c my-docs

# Generate vector embeddings for unembedded documents
# (the first run downloads the default embedding model)
qmd embed
qmd embed -c notes         # scope embedding generation to a collection
qmd embed --batch 500      # cap documents per run
qmd embed --force          # clear existing embeddings and rebuild

# Fast keyword search (BM25 full-text search)
qmd search "query expansion" -n 5
qmd search "authentication" --files            # output matching paths only
qmd search "error handling" -a --min-score 0.3 # return all matches above threshold

# Semantic vector similarity search
qmd vsearch "how to deploy locally" -n 5

# Hybrid search (BM25 + vector + RRF + rerank)
qmd query "quarterly planning process"
qmd query "api reference" --no-rerank         # bypass cross-encoder reranking
qmd query "auth tokens" --provider ollama     # LLM query expansion

# Expand search queries into lexical, vector, and HyDE sub-queries
qmd expand "authentication mechanisms"
qmd expand "auth" --json

# List collections and documents
qmd ls
qmd ls my-docs
qmd ls my-docs/api

# Get a document by collection/path or by #docid
qmd get my-docs/meeting-notes.md
qmd get "#a1b2c3"
qmd get "my-docs/notes.md:10:30" --line-numbers  # slice lines 10-39 with line numbering

# Batch retrieve documents by glob or comma-separated list
qmd multi-get "my-docs/meeting*.md" -l 20        # retrieve first 20 lines of each match
qmd multi-get "my-docs/a.md, my-docs/b.md" --max-bytes 20480

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

All commands operate on a SQLite index database (default: `$XDG_DATA_HOME/qmd/index.sqlite` or `~/.local/share/qmd/index.sqlite`). Use `--index <PATH>` or the `QMD_INDEX` environment variable to work against a specific index file (for example, for isolated agent or project indexes):

```bash
# Via command-line argument
qmd --index ~/indexes/docs.sqlite status

# Or via environment variable (ideal for per-agent or per-workspace routing)
export QMD_INDEX=~/.gemini/antigravity-cli/qmd.sqlite
qmd status
```

### Model Context Protocol (MCP) Server

`qmd` includes a built-in JSON-RPC 2.0 stdio MCP server for agentic workflows (compatible with Claude Desktop, Claude Code, and other MCP clients):

```bash
qmd mcp
```

**Claude Desktop Configuration** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
```

**Tools Exposed:**
- `query` — Hybrid search with sub-queries/pre-expansion, RRF fusion, and optional reranking
- `get` — Retrieve document content or sliced lines by path, docid, or `:from:count`
- `multi_get` — Batch retrieve documents by glob pattern or comma-separated list
- `search` — Fast BM25 keyword search
- `vsearch` — Semantic vector similarity search
- `status` — Index health, document counts, and collection statistics

## Retrieval-quality regression gate

The repository includes a small checked-in corpus and relevance judgments. The
gate exercises real FTS indexing, computes MRR, Recall@3, and nDCG@3, and checks
deterministic fusion without downloading a model:

```bash
cargo test -p qmd-rs --test retrieval_quality
```

## Reliability improvements

Recent releases include safeguards that keep indexing and retrieval predictable:

- Search snippets are Unicode-safe and never exceed the requested character bound, including when a match is surrounded by multibyte text.
- Full-text queries preserve the tokenizer's punctuation boundaries, so identifiers and paths keep their searchable terms; punctuation-only queries produce no direct FTS hits.
- During `qmd update`, a tracked Markdown file that becomes empty or whitespace-only is deactivated rather than left searchable as stale content.
- Vector candidate selection returns each active document at most once, even when a document has multiple matching chunks, and reciprocal-rank fusion suppresses duplicate votes within each ranked list.
- Embedding work runs in bounded groups and supports collection scoping (`-c`/`--collection`) and a per-run `--batch` cap. Completed groups are published independently, while embedding failures are reported by the CLI and make the command fail instead of being treated as success.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.
