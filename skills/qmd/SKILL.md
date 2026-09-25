---
name: qmd
description: Search local markdown knowledge bases, notes, docs, and wikis with QMD. Use when users ask to find notes, retrieve documents, inspect a wiki, answer from indexed markdown, or set up QMD access.
license: MIT
compatibility: Requires native Rust `qmd` CLI (qmd-rs). Available at `~/.local/bin/qmd` or via `cargo install qmd-rs-cli`.
metadata:
  author: tobi
  adapted-by: remedialchaos (qmd-rs)
  version: "0.5.0"
allowed-tools: Bash(qmd:*)
---

# QMD - Query Markdown Documents

## How search works

QMD searches local markdown collections: notes, docs, wikis, transcripts, and
project knowledge bases. Use it before web search when the answer may already be
in indexed local files.

The workflow is always:

1. Search for candidate documents using `qmd search` (BM25 keyword), `qmd vsearch` (vector semantic), or `qmd query` (hybrid reranked).
2. Retrieve the source with `qmd get` (or line slicing `qmd get <path>:<from>:<lines>`).
3. Answer from retrieved text, citing paths or docids.

Do not answer from snippets alone when the user needs facts, decisions, quotes,
or nuance. Snippets are only leads.

Typical loop:

```bash
# 1. Search for candidates
qmd search "merchant reality support interviews" -n 5
# or hybrid search with cross-encoder reranking:
qmd query "merchant reality support interviews" -n 5

# leads: #abc123 wiki/concepts/customer-proximity.md; #def432 wiki/sources/merchant-call.md

# 2. Retrieve full document source or sliced window
qmd get "#abc123"
qmd get "wiki/concepts/customer-proximity.md:1:30" --line-numbers
```

When reporting what you retrieved, a compact note is enough; do not paste whole
files unless needed:

```text
Retrieved:
- #abc123 wiki/concepts/customer-proximity.md
- #def432 wiki/sources/merchant-call.md
```

## Pick the right search mode

### 1. BM25 Lexical Keyword Search (`qmd search`)
Use **BM25 lexical search** (`qmd search`, alias `qmd fts`) when you know exact words, titles, names, code
symbols, error strings, or rare phrases. It is instant and requires no model or vector state:

```bash
qmd search "cockpit OKR Goodhart" -n 10
qmd search '"AI Before Headcount"' -c wiki -n 5
qmd search "struct ConversationMetrics" --json
qmd search "auth token" --files
```

### 2. Semantic Vector Search (`qmd vsearch`)
Use **vector search** (`qmd vsearch`, alias `vector-search`) for pure semantic similarity when looking for
conceptually related notes without requiring exact keyword overlap:

```bash
qmd vsearch "how we think about user research" -n 5
qmd vsearch "protecting client from excessive background polling" -c wiki -n 5
```

### 3. Hybrid Semantic Search & Reranking (`qmd query`)
Use **hybrid search** (`qmd query`, alias `deep-search`) for complex questions. It expands the query,
merges BM25 lexical and vector candidates via Reciprocal Rank Fusion (RRF), and applies cross-encoder reranking:

```bash
qmd query "metrics as instruments without letting OKRs replace judgment" -n 5
qmd query "customer proximity through support and interviews" -c wiki -n 5
qmd query "prompt cache ratio calculation" --min-score 0.5
qmd query "quick search without cross-encoder" --no-rerank
```

### 4. Structured Output & Flags
All search commands support:
- `--json`: Machine-readable output with `docid`, `collection`, `path`, `title`, and `score`.
- `--files`: Output matching file paths only (one per line, or JSON array with `--json`).
- `-a` / `--all`: Return all matching results without default limit capping.
- `--min-score <SCORE>`: Filter out lower-relevance results below score threshold.

## Retrieve sources (`qmd get` & `qmd multi-get`)

Search results include docids like `#abc123` and collection paths like `wiki/notes.md`. Fetch them:

```bash
# Full document
qmd get "#abc123"
qmd get "wiki/concepts/customer-proximity.md"

# Line slicing (start line 1, 30 lines) with line numbers
qmd get "wiki/concepts/customer-proximity.md:1:30" --line-numbers
qmd get "wiki/concepts/customer-proximity.md" --from 20 -l 15

# Batch retrieval with glob pattern or comma-separated list
qmd multi-get "wiki/concepts/customer*.md" -l 20
qmd multi-get "#abc123,#def432" --json
```

When citing sources in your response:
- Cite the collection path and `#docid` (e.g. `wiki/concepts/customer-proximity.md (#abc123)`).
- Provide verbatim excerpts relevant to the user's question.

## Discover what is indexed

Inspect registered collections, document listings, and index status:

```bash
# List collections or documents inside a collection subpath
qmd ls
qmd ls wiki
qmd ls wiki/concepts

qmd collection list
qmd status
qmd doctor
```

Add collection filters with `-c` or `--collection` when broad searches drift into the wrong corpus:

```bash
qmd search "headcount autonomous agents" -c wiki -n 10
qmd query "service cutover procedure" --collection wiki -n 5
```

Omit `-c` to search across all registered collections.

## Query craft

Good QMD searches combine:

1. **Title and alias anchors:** exact page titles, named entities, identifiers (`qmd search`).
2. **Semantic paraphrase:** how a human or documentation author describes the concept (`qmd query`).
3. **Collection targeting:** scoping to the right collection with `-c <collection>`.

Examples:

```bash
# Exact title lookup
qmd search '"arm the rebels" merchants tools' -c wiki

# Semantic concept lookup
qmd query "founder stays close to user reality through support channels" -c wiki

# Source / transcript lookup
qmd search "WhatsApp cadence Shawn Ryan" -c wiki -n 10
```

## Setup and maintenance

Only mutate indexes when the user asks for setup, indexing, or maintenance.
Searching and retrieving are read-only and safe.

```bash
# Register a collection of markdown files
qmd collection add ~/notes --name my-notes --pattern "**/*.md"

# Re-index all (or specific) collections
qmd update
qmd update -c my-notes

# Generate vector embeddings
qmd embed
qmd embed -c my-notes     # scope embedding generation to specific collection
qmd embed --batch 500     # cap documents per run
qmd embed --force         # clear existing embeddings and rebuild

# Clean up inactive documents and orphaned rows
qmd cleanup

# Reclaim space
qmd vacuum
```

### Health and diagnostics (`qmd doctor`)

`qmd doctor` checks SQLite integrity, FTS document synchronization, orphan rows/vectors,
collection path accessibility, and embedding completeness/fingerprints:

```bash
qmd doctor
qmd doctor --json
```

If hybrid search fails or warns of legacy embeddings, run `qmd doctor` to inspect the
status, followed by `qmd cleanup` and `qmd embed --force`.

## Pitfalls

- **Do not stop at snippets.** Fetch documents with `qmd get` before making factual claims or decisions.
- **Choose the right command:**
    - Use `qmd search` for exact technical terms, error logs, code symbols, and specific phrases (BM25 keyword).
    - Use `qmd vsearch` for conceptual similarity without requiring keyword matches (dense vector).
    - Use `qmd query` for complex questions requiring query expansion and cross-encoder reranking.
- **Do not mutate indexes casually.** `qmd collection add`, `qmd update`, `qmd embed`, and `qmd cleanup` change local state and can be CPU/GPU intensive.
- **Collection names matter.** Specify `-c <name>` to isolate search when multiple collections exist.
- **Inspect diagnostics when needed.** If vector search warns of fingerprint mismatches, verify with `qmd doctor` before running `qmd embed --force`.
