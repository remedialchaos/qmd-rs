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

1. Search for candidate documents using `qmd fts` (keyword) or `qmd search` (hybrid).
2. Retrieve the full source with `qmd get`.
3. Answer from retrieved text, citing paths or docids.

Do not answer from snippets alone when the user needs facts, decisions, quotes,
or nuance. Snippets are only leads.

Typical loop:

```bash
# 1. Search for candidates
qmd fts "merchant reality support interviews" -n 5
# or hybrid search:
qmd search "merchant reality support interviews" -n 5

# leads: #abc123 wiki/concepts/customer-proximity.md; #def432 wiki/sources/merchant-call.md

# 2. Retrieve full document source
qmd get "#abc123"
```

When reporting what you retrieved, a compact note is enough; do not paste whole
files unless needed:

```text
Retrieved:
- #abc123 wiki/concepts/customer-proximity.md
- #def432 wiki/sources/merchant-call.md
```

## Pick the right search mode

### 1. BM25 Lexical Keyword Search (`qmd fts`)
Use **BM25 lexical search** when you know exact words, titles, names, code
symbols, error strings, or rare phrases. It is instant and requires no model or vector state:

```bash
qmd fts "cockpit OKR Goodhart" -n 10
qmd fts '"AI Before Headcount"' -c wiki -n 5
qmd fts "struct ConversationMetrics" --json
```

### 2. Hybrid Semantic Search (`qmd search`)
Use **hybrid search** when the user describes an idea indirectly, uses different
wording than the source, or needs conceptual recall. It combines BM25 full-text matching,
vector semantic similarity, Reciprocal Rank Fusion (RRF), and local reranking:

```bash
qmd search "metrics as instruments without letting OKRs replace judgment" -n 5
qmd search "customer proximity through support and interviews" -c wiki -n 5
qmd search "how does prompt cache ratio get computed" --json
```

If vector search is unavailable or reports legacy fingerprints, use `qmd fts` with
targeted keywords.

### 3. Structured JSON for Agent Steps
Both `fts` and `search` support `--json` for structured parsing:

```bash
qmd fts "matrix bridge" --json
qmd search "context compactor" --json -n 5
```

JSON output includes `docid`, `collection`, `path`, `title`, and `score`.

## Retrieve sources (`qmd get`)

Search results include docids like `#abc123` and collection paths like `wiki/notes.md`. Fetch them:

```bash
qmd get "#abc123"
qmd get "wiki/concepts/customer-proximity.md"
qmd get "#abc123" --json
```

`qmd get` outputs the full document content. When citing sources in your response:
- Cite the collection path and `#docid` (e.g. `wiki/concepts/customer-proximity.md (#abc123)`).
- Provide verbatim excerpts relevant to the user's question.

## Discover what is indexed

Inspect registered collections and index status before searching unfamiliar environments:

```bash
qmd collection list
qmd status
qmd doctor
```

Add collection filters with `-c` or `--collection` when broad searches drift into the wrong corpus:

```bash
qmd fts "headcount autonomous agents" -c wiki -n 10
qmd search "service cutover procedure" --collection wiki -n 5
```

Omit `-c` to search across all registered collections.

## Query craft

Good QMD searches combine:

1. **Title and alias anchors:** exact page titles, named entities, identifiers (`qmd fts`).
2. **Semantic paraphrase:** how a human or documentation author describes the concept (`qmd search`).
3. **Collection targeting:** scoping to the right collection with `-c <collection>`.

Examples:

```bash
# Exact title lookup
qmd fts '"arm the rebels" merchants tools' -c wiki

# Semantic concept lookup
qmd search "founder stays close to user reality through support channels" -c wiki

# Source / transcript lookup
qmd fts "WhatsApp cadence Shawn Ryan" -c wiki -n 10
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
    - Use `qmd fts` for exact technical terms, error logs, code symbols, and specific phrases.
    - Use `qmd search` for natural language questions and conceptual exploration.
- **Do not mutate indexes casually.** `qmd collection add`, `qmd update`, `qmd embed`, and `qmd cleanup` change local state and can be CPU/GPU intensive.
- **Collection names matter.** Specify `-c <name>` to isolate search when multiple collections exist.
- **Inspect diagnostics when needed.** If vector search warns of fingerprint mismatches, verify with `qmd doctor` before running `qmd embed --force`.
