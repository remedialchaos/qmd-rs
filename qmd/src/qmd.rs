//! Top-level facade: [`Qmd`] owns the database and ML engines.
//!
//! ```rust,no_run
//! use qmd::{Qmd, Collection};
//!
//! let mut qmd = Qmd::open("./index.sqlite")?;
//! qmd.register_collection(&Collection::new("docs", "/path/to/docs"))?;
//! qmd.update(None)?;
//! qmd.embed()?;
//! let results = qmd.search("how does auth work?", 10)?;
//! # Ok::<(), qmd::Error>(())
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;

use ignore::WalkBuilder;

use crate::chunk::Chunker;
use crate::db::{
    Collection, CollectionInfo, Db, Document, IndexStatus, SearchResult, extract_title,
    hash_content,
};
use crate::embed::Embedder;
use crate::error::{Error, Result};
use crate::rerank::Reranker;
use crate::search::{self, Query, QueryType};

/// The main qmd handle.
///
/// Owns the SQLite database, embedding model, and reranker — all lazily
/// initialized on first use.
pub struct Qmd {
    /// SQLite database handle.
    db: Db,
    /// Lazily-loaded embedding engine.
    embedder: Option<Embedder>,
    /// Lazily-loaded reranking engine.
    reranker: Option<Reranker>,
    /// Document chunker for embedding.
    chunker: Chunker,
}

impl std::fmt::Debug for Qmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qmd")
            .field("embedder_loaded", &self.embedder.is_some())
            .field("reranker_loaded", &self.reranker.is_some())
            .finish_non_exhaustive()
    }
}

impl Qmd {
    /// Open (or create) a qmd index at the given SQLite path.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            db: Db::open(db_path.as_ref())?,
            embedder: None,
            reranker: None,
            chunker: Chunker::default(),
        })
    }

    /// Open an in-memory index (useful for tests).
    pub fn open_memory() -> Result<Self> {
        Ok(Self {
            db: Db::open_memory()?,
            embedder: None,
            reranker: None,
            chunker: Chunker::default(),
        })
    }

    /// Access the underlying database.
    #[must_use]
    pub const fn db(&self) -> &Db {
        &self.db
    }

    /// Set a custom chunker.
    pub const fn set_chunker(&mut self, chunker: Chunker) {
        self.chunker = chunker;
    }

    /// Ensure the embedder is loaded, lazily initializing on first call.
    fn ensure_embedder(&mut self) -> Result<()> {
        if self.embedder.is_none() {
            self.embedder = Some(Embedder::new()?);
        }
        Ok(())
    }

    // ── Collection management ───────────────────────────────────────────

    /// Register (or update) a collection. Does NOT index files.
    /// Call [`update`](Self::update) afterwards to scan the filesystem.
    pub fn register_collection(&self, coll: &Collection) -> Result<()> {
        let path = Path::new(&coll.path);
        if !path.is_dir() {
            return Err(Error::Config(format!("not a directory: {}", coll.path)));
        }
        self.db.upsert_collection(coll)
    }

    /// Remove a collection and all its documents.
    pub fn remove_collection(&self, name: &str) -> Result<usize> {
        self.db.delete_collection(name)
    }

    /// Rename a collection.
    pub fn rename_collection(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.db.rename_collection(old_name, new_name)
    }

    /// List all registered collections with stats.
    pub fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        Ok(self.db.status()?.collections)
    }

    // ── Context management ──────────────────────────────────────────────

    /// Set a path-scoped context for a collection.
    pub fn set_context(&self, collection: &str, path_prefix: &str, text: &str) -> Result<bool> {
        self.db.set_context(collection, path_prefix, text)
    }

    /// Remove a path-scoped context.
    pub fn remove_context(&self, collection: &str, path_prefix: &str) -> Result<bool> {
        self.db.remove_context(collection, path_prefix)
    }

    /// Set the global context (applies to all collections).
    pub fn set_global_context(&self, text: Option<&str>) -> Result<()> {
        self.db.set_global_context(text)
    }

    /// Get the global context.
    pub fn global_context(&self) -> Result<Option<String>> {
        self.db.global_context()
    }

    // ── Indexing ────────────────────────────────────────────────────────

    /// Scan registered collections and incrementally index files.
    ///
    /// If `collections` is `None`, all registered collections are scanned.
    /// Pass a slice of names to limit to specific collections.
    pub fn update(&self, collections: Option<&[&str]>) -> Result<UpdateResult> {
        let all_colls = self.db.list_collections()?;
        let colls: Vec<&Collection> = if let Some(names) = collections {
            let set: HashSet<&str> = names.iter().copied().collect();
            all_colls
                .iter()
                .filter(|c| set.contains(c.name.as_str()))
                .collect()
        } else {
            all_colls.iter().collect()
        };

        let mut total = UpdateResult::default();

        for coll in colls {
            let r = self.index_collection(coll)?;
            total.indexed += r.indexed;
            total.updated += r.updated;
            total.unchanged += r.unchanged;
            total.removed += r.removed;
            total.collections += 1;
        }

        Ok(total)
    }

    /// Index a single collection by scanning the filesystem.
    fn index_collection(&self, coll: &Collection) -> Result<IndexResult> {
        let base = Path::new(&coll.path);
        if !base.is_dir() {
            return Err(Error::Config(format!("not a directory: {}", coll.path)));
        }

        let files = walk_collection(base, &coll.pattern, &coll.ignore)?;

        let existing = self.db.active_paths(&coll.name)?;
        let existing_set: HashSet<&str> = existing.iter().map(String::as_str).collect();

        let mut indexed = 0usize;
        let mut updated = 0usize;
        let mut unchanged = 0usize;

        for file_path in &files {
            let rel = file_path
                .strip_prefix(base)
                .unwrap_or(file_path)
                .to_string_lossy()
                .replace('\\', "/");

            let Ok(content) = std::fs::read_to_string(file_path) else {
                continue;
            };
            if content.trim().is_empty() {
                continue;
            }

            let hash = hash_content(&content);
            let title = extract_title(&content, &rel);

            if let Some(existing_doc) = self.db.get_document(&coll.name, &rel)? {
                if existing_doc.hash == hash {
                    unchanged += 1;
                    continue;
                }
                updated += 1;
            } else {
                indexed += 1;
            }

            self.db.insert_content(&hash, &content)?;
            self.db.upsert_document(&coll.name, &rel, &title, &hash)?;
        }

        let new_paths: HashSet<String> = files
            .iter()
            .filter_map(|p| {
                p.strip_prefix(base)
                    .ok()
                    .map(|r| r.to_string_lossy().replace('\\', "/"))
            })
            .collect();

        let mut removed = 0usize;
        for path in &existing_set {
            if !new_paths.contains(*path) {
                self.db.deactivate(&coll.name, path)?;
                removed += 1;
            }
        }

        Ok(IndexResult {
            indexed,
            updated,
            unchanged,
            removed,
        })
    }

    // ── Embedding ───────────────────────────────────────────────────────

    /// Generate embeddings for all documents that need them.
    pub fn embed(&mut self) -> Result<EmbedResult> {
        self.embed_with_batch(None)
    }

    /// Generate embeddings for up to `batch` documents.
    pub fn embed_with_batch(&mut self, batch: Option<usize>) -> Result<EmbedResult> {
        let docs = self.db.unembedded_docs_with_limit(batch)?;
        if docs.is_empty() {
            return Ok(EmbedResult {
                embedded: 0,
                chunks: 0,
                remaining: self.db.needs_embedding_count()?,
                failures: 0,
                failure_messages: Vec::new(),
            });
        }

        self.ensure_embedder()?;

        let mut total_chunks = 0usize;
        let mut embedded = 0usize;
        let mut failures = 0usize;
        let mut failure_messages = Vec::new();
        for (hash, path, body) in &docs {
            let chunks = self.chunker.split(body);
            let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();

            let embed_result = self
                .embedder
                .as_mut()
                .unwrap_or_else(|| unreachable!())
                .embed_documents(&texts);
            let Ok(embeddings) = embed_result else {
                failures += 1;
                failure_messages.push(format!("failed: {path}"));
                continue;
            };

            let mut write_result = Ok(());
            for (seq, (chunk, emb)) in chunks.iter().zip(embeddings.iter()).enumerate().skip(1) {
                if let Err(e) = self
                    .db
                    .insert_embedding(hash, seq, chunk.pos, emb, "default")
                {
                    write_result = Err(e);
                    break;
                }
            }
            if write_result.is_ok()
                && let Some((chunk, emb)) = chunks.first().zip(embeddings.first())
            {
                write_result =
                    self.db
                        .insert_embedding(hash, 0, chunk.pos, emb, "default:complete");
            }
            if let Err(e) = write_result {
                failures += 1;
                failure_messages.push(format!("failed: {path}: {e}"));
                continue;
            }
            embedded += 1;
            total_chunks += chunks.len();
        }

        Ok(EmbedResult {
            embedded,
            chunks: total_chunks,
            remaining: self.db.needs_embedding_count()?,
            failures,
            failure_messages,
        })
    }

    // ── Search ──────────────────────────────────────────────────────────

    /// Full-text search (BM25 only, no ML).
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_fts_with_offset(query, limit, 0)
    }

    /// Full-text search with offset pagination.
    pub fn search_fts_with_offset(
        &self,
        query: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SearchResult>> {
        let fts_query = search::build_fts5_query(query).unwrap_or_else(|| query.to_string());
        self.db
            .search_fts_with_offset(&fts_query, limit, offset, None)
    }

    /// Vector similarity search.
    pub fn search_vec(&mut self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.ensure_embedder()?;
        let embedder = self.embedder.as_mut().unwrap_or_else(|| unreachable!());
        let emb = embedder.embed_query(query)?;
        self.db.search_vec(&emb, limit, None)
    }

    /// Hybrid search: FTS + vector + RRF fusion + optional reranking.
    ///
    /// This is the recommended search method for best quality.
    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_with_offset(query, limit, 0)
    }

    /// Hybrid search with offset pagination.
    pub fn search_with_offset(
        &mut self,
        query: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SearchResult>> {
        let queries = Query::expand_simple(query);
        self.search_with_queries_with_offset(query, &queries, limit, offset)
    }

    /// Search with pre-expanded queries (for external LLM integration).
    pub fn search_with_queries(
        &mut self,
        query: &str,
        queries: &[Query],
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_queries_with_offset(query, queries, limit, 0)
    }

    /// Search with pre-expanded queries and offset pagination.
    pub fn search_with_queries_with_offset(
        &mut self,
        query: &str,
        queries: &[Query],
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SearchResult>> {
        let requested = limit.saturating_add(offset);
        let fetch_limit = requested.saturating_mul(3);

        let mut all_lists: Vec<Vec<String>> = Vec::new();
        let mut all_weights: Vec<f64> = Vec::new();
        let mut result_map: HashMap<String, SearchResult> = HashMap::new();

        if let Some(fts_q) = search::build_fts5_query(query)
            && let Ok(hits) = self.db.search_fts(&fts_q, fetch_limit, None)
        {
            collect_results(&mut all_lists, &mut all_weights, &mut result_map, hits, 1.0);
        }

        self.ensure_embedder()?;

        for q in queries {
            match q.kind {
                QueryType::Lex => {
                    if let Some(fts_q) = search::build_fts5_query(&q.text)
                        && let Ok(hits) = self.db.search_fts(&fts_q, fetch_limit, None)
                    {
                        collect_results(
                            &mut all_lists,
                            &mut all_weights,
                            &mut result_map,
                            hits,
                            0.8,
                        );
                    }
                }
                QueryType::Vec | QueryType::Hyde => {
                    let embedder = self.embedder.as_mut().unwrap_or_else(|| unreachable!());
                    if let Ok(emb) = embedder.embed_query(&q.text)
                        && let Ok(hits) = self.db.search_vec(&emb, fetch_limit, None)
                    {
                        collect_results(
                            &mut all_lists,
                            &mut all_weights,
                            &mut result_map,
                            hits,
                            1.0,
                        );
                    }
                }
            }
        }

        let list_refs: Vec<&[String]> = all_lists.iter().map(Vec::as_slice).collect();
        let fused = search::rrf(&list_refs, Some(&all_weights), 60);

        let mut results: Vec<SearchResult> = fused
            .iter()
            .filter_map(|hit| {
                result_map.remove(&hit.key).map(|mut r| {
                    r.score = hit.score;
                    r
                })
            })
            .collect();

        if results.len() > 1 {
            self.apply_reranking(query, &mut results, requested);
        }

        if offset >= results.len() {
            return Ok(Vec::new());
        }
        results.drain(..offset);
        results.truncate(limit);
        Ok(results)
    }

    /// Apply cross-encoder reranking to top results.
    fn apply_reranking(&mut self, query: &str, results: &mut Vec<SearchResult>, limit: usize) {
        if self.reranker.is_none() {
            match Reranker::new() {
                Ok(r) => self.reranker = Some(r),
                Err(_) => return,
            }
        }

        let top_n = results.len().min(limit * 2);
        let snippets: Vec<String> = results[..top_n]
            .iter()
            .filter_map(|r| {
                self.db
                    .get_body(&r.doc.hash)
                    .ok()?
                    .map(|body| search::extract_snippet(&body, query, 4000))
            })
            .collect();

        if snippets.is_empty() {
            return;
        }

        let doc_refs: Vec<&str> = snippets.iter().map(String::as_str).collect();
        let reranker = self.reranker.as_mut().unwrap_or_else(|| unreachable!());
        let Ok(scored) = reranker.rerank(query, &doc_refs, top_n) else {
            return;
        };

        let mut reranked: Vec<SearchResult> = Vec::with_capacity(scored.len());
        let mut used: HashSet<usize> = HashSet::new();
        for s in &scored {
            if s.index < results.len() {
                let mut r = results[s.index].clone();
                r.score = f64::from(s.score);
                reranked.push(r);
                used.insert(s.index);
            }
        }
        for (i, r) in results.iter().enumerate() {
            if !used.contains(&i) {
                reranked.push(r.clone());
            }
        }
        *results = reranked;
    }

    // ── Document retrieval ──────────────────────────────────────────────

    /// Get a document by `collection/path` or by docid (`#abc123`).
    pub fn get(&self, path_or_docid: &str) -> Result<Document> {
        let clean = path_or_docid.trim_start_matches('#');
        if clean.len() == 6 && clean.chars().all(|c| c.is_ascii_hexdigit()) {
            if let Some((coll, path)) = self.db.find_by_docid(clean)? {
                return self
                    .db
                    .get_document(&coll, &path)?
                    .ok_or_else(|| Error::NotFound(path_or_docid.to_string()));
            }
        } else if let Some((coll, path)) = path_or_docid.split_once('/') {
            return self
                .db
                .get_document(coll, path)?
                .ok_or_else(|| Error::NotFound(path_or_docid.to_string()));
        }
        Err(Error::NotFound(path_or_docid.to_string()))
    }

    // ── Index health ────────────────────────────────────────────────────

    /// Get full index status.
    pub fn status(&self) -> Result<IndexStatus> {
        self.db.status()
    }

    /// Count active documents.
    pub fn doc_count(&self) -> Result<usize> {
        self.db.doc_count()
    }

    /// Count documents needing embedding.
    pub fn needs_embedding(&self) -> Result<usize> {
        self.db.needs_embedding_count()
    }

    // ── Maintenance ─────────────────────────────────────────────────────

    /// Delete inactive documents and orphaned data.
    pub fn cleanup(&self) -> Result<usize> {
        self.db.cleanup()
    }

    /// Clear all embeddings (forces re-embedding).
    pub fn clear_embeddings(&mut self) -> Result<usize> {
        self.db.clear_embeddings()
    }

    /// Vacuum the database.
    pub fn vacuum(&self) -> Result<()> {
        self.db.vacuum()
    }
}

/// Walk a collection directory using the `ignore` crate (gitignore-aware).
fn walk_collection(
    base: &Path,
    pattern: &str,
    ignore_patterns: &[String],
) -> Result<Vec<std::path::PathBuf>> {
    let mut builder = ignore::overrides::OverrideBuilder::new(base);
    builder
        .add(pattern)
        .map_err(|e| Error::Config(e.to_string()))?;

    for pat in ignore_patterns {
        let negated = format!("!{pat}");
        builder
            .add(&negated)
            .map_err(|e| Error::Config(e.to_string()))?;
    }

    for dir in EXCLUDE_DIRS {
        let neg = format!("!{dir}/");
        builder
            .add(&neg)
            .map_err(|e| Error::Config(e.to_string()))?;
    }

    let overrides = builder.build().map_err(|e| Error::Config(e.to_string()))?;

    let mut files = Vec::new();
    let walker = WalkBuilder::new(base)
        .overrides(overrides)
        .hidden(true)
        .git_ignore(true)
        .build();

    for dir_entry in walker {
        let entry = dir_entry.map_err(|e| Error::Config(e.to_string()))?;
        if entry.file_type().is_some_and(|ft| ft.is_file()) {
            files.push(entry.into_path());
        }
    }

    Ok(files)
}

/// Collect search results into shared lists for RRF fusion.
fn collect_results(
    lists: &mut Vec<Vec<String>>,
    weights: &mut Vec<f64>,
    map: &mut HashMap<String, SearchResult>,
    results: Vec<SearchResult>,
    weight: f64,
) {
    let keys: Vec<String> = results.iter().map(|r| r.doc.display_path()).collect();
    for r in results {
        map.entry(r.doc.display_path()).or_insert(r);
    }
    if !keys.is_empty() {
        lists.push(keys);
        weights.push(weight);
    }
}

/// Result of an [`update`](Qmd::update) operation across collections.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct UpdateResult {
    /// Number of collections processed.
    pub collections: usize,
    /// Newly indexed documents.
    pub indexed: usize,
    /// Updated documents (content changed).
    pub updated: usize,
    /// Unchanged documents.
    pub unchanged: usize,
    /// Removed (deactivated) documents.
    pub removed: usize,
}

/// Result of indexing a single collection.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct IndexResult {
    /// Newly indexed documents.
    pub indexed: usize,
    /// Updated documents (content changed).
    pub updated: usize,
    /// Unchanged documents.
    pub unchanged: usize,
    /// Removed (deactivated) documents.
    pub removed: usize,
}

/// Result of an embedding operation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmbedResult {
    /// Documents embedded.
    pub embedded: usize,
    /// Total chunks processed.
    pub chunks: usize,
    /// Documents still needing embedding after this run.
    pub remaining: usize,
    /// Documents that failed while embedding.
    pub failures: usize,
    /// Failure diagnostics for the CLI to display.
    pub failure_messages: Vec<String>,
}

/// Directories excluded from indexing.
const EXCLUDE_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    ".cache",
    "vendor",
    "dist",
    "build",
    "target",
];
