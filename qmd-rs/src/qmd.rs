//! Top-level facade: [`Qmd`] owns the database and ML engines.
//!
//! ```rust,no_run
//! use qmd_rs::{Qmd, Collection};
//!
//! let mut qmd = Qmd::open("./index.sqlite")?;
//! qmd.register_collection(&Collection::new("docs", "/path/to/docs"))?;
//! qmd.update(None)?;
//! qmd.embed()?;
//! let results = qmd.search("how does auth work?", 10)?;
//! # Ok::<(), qmd_rs::Error>(())
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;

use ignore::WalkBuilder;
use ignore::gitignore::GitignoreBuilder;

use crate::chunk::Chunker;
use crate::db::{
    Collection, CollectionInfo, Db, DoctorReport, Document, IndexStatus, SearchResult,
    extract_title, hash_content,
};
use crate::embed::{Embedder, EmbeddingEngine, default_embedding_fingerprint_with_chunker};
use crate::error::{Error, Result};
use crate::rerank::{Reranker, Scored};
use crate::search::{self, Query, QueryType};

/// The main qmd handle.
///
/// Owns the SQLite database, embedding model, and reranker — all lazily
/// initialized on first use.
pub struct Qmd {
    /// SQLite database handle.
    db: Db,
    /// Lazily-loaded embedding engine.
    embedder: Option<Box<dyn EmbeddingEngine>>,
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

/// Default number of unique content hashes processed per embedding work group.
///
/// A default embedding run streams the unembedded corpus through groups of this
/// size, so it never retains a whole corpus of documents or all generated
/// vectors at once.
pub const DEFAULT_EMBED_GROUP_SIZE: usize = 32;

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

    /// Return the fingerprint for this Qmd handle's active embedding contract.
    fn current_embedding_fingerprint(&self) -> String {
        default_embedding_fingerprint_with_chunker(384, self.chunker)
    }

    /// Ensure the embedder is loaded, lazily initializing on first call.
    fn ensure_embedder(&mut self) -> Result<()> {
        if self.embedder.is_none() {
            self.embedder = Some(Box::new(Embedder::new()?));
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
            total.failures.extend(r.failures);
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
        let mut failures = Vec::new();

        // Paths still present on disk after this scan. A path whose source is
        // empty/whitespace-only is removed from this set so the trailing
        // deactivation pass can retire its previously active document.
        let mut new_paths: HashSet<String> = files
            .iter()
            .filter_map(|p| {
                p.strip_prefix(base)
                    .ok()
                    .map(|r| r.to_string_lossy().replace('\\', "/"))
            })
            .collect();

        for file_path in &files {
            let rel = file_path
                .strip_prefix(base)
                .unwrap_or(file_path)
                .to_string_lossy()
                .replace('\\', "/");

            let content = match std::fs::read_to_string(file_path) {
                Ok(content) => content,
                Err(error) => {
                    failures.push(IndexFailure {
                        path: file_path.display().to_string(),
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            if content.trim().is_empty() {
                new_paths.remove(&rel);
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
            failures,
        })
    }

    // ── Embedding ───────────────────────────────────────────────────────

    /// Generate embeddings for all documents that need them.
    ///
    /// Work streams through bounded groups of [`DEFAULT_EMBED_GROUP_SIZE`]
    /// unique content hashes; each group is generated outside any write
    /// transaction and published only for the documents that completed.
    pub fn embed(&mut self) -> Result<EmbedResult> {
        self.embed_with_batch(None)
    }

    /// Generate embeddings for unembedded documents.
    ///
    /// A default run streams the whole unembedded corpus through bounded groups
    /// of [`DEFAULT_EMBED_GROUP_SIZE`] unique content hashes. `batch` caps the
    /// total number of documents embedded by this invocation while groups stay
    /// bounded. Each group is fetched, generated, and published independently,
    /// so completed work survives an unrelated failure and no whole-corpus or
    /// whole-vector collection is retained in memory.
    pub fn embed_with_batch(&mut self, batch: Option<usize>) -> Result<EmbedResult> {
        let fingerprint = self.current_embedding_fingerprint();
        self.db.validate_embedding_fingerprint(&fingerprint)?;

        let mut budget = batch;
        let mut cursor: i64 = 0;
        let mut embedded = 0usize;
        let mut total_chunks = 0usize;
        let mut failures = 0usize;
        let mut failure_messages: Vec<String> = Vec::new();
        let mut engine_ready = false;

        loop {
            let request = budget.map_or(DEFAULT_EMBED_GROUP_SIZE, |left| {
                left.min(DEFAULT_EMBED_GROUP_SIZE)
            });
            if request == 0 {
                break;
            }
            let group = self.db.unembedded_doc_group(request, cursor)?;
            let Some(last) = group.last() else {
                break;
            };
            cursor = last.3;
            if let Some(left) = budget.as_mut() {
                *left -= group.len();
            }
            if !engine_ready {
                self.ensure_embedder()?;
                engine_ready = true;
            }

            // Generate every document's vectors before touching the database.
            let mut pending: Vec<(&str, usize, usize, Vec<f32>)> = Vec::new();
            for (hash, alias, body, _) in &group {
                let chunks = self.chunker.split(body);
                let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
                let embed_result = self
                    .embedder
                    .as_mut()
                    .unwrap_or_else(|| unreachable!())
                    .embed_documents(&texts);
                match embed_result {
                    Ok(embeddings) if embeddings.len() == chunks.len() => {
                        for (seq, (chunk, emb)) in chunks.iter().zip(embeddings).enumerate() {
                            pending.push((hash.as_str(), seq, chunk.pos, emb));
                        }
                        embedded += 1;
                        total_chunks += chunks.len();
                    }
                    Ok(_) => {
                        failures += 1;
                        failure_messages.push(format!(
                            "failed: {alias} (hash {hash}): embedded chunk count mismatch"
                        ));
                    }
                    Err(error) => {
                        failures += 1;
                        failure_messages.push(format!("failed: {alias} (hash {hash}): {error}"));
                    }
                }
            }

            // Publish only this group's complete, successful documents.
            if !pending.is_empty() {
                self.db
                    .replace_embeddings_transactionally(&fingerprint, &pending)?;
            }
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
        self.search_fts_with_offset_in_collection(query, limit, offset, None)
    }

    /// Full-text search with offset pagination restricted to one collection.
    pub fn search_fts_with_offset_in_collection(
        &self,
        query: &str,
        limit: usize,
        offset: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let fts_query = search::build_fts5_query(query).unwrap_or_else(|| query.to_string());
        self.db
            .search_fts_with_offset(&fts_query, limit, offset, collection)
    }

    /// Vector similarity search.
    pub fn search_vec(&mut self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let fingerprint = self.current_embedding_fingerprint();
        self.db.validate_embedding_fingerprint(&fingerprint)?;
        self.ensure_embedder()?;
        let embedder = self.embedder.as_mut().unwrap_or_else(|| unreachable!());
        let emb = embedder.embed_query(query)?;
        self.db
            .search_vec_with_fingerprint(&emb, limit, None, &fingerprint)
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
        self.search_with_queries_with_offset_in_collection(query, &queries, limit, offset, None)
    }

    /// Hybrid search with offset pagination restricted to one collection.
    pub fn search_with_offset_in_collection(
        &mut self,
        query: &str,
        limit: usize,
        offset: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let queries = Query::expand_simple(query);
        self.search_with_queries_with_offset_in_collection(
            query, &queries, limit, offset, collection,
        )
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
        self.search_with_queries_with_offset_in_collection(query, queries, limit, offset, None)
    }

    /// Search with pre-expanded queries and collection-scoped pagination.
    pub fn search_with_queries_with_offset_in_collection(
        &mut self,
        query: &str,
        queries: &[Query],
        limit: usize,
        offset: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let fingerprint = self.current_embedding_fingerprint();
        self.db.validate_embedding_fingerprint(&fingerprint)?;
        let requested = limit.saturating_add(offset);
        let fetch_limit = requested.saturating_mul(3);

        let mut all_lists: Vec<Vec<String>> = Vec::new();
        let mut all_weights: Vec<f64> = Vec::new();
        let mut result_map: HashMap<String, SearchResult> = HashMap::new();

        if let Some(fts_q) = search::build_fts5_query(query)
            && let Ok(hits) = self.db.search_fts(&fts_q, fetch_limit, collection)
        {
            collect_results(&mut all_lists, &mut all_weights, &mut result_map, hits, 1.0);
        }

        self.ensure_embedder()?;

        for q in queries {
            match q.kind {
                QueryType::Lex => {
                    if let Some(fts_q) = search::build_fts5_query(&q.text)
                        && let Ok(hits) = self.db.search_fts(&fts_q, fetch_limit, collection)
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
                        && let Ok(hits) = self.db.search_vec_with_fingerprint(
                            &emb,
                            fetch_limit,
                            collection,
                            &fingerprint,
                        )
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
        let mut snippets: Vec<String> = Vec::new();
        let mut snippet_sources: Vec<usize> = Vec::new();
        for (index, r) in results[..top_n].iter().enumerate() {
            if let Ok(Some(body)) = self.db.get_body(&r.doc.hash) {
                snippets.push(search::extract_snippet(&body, query, 4000));
                snippet_sources.push(index);
            }
        }

        if snippets.is_empty() {
            return;
        }

        let doc_refs: Vec<&str> = snippets.iter().map(String::as_str).collect();
        let reranker = self.reranker.as_mut().unwrap_or_else(|| unreachable!());
        let Ok(scored) = reranker.rerank(query, &doc_refs, top_n) else {
            return;
        };

        let ordered = reranked_results(results, &snippet_sources, &scored);
        *results = ordered;
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
        self.db
            .status_with_expected_fingerprint(&self.current_embedding_fingerprint())
    }

    /// Diagnose an existing index through a strictly read-only connection.
    pub fn doctor(db_path: impl AsRef<Path>) -> Result<DoctorReport> {
        let db = Db::open_read_only(db_path.as_ref())?;
        let fingerprint = default_embedding_fingerprint_with_chunker(384, Chunker::default());
        db.doctor(&fingerprint, 384)
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

/// Order fusion results after cross-encoder reranking.
///
/// `snippet_sources[i]` records the position in `results` of the document whose
/// snippet was the `i`-th string handed to the reranker, so reranker indices can
/// be mapped back through stable document identity instead of snippet position.
fn reranked_results(
    results: &[SearchResult],
    snippet_sources: &[usize],
    scored: &[Scored],
) -> Vec<SearchResult> {
    let mut candidates: Vec<(usize, f32, usize)> = Vec::with_capacity(scored.len());
    let mut used: HashSet<usize> = HashSet::with_capacity(scored.len());
    for s in scored {
        // Map the reranker's snippet position back to the original document.
        let Some(&result_index) = snippet_sources.get(s.index) else {
            continue;
        };
        if result_index >= results.len() {
            continue;
        }
        if !used.insert(result_index) {
            continue;
        }
        candidates.push((result_index, s.score, s.index));
    }

    // Order only within the cross-encoder scale: score descending, then the
    // candidate's original fusion position. Untouched RRF scores are never
    // compared against cross-encoder scores.
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.2.cmp(&b.2)));

    let mut ordered: Vec<SearchResult> = Vec::with_capacity(results.len());
    for (result_index, score, _) in &candidates {
        let mut r = results[*result_index].clone();
        r.score = f64::from(*score);
        ordered.push(r);
    }
    // The unreranked tail keeps its prior fusion order.
    for (i, r) in results.iter().enumerate() {
        if !used.contains(&i) {
            ordered.push(r.clone());
        }
    }
    ordered
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

    for dir in EXCLUDE_DIRS {
        let neg = format!("!{dir}/");
        builder
            .add(&neg)
            .map_err(|e| Error::Config(e.to_string()))?;
    }

    let overrides = builder.build().map_err(|e| Error::Config(e.to_string()))?;
    let mut ignore_builder = GitignoreBuilder::new(base);
    for ignore_pattern in ignore_patterns {
        ignore_builder
            .add_line(None, ignore_pattern)
            .map_err(|e| Error::Config(e.to_string()))?;
    }
    let custom_ignores = ignore_builder
        .build()
        .map_err(|e| Error::Config(e.to_string()))?;

    let mut files = Vec::new();
    let walker = WalkBuilder::new(base).hidden(true).git_ignore(true).build();

    for dir_entry in walker {
        let entry = dir_entry.map_err(|e| Error::Config(e.to_string()))?;
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let relative = path
            .strip_prefix(base)
            .map_err(|e| Error::Config(e.to_string()))?;
        if relative.components().any(|component| {
            EXCLUDE_DIRS.contains(&component.as_os_str().to_string_lossy().as_ref())
        }) {
            continue;
        }
        if custom_ignores
            .matched_path_or_any_parents(path, false)
            .is_ignore()
        {
            continue;
        }
        if overrides.matched(relative, false).is_whitelist() {
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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::{DEFAULT_EMBED_GROUP_SIZE, Qmd, reranked_results};
    use crate::chunk::Chunker;
    use crate::db::Collection;
    use crate::db::hash_content;
    use crate::db::{Document, SearchResult, SearchSource};
    use crate::embed::{EmbeddingEngine, embedding_fingerprint};
    use crate::error::{Error, Result};
    use crate::rerank::Scored;
    use std::sync::{Arc, Mutex};

    /// Deterministic test double for the embedding engine: records every
    /// document batch it is asked to embed and can fail any batch whose text
    /// contains a marker. No ONNX model or network access is involved.
    #[derive(Default)]
    struct FakeEngine {
        /// One entry per `embed_documents` call, holding the requested texts.
        calls: Arc<Mutex<Vec<Vec<String>>>>,
        /// When set, any batch containing this marker fails.
        fail_marker: Option<String>,
        /// Vector width returned for successful embeddings.
        dims: usize,
    }

    impl FakeEngine {
        fn new(fail_marker: Option<&str>) -> Self {
            Self {
                fail_marker: fail_marker.map(str::to_string),
                dims: 8,
                ..Self::default()
            }
        }
    }

    impl EmbeddingEngine for FakeEngine {
        fn embed_query(&mut self, _query: &str) -> Result<Vec<f32>> {
            Ok(vec![0.0; self.dims])
        }

        fn embed_documents(&mut self, docs: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut calls = self
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            calls.push(docs.iter().map(|d| (*d).to_string()).collect());
            drop(calls);
            if let Some(marker) = &self.fail_marker
                && docs.iter().any(|d| d.contains(marker))
            {
                return Err(Error::Embedding(format!(
                    "synthetic failure for marker {marker}"
                )));
            }
            Ok(vec![vec![0.0; self.dims]; docs.len()])
        }
    }

    /// Insert a document body into the CAS and register it under `docs/<path>`.
    fn seed_embed_doc(qmd: &Qmd, path: &str, body: &str) {
        let hash = hash_content(body);
        qmd.db().insert_content(&hash, body).unwrap();
        qmd.db().upsert_document("docs", path, path, &hash).unwrap();
    }

    /// Build a fusion result for a `docs/<path>` document with `score`.
    fn search_result(path: &str, score: f64) -> SearchResult {
        SearchResult {
            doc: Document {
                collection: "docs".into(),
                path: path.into(),
                title: path.into(),
                hash: format!("hash-{path}"),
                modified_at: String::new(),
                body_len: 0,
                body: None,
            },
            score,
            source: SearchSource::Fts,
        }
    }

    fn paths(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|r| r.doc.path.as_str()).collect()
    }

    #[test]
    fn rerank_index_maps_through_original_identity_not_snippet_position() {
        // Fusion order is [a, b, c]; b has no readable body, so only a and c
        // were handed to the reranker (snippet positions 0 and 1).
        let results = vec![
            search_result("a.md", 1.0),
            search_result("b.md", 1.0),
            search_result("c.md", 1.0),
        ];
        let snippet_sources = [0usize, 2usize];
        // The reranker ranks its second snippet best, which is c.md.
        let scored = [Scored {
            index: 1,
            score: 0.9,
        }];

        let ordered = reranked_results(&results, &snippet_sources, &scored);

        // Exactly one document carries the cross-encoder score; it must be c.md.
        let winner = f64::from(0.9_f32);
        let reranked = ordered
            .iter()
            .find(|r| r.score.total_cmp(&winner).is_eq())
            .expect("reranked candidate should be present");
        assert_eq!(reranked.doc.path, "c.md");
        assert_ne!(reranked.doc.path, "b.md");
    }

    #[test]
    fn rerank_keeps_reranked_prefix_and_untouched_fusion_tail() {
        // RRF scores on the untouched tail intentionally dwarf the cross-encoder
        // scores, so any global score sort would reorder the tail ahead.
        let results = vec![
            search_result("a.md", 3.0),
            search_result("b.md", 2.0),
            search_result("c.md", 1.0),
            search_result("d.md", 0.5),
        ];
        let snippet_sources = [0usize, 1, 2, 3];
        // Reranker returns c then a; b and d fall outside the reranker limit.
        let scored = [
            Scored {
                index: 2,
                score: 0.9,
            },
            Scored {
                index: 0,
                score: 0.4,
            },
        ];

        let ordered = reranked_results(&results, &snippet_sources, &scored);

        assert_eq!(paths(&ordered), ["c.md", "a.md", "b.md", "d.md"]);
        // The tail keeps its fusion score and does not jump the prefix.
        assert_eq!(ordered[2].doc.path, "b.md");
        let b_fusion = results[1].score;
        assert!(ordered[2].score.total_cmp(&b_fusion).is_eq());
        assert!(ordered[2].score > ordered[1].score);
    }

    #[test]
    fn rerank_ties_keep_prior_fusion_order() {
        // Fusion order deliberately differs from lexical path order.
        let results = vec![
            search_result("b.md", 1.0),
            search_result("a.md", 1.0),
            search_result("c.md", 1.0),
        ];
        let snippet_sources = [0usize, 1, 2];
        // Equal cross-encoder scores, deliberately returned out of order.
        let scored = [
            Scored {
                index: 2,
                score: 0.5,
            },
            Scored {
                index: 0,
                score: 0.5,
            },
            Scored {
                index: 1,
                score: 0.5,
            },
        ];

        let ordered = reranked_results(&results, &snippet_sources, &scored);

        assert_eq!(paths(&ordered), ["b.md", "a.md", "c.md"]);
    }

    #[test]
    fn rerank_without_scores_keeps_prior_fusion_order() {
        let results = vec![
            search_result("a.md", 1.0),
            search_result("b.md", 3.0),
            search_result("c.md", 2.0),
        ];
        let snippet_sources = [0usize, 1, 2];

        let ordered = reranked_results(&results, &snippet_sources, &[]);

        assert_eq!(paths(&ordered), ["a.md", "b.md", "c.md"]);
    }

    #[test]
    fn default_collection_walk_excludes_hidden_and_gitignored_markdown() {
        use std::fs;
        use std::process::Command;

        let root = std::env::temp_dir().join(format!(
            "qmd-default-exclusions-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("visible/nested")).unwrap();
        fs::create_dir_all(root.join(".hidden-dir")).unwrap();
        fs::create_dir_all(root.join("ignored-dir")).unwrap();
        fs::create_dir_all(root.join("custom-dir/nested")).unwrap();
        fs::write(root.join("visible.md"), "visible searchable markdown").unwrap();
        fs::write(
            root.join("visible/nested/nested.md"),
            "nested searchable markdown",
        )
        .unwrap();
        fs::write(root.join(".hidden.md"), "hidden searchable markdown").unwrap();
        fs::write(
            root.join(".hidden-dir/hidden.md"),
            "hidden dir searchable markdown",
        )
        .unwrap();
        fs::write(root.join("ignored.md"), "ignored searchable markdown").unwrap();
        fs::write(
            root.join("ignored-dir/ignored.md"),
            "ignored dir searchable markdown",
        )
        .unwrap();
        fs::write(
            root.join("custom-dir/nested/custom.md"),
            "custom ignored searchable markdown",
        )
        .unwrap();
        fs::write(root.join("notes.txt"), "text searchable but not markdown").unwrap();
        fs::write(root.join(".gitignore"), "ignored.md\nignored-dir/\n").unwrap();
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .current_dir(&root)
                .status()
                .unwrap()
                .success()
        );

        let qmd = Qmd::open_memory().unwrap();
        qmd.register_collection(&Collection::new("docs", root.to_string_lossy()))
            .unwrap();
        qmd.update(None).unwrap();

        let hits = qmd.search_fts("searchable", 20).unwrap();
        let paths: Vec<_> = hits.iter().map(|hit| hit.doc.path.as_str()).collect();
        assert!(paths.iter().any(|path| path.ends_with("visible.md")));
        assert!(paths.iter().any(|path| path.ends_with("nested.md")));
        assert!(paths.iter().any(|path| path.ends_with("custom.md")));
        assert!(!paths.iter().any(|path| path.ends_with(".hidden.md")));
        assert!(!paths.iter().any(|path| path.ends_with("hidden.md")));
        assert!(!paths.iter().any(|path| path.ends_with("ignored.md")));
        assert!(!paths.iter().any(|path| path.ends_with("notes.txt")));

        qmd.register_collection(
            &Collection::new("docs", root.to_string_lossy())
                .with_ignore(vec!["custom-dir/".to_string()]),
        )
        .unwrap();
        qmd.update(None).unwrap();
        let filtered_hits = qmd.search_fts("searchable", 20).unwrap();
        let filtered_paths: Vec<_> = filtered_hits
            .iter()
            .map(|hit| hit.doc.path.as_str())
            .collect();
        assert!(
            !filtered_paths
                .iter()
                .any(|path| path.ends_with("custom.md"))
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn custom_chunker_changes_active_embedding_fingerprint() {
        let mut qmd = Qmd::open_memory().expect("in-memory qmd should open");
        qmd.set_chunker(Chunker::new(100, 10));
        assert_eq!(
            qmd.current_embedding_fingerprint(),
            embedding_fingerprint(384, 100, 10)
        );
    }

    #[test]
    fn update_keeps_successes_and_reports_file_failures() {
        use crate::db::Collection;
        use std::fs;

        let root = std::env::temp_dir().join(format!(
            "qmd-partial-update-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("good.md"), "# good\nsearchable text").unwrap();
        fs::write(root.join("bad.md"), [0xff, 0xfe]).unwrap();

        let qmd = Qmd::open_memory().unwrap();
        qmd.register_collection(&Collection::new("docs", root.to_string_lossy()))
            .unwrap();
        let result = qmd.update(None).unwrap();

        assert_eq!(result.indexed, 1);
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures[0].path.ends_with("bad.md"));
        assert!(!result.failures[0].reason.is_empty());
        assert_eq!(qmd.doc_count().unwrap(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn doctor_does_not_modify_the_index_file() {
        use std::fs;

        let path = std::env::temp_dir().join(format!(
            "qmd-doctor-read-only-{}.sqlite",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        {
            let _qmd = Qmd::open(&path).unwrap();
        }
        let before = fs::read(&path).unwrap();
        let report = Qmd::doctor(&path).unwrap();
        let after = fs::read(&path).unwrap();
        assert!(!report.has_errors());
        assert_eq!(before, after);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn emptied_source_deactivates_and_later_nonempty_update_reactivates() {
        use std::fs;

        let root = std::env::temp_dir().join(format!(
            "qmd-empty-source-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let file = root.join("note.md");
        fs::write(&file, "# note\nsearchable orange content").unwrap();

        let qmd = Qmd::open_memory().unwrap();
        qmd.register_collection(&Collection::new("docs", root.to_string_lossy()))
            .unwrap();
        let first = qmd.update(None).unwrap();
        assert_eq!(first.indexed, 1);
        assert_eq!(qmd.doc_count().unwrap(), 1);
        assert_eq!(qmd.search_fts("orange", 10).unwrap().len(), 1);

        // Emptied source: the previously active document must be deactivated
        // rather than left behind as a stale, still-searchable path.
        fs::write(&file, "").unwrap();
        let emptied = qmd.update(None).unwrap();
        assert_eq!(emptied.removed, 1);
        assert_eq!(qmd.doc_count().unwrap(), 0);
        assert!(qmd.search_fts("orange", 10).unwrap().is_empty());

        // Whitespace-only content is equivalent to empty for deactivation.
        fs::write(&file, "   \n\t\n").unwrap();
        let whitespace = qmd.update(None).unwrap();
        assert_eq!(whitespace.removed, 0);
        assert_eq!(qmd.doc_count().unwrap(), 0);

        // A later nonempty write reactivates the same path normally.
        fs::write(&file, "# note\nsearchable orange content again").unwrap();
        let restored = qmd.update(None).unwrap();
        assert_eq!(restored.indexed, 1);
        assert_eq!(qmd.doc_count().unwrap(), 1);
        assert_eq!(qmd.search_fts("orange", 10).unwrap().len(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn embed_schedules_one_job_per_content_hash_across_paths() {
        let mut qmd = Qmd::open_memory().unwrap();
        let shared = "# shared\nunique-token-alpha content body";
        seed_embed_doc(&qmd, "a.md", shared);
        seed_embed_doc(&qmd, "b.md", shared);
        seed_embed_doc(&qmd, "c.md", "# other\nunique-token-beta content body");

        let engine = FakeEngine::new(None);
        let calls = Arc::clone(&engine.calls);
        qmd.embedder = Some(Box::new(engine));

        let result = qmd.embed().unwrap();

        assert_eq!(result.failures, 0);
        assert_eq!(result.embedded, 2, "one job per unique content hash");
        assert_eq!(result.remaining, 0);
        let alpha_calls = calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| call.iter().any(|text| text.contains("unique-token-alpha")))
            .count();
        assert_eq!(alpha_calls, 1, "identical content must be embedded once");
    }

    #[test]
    fn embed_failure_alias_is_deterministic_and_preserves_hash_and_error() {
        let mut qmd = Qmd::open_memory().unwrap();
        let shared = "# shared\nconflict-token failure body";
        seed_embed_doc(&qmd, "b.md", shared);
        seed_embed_doc(&qmd, "a.md", shared);
        let hash = hash_content(shared);

        qmd.embedder = Some(Box::new(FakeEngine::new(Some("conflict-token"))));

        let result = qmd.embed().unwrap();

        assert_eq!(result.failures, 1);
        assert_eq!(result.embedded, 0);
        assert_eq!(result.remaining, 1);
        assert_eq!(qmd.db().vector_count().unwrap(), 0);
        let messages = result.failure_messages.join("\n");
        assert!(
            messages.contains("a.md"),
            "expected deterministic MIN(path) alias in: {messages}"
        );
        assert!(
            !messages.contains("b.md"),
            "non-deterministic alias leaked into: {messages}"
        );
        assert!(messages.contains(&hash), "hash missing from: {messages}");
        assert!(
            messages.contains("synthetic failure"),
            "underlying error missing from: {messages}"
        );
    }

    #[test]
    fn embed_preserves_completed_work_when_a_later_job_fails() {
        let mut qmd = Qmd::open_memory().unwrap();
        let total = DEFAULT_EMBED_GROUP_SIZE + 1;
        for i in 0..total - 1 {
            seed_embed_doc(
                &qmd,
                &format!("doc-{i:04}.md"),
                &format!("# doc {i}\ncontent number {i}"),
            );
        }
        seed_embed_doc(&qmd, "last.md", "# last\ntrigger-failure marker body");

        let engine = FakeEngine::new(Some("trigger-failure"));
        let calls = Arc::clone(&engine.calls);
        qmd.embedder = Some(Box::new(engine));

        let result = qmd.embed().unwrap();

        assert_eq!(result.failures, 1);
        assert_eq!(
            result.embedded,
            total - 1,
            "completed independent work must survive a later failure"
        );
        assert_eq!(result.remaining, 1);
        assert_eq!(qmd.db().vector_count().unwrap(), total - 1);
        let messages = result.failure_messages.join("\n");
        assert!(
            messages.contains("last.md"),
            "affected path missing from: {messages}"
        );
        assert!(
            messages.contains("synthetic failure"),
            "underlying error missing from: {messages}"
        );
        assert_eq!(calls.lock().unwrap().len(), total);
    }

    #[test]
    fn embed_batch_caps_the_total_documents_embedded() {
        let mut qmd = Qmd::open_memory().unwrap();
        for i in 0..5 {
            seed_embed_doc(&qmd, &format!("cap-{i}.md"), &format!("cap body {i}"));
        }
        qmd.embedder = Some(Box::new(FakeEngine::new(None)));

        let result = qmd.embed_with_batch(Some(2)).unwrap();

        assert_eq!(
            result.embedded, 2,
            "--batch caps total work, not the per-group bound"
        );
        assert_eq!(result.failures, 0);
        assert_eq!(result.remaining, 3);
        assert_eq!(qmd.db().vector_count().unwrap(), 2);
    }

    #[test]
    fn default_embed_group_is_declared_and_paged() {
        let qmd = Qmd::open_memory().unwrap();
        let total = DEFAULT_EMBED_GROUP_SIZE * 2 + 3;
        for i in 0..total {
            seed_embed_doc(&qmd, &format!("page-{i:04}.md"), &format!("page body {i}"));
        }

        let first = qmd
            .db()
            .unembedded_doc_group(DEFAULT_EMBED_GROUP_SIZE, 0)
            .unwrap();
        assert_eq!(first.len(), DEFAULT_EMBED_GROUP_SIZE);
        let second = qmd
            .db()
            .unembedded_doc_group(DEFAULT_EMBED_GROUP_SIZE, first.last().unwrap().3)
            .unwrap();
        assert_eq!(second.len(), DEFAULT_EMBED_GROUP_SIZE);
        let third = qmd
            .db()
            .unembedded_doc_group(DEFAULT_EMBED_GROUP_SIZE, second.last().unwrap().3)
            .unwrap();
        assert_eq!(third.len(), 3);

        let mut hashes: Vec<&str> = first
            .iter()
            .chain(&second)
            .chain(&third)
            .map(|row| row.0.as_str())
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), total, "pages must not re-select or skip work");
    }
}

/// Result of an [`update`](Qmd::update) operation across collections.
#[derive(Debug, Clone, Default)]
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
    /// Files that failed while successful files remained indexed.
    pub failures: Vec<IndexFailure>,
}

/// Structured per-file indexing failure.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct IndexFailure {
    /// Actionable source path.
    pub path: String,
    /// Underlying failure reason.
    pub reason: String,
}

/// Result of indexing a single collection.
#[derive(Debug, Clone)]
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
    /// Files that failed while successful files remained indexed.
    pub failures: Vec<IndexFailure>,
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
