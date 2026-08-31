//! SQLite database: schema, connection, and core operations.
//!
//! Integrates `sqlite-vec` for native KNN vector search and FTS5 for
//! full-text search, all within a single `.sqlite` file.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;
use std::time::SystemTime;

use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use zerocopy::IntoBytes;

use crate::error::{Error, Result};

/// Ensures sqlite-vec is registered once before SQLite connections are opened.
static SQLITE_VEC_EXTENSION: Once = Once::new();

/// RFC 3339 UTC timestamp from system clock.
fn now_rfc3339() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let rem = secs % 86400;
    let (year, month, day) = days_to_ymd(secs / 86400);
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
const fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_out = if m <= 2 { y + 1 } else { y };
    (y_out, m, d)
}

/// SHA-256 hash of content, lowercase hex.
#[must_use]
pub fn hash_content(content: &str) -> String {
    let mut h = Sha256::new();
    h.update(content.as_bytes());
    format!("{:x}", h.finalize())
}

/// A registered collection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Collection {
    /// Collection name (unique identifier).
    pub name: String,
    /// Absolute path to the collection root directory.
    pub path: String,
    /// Glob pattern for file matching (default: `**/*.md`).
    #[serde(default = "default_pattern")]
    pub pattern: String,
    /// Glob patterns to exclude.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
    /// Path-scoped context descriptions.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub context: HashMap<String, String>,
}

/// Default glob pattern.
fn default_pattern() -> String {
    "**/*.md".to_string()
}

impl Collection {
    /// Create a new collection with the given name and path.
    ///
    /// Uses `**/*.md` as the default pattern with no ignore rules or context.
    #[must_use]
    pub fn new(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            pattern: default_pattern(),
            ignore: Vec::new(),
            context: HashMap::new(),
        }
    }

    /// Set a custom glob pattern.
    #[must_use]
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = pattern.into();
        self
    }

    /// Set ignore patterns.
    #[must_use]
    pub fn with_ignore(mut self, ignore: Vec<String>) -> Self {
        self.ignore = ignore;
        self
    }
}

impl Default for Collection {
    fn default() -> Self {
        Self {
            name: String::new(),
            path: String::new(),
            pattern: default_pattern(),
            ignore: Vec::new(),
            context: HashMap::new(),
        }
    }
}

/// Collection info with document statistics.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct CollectionInfo {
    /// Collection configuration.
    #[serde(flatten)]
    pub collection: Collection,
    /// Total document count (active).
    pub doc_count: usize,
    /// Last modification timestamp.
    pub last_modified: Option<String>,
}

/// An indexed document.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct Document {
    /// Parent collection name.
    pub collection: String,
    /// Relative path within the collection.
    pub path: String,
    /// Document title.
    pub title: String,
    /// Content SHA-256 hash.
    pub hash: String,
    /// Last modification timestamp (RFC 3339).
    pub modified_at: String,
    /// Body length in bytes.
    pub body_len: usize,
    /// Body text (loaded on demand).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

impl Document {
    /// Short document id (first 6 hex chars of the hash).
    #[must_use]
    pub fn docid(&self) -> &str {
        &self.hash[..6.min(self.hash.len())]
    }

    /// Display path: `collection/path`.
    #[must_use]
    pub fn display_path(&self) -> String {
        format!("{}/{}", self.collection, self.path)
    }
}

/// Search result with relevance score.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct SearchResult {
    /// The matched document.
    pub doc: Document,
    /// Relevance score (higher is better).
    pub score: f64,
    /// Search backend that produced this result.
    pub source: SearchSource,
}

/// Which search backend produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum SearchSource {
    /// Full-text search (BM25).
    Fts,
    /// Vector similarity search.
    Vec,
}

/// Index health and status information.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct IndexStatus {
    /// Total active documents.
    pub total_documents: usize,
    /// Documents needing embedding.
    pub needs_embedding: usize,
    /// Whether a vector index exists.
    pub has_vector_index: bool,
    /// Per-collection info.
    pub collections: Vec<CollectionInfo>,
}

/// The database layer.
#[derive(Debug)]
pub struct Db {
    /// SQLite connection handle.
    pub(crate) conn: Connection,
    /// Embedding dimensionality (set after first embedding insert).
    dims: Option<usize>,
}

impl Db {
    /// Open (or create) a database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Self::register_sqlite_vec();
        let conn = Connection::open(path)?;
        let db = Self { conn, dims: None };
        db.migrate()?;
        Ok(db)
    }

    /// Open an in-memory database (useful for tests).
    pub fn open_memory() -> Result<Self> {
        Self::register_sqlite_vec();
        let conn = Connection::open_in_memory()?;
        let db = Self { conn, dims: None };
        db.migrate()?;
        Ok(db)
    }

    /// Register sqlite-vec before any SQLite connection is opened.
    fn register_sqlite_vec() {
        SQLITE_VEC_EXTENSION.call_once(|| {
            #[allow(unsafe_code, clippy::missing_transmute_annotations)]
            unsafe {
                rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                    sqlite_vec::sqlite3_vec_init as *const (),
                )));
            }
        });
    }

    /// Run schema migrations.
    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS content (
                hash       TEXT PRIMARY KEY,
                doc        TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                collection  TEXT NOT NULL,
                path        TEXT NOT NULL,
                title       TEXT NOT NULL,
                hash        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
                UNIQUE(collection, path)
            );

            CREATE INDEX IF NOT EXISTS idx_doc_coll ON documents(collection, active);
            CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(hash);
            CREATE INDEX IF NOT EXISTS idx_doc_path ON documents(path, active);

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                filepath, title, body,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS content_vectors (
                hash        TEXT NOT NULL,
                seq         INTEGER NOT NULL DEFAULT 0,
                pos         INTEGER NOT NULL DEFAULT 0,
                model       TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                PRIMARY KEY (hash, seq)
            );

            CREATE TABLE IF NOT EXISTS store_collections (
                name            TEXT PRIMARY KEY,
                path            TEXT NOT NULL,
                pattern         TEXT NOT NULL DEFAULT '**/*.md',
                ignore_patterns TEXT,
                context         TEXT
            );

            CREATE TABLE IF NOT EXISTS store_config (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            ",
        )?;
        self.ensure_fts_triggers()?;
        Ok(())
    }

    /// Create FTS synchronization triggers if absent.
    fn ensure_fts_triggers(&self) -> Result<()> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='documents_ai'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            self.conn.execute_batch(
                r"
                CREATE TRIGGER documents_ai AFTER INSERT ON documents
                WHEN new.active = 1
                BEGIN
                    INSERT INTO documents_fts(rowid, filepath, title, body)
                    SELECT new.id,
                           new.collection || '/' || new.path,
                           new.title,
                           (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;

                CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;
                    INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
                    SELECT new.id,
                           new.collection || '/' || new.path,
                           new.title,
                           (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;
                ",
            )?;
        }
        Ok(())
    }

    /// Create the sqlite-vec virtual table for the given dimensionality.
    fn ensure_vec_table(&self, dims: usize) -> Result<()> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_embeddings'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            self.conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE vec_embeddings USING vec0(embedding float[{dims}]);"
            ))?;
        }
        Ok(())
    }

    // ── Collection management ───────────────────────────────────────────

    /// Register or update a collection.
    pub fn upsert_collection(&self, coll: &Collection) -> Result<()> {
        let ignore_json = if coll.ignore.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&coll.ignore)?)
        };
        let ctx_json = if coll.context.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&coll.context)?)
        };
        self.conn.execute(
            r"INSERT INTO store_collections (name, path, pattern, ignore_patterns, context)
              VALUES (?1, ?2, ?3, ?4, ?5)
              ON CONFLICT(name) DO UPDATE SET
                  path = excluded.path,
                  pattern = excluded.pattern,
                  ignore_patterns = excluded.ignore_patterns,
                  context = excluded.context",
            params![coll.name, coll.path, coll.pattern, ignore_json, ctx_json],
        )?;
        Ok(())
    }

    /// Get a collection by name.
    pub fn get_collection(&self, name: &str) -> Result<Option<Collection>> {
        self.conn
            .query_row(
                "SELECT name, path, pattern, ignore_patterns, context FROM store_collections WHERE name = ?1",
                params![name],
                |row| Ok(row_to_collection(row)),
            )
            .optional()
            .map_err(Into::into)
    }

    /// List all registered collections.
    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, path, pattern, ignore_patterns, context FROM store_collections",
        )?;
        let colls = stmt
            .query_map([], |row| Ok(row_to_collection(row)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(colls)
    }

    /// Delete a collection registration and its documents.
    pub fn delete_collection(&self, name: &str) -> Result<usize> {
        let count = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE collection = ?1",
            params![name],
            |row| row.get::<_, i64>(0).map(|v| v as usize),
        )?;
        self.conn
            .execute("DELETE FROM documents WHERE collection = ?1", params![name])?;
        self.conn.execute(
            "DELETE FROM store_collections WHERE name = ?1",
            params![name],
        )?;
        self.cleanup()?;
        Ok(count)
    }

    /// Rename a collection.
    pub fn rename_collection(&self, old_name: &str, new_name: &str) -> Result<()> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM store_collections WHERE name = ?1",
                params![new_name],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if exists {
            return Err(Error::CollectionExists(new_name.to_string()));
        }
        self.conn.execute(
            "UPDATE store_collections SET name = ?1 WHERE name = ?2",
            params![new_name, old_name],
        )?;
        self.conn.execute(
            "UPDATE documents SET collection = ?1 WHERE collection = ?2",
            params![new_name, old_name],
        )?;
        Ok(())
    }

    // ── Context management ──────────────────────────────────────────────

    /// Set or update a path-scoped context for a collection.
    pub fn set_context(&self, collection: &str, path_prefix: &str, text: &str) -> Result<bool> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM store_collections WHERE name = ?1",
                params![collection],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return Ok(false);
        }
        let ctx_raw: Option<String> = self
            .conn
            .query_row(
                "SELECT context FROM store_collections WHERE name = ?1",
                params![collection],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        let mut ctx: HashMap<String, String> = ctx_raw
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        ctx.insert(path_prefix.to_string(), text.to_string());
        let json = serde_json::to_string(&ctx)?;
        self.conn.execute(
            "UPDATE store_collections SET context = ?1 WHERE name = ?2",
            params![json, collection],
        )?;
        Ok(true)
    }

    /// Remove a path-scoped context from a collection.
    pub fn remove_context(&self, collection: &str, path_prefix: &str) -> Result<bool> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM store_collections WHERE name = ?1",
                params![collection],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return Ok(false);
        }
        let ctx_raw: Option<String> = self
            .conn
            .query_row(
                "SELECT context FROM store_collections WHERE name = ?1",
                params![collection],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        let mut ctx: HashMap<String, String> = ctx_raw
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        if ctx.remove(path_prefix).is_none() {
            return Ok(false);
        }
        let json = if ctx.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&ctx)?)
        };
        self.conn.execute(
            "UPDATE store_collections SET context = ?1 WHERE name = ?2",
            params![json, collection],
        )?;
        Ok(true)
    }

    /// Set the global context (applies to all collections).
    pub fn set_global_context(&self, text: Option<&str>) -> Result<()> {
        if let Some(t) = text {
            self.conn.execute(
                r"INSERT INTO store_config (key, value) VALUES ('global_context', ?1)
                  ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![t],
            )?;
        } else {
            self.conn
                .execute("DELETE FROM store_config WHERE key = 'global_context'", [])?;
        }
        Ok(())
    }

    /// Get the global context.
    pub fn global_context(&self) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT value FROM store_config WHERE key = 'global_context'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Get the best matching context for a file path (longest prefix match).
    pub fn context_for_path(&self, collection: &str, file_path: &str) -> Result<Option<String>> {
        let mut parts: Vec<String> = Vec::new();

        if let Some(global) = self.global_context()? {
            parts.push(global);
        }

        if let Some(coll) = self.get_collection(collection)? {
            let normalized = if file_path.starts_with('/') {
                file_path.to_string()
            } else {
                format!("/{file_path}")
            };
            let mut matches: Vec<(usize, &String)> = coll
                .context
                .iter()
                .filter(|(prefix, _)| {
                    let np = if prefix.starts_with('/') {
                        (*prefix).clone()
                    } else {
                        format!("/{prefix}")
                    };
                    normalized.starts_with(&np)
                })
                .map(|(prefix, text)| (prefix.len(), text))
                .collect();
            matches.sort_by_key(|(len, _)| *len);
            for (_, text) in matches {
                parts.push(text.clone());
            }
        }

        if parts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(parts.join("\n\n")))
        }
    }

    // ── Content / Document CRUD ─────────────────────────────────────────

    /// Insert content into CAS. No-op if hash already exists.
    pub fn insert_content(&self, hash: &str, content: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            params![hash, content, now_rfc3339()],
        )?;
        Ok(())
    }

    /// Get document body by content hash.
    pub fn get_body(&self, hash: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT doc FROM content WHERE hash = ?1",
                params![hash],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Upsert a document record.
    pub fn upsert_document(
        &self,
        collection: &str,
        path: &str,
        title: &str,
        hash: &str,
    ) -> Result<()> {
        let now = now_rfc3339();
        self.conn.execute(
            r"INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
              VALUES (?1, ?2, ?3, ?4, ?5, ?5, 1)
              ON CONFLICT(collection, path) DO UPDATE SET
                  title = excluded.title, hash = excluded.hash,
                  modified_at = excluded.modified_at, active = 1",
            params![collection, path, title, hash, now],
        )?;
        Ok(())
    }

    /// Deactivate a document.
    pub fn deactivate(&self, collection: &str, path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2",
            params![collection, path],
        )?;
        Ok(())
    }

    /// Get all active paths for a collection.
    pub fn active_paths(&self, collection: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM documents WHERE collection = ?1 AND active = 1")?;
        let paths = stmt
            .query_map(params![collection], |row| row.get(0))?
            .collect::<std::result::Result<Vec<String>, _>>()?;
        Ok(paths)
    }

    /// Get a document by collection and path.
    pub fn get_document(&self, collection: &str, path: &str) -> Result<Option<Document>> {
        self.conn
            .query_row(
                r"SELECT d.title, d.hash, d.modified_at, c.doc, LENGTH(c.doc)
                  FROM documents d JOIN content c ON c.hash = d.hash
                  WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1",
                params![collection, path],
                |row| {
                    let body: String = row.get(3)?;
                    let body_len: i64 = row.get(4)?;
                    Ok(Document {
                        collection: collection.to_string(),
                        path: path.to_string(),
                        title: row.get(0)?,
                        hash: row.get(1)?,
                        modified_at: row.get(2)?,
                        body_len: body_len as usize,
                        body: Some(body),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// Find document by short docid (first 6 hex chars of hash).
    pub fn find_by_docid(&self, docid: &str) -> Result<Option<(String, String)>> {
        let clean = docid.trim_start_matches('#');
        self.conn
            .query_row(
                r"SELECT collection, path FROM documents
                  WHERE hash LIKE ?1 || '%' AND active = 1 LIMIT 1",
                params![clean],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(Into::into)
    }

    // ── Search ──────────────────────────────────────────────────────────

    /// Full-text search using FTS5 BM25.
    pub fn search_fts(
        &self,
        fts_query: &str,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        self.search_fts_with_offset(fts_query, limit, 0, collection)
    }

    /// Full-text search with a deterministic offset for pagination.
    pub fn search_fts_with_offset(
        &self,
        fts_query: &str,
        limit: usize,
        offset: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let (coll_filter, limit_param, offset_param) = if collection.is_some() {
            ("AND d.collection = ?2", "?3", "?4")
        } else {
            ("", "?2", "?3")
        };

        let sql = format!(
            r"SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                     bm25(documents_fts, 10.0, 1.0, 1.0) as score, LENGTH(c.doc)
              FROM documents_fts fts
              JOIN documents d ON d.id = fts.rowid
              JOIN content c ON c.hash = d.hash
              WHERE documents_fts MATCH ?1 {coll_filter} AND d.active = 1
              ORDER BY score
              LIMIT {limit_param} OFFSET {offset_param}"
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = |row: &rusqlite::Row<'_>| {
            let body_len: i64 = row.get(6)?;
            let raw_bm25: f64 = row.get(5)?;
            Ok(SearchResult {
                doc: Document {
                    collection: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    hash: row.get(3)?,
                    modified_at: row.get(4)?,
                    body_len: body_len as usize,
                    body: None,
                },
                score: crate::search::normalize_bm25(-raw_bm25),
                source: SearchSource::Fts,
            })
        };

        let results: Vec<SearchResult> = if let Some(coll) = collection {
            stmt.query_map(
                params![fts_query, coll, limit as i64, offset as i64],
                map_row,
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![fts_query, limit as i64, offset as i64], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(results)
    }

    // ── Embedding / Vector ──────────────────────────────────────────────

    /// Insert an embedding vector using sqlite-vec.
    ///
    /// The `rowid` in `vec_embeddings` is a sequential integer. We store a
    /// mapping in `content_vectors` from `(hash, seq)` → `rowid`.
    pub fn insert_embedding(
        &mut self,
        hash: &str,
        seq: usize,
        pos: usize,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        if self.dims.is_none() {
            self.dims = Some(embedding.len());
            self.ensure_vec_table(embedding.len())?;
        }

        let now = now_rfc3339();

        let existing_rowid: Option<i64> = self
            .conn
            .query_row(
                "SELECT rowid FROM content_vectors WHERE hash = ?1 AND seq = ?2",
                params![hash, seq as i64],
                |row| row.get(0),
            )
            .optional()?;

        let vec_bytes = embedding.as_bytes();

        if let Some(rid) = existing_rowid {
            self.conn.execute(
                "UPDATE vec_embeddings SET embedding = ?1 WHERE rowid = ?2",
                params![vec_bytes, rid],
            )?;
            self.conn.execute(
                "UPDATE content_vectors SET pos = ?1, model = ?2, embedded_at = ?3 WHERE hash = ?4 AND seq = ?5",
                params![pos as i64, model, now, hash, seq as i64],
            )?;
        } else {
            self.conn.execute(
                "INSERT INTO vec_embeddings (embedding) VALUES (?1)",
                params![vec_bytes],
            )?;
            let rowid = self.conn.last_insert_rowid();
            self.conn.execute(
                r"INSERT INTO content_vectors (rowid, hash, seq, pos, model, embedded_at)
                  VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![rowid, hash, seq as i64, pos as i64, model, now],
            )?;
        }

        Ok(())
    }

    /// Documents that need embedding.
    pub fn unembedded_docs(&self) -> Result<Vec<(String, String, String)>> {
        self.unembedded_docs_with_limit(None)
    }

    /// Documents that need embedding, optionally limited for a batch run.
    pub fn unembedded_docs_with_limit(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<(String, String, String)>> {
        let limit_clause = limit.map_or_else(String::new, |_| " LIMIT ?1".to_string());
        let sql = format!(
            r"SELECT DISTINCT d.hash, d.path, c.doc
              FROM documents d
              JOIN content c ON c.hash = d.hash
              LEFT JOIN content_vectors v
                ON d.hash = v.hash AND v.seq = 0 AND v.model = 'default:complete'
              WHERE d.active = 1 AND v.hash IS NULL
              ORDER BY d.id{limit_clause}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mapper = |row: &rusqlite::Row<'_>| Ok((row.get(0)?, row.get(1)?, row.get(2)?));
        let results = match limit {
            Some(n) => stmt
                .query_map(params![n as i64], mapper)?
                .collect::<std::result::Result<Vec<_>, _>>()?,
            None => stmt
                .query_map([], mapper)?
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        Ok(results)
    }

    /// Vector similarity search using sqlite-vec native KNN.
    pub fn search_vec(
        &self,
        query_embedding: &[f32],
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if self.dims.is_none() {
            return Ok(Vec::new());
        }

        let vec_bytes = query_embedding.as_bytes();

        let coll_filter = if collection.is_some() {
            "AND d.collection = ?3"
        } else {
            ""
        };

        let sql = format!(
            r"SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                     LENGTH(c.doc), ve.distance
              FROM vec_embeddings ve
              JOIN content_vectors cv ON cv.rowid = ve.rowid
              JOIN documents d ON d.hash = cv.hash AND d.active = 1
              JOIN content c ON c.hash = d.hash
              WHERE ve.embedding MATCH ?1
                AND k = ?2
                {coll_filter}
              ORDER BY ve.distance"
        );

        let mut stmt = self.conn.prepare(&sql)?;

        let map_row = |row: &rusqlite::Row<'_>| {
            let body_len: i64 = row.get(5)?;
            let distance: f64 = row.get(6)?;
            let similarity = 1.0 - distance;
            Ok(SearchResult {
                doc: Document {
                    collection: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    hash: row.get(3)?,
                    modified_at: row.get(4)?,
                    body_len: body_len as usize,
                    body: None,
                },
                score: similarity,
                source: SearchSource::Vec,
            })
        };

        let results: Vec<SearchResult> = if let Some(coll) = collection {
            stmt.query_map(params![vec_bytes, limit as i64, coll], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![vec_bytes, limit as i64], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(results)
    }

    // ── Index health ────────────────────────────────────────────────────

    /// Count active documents.
    pub fn doc_count(&self) -> Result<usize> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE active = 1",
            [],
            |row| row.get::<_, i64>(0).map(|v| v as usize),
        )?)
    }

    /// Count documents needing embedding.
    pub fn needs_embedding_count(&self) -> Result<usize> {
        Ok(self.conn.query_row(
            r"SELECT COUNT(DISTINCT d.hash)
              FROM documents d
              LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
                AND v.model = 'default:complete'
              WHERE d.active = 1 AND v.hash IS NULL",
            [],
            |row| row.get::<_, i64>(0).map(|v| v as usize),
        )?)
    }

    /// Get full index status.
    pub fn status(&self) -> Result<IndexStatus> {
        let total = self.doc_count()?;
        let needs = self.needs_embedding_count()?;
        let has_vec: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_embeddings'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        let collections = self.list_collections()?;
        let mut infos = Vec::with_capacity(collections.len());
        for coll in collections {
            let (count, last_mod): (usize, Option<String>) = self.conn.query_row(
                r"SELECT COUNT(*), MAX(modified_at)
                      FROM documents WHERE collection = ?1 AND active = 1",
                params![coll.name],
                |row| {
                    let c: i64 = row.get(0)?;
                    let m: Option<String> = row.get(1)?;
                    Ok((c as usize, m))
                },
            )?;
            infos.push(CollectionInfo {
                collection: coll,
                doc_count: count,
                last_modified: last_mod,
            });
        }

        Ok(IndexStatus {
            total_documents: total,
            needs_embedding: needs,
            has_vector_index: has_vec,
            collections: infos,
        })
    }

    // ── Maintenance ─────────────────────────────────────────────────────

    /// Delete inactive documents and orphaned content/vectors.
    pub fn cleanup(&self) -> Result<usize> {
        let c1 = self
            .conn
            .execute("DELETE FROM documents WHERE active = 0", [])?;
        let c2 = self.conn.execute(
            "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
        )?;
        let orphans: Vec<i64> = {
            let mut stmt = self.conn.prepare(
                r"SELECT cv.rowid FROM content_vectors cv
                  WHERE cv.hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            )?;
            stmt.query_map([], |row| row.get(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        for rid in &orphans {
            let _ = self
                .conn
                .execute("DELETE FROM vec_embeddings WHERE rowid = ?1", params![rid]);
        }
        let c3 = self.conn.execute(
            "DELETE FROM content_vectors WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
        )?;
        Ok(c1 + c2 + c3)
    }

    /// Clear all embeddings.
    pub fn clear_embeddings(&mut self) -> Result<usize> {
        let c = self.conn.execute("DELETE FROM content_vectors", [])?;
        let _ = self.conn.execute("DROP TABLE IF EXISTS vec_embeddings", []);
        self.dims = None;
        Ok(c)
    }

    /// Vacuum the database.
    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }
}

/// Parse a `store_collections` row into a [`Collection`].
fn row_to_collection(row: &rusqlite::Row<'_>) -> Collection {
    let name: String = row.get_unwrap(0);
    let path: String = row.get_unwrap(1);
    let pattern: String = row.get_unwrap(2);
    let ignore_raw: Option<String> = row.get_unwrap(3);
    let ctx_raw: Option<String> = row.get_unwrap(4);

    let ignore: Vec<String> = ignore_raw
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let context: HashMap<String, String> = ctx_raw
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();

    Collection {
        name,
        path,
        pattern,
        ignore,
        context,
    }
}

/// Extract a title from markdown content (H1/H2).
#[must_use]
pub fn extract_title(content: &str, filename: &str) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("# ")
            .or_else(|| trimmed.strip_prefix("## "))
        {
            let t = rest.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    let base = filename.rsplit(['/', '\\']).next().unwrap_or(filename);
    base.rfind('.')
        .map_or_else(|| base.to_string(), |i| base[..i].to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::unwrap_in_result)]
mod tests {
    use super::*;

    fn mem_db() -> Db {
        Db::open_memory().unwrap()
    }

    #[test]
    fn test_sqlite_vec_extension_is_available() {
        let db = mem_db();
        let version: String = db
            .conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(version.starts_with('v'));
    }

    #[test]
    fn test_hash_content() {
        let h1 = hash_content("hello");
        let h2 = hash_content("hello");
        assert_eq!(h1, h2);
        assert_ne!(hash_content("hello"), hash_content("world"));
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn test_extract_title_h1() {
        assert_eq!(extract_title("# My Title\nBody", "file.md"), "My Title");
    }

    #[test]
    fn test_extract_title_h2() {
        assert_eq!(extract_title("## Sub Title\nBody", "file.md"), "Sub Title");
    }

    #[test]
    fn test_extract_title_fallback() {
        assert_eq!(extract_title("No headings here", "notes.md"), "notes");
    }

    #[test]
    fn test_collection_crud() {
        let db = mem_db();

        let coll = Collection {
            name: "docs".into(),
            path: "/tmp/docs".into(),
            pattern: "**/*.md".into(),
            ..Default::default()
        };
        db.upsert_collection(&coll).unwrap();

        let got = db.get_collection("docs").unwrap().unwrap();
        assert_eq!(got.name, "docs");
        assert_eq!(got.path, "/tmp/docs");
        assert_eq!(got.pattern, "**/*.md");

        let all = db.list_collections().unwrap();
        assert_eq!(all.len(), 1);

        db.rename_collection("docs", "documents").unwrap();
        assert!(db.get_collection("docs").unwrap().is_none());
        assert!(db.get_collection("documents").unwrap().is_some());

        db.delete_collection("documents").unwrap();
        assert!(db.list_collections().unwrap().is_empty());
    }

    #[test]
    fn test_rename_collection_conflict() {
        let db = mem_db();

        db.upsert_collection(&Collection {
            name: "a".into(),
            path: "/tmp/a".into(),
            ..Default::default()
        })
        .unwrap();
        db.upsert_collection(&Collection {
            name: "b".into(),
            path: "/tmp/b".into(),
            ..Default::default()
        })
        .unwrap();

        let err = db.rename_collection("a", "b").unwrap_err();
        assert!(matches!(err, Error::CollectionExists(_)));
    }

    #[test]
    fn test_context_management() {
        let db = mem_db();

        db.upsert_collection(&Collection {
            name: "docs".into(),
            path: "/tmp/docs".into(),
            ..Default::default()
        })
        .unwrap();

        assert!(db.set_context("docs", "/", "Root context").unwrap());
        assert!(db.set_context("docs", "/api", "API docs").unwrap());

        let ctx = db.context_for_path("docs", "api/auth.md").unwrap().unwrap();
        assert!(ctx.contains("Root context"));
        assert!(ctx.contains("API docs"));

        assert!(db.remove_context("docs", "/api").unwrap());
        let ctx2 = db.context_for_path("docs", "api/auth.md").unwrap().unwrap();
        assert!(ctx2.contains("Root context"));
        assert!(!ctx2.contains("API docs"));
    }

    #[test]
    fn test_global_context() {
        let db = mem_db();

        db.upsert_collection(&Collection {
            name: "docs".into(),
            path: "/tmp/docs".into(),
            ..Default::default()
        })
        .unwrap();

        db.set_global_context(Some("Global note")).unwrap();
        assert_eq!(db.global_context().unwrap().as_deref(), Some("Global note"));

        let ctx = db.context_for_path("docs", "any.md").unwrap().unwrap();
        assert!(ctx.contains("Global note"));

        db.set_global_context(None).unwrap();
        assert!(db.global_context().unwrap().is_none());
    }

    #[test]
    fn test_document_crud() {
        let db = mem_db();

        let hash = hash_content("# Hello\nWorld");
        db.insert_content(&hash, "# Hello\nWorld").unwrap();
        db.upsert_document("docs", "hello.md", "Hello", &hash)
            .unwrap();

        let doc = db.get_document("docs", "hello.md").unwrap().unwrap();
        assert_eq!(doc.title, "Hello");
        assert_eq!(doc.hash, hash);
        assert_eq!(doc.docid(), &hash[..6]);
        assert_eq!(doc.display_path(), "docs/hello.md");

        assert_eq!(db.doc_count().unwrap(), 1);

        db.deactivate("docs", "hello.md").unwrap();
        assert!(db.get_document("docs", "hello.md").unwrap().is_none());
        assert_eq!(db.doc_count().unwrap(), 0);
    }

    #[test]
    fn test_find_by_docid() {
        let db = mem_db();

        let hash = hash_content("test content");
        db.insert_content(&hash, "test content").unwrap();
        db.upsert_document("lib", "test.md", "Test", &hash).unwrap();

        let (coll, path) = db.find_by_docid(&hash[..6]).unwrap().unwrap();
        assert_eq!(coll, "lib");
        assert_eq!(path, "test.md");

        let prefixed = format!("#{}", &hash[..6]);
        let (coll2, _) = db.find_by_docid(&prefixed).unwrap().unwrap();
        assert_eq!(coll2, "lib");
    }

    #[test]
    fn test_active_paths() {
        let db = mem_db();

        let h1 = hash_content("a");
        let h2 = hash_content("b");
        db.insert_content(&h1, "a").unwrap();
        db.insert_content(&h2, "b").unwrap();
        db.upsert_document("c", "a.md", "A", &h1).unwrap();
        db.upsert_document("c", "b.md", "B", &h2).unwrap();

        let paths = db.active_paths("c").unwrap();
        assert_eq!(paths.len(), 2);

        db.deactivate("c", "a.md").unwrap();
        let paths2 = db.active_paths("c").unwrap();
        assert_eq!(paths2.len(), 1);
        assert_eq!(paths2[0], "b.md");
    }

    #[test]
    fn test_cleanup() {
        let db = mem_db();

        let hash = hash_content("orphan");
        db.insert_content(&hash, "orphan").unwrap();
        db.upsert_document("x", "f.md", "F", &hash).unwrap();
        db.deactivate("x", "f.md").unwrap();

        let cleaned = db.cleanup().unwrap();
        assert!(cleaned > 0);
    }

    #[test]
    fn test_status() {
        let db = mem_db();

        db.upsert_collection(&Collection {
            name: "docs".into(),
            path: "/tmp/docs".into(),
            ..Default::default()
        })
        .unwrap();

        let h = hash_content("content");
        db.insert_content(&h, "content").unwrap();
        db.upsert_document("docs", "file.md", "File", &h).unwrap();

        let s = db.status().unwrap();
        assert_eq!(s.total_documents, 1);
        assert_eq!(s.needs_embedding, 1);
        assert_eq!(s.collections.len(), 1);
        assert_eq!(s.collections[0].doc_count, 1);
    }

    #[test]
    fn test_fts_search() {
        let db = mem_db();

        let hash = hash_content("# Rust Ownership\nRust has a unique ownership model.");
        db.insert_content(
            &hash,
            "# Rust Ownership\nRust has a unique ownership model.",
        )
        .unwrap();
        db.upsert_document("docs", "rust.md", "Rust Ownership", &hash)
            .unwrap();

        let results = db.search_fts("\"rust\"*", 10, None).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc.title, "Rust Ownership");
        assert_eq!(results[0].source, SearchSource::Fts);
    }

    #[test]
    fn test_search_fts_offset_paginates() {
        let db = mem_db();
        for i in 0..25 {
            let body = format!("Rust ownership document {i}");
            let hash = hash_content(&body);
            db.insert_content(&hash, &body).unwrap();
            db.upsert_document("docs", &format!("{i}.md"), &body, &hash)
                .unwrap();
        }

        let first = db.search_fts_with_offset("\"rust\"*", 10, 0, None).unwrap();
        let second = db
            .search_fts_with_offset("\"rust\"*", 10, 10, None)
            .unwrap();
        let third = db
            .search_fts_with_offset("\"rust\"*", 10, 20, None)
            .unwrap();
        let beyond = db
            .search_fts_with_offset("\"rust\"*", 10, 25, None)
            .unwrap();

        assert_eq!(first.len(), 10);
        assert_eq!(second.len(), 10);
        assert_eq!(third.len(), 5);
        assert!(beyond.is_empty());
        assert!(
            first
                .iter()
                .map(|r| r.doc.path.as_str())
                .collect::<std::collections::HashSet<_>>()
                .is_disjoint(
                    &second
                        .iter()
                        .map(|r| r.doc.path.as_str())
                        .collect::<std::collections::HashSet<_>>()
                )
        );
    }

    #[test]
    fn test_partial_embedding_with_seq_zero_is_still_unembedded() {
        let mut db = mem_db();
        let body = "A document with multiple chunks";
        let hash = hash_content(body);
        db.insert_content(&hash, body).unwrap();
        db.upsert_document("docs", "partial.md", "Partial", &hash)
            .unwrap();
        db.insert_embedding(&hash, 0, 0, &[0.0_f32; 3], "default")
            .unwrap();

        assert_eq!(db.unembedded_docs().unwrap().len(), 1);
    }

    #[test]
    fn test_unembedded_docs_respects_limit() {
        let db = mem_db();
        for i in 0..3 {
            let body = format!("document {i}");
            let hash = hash_content(&body);
            db.insert_content(&hash, &body).unwrap();
            db.upsert_document("docs", &format!("{i}.md"), "Doc", &hash)
                .unwrap();
        }
        assert_eq!(db.unembedded_docs_with_limit(Some(2)).unwrap().len(), 2);
    }

    #[test]
    fn test_needs_embedding_count() {
        let db = mem_db();

        let h = hash_content("some text");
        db.insert_content(&h, "some text").unwrap();
        db.upsert_document("c", "f.md", "F", &h).unwrap();

        assert_eq!(db.needs_embedding_count().unwrap(), 1);
    }
}
