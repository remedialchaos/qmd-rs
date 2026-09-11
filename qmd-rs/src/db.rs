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

use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
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

/// Insert an embedding row using the supplied SQLite connection.
fn insert_embedding_on_conn(
    conn: &Connection,
    hash: &str,
    seq: usize,
    pos: usize,
    embedding: &[f32],
    model: &str,
) -> Result<()> {
    let now = now_rfc3339();
    let existing_rowid: Option<i64> = conn
        .query_row(
            "SELECT rowid FROM content_vectors WHERE hash = ?1 AND seq = ?2",
            params![hash, seq as i64],
            |row| row.get(0),
        )
        .optional()?;
    let vec_bytes = embedding.as_bytes();
    if let Some(rid) = existing_rowid {
        conn.execute(
            "UPDATE vec_embeddings SET embedding = ?1 WHERE rowid = ?2",
            params![vec_bytes, rid],
        )?;
        conn.execute(
            "UPDATE content_vectors SET pos = ?1, model = ?2, embedded_at = ?3 WHERE hash = ?4 AND seq = ?5",
            params![pos as i64, model, now, hash, seq as i64],
        )?;
    } else {
        conn.execute(
            "INSERT INTO vec_embeddings (embedding) VALUES (?1)",
            params![vec_bytes],
        )?;
        let rowid = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO content_vectors (rowid, hash, seq, pos, model, embedded_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![rowid, hash, seq as i64, pos as i64, model, now],
        )?;
    }
    Ok(())
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
    /// Embedding contract fingerprint, if established.
    pub embedding_fingerprint: Option<String>,
    /// Compatibility between stored vectors and the active runtime contract.
    pub embedding_compatibility: EmbeddingCompatibility,
    /// Per-collection info.
    pub collections: Vec<CollectionInfo>,
}

/// Compatibility state for stored embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmbeddingCompatibility {
    /// No vectors are stored.
    NotPresent,
    /// Vectors match the active embedding contract.
    Compatible,
    /// Legacy vectors exist without a persisted contract.
    MissingLegacy,
    /// Stored vectors were produced under a different contract.
    Mismatched,
    /// A low-level caller did not provide an active contract for comparison.
    Unknown,
}

/// Severity of one read-only index diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DoctorCheckStatus {
    /// The check passed.
    Ok,
    /// The index is usable, but corrective action is recommended.
    Warning,
    /// The index is inconsistent or unsafe to use for the checked feature.
    Error,
}

/// One stable, machine-readable diagnostic result.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct DoctorCheck {
    /// Stable check identifier.
    pub name: &'static str,
    /// Check outcome.
    pub status: DoctorCheckStatus,
    /// Human-readable evidence and recovery guidance.
    pub detail: String,
}

/// Complete read-only index diagnostic report.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct DoctorReport {
    /// Checks in stable presentation order.
    pub checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    /// Whether any diagnostic reported an error.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == DoctorCheckStatus::Error)
    }
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

    /// Open an existing index without migrations or write access.
    pub fn open_read_only(path: &Path) -> Result<Self> {
        Self::register_sqlite_vec();
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Self { conn, dims: None })
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

    /// Create FTS synchronization triggers if absent, and upgrade a stale
    /// `documents_au` trigger on existing databases.
    ///
    /// Older qmd versions installed a `documents_au` whose delete arm was
    /// guarded by `new.active = 0`, so re-indexing a document through
    /// `INSERT ... ON CONFLICT DO UPDATE` left the stale FTS row in place and
    /// the subsequent `INSERT OR REPLACE` failed with SQLITE_CONSTRAINT
    /// (primary-key/rowid conflict in the FTS5 shadow table). Existing
    /// databases carry the old trigger, so the check must compare trigger
    /// SQL, not just presence of `documents_ai`.
    fn ensure_fts_triggers(&self) -> Result<()> {
        const AU_SQL: &str = r"CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id;
                    INSERT INTO documents_fts(rowid, filepath, title, body)
                    SELECT new.id,
                           new.collection || '/' || new.path,
                           new.title,
                           (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;";

        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='documents_ai'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if exists {
            // An existing DB may still carry the broken documents_au; repair
            // it only when its definition differs from the current one.
            let au_sql: Option<String> = self
                .conn
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='documents_au'",
                    [],
                    |row| row.get(0),
                )
                .optional()?;
            if au_sql.as_deref() != Some(AU_SQL) {
                self.conn
                    .execute_batch(&format!("DROP TRIGGER IF EXISTS documents_au; {AU_SQL};"))?;
            }
            return Ok(());
        }

        self.conn.execute_batch(&format!(
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

                {AU_SQL};
                ",
        ))?;
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
              ORDER BY score, d.collection, d.path, d.hash
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
        _hash: &str,
        _seq: usize,
        _pos: usize,
        _embedding: &[f32],
        _model: &str,
    ) -> Result<()> {
        Err(Error::Config(
            "embedding appends require the expected fingerprint; run qmd embed --force or use insert_embedding_with_fingerprint".into(),
        ))
    }

    /// Append one embedding only when it matches the established contract.
    pub fn insert_embedding_with_fingerprint(
        &mut self,
        hash: &str,
        seq: usize,
        pos: usize,
        embedding: &[f32],
        model: &str,
        expected_fingerprint: &str,
    ) -> Result<()> {
        self.validate_embedding_fingerprint(expected_fingerprint)?;
        if self.dims.is_none() {
            self.dims = Some(embedding.len());
            self.ensure_vec_table(embedding.len())?;
        } else if self.dims != Some(embedding.len()) {
            return Err(Error::Config("embedding dimensions do not match".into()));
        }
        let tx = self.conn.transaction()?;
        insert_embedding_on_conn(&tx, hash, seq, pos, embedding, model)?;
        tx.execute(
            "INSERT INTO store_config(key, value) VALUES ('embedding_fingerprint', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![expected_fingerprint],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Replace/add a batch and publish its fingerprint in one transaction.
    pub fn replace_embeddings_transactionally(
        &mut self,
        fingerprint: &str,
        embeddings: &[(&str, usize, usize, Vec<f32>)],
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        self.validate_embedding_fingerprint(fingerprint)?;
        let dims = embeddings[0].3.len();
        if embeddings.iter().any(|(_, _, _, e)| e.len() != dims) {
            return Err(Error::Config("embedding dimensions do not match".into()));
        }
        if self.dims.is_none() {
            self.dims = Some(dims);
            self.ensure_vec_table(dims)?;
        }
        let tx = self.conn.transaction()?;
        for (hash, seq, pos, embedding) in embeddings {
            let model = if *seq == 0 {
                "default:complete"
            } else {
                "default"
            };
            insert_embedding_on_conn(&tx, hash, *seq, *pos, embedding, model)?;
        }
        tx.execute(
            "INSERT INTO store_config(key, value) VALUES ('embedding_fingerprint', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![fingerprint],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Count stored embedding chunks.
    pub fn vector_count(&self) -> Result<usize> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |row| {
                row.get::<_, i64>(0).map(|v| v as usize)
            })?)
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
    ///
    /// Callers must provide the embedding contract fingerprint when vectors exist.
    pub fn search_vec(
        &self,
        _query_embedding: &[f32],
        _limit: usize,
        _collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if self.vector_count()? > 0 {
            return Err(Error::Config(
                "vector search requires the expected fingerprint; use search_vec_with_fingerprint or run qmd embed --force".into(),
            ));
        }
        Ok(Vec::new())
    }

    /// Vector search after validating the embedding contract.
    pub fn search_vec_with_fingerprint(
        &self,
        query_embedding: &[f32],
        limit: usize,
        collection: Option<&str>,
        expected_fingerprint: &str,
    ) -> Result<Vec<SearchResult>> {
        self.validate_embedding_fingerprint(expected_fingerprint)?;

        let vec_bytes = query_embedding.as_bytes();

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

        if let Some(coll) = collection {
            if self.vector_count()? == 0 {
                return Ok(Vec::new());
            }
            // Rank only active documents in the selected collection, one row
            // per (collection, path) keeping each document's best chunk by
            // distance, so duplicates cannot crowd out distinct documents
            // before the requested limit. Native KNN caps k at 4096 and
            // truncates ties before our document ordering; scalar L2 matches
            // vec0's default metric without either limitation.
            let scoped_sql = r"SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                     LENGTH(c.doc), MIN(vec_distance_L2(ve.embedding, ?1)) AS distance
              FROM documents d
              JOIN content_vectors cv ON cv.hash = d.hash
              JOIN vec_embeddings ve ON ve.rowid = cv.rowid
              JOIN content c ON c.hash = d.hash
              WHERE d.active = 1 AND d.collection = ?3
              GROUP BY d.collection, d.path
              ORDER BY distance, d.collection, d.path, d.hash
              LIMIT ?2";
            let mut stmt = self.conn.prepare(scoped_sql)?;
            let mut results: Vec<SearchResult> = stmt
                .query_map(params![vec_bytes, limit as i64, coll], map_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            results.truncate(limit);
            return Ok(results);
        }

        // Unscoped search uses the same exact active-document scalar L2
        // aggregation. Native KNN is not usable here: it caps k at 4096 and
        // truncates ties before active filtering or document aggregation, so
        // inactive or duplicate chunk rows can crowd distinct active
        // documents out of any fixed window. Exact aggregation over active
        // documents satisfies the contract without an oversampling constant
        // or best-effort cap. Performance optimization is deferred.
        if self.vector_count()? == 0 {
            return Ok(Vec::new());
        }
        let unscoped_sql = r"SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                     LENGTH(c.doc), MIN(vec_distance_L2(ve.embedding, ?1)) AS distance
              FROM documents d
              JOIN content_vectors cv ON cv.hash = d.hash
              JOIN vec_embeddings ve ON ve.rowid = cv.rowid
              JOIN content c ON c.hash = d.hash
              WHERE d.active = 1
              GROUP BY d.collection, d.path
              ORDER BY distance, d.collection, d.path, d.hash
              LIMIT ?2";
        let mut stmt = self.conn.prepare(unscoped_sql)?;
        let results: Vec<SearchResult> = stmt
            .query_map(params![vec_bytes, limit as i64], map_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

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
        self.status_inner(None)
    }

    /// Get index status compared with the active embedding contract.
    pub fn status_with_expected_fingerprint(&self, expected: &str) -> Result<IndexStatus> {
        self.status_inner(Some(expected))
    }

    /// Build status with an optional active fingerprint comparison.
    fn status_inner(&self, active_expected: Option<&str>) -> Result<IndexStatus> {
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

        let embedding_fingerprint = self.embedding_fingerprint()?;
        let embedding_compatibility = match (
            self.vector_count()?,
            embedding_fingerprint.as_deref(),
            active_expected,
        ) {
            (0, _, _) => EmbeddingCompatibility::NotPresent,
            (_, None, _) => EmbeddingCompatibility::MissingLegacy,
            (_, Some(actual), Some(candidate)) if actual == candidate => {
                EmbeddingCompatibility::Compatible
            }
            (_, Some(_), Some(_)) => EmbeddingCompatibility::Mismatched,
            _ => EmbeddingCompatibility::Unknown,
        };
        Ok(IndexStatus {
            total_documents: total,
            needs_embedding: needs,
            has_vector_index: has_vec,
            embedding_fingerprint,
            embedding_compatibility,
            collections: infos,
        })
    }

    /// Run stable, read-only index diagnostics.
    pub fn doctor(&self, expected_fingerprint: &str, expected_dims: usize) -> Result<DoctorReport> {
        let mut checks = Vec::with_capacity(8);
        let quick: String = self
            .conn
            .query_row("PRAGMA quick_check", [], |row| row.get(0))?;
        checks.push(DoctorCheck {
            name: "sqlite_quick_check",
            status: if quick == "ok" {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Error
            },
            detail: quick,
        });

        let missing_fts: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents d LEFT JOIN documents_fts f ON f.rowid = d.id WHERE d.active = 1 AND f.rowid IS NULL",
            [],
            |row| row.get(0),
        )?;
        let extra_fts: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents_fts f LEFT JOIN documents d ON d.id = f.rowid AND d.active = 1 WHERE d.id IS NULL",
            [],
            |row| row.get(0),
        )?;
        checks.push(DoctorCheck {
            name: "fts_documents",
            status: if missing_fts == 0 && extra_fts == 0 {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Error
            },
            detail: format!("missing={missing_fts}, extra={extra_fts}"),
        });

        let orphan_content: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM content c WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.hash = c.hash AND d.active = 1)",
            [],
            |row| row.get(0),
        )?;
        checks.push(DoctorCheck {
            name: "orphan_content",
            status: if orphan_content == 0 {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Warning
            },
            detail: format!("{orphan_content} unreferenced content rows; run qmd cleanup"),
        });

        let orphan_vectors: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM content_vectors v WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.hash = v.hash AND d.active = 1)",
            [],
            |row| row.get(0),
        )?;
        checks.push(DoctorCheck {
            name: "orphan_vectors",
            status: if orphan_vectors == 0 {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Error
            },
            detail: format!(
                "{orphan_vectors} vector rows lack an active document; run qmd cleanup"
            ),
        });

        let collections = self.list_collections()?;
        let missing_paths: Vec<&str> = collections
            .iter()
            .filter(|collection| !Path::new(&collection.path).is_dir())
            .map(|collection| collection.name.as_str())
            .collect();
        checks.push(DoctorCheck {
            name: "collection_paths",
            status: if missing_paths.is_empty() {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Warning
            },
            detail: if missing_paths.is_empty() {
                format!(
                    "{} registered collection paths are accessible",
                    collections.len()
                )
            } else {
                format!(
                    "missing or unreadable collection paths: {}",
                    missing_paths.join(", ")
                )
            },
        });

        let incomplete = self.needs_embedding_count()?;
        checks.push(DoctorCheck {
            name: "embedding_completeness",
            status: if incomplete == 0 {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Warning
            },
            detail: format!("{incomplete} active documents need embedding; run qmd embed"),
        });

        let vector_count = self.vector_count()?;
        let vec_table = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_embeddings'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);
        let stored_dims = if vector_count > 0 && vec_table {
            self.conn
                .query_row(
                    "SELECT length(embedding) / 4 FROM vec_embeddings LIMIT 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .map(|value| value as usize)
        } else {
            None
        };
        checks.push(DoctorCheck {
            name: "vector_dimensions",
            status: if vector_count == 0 || stored_dims == Some(expected_dims) {
                DoctorCheckStatus::Ok
            } else {
                DoctorCheckStatus::Error
            },
            detail: if vector_count == 0 {
                "no vectors stored".to_string()
            } else if let Some(actual) = stored_dims {
                format!("stored={actual}, expected={expected_dims}; run qmd embed --force")
            } else {
                "vector metadata exists but vec_embeddings is missing; run qmd embed --force"
                    .to_string()
            },
        });

        let stored_fingerprint = self.embedding_fingerprint()?;
        checks.push(DoctorCheck {
            name: "embedding_fingerprint",
            status: match stored_fingerprint.as_deref() {
                _ if vector_count == 0 => DoctorCheckStatus::Ok,
                Some(actual) if actual == expected_fingerprint => DoctorCheckStatus::Ok,
                _ => DoctorCheckStatus::Error,
            },
            detail: match stored_fingerprint {
                _ if vector_count == 0 => "no vectors stored".to_string(),
                Some(actual) if actual == expected_fingerprint => "compatible".to_string(),
                Some(actual) => format!(
                    "stored {actual}, expected {expected_fingerprint}; run qmd embed --force"
                ),
                None => "legacy vectors have no fingerprint; run qmd embed --force".to_string(),
            },
        });

        Ok(DoctorReport { checks })
    }

    // ── Maintenance ─────────────────────────────────────────────────────

    /// Read the persisted embedding contract fingerprint, if established.
    pub fn embedding_fingerprint(&self) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT value FROM store_config WHERE key = 'embedding_fingerprint'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Establish the embedding contract only while no vectors exist.
    ///
    /// Once vectors exist, changing this value would relabel incompatible data.
    /// Use [`Db::clear_embeddings`] before establishing a different fingerprint.
    pub fn set_embedding_fingerprint(&self, fingerprint: &str) -> Result<()> {
        if self.vector_count()? > 0 {
            return match self.embedding_fingerprint()? {
                Some(actual) if actual == fingerprint => Ok(()),
                _ => Err(Error::Config(
                    "cannot change the embedding fingerprint while existing vectors remain; run qmd embed --force"
                        .into(),
                )),
            };
        }
        self.conn.execute(
            "INSERT INTO store_config(key, value) VALUES ('embedding_fingerprint', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![fingerprint],
        )?;
        Ok(())
    }

    /// Reject vectors that were produced under another or unknown contract.
    pub fn validate_embedding_fingerprint(&self, expected: &str) -> Result<()> {
        let vectors: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM content_vectors", [], |row| row.get(0))?;
        match self.embedding_fingerprint()? {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(Error::Config(format!(
                "embedding fingerprint mismatch (stored {actual}, expected {expected}); run qmd embed --force"
            ))),
            None if vectors > 0 => Err(Error::Config(
                "legacy embeddings have no trustworthy fingerprint; run qmd embed --force".into(),
            )),
            None => Ok(()),
        }
    }

    // ── Maintenance ────────────────────────────────────────────────────

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
        self.conn.execute(
            "DELETE FROM store_config WHERE key = 'embedding_fingerprint'",
            [],
        )?;
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
    fn test_upsert_document_updates_fts() {
        let db = mem_db();

        let old_hash = hash_content("apples and oranges");
        db.insert_content(&old_hash, "apples and oranges").unwrap();
        db.upsert_document("docs", "fruit.md", "Fruit", &old_hash)
            .unwrap();

        let hits = db.search_fts("apples", 10, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc.path, "fruit.md");

        // Re-ingest the same (collection, path) with new content: the UPSERT
        // takes the DO UPDATE arm, which fires documents_au. The FTS row for
        // the old body must be replaced, not duplicated or stale.
        let new_hash = hash_content("zebras and giraffes");
        db.insert_content(&new_hash, "zebras and giraffes").unwrap();
        db.upsert_document("docs", "fruit.md", "Savanna", &new_hash)
            .unwrap();

        let hits_new = db.search_fts("zebras", 10, None).unwrap();
        assert_eq!(
            hits_new.len(),
            1,
            "updated doc must be findable by new body"
        );
        assert_eq!(hits_new[0].doc.title, "Savanna");
        assert_eq!(hits_new[0].doc.hash, new_hash);

        let hits_old = db.search_fts("apples", 10, None).unwrap();
        assert!(hits_old.is_empty(), "old body must no longer match");

        // FTS/document consistency: exactly one FTS row per active document.
        let fts_rows: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM documents_fts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts_rows, 1);
    }

    #[test]
    fn test_fts_trigger_upgrade_on_existing_db() {
        // Simulate an existing on-disk DB created by an older qmd that
        // installed the broken documents_au trigger. ensure_fts_triggers()
        // must upgrade it, not skip it.
        let dir = std::env::temp_dir().join(format!("qmd_fts_upgrade_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("legacy.sqlite");

        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                r"
                CREATE TABLE documents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection  TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    hash        TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    active      INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(collection, path)
                );
                CREATE VIRTUAL TABLE documents_fts USING fts5(filepath, title, body);
                CREATE TRIGGER documents_ai AFTER INSERT ON documents
                WHEN new.active = 1
                BEGIN
                    INSERT INTO documents_fts(rowid, filepath, title, body)
                    SELECT new.id, new.collection || '/' || new.path,
                           new.title, 'legacy body'
                    WHERE new.active = 1;
                END;
                CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;
                    INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
                    SELECT new.id, new.collection || '/' || new.path,
                           new.title, 'legacy body'
                    WHERE new.active = 1;
                END;
                INSERT INTO documents (collection, path, title, hash,
                                       created_at, modified_at, active)
                VALUES ('docs', 'legacy.md', 'Legacy', 'legacyhash',
                        't0', 't0', 1);
                ",
            )
            .unwrap();
        }

        // Opening the existing DB must repair the trigger (migration path).
        let db = Db::open(&db_path).unwrap();

        let old_hash = hash_content("apples and oranges");
        db.insert_content(&old_hash, "apples and oranges").unwrap();
        db.upsert_document("docs", "legacy.md", "Repaired", &old_hash)
            .unwrap();

        let hits = db.search_fts("apples", 10, None).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "upsert on upgraded DB must succeed and index"
        );

        let trigger_sql: String = db
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='documents_au'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        // The repaired trigger must unconditionally remove the old FTS row
        // before inserting the new one.
        assert!(
            trigger_sql.contains("DELETE FROM documents_fts WHERE rowid = old.id;")
                && !trigger_sql.contains("INSERT OR REPLACE"),
            "documents_au must delete-then-insert, got: {trigger_sql}"
        );

        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_dir_all(&dir);
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
        db.insert_embedding_with_fingerprint(
            &hash,
            0,
            0,
            &[0.0_f32; 3],
            "default",
            &crate::embed::embedding_fingerprint(3, 3200, 480),
        )
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
    fn collection_filter_and_tie_pagination_are_stable() {
        let db = mem_db();
        for (collection, path) in [("a", "z.md"), ("a", "a.md"), ("b", "b.md")] {
            let body = "same rust text";
            let hash = hash_content(&format!("{collection}/{path}"));
            db.insert_content(&hash, body).unwrap();
            db.upsert_document(collection, path, "same", &hash).unwrap();
        }
        let page = db
            .search_fts_with_offset("\"rust\"*", 1, 0, Some("a"))
            .unwrap();
        let next = db
            .search_fts_with_offset("\"rust\"*", 1, 1, Some("a"))
            .unwrap();
        assert_eq!(page[0].doc.collection, "a");
        assert_eq!(next[0].doc.collection, "a");
        assert_ne!(page[0].doc.path, next[0].doc.path);
        assert_eq!(page[0].doc.path, "a.md");
    }

    #[test]
    fn collection_filter_covers_selected_hits_beyond_global_fetch_cap() {
        let db = mem_db();
        for i in 0..5 {
            let body = "same rust text";
            let hash = hash_content(&format!("other/{i}"));
            db.insert_content(&hash, body).unwrap();
            db.upsert_document("other", &format!("{i}.md"), "same", &hash)
                .unwrap();
        }
        for i in 0..2 {
            let body = "same rust text";
            let hash = hash_content(&format!("selected/{i}"));
            db.insert_content(&hash, body).unwrap();
            db.upsert_document("selected", &format!("{i}.md"), "same", &hash)
                .unwrap();
        }

        let all = db.search_fts_with_offset("\"rust\"*", 7, 0, Some("selected"));
        assert_eq!(all.unwrap().len(), 2);
    }

    #[test]
    fn append_rejects_mismatched_embedding_fingerprint() {
        let mut db = mem_db();
        let expected = crate::embed::embedding_fingerprint(3, 3200, 480);
        let other = crate::embed::embedding_fingerprint(4, 3200, 480);
        db.set_embedding_fingerprint(&expected).unwrap();
        let err = db
            .insert_embedding_with_fingerprint("hash", 0, 0, &[0.0; 3], "default", &other)
            .unwrap_err();
        assert!(err.to_string().contains("fingerprint mismatch"));
    }

    #[test]
    fn fresh_checked_append_publishes_fingerprint_atomically() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(3, 3200, 480);
        db.insert_embedding_with_fingerprint("fresh", 0, 0, &[0.0; 3], "default", &fingerprint)
            .unwrap();
        assert_eq!(
            db.embedding_fingerprint().unwrap().as_deref(),
            Some(fingerprint.as_str())
        );
        assert_eq!(db.vector_count().unwrap(), 1);
    }

    #[test]
    fn legacy_checked_append_is_rejected() {
        let mut db = mem_db();
        db.conn.execute(
            "INSERT INTO content_vectors(hash, seq, pos, rowid, model, embedded_at) VALUES ('legacy', 0, 0, 1, 'default', 'legacy')",
            [],
        ).unwrap();
        let err = db
            .insert_embedding_with_fingerprint("new", 0, 0, &[0.0; 3], "default", "expected")
            .unwrap_err();
        assert!(err.to_string().contains("legacy embeddings"));
    }

    #[test]
    fn ungated_vector_search_is_rejected_when_vectors_exist() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(3, 3200, 480);
        db.insert_embedding_with_fingerprint("fresh", 0, 0, &[0.0; 3], "default", &fingerprint)
            .unwrap();
        let err = db.search_vec(&[0.0; 3], 1, None).unwrap_err();
        assert!(err.to_string().contains("expected fingerprint"));
    }

    #[test]
    fn gated_vector_search_rejects_mismatch() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(3, 3200, 480);
        db.insert_embedding_with_fingerprint("fresh", 0, 0, &[0.0; 3], "default", &fingerprint)
            .unwrap();
        let err = db
            .search_vec_with_fingerprint(&[0.0; 3], 1, None, "other")
            .unwrap_err();
        assert!(err.to_string().contains("fingerprint mismatch"));
    }

    #[test]
    fn transactional_embedding_failure_does_not_publish_fingerprint() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(3, 3200, 480);
        let err = db
            .replace_embeddings_transactionally(
                &fingerprint,
                &[("hash", 0, 0, vec![0.0; 3]), ("hash", 1, 1, vec![0.0; 2])],
            )
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        assert_eq!(db.embedding_fingerprint().unwrap(), None);
        assert_eq!(db.vector_count().unwrap(), 0);
    }

    #[test]
    fn collection_vector_knn_cap_does_not_hide_selected_hits() {
        let mut db = mem_db();
        db.upsert_collection(&Collection::new("other", "/tmp/other"))
            .unwrap();
        db.upsert_collection(&Collection::new("selected", "/tmp/selected"))
            .unwrap();
        let other_hash = hash_content("other");
        let selected_hash = hash_content("selected");
        for (collection, path, hash) in [
            ("other", "near.md", &other_hash),
            ("selected", "far.md", &selected_hash),
        ] {
            db.insert_content(hash, collection).unwrap();
            db.upsert_document(collection, path, path, hash).unwrap();
        }
        let fingerprint = crate::embed::embedding_fingerprint(3, 3200, 480);
        db.insert_embedding_with_fingerprint(
            &other_hash,
            0,
            0,
            &[0.0, 0.0, 0.0],
            "default",
            &fingerprint,
        )
        .unwrap();
        db.insert_embedding_with_fingerprint(
            &selected_hash,
            0,
            0,
            &[1.0, 0.0, 0.0],
            "default",
            &fingerprint,
        )
        .unwrap();

        // Every distractor is closer than the selected hit, so a global
        // top-4096 prefilter is incorrect as well as a k above sqlite-vec's cap.
        for seq in 1..4096 {
            db.insert_embedding_with_fingerprint(
                &other_hash,
                seq,
                0,
                &[0.0, 0.0, 0.0],
                "default",
                &fingerprint,
            )
            .unwrap();
        }
        assert_eq!(db.vector_count().unwrap(), 4097);

        let hits = db
            .search_vec_with_fingerprint(&[0.0, 0.0, 0.0], 1, Some("selected"), &fingerprint)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc.collection, "selected");
    }

    #[test]
    fn status_exposes_embedding_fingerprint_state() {
        let mut db = mem_db();
        assert_eq!(db.status().unwrap().embedding_fingerprint, None);
        let fingerprint = crate::embed::embedding_fingerprint(3, 100, 10);
        db.insert_embedding_with_fingerprint("fresh", 0, 0, &[0.0; 3], "default", &fingerprint)
            .unwrap();
        assert_eq!(
            db.status().unwrap().embedding_fingerprint,
            Some(fingerprint)
        );
    }

    #[test]
    fn status_json_reports_fingerprint_compatibility_states() {
        let mut db = mem_db();
        let expected = crate::embed::embedding_fingerprint(3, 100, 10);
        assert_eq!(
            db.status_with_expected_fingerprint(&expected)
                .unwrap()
                .embedding_compatibility,
            EmbeddingCompatibility::NotPresent
        );
        db.insert_embedding_with_fingerprint("fresh", 0, 0, &[0.0; 3], "default", &expected)
            .unwrap();
        assert_eq!(
            db.status_with_expected_fingerprint(&expected)
                .unwrap()
                .embedding_compatibility,
            EmbeddingCompatibility::Compatible
        );
        assert_eq!(
            db.status_with_expected_fingerprint("other")
                .unwrap()
                .embedding_compatibility,
            EmbeddingCompatibility::Mismatched
        );
        let json =
            serde_json::to_value(db.status_with_expected_fingerprint("other").unwrap()).unwrap();
        assert_eq!(json["embedding_compatibility"], "mismatched");
    }
    #[test]
    fn fingerprint_is_deterministic_and_persisted() {
        let db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(384, 3200, 480);
        assert_eq!(
            fingerprint,
            crate::embed::embedding_fingerprint(384, 3200, 480)
        );
        assert_ne!(
            fingerprint,
            crate::embed::embedding_fingerprint(385, 3200, 480)
        );
        assert_eq!(db.embedding_fingerprint().unwrap(), None);
        db.set_embedding_fingerprint(&fingerprint).unwrap();
        assert_eq!(db.embedding_fingerprint().unwrap(), Some(fingerprint));
    }

    #[test]
    fn collection_vector_search_returns_each_document_once_before_limit() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        // Two nearby chunks in one document and one chunk in another; a limit
        // of 2 must surface each (collection, path) exactly once, with the
        // duplicate-chunk document not crowding out the other document.
        for (path, vector) in [("dup.md", [0.05f32, 0.0]), ("other.md", [0.2, 0.0])] {
            let hash = hash_content(path);
            db.insert_content(&hash, path).unwrap();
            db.upsert_document("docs", path, path, &hash).unwrap();
            db.insert_embedding_with_fingerprint(&hash, 0, 0, &vector, "default", &fingerprint)
                .unwrap();
        }
        let dup = hash_content("dup.md");
        db.insert_embedding_with_fingerprint(&dup, 1, 0, &[0.1, 0.0], "default", &fingerprint)
            .unwrap();

        let hits = db
            .search_vec_with_fingerprint(&[0.0, 0.0], 2, Some("docs"), &fingerprint)
            .unwrap();
        let identities: Vec<(String, String)> = hits
            .iter()
            .map(|h| (h.doc.collection.clone(), h.doc.path.clone()))
            .collect();
        assert_eq!(
            identities,
            [
                ("docs".to_string(), "dup.md".to_string()),
                ("docs".to_string(), "other.md".to_string()),
            ],
            "each document identity must appear at most once: {identities:?}"
        );
        // Best (nearest) chunk of each document is retained.
        assert!((hits[0].score - 0.95).abs() < 1e-6);
        assert!((hits[1].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn collection_vector_search_filters_before_limit_and_preserves_limit() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        for (collection, path, vector) in [
            ("other", "nearest.md", [1.0, 0.0]),
            ("selected", "first.md", [0.9, 0.1]),
            ("selected", "second.md", [0.8, 0.2]),
        ] {
            let body = format!("{collection}/{path}");
            let hash = hash_content(&body);
            db.insert_content(&hash, &body).unwrap();
            db.upsert_document(collection, path, path, &hash).unwrap();
            db.insert_embedding_with_fingerprint(&hash, 0, 0, &vector, "default", &fingerprint)
                .unwrap();
        }

        let hits = db
            .search_vec_with_fingerprint(&[1.0, 0.0], 1, Some("selected"), &fingerprint)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc.collection, "selected");
        assert_eq!(hits[0].doc.path, "first.md");
    }

    #[test]
    fn collection_vector_search_orders_ties_before_large_limits() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        // Reverse insertion order makes rowid-based tie selection incorrect.
        for i in (0..4097).rev() {
            let path = format!("{i:04}.md");
            db.insert_content(&path, &path).unwrap();
            db.upsert_document("selected", &path, &path, &path).unwrap();
            db.insert_embedding_with_fingerprint(&path, 0, 0, &[1.0, 0.0], "default", &fingerprint)
                .unwrap();
        }
        // Shared content must not leak a document from another collection.
        db.upsert_document("other", "0000.md", "shared", "0000.md")
            .unwrap();
        db.upsert_document("selected", "inactive.md", "inactive", "0000.md")
            .unwrap();
        db.conn
            .execute(
                "UPDATE documents SET active = 0 WHERE path = 'inactive.md'",
                [],
            )
            .unwrap();

        for limit in [0, 1, 4096, 4097, 5000] {
            let hits = db
                .search_vec_with_fingerprint(&[0.0, 0.0], limit, Some("selected"), &fingerprint)
                .unwrap();
            let expected: Vec<_> = (0..limit.min(4097)).map(|i| format!("{i:04}.md")).collect();
            assert_eq!(
                hits.iter().map(|hit| &hit.doc.path).collect::<Vec<_>>(),
                expected.iter().collect::<Vec<_>>()
            );
            assert!(hits.iter().all(|hit| hit.doc.collection == "selected"));
            assert!(hits.iter().all(|hit| hit.score.abs() < f64::EPSILON));
        }
        assert!(
            db.search_vec_with_fingerprint(&[0.0, 0.0], 10, Some("missing"), &fingerprint)
                .unwrap()
                .is_empty()
        );
        assert_eq!(db.vector_count().unwrap(), 4097);
        assert_eq!(db.doc_count().unwrap(), 4098);
    }

    #[test]
    fn unscoped_vector_search_returns_one_result_per_document_identity() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        let alpha = hash_content("alpha");
        let beta = hash_content("beta");
        db.insert_content(&alpha, "alpha").unwrap();
        db.upsert_document("docs", "alpha.md", "alpha", &alpha)
            .unwrap();
        db.insert_content(&beta, "beta").unwrap();
        db.upsert_document("docs", "beta.md", "beta", &beta)
            .unwrap();
        for (seq, vector) in [(0usize, [0.1f32, 0.0]), (1, [0.12, 0.0]), (2, [0.9, 0.0])] {
            db.insert_embedding_with_fingerprint(&alpha, seq, 0, &vector, "default", &fingerprint)
                .unwrap();
        }
        db.insert_embedding_with_fingerprint(&beta, 0, 0, &[0.2, 0.0], "default", &fingerprint)
            .unwrap();

        let hits = db
            .search_vec_with_fingerprint(&[0.0, 0.0], 2, None, &fingerprint)
            .unwrap();
        let identities: Vec<(String, String)> = hits
            .iter()
            .map(|h| (h.doc.collection.clone(), h.doc.path.clone()))
            .collect();
        assert_eq!(
            identities,
            [
                ("docs".to_string(), "alpha.md".to_string()),
                ("docs".to_string(), "beta.md".to_string())
            ],
            "each document identity must appear at most once: {identities:?}"
        );
        // The best (nearest) chunk of each document is retained.
        assert!((hits[0].score - 0.9).abs() < 1e-6);
        assert!((hits[1].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn unscoped_vector_search_does_not_lose_active_docs_behind_inactive_rows() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        let inactive = hash_content("inactive");
        let active = hash_content("active");
        db.insert_content(&inactive, "inactive").unwrap();
        db.upsert_document("docs", "inactive.md", "inactive", &inactive)
            .unwrap();
        db.insert_content(&active, "active").unwrap();
        db.upsert_document("docs", "active.md", "active", &active)
            .unwrap();
        db.insert_embedding_with_fingerprint(&inactive, 0, 0, &[0.0, 0.0], "default", &fingerprint)
            .unwrap();
        db.insert_embedding_with_fingerprint(&active, 0, 0, &[0.5, 0.0], "default", &fingerprint)
            .unwrap();
        db.deactivate("docs", "inactive.md").unwrap();

        let hits = db
            .search_vec_with_fingerprint(&[0.0, 0.0], 1, None, &fingerprint)
            .unwrap();
        assert_eq!(hits.len(), 1, "active document must not be lost: {hits:?}");
        assert_eq!(hits[0].doc.path, "active.md");
    }

    #[test]
    fn unscoped_vector_search_exact_over_inactive_rows_beyond_native_cap() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        let inactive = hash_content("inactive");
        let active = hash_content("active");
        db.insert_content(&inactive, "inactive").unwrap();
        db.upsert_document("docs", "inactive.md", "inactive", &inactive)
            .unwrap();
        db.insert_content(&active, "active").unwrap();
        db.upsert_document("docs", "active.md", "active", &active)
            .unwrap();
        db.insert_embedding_with_fingerprint(&inactive, 0, 0, &[0.0, 0.0], "default", &fingerprint)
            .unwrap();
        db.insert_embedding_with_fingerprint(&active, 0, 0, &[0.5, 0.0], "default", &fingerprint)
            .unwrap();
        // More nearer rows than sqlite-vec's native k cap of 4096, all tied to
        // an inactive document. Only an exact active-document aggregation —
        // not a capped KNN window at any size — can satisfy limit 1 here.
        for seq in 1..4097 {
            db.insert_embedding_with_fingerprint(
                &inactive,
                seq,
                0,
                &[0.0, 0.0],
                "default",
                &fingerprint,
            )
            .unwrap();
        }
        db.deactivate("docs", "inactive.md").unwrap();
        let embedding_rows: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM vec_embeddings", [], |row| row.get(0))
            .unwrap();
        assert_eq!(embedding_rows, 4098, "beyond the native k = 4096 cap");

        let hits = db
            .search_vec_with_fingerprint(&[0.0, 0.0], 1, None, &fingerprint)
            .unwrap();
        assert_eq!(hits.len(), 1, "active document must not be lost: {hits:?}");
        assert_eq!(hits[0].doc.path, "active.md");
        assert!((hits[0].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn vector_search_limit_zero_is_empty_after_aggregation() {
        let mut db = mem_db();
        let fingerprint = crate::embed::embedding_fingerprint(2, 3200, 480);
        let hash = hash_content("doc");
        db.insert_content(&hash, "doc").unwrap();
        db.upsert_document("docs", "doc.md", "doc", &hash).unwrap();
        for seq in 0..2 {
            db.insert_embedding_with_fingerprint(
                &hash,
                seq,
                0,
                &[0.1, 0.0],
                "default",
                &fingerprint,
            )
            .unwrap();
        }
        assert!(
            db.search_vec_with_fingerprint(&[0.0, 0.0], 0, None, &fingerprint)
                .unwrap()
                .is_empty()
        );
        assert!(
            db.search_vec_with_fingerprint(&[0.0, 0.0], 0, Some("docs"), &fingerprint)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn established_fingerprint_cannot_be_reassigned_over_vectors() {
        let mut db = mem_db();
        db.insert_embedding_with_fingerprint("hash", 0, 0, &[1.0, 0.0], "default", "A")
            .unwrap();

        let err = db.set_embedding_fingerprint("B").unwrap_err();
        assert!(err.to_string().contains("existing vectors"));
        assert_eq!(db.embedding_fingerprint().unwrap().as_deref(), Some("A"));
    }

    #[test]
    fn doctor_reports_stable_warning_and_error_states() {
        let healthy = mem_db().doctor("expected", 384).unwrap();
        assert!(
            healthy
                .checks
                .iter()
                .all(|check| check.status == DoctorCheckStatus::Ok)
        );

        let warning_only = mem_db();
        warning_only
            .upsert_collection(&Collection::new("missing", "/definitely/not/here"))
            .unwrap();
        let warning_report = warning_only.doctor("expected", 384).unwrap();
        assert!(!warning_report.has_errors());
        assert!(warning_report.checks.iter().any(|check| {
            check.name == "collection_paths" && check.status == DoctorCheckStatus::Warning
        }));

        let db = mem_db();
        db.upsert_collection(&Collection::new("missing", "/definitely/not/here"))
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO content_vectors(hash, seq, pos, model, embedded_at) VALUES ('orphan', 0, 0, 'legacy', 'now')",
                [],
            )
            .unwrap();

        let report = db.doctor("expected", 384).unwrap();
        assert!(report.checks.iter().any(|check| {
            check.name == "collection_paths" && check.status == DoctorCheckStatus::Warning
        }));
        assert!(report.checks.iter().any(|check| {
            check.name == "orphan_vectors" && check.status == DoctorCheckStatus::Error
        }));
        assert!(report.has_errors());
        let json = serde_json::to_value(&report).unwrap();
        assert!(json["checks"].is_array());
        assert_eq!(json["checks"][0]["name"], "sqlite_quick_check");
    }
}
