//! Crate-level error types.

use thiserror::Error;

/// Unified error type for all qmd operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// SQLite / rusqlite error.
    #[error("database: {0}")]
    Database(#[from] rusqlite::Error),

    /// Filesystem I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// Embedding model error.
    #[error("embedding: {0}")]
    Embedding(String),

    /// Reranking model error.
    #[error("rerank: {0}")]
    Rerank(String),

    /// Configuration or path resolution error.
    #[error("config: {0}")]
    Config(String),

    /// Document not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Collection already exists.
    #[error("collection already exists: {0}")]
    CollectionExists(String),
}

/// Crate-level result alias.
pub type Result<T> = std::result::Result<T, Error>;
