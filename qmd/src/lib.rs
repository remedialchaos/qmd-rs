//! QMD — Query Markdown Documents.
//!
//! Local search engine for markdown files combining FTS5 (BM25),
//! sqlite-vec (vector KNN), and fastembed (ONNX embeddings + reranking).
//!
//! # Quick start
//!
//! ```rust,no_run
//! use qmd::{Qmd, Collection};
//!
//! let mut qmd = Qmd::open("./index.sqlite")?;
//! qmd.register_collection(&Collection::new("docs", "/path/to/docs"))?;
//! qmd.update(None)?;
//! qmd.embed()?;
//!
//! // Hybrid search (FTS + vector + RRF + rerank)
//! let results = qmd.search("how does auth work?", 10)?;
//!
//! // Or just BM25
//! let fts_results = qmd.search_fts("rust ownership", 10)?;
//! # Ok::<(), qmd::Error>(())
//! ```

pub mod chunk;
pub mod db;
pub mod embed;
pub mod error;
pub mod qmd;
pub mod rerank;
pub mod search;

pub use db::{
    Collection, CollectionInfo, DoctorCheck, DoctorCheckStatus, DoctorReport, Document,
    EmbeddingCompatibility, IndexStatus, SearchResult, SearchSource, hash_content,
};
pub use error::{Error, Result};
pub use qmd::{EmbedResult, IndexFailure, IndexResult, Qmd, UpdateResult};
pub use search::{Query, QueryType};
