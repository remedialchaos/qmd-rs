//! QMD — Query Markdown Documents.
//!
//! Local search engine for markdown files combining FTS5 (BM25) and
//! sqlite-vec (vector KNN) over a single local SQLite index, with ONNX
//! embeddings and reranking via opt-in `fastembed-*` features or fully
//! pluggable custom engines.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use qmd_rs::{Qmd, Collection};
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
//! # Ok::<(), qmd_rs::Error>(())
//! ```

// The two fastembed backends select mutually exclusive ONNX Runtime
// linking strategies; enabling both is never a meaningful build (ort would
// link no runtime at all), so reject it up front instead of letting it
// fail at first model use.
#[cfg(all(feature = "fastembed-static", feature = "fastembed-dynamic"))]
compile_error!(
    "fastembed-static and fastembed-dynamic are mutually exclusive ONNX Runtime linking \
     strategies; enable at most one of them"
);

pub mod chunk;
pub mod db;
pub mod embed;
pub mod error;
pub mod llm;
pub mod qmd;
pub mod rerank;
pub mod search;

pub use db::{
    Collection, CollectionInfo, DoctorCheck, DoctorCheckStatus, DoctorReport, Document,
    DocumentSummary, EmbeddingCompatibility, IndexStatus, OpenOptions, SearchResult, SearchSource,
    hash_content,
};
pub use embed::{EmbeddingEngine, embedding_fingerprint_for};
pub use error::{Error, Result};
pub use llm::{LlmConfig, LlmProvider, expand_query, parse_or_expand_query};
pub use qmd::{
    EmbedResult, IndexFailure, IndexResult, MultiGetItem, Qmd, UpdateResult, WriteBatch,
    parse_path_range, slice_body,
};
pub use rerank::{RerankingEngine, Scored};
pub use search::{Query, QueryType};
