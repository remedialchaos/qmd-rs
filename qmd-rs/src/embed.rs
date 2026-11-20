//! Embedding contract and engine seam.
//!
//! [`EmbeddingEngine`] and the fingerprint helpers are always available. The
//! concrete fastembed-backed `Embedder` is compiled only when a
//! `fastembed-*` feature is enabled.

#[cfg(feature = "fastembed")]
use fastembed::{EmbeddingModel, InitOptions, ModelTrait, TextEmbedding};
use sha2::{Digest, Sha256};

use crate::chunk::{Chunker, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS};

#[cfg(feature = "fastembed")]
use crate::error::Error;
use crate::error::Result;

/// Upper bound for the number of chunks embedded in one ONNX Runtime call.
#[cfg(feature = "fastembed")]
const EMBEDDING_BATCH_SIZE: usize = 16;

/// Stable fingerprint format for the embedding/index contract.
pub const FINGERPRINT_FORMAT: &str = "qmd-embedding-v1";

/// Identifier of the default embedding model (fastembed's `AllMiniLML6V2`).
///
/// Engines that do not override [`EmbeddingEngine::model_id`] report this
/// value, so their vectors share the historical embedding contract.
pub const DEFAULT_MODEL_ID: &str = "AllMiniLML6V2";

/// Vector width produced by [`DEFAULT_MODEL_ID`].
///
/// Engines that do not override [`EmbeddingEngine::dimensions`] report this
/// value.
pub const DEFAULT_EMBEDDING_DIMS: usize = 384;

/// Build the deterministic fingerprint for an arbitrary embedding contract.
///
/// The fingerprint pins the model identity, vector width, and chunking
/// geometry under which vectors were produced, so vectors from one contract
/// are never silently queried under another. For the default model this
/// matches [`embedding_fingerprint`] exactly; a custom `model_id` or
/// `dims` yields a distinct fingerprint so indexes never mix contracts.
#[must_use]
pub fn embedding_fingerprint_for(
    model_id: &str,
    dims: usize,
    max_chars: usize,
    overlap_chars: usize,
) -> String {
    let input = format!(
        "{FINGERPRINT_FORMAT}|model={model_id}|dims={dims}|max_chars={max_chars}|overlap_chars={overlap_chars}"
    );
    let mut hash = Sha256::new();
    hash.update(input.as_bytes());
    format!("{:x}", hash.finalize())
}

/// Build the deterministic fingerprint for the current embedding contract.
#[must_use]
pub fn embedding_fingerprint(dims: usize, max_chars: usize, overlap_chars: usize) -> String {
    embedding_fingerprint_for(DEFAULT_MODEL_ID, dims, max_chars, overlap_chars)
}

/// Fingerprint for the default embedder and chunker.
#[must_use]
pub fn default_embedding_fingerprint(dims: usize) -> String {
    embedding_fingerprint(dims, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS)
}

/// Fingerprint for a specific embedding chunker configuration.
#[must_use]
pub fn default_embedding_fingerprint_with_chunker(dims: usize, chunker: Chunker) -> String {
    embedding_fingerprint(dims, chunker.max_chars, chunker.overlap_chars)
}

/// Return contiguous ranges that limit a document embedding request to safe batches.
#[cfg(feature = "fastembed")]
fn batch_ranges(len: usize) -> Vec<std::ops::Range<usize>> {
    (0..len)
        .step_by(EMBEDDING_BATCH_SIZE)
        .map(|start| start..(start + EMBEDDING_BATCH_SIZE).min(len))
        .collect()
}

/// Narrow engine seam for embedding work.
///
/// Production uses the built-in fastembed engine (with a `fastembed-*`
/// feature); the trait exists so embedding scheduling and failure handling
/// can be exercised deterministically without loading an ONNX model or
/// touching the network, and so downstream callers can plug in their own
/// model or an API embedder.
///
/// # Default contract
///
/// Implementations that do not override [`EmbeddingEngine::model_id`] and
/// [`EmbeddingEngine::dimensions`] inherit the historical default contract
/// ([`DEFAULT_MODEL_ID`] / [`DEFAULT_EMBEDDING_DIMS`]), so they stay compatible
/// with indexes embedded by the default built-in engine. Override both to plug
/// in a different model: the derived fingerprint then changes, and indexes
/// embedded under the old contract are rejected instead of mixed.
pub trait EmbeddingEngine: Send {
    /// Embed a single query string.
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>>;

    /// Embed a batch of document texts.
    fn embed_documents(&mut self, docs: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Stable identifier of the model producing this engine's vectors.
    ///
    /// Part of the embedding-contract fingerprint: two engines reporting the
    /// same `model_id` and [`dimensions`](Self::dimensions) are assumed to
    /// produce interchangeable vectors.
    fn model_id(&self) -> &str {
        DEFAULT_MODEL_ID
    }

    /// Vector width this engine produces.
    fn dimensions(&self) -> usize {
        DEFAULT_EMBEDDING_DIMS
    }
}

#[cfg(feature = "fastembed")]
impl EmbeddingEngine for Embedder {
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        Self::embed_query(self, query)
    }

    fn embed_documents(&mut self, docs: &[&str]) -> Result<Vec<Vec<f32>>> {
        Self::embed_documents(self, docs)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

/// Text embedding engine.
#[cfg(feature = "fastembed")]
pub struct Embedder {
    /// Underlying fastembed model.
    model: TextEmbedding,
    /// Model identifier used for contract fingerprinting.
    model_id: String,
    /// Vector width produced by `model`.
    dims: usize,
}

#[cfg(feature = "fastembed")]
impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

#[cfg(feature = "fastembed")]
impl Embedder {
    /// Create an embedder with the default model ([`AllMiniLML6V2`]).
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create an embedder with a specific model.
    ///
    /// The model's fastembed metadata supplies the contract identity: the
    /// variant name becomes [`EmbeddingEngine::model_id`] (for the default
    /// model, exactly `"AllMiniLML6V2"`) and the table dimension becomes
    /// [`EmbeddingEngine::dimensions`].
    pub fn with_model(kind: EmbeddingModel) -> Result<Self> {
        let dims = EmbeddingModel::get_model_info(&kind)
            .ok_or_else(|| Error::Embedding(format!("no fastembed metadata for {kind:?}")))?
            .dim;
        let model_id = format!("{kind:?}");
        let opts = InitOptions::new(kind).with_show_download_progress(true);
        let model = TextEmbedding::try_new(opts).map_err(|e| Error::Embedding(e.to_string()))?;
        Ok(Self {
            model,
            model_id,
            dims,
        })
    }

    /// Embed a single query string.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let results = self
            .model
            .embed(vec![query], None)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty embedding result".into()))
    }

    /// Embed a batch of documents.
    pub fn embed_documents(&mut self, docs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(docs.len());
        for range in batch_ranges(docs.len()) {
            let batch = self
                .model
                .embed(&docs[range], None)
                .map_err(|e| Error::Embedding(e.to_string()))?;
            embeddings.extend(batch);
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CHUNK_CHARS, DEFAULT_MODEL_ID, DEFAULT_OVERLAP_CHARS, embedding_fingerprint,
        embedding_fingerprint_for,
    };

    #[cfg(feature = "fastembed")]
    use super::{EMBEDDING_BATCH_SIZE, batch_ranges};

    #[cfg(feature = "fastembed")]
    #[test]
    fn batch_ranges_bound_embedding_work() {
        let ranges = batch_ranges(EMBEDDING_BATCH_SIZE + 1);

        assert_eq!(
            ranges,
            [
                0..EMBEDDING_BATCH_SIZE,
                EMBEDDING_BATCH_SIZE..EMBEDDING_BATCH_SIZE + 1
            ]
        );
    }

    #[test]
    fn model_aware_fingerprint_keeps_the_default_contract_byte_identical() {
        // Digest pinned from the pre-change implementation: existing .sqlite
        // indexes must never require re-embedding.
        assert_eq!(
            embedding_fingerprint(384, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS),
            "a9a55493ecf9f92fe995bb64bd4ed8df54f05a74f68657b907ab29b910600524"
        );

        // The model-aware helper must reproduce the legacy fingerprint
        // exactly for the default model.
        assert_eq!(
            embedding_fingerprint_for(
                DEFAULT_MODEL_ID,
                384,
                DEFAULT_CHUNK_CHARS,
                DEFAULT_OVERLAP_CHARS
            ),
            embedding_fingerprint(384, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS)
        );
        assert_eq!(
            embedding_fingerprint_for("AllMiniLML6V2", 384, 100, 10),
            "a3f27375eceab1f247563c7aad86cd62c0ed236c4538436e314d84d6f6b8c3b0"
        );

        // A different model identity or width must change the contract.
        assert_ne!(
            embedding_fingerprint_for(
                "OtherModel",
                384,
                DEFAULT_CHUNK_CHARS,
                DEFAULT_OVERLAP_CHARS
            ),
            embedding_fingerprint(384, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS)
        );
        assert_ne!(
            embedding_fingerprint_for(
                DEFAULT_MODEL_ID,
                768,
                DEFAULT_CHUNK_CHARS,
                DEFAULT_OVERLAP_CHARS
            ),
            embedding_fingerprint(384, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS)
        );
    }
}
