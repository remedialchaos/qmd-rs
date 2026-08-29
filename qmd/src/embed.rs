//! Embedding engine backed by [`fastembed`].

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::{Error, Result};

/// Upper bound for the number of chunks embedded in one ONNX Runtime call.
const EMBEDDING_BATCH_SIZE: usize = 16;

/// Return contiguous ranges that limit a document embedding request to safe batches.
fn batch_ranges(len: usize) -> Vec<std::ops::Range<usize>> {
    (0..len)
        .step_by(EMBEDDING_BATCH_SIZE)
        .map(|start| start..(start + EMBEDDING_BATCH_SIZE).min(len))
        .collect()
}

/// Text embedding engine.
pub struct Embedder {
    /// Underlying fastembed model.
    model: TextEmbedding,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    /// Create an embedder with the default model ([`AllMiniLML6V2`]).
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create an embedder with a specific model.
    pub fn with_model(kind: EmbeddingModel) -> Result<Self> {
        let opts = InitOptions::new(kind).with_show_download_progress(true);
        let model = TextEmbedding::try_new(opts).map_err(|e| Error::Embedding(e.to_string()))?;
        Ok(Self { model })
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
    use super::{EMBEDDING_BATCH_SIZE, batch_ranges};

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
}
