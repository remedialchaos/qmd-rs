//! Reranking engine backed by [`fastembed`].

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

use crate::error::{Error, Result};

/// Scored rerank result.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct Scored {
    /// Document index in the input list.
    pub index: usize,
    /// Relevance score (higher is better).
    pub score: f32,
}

/// Cross-encoder reranking engine.
pub struct Reranker {
    /// Underlying fastembed model.
    model: TextRerank,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker").finish_non_exhaustive()
    }
}

impl Reranker {
    /// Create a reranker with the default model.
    pub fn new() -> Result<Self> {
        Self::with_model(RerankerModel::JINARerankerV2BaseMultiligual)
    }

    /// Create a reranker with a specific model.
    pub fn with_model(kind: RerankerModel) -> Result<Self> {
        let opts = RerankInitOptions::new(kind).with_show_download_progress(true);
        let model = TextRerank::try_new(opts).map_err(|e| Error::Rerank(e.to_string()))?;
        Ok(Self { model })
    }

    /// Rerank documents by relevance to the query.
    ///
    /// Returns scored indices sorted by descending relevance.
    pub fn rerank(&mut self, query: &str, documents: &[&str], limit: usize) -> Result<Vec<Scored>> {
        let results = self
            .model
            .rerank(query, documents, false, None)
            .map_err(|e| Error::Rerank(e.to_string()))?;

        let mut scored: Vec<Scored> = results
            .into_iter()
            .map(|r| Scored {
                index: r.index,
                score: r.score,
            })
            .collect();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(limit);
        Ok(scored)
    }
}
