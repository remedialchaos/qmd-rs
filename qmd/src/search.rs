//! Search utilities: FTS5 query building, RRF fusion, snippet extraction.

use std::collections::HashMap;

use regex::Regex;

/// Search backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum QueryType {
    /// Lexical (BM25).
    Lex,
    /// Vector (semantic).
    Vec,
    /// HyDE — Hypothetical Document Embedding.
    Hyde,
}

/// A typed search query destined for a specific backend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Query {
    /// Backend kind.
    pub kind: QueryType,
    /// Query text.
    pub text: String,
}

impl Query {
    /// Create a lexical (BM25) query.
    #[must_use]
    pub fn lex(text: impl Into<String>) -> Self {
        Self {
            kind: QueryType::Lex,
            text: text.into(),
        }
    }

    /// Create a vector (semantic) query.
    #[must_use]
    pub fn vec(text: impl Into<String>) -> Self {
        Self {
            kind: QueryType::Vec,
            text: text.into(),
        }
    }

    /// Create a HyDE query.
    #[must_use]
    pub fn hyde(text: impl Into<String>) -> Self {
        Self {
            kind: QueryType::Hyde,
            text: text.into(),
        }
    }

    /// Simple (non-LLM) expansion into lex + vec + hyde.
    #[must_use]
    pub fn expand_simple(query: &str) -> Vec<Self> {
        vec![
            Self::lex(query),
            Self::vec(query),
            Self::hyde(format!("Information about {query}")),
        ]
    }

    /// Parse structured LLM output into typed queries.
    ///
    /// Expected format (one per line): `lex:`, `vec:`, `hyde:`.
    /// Falls back to [`expand_simple`](Self::expand_simple) if no valid lines found.
    #[must_use]
    pub fn from_llm_output(output: &str, original: &str) -> Vec<Self> {
        let query_lower = original.to_lowercase();
        let line_re = Regex::new(r"^(lex|vec|hyde):\s*(.+)$").ok();
        let mut queries = Vec::new();

        for raw in output.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(ref re) = line_re
                && let Some(caps) = re.captures(line)
            {
                let kind = match &caps[1] {
                    "lex" => QueryType::Lex,
                    "vec" => QueryType::Vec,
                    "hyde" => QueryType::Hyde,
                    _ => continue,
                };
                let text = caps[2].trim();
                let text_lower = text.to_lowercase();
                let has_overlap = query_lower
                    .split_whitespace()
                    .any(|t| t.len() >= 3 && text_lower.contains(t));

                if has_overlap || query_lower.len() < 3 {
                    queries.push(Self {
                        kind,
                        text: text.to_string(),
                    });
                }
            }
        }

        if queries.is_empty() {
            Self::expand_simple(original)
        } else {
            queries
        }
    }
}

impl<'de> serde::Deserialize<'de> for QueryType {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "lex" => Ok(Self::Lex),
            "vec" => Ok(Self::Vec),
            "hyde" => Ok(Self::Hyde),
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &["lex", "vec", "hyde"],
            )),
        }
    }
}

/// Sanitize a term for FTS5 (keep only alphanumeric + apostrophes).
fn sanitize_fts5_term(term: &str) -> String {
    term.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .collect::<String>()
        .to_lowercase()
}

/// Build an FTS5 query from user-facing search syntax.
///
/// Supports quoted phrases, negation (`-term`), and prefix matching.
/// Returns `None` if no usable terms.
///
/// # Examples
///
/// ```
/// use qmd::search::build_fts5_query;
///
/// assert_eq!(
///     build_fts5_query("performance -sports"),
///     Some(r#""performance"* NOT "sports"*"#.to_string()),
/// );
/// ```
#[must_use]
pub fn build_fts5_query(query: &str) -> Option<String> {
    let mut positive: Vec<String> = Vec::new();
    let mut negative: Vec<String> = Vec::new();

    let s = query.trim();
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        let negated = bytes[i] == b'-';
        if negated {
            i += 1;
            if i >= bytes.len() {
                break;
            }
        }

        if bytes[i] == b'"' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            let phrase = &s[start..i];
            if i < bytes.len() {
                i += 1;
            }
            let sanitized: String = phrase
                .split_whitespace()
                .map(sanitize_fts5_term)
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !sanitized.is_empty() {
                let fts = format!("\"{sanitized}\"");
                if negated {
                    &mut negative
                } else {
                    &mut positive
                }
                .push(fts);
            }
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'"' {
                i += 1;
            }
            let sanitized = sanitize_fts5_term(&s[start..i]);
            if !sanitized.is_empty() {
                let fts = format!("\"{sanitized}\"*");
                if negated {
                    &mut negative
                } else {
                    &mut positive
                }
                .push(fts);
            }
        }
    }

    if positive.is_empty() {
        return None;
    }

    let mut result = positive.join(" ");
    for neg in &negative {
        result = format!("{result} NOT {neg}");
    }
    Some(result)
}

/// Monotonic mapping of raw BM25 to `[0, 1)`: `x / (1 + x)`.
#[must_use]
pub fn normalize_bm25(score: f64) -> f64 {
    let s = score.abs();
    s / (1.0 + s)
}

/// A fused RRF result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RrfHit {
    /// Document key.
    pub key: String,
    /// Merged RRF score.
    pub score: f64,
}

/// Reciprocal Rank Fusion over multiple ranked key lists.
///
/// `RRF(d) = Σ weight_i / (k + rank_i + 1)`.
#[must_use]
pub fn rrf(lists: &[&[String]], weights: Option<&[f64]>, k: usize) -> Vec<RrfHit> {
    let mut scores: HashMap<&str, f64> = HashMap::new();

    for (list_idx, keys) in lists.iter().enumerate() {
        let w = weights
            .and_then(|ws| ws.get(list_idx))
            .copied()
            .unwrap_or(1.0);
        for (rank, key) in keys.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let s = w / (k + rank + 1) as f64;
            *scores.entry(key.as_str()).or_default() += s;
        }
    }

    let mut hits: Vec<RrfHit> = scores
        .into_iter()
        .map(|(key, score)| RrfHit {
            key: key.to_string(),
            score,
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.key.cmp(&b.key)));
    hits
}

/// Extract a relevant snippet from `body` around query terms.
#[must_use]
pub fn extract_snippet(body: &str, query: &str, max_chars: usize) -> String {
    if body.len() <= max_chars {
        return body.to_string();
    }

    let body_lower = body.to_lowercase();
    let start_pos = query
        .split_whitespace()
        .filter(|t| t.len() >= 3)
        .find_map(|t| body_lower.find(&t.to_lowercase()))
        .map_or(0, |p| p.saturating_sub(50));

    let line_start = body[..start_pos].rfind('\n').map_or(0, |p| p + 1);
    let end_pos = (line_start + max_chars).min(body.len());
    let line_end = body[end_pos..]
        .find('\n')
        .map_or(body.len(), |p| end_pos + p);

    body[line_start..line_end].to_string()
}
