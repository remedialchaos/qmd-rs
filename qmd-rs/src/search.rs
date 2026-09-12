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

/// Normalize a term to the FTS5 tokenizer's own boundaries.
///
/// The index is built with the `porter unicode61` tokenizer, which splits on
/// every non-alphanumeric character. Deleting those separators here would glue
/// a multi-part identifier such as `qmd-rs` into the single token `qmdrs`,
/// which can never match an indexed `qmd` + `rs`. Replace each separator run
/// with a single space instead, so the emitted query keeps the same token
/// boundaries the tokenizer produced at index time.
fn sanitize_fts5_term(term: &str) -> String {
    let mut normalized = String::with_capacity(term.len());
    for ch in term.chars() {
        if ch.is_alphanumeric() {
            normalized.extend(ch.to_lowercase());
        } else if !normalized.is_empty() && !normalized.ends_with(' ') {
            normalized.push(' ');
        }
    }
    if normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

/// Build an FTS5 query from user-facing search syntax.
///
/// Supports quoted phrases, negation (`-term`), and prefix matching.
/// Returns `None` if no usable terms.
///
/// # Examples
///
/// ```
/// use qmd_rs::search::build_fts5_query;
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
        let mut counted: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (rank, key) in keys.iter().enumerate() {
            // A repeated key inside one backend list contributes only at its
            // first rank; later occurrences must not inflate its score.
            if !counted.insert(key.as_str()) {
                continue;
            }
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
    if max_chars == 0 {
        return String::new();
    }

    let chars: Vec<char> = body.chars().collect();
    if chars.len() <= max_chars {
        return body.to_string();
    }

    let body_lower = body.to_lowercase();
    let match_info = query
        .split_whitespace()
        .filter(|t| t.chars().count() >= 3)
        .find_map(|term| {
            body_lower
                .find(&term.to_lowercase())
                .map(|position| (position, term.chars().count()))
        });
    let match_char = match_info.map_or(0, |(lower_pos, _)| {
        let mut lower_offset = 0;
        for (char_index, ch) in body.char_indices() {
            if lower_pos <= lower_offset {
                return body[..char_index].chars().count();
            }
            let lower_len = ch.to_lowercase().collect::<String>().len();
            if lower_pos < lower_offset + lower_len {
                return body[..char_index].chars().count();
            }
            lower_offset += lower_len;
        }
        chars.len()
    });
    let start = match match_info {
        Some((_, term_len)) if term_len <= max_chars => {
            let centered = match_char.saturating_sub(max_chars / 2);
            let required = match_char
                .saturating_add(term_len)
                .saturating_sub(max_chars);
            centered.max(required).min(chars.len() - max_chars)
        }
        Some(_) => match_char.min(chars.len() - max_chars),
        None => match_char.saturating_sub(50),
    };
    let end = start + max_chars;

    chars[start..end].iter().collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::{build_fts5_query, extract_snippet, rrf};
    use rusqlite::Connection;

    /// Build an in-memory FTS5 fixture that mirrors the production index
    /// tokenizer so query building is exercised end to end.
    fn fixture(docs: &[&str]) -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        conn.execute_batch(
            "CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61');",
        )
        .expect("create fts5 fixture");
        {
            let mut insert = conn
                .prepare("INSERT INTO docs(body) VALUES (?1)")
                .expect("prepare insert");
            for doc in docs {
                insert.execute([doc]).expect("seed fixture");
            }
        }
        conn
    }

    /// Run the production query builder against the fixture and return the
    /// bodies of matching documents in rowid order.
    fn matching(query: &str, docs: &[&str]) -> Vec<String> {
        let conn = fixture(docs);
        let built = build_fts5_query(query).expect("query builder must yield a usable query");
        let mut stmt = conn
            .prepare("SELECT body FROM docs WHERE docs MATCH ?1 ORDER BY rowid")
            .expect("prepare match");
        let rows = stmt
            .query_map([built], |row| row.get::<_, String>(0))
            .expect("run match");
        rows.map(|row| row.expect("row")).collect()
    }

    #[test]
    fn hyphenated_identifier_matches_the_document_that_contains_it() {
        let docs = ["qmd-rs is the CLI", "an unrelated document"];

        assert_eq!(matching("qmd-rs", &docs), ["qmd-rs is the CLI"]);
    }

    #[test]
    fn path_like_input_and_extension_match_their_token_boundaries() {
        let docs = ["see qmd-rs/src/search.rs for details", "a different file"];

        assert_eq!(
            matching("qmd-rs/src/search.rs", &docs),
            ["see qmd-rs/src/search.rs for details"]
        );
        assert_eq!(
            matching("search.rs", &docs),
            ["see qmd-rs/src/search.rs for details"]
        );
    }

    #[test]
    fn version_like_token_matches_its_token_boundaries() {
        let docs = ["released v1.2.3 today", "released v123 today"];

        assert_eq!(matching("v1.2.3", &docs), ["released v1.2.3 today"]);
    }

    #[test]
    fn quoted_phrase_matches_across_separator_boundaries() {
        let docs = ["qmd-rs is the CLI", "qmd plus rs elsewhere"];

        assert_eq!(matching("\"qmd-rs\"", &docs), ["qmd-rs is the CLI"]);
    }

    #[test]
    fn established_grammar_and_safe_failure_are_preserved() {
        assert_eq!(
            build_fts5_query("performance -sports"),
            Some(r#""performance"* NOT "sports"*"#.to_string())
        );
        assert_eq!(build_fts5_query("perf"), Some(r#""perf"*"#.to_string()));
        assert_eq!(build_fts5_query("   "), None);
        assert_eq!(build_fts5_query("!!! --- ..."), None);
        assert_eq!(build_fts5_query("\"\""), None);
    }

    #[test]
    fn unicode_letters_numbers_and_apostrophes_still_match() {
        let docs = ["café 東京 don't stop"];

        assert_eq!(matching("café", &docs), ["café 東京 don't stop"]);
        assert_eq!(matching("東京", &docs), ["café 東京 don't stop"]);
        assert_eq!(matching("don't", &docs), ["café 東京 don't stop"]);
    }

    #[test]
    #[allow(clippy::panic)]
    fn rrf_counts_each_key_once_per_list_retaining_first_rank() {
        let list = vec!["B".to_string(), "A".to_string(), "A".to_string()];

        let hits = rrf(&[&list], None, 60);

        let score_of = |key: &str| {
            hits.iter()
                .find(|hit| hit.key == key)
                .unwrap_or_else(|| panic!("missing key {key}"))
                .score
        };

        assert!((score_of("B") - 1.0 / 61.0).abs() < 1e-12);
        assert!(
            (score_of("A") - 1.0 / 62.0).abs() < 1e-12,
            "duplicate keys must count once at their first rank"
        );
    }

    #[test]
    fn rrf_breaks_equal_scores_deterministically_by_key() {
        let lexical = vec!["A".to_string(), "B".to_string()];
        let semantic = vec!["B".to_string(), "A".to_string()];

        let hits = rrf(&[&lexical, &semantic], None, 60);

        let keys: Vec<&str> = hits.iter().map(|hit| hit.key.as_str()).collect();
        assert_eq!(keys, ["A", "B"]);
        assert!((hits[0].score - hits[1].score).abs() < 1e-12);
    }

    #[test]
    fn unicode_case_matching_never_slices_inside_utf8() {
        let body = "prefix 😀 café 東京 İSTANBUL target suffix";

        let snippet = extract_snippet(body, "istanbul", 12);

        assert!(snippet.is_char_boundary(snippet.len()));
        assert!(snippet.chars().count() <= 12);
    }

    #[test]
    fn snippet_never_exceeds_budget_when_line_is_longer() {
        let body = "before\nthis line contains the target and keeps going\nafter";

        let snippet = extract_snippet(body, "target", 10);

        assert_eq!(snippet.chars().count(), 10);
        assert!(snippet.contains("target"));
    }

    #[test]
    fn zero_budget_and_query_miss_are_empty_or_bounded() {
        let body = "accentué 中文 😀\nlong line without a match";

        assert_eq!(extract_snippet(body, "match", 0), "");
        assert!(extract_snippet(body, "absent", 5).chars().count() <= 5);
    }

    #[test]
    fn match_near_start_with_larger_budget_does_not_panic() {
        let body = "target and a long tail that exceeds the budget";

        let snippet = extract_snippet(body, "target", 10);

        assert_eq!(snippet.chars().count(), 10);
        assert!(snippet.contains("target"));
    }
}
