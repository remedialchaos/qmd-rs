//! Deterministic, model-free retrieval-quality regression gate.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::unwrap_used
)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use qmd::{Collection, Qmd};
use serde::Deserialize;

const JUDGMENTS: &str = include_str!("fixtures/retrieval-quality/judgments.json");

#[derive(Deserialize)]
struct Judgment {
    query: String,
    relevance: HashMap<String, u32>,
}

fn reciprocal_rank(ranked: &[String], relevant: &HashMap<String, u32>) -> f64 {
    ranked
        .iter()
        .position(|path| relevant.contains_key(path))
        .map_or(0.0, |rank| 1.0 / (rank + 1) as f64)
}

fn recall_at(ranked: &[String], relevant: &HashMap<String, u32>, k: usize) -> f64 {
    let found = ranked
        .iter()
        .take(k)
        .filter(|path| relevant.contains_key(*path))
        .count();
    found as f64 / relevant.len() as f64
}

fn dcg(grades: impl Iterator<Item = u32>) -> f64 {
    grades
        .enumerate()
        .map(|(rank, grade)| (2_f64.powi(grade as i32) - 1.0) / ((rank + 2) as f64).log2())
        .sum()
}

fn ndcg_at(ranked: &[String], relevant: &HashMap<String, u32>, k: usize) -> f64 {
    let actual = dcg(ranked
        .iter()
        .take(k)
        .map(|path| relevant.get(path).copied().unwrap_or(0)));
    let mut ideal: Vec<u32> = relevant.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let best = dcg(ideal.into_iter().take(k));
    if best == 0.0 { 0.0 } else { actual / best }
}

#[test]
fn fts_retrieval_quality_meets_checked_in_thresholds() {
    let corpus =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/retrieval-quality/corpus");
    let judgments: Vec<Judgment> = serde_json::from_str(JUDGMENTS).unwrap();
    let qmd = Qmd::open_memory().unwrap();
    qmd.register_collection(&Collection::new("quality", corpus.to_string_lossy()))
        .unwrap();
    let update = qmd.update(None).unwrap();
    assert!(update.failures.is_empty());

    let mut mrr_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut ndcg_sum = 0.0;
    for judgment in &judgments {
        let ranked: Vec<String> = qmd
            .search_fts(&judgment.query, 3)
            .unwrap()
            .iter()
            .map(|result| result.doc.display_path())
            .collect();
        mrr_sum += reciprocal_rank(&ranked, &judgment.relevance);
        recall_sum += recall_at(&ranked, &judgment.relevance, 3);
        ndcg_sum += ndcg_at(&ranked, &judgment.relevance, 3);
    }
    let count = judgments.len() as f64;
    let (mrr, recall, ndcg) = (mrr_sum / count, recall_sum / count, ndcg_sum / count);
    assert!(mrr >= 1.0, "MRR regression: {mrr:.3}");
    assert!(recall >= 0.83, "Recall@3 regression: {recall:.3}");
    assert!(ndcg >= 0.85, "nDCG@3 regression: {ndcg:.3}");
}

#[test]
fn fusion_order_is_deterministic_without_models() {
    let lexical = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let semantic = vec!["b".to_string(), "a".to_string(), "d".to_string()];
    let first = qmd::search::rrf(&[&lexical, &semantic], None, 60);
    let second = qmd::search::rrf(&[&lexical, &semantic], None, 60);
    let first_keys: Vec<&str> = first.iter().map(|hit| hit.key.as_str()).collect();
    let second_keys: Vec<&str> = second.iter().map(|hit| hit.key.as_str()).collect();
    assert_eq!(first_keys, second_keys);
    assert_eq!(&first_keys[..2], &["a", "b"]);
}

#[test]
fn public_fts_search_paginates_without_duplicates_and_empties_out_of_range() {
    let corpus =
        std::env::temp_dir().join(format!("qmd-retrieval-pagination-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&corpus);
    std::fs::create_dir_all(&corpus).unwrap();
    for (name, body) in [
        ("one.md", "# One\nPagination coverage one."),
        ("two.md", "# Two\nPagination coverage two."),
        ("three.md", "# Three\nPagination coverage three."),
        ("four.md", "# Four\nPagination coverage four."),
    ] {
        std::fs::write(corpus.join(name), body).unwrap();
    }

    let qmd = Qmd::open_memory().unwrap();
    qmd.register_collection(&Collection::new("pagination", corpus.to_string_lossy()))
        .unwrap();
    let update = qmd.update(None).unwrap();
    assert!(update.failures.is_empty());

    let page_one: Vec<String> = qmd
        .search_fts_with_offset("pagination", 2, 0)
        .unwrap()
        .iter()
        .map(|result| result.doc.display_path())
        .collect();
    let page_two: Vec<String> = qmd
        .search_fts_with_offset("pagination", 2, 2)
        .unwrap()
        .iter()
        .map(|result| result.doc.display_path())
        .collect();
    let out_of_range = qmd.search_fts_with_offset("pagination", 2, 4).unwrap();

    assert_eq!(page_one.len(), 2);
    assert_eq!(page_two.len(), 2);
    assert!(page_one.iter().all(|path| !page_two.contains(path)));
    let all_hits: HashSet<&str> = page_one
        .iter()
        .chain(page_two.iter())
        .map(String::as_str)
        .collect();
    assert_eq!(all_hits.len(), 4);
    assert!(out_of_range.is_empty());

    std::fs::remove_dir_all(corpus).unwrap();
}
