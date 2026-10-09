//! High-contention WAL benchmark: 12 readers and 4 writers on one SQLite file.
//!
//! Models agentic knowledge-base traffic (keyword and hybrid search, document
//! retrieval, collection checks against batch inserts, tag updates, and
//! collection context edits). Asserts that no `SQLITE_BUSY[_SNAPSHOT]` or
//! `SQLITE_LOCKED` surfaces under the default 5 s busy timeout, that no write
//! is lost, and that the database passes `PRAGMA integrity_check`.
//!
//! Run with `cargo test -p qmd-rs --test concurrency_benchmark -- --nocapture`
//! to see the throughput and latency report.

#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use qmd_rs::db::Db;
use qmd_rs::{Error, OpenOptions, Qmd, hash_content};
use rusqlite::{Connection, ErrorCode};

const FP: &str = "bench-fingerprint";
const READERS: usize = 12;
const WRITERS: usize = 4;
const READS_PER_READER: usize = 60;
const WRITES_PER_WRITER: usize = 40;
const NOTES_PER_BATCH: usize = 8;
const SEED_DOCS: usize = 24;

struct TempDb(PathBuf);

impl TempDb {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("qmd-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> PathBuf {
        self.0.join("index.sqlite")
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A committed note a writer expects to find at the end of the run.
struct Expected {
    collection: String,
    path: String,
    body: String,
    token: String,
    stale_tokens: Vec<String>,
}

/// Per-thread outcome: latencies in milliseconds, errors, and committed state.
#[derive(Default)]
struct Outcome {
    latencies_ms: Vec<(&'static str, f64)>,
    errors: Vec<Error>,
    expected: Vec<Expected>,
    contexts: Vec<(String, String)>,
    embedded: usize,
}

impl Outcome {
    fn timed<T>(&mut self, kind: &'static str, op: impl FnOnce() -> Result<T, Error>) -> Option<T> {
        let start = Instant::now();
        let result = op();
        self.latencies_ms
            .push((kind, start.elapsed().as_secs_f64() * 1e3));
        match result {
            Ok(value) => Some(value),
            Err(e) => {
                self.errors.push(e);
                None
            }
        }
    }
}

const fn is_lock_error(e: &Error) -> bool {
    matches!(
        e,
        Error::Database(rusqlite::Error::SqliteFailure(f, _))
            if matches!(f.code, ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
    )
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn report(label: &str, outcomes: &[Outcome], elapsed: Duration) {
    let mut all: Vec<f64> = outcomes
        .iter()
        .flat_map(|o| o.latencies_ms.iter().map(|(_, ms)| *ms))
        .collect();
    all.sort_by(f64::total_cmp);
    let ops = all.len();
    println!(
        "{label:<7} ops={ops:<5} ops/s={:<9.1} p50={:.2}ms p95={:.2}ms p99={:.2}ms max={:.2}ms",
        ops as f64 / elapsed.as_secs_f64(),
        percentile(&all, 0.50),
        percentile(&all, 0.95),
        percentile(&all, 0.99),
        all.last().copied().unwrap_or(0.0),
    );
}

fn report_kinds(outcomes: &[Outcome]) {
    let mut by_kind: HashMap<&str, Vec<f64>> = HashMap::new();
    for (kind, ms) in outcomes.iter().flat_map(|o| &o.latencies_ms) {
        by_kind.entry(kind).or_default().push(*ms);
    }
    let mut kinds: Vec<_> = by_kind.into_iter().collect();
    kinds.sort_by_key(|(k, _)| *k);
    for (kind, mut ms) in kinds {
        ms.sort_by(f64::total_cmp);
        println!(
            "  {kind:<13} n={:<4} p50={:.2}ms p95={:.2}ms p99={:.2}ms max={:.2}ms",
            ms.len(),
            percentile(&ms, 0.50),
            percentile(&ms, 0.95),
            percentile(&ms, 0.99),
            ms.last().copied().unwrap_or(0.0),
        );
    }
}

fn reader(path: &Path, id: usize, gate: &Barrier) -> Outcome {
    let db = Db::open_with_options(path, &OpenOptions::default().query_only(true)).unwrap();
    let mut out = Outcome::default();
    gate.wait();
    for i in 0..READS_PER_READER {
        match (id + i) % 4 {
            // Hybrid: BM25 and KNN legs, merged as the real pipeline would.
            0 => {
                out.timed("hybrid", || {
                    let fts = db.search_fts("alpha", 10, None)?;
                    let query = [0.1 + (i % 7) as f32 / 10.0, 0.2, 0.3, 0.4];
                    let vec = db.search_vec_with_fingerprint(&query, 10, None, FP)?;
                    assert!(fts.len() + vec.len() >= SEED_DOCS.min(10));
                    Ok(())
                });
            }
            // Keyword search on a seeded rare token, scoped to one collection.
            1 => {
                out.timed("keyword", || {
                    let hits =
                        db.search_fts(&format!("seedtok{}", i % SEED_DOCS), 5, Some("seed"))?;
                    assert_eq!(hits.len(), 1);
                    Ok(())
                });
            }
            // Document retrieval of a seeded document.
            2 => {
                out.timed("get", || {
                    let doc = db.get_document("seed", &format!("s{}.md", i % SEED_DOCS))?;
                    assert!(doc.is_some_and(|d| d.body.is_some()));
                    Ok(())
                });
            }
            // Collection checks.
            _ => {
                out.timed("collections", || {
                    let status = db.status()?;
                    assert!(
                        status
                            .collections
                            .iter()
                            .any(|c| c.collection.name == "seed")
                    );
                    db.list_collections()?;
                    db.global_context()?;
                    Ok(())
                });
            }
        }
    }
    out
}

fn note_body(token: &str, tag: &str) -> String {
    format!("---\ntags: [{tag}]\n---\n# Note\nalpha {token} filed under {tag}\n")
}

fn writer(path: &Path, id: usize, gate: &Barrier) -> Outcome {
    let qmd = Qmd::open(path).unwrap();
    let mut embed_db = Db::open(path).unwrap();
    let coll = format!("bench{id}");
    let mut out = Outcome::default();
    let mut notes: Vec<usize> = Vec::new();
    let mut versions: HashMap<usize, usize> = HashMap::new();
    gate.wait();
    for i in 0..WRITES_PER_WRITER {
        match i % 4 {
            // Batch insert of notes plus their embeddings.
            0 | 1 => {
                let first = notes.len();
                let bodies: Vec<(usize, String)> = (first..first + NOTES_PER_BATCH)
                    .map(|n| (n, note_body(&format!("zq{id}n{n}v0"), "inbox")))
                    .collect();
                let written = out.timed("insert_batch", || {
                    qmd.write_batch(|batch| {
                        for (n, body) in &bodies {
                            batch.upsert_record(&coll, &format!("n{n}.md"), "Note", body)?;
                        }
                        batch.cfg_set(&format!("last-{id}"), &i.to_string())
                    })
                });
                if written.is_none() {
                    continue;
                }
                notes.extend(first..first + NOTES_PER_BATCH);
                let hashes: Vec<String> = bodies.iter().map(|(_, b)| hash_content(b)).collect();
                let rows: Vec<_> = hashes
                    .iter()
                    .enumerate()
                    .map(|(k, h)| {
                        (
                            h.as_str(),
                            0,
                            0,
                            vec![0.4, 0.3, 0.2, 0.1 + k as f32 / 100.0],
                        )
                    })
                    .collect();
                if out
                    .timed("embed", || {
                        embed_db.replace_embeddings_transactionally(FP, &rows)
                    })
                    .is_some()
                {
                    out.embedded += rows.len();
                }
            }
            // Retag existing notes (frontmatter rewrite, new content hash).
            2 => {
                let picks: Vec<usize> = notes.iter().copied().step_by(3).take(4).collect();
                let rewrites: Vec<(usize, usize, String)> = picks
                    .iter()
                    .map(|&n| {
                        let v = versions.get(&n).copied().unwrap_or(0) + 1;
                        (
                            n,
                            v,
                            note_body(&format!("zq{id}n{n}v{v}"), &format!("tag{v}")),
                        )
                    })
                    .collect();
                let done = out.timed("tag_update", || {
                    qmd.write_batch(|batch| {
                        for (n, _, body) in &rewrites {
                            batch.upsert_record(&coll, &format!("n{n}.md"), "Note", body)?;
                        }
                        Ok(())
                    })
                });
                if done.is_some() {
                    for (n, v, _) in rewrites {
                        versions.insert(n, v);
                    }
                }
            }
            // Collection updates: re-register own collection, edit shared context.
            _ => {
                let prefix = format!("/w{id}/{i}");
                let text = format!("context from writer {id} step {i}");
                let done = out.timed("context", || {
                    qmd.register_direct_collection(&coll)?;
                    if qmd.set_context("seed", &prefix, &text)? {
                        Ok(())
                    } else {
                        Err(Error::NotFound("seed".into()))
                    }
                });
                if done.is_some() {
                    out.contexts.push((prefix, text));
                }
            }
        }
    }
    for &n in &notes {
        let v = versions.get(&n).copied().unwrap_or(0);
        let tag = if v == 0 {
            "inbox".to_string()
        } else {
            format!("tag{v}")
        };
        out.expected.push(Expected {
            collection: coll.clone(),
            path: format!("n{n}.md"),
            body: note_body(&format!("zq{id}n{n}v{v}"), &tag),
            token: format!("zq{id}n{n}v{v}"),
            stale_tokens: (0..v).map(|s| format!("zq{id}n{n}v{s}")).collect(),
        });
    }
    out
}

fn seed(path: &Path) -> usize {
    let qmd = Qmd::open(path).unwrap();
    let mut db = Db::open(path).unwrap();
    qmd.write_batch(|batch| {
        for s in 0..SEED_DOCS {
            let body = format!("# Seed\nalpha seedtok{s} reference material\n");
            batch.upsert_record("seed", &format!("s{s}.md"), "Seed", &body)?;
        }
        Ok(())
    })
    .unwrap();
    let hashes: Vec<String> = (0..SEED_DOCS)
        .map(|s| hash_content(&format!("# Seed\nalpha seedtok{s} reference material\n")))
        .collect();
    let rows: Vec<_> = hashes
        .iter()
        .enumerate()
        .map(|(s, h)| (h.as_str(), 0, 0, vec![0.1, 0.2, 0.3, s as f32 / 100.0]))
        .collect();
    db.replace_embeddings_transactionally(FP, &rows).unwrap();
    SEED_DOCS
}

// Collecting is required: every thread must be spawned before the first join.
#[allow(clippy::needless_collect)]
#[test]
fn wal_contention_16_workers_zero_busy() {
    let tmp = TempDb::new("bench");
    let path = tmp.path();
    let seeded = seed(&path);
    // Idle handle keeps the WAL file alive so its size can be reported.
    let db = Db::open(&path).unwrap();

    let gate = Arc::new(Barrier::new(READERS + WRITERS + 1));
    let spawn = |kind: &'static str, id: usize| {
        let (p, g) = (path.clone(), Arc::clone(&gate));
        thread::spawn(move || {
            if kind == "r" {
                reader(&p, id, &g)
            } else {
                writer(&p, id, &g)
            }
        })
    };
    let readers: Vec<_> = (0..READERS).map(|i| spawn("r", i)).collect();
    let writers: Vec<_> = (0..WRITERS).map(|i| spawn("w", i)).collect();

    gate.wait();
    let start = Instant::now();
    let read_out: Vec<Outcome> = readers.into_iter().map(|h| h.join().unwrap()).collect();
    let read_elapsed = start.elapsed();
    let write_out: Vec<Outcome> = writers.into_iter().map(|h| h.join().unwrap()).collect();
    let write_elapsed = start.elapsed();

    println!("\n== WAL contention: {READERS} readers + {WRITERS} writers ==");
    report("reads", &read_out, read_elapsed);
    report("writes", &write_out, write_elapsed);
    report_kinds(&read_out);
    report_kinds(&write_out);
    let total_ops: usize = read_out
        .iter()
        .chain(&write_out)
        .map(|o| o.latencies_ms.len())
        .sum();
    println!(
        "total   ops={total_ops} elapsed={:.2}s aggregate ops/s={:.1}",
        write_elapsed.as_secs_f64(),
        total_ops as f64 / write_elapsed.as_secs_f64()
    );
    let wal_len = std::fs::metadata(path.with_extension("sqlite-wal")).map_or(0, |m| m.len());
    println!("wal size after run: {wal_len} bytes");

    // Zero lock errors, zero errors of any kind.
    let all_errors: Vec<&Error> = read_out
        .iter()
        .chain(&write_out)
        .flat_map(|o| &o.errors)
        .collect();
    let busy = all_errors.iter().filter(|e| is_lock_error(e)).count();
    println!(
        "busy/locked errors: {busy}, other errors: {}",
        all_errors.len() - busy
    );
    assert_eq!(busy, 0, "SQLITE_BUSY/LOCKED surfaced: {all_errors:?}");
    assert!(all_errors.is_empty(), "unexpected errors: {all_errors:?}");
    assert!(total_ops >= 800, "workload too small: {total_ops}");

    // Database integrity (sqlite-vec is registered by the earlier Db::open).
    let conn = Connection::open(&path).unwrap();
    let integrity: String = conn
        .query_row("PRAGMA integrity_check", [], |r| r.get(0))
        .unwrap();
    assert_eq!(integrity, "ok");
    let fk: i64 = conn
        .query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(fk, 0, "foreign key violations");

    // Every committed write is stored, indexed, and retrievable.
    let mut checked = 0;
    for expected in write_out.iter().flat_map(|o| &o.expected) {
        let doc = db
            .get_document(&expected.collection, &expected.path)
            .unwrap()
            .unwrap_or_else(|| panic!("missing {}/{}", expected.collection, expected.path));
        assert_eq!(doc.body.as_deref(), Some(expected.body.as_str()));
        let hits = db
            .search_fts(&expected.token, 5, Some(&expected.collection))
            .unwrap();
        assert_eq!(hits.len(), 1, "token {} not indexed once", expected.token);
        assert_eq!(hits[0].doc.path, expected.path);
        for stale in &expected.stale_tokens {
            let stale_hits = db.search_fts(stale, 5, Some(&expected.collection)).unwrap();
            assert!(stale_hits.is_empty(), "stale token {stale} still indexed");
        }
        checked += 1;
    }
    assert_eq!(checked, WRITERS * (WRITES_PER_WRITER / 2) * NOTES_PER_BATCH);

    // Embeddings from every committed batch landed.
    let embedded: usize = write_out.iter().map(|o| o.embedded).sum();
    assert_eq!(db.vector_count().unwrap(), seeded + embedded);

    // No lost context updates from concurrent read-modify-write edits.
    let seed_coll = db.get_collection("seed").unwrap().unwrap();
    for (prefix, text) in write_out.iter().flat_map(|o| &o.contexts) {
        assert_eq!(
            seed_coll.context.get(prefix),
            Some(text),
            "lost context update for {prefix}"
        );
    }
    for id in 0..WRITERS {
        let last_batch = (0..WRITES_PER_WRITER).rev().find(|i| i % 4 < 2).unwrap();
        assert_eq!(
            db.cfg_get(&format!("last-{id}")).unwrap(),
            Some(last_batch.to_string())
        );
    }
}
