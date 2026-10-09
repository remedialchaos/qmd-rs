//! Concurrent readers and writers must not fail with `SQLITE_BUSY[_SNAPSHOT]`.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use qmd_rs::db::Db;
use qmd_rs::{OpenOptions, Qmd};

const FP: &str = "test-fingerprint";

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

#[test]
fn concurrent_first_open_migrates_once() {
    let tmp = TempDb::new("open");
    let path = tmp.path();
    let barrier = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (db_path, gate) = (path.clone(), Arc::clone(&barrier));
            thread::spawn(move || {
                gate.wait();
                Db::open(&db_path).map(|_| ())
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }
    let db = Db::open(&path).unwrap();
    assert_eq!(db.cfg_get("schema_version").unwrap().as_deref(), Some("2"));
}

#[test]
fn concurrent_reads_and_immediate_writes_do_not_hit_busy() {
    let tmp = TempDb::new("rw");
    let path = tmp.path();
    Qmd::open(&path).unwrap();

    let barrier = Arc::new(Barrier::new(7));
    let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

    // Record writers: upsert + config in one transaction.
    for w in 0..3 {
        let (db_path, gate) = (path.clone(), Arc::clone(&barrier));
        handles.push(thread::spawn(move || {
            let qmd = Qmd::open(&db_path).unwrap();
            gate.wait();
            for i in 0..40 {
                qmd.write_batch(|batch| {
                    batch.upsert_record("notes", &format!("w{w}/{i}.md"), "Title", "alpha body")?;
                    batch.cfg_set(&format!("last-{w}"), &i.to_string())
                })
                .unwrap();
            }
        }));
    }

    // Embedding writers: validate + vec table + insert under the write lock.
    for w in 0..2 {
        let (db_path, gate) = (path.clone(), Arc::clone(&barrier));
        handles.push(thread::spawn(move || {
            let mut db = Db::open_with_options(&db_path, &OpenOptions::default()).unwrap();
            gate.wait();
            for i in 0..40 {
                let hash = format!("hash-{w}-{i}");
                db.replace_embeddings_transactionally(
                    FP,
                    &[(hash.as_str(), 0, 0, vec![0.1, 0.2, 0.3, 0.4])],
                )
                .unwrap();
            }
        }));
    }

    // Readers: FTS, point reads, and counts while writers run.
    for _ in 0..2 {
        let (db_path, gate) = (path.clone(), Arc::clone(&barrier));
        handles.push(thread::spawn(move || {
            let qmd = Qmd::open(&db_path).unwrap();
            gate.wait();
            for _ in 0..80 {
                qmd.db().search_fts("alpha", 10, None).unwrap();
                qmd.get_record("notes", "w0/0.md").unwrap();
                qmd.db().vector_count().unwrap();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let qmd = Qmd::open(&path).unwrap();
    assert_eq!(qmd.db().search_fts("alpha", 200, None).unwrap().len(), 120);
    assert_eq!(qmd.db().vector_count().unwrap(), 80);
    assert_eq!(qmd.db().cfg_get("last-2").unwrap().as_deref(), Some("39"));
}

#[test]
fn write_batch_rolls_back_on_error() {
    let qmd = Qmd::open_memory().unwrap();
    let result: qmd_rs::Result<()> = qmd.write_batch(|batch| {
        batch.upsert_record("notes", "a.md", "A", "body")?;
        batch.cfg_set("k", "v")?;
        Err(qmd_rs::Error::Config("boom".into()))
    });
    assert!(result.is_err());
    assert!(qmd.get_record("notes", "a.md").unwrap().is_none());
    assert_eq!(qmd.db().cfg_get("k").unwrap(), None);
}

#[test]
fn query_only_connection_rejects_writes() {
    let tmp = TempDb::new("ro");
    let path = tmp.path();
    Db::open(&path).unwrap();
    let db = Db::open_with_options(&path, &OpenOptions::default().query_only(true)).unwrap();
    assert!(db.cfg_set("k", "v").is_err());
    assert_eq!(db.cfg_get("k").unwrap(), None);
}
