//! QMD CLI — local search engine for markdown files.

#![allow(
    clippy::missing_docs_in_private_items,
    clippy::print_stdout,
    clippy::print_stderr,
    missing_docs
)]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qmd_rs::{Collection, DoctorCheckStatus, Qmd};

/// QMD — local search engine for markdown files.
#[derive(Parser)]
#[command(name = "qmd", version, about)]
struct Cli {
    /// Path to the SQLite index file.
    #[arg(long)]
    index: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Manage collections.
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },
    /// Re-index all (or specific) collections.
    Update {
        /// Only index these collections.
        #[arg(short, long)]
        collection: Vec<String>,
    },
    /// Generate vector embeddings for unembedded documents.
    Embed {
        /// Clear all embeddings first.
        #[arg(long)]
        force: bool,
        /// Maximum number of documents to embed.
        #[arg(long)]
        batch: Option<usize>,
    },
    /// Hybrid search (FTS + vector + RRF + rerank).
    Search {
        /// Search query.
        query: String,
        /// Max results.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
        /// Number of matching results to skip.
        #[arg(long, default_value = "0")]
        offset: usize,
        /// Restrict results to this collection.
        #[arg(long)]
        collection: Option<String>,
    },
    /// Full-text keyword search (BM25 only).
    Fts {
        /// Search query.
        query: String,
        /// Max results.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
        /// Number of matching results to skip.
        #[arg(long, default_value = "0")]
        offset: usize,
        /// Restrict results to this collection.
        #[arg(long)]
        collection: Option<String>,
    },
    /// Get a document by collection/path or #docid.
    Get {
        /// Path (collection/file.md) or docid (#abc123).
        path: String,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show index status.
    Status {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run read-only index diagnostics.
    Doctor {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Manage contexts.
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },
    /// Clean up inactive documents and orphaned data.
    Cleanup,
    /// Vacuum the database to reclaim space.
    Vacuum,
}

#[derive(Subcommand)]
enum CollectionAction {
    /// Register a new collection.
    Add {
        /// Absolute path to the directory.
        path: PathBuf,
        /// Collection name.
        #[arg(long)]
        name: String,
        /// Glob pattern for files (default: **/*.md).
        #[arg(long, default_value = "**/*.md")]
        pattern: String,
    },
    /// List all collections.
    List {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove a collection.
    Remove {
        /// Collection name.
        name: String,
    },
    /// Rename a collection.
    Rename {
        /// Current name.
        old: String,
        /// New name.
        new: String,
    },
}

#[derive(Subcommand)]
enum ContextAction {
    /// Add context to a collection path.
    Add {
        /// Collection name.
        collection: String,
        /// Path prefix (e.g. "/" or "/api").
        path: String,
        /// Context description text.
        text: String,
    },
    /// List all contexts.
    List {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove context from a collection path.
    #[command(name = "rm")]
    Remove {
        /// Collection name.
        collection: String,
        /// Path prefix to remove.
        path: String,
    },
    /// Set or clear global context.
    Global {
        /// Context text (omit to clear).
        text: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let index = resolve_index_path(cli.index.as_deref());
    let result = (|| {
        if cli.index.is_none()
            && !matches!(&cli.command, Command::Doctor { .. })
            && let Some(notice) = prepare_default_index(&index, Path::new("index.sqlite"))?
        {
            eprintln!("notice: {notice}");
        }
        run(&index, cli.command)
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(index: &Path, command: Command) -> qmd_rs::Result<()> {
    match command {
        Command::Collection { action } => cmd_collection(index, action),
        Command::Update { collection } => cmd_update(index, &collection),
        Command::Embed { force, batch } => cmd_embed(index, force, batch),
        Command::Search {
            query,
            limit,
            json,
            offset,
            collection,
        } => cmd_search(index, &query, limit, offset, collection.as_deref(), json),
        Command::Fts {
            query,
            limit,
            json,
            offset,
            collection,
        } => cmd_fts(index, &query, limit, offset, collection.as_deref(), json),
        Command::Get { path, json } => cmd_get(index, &path, json),
        Command::Status { json } => cmd_status(index, json),
        Command::Doctor { json } => cmd_doctor(index, json),
        Command::Context { action } => cmd_context(index, action),
        Command::Cleanup => cmd_cleanup(index),
        Command::Vacuum => cmd_vacuum(index),
    }
}

fn default_index_path(xdg_data_home: Option<&OsStr>, home: Option<&OsStr>) -> PathBuf {
    let data_home = xdg_data_home
        .map_or_else(
            || PathBuf::from(home.unwrap_or_else(|| OsStr::new("."))),
            PathBuf::from,
        )
        .join(if xdg_data_home.is_some() {
            "qmd"
        } else {
            ".local/share/qmd"
        });
    data_home.join("index.sqlite")
}

fn resolve_index_path(explicit: Option<&Path>) -> PathBuf {
    explicit.map_or_else(
        || {
            default_index_path(
                std::env::var_os("XDG_DATA_HOME").as_deref(),
                std::env::var_os("HOME").as_deref(),
            )
        },
        Path::to_path_buf,
    )
}

fn prepare_default_index(new_index: &Path, legacy_index: &Path) -> qmd_rs::Result<Option<String>> {
    if let Some(parent) = new_index.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| qmd_rs::Error::Config(format!("{}: {e}", parent.display())))?;
    }
    if legacy_index.is_file() && !new_index.exists() {
        Ok(Some(format!(
            "legacy index exists at {}; using new default at {} (legacy data was not moved)",
            legacy_index.display(),
            new_index.display()
        )))
    } else {
        Ok(None)
    }
}

fn cmd_collection(index: &Path, action: CollectionAction) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    match action {
        CollectionAction::Add {
            path,
            name,
            pattern,
        } => {
            let abs = std::fs::canonicalize(&path)
                .map_err(|e| qmd_rs::Error::Config(format!("{}: {e}", path.display())))?;
            let coll = Collection::new(&name, abs.to_string_lossy().as_ref()).with_pattern(pattern);
            qmd.register_collection(&coll)?;
            println!("registered collection '{name}' at {}", abs.display());
        }
        CollectionAction::List { json } => {
            let colls = qmd.list_collections()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&colls)?);
            } else if colls.is_empty() {
                println!("no collections registered");
            } else {
                for c in &colls {
                    println!(
                        "{:<16} {:<40} {} docs",
                        c.collection.name, c.collection.path, c.doc_count
                    );
                }
            }
        }
        CollectionAction::Remove { name } => {
            let n = qmd.remove_collection(&name)?;
            println!("removed '{name}' ({n} documents)");
        }
        CollectionAction::Rename { old, new } => {
            qmd.rename_collection(&old, &new)?;
            println!("renamed '{old}' → '{new}'");
        }
    }
    Ok(())
}

fn cmd_update(index: &Path, collections: &[String]) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let filter: Option<Vec<&str>> = if collections.is_empty() {
        None
    } else {
        Some(collections.iter().map(String::as_str).collect())
    };
    let r = qmd.update(filter.as_deref())?;
    println!(
        "{} collections: {} indexed, {} updated, {} unchanged, {} removed",
        r.collections, r.indexed, r.updated, r.unchanged, r.removed
    );
    for failure in &r.failures {
        eprintln!("failed: {}: {}", failure.path, failure.reason);
    }
    if !r.failures.is_empty() {
        return Err(qmd_rs::Error::Config(format!(
            "{} files failed during update",
            r.failures.len()
        )));
    }
    Ok(())
}

fn cmd_embed(index: &Path, force: bool, batch: Option<usize>) -> qmd_rs::Result<()> {
    let mut qmd = Qmd::open(index)?;
    if force {
        qmd.clear_embeddings()?;
        println!("cleared all embeddings");
    }
    let r = qmd.embed_with_batch(batch)?;
    for message in &r.failure_messages {
        eprintln!("{message}");
    }
    embed_report(r.embedded, r.chunks, r.remaining, r.failures)
}

/// Print embed totals and return an error when any job failed.
///
/// Partial progress is reported as-is; a failure is never silently treated as
/// success, so `qmd embed` exits nonzero.
fn embed_report(
    embedded: usize,
    chunks: usize,
    remaining: usize,
    failures: usize,
) -> qmd_rs::Result<()> {
    println!(
        "{embedded} documents embedded, {chunks} chunks; {remaining} remaining; {failures} failures"
    );
    if failures > 0 {
        return Err(qmd_rs::Error::Config(format!(
            "{failures} embedding jobs failed; {embedded} documents ({chunks} chunks) published and {remaining} documents still need embedding"
        )));
    }
    Ok(())
}

fn cmd_search(
    index: &Path,
    query: &str,
    limit: usize,
    offset: usize,
    collection: Option<&str>,
    json: bool,
) -> qmd_rs::Result<()> {
    let mut qmd = Qmd::open(index)?;
    let results = qmd.search_with_offset_in_collection(query, limit, offset, collection)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if results.is_empty() {
        println!("no results");
    } else {
        for r in &results {
            println!(
                "{:.3}  #{} {} — {}",
                r.score,
                r.doc.docid(),
                r.doc.display_path(),
                r.doc.title,
            );
        }
    }
    Ok(())
}

fn cmd_fts(
    index: &Path,
    query: &str,
    limit: usize,
    offset: usize,
    collection: Option<&str>,
    json: bool,
) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let results = qmd.search_fts_with_offset_in_collection(query, limit, offset, collection)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if results.is_empty() {
        println!("no results");
    } else {
        for r in &results {
            println!(
                "{:.3}  #{} {} — {}",
                r.score,
                r.doc.docid(),
                r.doc.display_path(),
                r.doc.title,
            );
        }
    }
    Ok(())
}

fn cmd_get(index: &Path, path: &str, json: bool) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let doc = qmd.get(path)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&doc)?);
    } else {
        println!("# {}\n", doc.title);
        if let Some(body) = &doc.body {
            println!("{body}");
        }
    }
    Ok(())
}

fn cmd_status(index: &Path, json: bool) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let s = qmd.status()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&s)?);
    } else {
        println!("documents:        {}", s.total_documents);
        println!("needs embedding:  {}", s.needs_embedding);
        println!(
            "vector index:     {}",
            if s.has_vector_index { "yes" } else { "no" }
        );
        if !s.collections.is_empty() {
            println!("\ncollections:");
            for c in &s.collections {
                println!(
                    "  {:<16} {} docs  {}",
                    c.collection.name, c.doc_count, c.collection.path,
                );
            }
        }
    }
    Ok(())
}

fn cmd_doctor(index: &Path, json: bool) -> qmd_rs::Result<()> {
    let report = Qmd::doctor(index)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for check in &report.checks {
            let status = match check.status {
                DoctorCheckStatus::Ok => "ok",
                DoctorCheckStatus::Warning => "warning",
                DoctorCheckStatus::Error => "error",
                _ => "unknown",
            };
            println!("{status:<7} {:<24} {}", check.name, check.detail);
        }
    }
    if report.has_errors() {
        return Err(qmd_rs::Error::Config("doctor found index errors".into()));
    }
    Ok(())
}

fn cmd_context(index: &Path, action: ContextAction) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    match action {
        ContextAction::Add {
            collection,
            path,
            text,
        } => {
            if qmd.set_context(&collection, &path, &text)? {
                println!("context set for {collection}:{path}");
            } else {
                eprintln!("collection '{collection}' not found");
            }
        }
        ContextAction::List { json } => {
            let colls = qmd.db().list_collections()?;
            let global = qmd.global_context()?;
            if json {
                let mut all = Vec::new();
                if let Some(g) = &global {
                    all.push(serde_json::json!({"collection": "*", "path": "/", "context": g}));
                }
                for c in &colls {
                    for (p, t) in &c.context {
                        all.push(
                            serde_json::json!({"collection": c.name, "path": p, "context": t}),
                        );
                    }
                }
                println!("{}", serde_json::to_string_pretty(&all)?);
            } else {
                if let Some(g) = &global {
                    println!("*  /  {g}");
                }
                for c in &colls {
                    for (p, t) in &c.context {
                        println!("{}  {}  {t}", c.name, p);
                    }
                }
            }
        }
        ContextAction::Remove { collection, path } => {
            if qmd.remove_context(&collection, &path)? {
                println!("context removed for {collection}:{path}");
            } else {
                eprintln!("context not found");
            }
        }
        ContextAction::Global { text } => {
            qmd.set_global_context(text.as_deref())?;
            if text.is_some() {
                println!("global context set");
            } else {
                println!("global context cleared");
            }
        }
    }
    Ok(())
}

fn cmd_cleanup(index: &Path) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let n = qmd.cleanup()?;
    println!("{n} items cleaned up");
    Ok(())
}

fn cmd_vacuum(index: &Path) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    qmd.vacuum()?;
    println!("database vacuumed");
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("qmd-cli-{name}-{}", std::process::id()))
    }

    #[test]
    fn embed_report_fails_and_preserves_totals_when_jobs_fail() {
        let ok = embed_report(4, 9, 0, 0);
        assert!(ok.is_ok(), "clean run must succeed: {ok:?}");

        let err = embed_report(2, 5, 3, 1).expect_err("partial failure must exit nonzero");
        let message = err.to_string();
        assert!(
            message.contains("1 embedding job"),
            "failure count missing: {message}"
        );
        assert!(
            message.contains("2 documents"),
            "published total missing: {message}"
        );
        assert!(
            message.contains("5 chunks"),
            "chunk total missing: {message}"
        );
        assert!(
            message.contains("3 documents"),
            "remaining total missing: {message}"
        );
    }

    #[test]
    fn default_path_uses_xdg_data_home() {
        assert_eq!(
            default_index_path(Some(OsStr::new("/xdg")), Some(OsStr::new("/home/user"))),
            PathBuf::from("/xdg/qmd/index.sqlite")
        );
    }

    #[test]
    fn default_path_falls_back_to_home_local_share() {
        assert_eq!(
            default_index_path(None, Some(OsStr::new("/home/user"))),
            PathBuf::from("/home/user/.local/share/qmd/index.sqlite")
        );
    }

    #[test]
    fn explicit_index_is_preserved() {
        let explicit = PathBuf::from("relative/custom.sqlite");
        assert_eq!(resolve_index_path(Some(&explicit)), explicit);
    }

    #[test]
    fn default_parent_is_created_and_legacy_notice_is_non_destructive() {
        let temp = test_dir("legacy");
        let legacy = temp.join("index.sqlite");
        fs::create_dir_all(&temp).expect("tempdir");
        fs::write(&legacy, b"legacy").expect("legacy index");
        let new_index = temp.join("data/qmd/index.sqlite");

        let notice = prepare_default_index(&new_index, &legacy).expect("prepare index");

        assert!(new_index.parent().expect("parent").is_dir());
        assert!(legacy.is_file());
        assert!(!new_index.exists());
        assert!(notice.is_some());
        fs::remove_dir_all(temp).expect("cleanup");
    }

    #[test]
    fn no_legacy_notice_when_new_default_exists() {
        let temp = test_dir("new");
        let legacy = temp.join("index.sqlite");
        let new_index = temp.join("data/qmd/index.sqlite");
        fs::create_dir_all(new_index.parent().expect("parent")).expect("parent");
        fs::write(&legacy, b"legacy").expect("legacy index");
        fs::write(&new_index, b"new").expect("new index");

        assert!(
            prepare_default_index(&new_index, &legacy)
                .expect("prepare index")
                .is_none()
        );
        fs::remove_dir_all(temp).expect("cleanup");
    }
}
