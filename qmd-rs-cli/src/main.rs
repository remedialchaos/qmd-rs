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

mod mcp;

use clap::{Parser, Subcommand};
use qmd_rs::{
    Collection, DoctorCheckStatus, LlmConfig, LlmProvider, Qmd, parse_or_expand_query,
    parse_path_range, slice_body,
};

/// QMD — local search engine for markdown files.
#[derive(Parser)]
#[command(name = "qmd", version, about)]
struct Cli {
    /// Path to the SQLite index file.
    #[arg(long, env = "QMD_INDEX")]
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
        /// Restrict embedding to this collection.
        #[arg(short = 'c', long)]
        collection: Option<String>,
    },
    /// Full-text keyword search (BM25 only).
    #[command(alias = "fts")]
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
        #[arg(short = 'c', long)]
        collection: Option<String>,
        /// Run hybrid search instead of pure FTS.
        #[arg(long)]
        hybrid: bool,
        /// Output matching file paths only.
        #[arg(long)]
        files: bool,
        /// Return all matching results (no default limit).
        #[arg(short = 'a', long)]
        all: bool,
        /// Minimum score threshold.
        #[arg(long)]
        min_score: Option<f64>,
    },
    /// Vector similarity search (semantic).
    #[command(alias = "vector-search", alias = "v-search")]
    Vsearch {
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
        #[arg(short = 'c', long)]
        collection: Option<String>,
        /// Output matching file paths only.
        #[arg(long)]
        files: bool,
        /// Return all matching results (no default limit).
        #[arg(short = 'a', long)]
        all: bool,
        /// Minimum score threshold.
        #[arg(long)]
        min_score: Option<f64>,
    },
    /// Hybrid search (FTS + vector + RRF + rerank).
    #[command(alias = "deep-search")]
    Query {
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
        #[arg(short = 'c', long)]
        collection: Option<String>,
        /// LLM provider for query expansion (ollama, openai, anthropic, none).
        #[arg(long)]
        provider: Option<String>,
        /// LLM model name for query expansion.
        #[arg(long)]
        model: Option<String>,
        /// Output matching file paths only.
        #[arg(long)]
        files: bool,
        /// Return all matching results (no default limit).
        #[arg(short = 'a', long)]
        all: bool,
        /// Minimum score threshold.
        #[arg(long)]
        min_score: Option<f64>,
        /// Disable cross-encoder reranking.
        #[arg(long)]
        no_rerank: bool,
    },
    /// Expand a search query into lexical, vector, and HyDE variants.
    Expand {
        /// Search query or query document to expand.
        query: String,
        /// LLM provider (ollama, openai, anthropic, none).
        #[arg(long)]
        provider: Option<String>,
        /// LLM model name.
        #[arg(long)]
        model: Option<String>,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// List indexed documents in a collection or show collections overview.
    Ls {
        /// Collection name or collection/subpath prefix (e.g. "wiki" or "wiki/concepts").
        path: Option<String>,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Get a document by path or #docid, with optional line range.
    Get {
        /// Path (collection/file.md) or docid (#abc123), optionally with :from:count.
        path: String,
        /// Start line (1-indexed). Overrides line parsed from path.
        #[arg(long)]
        from: Option<usize>,
        /// Maximum number of lines to return. Overrides count parsed from path.
        #[arg(short = 'l', long = "lines")]
        lines: Option<usize>,
        /// Prefix lines with line numbers.
        #[arg(long)]
        line_numbers: bool,
        /// Disable line numbering.
        #[arg(long)]
        no_line_numbers: bool,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Batch retrieve documents by glob pattern or comma-separated list.
    #[command(name = "multi-get")]
    MultiGet {
        /// Glob pattern (e.g. 'wiki/concepts/*.md') or comma-separated list of paths or docids.
        pattern: String,
        /// Maximum number of lines per file.
        #[arg(short = 'l', long = "lines")]
        lines: Option<usize>,
        /// Skip files larger than this byte size (default: 65536).
        #[arg(long, default_value = "65536")]
        max_bytes: usize,
        /// Prefix lines with line numbers.
        #[arg(long)]
        line_numbers: bool,
        /// Disable line numbering.
        #[arg(long)]
        no_line_numbers: bool,
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
    /// Start the Model Context Protocol (MCP) server over stdio.
    Mcp,
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
        #[arg(long, alias = "mask", default_value = "**/*.md")]
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
        Command::Embed {
            force,
            batch,
            collection,
        } => cmd_embed(index, force, batch, collection.as_deref()),
        Command::Search {
            query,
            limit,
            json,
            offset,
            collection,
            hybrid,
            files,
            all,
            min_score,
        } => cmd_search(
            index,
            &query,
            if all { usize::MAX / 2 } else { limit },
            offset,
            collection.as_deref(),
            json,
            hybrid,
            files,
            min_score,
        ),
        Command::Vsearch {
            query,
            limit,
            json,
            offset,
            collection,
            files,
            all,
            min_score,
        } => cmd_vsearch(
            index,
            &query,
            if all { usize::MAX / 2 } else { limit },
            offset,
            collection.as_deref(),
            json,
            files,
            min_score,
        ),
        Command::Query {
            query,
            limit,
            json,
            offset,
            collection,
            provider,
            model,
            files,
            all,
            min_score,
            no_rerank,
        } => cmd_query(
            index,
            &query,
            if all { usize::MAX / 2 } else { limit },
            offset,
            collection.as_deref(),
            provider.as_deref(),
            model.as_deref(),
            json,
            files,
            min_score,
            no_rerank,
        ),
        Command::Expand {
            query,
            provider,
            model,
            json,
        } => cmd_expand(&query, provider.as_deref(), model.as_deref(), json),
        Command::Ls { path, json } => cmd_ls(index, path.as_deref(), json),
        Command::Get {
            path,
            from,
            lines,
            line_numbers,
            no_line_numbers,
            json,
        } => cmd_get(
            index,
            &path,
            from,
            lines,
            line_numbers,
            no_line_numbers,
            json,
        ),
        Command::MultiGet {
            pattern,
            lines,
            max_bytes,
            line_numbers,
            no_line_numbers,
            json,
        } => cmd_multi_get(
            index,
            &pattern,
            lines,
            max_bytes,
            line_numbers,
            no_line_numbers,
            json,
        ),
        Command::Status { json } => cmd_status(index, json),
        Command::Doctor { json } => cmd_doctor(index, json),
        Command::Context { action } => cmd_context(index, action),
        Command::Cleanup => cmd_cleanup(index),
        Command::Vacuum => cmd_vacuum(index),
        Command::Mcp => mcp::run_mcp(index),
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

fn cmd_embed(
    index: &Path,
    force: bool,
    batch: Option<usize>,
    collection: Option<&str>,
) -> qmd_rs::Result<()> {
    let mut qmd = Qmd::open(index)?;
    if force {
        qmd.clear_embeddings()?;
        println!("cleared all embeddings");
    }
    let r = qmd.embed_with_batch_in_collection(batch, collection)?;
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

#[allow(clippy::too_many_arguments)]
fn cmd_search(
    index: &Path,
    query: &str,
    limit: usize,
    offset: usize,
    collection: Option<&str>,
    json: bool,
    hybrid: bool,
    files: bool,
    min_score: Option<f64>,
) -> qmd_rs::Result<()> {
    if hybrid {
        let mut qmd = Qmd::open(index)?;
        let results = qmd.search_with_offset_in_collection(query, limit, offset, collection)?;
        print_search_results(&results, json, files, min_score)
    } else {
        let qmd = Qmd::open(index)?;
        let results = qmd.search_fts_with_offset_in_collection(query, limit, offset, collection)?;
        print_search_results(&results, json, files, min_score)
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_vsearch(
    index: &Path,
    query: &str,
    limit: usize,
    offset: usize,
    collection: Option<&str>,
    json: bool,
    files: bool,
    min_score: Option<f64>,
) -> qmd_rs::Result<()> {
    let mut qmd = Qmd::open(index)?;
    let results = qmd.search_vec_with_offset_in_collection(query, limit, offset, collection)?;
    print_search_results(&results, json, files, min_score)
}

#[allow(clippy::too_many_arguments)]
fn cmd_query(
    index: &Path,
    query: &str,
    limit: usize,
    offset: usize,
    collection: Option<&str>,
    provider: Option<&str>,
    model: Option<&str>,
    json: bool,
    files: bool,
    min_score: Option<f64>,
    no_rerank: bool,
) -> qmd_rs::Result<()> {
    let mut config = LlmConfig::from_env();
    if let Some(p) = provider {
        config.provider = p.parse().unwrap_or(LlmProvider::None);
    }
    if let Some(m) = model {
        config.model = m.to_string();
    }
    let mut qmd = Qmd::open(index)?;
    let results = qmd.query_with_options_with_offset_in_collection(
        query,
        limit,
        offset,
        collection,
        Some(&config),
        !no_rerank,
    )?;
    print_search_results(&results, json, files, min_score)
}

fn cmd_expand(
    query: &str,
    provider: Option<&str>,
    model: Option<&str>,
    json: bool,
) -> qmd_rs::Result<()> {
    let mut config = LlmConfig::from_env();
    if let Some(p) = provider {
        config.provider = p.parse().unwrap_or(LlmProvider::None);
    }
    if let Some(m) = model {
        config.model = m.to_string();
    }
    let queries = parse_or_expand_query(query, Some(&config));
    if json {
        println!("{}", serde_json::to_string_pretty(&queries)?);
    } else {
        for q in &queries {
            let kind_str = match q.kind {
                qmd_rs::QueryType::Lex => "lex",
                qmd_rs::QueryType::Vec => "vec",
                qmd_rs::QueryType::Hyde => "hyde",
                _ => "query",
            };
            println!("{kind_str}: {}", q.text);
        }
    }
    Ok(())
}

fn print_search_results(
    results: &[qmd_rs::SearchResult],
    json: bool,
    files: bool,
    min_score: Option<f64>,
) -> qmd_rs::Result<()> {
    let filtered: Vec<&qmd_rs::SearchResult> = results
        .iter()
        .filter(|r| min_score.is_none_or(|min| r.score >= min))
        .collect();

    if files {
        let mut seen = std::collections::HashSet::new();
        let paths: Vec<String> = filtered
            .iter()
            .map(|r| r.doc.display_path())
            .filter(|p| seen.insert(p.clone()))
            .collect();

        if json {
            println!("{}", serde_json::to_string_pretty(&paths)?);
        } else {
            for path in paths {
                println!("{path}");
            }
        }
        return Ok(());
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&filtered)?);
    } else if filtered.is_empty() {
        println!("no results");
    } else {
        for r in filtered {
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

fn cmd_ls(index: &Path, path: Option<&str>, json: bool) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    if let Some(p) = path {
        let (coll, subpath) = match p.split_once('/') {
            Some((c, s)) => (c, Some(s)),
            None => (p, None),
        };
        let docs = qmd.ls(Some(coll), subpath)?;
        if json {
            println!("{}", serde_json::to_string_pretty(&docs)?);
        } else if docs.is_empty() {
            println!("no matching documents");
        } else {
            for d in docs {
                println!("#{:<6} {} — {}", d.docid, d.path, d.title);
            }
        }
    } else {
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
    Ok(())
}

fn cmd_get(
    index: &Path,
    path: &str,
    from: Option<usize>,
    lines: Option<usize>,
    line_numbers: bool,
    no_line_numbers: bool,
    json: bool,
) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let (target_path, parsed_from, parsed_lines) = parse_path_range(path);
    let start_line = from.or(parsed_from);
    let line_count = lines.or(parsed_lines);
    let show_line_numbers = line_numbers && !no_line_numbers;

    let mut doc = qmd.get(target_path)?;
    if (start_line.is_some() || line_count.is_some() || show_line_numbers)
        && let Some(body) = &doc.body
    {
        let (sliced, _, _, _) = slice_body(body, start_line, line_count, show_line_numbers);
        doc.body = Some(sliced);
    }

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

fn cmd_multi_get(
    index: &Path,
    pattern: &str,
    lines: Option<usize>,
    max_bytes: usize,
    line_numbers: bool,
    no_line_numbers: bool,
    json: bool,
) -> qmd_rs::Result<()> {
    let qmd = Qmd::open(index)?;
    let mut items = qmd.multi_get(pattern, max_bytes)?;
    let show_line_numbers = line_numbers && !no_line_numbers;

    if lines.is_some() || show_line_numbers {
        for item in &mut items {
            if let Some(body) = &item.body {
                let (sliced, _, _, _) = slice_body(body, None, lines, show_line_numbers);
                item.body = Some(sliced);
            }
        }
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&items)?);
    } else if items.is_empty() {
        println!("no documents matched");
    } else {
        for item in items {
            println!("=== {} (#{})\n", item.path, item.docid);
            if item.skipped {
                if let Some(reason) = &item.skip_reason {
                    println!("[skipped: {reason}]\n");
                }
            } else if let Some(body) = &item.body {
                println!("{body}\n");
            }
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
#[allow(clippy::expect_used, clippy::panic)]
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
    #[allow(unsafe_code, clippy::disallowed_methods)]
    fn env_index_is_parsed_when_set() {
        let key = "QMD_INDEX";
        let expected = "/custom/env/index.sqlite";
        // SAFETY: Test sets and cleans up test environment variable.
        unsafe {
            std::env::set_var(key, expected);
        }
        let parsed = Cli::try_parse_from(["qmd", "status"]).expect("parse with env");
        assert_eq!(parsed.index, Some(PathBuf::from(expected)));

        let override_path = "/override/index.sqlite";
        let parsed_override = Cli::try_parse_from(["qmd", "--index", override_path, "status"])
            .expect("parse with override");
        assert_eq!(parsed_override.index, Some(PathBuf::from(override_path)));

        // SAFETY: Cleaning up test environment variable.
        unsafe {
            std::env::remove_var(key);
        }
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

    #[test]
    fn cli_parser_accepts_phase3_flags() {
        let parsed = Cli::try_parse_from([
            "qmd",
            "search",
            "query",
            "--files",
            "-a",
            "--min-score",
            "0.5",
        ])
        .expect("search flags");
        if let Command::Search {
            files,
            all,
            min_score,
            ..
        } = parsed.command
        {
            assert!(files);
            assert!(all);
            assert_eq!(min_score, Some(0.5));
        } else {
            panic!("expected Command::Search");
        }

        let parsed_query = Cli::try_parse_from([
            "qmd",
            "query",
            "query",
            "--files",
            "--no-rerank",
            "--min-score",
            "0.8",
        ])
        .expect("query flags");
        if let Command::Query {
            files,
            no_rerank,
            min_score,
            ..
        } = parsed_query.command
        {
            assert!(files);
            assert!(no_rerank);
            assert_eq!(min_score, Some(0.8));
        } else {
            panic!("expected Command::Query");
        }

        let parsed_coll = Cli::try_parse_from([
            "qmd",
            "collection",
            "add",
            ".",
            "--name",
            "test",
            "--mask",
            "*.markdown",
        ])
        .expect("collection add mask alias");
        if let Command::Collection {
            action: CollectionAction::Add { pattern, .. },
        } = parsed_coll.command
        {
            assert_eq!(pattern, "*.markdown");
        } else {
            panic!("expected CollectionAction::Add");
        }

        let parsed_mcp = Cli::try_parse_from(["qmd", "mcp"]).expect("mcp parser");
        assert!(matches!(parsed_mcp.command, Command::Mcp));

        let parsed_embed =
            Cli::try_parse_from(["qmd", "embed", "-c", "wiki", "--batch", "50", "--force"])
                .expect("embed parser");
        if let Command::Embed {
            force,
            batch,
            collection,
        } = parsed_embed.command
        {
            assert!(force);
            assert_eq!(batch, Some(50));
            assert_eq!(collection, Some("wiki".to_string()));
        } else {
            panic!("expected Command::Embed");
        }
    }
}
