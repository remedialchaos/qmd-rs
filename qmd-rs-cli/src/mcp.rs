//! MCP (Model Context Protocol) server for QMD.

#![allow(
    clippy::cast_possible_truncation,
    clippy::format_push_string,
    clippy::missing_docs_in_private_items,
    clippy::print_stdout,
    clippy::print_stderr,
    missing_docs
)]

use std::io::{BufRead, Write};
use std::path::Path;

use qmd_rs::{
    LlmConfig, Qmd, Query, SearchResult, parse_or_expand_query, parse_path_range, slice_body,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// Run the stdio MCP server loop.
pub fn run_mcp(index: &Path) -> qmd_rs::Result<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();

    run_mcp_io(index, &mut reader, &mut writer)
}

/// Process JSON-RPC messages from a buffered reader and write responses to a writer.
pub fn run_mcp_io<R: BufRead, W: Write>(
    index: &Path,
    reader: &mut R,
    writer: &mut W,
) -> qmd_rs::Result<()> {
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(trimmed) {
            if let Some(resp) = handle_request(index, req) {
                let serialized = serde_json::to_string(&resp)?;
                writer.write_all(serialized.as_bytes())?;
                writer.write_all(b"\n")?;
                writer.flush()?;
            }
        } else {
            // Invalid JSON
            let err_resp = JsonRpcResponse {
                jsonrpc: "2.0",
                id: Value::Null,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: "Parse error: invalid JSON".to_string(),
                    data: None,
                }),
            };
            let serialized = serde_json::to_string(&err_resp)?;
            writer.write_all(serialized.as_bytes())?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        line.clear();
    }
    Ok(())
}

fn handle_request(index: &Path, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    let id = req.id.unwrap_or(Value::Null);

    // Notifications (no ID) that need no response
    if id.is_null() && req.method.starts_with("notifications/") {
        return None;
    }

    if req.jsonrpc != "2.0" {
        return Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request: jsonrpc version must be '2.0'".to_string(),
                data: None,
            }),
        });
    }

    let response = match req.method.as_str() {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "qmd",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        },
        "ping" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        },
        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "tools": tools_list()
            })),
            error: None,
        },
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let result = handle_tool_call(index, &params);
            JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(result),
                error: None,
            }
        }
        _ => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
                data: None,
            }),
        },
    };

    Some(response)
}

fn tools_list() -> Value {
    json!([
        {
            "name": "query",
            "description": "Hybrid search combining full-text BM25, vector search, RRF fusion, and optional LLM reranking",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "searches": {
                        "type": "array",
                        "description": "Typed sub-queries (lex/vec/hyde)",
                        "items": { "type": "string" }
                    },
                    "collections": {
                        "type": "array",
                        "description": "Filter results to these collections",
                        "items": { "type": "string" }
                    },
                    "limit": { "type": "integer", "description": "Maximum number of results to return (default 10)" },
                    "minScore": { "type": "number", "description": "Minimum score threshold (0.0 to 1.0)" },
                    "rerank": { "type": "boolean", "description": "Run reranker (default true)" }
                }
            }
        },
        {
            "name": "get",
            "description": "Retrieve a document by path, docid (#abc123), or path:from:count",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": { "type": "string", "description": "File path, docid (#abc123), or path:from:count" },
                    "path": { "type": "string", "description": "Alias for file" },
                    "fromLine": { "type": "integer", "description": "Starting line number (1-indexed)" },
                    "maxLines": { "type": "integer", "description": "Maximum number of lines to return" },
                    "lineNumbers": { "type": "boolean", "description": "Prefix lines with line numbers (default true)" }
                }
            }
        },
        {
            "name": "multi_get",
            "description": "Batch retrieve documents by glob pattern or comma-separated list",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern (e.g. docs/*.md) or comma-separated paths/docids" },
                    "maxBytes": { "type": "integer", "description": "Maximum bytes per file (default 10240)" },
                    "maxLines": { "type": "integer", "description": "Maximum lines per file" },
                    "lineNumbers": { "type": "boolean", "description": "Prefix lines with line numbers (default true)" }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "search",
            "description": "Fast keyword/BM25 full-text search across documents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search keywords" },
                    "collection": { "type": "string", "description": "Collection name filter" },
                    "limit": { "type": "integer", "description": "Maximum number of results (default 10)" },
                    "minScore": { "type": "number", "description": "Minimum score threshold" },
                    "files": { "type": "boolean", "description": "Return file paths only" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "vsearch",
            "description": "Semantic vector similarity search across documents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Semantic search query" },
                    "collection": { "type": "string", "description": "Collection name filter" },
                    "limit": { "type": "integer", "description": "Maximum number of results (default 10)" },
                    "minScore": { "type": "number", "description": "Minimum score threshold" },
                    "files": { "type": "boolean", "description": "Return file paths only" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "status",
            "description": "Get index health, collection information, and document counts",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ])
}

fn handle_tool_call(index: &Path, params: &Value) -> Value {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = params.get("arguments").unwrap_or(&Value::Null);

    let result = match name {
        "query" => tool_query(index, args),
        "get" => tool_get(index, args),
        "multi_get" | "multi-get" => tool_multi_get(index, args),
        "search" => tool_search(index, args),
        "vsearch" => tool_vsearch(index, args),
        "status" => tool_status(index),
        _ => Err(format!("Unknown tool: {name}")),
    };

    match result {
        Ok(text) => json!({
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ],
            "isError": false
        }),
        Err(err_msg) => json!({
            "content": [
                {
                    "type": "text",
                    "text": err_msg
                }
            ],
            "isError": true
        }),
    }
}

fn tool_query(index: &Path, args: &Value) -> Result<String, String> {
    let query_str = args
        .get("query")
        .and_then(Value::as_str)
        .or_else(|| args.get("q").and_then(Value::as_str));

    let searches_val = args.get("searches").and_then(Value::as_array);

    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(10, |v| v as usize);

    let min_score = args
        .get("minScore")
        .or_else(|| args.get("min_score"))
        .and_then(Value::as_f64);

    let rerank = args
        .get("rerank")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            !args
                .get("noRerank")
                .or_else(|| args.get("no_rerank"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        });

    let collections: Vec<String> =
        if let Some(arr) = args.get("collections").and_then(Value::as_array) {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        } else if let Some(c) = args.get("collection").and_then(Value::as_str) {
            vec![c.to_string()]
        } else {
            Vec::new()
        };

    let mut qmd = Qmd::open(index).map_err(|e| e.to_string())?;

    let mut queries = Vec::new();
    let root_query = query_str.unwrap_or("").to_string();

    if let Some(searches) = searches_val {
        for s in searches {
            if let Some(s_str) = s.as_str() {
                if let Some(rest) = s_str.strip_prefix("lex:") {
                    queries.push(Query::lex(rest.trim()));
                } else if let Some(rest) = s_str.strip_prefix("vec:") {
                    queries.push(Query::vec(rest.trim()));
                } else if let Some(rest) = s_str.strip_prefix("hyde:") {
                    queries.push(Query::hyde(rest.trim()));
                } else {
                    queries.push(Query::lex(s_str.trim()));
                    queries.push(Query::vec(s_str.trim()));
                }
            } else if let Some(obj) = s.as_object() {
                let kind = obj.get("type").and_then(Value::as_str).unwrap_or("lex");
                let text = obj.get("query").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "vec" => queries.push(Query::vec(text)),
                    "hyde" => queries.push(Query::hyde(text)),
                    _ => queries.push(Query::lex(text)),
                }
            }
        }
    } else if !root_query.is_empty() {
        let config = LlmConfig::from_env();
        queries = parse_or_expand_query(&root_query, Some(&config));
    } else {
        return Err("Either 'query' or 'searches' must be provided".to_string());
    }

    let mut results: Vec<SearchResult> = if collections.is_empty() {
        qmd.search_with_queries_options(&root_query, &queries, limit, 0, None, rerank)
            .map_err(|e| e.to_string())?
    } else {
        let mut combined = Vec::new();
        for col in &collections {
            let mut hits = qmd
                .search_with_queries_options(&root_query, &queries, limit, 0, Some(col), rerank)
                .map_err(|e| e.to_string())?;
            combined.append(&mut hits);
        }
        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(limit);
        combined
    };

    if let Some(min) = min_score {
        results.retain(|r| r.score >= min);
    }

    if results.is_empty() {
        return Ok("No matching documents found.".to_string());
    }

    let mut out = format!("Found {} results:\n\n", results.len());
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. **{}** (`{}`) — Score: {:.3} — docid: #{}\n",
            i + 1,
            r.doc.title,
            r.doc.display_path(),
            r.score,
            r.doc.docid()
        ));
    }
    Ok(out)
}

fn tool_get(index: &Path, args: &Value) -> Result<String, String> {
    let file_arg = args
        .get("file")
        .or_else(|| args.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing required parameter 'file'".to_string())?;

    let from_arg = args
        .get("fromLine")
        .or_else(|| args.get("from"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    let lines_arg = args
        .get("maxLines")
        .or_else(|| args.get("lines"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    let line_numbers = args
        .get("lineNumbers")
        .or_else(|| args.get("line_numbers"))
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let qmd = Qmd::open(index).map_err(|e| e.to_string())?;
    let (target, path_from, path_lines) = parse_path_range(file_arg);
    let start_line = from_arg.or(path_from);
    let line_count = lines_arg.or(path_lines);

    let mut doc = qmd.get(target).map_err(|e| e.to_string())?;

    if (start_line.is_some() || line_count.is_some() || line_numbers)
        && let Some(body) = &doc.body
    {
        let (sliced, _, _, _) = slice_body(body, start_line, line_count, line_numbers);
        doc.body = Some(sliced);
    }

    let docid = doc.docid().to_string();
    let body_text = doc.body.as_deref().unwrap_or_default();
    Ok(format!(
        "# {} (`{}`)\nDocid: #{docid}\n\n{body_text}",
        doc.title, doc.path,
    ))
}

fn tool_multi_get(index: &Path, args: &Value) -> Result<String, String> {
    let pattern = args
        .get("pattern")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing required parameter 'pattern'".to_string())?;

    let max_bytes = args
        .get("maxBytes")
        .or_else(|| args.get("max_bytes"))
        .and_then(Value::as_u64)
        .map_or(10240, |v| v as usize);

    let lines = args
        .get("maxLines")
        .or_else(|| args.get("lines"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    let line_numbers = args
        .get("lineNumbers")
        .or_else(|| args.get("line_numbers"))
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let qmd = Qmd::open(index).map_err(|e| e.to_string())?;
    let mut items = qmd
        .multi_get(pattern, max_bytes)
        .map_err(|e| e.to_string())?;

    if items.is_empty() {
        return Ok(format!("No documents matched pattern: '{pattern}'"));
    }

    let mut out = String::new();
    for item in &mut items {
        let display = if item.collection.is_empty() {
            item.path.clone()
        } else {
            format!("{}/{}", item.collection, item.path)
        };
        out.push_str(&format!("=== {} (#{})\n", display, item.docid));
        if item.skipped {
            if let Some(reason) = &item.skip_reason {
                out.push_str(&format!("[skipped: {reason}]\n\n"));
            }
        } else if let Some(body) = &item.body {
            let (sliced, _, _, _) = slice_body(body, None, lines, line_numbers);
            out.push_str(&sliced);
            out.push_str("\n\n");
        }
    }
    Ok(out)
}

fn tool_search(index: &Path, args: &Value) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing required parameter 'query'".to_string())?;

    let collection = args.get("collection").and_then(Value::as_str);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(10, |v| v as usize);
    let min_score = args
        .get("minScore")
        .or_else(|| args.get("min_score"))
        .and_then(Value::as_f64);
    let files = args.get("files").and_then(Value::as_bool).unwrap_or(false);

    let qmd = Qmd::open(index).map_err(|e| e.to_string())?;
    let mut results = qmd
        .search_fts_with_offset_in_collection(query, limit, 0, collection)
        .map_err(|e| e.to_string())?;

    if let Some(min) = min_score {
        results.retain(|r| r.score >= min);
    }

    if files {
        let mut seen = std::collections::HashSet::new();
        let paths: Vec<String> = results
            .iter()
            .map(|r| r.doc.display_path())
            .filter(|p| seen.insert(p.clone()))
            .collect();
        return Ok(paths.join("\n"));
    }

    if results.is_empty() {
        return Ok("No matching documents found.".to_string());
    }

    let mut out = format!("Found {} results:\n\n", results.len());
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. **{}** (`{}`) — Score: {:.3} — docid: #{}\n",
            i + 1,
            r.doc.title,
            r.doc.display_path(),
            r.score,
            r.doc.docid()
        ));
    }
    Ok(out)
}

fn tool_vsearch(index: &Path, args: &Value) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing required parameter 'query'".to_string())?;

    let collection = args.get("collection").and_then(Value::as_str);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(10, |v| v as usize);
    let min_score = args
        .get("minScore")
        .or_else(|| args.get("min_score"))
        .and_then(Value::as_f64);
    let files = args.get("files").and_then(Value::as_bool).unwrap_or(false);

    let mut qmd = Qmd::open(index).map_err(|e| e.to_string())?;
    let mut results = qmd
        .search_vec_with_offset_in_collection(query, limit, 0, collection)
        .map_err(|e| e.to_string())?;

    if let Some(min) = min_score {
        results.retain(|r| r.score >= min);
    }

    if files {
        let mut seen = std::collections::HashSet::new();
        let paths: Vec<String> = results
            .iter()
            .map(|r| r.doc.display_path())
            .filter(|p| seen.insert(p.clone()))
            .collect();
        return Ok(paths.join("\n"));
    }

    if results.is_empty() {
        return Ok("No matching documents found.".to_string());
    }

    let mut out = format!("Found {} results:\n\n", results.len());
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. **{}** (`{}`) — Score: {:.3} — docid: #{}\n",
            i + 1,
            r.doc.title,
            r.doc.display_path(),
            r.score,
            r.doc.docid()
        ));
    }
    Ok(out)
}

fn tool_status(index: &Path) -> Result<String, String> {
    let qmd = Qmd::open(index).map_err(|e| e.to_string())?;
    let status = qmd.status().map_err(|e| e.to_string())?;

    let mut out = format!(
        "QMD Status:\n- Total documents: {}\n- Needs embedding: {}\n- Vector index: {}\n- Collections: {}\n",
        status.total_documents,
        status.needs_embedding,
        if status.has_vector_index { "yes" } else { "no" },
        status.collections.len()
    );

    for c in &status.collections {
        out.push_str(&format!(
            "  - {} ({}): {} docs\n",
            c.collection.name, c.collection.path, c.doc_count
        ));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_initialize_and_tools_list() {
        let temp = std::env::temp_dir().join(format!("qmd-mcp-test-{}", std::process::id()));
        let db_path = temp.join("index.sqlite");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).expect("tempdir");

        let input = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":3,"method":"ping"}"#,
            "\n"
        );

        let mut reader = Cursor::new(input);
        let mut writer = Vec::new();

        run_mcp_io(&db_path, &mut reader, &mut writer).expect("mcp loop");

        let output = String::from_utf8(writer).expect("utf8");
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "Expected 3 responses (init, tools/list, ping)"
        );

        let init_resp: Value = serde_json::from_str(lines[0]).expect("init json");
        assert_eq!(init_resp["id"], 1);
        assert_eq!(init_resp["result"]["serverInfo"]["name"], "qmd");

        let tools_resp: Value = serde_json::from_str(lines[1]).expect("tools json");
        assert_eq!(tools_resp["id"], 2);
        let tools = tools_resp["result"]["tools"]
            .as_array()
            .expect("tools array");
        assert!(tools.iter().any(|t| t["name"] == "query"));
        assert!(tools.iter().any(|t| t["name"] == "get"));
        assert!(tools.iter().any(|t| t["name"] == "multi_get"));
        assert!(tools.iter().any(|t| t["name"] == "status"));

        let ping_resp: Value = serde_json::from_str(lines[2]).expect("ping json");
        assert_eq!(ping_resp["id"], 3);

        let _ = std::fs::remove_dir_all(temp);
    }

    #[test]
    fn test_tool_calls_end_to_end() {
        let temp = std::env::temp_dir().join(format!("qmd-mcp-e2e-{}", std::process::id()));
        let db_path = temp.join("index.sqlite");
        let docs_dir = temp.join("docs");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&docs_dir).expect("docsdir");

        let doc1 = docs_dir.join("intro.md");
        std::fs::write(
            &doc1,
            "# Welcome\n\nThis is an introduction to QMD.\nIt supports local search.\n",
        )
        .expect("write doc1");

        let qmd = Qmd::open(&db_path).expect("open");
        let coll = qmd_rs::Collection::new("docs", docs_dir.to_str().expect("str"));
        qmd.register_collection(&coll).expect("register");
        qmd.update(Some(&["docs"])).expect("update");

        let input = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get","arguments":{"file":"docs/intro.md"}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search","arguments":{"query":"introduction"}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"query","arguments":{"query":"search"}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"multi_get","arguments":{"pattern":"docs/*.md"}}}"#,
            "\n"
        );

        let mut reader = Cursor::new(input);
        let mut writer = Vec::new();

        run_mcp_io(&db_path, &mut reader, &mut writer).expect("mcp loop");

        let output = String::from_utf8(writer).expect("utf8");
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 5, "Expected 5 responses");

        // Status
        let status_resp: Value = serde_json::from_str(lines[0]).expect("status json");
        assert_eq!(status_resp["id"], 1);
        assert_eq!(status_resp["result"]["isError"], false);
        let status_text = status_resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(status_text.contains("Total documents: 1"));

        // Get
        let get_resp: Value = serde_json::from_str(lines[1]).expect("get json");
        assert_eq!(get_resp["id"], 2);
        assert_eq!(get_resp["result"]["isError"], false);
        let get_text = get_resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(get_text.contains("Welcome"));
        assert!(get_text.contains("This is an introduction"));

        // Search
        let search_resp: Value = serde_json::from_str(lines[2]).expect("search json");
        assert_eq!(search_resp["id"], 3);
        assert_eq!(search_resp["result"]["isError"], false);
        let search_text = search_resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(search_text.contains("Found 1 results"));
        assert!(search_text.contains("Welcome"));

        // Query
        let query_resp: Value = serde_json::from_str(lines[3]).expect("query json");
        assert_eq!(query_resp["id"], 4);
        assert_eq!(query_resp["result"]["isError"], false);
        let query_text = query_resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(query_text.contains("Found 1 results"));

        // Multi-get
        let multi_resp: Value = serde_json::from_str(lines[4]).expect("multi json");
        assert_eq!(multi_resp["id"], 5);
        assert_eq!(multi_resp["result"]["isError"], false);
        let multi_text = multi_resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(multi_text.contains("=== docs/intro.md"));

        let _ = std::fs::remove_dir_all(temp);
    }
}
