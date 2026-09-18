//! LLM query expansion adapters (Ollama, OpenAI-compatible, Anthropic).
//!
//! Provides automated query expansion into lexical (`lex:`), semantic vector (`vec:`),
//! and hypothetical document embedding (`hyde:`) variants to maximize retrieval recall.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::search::Query;

/// Supported LLM provider kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum LlmProvider {
    /// Local Ollama server (`/api/chat`).
    Ollama,
    /// OpenAI or OpenAI-compatible endpoint (`/chat/completions`).
    #[serde(rename = "openai")]
    OpenAi,
    /// Anthropic Messages API (`/v1/messages`).
    Anthropic,
    /// No external LLM (heuristic/simple expansion only).
    #[default]
    None,
}

impl std::str::FromStr for LlmProvider {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "ollama" => Ok(Self::Ollama),
            "openai" | "open-ai" | "compatible" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            _ => Ok(Self::None),
        }
    }
}

/// Configuration for external LLM query expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LlmConfig {
    /// Active provider.
    pub provider: LlmProvider,
    /// Model name identifier.
    pub model: String,
    /// Base URL for the provider API.
    pub url: String,
    /// Optional API authentication key.
    pub api_key: Option<String>,
    /// Request timeout in milliseconds (default: 5000 ms).
    pub timeout_ms: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::None,
            model: String::new(),
            url: String::new(),
            api_key: None,
            timeout_ms: 5000,
        }
    }
}

impl LlmConfig {
    /// Construct configuration from environment variables.
    ///
    /// Reads:
    /// - `QMD_LLM_PROVIDER`: `ollama`, `openai`, `anthropic`, `none`
    /// - `QMD_LLM_MODEL`: model name
    /// - `QMD_LLM_URL` / `OPENAI_BASE_URL` / `OLLAMA_HOST` / `ANTHROPIC_BASE_URL`
    /// - `QMD_LLM_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
    /// - `QMD_LLM_TIMEOUT_MS`: timeout in milliseconds
    #[must_use]
    pub fn from_env() -> Self {
        let provider_str = std::env::var("QMD_LLM_PROVIDER").unwrap_or_default();
        let provider: LlmProvider = provider_str.parse().unwrap_or(LlmProvider::None);

        let url = std::env::var("QMD_LLM_URL").unwrap_or_else(|_| match provider {
            LlmProvider::Ollama => std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            LlmProvider::OpenAi => std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            LlmProvider::Anthropic => std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            LlmProvider::None => String::new(),
        });

        let api_key = std::env::var("QMD_LLM_API_KEY")
            .ok()
            .or_else(|| match provider {
                LlmProvider::OpenAi => std::env::var("OPENAI_API_KEY").ok(),
                LlmProvider::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok(),
                _ => None,
            });

        let model = std::env::var("QMD_LLM_MODEL").unwrap_or_else(|_| match provider {
            LlmProvider::Ollama => "llama3.2".to_string(),
            LlmProvider::OpenAi => "gpt-4o-mini".to_string(),
            LlmProvider::Anthropic => "claude-3-5-haiku-20241022".to_string(),
            LlmProvider::None => String::new(),
        });

        let timeout_ms = std::env::var("QMD_LLM_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000);

        Self {
            provider,
            model,
            url,
            api_key,
            timeout_ms,
        }
    }
}

/// System prompt used for query expansion.
pub const EXPANSION_SYSTEM_PROMPT: &str = "\
You are a search query expansion assistant for a markdown documentation and knowledge base.
Given the user's search query, output 3 to 5 complementary query variations to improve retrieval recall.
You MUST format each variation on its own line using one of these prefixes:
lex: <keywords and synonyms for BM25 lexical search>
vec: <natural language concept or question for semantic vector search>
hyde: <a concise 1-2 sentence hypothetical answer snippet that would appear in an ideal document>

Requirements:
- Output at least one 'lex:' line and at least one 'vec:' line.
- Do NOT output explanations, markdown code blocks, numbering, or introductory text.
- Output ONLY the prefixed lines.";

/// Expand a query using the configured LLM provider, falling back to simple expansion on failure.
#[must_use]
pub fn expand_query(query: &str, config: Option<&LlmConfig>) -> Vec<Query> {
    let Some(cfg) = config else {
        return Query::expand_simple(query);
    };

    if cfg.provider == LlmProvider::None {
        return Query::expand_simple(query);
    }

    match call_provider(query, cfg) {
        Ok(raw_output) => Query::from_llm_output(&raw_output, query),
        Err(_) => Query::expand_simple(query),
    }
}

/// Parse explicit query document input or expand single-line query.
///
/// Matches upstream `tobi/qmd` behavior:
/// - If input contains explicit `lex:`, `vec:`, or `hyde:` prefixes on any line, it parses them directly.
/// - If input begins with `expand:`, it strips the prefix and runs expansion.
/// - Otherwise, it expands the plain query text.
#[must_use]
pub fn parse_or_expand_query(input: &str, config: Option<&LlmConfig>) -> Vec<Query> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Check for explicit query document prefixes
    let has_typed_lines = trimmed.lines().any(|l| {
        let line = l.trim();
        line.starts_with("lex:") || line.starts_with("vec:") || line.starts_with("hyde:")
    });

    if has_typed_lines {
        return Query::from_llm_output(trimmed, trimmed);
    }

    let search_term = trimmed.strip_prefix("expand:").unwrap_or(trimmed).trim();

    expand_query(search_term, config)
}

/// Send request to the designated provider.
fn call_provider(query: &str, config: &LlmConfig) -> Result<String> {
    let timeout = Duration::from_millis(config.timeout_ms);
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();

    match config.provider {
        LlmProvider::Ollama => call_ollama(&agent, query, config),
        LlmProvider::OpenAi => call_openai(&agent, query, config),
        LlmProvider::Anthropic => call_anthropic(&agent, query, config),
        LlmProvider::None => Err(Error::Config("no provider configured".into())),
    }
}

/// Query expansion via Ollama `/api/chat`.
fn call_ollama(agent: &ureq::Agent, query: &str, config: &LlmConfig) -> Result<String> {
    let base_url = config.url.trim_end_matches('/');
    let endpoint = format!("{base_url}/api/chat");

    let payload = serde_json::json!({
        "model": config.model,
        "messages": [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        "stream": false
    });

    let resp = agent
        .post(&endpoint)
        .send_json(payload)
        .map_err(|e| Error::Config(format!("ollama request failed: {e}")))?;

    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| Error::Config(format!("invalid ollama response: {e}")))?;

    json.get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(str::to_string)
        .ok_or_else(|| Error::Config("missing message.content in ollama response".into()))
}

/// Query expansion via OpenAI-compatible `/chat/completions`.
fn call_openai(agent: &ureq::Agent, query: &str, config: &LlmConfig) -> Result<String> {
    let base_url = config.url.trim_end_matches('/');
    let endpoint = if base_url.ends_with("/v1") {
        format!("{base_url}/chat/completions")
    } else {
        format!("{base_url}/v1/chat/completions")
    };

    let mut req = agent.post(&endpoint);
    if let Some(key) = &config.api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }

    let payload = serde_json::json!({
        "model": config.model,
        "messages": [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        "temperature": 0.3
    });

    let resp = req
        .send_json(payload)
        .map_err(|e| Error::Config(format!("openai request failed: {e}")))?;

    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| Error::Config(format!("invalid openai response: {e}")))?;

    json.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            Error::Config("missing choices[0].message.content in openai response".into())
        })
}

/// Query expansion via Anthropic Messages API.
fn call_anthropic(agent: &ureq::Agent, query: &str, config: &LlmConfig) -> Result<String> {
    let base_url = config.url.trim_end_matches('/');
    let endpoint = if base_url.ends_with("/v1") {
        format!("{base_url}/messages")
    } else {
        format!("{base_url}/v1/messages")
    };

    let mut req = agent
        .post(&endpoint)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json");

    if let Some(key) = &config.api_key {
        req = req.set("x-api-key", key);
    }

    let payload = serde_json::json!({
        "model": config.model,
        "system": EXPANSION_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": 512
    });

    let resp = req
        .send_json(payload)
        .map_err(|e| Error::Config(format!("anthropic request failed: {e}")))?;

    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| Error::Config(format!("invalid anthropic response: {e}")))?;

    json.get("content")
        .and_then(|c| c.get(0))
        .and_then(|b| b.get("text"))
        .and_then(|t| t.as_str())
        .map(str::to_string)
        .ok_or_else(|| Error::Config("missing content[0].text in anthropic response".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_or_expand_handles_typed_lines_directly() {
        let input =
            "lex: auth middleware\nvec: token verification\nhyde: Validates JSON web tokens.";
        let queries = parse_or_expand_query(input, None);
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0].kind, crate::search::QueryType::Lex);
        assert_eq!(queries[0].text, "auth middleware");
        assert_eq!(queries[1].kind, crate::search::QueryType::Vec);
        assert_eq!(queries[1].text, "token verification");
        assert_eq!(queries[2].kind, crate::search::QueryType::Hyde);
        assert_eq!(queries[2].text, "Validates JSON web tokens.");
    }

    #[test]
    fn parse_or_expand_handles_expand_prefix() {
        let input = "expand: sqlite database";
        let queries = parse_or_expand_query(input, None);
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0].text, "sqlite database");
    }

    #[test]
    fn expand_query_falls_back_to_simple_when_no_provider() {
        let queries = expand_query("rust ownership", None);
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0].kind, crate::search::QueryType::Lex);
        assert_eq!(queries[0].text, "rust ownership");
        assert_eq!(queries[1].kind, crate::search::QueryType::Vec);
        assert_eq!(queries[1].text, "rust ownership");
        assert_eq!(queries[2].kind, crate::search::QueryType::Hyde);
    }

    #[test]
    fn llm_config_defaults_to_none() {
        let config = LlmConfig::default();
        assert_eq!(config.provider, LlmProvider::None);
    }
}
