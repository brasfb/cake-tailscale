//! Ollama-compatible API endpoints.
//!
//! Implements the subset of the [ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)
//! that chat clients (Open WebUI, IDE plugins, the ollama CLI) actually use:
//! `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps` and
//! `/api/version`. Streaming responses use NDJSON (one JSON object per line)
//! and default to `stream: true`, matching ollama semantics — the opposite
//! of the OpenAI endpoints.
//!
//! Cake serves one model per process, so `/api/chat` accepts any `model`
//! value with a warning instead of a 404 when it doesn't match the loaded
//! model, and `/api/tags` lists the loaded model first followed by other
//! locally cached models.

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::cake::Master;
use crate::models::chat::Message;
use crate::models::Model;
use crate::utils::models as model_registry;

use super::text::{run_generation, spawn_generation};

// ── Requests ────────────────────────────────────────────────────────────────

/// Generation options accepted inside `options` (ollama Modelfile params).
/// Only the ones cake can honor are deserialized; the rest are ignored.
#[derive(Default, Deserialize)]
pub struct OllamaOptions {
    #[serde(default)]
    pub num_predict: Option<usize>,
}

#[derive(Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Deserialize)]
pub struct OllamaShowRequest {
    pub model: String,
}

// ── Responses ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

/// A `/api/chat` NDJSON object (intermediate or final).
#[derive(Serialize)]
struct OllamaChatChunk {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u128>,
}

/// A `/api/generate` NDJSON object (intermediate or final).
#[derive(Serialize)]
struct OllamaGenerateChunk {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u128>,
}

#[derive(Serialize)]
struct OllamaModelDetails {
    format: String,
    family: String,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Serialize)]
struct OllamaModelEntry {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: OllamaModelDetails,
}

#[derive(Serialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelEntry>,
}

// ── Time helpers (no chrono dependency) ─────────────────────────────────────

/// Format unix seconds as an RFC 3339 UTC timestamp ("2026-06-10T12:34:56Z").
/// Uses Howard Hinnant's civil-from-days algorithm.
pub(crate) fn rfc3339_utc(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs_of_day = unix_secs % 86_400;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

fn now_rfc3339() -> String {
    rfc3339_utc(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
}

fn mtime_rfc3339(path: &std::path::Path) -> String {
    let secs = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rfc3339_utc(secs)
}

// ── Model registry helpers ──────────────────────────────────────────────────

/// Pseudo-digest: cake models aren't content-addressed blobs like ollama's,
/// so the digest is a stable hash of the model name.
fn name_digest(name: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hex::encode(hasher.finalize())
}

/// Read `model_type` from a model directory's config.json ("llama", "qwen3", ...).
fn read_model_family(path: &std::path::Path) -> String {
    std::fs::read_to_string(path.join("config.json"))
        .ok()
        .and_then(|data| serde_json::from_str::<serde_json::Value>(&data).ok())
        .and_then(|cfg| cfg.get("model_type").and_then(|v| v.as_str()).map(String::from))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Rough parameter count from on-disk size assuming F16 weights.
fn parameter_size_label(size_bytes: u64) -> String {
    let params = size_bytes as f64 / 2.0;
    if params >= 1e9 {
        format!("{:.1}B", params / 1e9)
    } else {
        format!("{:.0}M", params / 1e6)
    }
}

fn model_entry(m: &model_registry::LocalModel) -> OllamaModelEntry {
    OllamaModelEntry {
        name: m.name.clone(),
        model: m.name.clone(),
        modified_at: mtime_rfc3339(&m.path),
        size: m.size_bytes,
        digest: name_digest(&m.name),
        details: OllamaModelDetails {
            format: "safetensors".to_string(),
            family: read_model_family(&m.path),
            parameter_size: parameter_size_label(m.size_bytes),
            quantization_level: "F16".to_string(),
        },
    }
}

/// Name the loaded model is served under: the model id/path the server was
/// started with (e.g. "evilsocket/Qwen3-0.6B"), falling back to the
/// architecture name when unset.
fn loaded_model_name<M: Model>(master: &Master<M>) -> String {
    let id = master.ctx.args.model.trim();
    if id.is_empty() {
        M::MODEL_NAME.to_string()
    } else {
        id.to_string()
    }
}

/// Warn when the client asked for a model other than the one loaded:
/// cake serves a single model per process.
fn warn_model_mismatch<M: Model>(requested: &Option<String>, loaded: &str) {
    if let Some(requested) = requested {
        if requested != loaded && requested != M::MODEL_NAME {
            log::warn!(
                "client requested model '{}' but '{}' is loaded — serving with the loaded model",
                requested,
                loaded
            );
        }
    }
}

// ── Endpoints ───────────────────────────────────────────────────────────────

/// GET /api/version
pub async fn version() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "version": concat!(env!("CARGO_PKG_VERSION"), "-cake"),
    }))
}

/// GET /api/tags — locally available models, loaded model first.
pub async fn tags<M: Model>(state: web::Data<Arc<RwLock<Master<M>>>>) -> impl Responder {
    let loaded = {
        let master = state.read().await;
        master.model.as_ref().map(|_| loaded_model_name(&master))
    };

    let cached = tokio::task::spawn_blocking(model_registry::list_models)
        .await
        .unwrap_or_else(|e| Err(anyhow::anyhow!("{e}")))
        .unwrap_or_default();

    let mut models: Vec<OllamaModelEntry> = Vec::new();
    if let Some(loaded_name) = &loaded {
        // The loaded model first; use registry info when it's cached locally.
        match cached.iter().find(|m| &m.name == loaded_name) {
            Some(m) => models.push(model_entry(m)),
            None => models.push(OllamaModelEntry {
                name: loaded_name.clone(),
                model: loaded_name.clone(),
                modified_at: now_rfc3339(),
                size: 0,
                digest: name_digest(loaded_name),
                details: OllamaModelDetails {
                    format: "safetensors".to_string(),
                    family: "unknown".to_string(),
                    parameter_size: "unknown".to_string(),
                    quantization_level: "F16".to_string(),
                },
            }),
        }
    }
    for m in &cached {
        if Some(&m.name) != loaded.as_ref() {
            models.push(model_entry(m));
        }
    }

    HttpResponse::Ok().json(OllamaTagsResponse { models })
}

/// GET /api/ps — the currently loaded model (cake loads exactly one).
pub async fn ps<M: Model>(state: web::Data<Arc<RwLock<Master<M>>>>) -> impl Responder {
    let master = state.read().await;
    let name = loaded_model_name(&master);
    let models: Vec<serde_json::Value> = master
        .model
        .as_ref()
        .map(|_| {
            vec![serde_json::json!({
                "name": name,
                "model": name,
                "digest": name_digest(&name),
                "expires_at": "never",
            })]
        })
        .unwrap_or_default();
    HttpResponse::Ok().json(serde_json::json!({ "models": models }))
}

/// POST /api/show — details for a locally cached model.
pub async fn show(body: web::Json<OllamaShowRequest>) -> impl Responder {
    let name = body.0.model;
    let found = tokio::task::spawn_blocking(move || model_registry::find_model(&name))
        .await
        .unwrap_or_else(|e| Err(anyhow::anyhow!("{e}")));

    match found {
        Ok(Some(m)) => {
            let model_info = std::fs::read_to_string(m.path.join("config.json"))
                .ok()
                .and_then(|data| serde_json::from_str::<serde_json::Value>(&data).ok())
                .unwrap_or(serde_json::Value::Null);
            let entry = model_entry(&m);
            HttpResponse::Ok().json(serde_json::json!({
                "details": entry.details,
                "model_info": model_info,
                "capabilities": ["completion"],
            }))
        }
        Ok(None) => HttpResponse::NotFound()
            .json(serde_json::json!({"error": "model not found"})),
        Err(e) => HttpResponse::InternalServerError()
            .json(serde_json::json!({"error": format!("{e}")})),
    }
}

/// POST /api/chat — chat completion, NDJSON streaming by default.
pub async fn chat<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    body: web::Json<OllamaChatRequest>,
) -> impl Responder {
    let request = body.0;
    let loaded = {
        let master = state.read().await;
        loaded_model_name(&master)
    };
    warn_model_mismatch::<M>(&request.model, &loaded);
    let max_tokens = request.options.unwrap_or_default().num_predict;

    if request.stream.unwrap_or(true) {
        chat_stream::<M>(state, loaded, request.messages, max_tokens)
    } else {
        chat_blocking::<M>(state, loaded, request.messages, max_tokens).await
    }
}

/// POST /api/generate — raw completion, NDJSON streaming by default.
/// The prompt is wrapped as a single user message (plus optional system
/// message) and runs through the same chat plumbing.
pub async fn generate<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    body: web::Json<OllamaGenerateRequest>,
) -> impl Responder {
    let request = body.0;
    let loaded = {
        let master = state.read().await;
        loaded_model_name(&master)
    };
    warn_model_mismatch::<M>(&request.model, &loaded);
    let max_tokens = request.options.unwrap_or_default().num_predict;

    let mut messages = Vec::new();
    if let Some(system) = request.system {
        messages.push(Message::system(system));
    }
    messages.push(Message::user(request.prompt));

    if request.stream.unwrap_or(true) {
        generate_stream::<M>(state, loaded, messages, max_tokens)
    } else {
        generate_blocking::<M>(state, loaded, messages, max_tokens).await
    }
}

// ── Chat implementations ────────────────────────────────────────────────────

async fn chat_blocking<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
) -> HttpResponse {
    let started = Instant::now();
    match run_generation(&state, messages, max_tokens).await {
        Ok(outcome) => HttpResponse::Ok().json(OllamaChatChunk {
            model,
            created_at: now_rfc3339(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: outcome.text,
            },
            done: true,
            done_reason: Some(outcome.finish_reason),
            total_duration: Some(started.elapsed().as_nanos()),
            eval_count: Some(outcome.completion_tokens),
            eval_duration: Some(started.elapsed().as_nanos()),
        }),
        Err(e) => e.to_response(),
    }
}

fn chat_stream<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
) -> HttpResponse {
    let mut rx = spawn_generation(state, messages, max_tokens);

    let stream = async_stream::stream! {
        let started = Instant::now();
        let mut eval_count = 0usize;

        while let Some(msg) = rx.recv().await {
            match msg {
                Some(content) => {
                    eval_count += 1;
                    let chunk = OllamaChatChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        message: OllamaMessage {
                            role: "assistant".to_string(),
                            content,
                        },
                        done: false,
                        done_reason: None,
                        total_duration: None,
                        eval_count: None,
                        eval_duration: None,
                    };
                    yield Ok::<_, actix_web::Error>(ndjson_line(&chunk));
                }
                None => {
                    let done = OllamaChatChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        message: OllamaMessage {
                            role: "assistant".to_string(),
                            content: String::new(),
                        },
                        done: true,
                        done_reason: Some("stop".to_string()),
                        total_duration: Some(started.elapsed().as_nanos()),
                        eval_count: Some(eval_count),
                        eval_duration: Some(started.elapsed().as_nanos()),
                    };
                    yield Ok(ndjson_line(&done));
                    break;
                }
            }
        }
    };

    HttpResponse::Ok()
        .content_type("application/x-ndjson")
        .insert_header(("Cache-Control", "no-cache"))
        .streaming(stream)
}

async fn generate_blocking<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
) -> HttpResponse {
    let started = Instant::now();
    match run_generation(&state, messages, max_tokens).await {
        Ok(outcome) => HttpResponse::Ok().json(OllamaGenerateChunk {
            model,
            created_at: now_rfc3339(),
            response: outcome.text,
            done: true,
            done_reason: Some(outcome.finish_reason),
            total_duration: Some(started.elapsed().as_nanos()),
            eval_count: Some(outcome.completion_tokens),
            eval_duration: Some(started.elapsed().as_nanos()),
        }),
        Err(e) => e.to_response(),
    }
}

fn generate_stream<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
) -> HttpResponse {
    let mut rx = spawn_generation(state, messages, max_tokens);

    let stream = async_stream::stream! {
        let started = Instant::now();
        let mut eval_count = 0usize;

        while let Some(msg) = rx.recv().await {
            match msg {
                Some(content) => {
                    eval_count += 1;
                    let chunk = OllamaGenerateChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        response: content,
                        done: false,
                        done_reason: None,
                        total_duration: None,
                        eval_count: None,
                        eval_duration: None,
                    };
                    yield Ok::<_, actix_web::Error>(ndjson_line(&chunk));
                }
                None => {
                    let done = OllamaGenerateChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        response: String::new(),
                        done: true,
                        done_reason: Some("stop".to_string()),
                        total_duration: Some(started.elapsed().as_nanos()),
                        eval_count: Some(eval_count),
                        eval_duration: Some(started.elapsed().as_nanos()),
                    };
                    yield Ok(ndjson_line(&done));
                    break;
                }
            }
        }
    };

    HttpResponse::Ok()
        .content_type("application/x-ndjson")
        .insert_header(("Cache-Control", "no-cache"))
        .streaming(stream)
}

/// Encode one NDJSON line: the serialized object followed by a newline.
fn ndjson_line<T: Serialize>(value: &T) -> web::Bytes {
    web::Bytes::from(format!("{}\n", serde_json::to_string(value).unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Request deserialization ─────────────────────────────────

    #[test]
    fn test_chat_request_minimal() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: OllamaChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert!(req.model.is_none());
        assert!(req.stream.is_none());
        assert!(req.options.is_none());
    }

    #[test]
    fn test_chat_request_with_options() {
        let json = r#"{
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "options": {"num_predict": 64, "temperature": 0.5}
        }"#;
        let req: OllamaChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("llama3.2"));
        assert_eq!(req.stream, Some(false));
        assert_eq!(req.options.unwrap().num_predict, Some(64));
    }

    #[test]
    fn test_chat_request_missing_messages_fails() {
        let json = r#"{"model": "x"}"#;
        assert!(serde_json::from_str::<OllamaChatRequest>(json).is_err());
    }

    #[test]
    fn test_generate_request_minimal() {
        let json = r#"{"prompt": "Why is the sky blue?"}"#;
        let req: OllamaGenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Why is the sky blue?");
        assert!(req.system.is_none());
    }

    #[test]
    fn test_generate_request_with_system() {
        let json = r#"{"prompt": "hi", "system": "be brief", "stream": true}"#;
        let req: OllamaGenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.as_deref(), Some("be brief"));
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn test_show_request() {
        let req: OllamaShowRequest =
            serde_json::from_str(r#"{"model": "evilsocket/Qwen3-0.6B"}"#).unwrap();
        assert_eq!(req.model, "evilsocket/Qwen3-0.6B");
    }

    // ── NDJSON chunk format ─────────────────────────────────────

    #[test]
    fn test_chat_chunk_intermediate_format() {
        let chunk = OllamaChatChunk {
            model: "m".into(),
            created_at: "2026-06-10T00:00:00Z".into(),
            message: OllamaMessage {
                role: "assistant".into(),
                content: "tok".into(),
            },
            done: false,
            done_reason: None,
            total_duration: None,
            eval_count: None,
            eval_duration: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["message"]["content"], "tok");
        assert_eq!(json["done"], false);
        // Stats must be omitted (not null) on intermediate chunks
        assert!(json.get("done_reason").is_none());
        assert!(json.get("eval_count").is_none());
        assert!(json.get("total_duration").is_none());
    }

    #[test]
    fn test_chat_chunk_terminal_format() {
        let chunk = OllamaChatChunk {
            model: "m".into(),
            created_at: "2026-06-10T00:00:00Z".into(),
            message: OllamaMessage {
                role: "assistant".into(),
                content: String::new(),
            },
            done: true,
            done_reason: Some("stop".into()),
            total_duration: Some(123),
            eval_count: Some(42),
            eval_duration: Some(120),
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["done"], true);
        assert_eq!(json["done_reason"], "stop");
        assert_eq!(json["eval_count"], 42);
    }

    #[test]
    fn test_generate_chunk_uses_response_field() {
        let chunk = OllamaGenerateChunk {
            model: "m".into(),
            created_at: "2026-06-10T00:00:00Z".into(),
            response: "tok".into(),
            done: false,
            done_reason: None,
            total_duration: None,
            eval_count: None,
            eval_duration: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["response"], "tok");
        assert!(json.get("message").is_none());
    }

    #[test]
    fn test_ndjson_line_format() {
        let chunk = OllamaChatChunk {
            model: "m".into(),
            created_at: "t".into(),
            message: OllamaMessage {
                role: "assistant".into(),
                content: "x".into(),
            },
            done: false,
            done_reason: None,
            total_duration: None,
            eval_count: None,
            eval_duration: None,
        };
        let line = ndjson_line(&chunk);
        let s = std::str::from_utf8(&line).unwrap();
        assert!(s.ends_with('\n'), "NDJSON lines must end with a newline");
        assert_eq!(s.matches('\n').count(), 1, "exactly one newline per object");
        let _: serde_json::Value = serde_json::from_str(s.trim()).unwrap();
    }

    // ── rfc3339_utc ─────────────────────────────────────────────

    #[test]
    fn test_rfc3339_epoch() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_rfc3339_known_value() {
        // 2026-06-10 12:34:56 UTC
        assert_eq!(rfc3339_utc(1_781_094_896), "2026-06-10T12:34:56Z");
    }

    #[test]
    fn test_rfc3339_leap_year() {
        // 2024-02-29 00:00:00 UTC
        assert_eq!(rfc3339_utc(1_709_164_800), "2024-02-29T00:00:00Z");
    }

    // ── Misc helpers ────────────────────────────────────────────

    #[test]
    fn test_name_digest_stable_and_hex() {
        let d1 = name_digest("org/model");
        let d2 = name_digest("org/model");
        assert_eq!(d1, d2);
        assert_eq!(d1.len(), 64);
        assert!(d1.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(name_digest("org/other"), d1);
    }

    #[test]
    fn test_parameter_size_label() {
        assert_eq!(parameter_size_label(2 * 1_000_000_000), "1.0B");
        assert_eq!(parameter_size_label(1_600_000_000), "800M");
        assert_eq!(parameter_size_label(700_000_000), "350M");
    }

    #[test]
    fn test_version_string_format() {
        let v = concat!(env!("CARGO_PKG_VERSION"), "-cake");
        assert!(v.ends_with("-cake"));
    }
}
