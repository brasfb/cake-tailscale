use crate::cake::Master;
use crate::models::chat::Message;
use crate::models::{Model, OutputModality};
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Serialize)]
struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn new(model: String, message: String, prompt_tokens: usize, completion_tokens: usize, finish_reason: String) -> Self {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let object = String::from("chat.completion");
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let choices = vec![Choice {
            index: 0,
            message: Message::assistant(message),
            finish_reason,
        }];

        Self {
            id,
            object,
            created,
            model,
            choices,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

// SSE streaming types
#[derive(Serialize)]
struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

pub async fn generate_text<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    req: HttpRequest,
    body: web::Json<ChatRequest>,
) -> impl Responder {
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let stream = body.0.stream.unwrap_or(false);

    log::info!("starting chat for {} (stream={}) ...", &client, stream);

    if stream {
        generate_text_stream(state, body.0).await
    } else {
        generate_text_blocking(state, body.0).await
    }
}

/// Why a blocking generation run failed.
pub(crate) enum GenerationError {
    NoModel,
    Internal(String),
}

impl GenerationError {
    pub(crate) fn to_response(&self) -> HttpResponse {
        match self {
            GenerationError::NoModel => HttpResponse::NotFound()
                .json(serde_json::json!({"error": "No text model loaded"})),
            GenerationError::Internal(e) => {
                HttpResponse::InternalServerError().json(serde_json::json!({"error": e}))
            }
        }
    }
}

/// Result of a completed blocking generation run.
pub(crate) struct GenerationOutcome {
    pub text: String,
    pub finish_reason: String,
    pub completion_tokens: usize,
}

/// Run a full generation to completion: reset the master, feed `messages`,
/// generate up to `max_tokens`, and disconnect. Shared by the OpenAI and
/// ollama non-streaming endpoints.
pub(crate) async fn run_generation<M: Model>(
    state: &web::Data<Arc<RwLock<Master<M>>>>,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
) -> Result<GenerationOutcome, GenerationError> {
    let mut master = state.write().await;

    if !master.model.as_ref().is_some_and(|m| m.output_modality() == OutputModality::Text) {
        return Err(GenerationError::NoModel);
    }

    master
        .reset()
        .map_err(|e| GenerationError::Internal(format!("{e}")))?;

    let model = master.model.as_mut().unwrap();
    for message in messages {
        model
            .add_message(message)
            .map_err(|e| GenerationError::Internal(format!("{e}")))?;
    }

    let mut text = String::new();
    let mut finish_reason = "length".to_string();

    let gen_result = master
        .generate_text(max_tokens, |data| {
            if data.is_empty() {
                finish_reason = "stop".to_string();
            } else {
                text += data;
            }
        })
        .await;

    println!();

    let completion_tokens = master
        .model
        .as_ref()
        .map(|m| m.generated_tokens())
        .unwrap_or(0);

    let _ = master.goodbye().await;

    gen_result.map_err(|e| GenerationError::Internal(format!("{e}")))?;

    Ok(GenerationOutcome {
        text,
        finish_reason,
        completion_tokens,
    })
}

/// Spawn a generation task feeding tokens into an unbounded channel.
/// `Some(token)` per generated piece, `None` exactly once at the end
/// (whether by stop token, token budget, or error). Shared by the OpenAI
/// SSE and ollama NDJSON streaming endpoints.
pub(crate) fn spawn_generation<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    messages: Vec<Message>,
    max_tokens: Option<usize>,
) -> tokio::sync::mpsc::UnboundedReceiver<Option<String>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Option<String>>();

    tokio::spawn(async move {
        let mut master = state.write().await;

        if let Err(e) = master.reset() {
            log::error!("reset error: {e}");
            let _ = tx.send(None);
            return;
        }

        let model = match master.model.as_mut() {
            Some(m) if m.output_modality() == OutputModality::Text => m,
            _ => {
                log::error!("no text model loaded");
                let _ = tx.send(None);
                return;
            }
        };
        for message in messages {
            if let Err(e) = model.add_message(message) {
                log::error!("add_message error: {e}");
                let _ = tx.send(None);
                return;
            }
        }

        if let Err(e) = master
            .generate_text(max_tokens, |data| {
                if data.is_empty() {
                    let _ = tx.send(None);
                } else {
                    let _ = tx.send(Some(data.to_string()));
                }
            })
            .await
        {
            log::error!("generate_text error: {e}");
            let _ = tx.send(None);
        }

        let _ = master.goodbye().await;
    });

    rx
}

async fn generate_text_blocking<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    request: ChatRequest,
) -> HttpResponse {
    let num_messages = request.messages.len();

    match run_generation(&state, request.messages, request.max_tokens).await {
        Ok(outcome) => {
            let response = ChatResponse::new(
                M::MODEL_NAME.to_string(),
                outcome.text,
                num_messages,
                outcome.completion_tokens,
                outcome.finish_reason,
            );
            HttpResponse::Ok().json(response)
        }
        Err(e) => e.to_response(),
    }
}

async fn generate_text_stream<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    request: ChatRequest,
) -> HttpResponse {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let model = M::MODEL_NAME.to_string();

    let mut rx = spawn_generation(state.clone(), request.messages, request.max_tokens);

    let stream = async_stream::stream! {
        // Send initial role chunk
        let initial = StreamResponse {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok::<_, actix_web::Error>(
            web::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&initial).unwrap()))
        );

        // Stream content chunks
        while let Some(msg) = rx.recv().await {
            match msg {
                Some(content) => {
                    let chunk = StreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta {
                                role: None,
                                content: Some(content),
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok(web::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())));
                }
                None => {
                    // Final chunk with finish_reason
                    let done = StreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    yield Ok(web::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&done).unwrap())));
                    yield Ok(web::Bytes::from("data: [DONE]\n\n"));
                    break;
                }
            }
        }
    };

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatRequest deserialization ──────────────────────────────

    #[test]
    fn test_chat_request_full() {
        let json = r#"{
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "model": "test-model",
            "stream": true,
            "max_tokens": 100,
            "temperature": 0.7
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.model.as_deref(), Some("test-model"));
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.max_tokens, Some(100));
        assert!((req.temperature.unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_chat_request_minimal() {
        // Only messages required; all other fields default to None
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert!(req.model.is_none());
        assert!(req.stream.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
    }

    #[test]
    fn test_chat_request_empty_messages() {
        let json = r#"{"messages": []}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
    }

    #[test]
    fn test_chat_request_stream_false() {
        let json = r#"{"messages": [{"role": "user", "content": "test"}], "stream": false}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn test_chat_request_missing_messages_fails() {
        let json = r#"{"model": "test"}"#;
        let result = serde_json::from_str::<ChatRequest>(json);
        assert!(result.is_err(), "missing messages field should fail");
    }

    // ── ChatResponse serialization ──────────────────────────────

    #[test]
    fn test_chat_response_new_structure() {
        let resp = ChatResponse::new(
            "gpt-test".into(),
            "Hello there!".into(),
            5,
            10,
            "stop".into(),
        );
        assert!(resp.id.starts_with("chatcmpl-"));
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.model, "gpt-test");
        assert!(resp.created > 0);
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.completion_tokens, 10);
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[test]
    fn test_chat_response_serializes_to_json() {
        let resp = ChatResponse::new(
            "model-x".into(),
            "answer".into(),
            3,
            7,
            "length".into(),
        );
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "model-x");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "answer");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(json["usage"]["prompt_tokens"], 3);
        assert_eq!(json["usage"]["completion_tokens"], 7);
        assert_eq!(json["usage"]["total_tokens"], 10);
    }

    #[test]
    fn test_chat_response_unique_ids() {
        let r1 = ChatResponse::new("m".into(), "a".into(), 0, 0, "stop".into());
        let r2 = ChatResponse::new("m".into(), "a".into(), 0, 0, "stop".into());
        assert_ne!(r1.id, r2.id, "each response should get a unique UUID-based id");
    }

    #[test]
    fn test_chat_response_zero_tokens() {
        let resp = ChatResponse::new("m".into(), "".into(), 0, 0, "stop".into());
        assert_eq!(resp.usage.total_tokens, 0);
        assert_eq!(resp.choices[0].message.content, "");
    }

    // ── Usage serialization ─────────────────────────────────────

    #[test]
    fn test_usage_serialization() {
        let usage = Usage {
            prompt_tokens: 42,
            completion_tokens: 58,
            total_tokens: 100,
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["prompt_tokens"], 42);
        assert_eq!(json["completion_tokens"], 58);
        assert_eq!(json["total_tokens"], 100);
    }

    // ── StreamDelta serialization ───────────────────────────────

    #[test]
    fn test_stream_delta_skips_none_fields() {
        let delta = StreamDelta {
            role: None,
            content: Some("hello".into()),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("role").is_none(), "None role should be skipped");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn test_stream_delta_both_none() {
        let delta = StreamDelta {
            role: None,
            content: None,
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("role").is_none());
        assert!(json.get("content").is_none());
    }

    #[test]
    fn test_stream_delta_role_only() {
        let delta = StreamDelta {
            role: Some("assistant".into()),
            content: None,
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert_eq!(json["role"], "assistant");
        assert!(json.get("content").is_none());
    }

    // ── StreamResponse serialization ────────────────────────────

    #[test]
    fn test_stream_response_chunk_format() {
        let resp = StreamResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk".into(),
            created: 1700000000,
            model: "test-model".into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: Some("word".into()),
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "word");
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn test_stream_response_final_chunk() {
        let resp = StreamResponse {
            id: "chatcmpl-done".into(),
            object: "chat.completion.chunk".into(),
            created: 1700000000,
            model: "m".into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".into()),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    // ── SSE format ──────────────────────────────────────────────

    #[test]
    fn test_sse_data_line_format() {
        // Verify the "data: {json}\n\n" format used in streaming
        let resp = StreamResponse {
            id: "id".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "m".into(),
            choices: vec![],
        };
        let line = format!("data: {}\n\n", serde_json::to_string(&resp).unwrap());
        assert!(line.starts_with("data: {"));
        assert!(line.ends_with("}\n\n"));
        // The JSON inside should be parseable
        let inner = line.strip_prefix("data: ").unwrap().trim();
        let _: serde_json::Value = serde_json::from_str(inner).unwrap();
    }
}
