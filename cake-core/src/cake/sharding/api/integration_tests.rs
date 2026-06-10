//! API integration tests using actix_web::test with mocked models.
//!
//! All tests run offline — no GPU, no model files, no network servers.

use actix_web::test;

use super::test_helpers;

// ═══════════════════════════════════════════════════════════════════
// Chat endpoint tests (/v1/chat/completions)
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_chat_completions_blocking() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/v1/chat/completions")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    // MockTextGenerator emits "Hello" + " world"
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

#[actix_web::test]
async fn test_chat_completions_streaming() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/v1/chat/completions")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    let body = test::read_body(resp).await;
    let body_str = std::str::from_utf8(&body).unwrap();

    // Should contain SSE data lines
    assert!(body_str.contains("data: "));
    // Should have [DONE] sentinel
    assert!(body_str.contains("data: [DONE]"));
    // Should contain role chunk
    assert!(body_str.contains("\"role\":\"assistant\""));
}

#[actix_web::test]
async fn test_chat_completions_minimal_request() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/v1/chat/completions")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "test"}]
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}

#[actix_web::test]
async fn test_chat_completions_no_model_loaded() {
    let app = test::init_service(test_helpers::test_app_none()).await;
    let req = test::TestRequest::post()
        .uri("/v1/chat/completions")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["error"].as_str().unwrap().contains("No text model"));
}

#[actix_web::test]
async fn test_chat_completions_backwards_compat() {
    // Both /v1/chat/completions and /api/v1/chat/completions should work
    let app = test::init_service(test_helpers::test_app_text()).await;

    let req1 = test::TestRequest::post()
        .uri("/v1/chat/completions")
        .set_json(serde_json::json!({"messages": [{"role": "user", "content": "a"}]}))
        .to_request();
    let resp1 = test::call_service(&app, req1).await;
    assert!(resp1.status().is_success());

    let req2 = test::TestRequest::post()
        .uri("/api/v1/chat/completions")
        .set_json(serde_json::json!({"messages": [{"role": "user", "content": "b"}]}))
        .to_request();
    let resp2 = test::call_service(&app, req2).await;
    assert!(resp2.status().is_success());
}

// ═══════════════════════════════════════════════════════════════════
// Audio endpoint tests (/v1/audio/speech)
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_audio_speech_basic() {
    let app = test::init_service(test_helpers::test_app_audio()).await;
    let req = test::TestRequest::post()
        .uri("/v1/audio/speech")
        .set_json(serde_json::json!({
            "input": "Hello world"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "audio/wav"
    );

    let body = test::read_body(resp).await;
    // Valid RIFF header
    assert_eq!(&body[0..4], b"RIFF");
    assert_eq!(&body[8..12], b"WAVE");
}

#[actix_web::test]
async fn test_audio_speech_pcm_format() {
    let app = test::init_service(test_helpers::test_app_audio()).await;
    let req = test::TestRequest::post()
        .uri("/v1/audio/speech")
        .set_json(serde_json::json!({
            "input": "Test",
            "response_format": "pcm"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "audio/pcm"
    );

    let body = test::read_body(resp).await;
    // PCM f32 = 2400 samples * 4 bytes = 9600
    assert_eq!(body.len(), 2400 * 4);
}

#[actix_web::test]
async fn test_audio_speech_no_model_loaded() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/v1/audio/speech")
        .set_json(serde_json::json!({"input": "hello"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["error"].as_str().unwrap().contains("No audio model"));
}

// ═══════════════════════════════════════════════════════════════════
// Image endpoint tests (/v1/images/generations)
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_images_generations_raw_png() {
    let app = test::init_service(test_helpers::test_app_image()).await;
    let req = test::TestRequest::post()
        .uri("/v1/images/generations")
        .set_json(serde_json::json!({
            "prompt": "A rusty robot"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "image/png"
    );

    let body = test::read_body(resp).await;
    // PNG magic bytes
    assert_eq!(&body[0..4], &[0x89, 0x50, 0x4E, 0x47]);
}

#[actix_web::test]
async fn test_images_generations_b64_json() {
    let app = test::init_service(test_helpers::test_app_image()).await;
    let req = test::TestRequest::post()
        .uri("/v1/images/generations")
        .set_json(serde_json::json!({
            "prompt": "A rusty robot",
            "response_format": "b64_json"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["created"].as_u64().unwrap() > 0);
    let data = body["data"].as_array().unwrap();
    assert!(!data.is_empty());
    assert!(data[0]["b64_json"].as_str().unwrap().len() > 10);
}

#[actix_web::test]
async fn test_images_generations_no_model_loaded() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/v1/images/generations")
        .set_json(serde_json::json!({"prompt": "test"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["error"].as_str().unwrap().contains("No image model"));
}

// ═══════════════════════════════════════════════════════════════════
// Legacy image endpoint (/api/v1/image)
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_legacy_image_no_model_loaded() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/v1/image")
        .set_json(serde_json::json!({
            "image_args": {
                "prompt": "test"
            }
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

// ═══════════════════════════════════════════════════════════════════
// Models endpoint (/v1/models)
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_list_models_text() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::get().uri("/v1/models").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "mock-text");
    assert_eq!(data[0]["object"], "model");
}

#[actix_web::test]
async fn test_list_models_audio() {
    let app = test::init_service(test_helpers::test_app_audio()).await;
    let req = test::TestRequest::get().uri("/v1/models").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "mock-audio");
}

#[actix_web::test]
async fn test_list_models_none() {
    let app = test::init_service(test_helpers::test_app_none()).await;
    let req = test::TestRequest::get().uri("/v1/models").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["data"].as_array().unwrap().len(), 0);
}

#[actix_web::test]
async fn test_list_models_format() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::get().uri("/v1/models").to_request();
    let resp = test::call_service(&app, req).await;
    let body: serde_json::Value = test::read_body_json(resp).await;
    // OpenAI-compatible structure
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "model");
    assert!(body["data"][0]["owned_by"].as_str().is_some());
}

// ═══════════════════════════════════════════════════════════════════
// Cross-cutting
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_all_endpoints_404_when_no_models() {
    let app = test::init_service(test_helpers::test_app_none()).await;

    // Chat
    let req = test::TestRequest::post()
        .uri("/v1/chat/completions")
        .set_json(serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    // Audio
    let req = test::TestRequest::post()
        .uri("/v1/audio/speech")
        .set_json(serde_json::json!({"input": "hi"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    // Image (OpenAI)
    let req = test::TestRequest::post()
        .uri("/v1/images/generations")
        .set_json(serde_json::json!({"prompt": "hi"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);

    // Image (legacy)
    let req = test::TestRequest::post()
        .uri("/api/v1/image")
        .set_json(serde_json::json!({"image_args": {"prompt": "hi"}}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

// ═══════════════════════════════════════════════════════════════════
// Ollama-compatible endpoint tests (/api/chat, /api/generate, ...)
// ═══════════════════════════════════════════════════════════════════

#[actix_web::test]
async fn test_ollama_version() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::get().uri("/api/version").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["version"].as_str().unwrap().ends_with("-cake"));
}

#[actix_web::test]
async fn test_ollama_tags_lists_loaded_model_first() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::get().uri("/api/tags").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty());
    // MockTextGenerator is the loaded model — must be listed first
    assert_eq!(models[0]["name"], "mock-text");
    assert_eq!(models[0]["model"], "mock-text");
    assert_eq!(models[0]["digest"].as_str().unwrap().len(), 64);
    assert!(models[0]["modified_at"].as_str().unwrap().ends_with('Z'));
}

#[actix_web::test]
async fn test_ollama_ps_shows_loaded_model() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::get().uri("/api/ps").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "mock-text");
}

#[actix_web::test]
async fn test_ollama_ps_empty_when_no_model() {
    let app = test::init_service(test_helpers::test_app_none()).await;
    let req = test::TestRequest::get().uri("/api/ps").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["models"].as_array().unwrap().is_empty());
}

#[actix_web::test]
async fn test_ollama_show_unknown_model_404() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/show")
        .set_json(serde_json::json!({"model": "definitely/not-a-cached-model"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn test_ollama_chat_blocking() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/chat")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], "Hello world");
    assert_eq!(body["done"], true);
    assert!(body["done_reason"].is_string());
    assert!(body["eval_count"].as_u64().unwrap() > 0);
    assert!(body["total_duration"].as_u64().is_some());
}

#[actix_web::test]
async fn test_ollama_chat_streams_ndjson_by_default() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/chat")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers().get("content-type").unwrap().to_str().unwrap(),
        "application/x-ndjson"
    );

    let body = test::read_body(resp).await;
    let body_str = std::str::from_utf8(&body).unwrap();

    // Every line must be a standalone JSON object
    let lines: Vec<&str> = body_str.lines().filter(|l| !l.is_empty()).collect();
    assert!(lines.len() >= 2, "expected at least one token + terminal object");
    for line in &lines {
        let _: serde_json::Value = serde_json::from_str(line).unwrap();
    }

    // All but the last have done:false; the last has done:true + stats
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["done"], true);
    assert_eq!(last["done_reason"], "stop");
    assert!(last["eval_count"].as_u64().unwrap() > 0);
    for line in &lines[..lines.len() - 1] {
        let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(chunk["done"], false);
        assert!(chunk["message"]["content"].is_string());
    }
}

#[actix_web::test]
async fn test_ollama_chat_ignores_mismatched_model_name() {
    // Cake serves one model per process: mismatched names warn, not 404
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/chat")
        .set_json(serde_json::json!({
            "model": "some-other-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}

#[actix_web::test]
async fn test_ollama_chat_no_model_404() {
    let app = test::init_service(test_helpers::test_app_none()).await;
    let req = test::TestRequest::post()
        .uri("/api/chat")
        .set_json(serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn test_ollama_generate_blocking() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/generate")
        .set_json(serde_json::json!({
            "prompt": "Why is the sky blue?",
            "stream": false
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["response"], "Hello world");
    assert_eq!(body["done"], true);
}

#[actix_web::test]
async fn test_ollama_generate_streams_ndjson() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/generate")
        .set_json(serde_json::json!({"prompt": "Hi"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers().get("content-type").unwrap().to_str().unwrap(),
        "application/x-ndjson"
    );

    let body = test::read_body(resp).await;
    let body_str = std::str::from_utf8(&body).unwrap();
    let lines: Vec<&str> = body_str.lines().filter(|l| !l.is_empty()).collect();
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["done"], true);
    // generate uses "response", not "message"
    for line in &lines {
        let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(chunk["response"].is_string());
        assert!(chunk.get("message").is_none());
    }
}

#[actix_web::test]
async fn test_ollama_generate_with_options_num_predict() {
    let app = test::init_service(test_helpers::test_app_text()).await;
    let req = test::TestRequest::post()
        .uri("/api/generate")
        .set_json(serde_json::json!({
            "prompt": "Hi",
            "stream": false,
            "options": {"num_predict": 1}
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}
