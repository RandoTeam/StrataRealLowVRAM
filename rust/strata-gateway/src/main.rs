mod engine;
mod openai;
mod protocol;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use serde::Deserialize;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;
use uuid::Uuid;

use crate::engine::{EngineConfig, StrataEngine};
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChoiceDelta, ChunkChoice, ModelCard,
    ModelListResponse,
};
use crate::protocol::{EngineCommand, EngineResponse, SamplingParams};

#[derive(Parser, Debug)]
#[command(author, version, about = "High-performance Rust gateway for Strata inference engine")]
struct Cli {
    #[arg(short, long, default_value = "strata-coder-iq1_m.json")]
    config: PathBuf,

    #[arg(short, long, default_value = "8080")]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[derive(Debug, Deserialize)]
struct StrataJsonConfig {
    exe: String,
    args: Vec<String>,
    cwd: Option<String>,
    env: Option<std::collections::HashMap<String, String>>,
    model_name: Option<String>,
}

struct AppState {
    engine: Arc<StrataEngine>,
    model_name: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();
    info!("Loading Strata config from {:?}", cli.config);

    let config_content = std::fs::read_to_string(&cli.config)
        .map_err(|e| format!("Failed to read config file {:?}: {}", cli.config, e))?;
    let json_cfg: StrataJsonConfig = serde_json::from_str(&config_content)
        .map_err(|e| format!("Failed to parse config JSON: {}", e))?;

    let base_dir = cli.config.parent().unwrap_or(Path::new("."));
    let exe_path = PathBuf::from(&json_cfg.exe);
    let cwd = json_cfg.cwd.map(PathBuf::from).unwrap_or_else(|| base_dir.to_path_buf());

    let mut env_pairs = Vec::new();
    if let Some(env_map) = json_cfg.env {
        for (k, v) in env_map {
            env_pairs.push((k, v));
        }
    }

    let engine_cfg = EngineConfig {
        exe_path,
        args: json_cfg.args,
        cwd,
        env: env_pairs,
    };

    let engine = StrataEngine::start(engine_cfg).await?;
    let model_name = json_cfg.model_name.unwrap_or_else(|| "qwen3.8-flash-next-coder-iq1_m".to_string());

    let state = Arc::new(AppState {
        engine,
        model_name,
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    info!("Strata Rust Gateway listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "max_context": state.engine.max_context(),
        "model": state.model_name,
        "engine": "strata-gateway-rust"
    }))
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "strata".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let max_new = req.max_tokens.unwrap_or(2048);
    let sampling = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        ..Default::default()
    };

    // Synthesize simple prompt token IDs placeholder or tokenized IDs
    // For raw token input or template rendering
    let token_ids: Vec<i32> = vec![151644, 872, 198]; // standard Qwen header placeholder

    let (tx, mut rx) = mpsc::channel::<EngineResponse>(128);
    let cmd = EngineCommand::Gen {
        max_new,
        sampling,
        token_ids,
    };

    let engine = state.engine.clone();
    tokio::spawn(async move {
        if let Err(e) = engine.execute_command(cmd, tx).await {
            error!("Engine execution failed: {}", e);
        }
    });

    let req_id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = state.model_name.clone();

    if req.stream {
        let stream = async_stream::stream! {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            while let Some(resp) = rx.recv().await {
                match resp {
                    EngineResponse::Token(token_id) => {
                        let chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: now,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChoiceDelta {
                                    role: None,
                                    content: Some(format!(" {}", token_id)), // Token ID decoding
                                    reasoning_content: None,
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        yield Ok::<Event, std::convert::Infallible>(Event::default().data(data));
                    }
                    EngineResponse::Done(stats) => {
                        let chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: now,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChoiceDelta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                },
                                finish_reason: Some(stats.finish_reason),
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        yield Ok::<Event, std::convert::Infallible>(Event::default().data(data));
                        yield Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"));
                        break;
                    }
                    _ => {}
                }
            }
        };

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let mut tokens = Vec::new();
        let mut finish = "stop".to_string();
        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token(id) => tokens.push(id),
                EngineResponse::Done(stats) => {
                    finish = stats.finish_reason;
                    break;
                }
                _ => {}
            }
        }

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Json(serde_json::json!({
            "id": req_id,
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": format!("Tokens: {:?}", tokens)
                },
                "finish_reason": finish
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens.len(),
                "total_tokens": tokens.len()
            }
        })).into_response()
    }
}
