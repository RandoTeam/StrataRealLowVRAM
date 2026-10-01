use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::protocol::{EngineCommand, EngineResponse};

pub struct EngineConfig {
    pub exe_path: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub env: Vec<(String, String)>,
}

pub struct StrataEngine {
    stdin: Mutex<ChildStdin>,
    max_context: u64,
    can_stop: bool,
    response_rx: Mutex<mpsc::Receiver<EngineResponse>>,
    fifo: Mutex<()>,
}

impl StrataEngine {
    pub async fn start(config: EngineConfig) -> Result<Arc<Self>, String> {
        info!("Spawning Strata engine: {:?} with {} args", config.exe_path, config.args.len());

        let mut cmd = Command::new(&config.exe_path);
        cmd.arg("--serve");
        for arg in &config.args {
            cmd.arg(arg);
        }
        cmd.current_dir(&config.cwd);
        for (k, v) in &config.env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit());

        let mut child: Child = cmd.spawn().map_err(|e| format!("Failed to spawn engine: {}", e))?;

        let stdin = child.stdin.take().ok_or("Failed to acquire child stdin")?;
        let stdout = child.stdout.take().ok_or("Failed to acquire child stdout")?;

        let (tx, rx) = mpsc::channel::<EngineResponse>(1024);

        // Background reader task for stdout
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                let resp = EngineResponse::parse(&line);
                if tx.send(resp).await.is_err() {
                    break;
                }
            }
            warn!("Engine stdout reader task terminated");
        });

        let mut rx = rx;
        let mut max_context = 0;
        let mut can_stop = false;

        info!("Awaiting READY handshake from Strata engine...");
        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Ready { max_context: ctx, can_stop: cs } => {
                    max_context = ctx;
                    can_stop = cs;
                    info!("Received READY: max_context={}, can_stop={}", max_context, can_stop);
                    break;
                }
                EngineResponse::Info(info_map) => {
                    info!("Engine INFO: {:?}", info_map);
                }
                EngineResponse::Err(err) => {
                    return Err(format!("Engine startup error: {}", err));
                }
                _ => {}
            }
        }

        if max_context == 0 {
            return Err("Engine exited before reaching READY state".to_string());
        }

        Ok(Arc::new(Self {
            stdin: Mutex::new(stdin),
            max_context,
            can_stop,
            response_rx: Mutex::new(rx),
            fifo: Mutex::new(()),
        }))
    }

    pub fn max_context(&self) -> u64 {
        self.max_context
    }

    pub async fn execute_command(
        &self,
        cmd: EngineCommand,
        out_tx: mpsc::Sender<EngineResponse>,
    ) -> Result<(), String> {
        let _guard = self.fifo.lock().await;

        let line = cmd.to_line();
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await.map_err(|e| format!("Failed to write to engine: {}", e))?;
            stdin.flush().await.map_err(|e| format!("Failed to flush engine stdin: {}", e))?;
        }

        let mut rx = self.response_rx.lock().await;
        while let Some(resp) = rx.recv().await {
            let is_terminal = matches!(&resp, EngineResponse::Done(_) | EngineResponse::Err(_));
            if out_tx.send(resp).await.is_err() {
                // Client cancelled or disconnected
                if self.can_stop {
                    info!("Client disconnected, sending STOP to engine");
                    let mut stdin = self.stdin.lock().await;
                    let _ = stdin.write_all(b"STOP\n").await;
                    let _ = stdin.flush().await;
                }
                // Drain until terminal response
                while let Some(drain_resp) = rx.recv().await {
                    if matches!(drain_resp, EngineResponse::Done(_) | EngineResponse::Err(_)) {
                        break;
                    }
                }
                return Ok(());
            }
            if is_terminal {
                break;
            }
        }

        Ok(())
    }
}
