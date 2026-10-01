use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub penalty_repeat: Option<f32>,
    pub penalty_freq: Option<f32>,
    pub penalty_present: Option<f32>,
    pub penalty_last_n: Option<i32>,
    pub seed: Option<u64>,
    pub pcie_frac: Option<f32>,
    pub spec_min_p: Option<f32>,
    pub cvec: Option<i32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.05),
            penalty_repeat: Some(1.05),
            penalty_freq: None,
            penalty_present: Some(0.2),
            penalty_last_n: Some(256),
            seed: None,
            pcie_frac: None,
            spec_min_p: None,
            cvec: None,
        }
    }
}

impl SamplingParams {
    pub fn to_inline_keys(&self) -> String {
        let mut parts = Vec::new();
        if let Some(t) = self.temperature {
            if t > 0.0 { parts.push(format!("temperature={:.4}", t)); }
        }
        if let Some(p) = self.top_p {
            if p > 0.0 && p < 1.0 { parts.push(format!("top_p={:.4}", p)); }
        }
        if let Some(k) = self.top_k {
            if k > 0 { parts.push(format!("top_k={}", k.min(64))); }
        }
        if let Some(mp) = self.min_p {
            if mp > 0.0 && mp <= 1.0 { parts.push(format!("min_p={:.4}", mp)); }
        }
        if let Some(pr) = self.penalty_repeat {
            if (pr - 1.0).abs() > 1e-4 { parts.push(format!("penalty_repeat={:.4}", pr)); }
        }
        if let Some(pf) = self.penalty_freq {
            if pf.abs() > 1e-4 { parts.push(format!("penalty_freq={:.4}", pf)); }
        }
        if let Some(pp) = self.penalty_present {
            if pp.abs() > 1e-4 { parts.push(format!("penalty_present={:.4}", pp)); }
        }
        if let Some(pln) = self.penalty_last_n {
            if pln > 0 { parts.push(format!("penalty_last_n={}", pln)); }
        }
        if let Some(s) = self.seed {
            parts.push(format!("seed={}", s));
        }
        if let Some(pf) = self.pcie_frac {
            parts.push(format!("pcie_frac={:.4}", pf));
        }
        if let Some(smp) = self.spec_min_p {
            parts.push(format!("spec_min_p={:.4}", smp));
        }
        if let Some(c) = self.cvec {
            parts.push(format!("cvec={}", c));
        }
        parts.join(" ")
    }
}

#[derive(Debug, Clone)]
pub enum EngineCommand {
    Gen {
        max_new: u32,
        sampling: SamplingParams,
        token_ids: Vec<i32>,
    },
    GenI {
        max_new: u32,
        sampling: SamplingParams,
        embeddings_path: String,
        token_ids: Vec<i32>,
    },
    Stop,
    Quit,
}

impl EngineCommand {
    pub fn to_line(&self) -> String {
        match self {
            Self::Gen { max_new, sampling, token_ids } => {
                let keys = sampling.to_inline_keys();
                let ids: Vec<String> = token_ids.iter().map(|id| id.to_string()).collect();
                if keys.is_empty() {
                    format!("GEN {} {}\n", max_new, ids.join(","))
                } else {
                    format!("GEN {} {} {}\n", max_new, keys, ids.join(","))
                }
            }
            Self::GenI { max_new, sampling, embeddings_path, token_ids } => {
                let keys = sampling.to_inline_keys();
                let ids: Vec<String> = token_ids.iter().map(|id| id.to_string()).collect();
                if keys.is_empty() {
                    format!("GENI {} {} {}\n", max_new, embeddings_path, ids.join(","))
                } else {
                    format!("GENI {} {} {} {}\n", max_new, keys, embeddings_path, ids.join(","))
                }
            }
            Self::Stop => "STOP\n".to_string(),
            Self::Quit => "QUIT\n".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoneStats {
    pub generated: i64,
    pub prompt_tokens: i64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub finish_reason: String,
    pub drafts_accepted: i64,
    pub drafts_offered: i64,
    pub reused: i64,
    pub hits: Option<i64>,
    pub lookups: Option<i64>,
}

#[derive(Debug, Clone)]
pub enum EngineResponse {
    Info(HashMap<String, String>),
    Ready { max_context: u64, can_stop: bool },
    Resume(u64),
    PromptProgress { done: u64, total: u64, ms: f64, tok_s: f64 },
    Reused(u64),
    Token(i32),
    Done(DoneStats),
    Err(String),
    Other(String),
}

impl EngineResponse {
    pub fn parse(line: &str) -> Self {
        let s = line.trim();
        if s.starts_with("T ") {
            if let Ok(id) = s[2..].trim().parse::<i32>() {
                return Self::Token(id);
            }
        } else if s.starts_with("READY") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            let max_context = parts.get(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            let can_stop = parts.iter().any(|&x| x == "stop");
            return Self::Ready { max_context, can_stop };
        } else if s.starts_with("DONE") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() >= 6 {
                let stats = DoneStats {
                    generated: parts.get(1).and_then(|x| x.parse().ok()).unwrap_or(0),
                    prompt_tokens: parts.get(2).and_then(|x| x.parse().ok()).unwrap_or(0),
                    prompt_ms: parts.get(3).and_then(|x| x.parse().ok()).unwrap_or(0.0),
                    decode_ms: parts.get(4).and_then(|x| x.parse().ok()).unwrap_or(0.0),
                    finish_reason: parts.get(5).unwrap_or(&"stop").to_string(),
                    drafts_accepted: parts.get(6).and_then(|x| x.parse().ok()).unwrap_or(0),
                    drafts_offered: parts.get(7).and_then(|x| x.parse().ok()).unwrap_or(0),
                    reused: parts.get(8).and_then(|x| x.parse().ok()).unwrap_or(0),
                    hits: parts.get(9).and_then(|x| x.parse().ok()),
                    lookups: parts.get(10).and_then(|x| x.parse().ok()),
                };
                return Self::Done(stats);
            }
        } else if s.starts_with("ERR ") {
            return Self::Err(s[4..].to_string());
        } else if s.starts_with("INFO ") {
            let mut map = HashMap::new();
            for kv in s[5..].split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    map.insert(k.to_string(), v.to_string());
                }
            }
            return Self::Info(map);
        } else if s.starts_with("PP ") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() >= 5 {
                let done = parts[1].parse().unwrap_or(0);
                let total = parts[2].parse().unwrap_or(0);
                let ms = parts[3].parse().unwrap_or(0.0);
                let tok_s = parts[4].parse().unwrap_or(0.0);
                return Self::PromptProgress { done, total, ms, tok_s };
            }
        } else if s.starts_with("RESUME ") {
            if let Ok(n) = s[7..].trim().parse::<u64>() {
                return Self::Resume(n);
            }
        } else if s.starts_with("REUSED ") {
            if let Ok(n) = s[7..].trim().parse::<u64>() {
                return Self::Reused(n);
            }
        }
        Self::Other(s.to_string())
    }
}
