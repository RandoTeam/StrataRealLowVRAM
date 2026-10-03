import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
CONFIG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.json")
LOG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.log")
PYTHON_EXE = os.path.join(STRATA_DIR, r".venv\Scripts\python.exe")
SERVER_PY = os.path.join(STRATA_DIR, r"serve\server.py")
OPTIMIZER_PS1 = os.path.join(STRATA_DIR, r"tools\optimize_memory.ps1")
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"
RESULTS_FILE = os.path.join(STRATA_DIR, "bench", "coder_benchmark_results.json")

def kill_strata():
    subprocess.run(["powershell", "-NoProfile", "-Command", "Stop-Process -Name strata -Force -ErrorAction SilentlyContinue"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

def run_ram_optimizer():
    if os.path.exists(OPTIMIZER_PS1):
        try:
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", OPTIMIZER_PS1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(2)

def wait_for_server(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(HEALTH_URL)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def parse_latest_log():
    if not os.path.exists(LOG_FILE):
        return {}
    
    res = {}
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        for l in reversed(lines):
            if "decode expert cache hit rate:" in l and "hit_rate" not in res:
                try:
                    hit_str = l.split("decode expert cache hit rate:")[1].split("%")[0].strip()
                    res["hit_rate"] = float(hit_str)
                except Exception:
                    pass
            if "strata serve: prompt" in l and "generated in" in l and "engine_prefill_tok_s" not in res:
                parts = l.split(",")
                for sub in parts:
                    if "read in" in sub and "tok/s" in sub:
                        p_val = sub.split("(")[1].split("tok/s")[0].strip()
                        res["engine_prefill_tok_s"] = float(p_val)
                    if "generated in" in sub and "tok/s" in sub:
                        d_val = sub.split("(")[1].split("tok/s")[0].strip()
                        res["engine_decode_tok_s"] = float(d_val)
                    if "drafts accepted" in sub:
                        d_parts = sub.split("drafts accepted")[1].split(",")
                        acc_str = d_parts[0].strip()
                        if " of " in acc_str:
                            acc, total = acc_str.split(" of ")
                            if int(total) > 0:
                                res["mtp_acc"] = round(int(acc) / int(total) * 100, 1)
            if "expert cache" in l and "slots" in l and "expert_slots" not in res:
                try:
                    parts = l.split("expert cache")[1].split("slots")[0].strip()
                    res["expert_slots"] = int(parts)
                except Exception:
                    pass
            if len(res) >= 4:
                break
    except Exception as e:
        res["parse_error"] = str(e)
    return res

def run_prompt_streaming(prompt_def, timeout_s=400):
    payload = {
        "model": "qwen3.8-flash-next-coder-iq1_m",
        "messages": [
            {"role": "system", "content": prompt_def.get("system", "You are an expert programming assistant.")},
            {"role": "user", "content": prompt_def["user"]}
        ],
        "temperature": 0.6,
        "max_tokens": prompt_def["max_tokens"],
        "reasoning_budget_tokens": 80,
        "stream": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER_URL, data=data, headers={"Content-Type": "application/json"})

    start_t = time.time()
    reasoning_chunks = []
    content_chunks = []
    first_token_t = None
    reasoning_end_t = None
    finish_reason = None
    usage = {}
    timings = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                
                now = time.time()
                if first_token_t is None:
                    first_token_t = now

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                r_text = delta.get("reasoning_content")
                c_text = delta.get("content")

                if r_text:
                    reasoning_chunks.append((now, r_text))
                    print(r_text, end="", flush=True)
                if c_text:
                    if reasoning_end_t is None and reasoning_chunks:
                        reasoning_end_t = now
                        print("\n[END OF THINKING]\n", flush=True)
                    content_chunks.append((now, c_text))
                    print(c_text, end="", flush=True)

                fr = chunk.get("choices", [{}])[0].get("finish_reason")
                if fr:
                    finish_reason = fr
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                if "timings" in chunk and chunk["timings"]:
                    timings = chunk["timings"]

        end_t = time.time()
        elapsed = end_t - start_t
        ttft = (first_token_t - start_t) if first_token_t else elapsed

        full_reasoning = "".join([t[1] for t in reasoning_chunks])
        full_content = "".join([t[1] for t in content_chunks])

        total_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        reasoning_chars = len(full_reasoning)
        content_chars = len(full_content)

        if total_tokens > 0 and (reasoning_chars + content_chars) > 0:
            r_tokens = round(total_tokens * (reasoning_chars / (reasoning_chars + content_chars)))
            c_tokens = total_tokens - r_tokens
        else:
            r_tokens = max(1, round(reasoning_chars / 3.8)) if reasoning_chars else 0
            c_tokens = max(1, round(content_chars / 3.8)) if content_chars else 0
            total_tokens = r_tokens + c_tokens

        r_time = (reasoning_end_t - first_token_t) if (reasoning_end_t and first_token_t) else (
            (end_t - first_token_t) if (not content_chunks and first_token_t) else 0.0
        )
        c_time = (end_t - reasoning_end_t) if (reasoning_end_t and end_t > reasoning_end_t) else (
            (end_t - first_token_t) if (not reasoning_chunks and first_token_t) else 0.0
        )

        r_tok_s = round(r_tokens / r_time, 2) if r_time > 0 and r_tokens > 0 else 0
        c_tok_s = round(c_tokens / c_time, 2) if c_time > 0 and c_tokens > 0 else 0
        total_tok_s = round(total_tokens / (end_t - first_token_t), 2) if (first_token_t and end_t > first_token_t) else 0

        # Log stats
        log_stats = parse_latest_log()

        return {
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            "ttft_s": round(ttft, 2),
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "total_tok_s": total_tok_s,
            "reasoning_tokens": r_tokens,
            "reasoning_time_s": round(r_time, 2),
            "reasoning_tok_s": r_tok_s,
            "content_tokens": c_tokens,
            "content_time_s": round(c_time, 2),
            "content_tok_s": c_tok_s,
            "finish_reason": finish_reason,
            "timings": timings,
            "engine_prefill_tok_s": timings.get("prompt_per_second") or log_stats.get("engine_prefill_tok_s"),
            "engine_decode_tok_s": timings.get("predicted_per_second") or log_stats.get("engine_decode_tok_s"),
            "hit_rate_pct": log_stats.get("hit_rate"),
            "mtp_acc_pct": log_stats.get("mtp_acc"),
            "expert_slots": log_stats.get("expert_slots"),
            "reasoning_full": full_reasoning,
            "content_full": full_content,
            "content_length": len(full_content)
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed_s": round(time.time() - start_t, 2)}

def main():
    print("=" * 75)
    print(" STRATA 125B CODER BENCHMARK: DECODE (THINKING VS ANSWER) & PREFILL")
    print("=" * 75)

    kill_strata()
    run_ram_optimizer()

    # Clear log
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    print("[1/4] Launching Strata Server with real Strata C++ Engine...")
    cmd = [PYTHON_EXE, SERVER_PY, "--engine", "strata", "--config", CONFIG_FILE, "--port", "8080"]
    env = os.environ.copy()
    env["PATH"] = r"C:\Python312\Lib\site-packages\nvidia\cu13\bin\x86_64;C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\engine;" + env.get("PATH", "")
    env["STRATA_ARENA_LOCK"] = "0"
    
    server_proc = subprocess.Popen(
        cmd,
        cwd=STRATA_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env
    )

    try:
        print("[2/4] Awaiting server readiness on http://127.0.0.1:8080/health ...")
        if not wait_for_server(180):
            print("[ERROR] Server failed to become ready within timeout!")
            return

        print("[SUCCESS] Server is online and ready!\n")

        # ---------------------------------------------------------
        # Benchmark 1: Standard Coding Tests (Decode & Thinking)
        # ---------------------------------------------------------
        test_prompts = [
            {
                "id": "Test A",
                "name": "Short Logic & Concurrency",
                "system": "You are an expert systems programmer. Think step by step.",
                "user": "Explain the difference between a mutex, a spinlock, and a counting semaphore in concurrent programming in 3 short bullet points. Provide a one-line modern C++ example for each.",
                "max_tokens": 256
            },
            {
                "id": "Test B",
                "name": "Medium Data Structures (LRU Cache)",
                "system": "You are a senior algorithms engineer. Think step by step.",
                "user": "Implement a high-performance LRU Cache in Python using a doubly linked list and a dictionary without OrderedDict. Include get and put methods, type annotations, and docstrings.",
                "max_tokens": 384
            },
            {
                "id": "Test C",
                "name": "Sustained Architecture (Concurrent Worker Pool)",
                "system": "You are a principal software architect. Think step by step.",
                "user": "Design and implement a robust thread-safe Worker Pool in Python using threading, queue, and dataclasses with priority tasks, graceful shutdown, and timeout handling.",
                "max_tokens": 512
            }
        ]

        results = {"decode_tests": {}, "prefill_tests": {}}

        print("=" * 75)
        print(" PHASE 1: DECODE BENCHMARKS (MEASURING THINKING VS ANSWER TOK/S)")
        print("=" * 75)

        for p in test_prompts:
            print(f"\n=======================================================================")
            print(f"--> Executing {p['id']}: {p['name']} (max_tokens={p['max_tokens']})")
            print(f"=======================================================================")
            res = run_prompt_streaming(p)
            results["decode_tests"][p["id"]] = res

            if res["ok"]:
                print(f"\n--- METRICS SUMMARY FOR {p['id']} ---")
                print(f"    Total Generated: {res['total_tokens']} tokens in {res['elapsed_s']}s (Overall: {res['total_tok_s']} tok/s)")
                print(f"    Thinking Stream: {res['reasoning_tokens']} tok in {res['reasoning_time_s']}s => {res['reasoning_tok_s']} tok/s")
                print(f"    Answer Stream:   {res['content_tokens']} tok in {res['content_time_s']}s => {res['content_tok_s']} tok/s")
                print(f"    Engine Hardware: Prefill {res.get('engine_prefill_tok_s')} tok/s | Decode {res.get('engine_decode_tok_s')} tok/s")
                print(f"    VRAM Cache Hit:  {res.get('hit_rate_pct')}% | MTP Acceptance: {res.get('mtp_acc_pct')}%")
                print(f"    Resident Slots:  {res.get('expert_slots')}")
            else:
                print(f"\n[ERROR] FAILED: {res.get('error')}")

            time.sleep(3)

        # ---------------------------------------------------------
        # Benchmark 2: Prefill Throughput on Long Texts
        # ---------------------------------------------------------
        print("\n" + "=" * 75)
        print(" PHASE 2: PREFILL THROUGHPUT (PROMPTS UP TO 8,192 TOKENS)")
        print("=" * 75)

        sample_code = """
// Distributed consensus Raft state machine node implementation
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

enum class NodeRole { Follower, Candidate, Leader };

struct LogEntry {
    uint64_t term;
    uint64_t index;
    std::string command;
};

class RaftNode {
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    NodeRole role_{NodeRole::Follower};
    uint64_t current_term_{0};
    int voted_for_{-1};
    std::vector<LogEntry> log_;
    uint64_t commit_index_{0};
    uint64_t last_applied_{0};
    std::atomic<bool> running_{true};
public:
    RaftNode() {
        log_.push_back({0, 0, "NOOP"});
    }
    void start_election() {
        std::unique_lock<std::mutex> lock(mtx_);
        role_ = NodeRole::Candidate;
        current_term_++;
        voted_for_ = 0; // self
    }
    void append_entries(uint64_t term, int leader_id, uint64_t prev_index, uint64_t prev_term, const std::vector<LogEntry>& entries) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (term < current_term_) return;
        if (term > current_term_) {
            current_term_ = term;
            role_ = NodeRole::Follower;
            voted_for_ = -1;
        }
        if (log_.size() <= prev_index || log_[prev_index].term != prev_term) return;
        log_.erase(log_.begin() + prev_index + 1, log_.end());
        log_.insert(log_.end(), entries.begin(), entries.end());
    }
};
"""
        prefill_tests = [
            ("1024_tokens", 4),
            ("2048_tokens", 8),
            ("4096_tokens", 16),
            ("8192_tokens", 32)
        ]

        for label, mult in prefill_tests:
            long_context = sample_code * mult
            prompt_p = {
                "id": f"Prefill_{label}",
                "name": f"Codebase Context ({label})",
                "system": "You are a senior concurrency reviewer.",
                "user": f"Review this consensus codebase. State whether it correctly prevents split-brain elections:\n\n{long_context}",
                "max_tokens": 48
            }
            print(f"\n=======================================================================")
            print(f"--> Testing Prefill Throughput on {label}...")
            print(f"=======================================================================")
            res = run_prompt_streaming(prompt_p, timeout_s=400)
            results["prefill_tests"][label] = res

            if res["ok"]:
                print(f"\n--- PREFILL METRICS FOR {label} ---")
                print(f"    Prompt Tokens:   {res['prompt_tokens']}")
                print(f"    TTFT (Latency):  {res['ttft_s']}s")
                print(f"    Prefill Rate:    {res.get('engine_prefill_tok_s')} tok/s")
                print(f"    Decode Rate:     {res['total_tok_s']} tok/s (Tokens: {res['total_tokens']})")
                print(f"    Cache Hit Rate:  {res.get('hit_rate_pct')}%")
            else:
                print(f"\n[ERROR] FAILED: {res.get('error')}")

            time.sleep(3)

        # Save complete results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] All benchmarks completed and saved to {RESULTS_FILE}")

    finally:
        print("\n[4/4] Shutting down Strata Server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        kill_strata()
        print("Engine cleanly terminated.")

if __name__ == "__main__":
    main()
