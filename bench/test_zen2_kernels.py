import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
CONFIG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.json")
LOG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.log")
PYTHON_EXE = os.path.join(STRATA_DIR, r".venv\Scripts\python.exe")
SERVER_PY = os.path.join(STRATA_DIR, r"serve\server.py")
OPTIMIZER_PS1 = os.path.join(STRATA_DIR, r"tools\optimize_memory.ps1")
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"

TEST_PROMPTS = [
    {
        "id": "A",
        "name": "Short Logic & Concurrency (Test A)",
        "system": "You are a concise, precise AI assistant.",
        "user": "Explain the difference between a mutex and a semaphore in concurrent programming in 3 short bullet points. Provide a one-line C++ code example for each.",
        "max_tokens": 192,
        "thinking": "low"
    },
    {
        "id": "B",
        "name": "Medium Data Structures (Test B)",
        "system": "You are an expert systems engineer.",
        "user": "Implement a circular ring buffer in C++ using a fixed-size array and atomic head/tail indices for lock-free single-producer single-consumer operation. Provide clean, well-commented code.",
        "max_tokens": 256,
        "thinking": "low"
    }
]

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

def wait_for_server(timeout=120):
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

def run_prompt(prompt_def, timeout_s=180):
    payload = {
        "model": "qwen3.8-flash-next-coder-iq1_m",
        "messages": [
            {"role": "system", "content": prompt_def["system"]},
            {"role": "user", "content": prompt_def["user"]}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": prompt_def["max_tokens"],
        "stream": True
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER_URL, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    first_token_time = None
    chunks = 0
    token_chars = 0
    reasoning_tokens = 0
    completion_tokens = 0
    
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.time() - t0
                    chunks += 1
                    try:
                        chunk_json = json.loads(line_str[6:])
                        choices = chunk_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                token_chars += len(delta["content"])
                                completion_tokens += 1
                            if delta.get("reasoning_content"):
                                reasoning_tokens += 1
                    except Exception:
                        pass
        total_time = time.time() - t0
        total_tokens = completion_tokens + reasoning_tokens
        gen_time = total_time - (first_token_time if first_token_time else 0)
        tok_s = round(total_tokens / gen_time, 2) if gen_time > 0 and total_tokens > 0 else 0.0
        return {
            "ok": True,
            "ttft_s": round(first_token_time, 3) if first_token_time else None,
            "total_time_s": round(total_time, 2),
            "decode_tok_s": tok_s,
            "completion_tokens": total_tokens
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def parse_last_log_stats():
    if not os.path.exists(LOG_FILE):
        return None, None, None, None
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        prefill_tok_s = None
        decode_tok_s = None
        hit_rate = None
        mtp_acc = None
        
        for l in reversed(lines):
            if (prefill_tok_s is None or decode_tok_s is None) and "strata serve: prompt" in l and "generated in" in l:
                parts = l.split(",")
                for sub in parts:
                    if "read in" in sub and "tok/s" in sub:
                        p_val = sub.split("(")[1].split("tok/s")[0].strip()
                        prefill_tok_s = float(p_val)
                    if "generated in" in sub and "tok/s" in sub:
                        d_val = sub.split("(")[1].split("tok/s")[0].strip()
                        decode_tok_s = float(d_val)
                    if "drafts accepted" in sub:
                        d_parts = sub.split("drafts accepted")[1].split(",")
                        acc_str = d_parts[0].strip()
                        if " of " in acc_str:
                            acc, total = acc_str.split(" of ")
                            if int(total) > 0:
                                mtp_acc = round(int(acc) / int(total) * 100, 1)

            if hit_rate is None and "decode expert cache hit rate:" in l:
                try:
                    hit_str = l.split("decode expert cache hit rate:")[1].split("%")[0].strip()
                    hit_rate = float(hit_str)
                except Exception:
                    pass
                    
        return prefill_tok_s, decode_tok_s, hit_rate, mtp_acc
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None, None, None, None

def evaluate_candidate(cand_name, env_overrides):
    print(f"\n==========================================", flush=True)
    print(f"EVALUATING CANDIDATE: {cand_name}", flush=True)
    print(f"ENV: {env_overrides}", flush=True)
    print(f"==========================================", flush=True)
    
    kill_strata()
    run_ram_optimizer()
    
    # Load base config
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Apply env overrides
    base_env = cfg.get("env", {})
    new_env = dict(base_env)
    new_env.update(env_overrides)
    cfg["env"] = new_env
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
        
    print("  -> Starting server with candidate environment...", flush=True)
    cmd = [PYTHON_EXE, SERVER_PY, "--engine", "strata", "--config", CONFIG_FILE, "--port", "8080"]
    srv_proc = subprocess.Popen(cmd, cwd=STRATA_DIR, creationflags=subprocess.HIGH_PRIORITY_CLASS)
    
    if not wait_for_server(timeout=120):
        print("  [ERROR] Server failed to start within timeout!", flush=True)
        kill_strata()
        return None
        
    print("  -> Server ONLINE! Running test battery...", flush=True)
    
    # Warm-up / Test A
    print("     [1/2] Test A (Short Concurrency, 192 tokens)...", flush=True)
    p1 = run_prompt(TEST_PROMPTS[0], timeout_s=120)
    print(f"           Result: ok={p1['ok']}, speed={p1.get('decode_tok_s', 0)} tok/s, TTFT={p1.get('ttft_s')}s", flush=True)
    time.sleep(2)
    
    # Test B
    print("     [2/2] Test B (Medium Ring Buffer, 256 tokens)...", flush=True)
    p2 = run_prompt(TEST_PROMPTS[1], timeout_s=150)
    print(f"           Result: ok={p2['ok']}, speed={p2.get('decode_tok_s', 0)} tok/s, TTFT={p2.get('ttft_s')}s", flush=True)
    
    prefill_s, decode_s, hit_rate, mtp_acc = parse_last_log_stats()
    kill_strata()
    
    speeds = [p["decode_tok_s"] for p in [p1, p2] if p.get("ok")]
    avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else 0.0
    
    res = {
        "candidate": cand_name,
        "env": env_overrides,
        "test_a_tok_s": p1.get("decode_tok_s", 0.0),
        "test_b_tok_s": p2.get("decode_tok_s", 0.0),
        "avg_client_decode_tok_s": avg_speed,
        "engine_decode_tok_s": decode_s if decode_s is not None else avg_speed,
        "engine_prefill_tok_s": prefill_s if prefill_s is not None else 0.0,
        "hit_rate_pct": hit_rate if hit_rate is not None else 0.0,
        "mtp_acc_pct": mtp_acc if mtp_acc is not None else 0.0,
        "ttft_a": p1.get("ttft_s"),
        "ttft_b": p2.get("ttft_s")
    }
    
    print(f"  -> SUMMARY for {cand_name}:")
    print(f"     Engine Decode: {res['engine_decode_tok_s']} tok/s | Client Avg: {res['avg_client_decode_tok_s']} tok/s")
    print(f"     Prefill: {res['engine_prefill_tok_s']} tok/s | Hit Rate: {res['hit_rate_pct']}% | MTP Acc: {res['mtp_acc_pct']}%")
    return res

def main():
    # Save original config
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        orig_cfg = json.load(f)
        
    candidates = [
        ("Baseline Champion", {}),
        ("Candidate A (MT_MIN=1)", {"STRATA_IQ_MT_MIN": "1"}),
        ("Candidate B (MT_MIN=1 + PREFETCH=1024)", {"STRATA_IQ_MT_MIN": "1", "STRATA_IQ_PREFETCH": "1024"}),
        ("Candidate C (MT_MIN=1 + PREFETCH=512)", {"STRATA_IQ_MT_MIN": "1", "STRATA_IQ_PREFETCH": "512"}),
    ]
    
    all_results = []
    try:
        for name, env_diff in candidates:
            r = evaluate_candidate(name, env_diff)
            if r:
                all_results.append(r)
            print("  Cooldown 20 seconds...", flush=True)
            time.sleep(20)
    finally:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(orig_cfg, f, indent=1)
        kill_strata()
        
    out_file = os.path.join(STRATA_DIR, r"bench\zen2_kernel_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll tests completed! Saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
