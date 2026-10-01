import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
CONFIG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.json")
BASELINE_FILE = os.path.join(STRATA_DIR, "bench", "strata-coder-iq1_m.baseline.json")
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
        "max_tokens": 256,
        "thinking": "low"
    },
    {
        "id": "B",
        "name": "Medium Data Structures & LRU (Test B)",
        "system": "You are an expert Python engineer.",
        "user": "Implement an LRU (Least Recently Used) cache in Python without using collections.OrderedDict. Use a doubly linked list and dict for O(1) ops. Provide full LRUCache class with get and put, plus docstrings.",
        "max_tokens": 384,
        "thinking": "low"
    },
    {
        "id": "C",
        "name": "Sustained Architecture & Worker Pool (Test C)",
        "system": "You are a principal software systems architect.",
        "user": "Implement a thread-safe asynchronous Task Worker Pool in Python using threading, queue, and dataclasses. Include task priority, worker idle timeouts, graceful shutdown with worker draining, and exception aggregation. Provide full implementation with docstrings and usage example.",
        "max_tokens": 512,
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

def wait_for_server(timeout=110):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(HEALTH_URL)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def run_prompt(prompt_data, timeout_s=180):
    body = {
        "model": "qwen3.8-flash-next-coder-iq1_m",
        "messages": [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["user"]}
        ],
        "max_tokens": prompt_data["max_tokens"],
        "reasoning_effort": prompt_data["thinking"],
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(SERVER_URL, data=json.dumps(body).encode("utf-8"), headers=headers)
    
    start_t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            elapsed = time.time() - start_t
            res_data = json.loads(resp.read().decode("utf-8"))
            choice_msg = res_data.get("choices", [{}])[0].get("message", {})
            content = choice_msg.get("content") or choice_msg.get("reasoning_content") or ""
            usage = res_data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tok_s = round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens > 0 else 0
            return {
                "ok": True,
                "elapsed_s": round(elapsed, 2),
                "completion_tokens": completion_tokens,
                "decode_tok_s": tok_s,
                "content_preview": content[:150] if content else "",
                "has_content": bool(content and len(content.strip()) > 30)
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed_s": round(time.time() - start_t, 2), "decode_tok_s": 0, "has_content": False}

def parse_last_log_stats():
    """
    Parses engine metrics from strata-coder-iq1_m.log.
    """
    if not os.path.exists(LOG_FILE):
        return None, None, None, None, None, None, None, None
        
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        prefill_tok_s = None
        decode_tok_s = None
        hit_rate = None
        mtp_acc = None
        expert_slots = None
        vram_free_mib = None
        kv_streaming_hit = None
        ram_read_mib = None
        
        for l in reversed(lines):
            # Parse expert slots
            if expert_slots is None and "expert cache" in l and "slots" in l:
                try:
                    parts = l.split("expert cache")[1].split("slots")[0].strip()
                    expert_slots = int(parts)
                except Exception:
                    pass
                    
            # Parse VRAM free
            if vram_free_mib is None and "MiB of VRAM free with everything loaded" in l:
                try:
                    parts = l.split("MiB of VRAM free with everything loaded")[0].split()[-1]
                    vram_free_mib = int(parts)
                except Exception:
                    pass

            # Parse prompt / decode speed
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

            # Parse hit rate
            if hit_rate is None and "decode expert cache hit rate:" in l:
                hr_part = l.split("hit rate:")[1].split("%")[0].strip()
                hit_rate = float(hr_part)
                
            # Parse KV streaming telemetry
            if kv_streaming_hit is None and "KV streaming:" in l and "block reads hit VRAM" in l:
                try:
                    hit_part = l.split("KV streaming:")[1].split("%")[0].strip()
                    kv_streaming_hit = float(hit_part)
                    if "read from RAM" in l:
                        ram_part = l.split("read from RAM")[0].split()[-2].strip()
                        ram_read_mib = float(ram_part)
                except Exception:
                    pass
                
        return prefill_tok_s, decode_tok_s, hit_rate, mtp_acc, expert_slots, vram_free_mib, kv_streaming_hit, ram_read_mib
    except Exception:
        return None, None, None, None, None, None, None, None

def build_config_from_args(args_list, env_dict=None, config_path=CONFIG_FILE):
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["args"] = args_list
    if env_dict:
        cfg["env"] = env_dict
    elif "env" in cfg:
        del cfg["env"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

def evaluate_tournament4_candidate(candidate_id, candidate_name, candidate_category, args_list, env_dict=None):
    print(f"\n==================================================================", flush=True)
    print(f"  TOURNAMENT 4.0: EVALUATING CANDIDATE {candidate_id}: {candidate_name}", flush=True)
    print(f"  Category: {candidate_category}", flush=True)
    print(f"  Args count: {len(args_list)}", flush=True)
    if env_dict:
        print(f"  Custom Env: {env_dict}", flush=True)
    print(f"==================================================================", flush=True)
    
    kill_strata()
    time.sleep(3)
    build_config_from_args(args_list, env_dict)
    run_ram_optimizer()
    time.sleep(4) # Extended thermal and memory stabilization
    
    # Start Strata server
    cmd = [PYTHON_EXE, SERVER_PY, "--engine", "strata", "--config", CONFIG_FILE, "--port", "8080"]
    sub_env = os.environ.copy()
    if env_dict:
        for k, v in env_dict.items():
            sub_env[k] = str(v)
            
    proc = subprocess.Popen(cmd, cwd=STRATA_DIR, env=sub_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("  -> Booting Strata server and validating VRAM allocations (timeout 110s)...", flush=True)
    ready = wait_for_server(timeout=110)
    if not ready:
        print("  -> [DISQUALIFIED] Server failed to initialize within timeout or crashed on VRAM alloc.", flush=True)
        kill_strata()
        return {
            "id": candidate_id,
            "name": candidate_name,
            "category": candidate_category,
            "status": "DISQUALIFIED (Startup Crash / OOM)",
            "score": 0.0,
            "decode_tok_s": 0.0,
            "prefill_tok_s": 0.0,
            "hit_rate_pct": 0.0,
            "mtp_acc_pct": 0.0,
            "expert_slots": 0,
            "vram_free_mib": 0
        }
    
    print("  -> Server ONLINE! Executing 3-Tier Tournament 4 Benchmark...", flush=True)
    
    # Test A: Short Logic
    print("     [1/3] Executing Test A (Short Logic & Concurrency, 256 tokens)...", flush=True)
    p1 = run_prompt(TEST_PROMPTS[0], timeout_s=120)
    print(f"           Test A Result: ok={p1['ok']}, speed={p1.get('decode_tok_s', 0)} tok/s, tokens={p1.get('completion_tokens', 0)}", flush=True)
    time.sleep(2)
    
    # Test B: Medium Algorithm
    print("     [2/3] Executing Test B (Medium Algorithm & LRU Cache, 384 tokens)...", flush=True)
    p2 = run_prompt(TEST_PROMPTS[1], timeout_s=180)
    print(f"           Test B Result: ok={p2['ok']}, speed={p2.get('decode_tok_s', 0)} tok/s, tokens={p2.get('completion_tokens', 0)}", flush=True)
    time.sleep(2)
    
    # Test C: Sustained Code Architecture
    print("     [3/3] Executing Test C (Sustained Worker Pool & Reasoning, 512 tokens)...", flush=True)
    p3 = run_prompt(TEST_PROMPTS[2], timeout_s=220)
    print(f"           Test C Result: ok={p3['ok']}, speed={p3.get('decode_tok_s', 0)} tok/s, tokens={p3.get('completion_tokens', 0)}", flush=True)
    
    prefill_s, decode_s, hit_rate, mtp_acc, expert_slots, vram_free_mib, kv_hit, ram_read = parse_last_log_stats()
    kill_strata()
    
    p_metric = prefill_s if prefill_s is not None else p1.get("decode_tok_s", 0.0)
    d_metric = decode_s if decode_s is not None else p2.get("decode_tok_s", 0.0)
    hr_metric = hit_rate if hit_rate is not None else 0.0
    mtp_metric = mtp_acc if mtp_acc is not None else 0.0
    slots_metric = expert_slots if expert_slots is not None else 0
    free_metric = vram_free_mib if vram_free_mib is not None else 0
    kv_hit_metric = kv_hit if kv_hit is not None else 0.0
    ram_read_metric = ram_read if ram_read is not None else 0.0
    
    p1_pass = p1.get("ok", False) and p1.get("has_content", False)
    p2_pass = p2.get("ok", False) and p2.get("has_content", False)
    p3_pass = p3.get("ok", False) and p3.get("has_content", False)
    
    passed_count = sum([p1_pass, p2_pass, p3_pass])
    if passed_count == 3:
        quality_bonus = 10.0
        status = "PASSED (Full 3/3)"
    elif passed_count == 2:
        quality_bonus = 6.0
        status = "PARTIAL (2/3)"
    elif passed_count == 1:
        quality_bonus = 3.0
        status = "PARTIAL (1/3)"
    else:
        quality_bonus = 0.0
        status = "FAILED (0/3)"
        
    # Tournament 4 Focus Formula (heavily weighted towards Decode tok/s to drive 8.0+ tok/s goal):
    # Score = (Decode * 10.0) + (Prefill * 0.5) + (Hit Rate * 1.5) + (MTP Acc * 0.1) + Quality Bonus
    total_score = round(
        (d_metric * 10.0) +
        (p_metric * 0.5) +
        (hr_metric * 1.5) +
        (mtp_metric * 0.1) +
        quality_bonus,
        2
    )
    
    result = {
        "id": candidate_id,
        "name": candidate_name,
        "category": candidate_category,
        "status": status,
        "decode_tok_s": d_metric,
        "prefill_tok_s": p_metric,
        "hit_rate_pct": hr_metric,
        "mtp_acc_pct": mtp_metric,
        "expert_slots": slots_metric,
        "vram_free_mib": free_metric,
        "kv_streaming_hit_pct": kv_hit_metric,
        "ram_read_mib": ram_read_metric,
        "quality_bonus": quality_bonus,
        "score": total_score,
        "test_a": p1,
        "test_b": p2,
        "test_c": p3
    }
    
    print(f"\n  -> TOURNAMENT 4 SCORE: {total_score} | Decode: {d_metric} tok/s | Prefill: {p_metric} tok/s | Hit Rate: {hr_metric}% | MTP: {mtp_metric}% | Slots: {slots_metric} | Quality: +{quality_bonus}", flush=True)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner_t4.py <candidate_json_file>")
        sys.exit(1)
        
    cand_file = sys.argv[1]
    with open(cand_file, "r", encoding="utf-8") as f:
        cand_data = json.load(f)
        
    res = evaluate_tournament4_candidate(
        cand_data["id"],
        cand_data["name"],
        cand_data.get("category", "General"),
        cand_data["args"],
        cand_data.get("env", None)
    )
    
    out_file = cand_file.replace(".json", "_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {out_file}", flush=True)
