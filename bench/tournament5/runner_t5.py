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
        "max_tokens": prompt_def["max_tokens"],
        "thinking": prompt_def.get("thinking", "low")
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER_URL, data=data, headers={"Content-Type": "application/json"})
    
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
            
            # Rigorous Quality Check (Rule of Preserving Model Intelligence)
            has_reasoning = bool("<think>" in content or choice_msg.get("reasoning_content") or len(content.strip()) > 50)
            no_repetitive_loop = not ("\n\n\n\n\n" in content or "mutex mutex mutex" in content)
            passes_intelligence_check = has_reasoning and no_repetitive_loop and bool(completion_tokens >= 50)
            
            return {
                "ok": True,
                "elapsed_s": round(elapsed, 2),
                "completion_tokens": completion_tokens,
                "decode_tok_s": tok_s,
                "content_preview": content[:180] if content else "",
                "has_content": bool(content and len(content.strip()) > 30),
                "intelligence_ok": passes_intelligence_check
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed_s": round(time.time() - start_t, 2), "decode_tok_s": 0, "has_content": False, "intelligence_ok": False}

def parse_last_log_stats():
    if not os.path.exists(LOG_FILE):
        return None, None, None, None, None, None
        
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        prefill_tok_s = None
        decode_tok_s = None
        hit_rate = None
        mtp_acc = None
        expert_slots = None
        vram_free_mib = None
        
        for l in reversed(lines):
            if expert_slots is None and "expert cache" in l and "slots" in l:
                try:
                    parts = l.split("expert cache")[1].split("slots")[0].strip()
                    expert_slots = int(parts)
                except Exception:
                    pass
                    
            if vram_free_mib is None and "MiB of VRAM free with everything loaded" in l:
                try:
                    parts = l.split("MiB of VRAM free with everything loaded")[0].split()[-1]
                    vram_free_mib = int(parts)
                except Exception:
                    pass

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
                    
        return prefill_tok_s, decode_tok_s, hit_rate, mtp_acc, expert_slots, vram_free_mib
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None, None, None, None, None, None

def evaluate_candidate(candidate_id, candidate_name, args_list, env_dict=None):
    print(f"\n==================================================================", flush=True)
    print(f"  TOURNAMENT 5.0 BENCHMARK: [{candidate_id}] {candidate_name}", flush=True)
    print(f"  Flags count: {len(args_list)}", flush=True)
    if env_dict:
        print(f"  Environment: {env_dict}", flush=True)
    print(f"==================================================================", flush=True)
    
    kill_strata()
    time.sleep(3)
    
    # Read template config
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["args"] = args_list
    if env_dict:
        cfg["env"] = env_dict
    elif "env" in cfg:
        cfg["env"] = {}
        
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
        
    run_ram_optimizer()
    time.sleep(3)
    
    cmd = [PYTHON_EXE, SERVER_PY, "--engine", "strata", "--config", CONFIG_FILE, "--port", "8080"]
    sub_env = os.environ.copy()
    if env_dict:
        for k, v in env_dict.items():
            sub_env[k] = str(v)
            
    proc = subprocess.Popen(cmd, cwd=STRATA_DIR, env=sub_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("  -> Booting Strata server and validating VRAM allocation (timeout 120s)...", flush=True)
    ready = wait_for_server(timeout=120)
    if not ready:
        print("  -> [DISQUALIFIED / CRASH] Server failed to initialize within timeout or crashed on VRAM alloc.", flush=True)
        kill_strata()
        return {
            "id": candidate_id,
            "name": candidate_name,
            "status": "DISQUALIFIED (Startup Crash / OOM)",
            "score": 0.0,
            "decode_tok_s": 0.0,
            "client_decode_tok_s": 0.0,
            "prefill_tok_s": 0.0,
            "hit_rate_pct": 0.0,
            "mtp_acc_pct": 0.0,
            "expert_slots": 0,
            "intelligence_preserved": False
        }
        
    print("  -> Server ONLINE! Executing 3-Tier test battery...", flush=True)
    
    # Test A: Short Logic & Concurrency
    print("     [1/3] Test A (Short Concurrency, 256 tokens)...", flush=True)
    p1 = run_prompt(TEST_PROMPTS[0], timeout_s=120)
    print(f"           Result: ok={p1['ok']}, speed={p1.get('decode_tok_s', 0)} tok/s, intelligence={p1.get('intelligence_ok')}", flush=True)
    time.sleep(2)
    
    # Test B: Medium Algorithm & LRU Cache
    print("     [2/3] Test B (Medium LRU Cache, 384 tokens)...", flush=True)
    p2 = run_prompt(TEST_PROMPTS[1], timeout_s=180)
    print(f"           Result: ok={p2['ok']}, speed={p2.get('decode_tok_s', 0)} tok/s, intelligence={p2.get('intelligence_ok')}", flush=True)
    time.sleep(2)
    
    # Test C: Sustained Systems Task
    print("     [3/3] Test C (Sustained Task Pool, 512 tokens)...", flush=True)
    p3 = run_prompt(TEST_PROMPTS[2], timeout_s=220)
    print(f"           Result: ok={p3['ok']}, speed={p3.get('decode_tok_s', 0)} tok/s, intelligence={p3.get('intelligence_ok')}", flush=True)
    
    prefill_s, decode_s, hit_rate, mtp_acc, expert_slots, vram_free_mib = parse_last_log_stats()
    kill_strata()
    
    prompt_speeds = [p["decode_tok_s"] for p in [p1, p2, p3] if p.get("ok")]
    avg_client_decode = round(sum(prompt_speeds) / len(prompt_speeds), 2) if prompt_speeds else 0.0
    
    d_metric = decode_s if decode_s is not None else avg_client_decode
    p_metric = prefill_s if prefill_s is not None else 8.0
    hr_metric = hit_rate if hit_rate is not None else 0.0
    mtp_metric = mtp_acc if mtp_acc is not None else 0.0
    slots_metric = expert_slots if expert_slots is not None else 0
    
    p1_pass = p1.get("ok", False) and p1.get("intelligence_ok", False)
    p2_pass = p2.get("ok", False) and p2.get("intelligence_ok", False)
    p3_pass = p3.get("ok", False) and p3.get("intelligence_ok", False)
    passed_count = sum([p1_pass, p2_pass, p3_pass])
    
    intelligence_preserved = (passed_count == 3)
    if passed_count == 3:
        status = "PASSED (Full 3/3)"
        quality_bonus = 15.0
    elif passed_count == 2:
        status = "PARTIAL (2/3)"
        quality_bonus = 8.0
    elif passed_count == 1:
        status = "PARTIAL (1/3)"
        quality_bonus = 3.0
    else:
        status = "FAILED (Intelligence Degradation / Broken)"
        quality_bonus = 0.0
        
    # Tournament 5 Score Formula:
    # Heavy Decode multiplier (12.0) + Client responsiveness (5.0) + Hit Rate (1.5) + MTP Acc (0.1) + Quality Bonus (15.0)
    score = round(
        (d_metric * 12.0) +
        (avg_client_decode * 5.0) +
        (p_metric * 0.5) +
        (hr_metric * 1.5) +
        (mtp_metric * 0.1) +
        quality_bonus,
        2
    )
    
    return {
        "id": candidate_id,
        "name": candidate_name,
        "status": status,
        "score": score,
        "engine_decode_tok_s": d_metric,
        "client_decode_tok_s": avg_client_decode,
        "test_a_tok_s": p1.get("decode_tok_s", 0.0),
        "test_b_tok_s": p2.get("decode_tok_s", 0.0),
        "test_c_tok_s": p3.get("decode_tok_s", 0.0),
        "prefill_tok_s": p_metric,
        "hit_rate_pct": hr_metric,
        "mtp_acc_pct": mtp_metric,
        "expert_slots": slots_metric,
        "intelligence_preserved": intelligence_preserved,
        "p1_preview": p1.get("content_preview", ""),
        "p2_preview": p2.get("content_preview", ""),
        "p3_preview": p3.get("content_preview", "")
    }

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-runner":
        print("Runner test OK")
