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
        "name": "Short Logic (Test A)",
        "system": "You are a concise, precise AI assistant.",
        "user": "Explain the difference between a mutex and a semaphore in concurrent programming in 3 short bullet points. Provide a one-line C++ code example for each.",
        "max_tokens": 256,
        "thinking": "low"
    },
    {
        "id": "B",
        "name": "Medium LRU (Test B)",
        "system": "You are an expert Python engineer.",
        "user": "Implement an LRU (Least Recently Used) cache in Python without using collections.OrderedDict. Use a doubly linked list and dict for O(1) ops. Provide full LRUCache class with get and put, plus docstrings.",
        "max_tokens": 384,
        "thinking": "low"
    }
]

def kill_strata():
    subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def run_ram_optimizer():
    if os.path.exists(OPTIMIZER_PS1):
        try:
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", OPTIMIZER_PS1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

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

def run_prompt(prompt_data):
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
        with urllib.request.urlopen(req, timeout=180) as resp:
            elapsed = time.time() - start_t
            res_data = json.loads(resp.read().decode("utf-8"))
            content = res_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = res_data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tok_s = round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens > 0 else 0
            return {
                "ok": True,
                "elapsed_s": round(elapsed, 2),
                "completion_tokens": completion_tokens,
                "decode_tok_s": tok_s,
                "content_preview": content[:120] if content else "",
                "has_content": bool(content and len(content.strip()) > 20)
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed_s": round(time.time() - start_t, 2), "decode_tok_s": 0, "has_content": False}

def parse_last_log_stats():
    """
    Parses the most recent engine metrics from strata-coder-iq1_m.log.
    """
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
            # Parse expert slots
            if expert_slots is None and "expert cache" in l and "slots" in l:
                # e.g.: expert cache 220 slots, 0.42 GiB of VRAM
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
                # e.g.: strata serve: prompt 175 tokens = 0 reused + 175 read in 21875 ms (8.0 tok/s), 143 generated in 24656 ms (5.8 tok/s), drafts accepted 361 of 402
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
                # strata serve: decode expert cache hit rate: 5.1% (10172 hits / 198720 lookups)
                hr_part = l.split("hit rate:")[1].split("%")[0].strip()
                hit_rate = float(hr_part)
                
        return prefill_tok_s, decode_tok_s, hit_rate, mtp_acc, expert_slots, vram_free_mib
    except Exception:
        return None, None, None, None, None, None

def build_config_from_args(args_list, config_path=CONFIG_FILE):
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["args"] = args_list
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

def evaluate_candidate(candidate_id, candidate_name, candidate_category, args_list, env_dict=None):
    print(f"\n=======================================================")
    print(f"  TOURNAMENT 2: TESTING CANDIDATE {candidate_id}: {candidate_name}")
    print(f"  Category: {candidate_category}")
    print(f"  Args count: {len(args_list)}")
    if env_dict:
        print(f"  Custom Env: {env_dict}")
    print(f"=======================================================")
    
    kill_strata()
    time.sleep(2)
    build_config_from_args(args_list)
    run_ram_optimizer()
    
    # Start Strata server
    cmd = [PYTHON_EXE, SERVER_PY, "--engine", "strata", "--config", CONFIG_FILE, "--port", "8080"]
    sub_env = os.environ.copy()
    if env_dict:
        for k, v in env_dict.items():
            sub_env[k] = str(v)
    proc = subprocess.Popen(cmd, cwd=STRATA_DIR, env=sub_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("  -> Waiting for Strata server to boot and allocate VRAM (timeout 110s)...")
    ready = wait_for_server(timeout=110)
    if not ready:
        print("  -> [DISQUALIFIED] Server failed to start within timeout.")
        kill_strata()
        return {
            "id": candidate_id,
            "name": candidate_name,
            "category": candidate_category,
            "status": "DISQUALIFIED (Startup Crash)",
            "score": 0.0,
            "decode_tok_s": 0.0,
            "prefill_tok_s": 0.0,
            "hit_rate_pct": 0.0,
            "mtp_acc_pct": 0.0,
            "expert_slots": 0,
            "vram_free_mib": 0
        }
    
    print("  -> Server ONLINE! Running Test A (Short Logic)...")
    p1 = run_prompt(TEST_PROMPTS[0])
    print(f"     Test A: ok={p1['ok']}, tok/s={p1.get('decode_tok_s', 0)}, tokens={p1.get('completion_tokens', 0)}")
    time.sleep(1)
    
    print("  -> Running Test B (Medium Algorithm)...")
    p2 = run_prompt(TEST_PROMPTS[1])
    print(f"     Test B: ok={p2['ok']}, tok/s={p2.get('decode_tok_s', 0)}, tokens={p2.get('completion_tokens', 0)}")
    
    prefill_s, decode_s, hit_rate, mtp_acc, expert_slots, vram_free_mib = parse_last_log_stats()
    kill_strata()
    
    # Metrics fallback
    p_metric = prefill_s if prefill_s is not None else p1.get("decode_tok_s", 0.0)
    d_metric = decode_s if decode_s is not None else p2.get("decode_tok_s", 0.0)
    hr_metric = hit_rate if hit_rate is not None else 0.0
    mtp_metric = mtp_acc if mtp_acc is not None else 0.0
    slots_metric = expert_slots if expert_slots is not None else 0
    free_metric = vram_free_mib if vram_free_mib is not None else 0
    
    # Quality Bonus
    p1_pass = p1.get("ok", False) and p1.get("has_content", False)
    p2_pass = p2.get("ok", False) and p2.get("has_content", False)
    
    if p1_pass and p2_pass:
        quality_bonus = 10.0
        status = "PASSED"
    elif p1_pass or p2_pass:
        quality_bonus = 5.0
        status = "PARTIAL"
    else:
        quality_bonus = 0.0
        status = "FAILED"
        
    # Tournament 2 Score Formula:
    # Score = (Decode * 5.0) + (Prefill * 0.5) + (Hit Rate * 2.0) + (MTP Acc * 0.15) + Quality Bonus
    total_score = round(
        (d_metric * 5.0) +
        (p_metric * 0.5) +
        (hr_metric * 2.0) +
        (mtp_metric * 0.15) +
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
        "quality_bonus": quality_bonus,
        "score": total_score,
        "test_a": p1,
        "test_b": p2
    }
    
    print(f"  -> SCORE: {total_score} | Decode: {d_metric} tok/s | Prefill: {p_metric} tok/s | Hit Rate: {hr_metric}% | MTP: {mtp_metric}% | Slots: {slots_metric}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner.py <candidate_json_file>")
        sys.exit(1)
        
    cand_file = sys.argv[1]
    with open(cand_file, "r", encoding="utf-8") as f:
        cand_data = json.load(f)
        
    res = evaluate_candidate(
        cand_data["id"],
        cand_data["name"],
        cand_data.get("category", "General"),
        cand_data["args"],
        cand_data.get("env", None)
    )
    
    out_file = cand_file.replace(".json", "_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {out_file}")
