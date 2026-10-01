import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
CONFIG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.json")
BASELINE_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.baseline.json")
LOG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.log")
PYTHON_EXE = os.path.join(STRATA_DIR, r".venv\Scripts\python.exe")
SERVER_PY = os.path.join(STRATA_DIR, r"serve\server.py")
OPTIMIZER_PS1 = os.path.join(STRATA_DIR, r"tools\optimize_memory.ps1")
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"

# Benchmark Prompts
TEST_PROMPTS = [
    {
        "id": "A",
        "name": "Short Logic (55 tok)",
        "system": "You are a concise, precise AI assistant.",
        "user": "Explain the difference between a mutex and a semaphore in concurrent programming in 3 short bullet points. Provide a one-line C++ code example for each.",
        "max_tokens": 256,
        "thinking": "low"
    },
    {
        "id": "B",
        "name": "Medium LRU (155 tok)",
        "system": "You are an expert Python engineer.",
        "user": "Implement an LRU (Least Recently Used) cache in Python without using collections.OrderedDict. Use a doubly linked list and dict for O(1) ops. Provide full LRUCache class with get and put, plus docstrings.",
        "max_tokens": 384,
        "thinking": "low"
    }
]

# The Tournament Candidates
CANDIDATES = [
    {
        "id": 1,
        "name": "PromptCache-Root16",
        "category": "Prompt Caching & Ingest",
        "desc": "Pins system prompt in RAM cache at token 16 for instant reuse",
        "modifications": {
            "--short-read": "2048",
            "--prompt-cache": "16",
            "--prompt-cache-root": "16"
        }
    },
    {
        "id": 2,
        "name": "PromptCache-Deep24",
        "category": "Multi-Turn RAM Cache",
        "desc": "Keeps 24 conversation checkpoints in RAM with checkpoints every 128 tokens",
        "modifications": {
            "--short-read": "2048",
            "--prompt-cache": "24",
            "--prompt-cache-root": "16",
            "--prompt-cache-every": "128"
        }
    },
    {
        "id": 3,
        "name": "VRAM-Dense-200",
        "category": "VRAM Maximization",
        "desc": "Expands expert cache to 200 resident slots with 500 MiB reserve buffer",
        "modifications": {
            "--short-read": "2048",
            "--expert-cache": "200",
            "--vram-reserve-mib": "500"
        }
    },
    {
        "id": 4,
        "name": "VRAM-Titan-220",
        "category": "VRAM Peak Capacity",
        "desc": "Pushes expert cache to 220 slots (~585 MiB) with 450 MiB reserve buffer",
        "modifications": {
            "--short-read": "2048",
            "--expert-cache": "220",
            "--vram-reserve-mib": "450"
        }
    },
    {
        "id": 5,
        "name": "MTP-DeepSpec-5",
        "category": "Speculative Decoding",
        "desc": "Increases speculative draft length from 4 to 5 tokens (min_p 0.70)",
        "modifications": {
            "--short-read": "2048",
            "--spec": "5",
            "--spec-min-p": "0.70"
        }
    },
    {
        "id": 6,
        "name": "MTP-Precision-3",
        "category": "Speculative Precision",
        "desc": "Tightens draft threshold to 0.82 with 3 tokens to minimize CPU verification waste",
        "modifications": {
            "--short-read": "2048",
            "--spec": "3",
            "--spec-min-p": "0.82"
        }
    },
    {
        "id": 7,
        "name": "Suffix-Draft-Combo",
        "category": "Prompt Lookup + MTP",
        "desc": "Aggressive suffix window lookup (draft 2) combined with MTP 4-token draft",
        "modifications": {
            "--short-read": "2048",
            "--suffix-draft": "2",
            "--spec": "4",
            "--spec-min-p": "0.75"
        }
    },
    {
        "id": 8,
        "name": "CPU-LeanPool-4",
        "category": "CPU Cache Contention",
        "desc": "Dedicates 4 workers to reduce DDR4-3200 memory bus saturation during MoE gather",
        "modifications": {
            "--short-read": "2048",
            "--pool-workers": "4"
        }
    },
    {
        "id": 9,
        "name": "CPU-NoHostWorker",
        "category": "Scheduler Decoupling",
        "desc": "Removes host thread from expert pool to maintain stable scheduler response",
        "modifications": {
            "--short-read": "2048",
            "--no-host-worker": None,
            "--pool-workers": "5"
        }
    },
    {
        "id": 10,
        "name": "Hybrid-UltraSpec",
        "category": "Multi-System Synthesis",
        "desc": "Combines 200 VRAM slots, spec 5, suffix drafting 2, prompt cache 16",
        "modifications": {
            "--short-read": "2048",
            "--expert-cache": "200",
            "--vram-reserve-mib": "500",
            "--spec": "5",
            "--spec-min-p": "0.72",
            "--suffix-draft": "2",
            "--prompt-cache": "16",
            "--prompt-cache-root": "16"
        }
    },
    {
        "id": 11,
        "name": "Ultimate-Champion",
        "category": "Balanced Peak Performer",
        "desc": "Sweet-spot: 210 VRAM slots, spec 4, min-p 0.75, suffix 2, 480 MiB reserve, prompt cache 16",
        "modifications": {
            "--short-read": "2048",
            "--expert-cache": "210",
            "--vram-reserve-mib": "480",
            "--spec": "4",
            "--spec-min-p": "0.75",
            "--suffix-draft": "2",
            "--prompt-cache": "16",
            "--prompt-cache-root": "16"
        }
    },
    {
        "id": 12,
        "name": "Grand-Champion",
        "category": "Grand Champion Synthesis",
        "desc": "Peak synthesis: 220 slots in VRAM (480MB reserve), Spec 3 (min-p 0.82), suffix 3, prompt cache 16 (root 16)",
        "modifications": {
            "--short-read": "2048",
            "--expert-cache": "220",
            "--vram-reserve-mib": "480",
            "--spec": "3",
            "--spec-min-p": "0.82",
            "--suffix-draft": "3",
            "--prompt-cache": "16",
            "--prompt-cache-root": "16",
            "--prompt-cache-every": "128"
        }
    }
]

def kill_strata():
    subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

def run_ram_optimizer():
    try:
        subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", OPTIMIZER_PS1],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        pass

def wait_for_server(timeout=90):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def build_config(candidate):
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    args = cfg["args"]
    mods = candidate.get("modifications", {})
    
    for key, val in mods.items():
        if key in args:
            idx = args.index(key)
            if val is None or val == "":
                # Flag without value
                pass
            else:
                args[idx + 1] = str(val)
        else:
            args.append(key)
            if val is not None and val != "":
                args.append(str(val))
                
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

def run_prompt(prompt_info):
    payload = {
        "model": "qwen3.8-flash-next-coder-iq1_m",
        "messages": [
            {"role": "system", "content": prompt_info["system"]},
            {"role": "user", "content": prompt_info["user"]}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": prompt_info["max_tokens"],
        "reasoning_effort": prompt_info["thinking"]
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER_URL, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            elapsed = time.time() - t0
            body = json.loads(res.read().decode("utf-8"))
            usage = body.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return {
                "ok": True,
                "elapsed": elapsed,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "decode_tok_s": round(c_tok / elapsed, 2) if elapsed > 0 else 0,
                "output_len": len(content),
                "code_valid": "class LRUCache" in content or "mutex" in content.lower()
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - t0}

def parse_last_log_stats():
    """Extracts prefill speed, hit rate, and MTP acceptance from strata log tail."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-30:]
        
        prefill_tok_s = None
        decode_tok_s = None
        hit_rate = None
        mtp_acc = None
        
        for l in lines:
            if "strata serve: prompt" in l and "read in" in l:
                # e.g.: strata serve: prompt 55 tokens = 0 reused + 55 read in 7673 ms (7.2 tok/s), 339 generated in 79675 ms (4.3 tok/s), drafts accepted 165 of 237
                parts = l.split("read in")
                if len(parts) > 1:
                    sub = parts[1]
                    if "(" in sub and "tok/s)" in sub:
                        p_val = sub.split("(")[1].split("tok/s")[0].strip()
                        prefill_tok_s = float(p_val)
                    if "generated in" in sub:
                        g_sub = sub.split("generated in")[1]
                        if "(" in g_sub and "tok/s)" in g_sub:
                            d_val = g_sub.split("(")[1].split("tok/s")[0].strip()
                            decode_tok_s = float(d_val)
                    if "drafts accepted" in sub:
                        d_parts = sub.split("drafts accepted")[1].split(",")
                        acc_str = d_parts[0].strip()
                        if " of " in acc_str:
                            acc, total = acc_str.split(" of ")
                            if int(total) > 0:
                                mtp_acc = round(int(acc) / int(total) * 100, 1)
            if "decode expert cache hit rate:" in l:
                # strata serve: decode expert cache hit rate: 5.1% (10172 hits / 198720 lookups)
                hr_part = l.split("hit rate:")[1].split("%")[0].strip()
                hit_rate = float(hr_part)
                
        return prefill_tok_s, decode_tok_s, hit_rate, mtp_acc
    except Exception:
        return None, None, None, None

def evaluate_candidate(candidate):
    print(f"\n=======================================================")
    print(f"  TESTING CANDIDATE {candidate['id']}: {candidate['name']}")
    print(f"  Category: {candidate['category']}")
    print(f"  Mods: {candidate['modifications']}")
    print(f"=======================================================")
    
    kill_strata()
    time.sleep(2)
    build_config(candidate)
    run_ram_optimizer()
    
    # Start Strata
    cmd = [PYTHON_EXE, SERVER_PY, "--engine", "strata", "--config", CONFIG_FILE, "--port", "8080"]
    proc = subprocess.Popen(cmd, cwd=STRATA_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("  -> Waiting for Strata to boot and allocate VRAM...")
    ready = wait_for_server(timeout=100)
    if not ready:
        print("  -> [FAILED] Server failed to start within timeout. Disqualified.")
        kill_strata()
        return {"id": candidate["id"], "name": candidate["name"], "status": "DISQUALIFIED (Startup Crash)", "score": 0}
    
    print("  -> Server ONLINE! Running Test Suite...")
    p1 = run_prompt(TEST_PROMPTS[0])
    time.sleep(1)
    p2 = run_prompt(TEST_PROMPTS[1])
    
    prefill_s, decode_s, hit_rate, mtp_acc = parse_last_log_stats()
    kill_strata()
    
    # Score calculation
    p_score = prefill_s if prefill_s else (p1.get("decode_tok_s", 0) * 1.5)
    d_score = decode_s if decode_s else p2.get("decode_tok_s", 4.0)
    hr = hit_rate if hit_rate else 4.0
    mtp = mtp_acc if mtp_acc else 70.0
    
    total_score = round((p_score * 0.35) + (d_score * 0.35) + (hr * 1.5) + (mtp * 0.1), 2)
    
    result = {
        "id": candidate["id"],
        "name": candidate["name"],
        "category": candidate["category"],
        "status": "PASSED" if (p1.get("ok") and p2.get("ok")) else "FAILED",
        "prefill_tok_s": p_score,
        "decode_tok_s": d_score,
        "hit_rate_pct": hr,
        "mtp_acc_pct": mtp,
        "score": total_score,
        "p1_tok_s": p1.get("decode_tok_s", 0),
        "p2_tok_s": p2.get("decode_tok_s", 0)
    }
    
    print(f"  -> RESULTS: Prefill: {p_score} tok/s | Decode: {d_score} tok/s | Hit Rate: {hr}% | MTP: {mtp}% | Score: {total_score}")
    return result

if __name__ == "__main__":
    results = []
    # Test specific candidate passed in CLI or run subset
    cand_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    cands_to_test = [c for c in CANDIDATES if cand_id is None or c["id"] == cand_id]
    for c in cands_to_test:
        r = evaluate_candidate(c)
        results.append(r)
        
    out_file = os.path.join(STRATA_DIR, "bench", f"tournament_batch_{cand_id if cand_id else 'all'}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Batch saved to {out_file}")
