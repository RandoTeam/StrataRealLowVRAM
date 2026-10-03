#!/usr/bin/env python3
"""
Automated benchmark and HTML game generation for:
1. Qwen3.8 Coder IQ1_M
2. Qwen3.8 Full Q2_0
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error
import re
import ctypes

PORT = 8080

def get_free_ram_gb():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024 ** 3)

def wait_for_server(port=PORT, timeout_s=360):
    url = f"http://127.0.0.1:{port}/status"
    start = time.time()
    last_print = 0
    while time.time() - start < timeout_s:
        elapsed = int(time.time() - start)
        if elapsed - last_print >= 10:
            print(f"  [Wait] Server initializing... ({elapsed}s elapsed, Free RAM: {get_free_ram_gb():.2f} GB)", flush=True)
            last_print = elapsed
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Bench"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if not data.get("busy"):
                        return data
        except Exception:
            pass
        time.sleep(2)
    return None

def run_prompt_streaming(prompt, max_tokens=1000, port=PORT):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    first_token_time = None
    chunks = []
    
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                continue
            try:
                chunk_json = json.loads(line_str[6:])
            except Exception:
                continue
            if first_token_time is None:
                first_token_time = time.time() - t0
            choices = chunk_json.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    chunks.append(delta["content"])
    
    t_end = time.time()
    total_time = t_end - t0
    gen_time = (t_end - (t0 + first_token_time)) if first_token_time else total_time
    full_text = "".join(chunks)
    return {
        "text": full_text,
        "ttft_s": first_token_time or 0.0,
        "total_time_s": total_time,
        "gen_time_s": gen_time
    }

def extract_html(raw_text):
    # Try finding ```html ... ```
    match = re.search(r"```html\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try finding <!DOCTYPE html> ... </html>
    match = re.search(r"(<!DOCTYPE html.*?>.*?</html>)", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try finding <html> ... </html>
    match = re.search(r"(<html.*?>.*?</html>)", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Otherwise return raw text
    return raw_text.strip()

def run_model_test(config_file, out_dir, model_name):
    print(f"\n=======================================================", flush=True)
    print(f"Starting Benchmark for: {model_name} ({config_file})", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Compact memory
    print("[1/4] Compacting RAM...", flush=True)
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "tools\\optimize_memory.ps1"], check=False)
    initial_ram = get_free_ram_gb()
    print(f"      Free RAM before server start: {initial_ram:.2f} GB", flush=True)

    # 2. Launch server
    python_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    server_cmd = [
        python_exe,
        "serve/server.py",
        "--engine", "strata",
        "--config", config_file,
        "--port", str(PORT)
    ]
    proc = subprocess.Popen(server_cmd, cwd=os.getcwd())
    try:
        print(f"[2/4] Launched PID {proc.pid}. Waiting for weights to load...", flush=True)
        status = wait_for_server(port=PORT)
        if not status:
            print("[-] Server failed to start within timeout", flush=True)
            return None
        
        ready_ram = get_free_ram_gb()
        print(f"      [OK] Server online! Context: {status.get('max_context')}, Free RAM: {ready_ram:.2f} GB", flush=True)

        # 3. Prompt generation
        print("[3/4] Sending Tic-Tac-Toe generation prompt...", flush=True)
        prompt = (
            "Write a complete, single-file Tic-Tac-Toe (крестики-нолики) game in HTML with embedded CSS and JavaScript. "
            "It must be fully playable, responsive, beautifully styled with a modern dark theme, score counter, "
            "win/draw detection with line/cell highlighting, reset button, and an option to play vs simple AI or 2-player mode. "
            "Output ONLY the complete HTML code enclosed in ```html ... ``` code block."
        )

        res = run_prompt_streaming(prompt, max_tokens=1000, port=PORT)
        peak_ram = get_free_ram_gb()

        # Metrics query
        metrics = {}
        try:
            req_m = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers={"User-Agent": "Bench"})
            with urllib.request.urlopen(req_m, timeout=5) as resp_m:
                metrics = json.loads(resp_m.read().decode())
        except Exception:
            pass

        last_req = metrics.get("last_request", {})
        prompt_tokens = last_req.get("prompt_tokens", 0)
        prompt_ms = last_req.get("prompt_ms", 0.0)
        output_tokens = last_req.get("output_tokens", 0)
        decode_ms = last_req.get("decode_ms", 0.0)

        prefill_tok_s = (prompt_tokens / (prompt_ms / 1000.0)) if prompt_ms > 0 else (prompt_tokens / res["ttft_s"] if res["ttft_s"] > 0 else 0)
        decode_tok_s = (output_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else (output_tokens / res["gen_time_s"] if res["gen_time_s"] > 0 else 0)

        # 4. Save HTML game
        print("[4/4] Extracting HTML and saving game...", flush=True)
        html_code = extract_html(res["text"])
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "index.html")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_code)
        
        file_size_kb = os.path.getsize(out_file) / 1024
        print(f"      Saved HTML game to: {out_file} ({file_size_kb:.1f} KB)", flush=True)

        result = {
            "model": model_name,
            "config": config_file,
            "initial_ram_gb": round(initial_ram, 2),
            "ready_ram_gb": round(ready_ram, 2),
            "peak_free_ram_gb": round(peak_ram, 2),
            "prompt_tokens": prompt_tokens,
            "prompt_ms": round(prompt_ms, 1),
            "prefill_tok_s": round(prefill_tok_s, 2),
            "ttft_s": round(res["ttft_s"], 3),
            "output_tokens": output_tokens,
            "decode_ms": round(decode_ms, 1),
            "decode_tok_s": round(decode_tok_s, 2),
            "total_time_s": round(res["total_time_s"], 2),
            "game_path": out_file,
            "game_size_kb": round(file_size_kb, 1),
            "expert_cache_hits": last_req.get("hits", 0),
            "expert_cache_lookups": last_req.get("lookups", 0)
        }
        print(f"      Result: {output_tokens} tokens in {decode_ms/1000:.2f}s ({decode_tok_s:.2f} tok/s), TTFT: {res['ttft_s']:.3f}s, Free RAM: {peak_ram:.2f} GB", flush=True)
        return result
    finally:
        print("  [Cleanup] Shutting down server...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  [Cleanup] Server stopped.", flush=True)

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = []

    if target in ("all", "coder"):
        res_coder = run_model_test(
            "strata-coder-iq1_m.json",
            "test_results/tic_tac_toe/coder_iq1_m",
            "Qwen3.8-Flash-Next-Coder-IQ1_M"
        )
        if res_coder:
            results.append(res_coder)

    if target in ("all", "q2_0"):
        res_q2 = run_model_test(
            "strata-q2_0.json",
            "test_results/tic_tac_toe/q2_0",
            "Qwen3.8-Flash-Next-Q2_0"
        )
        if res_q2:
            results.append(res_q2)

    with open("test_results/tic_tac_toe/summary_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================== SUMMARY ==================", flush=True)
    print(json.dumps(results, indent=2), flush=True)

if __name__ == "__main__":
    main()
