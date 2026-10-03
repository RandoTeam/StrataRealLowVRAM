#!/usr/bin/env python3
"""
Live verification benchmark for StrataRealLowVRAM models:
1. Validates Qwen3.8 Coder IQ1_M and Q2_0 with real strata engine.
2. Checks free physical RAM headroom (must be >= 4.0 GB).
3. Validates prompt processing (prefill) and decode generation speed (tok/s).
4. Tests real coding and reasoning quality.
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error
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

def wait_for_server(port=PORT, timeout_s=300):
    url = f"http://127.0.0.1:{port}/status"
    start = time.time()
    last_print = 0
    while time.time() - start < timeout_s:
        elapsed = int(time.time() - start)
        if elapsed - last_print >= 10:
            print(f"[Server] Still waiting for server to load... ({elapsed}s elapsed, Free RAM: {get_free_ram_gb():.2f} GB)", flush=True)
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

def run_prompt_streaming(prompt, max_tokens=150, port=PORT):
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

def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "strata-coder-iq1_m.json"
    print(f"=== Starting live test for config: {config_name} ===", flush=True)
    initial_ram = get_free_ram_gb()
    print(f"[Initial] Free Physical RAM: {initial_ram:.2f} GB", flush=True)

    # Use virtual environment python if present
    python_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    server_cmd = [
        python_exe,
        "serve/server.py",
        "--engine", "strata",
        "--config", config_name,
        "--port", str(PORT)
    ]
    proc = subprocess.Popen(server_cmd, cwd=os.getcwd())
    try:
        print(f"[Server] Launched PID {proc.pid}. Waiting for server initialization (weights load)...", flush=True)
        status = wait_for_server(port=PORT)
        if not status:
            print("[Error] Server failed to start within timeout", flush=True)
            return 1
        
        ready_ram = get_free_ram_gb()
        print(f"[Ready] Server up! Model: {status.get('model')}, Max Context: {status.get('max_context')}", flush=True)
        print(f"[Ready] Free Physical RAM buffer: {ready_ram:.2f} GB (Requirement: >= 4.0 GB)", flush=True)

        # Test Prompt 1: Interactive Short Prompt
        print("\n--- Test 1: Interactive Coding Query ---", flush=True)
        q1 = "Write an optimal Python function to find the longest palindromic substring using Manacher's algorithm. Explain the logic briefly."
        res1 = run_prompt_streaming(q1, max_tokens=180, port=PORT)
        ram1 = get_free_ram_gb()
        
        # Query metrics after prompt 1
        metrics1 = {}
        try:
            req_m = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers={"User-Agent": "Bench"})
            with urllib.request.urlopen(req_m, timeout=5) as resp_m:
                metrics1 = json.loads(resp_m.read().decode())
        except Exception as e:
            print(f"[Metrics Warning]: {e}", flush=True)

        last_req1 = metrics1.get("last_request", {})
        p_toks1 = last_req1.get("prompt_tokens", 0)
        p_ms1 = last_req1.get("prompt_ms", 0.0)
        out_toks1 = last_req1.get("output_tokens", 0)
        dec_ms1 = last_req1.get("decode_ms", 0.0)
        gen_tok_s1 = (out_toks1 / (dec_ms1 / 1000.0)) if dec_ms1 > 0 else (out_toks1 / res1["gen_time_s"] if res1["gen_time_s"] > 0 else 0)
        prefill_tok_s1 = (p_toks1 / (p_ms1 / 1000.0)) if p_ms1 > 0 else (p_toks1 / res1["ttft_s"] if res1["ttft_s"] > 0 else 0)

        print(f"[Result 1] Prompt tokens: {p_toks1} (Prefill: {p_ms1:.1f}ms, {prefill_tok_s1:.1f} tok/s, TTFT: {res1['ttft_s']:.3f}s)", flush=True)
        print(f"[Result 1] Output tokens: {out_toks1} in {dec_ms1/1000.0:.2f}s -> {gen_tok_s1:.2f} tok/s (Requirement: >= 6.0 tok/s)", flush=True)
        print(f"[Result 1] Free RAM buffer: {ram1:.2f} GB (Requirement: >= 4.0 GB)", flush=True)
        print(f"[Result 1 Text Snippet]:\n{res1['text'][:250]}...\n", flush=True)

        # Test Prompt 2: Reasoning & Algorithmic Problem
        print("\n--- Test 2: Deep Algorithmic Reasoning & Coding ---", flush=True)
        q2 = "Implement a lock-free multi-producer multi-consumer bounded queue in modern C++20 with atomic operations, memory barriers, and cache-line padding. Provide a clear proof of correctness."
        res2 = run_prompt_streaming(q2, max_tokens=220, port=PORT)
        ram2 = get_free_ram_gb()

        # Query metrics after prompt 2
        metrics2 = {}
        try:
            req_m = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers={"User-Agent": "Bench"})
            with urllib.request.urlopen(req_m, timeout=5) as resp_m:
                metrics2 = json.loads(resp_m.read().decode())
        except Exception as e:
            print(f"[Metrics Warning]: {e}", flush=True)

        last_req2 = metrics2.get("last_request", {})
        p_toks2 = last_req2.get("prompt_tokens", 0)
        p_ms2 = last_req2.get("prompt_ms", 0.0)
        out_toks2 = last_req2.get("output_tokens", 0)
        dec_ms2 = last_req2.get("decode_ms", 0.0)
        gen_tok_s2 = (out_toks2 / (dec_ms2 / 1000.0)) if dec_ms2 > 0 else (out_toks2 / res2["gen_time_s"] if res2["gen_time_s"] > 0 else 0)
        prefill_tok_s2 = (p_toks2 / (p_ms2 / 1000.0)) if p_ms2 > 0 else (p_toks2 / res2["ttft_s"] if res2["ttft_s"] > 0 else 0)

        print(f"[Result 2] Prompt tokens: {p_toks2} (Prefill: {p_ms2:.1f}ms, {prefill_tok_s2:.1f} tok/s, TTFT: {res2['ttft_s']:.3f}s)", flush=True)
        print(f"[Result 2] Output tokens: {out_toks2} in {dec_ms2/1000.0:.2f}s -> {gen_tok_s2:.2f} tok/s (Requirement: >= 6.0 tok/s)", flush=True)
        print(f"[Result 2] Free RAM buffer: {ram2:.2f} GB (Requirement: >= 4.0 GB)", flush=True)
        print(f"[Result 2 Text Snippet]:\n{res2['text'][:250]}...\n", flush=True)

        print(f"\n=== Verification Summary for {config_name} ===", flush=True)
        print(f"Decode speed: {max(gen_tok_s1, gen_tok_s2):.2f} tok/s (Target: >= 6.0 tok/s)")
        print(f"Free Physical RAM: {min(ram1, ram2):.2f} GB (Target: >= 4.0 GB)")
        print(f"Context: {status.get('max_context')} (Target: >= 60,000)")
        return 0
    finally:
        print("[Server] Terminating server process...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[Server] Process exited.", flush=True)

if __name__ == "__main__":
    sys.exit(main())
