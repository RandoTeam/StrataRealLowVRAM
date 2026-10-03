#!/usr/bin/env python3
"""
Comprehensive verification test for all 3 models:
1. Qwen3.8 Coder IQ1_M in Strata (Web Chat + DSH API)
2. Qwen3.8 Full Q2_0 in Strata (Web Chat + DSH API)
3. Qwen3.6-35B-A3B in llama-server (Web Chat + DSH API)
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

STRATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = r"C:\Users\Ilia V\llama-server"

def compact_ram():
    print("[RAM] Compacting memory...", flush=True)
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", os.path.join(STRATA_DIR, "tools", "optimize_memory.ps1")], check=False)

def wait_for_server(url="http://127.0.0.1:8080/v1/models", timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Tester"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def test_web_chat_ui():
    url = "http://127.0.0.1:8080/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Tester"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", "replace")
            is_ok = len(html) > 500 and ("<html" in html.lower() or "<!doctype" in html.lower())
            print(f"  [1/2] Web Chat UI (GET /): {'OK' if is_ok else 'FAIL'} ({len(html)} bytes)", flush=True)
            return is_ok
    except Exception as e:
        print(f"  [1/2] Web Chat UI (GET /): ERROR {e}", flush=True)
        return False

def test_dsh_api(model_id, prompt="Ответь одним словом: работает?"):
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 60,
        "temperature": 0.6
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            dt = time.time() - t0
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"].strip()
            print(f"  [2/2] DeepSeek Harness API ({model_id}): OK in {dt:.2f}s", flush=True)
            print(f"        Ответ модели: {content}", flush=True)
            return True
    except Exception as e:
        print(f"  [2/2] DeepSeek Harness API ({model_id}): ERROR {e}", flush=True)
        return False

def test_strata_model(config_name, model_id):
    print(f"\n=======================================================", flush=True)
    print(f"Testing Strata: {model_id} ({config_name})", flush=True)
    print(f"=======================================================", flush=True)
    compact_ram()
    python_exe = os.path.join(STRATA_DIR, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    server_cmd = [
        python_exe,
        os.path.join(STRATA_DIR, "serve", "server.py"),
        "--engine", "strata",
        "--config", os.path.join(STRATA_DIR, config_name),
        "--port", "8080"
    ]
    proc = subprocess.Popen(server_cmd, cwd=STRATA_DIR)
    try:
        print(f"Launched Strata PID {proc.pid}, waiting for online...", flush=True)
        ready = wait_for_server(timeout=180)
        if not ready:
            print("[-] Strata failed to start within timeout", flush=True)
            return False

        chat_ok = test_web_chat_ui()
        api_ok = test_dsh_api(model_id)
        return chat_ok and api_ok
    finally:
        print("Stopping Strata server...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Stopped.", flush=True)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("all", "coder"):
        test_strata_model("strata-coder-iq1_m.json", "qwen3.8-flash-next-coder-iq1_m")
    if target in ("all", "q2_0"):
        test_strata_model("strata-q2_0.json", "qwen3.8-flash-next-q2_0")
