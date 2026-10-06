import subprocess
import time
import json
import urllib.request
import urllib.error
import sys
import os
import signal

def run_test(config_path, test_prompt="Write a Python quicksort algorithm.", max_tokens=100):
    cmd = [
        os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe"),
        os.path.join(os.getcwd(), "serve", "server.py"),
        "--engine", "strata",
        "--config", config_path,
        "--port", "8080"
    ]
    print(f"Starting server with config: {config_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    ready = False
    start_time = time.time()
    server_output = []

    # Monitor output until ready (timeout 180s)
    while time.time() - start_time < 180:
        line = proc.stdout.readline()
        if line:
            server_output.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            if "ready: http://" in line or "Uvicorn running" in line or "Serving on" in line or "ready: 127.0.0.1" in line:
                ready = True
                break
        if proc.poll() is not None:
            print(f"Server exited early with code {proc.returncode}")
            return False, "Exited early"

    if not ready:
        print("Server did not become ready within timeout.")
        proc.terminate()
        return False, "Timeout"

    print("\n--- Server is READY! Sending inference request ---")
    req_data = {
        "model": "default",
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }
    req_body = json.dumps(req_data).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=req_body,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - t0
            res = json.loads(resp.read().decode("utf-8"))
            choice = res["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            usage = res.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tok_s = completion_tokens / elapsed if elapsed > 0 else 0

            print("\n=== INFERENCE RESULT ===")
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"Generated tokens: {completion_tokens}")
            print(f"Speed: {tok_s:.2f} tok/s")
            print(f"Usage details: {usage}")
            if reasoning:
                print(f"Reasoning preview:\n{reasoning[:200]}...\n")
            if content:
                print(f"Content preview:\n{content[:200]}...\n")
            print("========================\n")
    except Exception as e:
        print(f"Inference request failed: {e}")
        proc.terminate()
        return False, str(e)

    # Stop server gracefully
    print("Terminating server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Server stopped cleanly.")
    return True, f"{tok_s:.2f} tok/s"

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "strata-coder-iq1_m.json"
    ok, msg = run_test(cfg)
    sys.exit(0 if ok else 1)
