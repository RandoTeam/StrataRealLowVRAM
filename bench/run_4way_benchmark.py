import subprocess
import time
import json
import urllib.request
import urllib.error
import psutil
import os
import sys

PROMPT = "Реши задачу на Python: дан массив целых чисел, найди длину наибольшей строго возрастающей подпоследовательности за время O(n log n). Напиши эффективный код с подробным объяснением алгоритма."
MAX_TOKENS = 180

def get_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        ).strip().split(",")
        return float(out[0].strip()), float(out[1].strip()), float(out[2].strip())
    except Exception:
        return 0.0, 0.0, 0.0

def get_ram():
    vm = psutil.virtual_memory()
    return vm.used / (1024**3), vm.available / (1024**3), vm.percent

def kill_port(port=8080):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info['name'].lower()
            if "strata" in name or "python" in name:
                conns = proc.net_connections(kind='inet') if hasattr(proc, 'net_connections') else proc.connections(kind='inet')
                for conn in conns:
                    if conn.laddr.port == port:
                        print(f"[*] Killing PID {proc.pid} holding port {port}")
                        proc.kill()
        except Exception:
            pass
    time.sleep(2)

def wait_server_ready(server_proc, timeout=180):
    t0 = time.time()
    url = "http://127.0.0.1:8080/health"
    while time.time() - t0 < timeout:
        if server_proc.poll() is not None:
            print(f"[!] Server process exited unexpectedly with code {server_proc.returncode}!")
            return False
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "ok" and data.get("max_context", 0) > 0 and data.get("loaded", False):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def run_single_benchmark(test_name, config_file):
    print(f"\n{'='*70}")
    print(f"STARTING TEST: {test_name}")
    print(f"Config: {config_file}")
    print(f"{'='*70}")

    kill_port(8080)
    time.sleep(2)

    python_exe = "C:\\Users\\Ilia V\\Documents\\antigravity\\calm-noether\\Strata\\.venv\\Scripts\\python.exe"
    cmd = [python_exe, "serve/server.py", "--engine", "strata", "--config", config_file, "--port", "8080"]
    print(f"[*] Launching: {' '.join(cmd)}")
    server_proc = subprocess.Popen(cmd, cwd="C:\\Users\\Ilia V\\Documents\\antigravity\\calm-noether\\Strata")

    print("[*] Waiting for engine and server initialization...")
    ready = wait_server_ready(server_proc, timeout=180)
    if not ready:
        print("[!] Server failed to become ready within timeout!")
        server_proc.kill()
        return None

    ram_used, ram_avail, ram_pct = get_ram()
    vram_used, vram_total, gpu_util = get_vram()
    print(f"[*] Server READY. Initial RAM used: {ram_used:.2f} GB (free: {ram_avail:.2f} GB) | VRAM used: {vram_used:.0f}/{vram_total:.0f} MB")

    # Send streaming benchmark request
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "stream": True
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})

    t_start = time.time()
    t_first = None
    t_think_end = None
    t_end = None

    tokens_think = 0
    tokens_decode = 0
    full_text = ""
    in_think = False

    print("[*] Streaming response...")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    # Anthropic or OpenAI reasoning delta format
                    reasoning = delta.get("reasoning_content") or delta.get("thinking")
                    content = delta.get("content", "")

                    now = time.time()
                    if t_first is None:
                        t_first = now
                        print(f"[*] TTFT (Time to first token): {t_first - t_start:.2f} s")

                    if reasoning:
                        tokens_think += 1
                        full_text += reasoning
                    elif content:
                        if "<think>" in content:
                            in_think = True
                        if in_think:
                            tokens_think += 1
                            if "</think>" in content:
                                in_think = False
                                t_think_end = now
                                print(f"[*] Thinking phase finished: {tokens_think} tokens in {t_think_end - t_first:.2f} s")
                        else:
                            if t_think_end is None and tokens_think > 0:
                                t_think_end = now
                            tokens_decode += 1
                        full_text += content
                except Exception:
                    pass
        t_end = time.time()
    except Exception as e:
        print(f"[!] Request error: {e}")
        t_end = time.time()

    total_time = t_end - t_start
    ttft = (t_first - t_start) if t_first else 0.0

    if t_think_end is None:
        t_think_end = t_first if t_first else t_start

    think_duration = max(0.001, t_think_end - (t_first or t_start))
    decode_duration = max(0.001, t_end - t_think_end)

    total_tokens = tokens_think + tokens_decode
    gen_duration = max(0.001, t_end - (t_first or t_start))

    overall_tok_s = total_tokens / gen_duration if gen_duration > 0 else 0.0
    think_tok_s = tokens_think / think_duration if tokens_think > 0 else 0.0
    decode_tok_s = tokens_decode / decode_duration if tokens_decode > 0 else 0.0

    peak_ram_used, peak_ram_avail, _ = get_ram()
    peak_vram_used, _, _ = get_vram()

    result = {
        "test_name": test_name,
        "config": config_file,
        "total_tokens": total_tokens,
        "tokens_think": tokens_think,
        "tokens_decode": tokens_decode,
        "total_time_s": round(total_time, 2),
        "ttft_s": round(ttft, 2),
        "overall_tok_s": round(overall_tok_s, 2),
        "think_tok_s": round(think_tok_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "ram_used_gb": round(peak_ram_used, 2),
        "ram_avail_gb": round(peak_ram_avail, 2),
        "vram_used_mb": round(peak_vram_used, 0)
    }

    print(f"\n--- TEST RESULTS: {test_name} ---")
    print(f"Total Tokens: {total_tokens} (Thinking: {tokens_think}, Decode: {tokens_decode})")
    print(f"TTFT: {ttft:.2f} s")
    print(f"Thinking Speed: {think_tok_s:.2f} tok/s")
    print(f"Decode Speed:   {decode_tok_s:.2f} tok/s")
    print(f"Overall Rate:   {overall_tok_s:.2f} tok/s")
    print(f"RAM Used/Free:  {peak_ram_used:.2f} GB / {peak_ram_avail:.2f} GB")
    print(f"VRAM Used:      {peak_vram_used:.0f} MB")
    print(f"---------------------------------\n")

    # Clean termination
    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except Exception:
        server_proc.kill()
    kill_port(8080)

    print("[*] Cooldown 10s...")
    time.sleep(10)
    return result

if __name__ == "__main__":
    tests = [
        ("Model 1 [Coder IQ1_M] - StrataRealLowVRAM (Fork + v0.1.32)", "strata-coder-our.json"),
        ("Model 1 [Coder IQ1_M] - Vanilla Upstream Strata", "strata-coder-vanilla.json"),
        ("Model 2 [Full Q2_0] - StrataRealLowVRAM (Fork + v0.1.32)", "strata-q2_0-our.json"),
        ("Model 2 [Full Q2_0] - Vanilla Upstream Strata", "strata-q2_0-vanilla.json")
    ]

    all_results = []
    for name, cfg in tests:
        res = run_single_benchmark(name, cfg)
        if res:
            all_results.append(res)
        with open("bench/4way_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("ALL 4 BENCHMARKS COMPLETE. RESULTS SUMMARY:")
    print("="*80)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))
