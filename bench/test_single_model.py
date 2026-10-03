import subprocess
import time
import json
import urllib.request
import urllib.error
import psutil
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

PROMPT = "Реши задачу на Python: дан массив целых чисел, найди длину наибольшей строго возрастающей подпоследовательности за время O(n log n). Напиши эффективный код с подробным объяснением алгоритма."
MAX_TOKENS = 250

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
                        print(f"[*] Stopping existing PID {proc.pid} on port {port}")
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

def test_model(config_path, model_label):
    print(f"\n{'='*80}")
    print(f"STARTING TRANSPARENT RUN: {model_label}")
    print(f"Config: {config_path}")
    print(f"{'='*80}\n")

    kill_port(8080)
    time.sleep(2)

    python_exe = "C:\\Users\\Ilia V\\Documents\\antigravity\\calm-noether\\Strata\\.venv\\Scripts\\python.exe"
    cmd = [python_exe, "serve/server.py", "--engine", "strata", "--config", config_path, "--port", "8080"]
    print(f"[*] Starting server command: {' '.join(cmd)}")
    server_proc = subprocess.Popen(cmd, cwd="C:\\Users\\Ilia V\\Documents\\antigravity\\calm-noether\\Strata")

    print("[*] Loading model weights and allocating engine structures...")
    ready = wait_server_ready(server_proc, timeout=180)
    if not ready:
        print("[!] Server failed to report READY!")
        server_proc.kill()
        return None

    ram_used, ram_avail, ram_pct = get_ram()
    vram_used, vram_total, gpu_util = get_vram()
    print(f"\n[+] Engine is READY and initialized:")
    print(f"    - RAM in use:   {ram_used:.2f} GB (Available buffer: {ram_avail:.2f} GB)")
    print(f"    - VRAM in use:  {vram_used:.0f} / {vram_total:.0f} MB")
    print(f"    - CPU load:     {psutil.cpu_percent(interval=0.5):.1f}%")

    print(f"\n[*] Sending Prompt:")
    print(f"    \"{PROMPT}\"")
    print(f"    Max tokens: {MAX_TOKENS}\n")
    print("-" * 80)
    print("LIVE STREAMING MODEL OUTPUT:")
    print("-" * 80)

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
    full_output = ""
    in_think = False

    cpu_samples = []
    gpu_samples = []

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
                    reasoning = delta.get("reasoning_content") or delta.get("thinking")
                    content = delta.get("content", "")

                    now = time.time()
                    if t_first is None:
                        t_first = now
                        ttft = t_first - t_start
                        print(f"\n[TTFT: {ttft:.2f} s]")

                    if reasoning:
                        tokens_think += 1
                        full_output += reasoning
                        sys.stdout.write(reasoning)
                        sys.stdout.flush()
                    elif content:
                        if "<think>" in content:
                            in_think = True
                        if in_think:
                            tokens_think += 1
                            if "</think>" in content:
                                in_think = False
                                t_think_end = now
                        else:
                            if t_think_end is None and tokens_think > 0:
                                t_think_end = now
                            tokens_decode += 1
                        full_output += content
                        sys.stdout.write(content)
                        sys.stdout.flush()

                    if (tokens_think + tokens_decode) % 15 == 0:
                        cpu_samples.append(psutil.cpu_percent())
                        gpu_vram, _, gpu_u = get_vram()
                        gpu_samples.append(gpu_u)

                except Exception:
                    pass
        t_end = time.time()
    except Exception as e:
        print(f"\n[!] Streaming request exception: {e}")
        t_end = time.time()

    print("\n" + "-" * 80)
    print("STREAMING COMPLETE\n")

    ttft = (t_first - t_start) if t_first else 0.0
    if t_think_end is None:
        t_think_end = t_first if t_first else t_start

    think_dur = max(0.001, t_think_end - (t_first or t_start))
    decode_dur = max(0.001, t_end - t_think_end)
    total_tokens = tokens_think + tokens_decode
    gen_dur = max(0.001, t_end - (t_first or t_start))

    think_tok_s = tokens_think / think_dur if tokens_think > 0 else 0.0
    decode_tok_s = tokens_decode / decode_dur if tokens_decode > 0 else 0.0
    overall_tok_s = total_tokens / gen_dur if gen_dur > 0 else 0.0

    peak_ram_used, peak_ram_avail, _ = get_ram()
    peak_vram_used, _, _ = get_vram()
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else psutil.cpu_percent()
    avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0.0

    metrics = {
        "model": model_label,
        "config": config_path,
        "total_tokens": total_tokens,
        "tokens_think": tokens_think,
        "tokens_decode": tokens_decode,
        "ttft_s": round(ttft, 2),
        "think_tok_s": round(think_tok_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "overall_tok_s": round(overall_tok_s, 2),
        "ram_used_gb": round(peak_ram_used, 2),
        "ram_avail_gb": round(peak_ram_avail, 2),
        "vram_used_mb": round(peak_vram_used, 0),
        "avg_cpu_percent": round(avg_cpu, 1),
        "avg_gpu_util_percent": round(avg_gpu, 1)
    }

    print(f"{'='*80}")
    print(f"AUDIT SUMMARY: {model_label}")
    print(f"{'='*80}")
    print(f"Generated Tokens:  {total_tokens} (Thinking: {tokens_think}, Code/Answer: {tokens_decode})")
    print(f"TTFT (Prefill):    {ttft:.2f} s")
    print(f"Thinking Speed:    {think_tok_s:.2f} tok/s")
    print(f"Decode/Code Speed: {decode_tok_s:.2f} tok/s")
    print(f"Overall Speed:     {overall_tok_s:.2f} tok/s")
    print(f"Hardware State:")
    print(f"  - RAM in use:    {peak_ram_used:.2f} GB")
    print(f"  - FREE RAM:      {peak_ram_avail:.2f} GB (BUFFER HEALTHY: {'YES' if peak_ram_avail >= 2.0 else 'NO'})")
    print(f"  - VRAM in use:   {peak_vram_used:.0f} MB")
    print(f"  - Average CPU%:  {avg_cpu:.1f}%")
    print(f"{'='*80}\n")

    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except Exception:
        server_proc.kill()
    kill_port(8080)
    print("[*] Server terminated. Cooldown 10s...")
    time.sleep(10)

    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_single_model.py <config_path> <model_label>")
        sys.exit(1)
    cfg = sys.argv[1]
    label = sys.argv[2]
    res = test_model(cfg, label)
    if res:
        with open(f"bench/{label.replace(' ', '_').lower()}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
