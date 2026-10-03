import subprocess
import time
import json
import urllib.request
import psutil
import sys

def run():
    # Kill any existing server
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'strata' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                for c in (proc.net_connections(kind='inet') if hasattr(proc, 'net_connections') else proc.connections(kind='inet')):
                    if c.laddr.port == 8080:
                        proc.kill()
        except Exception:
            pass
    time.sleep(2)
    
    cmd = [r".venv\Scripts\python.exe", "serve/server.py", "--engine", "strata", "--config", "strata-coder-iq1_m.json", "--port", "8080"]
    print("[*] Launching Coder IQ1_M server...")
    p = subprocess.Popen(cmd)
    
    # Wait for ready
    ready = False
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as r:
                d = json.loads(r.read().decode())
                if d.get("loaded"):
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(1)
        
    if not ready:
        print("[!] Server failed to load!")
        p.kill()
        return
        
    print("[+] Server READY!")
    time.sleep(2)
    
    # 1. Test short prompt (100 tokens)
    print("\n--- TEST 1: 100 Tokens ---")
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "Explain binary search in 3 sentences."}], "max_tokens": 10, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read().decode())
        print(f"Elapsed: {time.time()-t0:.2f}s, tokens: {res.get('usage', {})}")

    # 2. Test medium prompt (500 tokens)
    print("\n--- TEST 2: ~500 Tokens Fresh ---")
    long_p = "The quick brown fox jumps over the lazy dog. Artificial intelligence and tensor core computing accelerate neural models. " * 35
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": long_p}], "max_tokens": 5, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        res = json.loads(r.read().decode())
        print(f"Elapsed: {time.time()-t0:.2f}s, tokens: {res.get('usage', {})}")

    # 3. Test large prompt (1500 tokens)
    print("\n--- TEST 3: ~1500 Tokens Fresh ---")
    long_p2 = "In computer science, algorithms and data structures are the foundational pillars of software engineering. High throughput matrix multiplication requires efficient memory hierarchies. " * 65
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": long_p2}], "max_tokens": 5, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        res = json.loads(r.read().decode())
        print(f"Elapsed: {time.time()-t0:.2f}s, tokens: {res.get('usage', {})}")

    p.kill()

if __name__ == "__main__":
    run()
