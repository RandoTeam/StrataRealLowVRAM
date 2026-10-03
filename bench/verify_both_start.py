import subprocess
import time
import json
import urllib.request
import os
import sys
import psutil

def kill_port(port=8080):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info['name'].lower()
            if "strata" in name or "python" in name:
                conns = proc.net_connections(kind='inet') if hasattr(proc, 'net_connections') else proc.connections(kind='inet')
                for conn in conns:
                    if conn.laddr.port == port:
                        proc.kill()
        except Exception:
            pass
    time.sleep(1)

def verify_config(config_name):
    print(f"\n[*] Testing startup for: {config_name}")
    kill_port(8080)
    python_exe = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\.venv\Scripts\python.exe"
    cmd = [python_exe, "serve/server.py", "--engine", "strata", "--config", config_name, "--port", "8080"]
    p = subprocess.Popen(cmd, cwd=r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata")
    
    t0 = time.time()
    ready = False
    while time.time() - t0 < 60:
        if p.poll() is not None:
            print(f"[!] Engine exited with code {p.returncode}")
            return False
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get("status") == "ok" and data.get("loaded", False):
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(1)
        
    p.terminate()
    try:
        p.wait(timeout=3)
    except Exception:
        p.kill()
    kill_port(8080)
    
    if ready:
        print(f"[+] SUCCESS: {config_name} reached READY in {time.time()-t0:.1f}s")
        return True
    else:
        print(f"[!] FAILED: {config_name} did not reach ready within 60s")
        return False

if __name__ == "__main__":
    q2_ok = verify_config("strata-q2_0.json")
    coder_ok = verify_config("strata-coder-iq1_m.json")
    print(f"\nFinal Verification: Q2_0={'PASS' if q2_ok else 'FAIL'}, Coder={'PASS' if coder_ok else 'FAIL'}")
    sys.exit(0 if (q2_ok and coder_ok) else 1)
