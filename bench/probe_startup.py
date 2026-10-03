import subprocess
import os
import sys
import time
import json
import threading

def test_engine_startup(exe_name, env_overrides=None, extra_args=None):
    STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
    CONFIG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.json")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exe = os.path.join(STRATA_DIR, "engine", exe_name)
    args = ["--serve"] + list(cfg["args"])
    if extra_args:
        args.extend(extra_args)

    env = os.environ.copy()
    env["PATH"] = r"C:\Python312\Lib\site-packages\nvidia\cu13\bin\x86_64;C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\engine;" + env.get("PATH", "")
    base_env = cfg.get("env", {})
    env.update(base_env)
    if env_overrides:
        env.update(env_overrides)

    print(f"\n=======================================================", flush=True)
    print(f"Testing {exe_name}...", flush=True)
    print(f"ENV: {env_overrides}", flush=True)
    print(f"=======================================================", flush=True)

    p = subprocess.Popen(
        [exe] + args,
        cwd=cfg["cwd"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env
    )

    output_lines = []
    ready = False

    def reader(stream, label):
        nonlocal ready
        for line in stream:
            line_s = line.rstrip()
            output_lines.append(f"[{label}] {line_s}")
            print(f"[{label}] {line_s}", flush=True)
            if label == "STDOUT" and "READY" in line_s:
                ready = True

    t_err = threading.Thread(target=reader, args=(p.stderr, "STDERR"), daemon=True)
    t_out = threading.Thread(target=reader, args=(p.stdout, "STDOUT"), daemon=True)
    t_err.start()
    t_out.start()

    t0 = time.time()
    while time.time() - t0 < 60:
        if ready:
            print(f"SUCCESS: Engine reached ready state in {time.time()-t0:.1f}s!", flush=True)
            break
        if p.poll() is not None:
            print(f"EXITED with code {p.poll()} after {time.time()-t0:.1f}s", flush=True)
            break
        time.sleep(1)

    try:
        p.stdin.write("quit\n")
        p.stdin.flush()
    except Exception:
        pass

    time.sleep(2)
    if p.poll() is None:
        p.kill()
        p.wait()

if __name__ == "__main__":
    exe_choice = sys.argv[1] if len(sys.argv) > 1 else "strata.exe"
    pin = sys.argv[2] if len(sys.argv) > 2 else "1"
    test_engine_startup(exe_choice, env_overrides={"STRATA_ARENA_PIN_GIB": pin})
