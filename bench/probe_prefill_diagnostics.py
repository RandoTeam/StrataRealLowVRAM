import subprocess
import os
import sys
import time
import json
import threading

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
CONFIG_FILE = os.path.join(STRATA_DIR, "strata-q2_0.json")

def run_diagnostics():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exe = cfg.get("exe", os.path.join(STRATA_DIR, "engine", "strata.exe"))
    args = ["--serve"] + list(cfg["args"])
    if "--short-read" in args:
        idx = args.index("--short-read")
        args[idx + 1] = "64"
    else:
        args += ["--short-read", "64"]
    env = os.environ.copy()
    env["PATH"] = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\.venv\Lib\site-packages\nvidia\cu13\bin;" + env.get("PATH", "")
    env.update(cfg.get("env", {}))
    env["STRATA_PREFILL_TIMING"] = "1"
    env["STRATA_TRACE"] = "1"
    env["STRATA_PF_FUSED"] = "1"
    env["STRATA_PARTIAL_PIN"] = "1"
    env["STRATA_PARTIAL_PIN_GIB"] = "4"
    env["STRATA_STAGER_THREADS"] = "8"
    env["STRATA_STAGER_RING"] = "64"
    env["STRATA_PREFILL_STREAM_MIN"] = "32"
    env["STRATA_PREFILL_RING"] = "512"

    print("=======================================================", flush=True)
    print("Launching strata for Prefill Diagnostics...", flush=True)
    print("Args:", " ".join(args[:15]), "...", flush=True)
    print("=======================================================", flush=True)

    p = subprocess.Popen(
        [exe] + args,
        cwd=cfg.get("cwd", STRATA_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env
    )

    ready = threading.Event()

    def err_reader():
        for line in p.stderr:
            line_s = line.rstrip()
            print(f"[STDERR] {line_s}", flush=True)

    done_event = threading.Event()
    def out_reader():
        for line in p.stdout:
            line_s = line.rstrip()
            if line_s.startswith("T ") or line_s.startswith("DONE ") or line_s.startswith("READY") or line_s.startswith("REUSED") or line_s.startswith("RESUME"):
                print(f"[STDOUT] {line_s}", flush=True)
            if line_s.startswith("READY"):
                ready.set()
            if line_s.startswith("DONE "):
                done_event.set()

    t_err = threading.Thread(target=err_reader, daemon=True)
    t_out = threading.Thread(target=out_reader, daemon=True)
    t_err.start()
    t_out.start()

    if not ready.wait(120):
        print("[!] Engine failed to reach READY within 120s", flush=True)
        p.kill()
        return

    print("\n[+] Engine is READY! Running Prefill Test Cases...", flush=True)

    test_lengths = [50, 250, 1024]
    for tl in test_lengths:
        print(f"\n---> Testing prompt length {tl} tokens...", flush=True)
        tokens = [151644, 872, 198] + [99] * (tl - 4) + [151645]
        gen_line = f"GEN 5 temperature=0.0 " + " ".join(str(t) for t in tokens) + "\n"
        t0 = time.time()
        done_event.clear()
        p.stdin.write(gen_line)
        p.stdin.flush()
        if done_event.wait(180):
            print(f"[+] Finished {tl} tokens in {time.time() - t0:.2f}s", flush=True)
        else:
            print(f"[!] Timed out waiting for {tl} tokens", flush=True)
            break

    print("\nShutting down engine...", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    print("Done!")

if __name__ == "__main__":
    run_diagnostics()
