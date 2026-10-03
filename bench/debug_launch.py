import subprocess
import os
import sys
import time
import json

config_path = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\strata-coder-iq1_m.json"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
exe = os.path.join(STRATA_DIR, "engine", "strata.exe")
raw_args = list(cfg["args"])
# remove --expert-profile and its path
args = []
skip_next = False
for i, a in enumerate(raw_args):
    if skip_next:
        skip_next = False
        continue
    if a == "--expert-cache":
        args.extend(["--expert-cache", "0"])
        skip_next = True
        continue
    args.append(a)
args = ["--serve"] + args
env = os.environ.copy()
env["PATH"] = r"C:\Python312\Lib\site-packages\nvidia\cu13\bin\x86_64;C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\engine;" + env.get("PATH", "")
env["STRATA_ARENA_PIN_GIB"] = "1"
env["STRATA_ARENA_LOCK"] = "0"
env["STRATA_POOL_SPIN_US"] = "50000"

print(f"Launching {exe} with PIN_GIB=1...", flush=True)
p = subprocess.Popen([exe] + args, cwd=cfg["cwd"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env)

t0 = time.time()
while time.time() - t0 < 60:
    ret = p.poll()
    if ret is not None:
        print(f"Process exited with code {ret} after {time.time()-t0:.1f}s", flush=True)
        break
    time.sleep(2)
    # Check if there is anything on stdout
    # p.stdout might block if read directly without select or thread, so let's keep waiting or check ret

print(f"Process poll: {p.poll()}")
# Read all remaining output
err_out = p.stderr.read()
print("STDERR:\n", err_out)

if p.poll() is None:
    print("Process is still alive! Reading one line of stdout...")
    p.kill()
