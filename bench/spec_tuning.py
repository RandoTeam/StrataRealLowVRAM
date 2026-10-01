"""bench/spec_tuning.py - Systematic Speculative Decoding & MTP Tuning Matrix

Evaluates combinations of speculative decoding parameters:
- --spec (draft count: 3, 4, 5, 6)
- --spec-min-p (acceptance threshold: 0.75, 0.80, 0.84, 0.88)
- --suffix-draft (ngram suffix draft: 0, 3, 4)
- --mtp-max-t (MTP steps: 2, 3)
"""
import argparse
import json
import os
import subprocess
import sys
import time

STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
CONFIG_FILE = os.path.join(STRATA_DIR, "strata-coder-iq1_m.json")

def run_single_config(spec, min_p, suffix, mtp_t, max_tokens=128):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exe = cfg["exe"]
    args = list(cfg["args"])

    # Override parameters
    def set_arg(flag, val):
        if flag in args:
            idx = args.index(flag)
            args[idx + 1] = str(val)
        else:
            args.extend([flag, str(val)])

    set_arg("--spec", spec)
    set_arg("--spec-min-p", min_p)
    set_arg("--suffix-draft", suffix)
    set_arg("--mtp-max-t", mtp_t)

    env = os.environ.copy()
    env["PATH"] = r"C:\Python312\Lib\site-packages\nvidia\cu13\bin\x86_64;C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\engine;" + env.get("PATH", "")
    env.update(cfg.get("env", {}))
    env["STRATA_ARENA_PIN_GIB"] = "1"

    print(f"\n[Bench] Testing --spec {spec} --spec-min-p {min_p} --suffix-draft {suffix} --mtp-max-t {mtp_t}...", flush=True)

    cmd = [exe, "--serve"] + args
    p = subprocess.Popen(
        cmd,
        cwd=cfg["cwd"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env
    )

    ready = False
    stats = None
    t0 = time.time()

    try:
        # Wait for READY
        while time.time() - t0 < 45:
            line = p.stdout.readline()
            if not line:
                if p.poll() is not None:
                    break
                continue
            line = line.strip()
            if "READY" in line:
                ready = True
                print("  Engine READY!", flush=True)
                break

        if not ready:
            print("  Engine failed to reach READY", flush=True)
            p.kill()
            return None

        # Send test prompt (Python Fibonacci test prompt)
        test_prompt_ids = "151644,872,198,2610,525,264,10933,10156,311,9113,198,151645,198,151644,77091,198"
        gen_cmd = f"GEN {max_tokens} temperature=0 {test_prompt_ids}\n"
        p.stdin.write(gen_cmd)
        p.stdin.flush()

        t_gen_start = time.time()
        tokens = []

        while time.time() - t_gen_start < 60:
            line = p.stdout.readline()
            if not line:
                if p.poll() is not None:
                    break
                continue
            line = line.strip()
            if line.starts_with("T "):
                tokens.append(int(line.split()[1]))
            elif line.starts_with("DONE"):
                parts = line.split()
                # DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> [hits] [lookups]
                stats = {
                    "spec": spec,
                    "spec_min_p": min_p,
                    "suffix_draft": suffix,
                    "mtp_max_t": mtp_t,
                    "generated": int(parts[1]),
                    "prompt_tokens": int(parts[2]),
                    "prompt_ms": float(parts[3]),
                    "decode_ms": float(parts[4]),
                    "finish_reason": parts[5],
                    "drafts_accepted": int(parts[6]) if len(parts) > 6 else 0,
                    "drafts_offered": int(parts[7]) if len(parts) > 7 else 0,
                    "reused": int(parts[8]) if len(parts) > 8 else 0,
                    "hits": int(parts[9]) if len(parts) > 9 else 0,
                    "lookups": int(parts[10]) if len(parts) > 10 else 0,
                }
                stats["tok_s"] = stats["generated"] / (stats["decode_ms"] / 1000.0) if stats["decode_ms"] > 0 else 0
                stats["accept_rate"] = stats["drafts_accepted"] / stats["drafts_offered"] if stats["drafts_offered"] > 0 else 0
                print(f"  Result: {stats['tok_s']:.2f} tok/s | Acceptance: {stats['accept_rate']*100:.1f}% ({stats['drafts_accepted']}/{stats['drafts_offered']})", flush=True)
                break
    finally:
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
        except Exception:
            pass
        time.sleep(1)
        if p.poll() is None:
            p.kill()
            p.wait()

    return stats

def main():
    parser = argparse.ArgumentParser(description="Speculative decode tuning")
    parser.add_argument("--output", default="bench/results/spec_tuning_results.json")
    args = parser.parse_args()

    matrix = [
        # (spec, min_p, suffix, mtp_t)
        (4, 0.84, 4, 3),
        (5, 0.84, 4, 3),
        (6, 0.84, 4, 3),
        (6, 0.80, 4, 3),
        (6, 0.88, 4, 3),
        (6, 0.84, 0, 3),
        (6, 0.84, 3, 2),
    ]

    results = []
    for spec, min_p, suffix, mtp_t in matrix:
        res = run_single_config(spec, min_p, suffix, mtp_t)
        if res:
            results.append(res)
        time.sleep(5)  # Cooldown between runs

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTuning complete. Results written to {args.output}")

if __name__ == "__main__":
    main()
