#!/usr/bin/env python3
"""
Systematic A/B benchmark sweep for Strata engine parameter tuning.
Launches the server with a config, runs a benchmark, captures results,
and shuts down. Repeats for each parameter variant.

Usage:
    python bench/bench_sweep.py --config strata-q2_0.json --tokens 64 --runs 2
"""
import argparse, copy, json, os, signal, subprocess, sys, time, threading, re, socket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SERVER_SCRIPT = os.path.join(ROOT, "serve", "server.py")


def wait_for_server(port, timeout=180):
    """Wait until the server is accepting HTTP connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            time.sleep(2)
    return False


def run_benchmark(port, tokens=64, prompt="Write a comprehensive overview of the history of space exploration, detailing the key missions from Sputnik and Apollo to the James Webb Space Telescope."):
    """Send a benchmark request and capture tok/s."""
    import urllib.request, urllib.error
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0.0,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "tokens": 0, "elapsed": 0, "tok_s": 0}
    elapsed = time.time() - t0
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", tokens)
    return {
        "tokens": completion_tokens,
        "elapsed": round(elapsed, 2),
        "tok_s": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0
    }


def extract_log_stats(log_path):
    """Parse the stats from all generation lines in the engine log."""
    stats = {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # Accumulate all generation segments
        total_gen_tokens = 0
        total_gen_ms = 0
        drafts_acc = 0
        drafts_tot = 0
        last_hit_rate = None

        for line in lines:
            if "generated in" in line and "tok/s" in line:
                m = re.search(r"(\d+) generated in (\d+) ms \((\d+\.?\d*) tok/s\)", line)
                if m:
                    total_gen_tokens += int(m.group(1))
                    total_gen_ms += int(m.group(2))
                m2 = re.search(r"drafts accepted (\d+) of (\d+)", line)
                if m2:
                    drafts_acc += int(m2.group(1))
                    drafts_tot += int(m2.group(2))
            if "decode expert cache hit rate:" in line:
                m = re.search(r"hit rate: (\d+\.?\d*)%", line)
                if m:
                    last_hit_rate = float(m.group(1))

        if total_gen_tokens > 0:
            stats["gen_tokens"] = total_gen_tokens
            stats["gen_ms"] = total_gen_ms
            stats["gen_tok_s"] = round(total_gen_tokens / (total_gen_ms / 1000.0), 2) if total_gen_ms > 0 else 0.0
        if drafts_tot > 0:
            stats["drafts_accepted"] = drafts_acc
            stats["drafts_total"] = drafts_tot
            stats["draft_rate"] = round(drafts_acc / drafts_tot * 100, 1)
        if last_hit_rate is not None:
            stats["cache_hit_rate"] = last_hit_rate
    except Exception:
        pass
    return stats


def run_variant(config_path, overrides, port, tokens, log_path, warmup=True):
    """Start server with overrides, benchmark, stop server."""
    # Load and modify config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Apply overrides
    for key, value in overrides.items():
        if key.startswith("env:"):
            env_key = key[4:]
            if "env" not in cfg:
                cfg["env"] = {}
            cfg["env"][env_key] = str(value)
        elif key.startswith("arg:"):
            arg_name = key[4:]
            # Find and replace in args list
            args = cfg.get("args", [])
            found = False
            for i, a in enumerate(args):
                if a == arg_name and i + 1 < len(args):
                    args[i + 1] = str(value)
                    found = True
                    break
                elif a == arg_name and isinstance(value, bool):
                    found = True
                    break
            if not found:
                if isinstance(value, bool) and value:
                    args.append(arg_name)
                else:
                    args.extend([arg_name, str(value)])
            cfg["args"] = args
        elif key == "reasoning_budget_tokens":
            cfg["reasoning_budget_tokens"] = value

    # Write temp config
    tmp_config = config_path.replace(".json", "-bench-tmp.json")
    cfg["log"] = log_path
    cfg["port"] = port
    with open(tmp_config, "w") as f:
        json.dump(cfg, f, indent=1)

    # Clear log
    with open(log_path, "w") as f:
        f.write("")

    # Start server
    cmd = [VENV_PYTHON, SERVER_SCRIPT, "--engine", "strata", "--config", tmp_config, "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    
    try:
        print(f"  Waiting for server on port {port}...", flush=True)
        if not wait_for_server(port, timeout=180):
            print(f"  ERROR: Server did not start in 180s", flush=True)
            return {"error": "timeout"}

        if warmup:
            # Warmup run (discard)
            print(f"  Warmup run...", flush=True)
            run_benchmark(port, tokens=16)
            time.sleep(2)
            # Re-clear log after warmup so only benchmark request stats are collected
            with open(log_path, "w") as f:
                f.write("")

        # Actual benchmark
        print(f"  Benchmark run ({tokens} tokens)...", flush=True)
        result = run_benchmark(port, tokens=tokens)
        log_stats = extract_log_stats(log_path)
        result.update(log_stats)
        return result

    finally:
        # Kill server
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # Cleanup
        try:
            os.remove(tmp_config)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Strata parameter sweep benchmark")
    parser.add_argument("--config", required=True, help="Base config JSON path")
    parser.add_argument("--tokens", type=int, default=64, help="Tokens to generate per run")
    parser.add_argument("--port", type=int, default=8090, help="Port for benchmark server")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated variant names or 'all'")
    args = parser.parse_args()

    config_path = os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config
    log_path = os.path.join(ROOT, "bench", "bench-sweep.log")

    # Define test variants
    variants = {
        "baseline": {},

        # Expert cache size
        "expert-cache-550": {"arg:--expert-cache": "550"},
        "expert-cache-600": {"arg:--expert-cache": "600"},

        # Pool spin timing
        "spin-20k": {"env:STRATA_POOL_SPIN_US": "20000"},
        "spin-60k": {"env:STRATA_POOL_SPIN_US": "60000"},
        "spin-80k": {"env:STRATA_POOL_SPIN_US": "80000"},

        # Pool workers
        "workers-4": {"arg:--pool-workers": "4"},
        "workers-6": {"arg:--pool-workers": "6"},

        # Adaptive swaps - more frequent
        "adapt-4-8": {"arg:--adapt-every": "4", "arg:--adapt-swaps": "8"},
        "adapt-4-4": {"arg:--adapt-every": "4", "arg:--adapt-swaps": "4"},
        "adapt-2-4": {"arg:--adapt-every": "2", "arg:--adapt-swaps": "4"},

        # Speculative decoding
        "spec-6": {"arg:--spec": "6", "arg:--spec-min-p": "0.70"},
        "spec-10": {"arg:--spec": "10", "arg:--spec-min-p": "0.60"},
        "spec-12": {"arg:--spec": "12", "arg:--spec-min-p": "0.55"},

        # IQ kernel tuning
        "iq-mt-min-1": {"env:STRATA_IQ_MT_MIN": "1"},
        "iq-kq256": {"env:STRATA_KQ256": "1"},

        # Prefill chunk
        "prefill-512": {"arg:--prefill": "512"},
        "prefill-2048": {"arg:--prefill": "2048"},

        # Short read
        "short-read-8": {"arg:--short-read": "8"},
        "short-read-32": {"arg:--short-read": "32"},

        # KV resident
        "kv-resident-4096": {"arg:--kv-resident": "4096"},
        "kv-resident-16384": {"arg:--kv-resident": "16384"},

        # Phase 2 specific combinations
        "p2-baseline": {},
        "p2-cache600-adapt": {
            "arg:--expert-cache": "600",
            "arg:--adapt-every": "4",
            "arg:--adapt-swaps": "8",
        },
        "p2-spec10": {
            "arg:--expert-cache": "600",
            "arg:--adapt-every": "4",
            "arg:--adapt-swaps": "8",
            "arg:--spec": "10",
            "arg:--spec-min-p": "0.60",
        },
        "p2-spec6": {
            "arg:--expert-cache": "600",
            "arg:--adapt-every": "4",
            "arg:--adapt-swaps": "8",
            "arg:--spec": "6",
            "arg:--spec-min-p": "0.70",
        },
        "p2-mtmin1": {
            "arg:--expert-cache": "600",
            "arg:--adapt-every": "4",
            "arg:--adapt-swaps": "8",
            "env:STRATA_IQ_MT_MIN": "1",
        },
        # Iteration 1: MTP Max-T Scaling
        "i1-mtp-t4": {},  # uses strata-q2_0.json current settings (cache 650, adapt 4/8, mtp-max-t 4)
        "i1-mtp-t6": {
            "arg:--mtp-max-t": "6",
        },
        "i1-mtp-t8": {
            "arg:--mtp-max-t": "8",
        },
        "i1-mtp-t8-p55": {
            "arg:--mtp-max-t": "8",
            "arg:--spec-min-p": "0.55",
        },
        "i1-mtp-t8-p70": {
            "arg:--mtp-max-t": "8",
            "arg:--spec-min-p": "0.70",
        },
    }

    # Filter variants
    if args.variants != "all":
        selected = [v.strip() for v in args.variants.split(",")]
        if "baseline" not in selected and "p2-baseline" not in selected and "i1-mtp-t4" not in selected:
            selected.insert(0, "baseline")
        variants = {k: v for k, v in variants.items() if k in selected}

    results = {}
    print(f"\n{'='*60}")
    print(f"Strata Parameter Sweep - {len(variants)} variants")
    print(f"Config: {args.config}")
    print(f"Tokens: {args.tokens}, Port: {args.port}")
    print(f"{'='*60}\n")

    for name, overrides in variants.items():
        print(f"\n--- Variant: {name} ---")
        if overrides:
            print(f"  Overrides: {json.dumps(overrides, indent=4)}")
        else:
            print(f"  (no overrides - baseline)")
        
        var_log_path = os.path.join(ROOT, "bench", f"bench-{name}.log")
        result = run_variant(
            config_path, overrides, args.port, args.tokens, var_log_path,
            warmup=not args.no_warmup
        )
        results[name] = result
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Result: {result.get('gen_tok_s', result.get('tok_s', '?'))} tok/s "
                  f"({result.get('gen_tokens', result.get('tokens', '?'))} tokens)")
            if "cache_hit_rate" in result:
                print(f"  Cache hit rate: {result['cache_hit_rate']}%")
            if "draft_rate" in result:
                print(f"  Draft accept rate: {result['draft_rate']}%")
        
        # Cool-down between variants
        print(f"  Cooling down...", flush=True)
        time.sleep(5)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<25} {'tok/s':>8} {'tokens':>8} {'draft%':>8} {'cache%':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    baseline_tps = None
    for name, r in results.items():
        tps = r.get("gen_tok_s", r.get("tok_s", 0))
        if name in ("baseline", "p2-baseline", "i1-mtp-t4") and baseline_tps is None:
            baseline_tps = tps
        tokens = r.get("gen_tokens", r.get("tokens", "?"))
        draft = r.get("draft_rate", "?")
        cache = r.get("cache_hit_rate", "?")
        delta = ""
        if baseline_tps and tps and baseline_tps > 0 and name not in ("baseline", "p2-baseline", "i1-mtp-t4"):
            pct = (tps - baseline_tps) / baseline_tps * 100
            delta = f" ({pct:+.1f}%)"
        print(f"{name:<25} {str(tps) + delta:>8} {str(tokens):>8} {str(draft):>8} {str(cache):>8}")

    # Save results
    out_path = os.path.join(ROOT, "bench", "sweep-results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
