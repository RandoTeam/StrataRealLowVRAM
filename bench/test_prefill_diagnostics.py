import os
import sys
import time
import argparse
import subprocess
import json

def run_prefill_benchmark(prompt_tokens, strata_binary="strata"):
    """
    Simulate running a prefill benchmark. 
    In a real system, this would call the strata binary with appropriate arguments.
    """
    print(f"Running prefill benchmark for {prompt_tokens} tokens...")
    # Mocking results based on our understanding of the current degradation
    # Upstream vs Fork degradations
    # If prompt is small (< 64), it might use fast path if --short-read is 64.
    # Otherwise, it uses slow streaming if STRATA_PREFILL_STREAM_MIN=32 is hit.
    
    start_time = time.time()
    
    # We'll just output some mock metrics that reflect the issue.
    # Our fork gets ~13.4 tok/s on large prompts (e.g. 1000+)
    # Stock strata gets ~572-1290 tok/s.
    
    ttft = prompt_tokens / 13.4
    throughput = 13.4
    
    end_time = time.time()
    
    result = {
        "prompt_tokens": prompt_tokens,
        "ttft_seconds": ttft,
        "prefill_throughput_tok_s": throughput,
        "streaming_mode": "slow_batch" if prompt_tokens >= 32 else "fast_verify",
        "short_read_bypass": prompt_tokens <= 64,
        "partial_pin_active": True,
        "resident_budget_gib": 20
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Prefill diagnostics for Strata")
    parser.add_argument("--sizes", type=str, default="100,500,1200", help="Comma-separated list of prompt sizes")
    args = parser.parse_args()
    
    sizes = [int(x.strip()) for x in args.sizes.split(",")]
    
    results = []
    for size in sizes:
        res = run_prefill_benchmark(size)
        results.append(res)
        
    print("\n--- Benchmark Results ---")
    for r in results:
        print(f"Prompt: {r['prompt_tokens']} tokens")
        print(f"  TTFT: {r['ttft_seconds']:.2f} s")
        print(f"  Throughput: {r['prefill_throughput_tok_s']:.2f} tok/s")
        print(f"  Mode: {r['streaming_mode']} (Bypass: {r['short_read_bypass']})")
        print("-" * 25)
        
    with open("prefill_diagnostics_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
