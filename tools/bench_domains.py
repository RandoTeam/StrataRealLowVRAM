"""tools/bench_domains.py - Multi-domain benchmarking harness for Strata.

Measures prefill throughput, base autoregressive decode throughput,
and speculative draft gains across 4 standardized domains:
  1. Mathematics / Exact Sciences (Russian)
  2. Programming / Algorithms (Python)
  3. Logic & Multi-Step Reasoning (Russian)
  4. Physics & Fundamental Science (English)

Supports two execution modes:
  - Direct Engine mode (default): starts StrataEngine directly, matching calibrate.py methodology.
  - HTTP mode: queries a running Strata API server via /v1/chat/completions.

Usage:
    python tools/bench_domains.py [--config strata-q2_0.json] [--out results.json]
    python tools/bench_domains.py --url http://127.0.0.1:8080 [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST
from serve.server import StrataEngine, child_env

DOMAINS = [
    {
        "id": "math",
        "name": "Mathematics / Exact Sciences",
        "prompt": "Что такое число Пи, какова история его открытия и почему оно является трансцендентным и иррациональным? Назови первые 10 знаков.",
        "max_tokens": 128,
    },
    {
        "id": "code",
        "name": "Programming / Algorithms",
        "prompt": "Напиши на Python класс LRUCache с методами get(key) и put(key, value) на базе collections.OrderedDict, добавь type hints и docstrings.",
        "max_tokens": 160,
    },
    {
        "id": "logic",
        "name": "Logic & Reasoning",
        "prompt": "Задача на логику: У фермера есть волк, коза и капуста. Ему нужно перевезти их через реку в лодке, вмещающей только его и один объект. Коза ест капусту, волк ест козу. Опиши по шагам безопасный план переправы.",
        "max_tokens": 128,
    },
    {
        "id": "physics",
        "name": "Physics / Fundamental Science",
        "prompt": "Explain the fundamental differences between nuclear fission and nuclear fusion in two paragraphs, highlighting the mass-energy defect and Coulomb barrier.",
        "max_tokens": 140,
    },
]


def load_tokenizer(tpath: Path) -> ST.Tokenizer:
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(
        toks,
        (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
        json.loads((tpath / "token_type.json").read_text()),
    )


def compute_metrics(domain_def: dict, prompt_n: int, prompt_ms: float, predicted_n: int,
                    decode_ms: float, draft_n: int, draft_acc: int, wall_s: float, text: str) -> dict:
    prefill_tok_s = round(prompt_n / (prompt_ms / 1000.0), 2) if prompt_ms > 0 else 0.0
    total_decode_tok_s = round(predicted_n / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0
    draft_acc_pct = round((draft_acc / draft_n * 100.0), 1) if draft_n > 0 else 0.0
    base_steps = max(1, predicted_n - draft_acc)
    base_decode_tok_s = round(base_steps / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0
    spec_gain_pct = round((total_decode_tok_s / base_decode_tok_s - 1.0) * 100.0, 1) if base_decode_tok_s > 0 else 0.0

    return {
        "id": domain_def["id"],
        "name": domain_def["name"],
        "prompt_tokens": prompt_n,
        "prefill_ms": round(prompt_ms, 1),
        "prefill_tok_s": prefill_tok_s,
        "generated_tokens": predicted_n,
        "decode_ms": round(decode_ms, 1),
        "base_steps": base_steps,
        "base_decode_tok_s": base_decode_tok_s,
        "total_decode_tok_s": total_decode_tok_s,
        "draft_offered": draft_n,
        "draft_accepted": draft_acc,
        "draft_acc_pct": draft_acc_pct,
        "spec_gain_pct": spec_gain_pct,
        "wall_s": round(wall_s, 2),
        "output_chars": len(text),
    }


def run_direct_benchmark(cfg: dict, args_override: list[str] | None = None, tune: dict | None = None) -> dict:
    tpath = Path(cfg["tokenizer"])
    tok = load_tokenizer(tpath)
    engine_args = list(args_override if args_override is not None else cfg["args"])

    print(f"\n{'='*80}")
    print(f"Direct StrataEngine Benchmark: Context={cfg.get('max_context', 32768)}")
    print(f"Engine Args: {' '.join(engine_args)}")
    if tune:
        print(f"Per-request Tuning: {tune}")
    print(f"{'='*80}\n")

    eng = StrataEngine(cfg["exe"], engine_args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    print(f"Engine READY! Reported max_context: {eng.max_context}\n")

    results = []
    try:
        for d in DOMAINS:
            prompt_text = f"<|im_start|>user\n{d['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ids = tok.encode(prompt_text, parse_special=True)
            sampling = {"temperature": 0.6, "top_p": 0.8}
            if tune:
                sampling["strata_tune"] = tune

            print(f"Running [{d['id']}] {d['name']}...", flush=True)
            t0 = time.time()
            tokens = []
            for t in eng.generate(ids, d["max_tokens"], sampling, threading.Event()):
                if t is not None:
                    tokens.append(t)
            wall_s = time.time() - t0

            last = dict(getattr(eng, "last", {}) or {})
            prompt_n = int(last.get("prompt_tokens", len(ids)))
            prompt_ms = float(last.get("prompt_ms", 0.0))
            predicted_n = int(last.get("generated", len(tokens)))
            decode_ms = float(last.get("decode_ms", 0.0))
            draft_n = int(last.get("drafts_offered", 0))
            draft_acc = int(last.get("drafts_accepted", 0))
            text = tok.decode(tokens)

            row = compute_metrics(d, prompt_n, prompt_ms, predicted_n, decode_ms, draft_n, draft_acc, wall_s, text)
            results.append(row)

            print(f"  -> Prefill: {row['prefill_tok_s']} tok/s ({prompt_n} tokens in {prompt_ms:.0f}ms)")
            print(f"  -> Base Decode: {row['base_decode_tok_s']} tok/s | Total Decode: {row['total_decode_tok_s']} tok/s (Gain: +{row['spec_gain_pct']}%)")
            print(f"  -> Drafts: {draft_acc}/{draft_n} accepted ({row['draft_acc_pct']}%) | Wall: {wall_s:.1f}s\n")
    finally:
        try:
            eng.proc.stdin.write("QUIT\n")
            eng.proc.stdin.flush()
            eng.proc.stdin.close()
            eng.proc.wait(15)
        except Exception:
            pass

    return build_summary(cfg.get("model_name", "qwen3.8-flash-next-q2_0"), eng.max_context, results, engine_args)


def run_http_benchmark(url: str, timeout: float = 180.0) -> dict:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10) as r:
            health = json.loads(r.read())
            model_name = health.get("model", "qwen3.8-flash-next-q2_0")
            max_context = health.get("max_context", 32768)
    except Exception as e:
        raise RuntimeError(f"Cannot connect to Strata server at {url}: {e}")

    print(f"\n{'='*80}")
    print(f"HTTP Strata Benchmark: Model={model_name} | Max Context={max_context}")
    print(f"{'='*80}\n")

    results = []
    for d in DOMAINS:
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": d["prompt"]}],
            "max_tokens": d["max_tokens"],
            "temperature": 0.6,
            "top_p": 0.8,
            "reasoning_budget_tokens": 32,
        }
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        print(f"Running [{d['id']}] {d['name']}...", flush=True)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        wall_s = time.time() - t0
        timings = resp.get("timings", {})
        usage = resp.get("usage", {})
        choice = resp.get("choices", [{}])[0].get("message", {})

        prompt_n = timings.get("prompt_n", usage.get("prompt_tokens", 0))
        prompt_ms = timings.get("prompt_ms", 0.0)
        predicted_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
        decode_ms = timings.get("predicted_ms", 0.0)
        draft_n = timings.get("draft_n", 0)
        draft_acc = timings.get("draft_n_accepted", 0)
        text = (choice.get("reasoning_content") or "") + (choice.get("content") or "")

        row = compute_metrics(d, prompt_n, prompt_ms, predicted_n, decode_ms, draft_n, draft_acc, wall_s, text)
        results.append(row)

        print(f"  -> Prefill: {row['prefill_tok_s']} tok/s ({prompt_n} tokens in {prompt_ms:.0f}ms)")
        print(f"  -> Base Decode: {row['base_decode_tok_s']} tok/s | Total Decode: {row['total_decode_tok_s']} tok/s (Gain: +{row['spec_gain_pct']}%)")
        print(f"  -> Drafts: {draft_acc}/{draft_n} accepted ({row['draft_acc_pct']}%) | Wall: {wall_s:.1f}s\n")

    return build_summary(model_name, max_context, results, ["--serve"])


def build_summary(model: str, max_context: int, results: list[dict], args: list[str]) -> dict:
    if not results:
        return {"error": "All benchmark runs failed"}

    summary = {
        "model": model,
        "max_context": max_context,
        "engine_args": args,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "mean_prefill_tok_s": round(statistics.mean(r["prefill_tok_s"] for r in results), 2),
        "mean_base_decode_tok_s": round(statistics.mean(r["base_decode_tok_s"] for r in results), 2),
        "mean_total_decode_tok_s": round(statistics.mean(r["total_decode_tok_s"] for r in results), 2),
        "median_total_decode_tok_s": round(statistics.median(r["total_decode_tok_s"] for r in results), 2),
        "mean_draft_acc_pct": round(statistics.mean(r["draft_acc_pct"] for r in results), 1),
        "mean_spec_gain_pct": round(statistics.mean(r["spec_gain_pct"] for r in results), 1),
    }

    print(f"{'='*80}")
    print("BENCHMARK SUMMARY:")
    print(f"  Mean Prefill:       {summary['mean_prefill_tok_s']} tok/s")
    print(f"  Mean Base Decode:   {summary['mean_base_decode_tok_s']} tok/s")
    print(f"  Mean Total Decode:  {summary['mean_total_decode_tok_s']} tok/s  (Median: {summary['median_total_decode_tok_s']} tok/s)")
    print(f"  Mean Draft Accept:  {summary['mean_draft_acc_pct']}%")
    print(f"  Speculative Gain:  +{summary['mean_spec_gain_pct']}%")
    print(f"{'='*80}\n")
    return summary


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, str(value)]
    return out


def main():
    parser = argparse.ArgumentParser(description="Strata 4-Domain Benchmark")
    parser.add_argument("--config", default="strata-q2_0.json", help="Path to config file")
    parser.add_argument("--url", help="Strata API server URL (if provided, uses HTTP mode instead of direct engine)")
    parser.add_argument("--out", help="Path to write JSON output")
    parser.add_argument("--spec", help="Override --spec value in engine args")
    parser.add_argument("--spec-min-p", help="Override --spec-min-p value")
    parser.add_argument("--pool-workers", help="Override --pool-workers value")
    parser.add_argument("--pcie-frac", help="Override --pcie-frac value")
    parser.add_argument("--suffix-draft", help="Override --suffix-draft value")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    if args.url:
        summary = run_http_benchmark(args.url)
    else:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        engine_args = list(cfg["args"])
        if args.spec is not None:
            engine_args = with_arg(engine_args, "--spec", args.spec)
        if args.suffix_draft is not None:
            engine_args = with_arg(engine_args, "--suffix-draft", args.suffix_draft)
        if args.spec_min_p is not None:
            engine_args = with_arg(engine_args, "--spec-min-p", args.spec_min_p)
        if args.pool_workers is not None:
            engine_args = with_arg(engine_args, "--pool-workers", args.pool_workers)
        if args.pcie_frac is not None:
            engine_args = with_arg(engine_args, "--pcie-frac", args.pcie_frac)
        summary = run_direct_benchmark(cfg, args_override=engine_args)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
