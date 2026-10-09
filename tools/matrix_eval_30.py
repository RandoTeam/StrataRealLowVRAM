"""tools/matrix_eval_30.py - Automated 30-Iteration Quality & Performance Matrix Harness.

Measures:
  - Time To First Token (TTFT in ms)
  - Prefill throughput (tokens/second)
  - Autoregressive decode throughput (tokens/second)
  - Output coherence & reasoning loop detection across 30 iterations (Math, Code, Logic, Physics in RU and EN)
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 30 Curated Prompts across Russian and English in 4 Core Domains
PROMPTS_30 = [
    # Mathematics (RU)
    {"id": "math_ru_1", "domain": "Math", "lang": "RU", "prompt": "Реши уравнение 3*x^2 - 12*x + 9 = 0. Найди дискриминант и корни x1, x2.", "max_tokens": 120},
    {"id": "math_ru_2", "domain": "Math", "lang": "RU", "prompt": "Найди производную функции f(x) = x^3 * sin(x). Примени правило произведения.", "max_tokens": 120},
    {"id": "math_ru_3", "domain": "Math", "lang": "RU", "prompt": "Вычисли определенный интеграл от 2*x от 0 до 4.", "max_tokens": 100},
    {"id": "math_ru_4", "domain": "Math", "lang": "RU", "prompt": "Какова вероятность выпадения двух шестерок при броске двух игральных костей?", "max_tokens": 100},
    # Mathematics (EN)
    {"id": "math_en_1", "domain": "Math", "lang": "EN", "prompt": "Evaluate the limit as x approaches 0 of sin(x)/x using L'Hopital's rule or Taylor expansion.", "max_tokens": 120},
    {"id": "math_en_2", "domain": "Math", "lang": "EN", "prompt": "Find the eigenvalues of the 2x2 matrix [[2, 1], [1, 2]].", "max_tokens": 120},
    {"id": "math_en_3", "domain": "Math", "lang": "EN", "prompt": "Compute the sum of the infinite geometric series 1 + 1/2 + 1/4 + 1/8 + ...", "max_tokens": 100},
    {"id": "math_en_4", "domain": "Math", "lang": "EN", "prompt": "State Bayes' theorem and define prior, likelihood, and posterior probability.", "max_tokens": 120},
    # Programming & Algorithms (RU)
    {"id": "code_ru_1", "domain": "Code", "lang": "RU", "prompt": "Напиши функцию на Python для бинарного поиска в отсортированном списке с типизацией.", "max_tokens": 140},
    {"id": "code_ru_2", "domain": "Code", "lang": "RU", "prompt": "Реализуй функцию на Python для разворота односвязного списка.", "max_tokens": 140},
    {"id": "code_ru_3", "domain": "Code", "lang": "RU", "prompt": "Как работает алгоритм быстрой сортировки (Quicksort)? Назови среднюю и худшую сложность.", "max_tokens": 120},
    {"id": "code_ru_4", "domain": "Code", "lang": "RU", "prompt": "Напиши генератор Фибоначчи на Python с использованием yield.", "max_tokens": 100},
    # Programming & Algorithms (EN)
    {"id": "code_en_1", "domain": "Code", "lang": "EN", "prompt": "Write a clean Python function to find the maximum subarray sum (Kadane's algorithm).", "max_tokens": 140},
    {"id": "code_en_2", "domain": "Code", "lang": "EN", "prompt": "Explain the difference between a mutex and a semaphore with a code example in C++.", "max_tokens": 140},
    {"id": "code_en_3", "domain": "Code", "lang": "EN", "prompt": "Implement a function in Python that checks if a string has balanced parentheses using a stack.", "max_tokens": 140},
    {"id": "code_en_4", "domain": "Code", "lang": "EN", "prompt": "Explain time and space complexity of Dijkstra's algorithm with a min-heap priority queue.", "max_tokens": 120},
    # Logic & Reasoning (RU)
    {"id": "logic_ru_1", "domain": "Logic", "lang": "RU", "prompt": "У фермера есть волк, коза и капуста. Как перевезти их через реку в лодке с одним пассажиром?", "max_tokens": 140},
    {"id": "logic_ru_2", "domain": "Logic", "lang": "RU", "prompt": "Один брат всегда говорит правду, другой всегда врет. Какой один вопрос задать на развилке?", "max_tokens": 140},
    {"id": "logic_ru_3", "domain": "Logic", "lang": "RU", "prompt": "В комнате 3 выключателя и в закрытой комнате 3 лампочки. Как за один вход определить соответствие?", "max_tokens": 140},
    {"id": "logic_ru_4", "domain": "Logic", "lang": "RU", "prompt": "Если все розы — цветы, а некоторые цветы быстро вянут, следует ли, что некоторые розы быстро вянут?", "max_tokens": 120},
    # Logic & Reasoning (EN)
    {"id": "logic_en_1", "domain": "Logic", "lang": "EN", "prompt": "Three boxes: Apples, Oranges, Mixed. All labeled incorrectly. You pick one fruit from one box. How to fix labels?", "max_tokens": 140},
    {"id": "logic_en_2", "domain": "Logic", "lang": "EN", "prompt": "Analyze: 'If it rains, the grass is wet. The grass is wet. Therefore it rained.' Is this deduction valid?", "max_tokens": 120},
    {"id": "logic_en_3", "domain": "Logic", "lang": "EN", "prompt": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Show math.", "max_tokens": 100},
    # Physics & Science (RU)
    {"id": "phys_ru_1", "domain": "Physics", "lang": "RU", "prompt": "Сформулируй три закона Ньютона и напиши математическую формулу для второго закона.", "max_tokens": 140},
    {"id": "phys_ru_2", "domain": "Physics", "lang": "RU", "prompt": "В чем физический смысл уравнения Эйнштейна E = mc^2? Объясни дефект массы.", "max_tokens": 140},
    {"id": "phys_ru_3", "domain": "Physics", "lang": "RU", "prompt": "Объясни явление фотоэффекта и почему классическая волновая теория не могла его объяснить.", "max_tokens": 140},
    # Physics & Science (EN)
    {"id": "phys_en_1", "domain": "Physics", "lang": "EN", "prompt": "Explain the four Maxwell equations and their physical meaning in electromagnetism.", "max_tokens": 150},
    {"id": "phys_en_2", "domain": "Physics", "lang": "EN", "prompt": "Describe the Carnot cycle and why its efficiency represents the theoretical maximum.", "max_tokens": 140},
    {"id": "phys_en_3", "domain": "Physics", "lang": "EN", "prompt": "What is the Pauli exclusion principle and how does it explain atomic electron shells?", "max_tokens": 140},
    {"id": "phys_en_4", "domain": "Physics", "lang": "EN", "prompt": "Explain quantum tunneling and how alpha decay occurs through the Coulomb barrier.", "max_tokens": 140},
]


def check_for_loops_and_gibberish(text: str) -> tuple[bool, str]:
    """Inspects text for repetitive loops, formula noise (e.g. b^*(-b)*b), or emptiness."""
    if not text.strip():
        return False, "EMPTY_OUTPUT"
    # Check for corruption patterns
    for bad in ("b^*(-b)", "+C/C+C/C", "m[ant]l[ant]", "b*(-b)"):
        if bad in text:
            return False, f"CORRUPTION_PATTERN_{bad}"
    # Check n-gram repetition
    words = text.split()
    if len(words) >= 12:
        for n in (3, 4):
            ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            counts: dict[tuple, int] = {}
            for ng in ngrams:
                counts[ng] = counts.get(ng, 0) + 1
                if counts[ng] >= 5:
                    return False, f"REPETITIVE_NGRAM_{' '.join(ng)}"
    return True, "CLEAN"


def run_matrix_eval_30(url: str = "http://127.0.0.1:8080", out_path: str = "reports/matrix_eval_30_report.json"):
    print(f"\n{'='*80}")
    print(f"AUTOMATED CLOSED-LOOP MATRIX EVALUATION: 30 ITERATIONS")
    print(f"Target Server: {url}")
    print(f"{'='*80}\n")

    # Verify server availability
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10) as r:
            health = json.loads(r.read())
            model_name = health.get("model", "qwen3.8-flash-next-q2_0")
            max_context = health.get("max_context", 32768)
            print(f"Server Health OK: Model={model_name}, Max Context={max_context}\n")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Strata server at {url}: {e}")
        return

    results = []
    rejected_count = 0

    for i, p in enumerate(PROMPTS_30, 1):
        prompt_id = p["id"]
        domain = p["domain"]
        lang = p["lang"]
        prompt = p["prompt"]
        max_tokens = p["max_tokens"]

        print(f"[{i:02d}/30] [{domain} | {lang}] {prompt_id}: '{prompt[:45]}...'")

        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic decoding for exact quality verification
            "top_p": 0.95,
            "reasoning_budget_tokens": 1024,
        }

        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  FAILED: {e}\n")
            rejected_count += 1
            continue
        wall_s = time.time() - t0

        choice = data.get("choices", [{}])[0].get("message", {})
        timings = data.get("timings", {})
        usage = data.get("usage", {})

        prompt_n = timings.get("prompt_n", usage.get("prompt_tokens", 0))
        prompt_ms = timings.get("prompt_ms", 0.0)
        predicted_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
        decode_ms = timings.get("predicted_ms", 0.0)

        prefill_tok_s = round(prompt_n / (prompt_ms / 1000.0), 2) if prompt_ms > 0 else 0.0
        decode_tok_s = round(predicted_n / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0
        ttft_ms = round(prompt_ms, 1)

        reasoning = choice.get("reasoning_content") or ""
        content = choice.get("content") or ""
        full_text = reasoning + "\n" + content

        is_valid, quality_note = check_for_loops_and_gibberish(full_text)
        if not is_valid:
            rejected_count += 1
            status = f"REJECTED ({quality_note})"
        else:
            status = "PASSED"

        print(f"  -> TTFT: {ttft_ms}ms | Prefill: {prefill_tok_s} tok/s | Decode: {decode_tok_s} tok/s | Gen: {predicted_n} tok | {status}")

        results.append({
            "iteration": i,
            "id": prompt_id,
            "domain": domain,
            "lang": lang,
            "prompt": prompt,
            "prompt_tokens": prompt_n,
            "generated_tokens": predicted_n,
            "ttft_ms": ttft_ms,
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
            "wall_s": round(wall_s, 2),
            "is_valid": is_valid,
            "quality_note": quality_note,
            "content_preview": content[:160] if content else reasoning[:160],
        })

    if not results:
        print("[ERROR] No valid benchmark results collected!")
        return

    summary = {
        "model": model_name,
        "max_context": max_context,
        "total_iterations": len(results),
        "passed_iterations": len(results) - rejected_count,
        "rejected_iterations": rejected_count,
        "quality_pass_rate_pct": round((len(results) - rejected_count) / len(results) * 100.0, 1),
        "mean_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in results), 1),
        "median_ttft_ms": round(statistics.median(r["ttft_ms"] for r in results), 1),
        "mean_prefill_tok_s": round(statistics.mean(r["prefill_tok_s"] for r in results), 2),
        "mean_decode_tok_s": round(statistics.mean(r["decode_tok_s"] for r in results), 2),
        "median_decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in results), 2),
        "max_decode_tok_s": max(r["decode_tok_s"] for r in results),
        "min_decode_tok_s": min(r["decode_tok_s"] for r in results),
        "results": results,
    }

    print(f"\n{'='*80}")
    print("30-ITERATION CLOSED-LOOP EVALUATION SUMMARY:")
    print(f"  Quality Pass Rate:   {summary['quality_pass_rate_pct']}% ({summary['passed_iterations']}/{summary['total_iterations']})")
    print(f"  Mean TTFT:           {summary['mean_ttft_ms']} ms (Median: {summary['median_ttft_ms']} ms)")
    print(f"  Mean Prefill Speed:  {summary['mean_prefill_tok_s']} tok/s")
    print(f"  Mean Decode Speed:   {summary['mean_decode_tok_s']} tok/s (Median: {summary['median_decode_tok_s']} tok/s)")
    print(f"  Speed Range:         {summary['min_decode_tok_s']} - {summary['max_decode_tok_s']} tok/s")
    print(f"{'='*80}\n")

    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to: {p_out.resolve()}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    out = sys.argv[2] if len(sys.argv) > 2 else "reports/matrix_eval_30_report.json"
    run_matrix_eval_30(url, out)
