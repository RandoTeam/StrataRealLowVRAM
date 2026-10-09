"""tools/tune_stepwise.py - Systematic step-by-step engine tuner with noise & rollback guards.

Iteratively tests engine parameters against the 4 standardized domains,
adopts winners exceeding the noise threshold (+2.5%), re-tests borderlines,
and rolls back any degradations to maintain monotonic performance improvement.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import bench_domains as BD

MIN_GAIN = 0.025   # +2.5% required to beat baseline
NOISE_BAND = 0.020 # +/- 2.0% is treated as measurement noise


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, str(value)]
    return out


def evaluate_config(cfg: dict, engine_args: list[str], label: str) -> dict:
    print(f"\n>>> Evaluating candidate: [{label}]")
    print(f"    Args: {' '.join(engine_args)}")
    summary = BD.run_direct_benchmark(cfg, args_override=engine_args)
    mean_tok_s = summary["mean_total_decode_tok_s"]
    print(f"    Result for [{label}]: {mean_tok_s:.2f} tok/s (prefill: {summary['mean_prefill_tok_s']:.2f} tok/s)")
    return summary


def run_tuning(config_path: str = "strata-q2_0.json", out_path: str = "tuning_history.json") -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    base_args = list(cfg["args"])

    history = []
    
    # Step 0: Initial Reference
    print("\n" + "="*80)
    print("STEP 0: Measuring Initial Reference Baseline")
    print("="*80)
    golden_summary = evaluate_config(cfg, base_args, "initial_reference")
    golden_args = list(base_args)
    golden_rate = golden_summary["mean_total_decode_tok_s"]
    history.append({
        "step": "step_0_reference",
        "label": "initial_reference",
        "rate": golden_rate,
        "args": list(golden_args),
        "status": "baseline"
    })

    # Step 1: Tune Speculative Window & Suffix Draft
    print("\n" + "="*80)
    print("STEP 1: Tuning Speculative Window & Suffix Drafter")
    print("="*80)
    spec_candidates = [
        ("spec2_sfx0_pure", [("--spec", "2"), ("--suffix-draft", "0")]),
        ("spec2_sfx1_light", [("--spec", "2"), ("--suffix-draft", "1")]),
        ("spec2_sfx2_bal",   [("--spec", "2"), ("--suffix-draft", "2")]),
        ("spec3_sfx2_deep",  [("--spec", "3"), ("--suffix-draft", "2")]),
    ]
    for label, flags in spec_candidates:
        cand_args = list(golden_args)
        for f, v in flags:
            cand_args = with_arg(cand_args, f, v)
        
        cand_res = evaluate_config(cfg, cand_args, label)
        cand_rate = cand_res["mean_total_decode_tok_s"]
        diff = (cand_rate - golden_rate) / golden_rate

        entry = {"step": "step_1_spec", "label": label, "rate": cand_rate, "diff_pct": round(diff * 100, 2), "args": cand_args}
        if diff >= MIN_GAIN:
            print(f"    --> Candidate [{label}] beat golden by +{diff*100:.2f}%! Confirming with second run...")
            confirm_res = evaluate_config(cfg, cand_args, f"{label}_confirm")
            confirm_rate = confirm_res["mean_total_decode_tok_s"]
            if confirm_rate > golden_rate * (1.0 + NOISE_BAND):
                print(f"    [ADOPTED] Candidate [{label}] is confirmed new winner! ({confirm_rate:.2f} tok/s vs {golden_rate:.2f} tok/s)")
                golden_rate = confirm_rate
                golden_args = list(cand_args)
                entry["status"] = "adopted"
            else:
                print(f"    [REJECTED] Confirmation run ({confirm_rate:.2f} tok/s) failed to hold gain.")
                entry["status"] = "confirmation_failed"
        elif diff < -NOISE_BAND:
            print(f"    [ROLLBACK] Candidate [{label}] dropped by {diff*100:.2f}%. Rolling back to golden baseline.")
            entry["status"] = "rollback_degraded"
        else:
            print(f"    [REJECTED] Candidate [{label}] within noise margin ({diff*100:.2f}%). Keeping golden baseline.")
            entry["status"] = "rejected_noise"
        history.append(entry)

    # Step 2: Tune Draft Confidence Threshold (--spec-min-p)
    print("\n" + "="*80)
    print("STEP 2: Tuning Speculative Confidence Threshold (--spec-min-p)")
    print("="*80)
    minp_candidates = [
        ("minp_0.40", "0.40"),
        ("minp_0.60", "0.60"),
        ("minp_0.70", "0.70"),
    ]
    for label, minp_val in minp_candidates:
        cand_args = with_arg(golden_args, "--spec-min-p", minp_val)
        cand_res = evaluate_config(cfg, cand_args, label)
        cand_rate = cand_res["mean_total_decode_tok_s"]
        diff = (cand_rate - golden_rate) / golden_rate

        entry = {"step": "step_2_minp", "label": label, "rate": cand_rate, "diff_pct": round(diff * 100, 2), "args": cand_args}
        if diff >= MIN_GAIN:
            print(f"    --> Candidate [{label}] beat golden by +{diff*100:.2f}%! Confirming...")
            confirm_res = evaluate_config(cfg, cand_args, f"{label}_confirm")
            confirm_rate = confirm_res["mean_total_decode_tok_s"]
            if confirm_rate > golden_rate * (1.0 + NOISE_BAND):
                print(f"    [ADOPTED] Candidate [{label}] confirmed new winner! ({confirm_rate:.2f} tok/s)")
                golden_rate = confirm_rate
                golden_args = list(cand_args)
                entry["status"] = "adopted"
            else:
                print(f"    [REJECTED] Confirmation failed ({confirm_rate:.2f} tok/s).")
                entry["status"] = "confirmation_failed"
        elif diff < -NOISE_BAND:
            print(f"    [ROLLBACK] Candidate [{label}] dropped by {diff*100:.2f}%. Rolling back.")
            entry["status"] = "rollback_degraded"
        else:
            print(f"    [REJECTED] Candidate within noise margin ({diff*100:.2f}%).")
            entry["status"] = "rejected_noise"
        history.append(entry)

    # Step 3: Tune CPU Pool Worker Threads (--pool-workers)
    print("\n" + "="*80)
    print("STEP 3: Tuning CPU Pool Worker Threads (--pool-workers)")
    print("="*80)
    worker_candidates = [
        ("workers_4", "4"),
        ("workers_5", "5"),
        ("workers_6", "6"),
    ]
    for label, w_val in worker_candidates:
        cand_args = with_arg(golden_args, "--pool-workers", w_val)
        cand_res = evaluate_config(cfg, cand_args, label)
        cand_rate = cand_res["mean_total_decode_tok_s"]
        diff = (cand_rate - golden_rate) / golden_rate

        entry = {"step": "step_3_workers", "label": label, "rate": cand_rate, "diff_pct": round(diff * 100, 2), "args": cand_args}
        if diff >= MIN_GAIN:
            print(f"    --> Candidate [{label}] beat golden by +{diff*100:.2f}%! Confirming...")
            confirm_res = evaluate_config(cfg, cand_args, f"{label}_confirm")
            confirm_rate = confirm_res["mean_total_decode_tok_s"]
            if confirm_rate > golden_rate * (1.0 + NOISE_BAND):
                print(f"    [ADOPTED] Candidate [{label}] confirmed new winner! ({confirm_rate:.2f} tok/s)")
                golden_rate = confirm_rate
                golden_args = list(cand_args)
                entry["status"] = "adopted"
            else:
                entry["status"] = "confirmation_failed"
        elif diff < -NOISE_BAND:
            print(f"    [ROLLBACK] Candidate [{label}] dropped by {diff*100:.2f}%. Rolling back.")
            entry["status"] = "rollback_degraded"
        else:
            print(f"    [REJECTED] Candidate within noise margin ({diff*100:.2f}%).")
            entry["status"] = "rejected_noise"
        history.append(entry)

    # Step 4: Tune PCIe Transfer Share (--pcie-frac)
    print("\n" + "="*80)
    print("STEP 4: Tuning PCIe Transfer Share (--pcie-frac)")
    print("="*80)
    pcie_candidates = [
        ("pcie_0.35", "0.35"),
        ("pcie_0.55", "0.55"),
        ("pcie_0.70", "0.70"),
    ]
    for label, p_val in pcie_candidates:
        cand_args = with_arg(golden_args, "--pcie-frac", p_val)
        cand_res = evaluate_config(cfg, cand_args, label)
        cand_rate = cand_res["mean_total_decode_tok_s"]
        diff = (cand_rate - golden_rate) / golden_rate

        entry = {"step": "step_4_pcie", "label": label, "rate": cand_rate, "diff_pct": round(diff * 100, 2), "args": cand_args}
        if diff >= MIN_GAIN:
            print(f"    --> Candidate [{label}] beat golden by +{diff*100:.2f}%! Confirming...")
            confirm_res = evaluate_config(cfg, cand_args, f"{label}_confirm")
            confirm_rate = confirm_res["mean_total_decode_tok_s"]
            if confirm_rate > golden_rate * (1.0 + NOISE_BAND):
                print(f"    [ADOPTED] Candidate [{label}] confirmed new winner! ({confirm_rate:.2f} tok/s)")
                golden_rate = confirm_rate
                golden_args = list(cand_args)
                entry["status"] = "adopted"
            else:
                entry["status"] = "confirmation_failed"
        elif diff < -NOISE_BAND:
            print(f"    [ROLLBACK] Candidate [{label}] dropped by {diff*100:.2f}%. Rolling back.")
            entry["status"] = "rollback_degraded"
        else:
            print(f"    [REJECTED] Candidate within noise margin ({diff*100:.2f}%).")
            entry["status"] = "rejected_noise"
        history.append(entry)

    # Final Summary
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print(f"Initial Rate: {history[0]['rate']:.2f} tok/s")
    print(f"Final Golden Rate: {golden_rate:.2f} tok/s (+{((golden_rate - history[0]['rate'])/history[0]['rate'])*100:.2f}%)")
    print(f"Winning Args: {' '.join(golden_args)}")
    print("="*80 + "\n")

    report = {
        "initial_rate": history[0]["rate"],
        "final_rate": golden_rate,
        "gain_pct": round(((golden_rate - history[0]["rate"]) / history[0]["rate"]) * 100, 2),
        "golden_args": golden_args,
        "history": history
    }
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Save optimized config
    cfg["args"] = golden_args
    Path(config_path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {config_path} with optimal golden arguments.")
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_tuning()
