import os
import sys
import glob
import json
import subprocess
import time

TOURNAMENT3_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\bench\tournament3"
CANDIDATES_DIR = os.path.join(TOURNAMENT3_DIR, "candidates")
RUNNER_PY = os.path.join(TOURNAMENT3_DIR, "runner_olympic.py")
LEADERBOARD_JSON = os.path.join(TOURNAMENT3_DIR, "leaderboard.json")
LEADERBOARD_MD = os.path.join(TOURNAMENT3_DIR, "leaderboard.md")

CANDIDATE_ORDER = ["O01", "O02", "O03", "O04"]

def run_olympic_orchestrator():
    print("==================================================================", flush=True)
    print("   STARTING OLYMPIC TOURNAMENT 3.0: 4 ELITE OLYMPIANS BATCH", flush=True)
    print("==================================================================", flush=True)
    
    results = []
    for cand_id in CANDIDATE_ORDER:
        cand_files = glob.glob(os.path.join(CANDIDATES_DIR, f"{cand_id}*_candidate.json"))
        if not cand_files:
            print(f"[-] Candidate file for {cand_id} not found! Skipping...", flush=True)
            continue
            
        cand_file = cand_files[0]
        res_file = cand_file.replace(".json", "_result.json")
        
        # Check if already executed
        if os.path.exists(res_file):
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                if "score" in res_data and res_data.get("status") != "DISQUALIFIED":
                    print(f"[+] Candidate {cand_id} already evaluated: Score={res_data.get('score')}. Skipping rerun.", flush=True)
                    results.append(res_data)
                    continue
            except Exception:
                pass

        print(f"\n[>>>] Running Olympic Evaluation for {cand_id}...", flush=True)
        cmd = [sys.executable, "-u", RUNNER_PY, cand_file]
        proc = subprocess.run(cmd, cwd=TOURNAMENT3_DIR)
        
        if os.path.exists(res_file):
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                results.append(res_data)
            except Exception as e:
                print(f"[!] Error reading result for {cand_id}: {e}", flush=True)
        else:
            print(f"[!] Result file not generated for {cand_id}!", flush=True)
            
        print(f"[<<<] Completed {cand_id}. Pausing 10s for thermal and memory reset...", flush=True)
        time.sleep(10)
        
    # Generate Leaderboard
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    with open(LEADERBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    md_lines = [
        "# Таблица лидеров: Олимпийский Турнир 3.0 (4 Олимпийца)",
        "",
        f"**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Модель:** Qwen 3.8 Flash Next Coder (IQ1_M) — 125B MoE",
        "**Железо:** RTX 3050 Laptop 4GB + Ryzen 5 5500U + 32GB DDR4-3200",
        "",
        "| Ранг | ID | Название | Категория | Статус | Decode (tok/s) | Prefill (tok/s) | Hit Rate (%) | MTP Acc (%) | Слоты | Quality | Score |",
        "|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    
    for rank, r in enumerate(results, 1):
        md_lines.append(
            f"| **{rank}** | `{r.get('id')}` | {r.get('name')} | {r.get('category')} | {r.get('status')} | "
            f"**{r.get('decode_tok_s')}** | {r.get('prefill_tok_s')} | {r.get('hit_rate_pct')}% | "
            f"{r.get('mtp_acc_pct')}% | {r.get('expert_slots')} | +{r.get('quality_bonus')} | **{r.get('score')}** |"
        )
        
    with open(LEADERBOARD_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print("\n==================================================================", flush=True)
    print("   OLYMPIC TOURNAMENT 3.0 COMPLETED! FINAL LEADERBOARD:", flush=True)
    print("==================================================================", flush=True)
    for rank, r in enumerate(results, 1):
        print(f" {rank}. [{r.get('id')}] {r.get('name')}: Score={r.get('score')} | Decode={r.get('decode_tok_s')} tok/s | HitRate={r.get('hit_rate_pct')}% | Slots={r.get('expert_slots')}", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_olympic_orchestrator()
