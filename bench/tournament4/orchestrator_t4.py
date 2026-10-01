import os
import sys
import time
import json
import subprocess
from datetime import datetime

BENCH_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\bench\tournament4"
CANDIDATES_DIR = os.path.join(BENCH_DIR, "candidates")
RUNNER_SCRIPT = os.path.join(BENCH_DIR, "runner_t4.py")
LEADERBOARD_JSON = os.path.join(BENCH_DIR, "leaderboard.json")
LEADERBOARD_MD = os.path.join(BENCH_DIR, "leaderboard.md")
PYTHON_EXE = sys.executable

def run_candidate(cand_file):
    print(f"\n>>> [TOURNAMENT 4 ORCHESTRATOR] Launching evaluation for: {os.path.basename(cand_file)} <<<", flush=True)
    cmd = [PYTHON_EXE, RUNNER_SCRIPT, cand_file]
    res = subprocess.run(cmd)
    
    result_file = cand_file.replace(".json", "_result.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def update_leaderboard(results):
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    with open(LEADERBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    md_content = f"# Таблица лидеров: Турнир 4.0 (Цель 8.0+ tok/s)\n\n"
    md_content += f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_content += f"**Модель:** Qwen 3.8 Flash Next Coder (IQ1_M) — 125B MoE\n"
    md_content += f"**Железо:** RTX 3050 Laptop 4GB + Ryzen 5 5500U + 32GB DDR4-3200\n\n"
    md_content += "| Ранг | ID | Название | Категория | Статус | Decode (tok/s) | Prefill (tok/s) | Hit Rate (%) | MTP Acc (%) | Слоты | Quality | Score |\n"
    md_content += "|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    
    for idx, r in enumerate(results, 1):
        md_content += f"| **{idx}** | `{r.get('id')}` | {r.get('name')} | {r.get('category')} | {r.get('status')} | **{r.get('decode_tok_s')}** | {r.get('prefill_tok_s')} | {r.get('hit_rate_pct')}% | {r.get('mtp_acc_pct')}% | {r.get('expert_slots')} | +{r.get('quality_bonus')} | **{r.get('score')}** |\n"
        
    with open(LEADERBOARD_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[ORCHESTRATOR] Leaderboard updated at {LEADERBOARD_MD}", flush=True)

def main():
    if not os.path.exists(CANDIDATES_DIR):
        print(f"Error: {CANDIDATES_DIR} does not exist.")
        sys.exit(1)
        
    files = [f for f in os.listdir(CANDIDATES_DIR) if f.endswith(".json") and not f.endswith("_result.json")]
    files.sort()
    
    print(f"Found {len(files)} candidate configs to evaluate: {files}")
    
    results = []
    for f in files:
        cand_path = os.path.join(CANDIDATES_DIR, f)
        res = run_candidate(cand_path)
        if res:
            results.append(res)
            update_leaderboard(results)
            
        print("\n  -> Thermal cooldown pause (15 seconds) to ensure thermal stability of Ryzen 5 5500U...", flush=True)
        time.sleep(15)
        
    print("\n=======================================================")
    print("  ALL TOURNAMENT 4 CANDIDATES EVALUATED SUCCESSFULLY!  ")
    print("=======================================================")

if __name__ == "__main__":
    main()
