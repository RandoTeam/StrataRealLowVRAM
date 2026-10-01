import os
import sys
import time
import json
import subprocess
from datetime import datetime

BENCH_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\bench\tournament5"
CANDIDATES_DIR = os.path.join(BENCH_DIR, "candidates")
RUNNER_SCRIPT = os.path.join(BENCH_DIR, "runner_t5.py")
LEADERBOARD_JSON = os.path.join(BENCH_DIR, "leaderboard.json")
LEADERBOARD_MD = os.path.join(BENCH_DIR, "leaderboard.md")
PYTHON_EXE = sys.executable

def run_candidate(cand_file):
    print(f"\n>>> [TOURNAMENT 5.0 ORCHESTRATOR] Launching evaluation for: {os.path.basename(cand_file)} <<<", flush=True)
    with open(cand_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    c_id = data.get("id", os.path.splitext(os.path.basename(cand_file))[0])
    c_name = data.get("name", "Unnamed Candidate")
    args = data.get("args", [])
    env = data.get("env", {})
    
    # Import evaluate_candidate from runner_t5
    sys.path.insert(0, BENCH_DIR)
    from runner_t5 import evaluate_candidate
    
    res = evaluate_candidate(c_id, c_name, args, env)
    
    result_file = cand_file.replace(".json", "_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res

def update_leaderboard(results, round_name="Круг 1 (Индивидуальный)"):
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    with open(LEADERBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    md_content = f"# Таблица лидеров: Олимпийский Турнир 5.0 ({round_name})\n\n"
    md_content += f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    md_content += f"**Цель:** Прорыв барьера $\\ge 8.0$ токенов/сек  \n"
    md_content += f"**Правила:** Контекст строго 65 536 токенов + Сохранение ума и логики модели  \n"
    md_content += f"**Платформа:** NVIDIA RTX 3050 Laptop 4GB + AMD Ryzen 5 5500U + 32GB DDR4-3200  \n\n"
    md_content += "| Ранг | ID | Название кандидата | Статус | Чистый Decode | Клиентский Decode | Prefill | Hit Rate | MTP Acc | Ум сохранён? | Итоговый Score |\n"
    md_content += "|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    
    for idx, r in enumerate(results, 1):
        intel_str = "🟢 ДА" if r.get("intelligence_preserved") else "🔴 НЕТ"
        md_content += (
            f"| **{idx}** | `{r.get('id')}` | {r.get('name')} | {r.get('status')} | "
            f"**{r.get('engine_decode_tok_s')} tok/s** | {r.get('client_decode_tok_s')} tok/s | "
            f"{r.get('prefill_tok_s')} tok/s | {r.get('hit_rate_pct')}% | {r.get('mtp_acc_pct')}% | "
            f"{intel_str} | **{r.get('score')}** |\n"
        )
        
    with open(LEADERBOARD_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[ORCHESTRATOR] Leaderboard updated at {LEADERBOARD_MD}", flush=True)

def main():
    if not os.path.exists(CANDIDATES_DIR):
        os.makedirs(CANDIDATES_DIR, exist_ok=True)
        print(f"Created candidates directory: {CANDIDATES_DIR}")
        return
        
    files = [f for f in os.listdir(CANDIDATES_DIR) if f.endswith(".json") and not f.endswith("_result.json")]
    files.sort()
    
    if not files:
        print("No candidates found in candidates/ directory yet.")
        return
        
    print(f"Found {len(files)} candidate configs to evaluate: {files}")
    results = []
    for f in files:
        cand_path = os.path.join(CANDIDATES_DIR, f)
        res = run_candidate(cand_path)
        if res:
            results.append(res)
            # Mandatory 120s Thermal Cooldown between candidates
            print("\n[ORCHESTRATOR] Thermal cooldown (120s) between tournament runs...", flush=True)
            for sec in range(120, 0, -20):
                print(f"  Cooling down... {sec}s remaining", flush=True)
                time.sleep(20)
                
    update_leaderboard(results)

if __name__ == "__main__":
    main()
