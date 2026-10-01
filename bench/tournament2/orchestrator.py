import os
import sys
import glob
import json
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_DIR = os.path.join(SCRIPT_DIR, "candidates")
RUNNER_PY = os.path.join(SCRIPT_DIR, "runner.py")
LEADERBOARD_JSON = os.path.join(SCRIPT_DIR, "leaderboard.json")
LEADERBOARD_MD = os.path.join(SCRIPT_DIR, "leaderboard.md")

def get_candidates():
    pattern = os.path.join(CANDIDATES_DIR, "S*_candidate.json")
    files = glob.glob(pattern)
    # Filter out result files
    cand_files = [f for f in files if not f.endswith("_result.json")]
    cand_files.sort(key=lambda x: os.path.basename(x))
    return cand_files

def run_tournament():
    cand_files = get_candidates()
    print(f"=== TOURNAMENT 2: EVALUATING {len(cand_files)} CANDIDATES ===")
    
    for i, c_file in enumerate(cand_files, 1):
        res_file = c_file.replace(".json", "_result.json")
        cid = os.path.basename(c_file).split("_")[0]
        
        if os.path.exists(res_file):
            print(f"[{i}/{len(cand_files)}] {cid} already evaluated, skipping.")
            continue
            
        print(f"\n>>> [{i}/{len(cand_files)}] Launching {os.path.basename(c_file)} ...", flush=True)
        cmd = [sys.executable, "-u", RUNNER_PY, c_file]
        try:
            # Run synchronously so we capture full output and don't overlap
            proc = subprocess.run(cmd, cwd=SCRIPT_DIR, text=True, capture_output=False)
            if proc.returncode != 0:
                print(f"Runner failed for {cid} with code {proc.returncode}")
        except Exception as e:
            print(f"Execution error on {cid}: {e}")
            
        time.sleep(3)
        
    print("\n=== TOURNAMENT 2 BENCHMARKS FINISHED ===")
    collate_leaderboard()

def collate_leaderboard():
    pattern = os.path.join(CANDIDATES_DIR, "S*_candidate_result.json")
    res_files = glob.glob(pattern)
    
    results = []
    for rf in res_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error reading {rf}: {e}")
            
    # Sort by score descending
    results.sort(key=lambda x: (x.get("score", 0.0), x.get("decode_tok_s", 0.0)), reverse=True)
    
    with open(LEADERBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved leaderboard with {len(results)} candidates to {LEADERBOARD_JSON}")
    
    # Generate Markdown table
    md_lines = [
        "# Таблица лидеров: Второй турнир оптимизации Strata (20 субагентов)",
        "",
        f"**Дата проведения:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Модель:** Qwen 3.8 Flash Next Coder (IQ1_M) — 125B MoE (256/512 экспертов, 48 слоёв, 10 активных на токен)",
        "**Железо:** RTX 3050 Laptop 4GB GDDR6 (GA107, PCIe 3.0 x8) + Ryzen 5 5500U (6C/12T Zen 2 AVX2) + 32GB DDR4-3200",
        "",
        "| Ранг | ID | Название | Категория | Статус | Decode (tok/s) | Prefill (tok/s) | Hit Rate (%) | MTP Acc (%) | Слоты VRAM | Quality | Итоговый Score |",
        "|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    
    for rank, r in enumerate(results, 1):
        md_lines.append(
            f"| **{rank}** | `{r.get('id', '')}` | {r.get('name', '')} | {r.get('category', '')} | {r.get('status', '')} | "
            f"**{r.get('decode_tok_s', 0.0)}** | {r.get('prefill_tok_s', 0.0)} | {r.get('hit_rate_pct', 0.0)}% | "
            f"{r.get('mtp_acc_pct', 0.0)}% | {r.get('expert_slots', 0)} | +{r.get('quality_bonus', 0.0)} | "
            f"**{r.get('score', 0.0)}** |"
        )
        
    with open(LEADERBOARD_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Saved leaderboard markdown to {LEADERBOARD_MD}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--collate":
        collate_leaderboard()
    else:
        run_tournament()
