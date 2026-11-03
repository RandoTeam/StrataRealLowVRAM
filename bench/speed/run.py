"""Measure one model's speed and record it with its conditions.  Nothing here deletes a log.

Two tasks, because they are not the same thing:

    chat  a chat request through the server, exactly what a person sees: the prompt goes through the chat template
          and the model answers the question at the end of bench/speed/prompt.txt.
    raw   the engine's own --tokens mode with those ids and no chat template: the model continues the text instead of
          answering it.  This is the task the first published table was measured on, so it is kept for comparison —
          never as the headline number.

    python bench/speed/run.py --config strata-clover-apex-i-nanoplus.json --model clover --task chat --runs 8
    python bench/speed/run.py --config strata-flint-ud-q3_k_xl.json --model flint --task raw --runs 8

Every run appends one line to bench/speed/runs.log; the JSON beside the results keeps the same rows plus the engine's
SHA-256, its settings and the prompt's token count.  The median and the range are printed, never a single best run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

PROMPT = Path(__file__).resolve().parent / "prompt.txt"
RUNS_LOG = Path(__file__).resolve().parent / "runs.log"
BASE = "http://127.0.0.1:8080"
BASE_PORT = 8080
SERVER_LINE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                         r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
CLI_LINE = re.compile(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s \([^)]*\); "
                      r"experts from VRAM ([\d.]+)%.*?(\d+) of (\d+) guessed tokens right")


def config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))          # setup writes the configs with a BOM
    cfg["args"] = list(cfg["args"])
    return cfg


def tokenizer_dir(cfg: dict) -> Path:
    return Path(cfg["tokenizer"])


def prompt_ids(cfg: dict) -> list[int]:
    tok = Tokenizer.from_dir(tokenizer_dir(cfg))
    return tok.encode(PROMPT.read_text(encoding="utf-8"))


def engine_sha(cfg: dict) -> str:
    return hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest()


def wait_ready(timeout: float = 240.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=2) as answer:
                if answer.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - the server is not up yet
            pass
        time.sleep(1)
    return False


def run_chat(cfg: dict, ids: list[int], text: str, max_new: int) -> dict:
    log = Path(cfg["log"])
    start = log.stat().st_size if log.exists() else 0          # parse only what this run appended
    server = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                               str(cfg["_path"]), "--port", str(BASE_PORT)], cwd=ROOT, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        if not wait_ready():
            raise RuntimeError("the server did not start")
        body = json.dumps({"model": "local", "temperature": 0, "max_tokens": max_new,
                           "messages": [{"role": "user", "content": text}]}).encode()
        request = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3600) as answer:
            answer.read()
    finally:
        subprocess.run(["powershell", "-c", "Get-Process strata-fourgbvram -ErrorAction SilentlyContinue | "
                                            "Stop-Process -Force -ErrorAction SilentlyContinue"])
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        time.sleep(3)
    fresh = log.read_text(encoding="utf-8", errors="replace")[start:] if log.exists() else ""
    found = SERVER_LINE.findall(fresh)
    if not found:
        raise RuntimeError("no serve line in " + str(log))
    row = found[-1]
    return {"prompt_tokens": int(row[0]), "prefill": float(row[4]), "generated": int(row[5]),
            "decode": float(row[7]), "drafts": row[8] + "/" + row[9] if row[9] else "?"}


def run_raw(cfg: dict, ids: list[int], max_new: int) -> dict:
    profile = Path(rf"C:\Temp\opencode\speed-profile.txt")
    profile.write_bytes(Path(cfg["_profile"]).read_bytes())
    args = [cfg["exe"], "--model", cfg["_model"], "--tokens", ",".join(map(str, ids)), "--max-new", str(max_new),
            "--max-context", "32768", "--expert-profile", str(profile)]
    skip = {"--model", "--tokens", "--max-new", "--max-context", "--expert-profile", "--serve"}
    extra, i = [], 0
    while i < len(cfg["args"]):                                  # the config's own flags, minus what we set here
        flag = cfg["args"][i]
        has_value = i + 1 < len(cfg["args"]) and not cfg["args"][i + 1].startswith("--")
        if flag in skip:
            i += 2 if has_value else 1
        elif has_value:
            extra += [flag, cfg["args"][i + 1]]
            i += 2
        else:
            extra.append(flag)
            i += 1
    out = subprocess.run(args + extra, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=3600)
    found = CLI_LINE.search(out.stderr or "")
    if not found:
        raise RuntimeError("no stats line: " + (out.stderr or "")[-400:])
    return {"prompt_tokens": int(found.group(1)), "prefill": float(found.group(2)),
            "generated": int(found.group(3)), "decode": float(found.group(4)),
            "vram_experts": float(found.group(5)), "drafts": found.group(6) + "/" + found.group(7)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the run config setup wrote (its args and paths are the settings)")
    ap.add_argument("--model", required=True, help="a name for the rows: clover, flint")
    ap.add_argument("--task", choices=("chat", "raw"), required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", help="where the JSON goes (default bench/speed/results/<model>-<task>.json)")
    a = ap.parse_args()

    cfg = config(Path(a.config))
    cfg["_path"] = Path(a.config).resolve()
    cfg["_model"] = cfg["args"][cfg["args"].index("--model") + 1]
    cfg["_profile"] = (cfg["args"][cfg["args"].index("--expert-profile") + 1]
                       if "--expert-profile" in cfg["args"] else "")
    ids = prompt_ids(cfg)
    text = PROMPT.read_text(encoding="utf-8")
    sha = engine_sha(cfg)
    settings, i = {}, 0
    while i < len(cfg["args"]):
        flag = cfg["args"][i]
        if flag.startswith("--") and i + 1 < len(cfg["args"]) and not cfg["args"][i + 1].startswith("--"):
            settings[flag] = cfg["args"][i + 1]
            i += 2
        else:
            if flag.startswith("--"):
                settings[flag] = True
            i += 1

    rows = []
    for n in range(1, a.runs + 1):
        row = run_chat(cfg, ids, text, a.max_new) if a.task == "chat" else run_raw(cfg, ids, a.max_new)
        rows.append(row)
        print(f"{a.model} {a.task} run {n}/{a.runs}: read {row['prefill']:.0f} tok/s  write {row['decode']:.1f} tok/s  "
              f"prompt {row['prompt_tokens']}  drafts {row['drafts']}", flush=True)

    med = statistics.median(r["decode"] for r in rows)
    lo, hi = min(r["decode"] for r in rows), max(r["decode"] for r in rows)
    pre = statistics.median(r["prefill"] for r in rows)
    result = {"model": a.model, "task": a.task, "when": time.strftime("%Y-%m-%d %H:%M"), "runs": a.runs,
              "prompt": str(PROMPT.relative_to(ROOT)), "prompt_tokens": rows[0]["prompt_tokens"],
              "max_new": a.max_new, "engine_sha256": sha, "config": str(cfg["_path"]),
              "settings": settings, "median_decode": med, "min_decode": lo, "max_decode": hi,
              "median_prefill": pre, "rows": rows}
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / f"{a.model}-{a.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    with RUNS_LOG.open("a", encoding="utf-8") as fh:                   # appended, never truncated
        fh.write(f"{result['when']}\t{a.model}\t{a.task}\t{med:.1f}\t{lo:.1f}-{hi:.1f}\t{pre:.0f}\t"
                 f"{rows[0]['prompt_tokens']}\t{sha[:16]}\t{out.name}\n")
    print(f"\n{a.model} {a.task}: {med:.1f} tok/s written (median of {a.runs}, {lo:.1f}-{hi:.1f}), "
          f"prompt read {pre:.0f} tok/s, {rows[0]['prompt_tokens']} tokens, engine {sha[:16]}")
    print(f"written to {out} and appended to {RUNS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
