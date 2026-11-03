"""Run bench/quality/prompts.json against a running 4gbVramStrata server and score the answers.

    python bench/quality/run.py --model clover-apex-i-nanoplus --port 8080
    python bench/quality/run.py --model flint-ud-q3_k_xl --out bench/quality/results/flint.json

The server must already be running with the model loaded (START-HERE.bat, or serve/server.py --config ...).  Every
prompt is greedy (temperature 0) with a generous max_tokens, because the reasoning block counts against it.  The checks
are deliberately shallow - a smoke test that the model still answers the question - not a benchmark: they catch an
answer that is empty, off-topic, in the wrong language or refuses, and they say pass or fail, never a score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).resolve().parent / "prompts.json"
CYRILLIC = re.compile("[\u0400-\u04FF]")


def json_objects(text: str) -> list[str]:
    """Every balanced {...} run in the text, braces inside strings included: a tool call has nested arguments."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append(text[start : i + 1])
    return out


def check(kind: str, value, answer: str) -> bool:
    text = (answer or "").strip()
    low = text.lower()
    if not text:
        return False
    if kind == "contains":
        return str(value).lower() in low
    if kind == "contains_all":
        return all(str(v).lower() in low for v in value)
    if kind == "json_keys":
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return False
        return isinstance(data, dict) and all(k in data for k in value)
    if kind == "palindrome_code":
        # any honest palindrome function: def, a case fold, and whitespace handling (replace or split are both fine)
        return ("def is_palindrome" in low and "lower" in low and ("replace" in low or "split" in low or "join" in low))
    if kind == "russian_list":
        items = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s", text))
        return len(CYRILLIC.findall(text)) > 80 and items >= int(value)
    if kind == "cyrillic_and_count":
        cyr = len(CYRILLIC.findall(text))
        return cyr > 40 and len(re.findall(r"(?m)^\s*\d[.)]", text)) >= int(value)
    if kind == "word_count":
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        return len([w for w in words if re.search(r"[A-Za-z\u0400-\u04FF]", w)]) == int(value)
    if kind == "max_words":
        return len(text.split()) <= int(value)
    if kind == "fib8":
        nums = [p.strip() for p in text.replace("\n", " ").split(",")]
        try:
            vals = [int(p) for p in nums]
        except ValueError:
            return False
        return vals in ([0, 1, 1, 2, 3, 5, 8, 13], [1, 1, 2, 3, 5, 8, 13, 21])
    if kind == "comma_count":
        parts = [p for p in text.replace("\n", " ").split(",") if p.strip()]
        return len(parts) == int(value) and all(re.fullmatch(r"\s*\d+\s*", p) for p in parts)
    if kind == "json_array_len":
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return False
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return False
        return isinstance(data, list) and len(data) == int(value) and all(
            isinstance(x, dict) and {"name", "size"} <= set(x) for x in data)
    if kind == "json_calls":
        # one JSON object per line, each a {"name": ..., "arguments": {...}} call: the names must appear, in order
        calls = []
        for obj in json_objects(text):
            try:
                data = json.loads(obj)
            except ValueError:
                continue
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                calls.append(data["name"].strip().rsplit(".", 1)[-1])
        want = [str(x) for x in value]
        i = 0
        for name in calls:
            if i < len(want) and name == want[i]:
                i += 1
        return i == len(want)
    if kind == "runs_python":
        # the code has to run: exec it, call the function, compare the results as JSON so tuples and lists match
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
        code = m.group(1) if m else text
        space: dict = {}
        try:
            exec(code, space)  # noqa: S102 - the point of the check is to run the model's code
            fn = space[value["function"]]
            for case in value["cases"]:
                got = fn(*case["args"])
                if json.loads(json.dumps(got)) != case["want"]:
                    return False
        except Exception:  # noqa: BLE001 - a code check that throws is a fail, and its reason is not needed
            return False
        return True
    raise ValueError(f"unknown check {kind!r}")


def with_context(p: dict) -> str:
    """A long-context prompt: the facts placed at 15%, 50% and 85% of the document, the questions at the end."""
    long_ = p.get("long")
    if not long_:
        return p["prompt"]
    words = max(50, int(long_["fill_words"]))
    gap = long_["filler"] * max(1, words // max(1, len(long_["filler"].split())))
    parts = [gap[: len(gap) // 3], long_["facts"][0], gap, long_["facts"][1], gap, long_["facts"][2], gap]
    return p["prompt"] + "\n\n" + "\n".join(parts) + "\n\n" + long_["questions"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the name the server answers as")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--repeat", type=int, default=1, help="asks every prompt N times: a greedy answer can still move "
                                                          "when the expert cache does")
    ap.add_argument("--out", help="where to write the JSON result (default: bench/quality/results/<model>.json)")
    ap.add_argument("--no-long", action="store_true", help="leave out the long-context prompt (a quick run)")
    ap.add_argument("--only", help="run only these prompt ids, comma-separated")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    try:
        requests.get(base + "/v1/models", timeout=5).raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"no server on {base}: {e}")

    prompts = json.loads(PROMPTS.read_text(encoding="utf-8-sig"))
    if a.no_long:
        prompts = [p for p in prompts if "long" not in p]
    if a.only:
        wanted = {x.strip() for x in a.only.split(",")}
        prompts = [p for p in prompts if p["id"] in wanted]
    rows, passed, passed_any = [], 0, 0
    for p in prompts:
        question = with_context(p)
        attempts = []
        for _ in range(max(1, a.repeat)):
            t0 = time.time()
            r = requests.post(base + "/v1/chat/completions", timeout=1800, json={
                "model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
                "messages": [{"role": "user", "content": question}]})
            r.raise_for_status()
            d = r.json()
            msg = d["choices"][0]["message"]
            answer = msg.get("content") or ""
            attempts.append({"answer": answer, "pass": check(p["check"]["kind"], p["check"]["value"], answer),
                             "reasoning_chars": len(msg.get("reasoning_content") or ""),
                             "seconds": round(time.time() - t0, 1), "usage": d.get("usage")})
        ok = all(x["pass"] for x in attempts)
        any_ok = any(x["pass"] for x in attempts)
        passed += ok
        passed_any += any_ok
        label = "ok  " if ok else ("FLAKY" if any_ok else "FAIL ")
        rows.append({"id": p["id"], "domain": p["domain"], "prompt": p["prompt"], "pass": ok,
                     "passes": sum(x["pass"] for x in attempts), "attempts": attempts})
        print(f"{label} {p['id']:20s} {rows[-1]['passes']}/{len(attempts)}  "
              f"{attempts[-1]['answer'].strip()[:70]!r}")

    out = Path(a.out) if a.out else ROOT / "bench" / "quality" / "results" / f"{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "when": time.strftime("%Y-%m-%d %H:%M"),
                               "engine": "strata-fourgbvram", "passed": passed, "passed_any": passed_any,
                               "total": len(rows), "repeat": max(1, a.repeat),
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{passed} of {len(rows)} passed every attempt ({passed_any} passed at least one); written to {out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
