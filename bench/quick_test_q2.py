import json
import time
import urllib.request
import sys

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

URL = "http://127.0.0.1:8080/v1/chat/completions"

payload = {
    "model": "qwen3.8-flash-next-q2_0",
    "messages": [
        {"role": "system", "content": "You are a senior systems engineer. Think step by step and then answer."},
        {"role": "user", "content": "Explain why lock-free data structures often suffer from the ABA problem, and how a hazard pointer solves it. Give a concise summary."}
    ],
    "max_tokens": 180,
    "reasoning_budget_tokens": 50,
    "stream": True
}

print("[Q2_0 BENCHMARK] Sending completion request...")
start = time.time()
first_token = None
reasoning_end = None
tokens_generated = 0
reasoning_tokens = 0
content_tokens = 0

req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except Exception:
            continue
        
        now = time.time()
        if first_token is None:
            first_token = now
            print(f"[Q2_0] Time to first token (TTFT): {first_token - start:.2f}s\n[THINKING START]")
        
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        r = delta.get("reasoning_content")
        c = delta.get("content")
        if r:
            reasoning_tokens += 1
            tokens_generated += 1
            print(r, end="", flush=True)
        if c:
            if reasoning_end is None and reasoning_tokens > 0:
                reasoning_end = now
                print("\n\n[ANSWER START]", flush=True)
            content_tokens += 1
            tokens_generated += 1
            print(c, end="", flush=True)

end = time.time()
total_gen_time = end - (first_token or start)
r_time = (reasoning_end - first_token) if (reasoning_end and first_token) else 0.0
c_time = (end - reasoning_end) if (reasoning_end and end > reasoning_end) else 0.0

r_tok_s = (reasoning_tokens / r_time) if r_time > 0 else 0
c_tok_s = (content_tokens / c_time) if c_time > 0 else 0
overall_tok_s = (tokens_generated / total_gen_time) if total_gen_time > 0 else 0

print(f"\n\n==========================================")
print(f"       Q2_0 BENCHMARK RESULTS")
print(f"==========================================")
print(f"Total tokens:      {tokens_generated}")
print(f"Total time:        {total_gen_time:.2f}s")
print(f"Overall throughput:{overall_tok_s:.2f} tok/s")
print(f"Thinking tokens:   {reasoning_tokens} in {r_time:.2f}s => {r_tok_s:.2f} tok/s")
print(f"Answer tokens:     {content_tokens} in {c_time:.2f}s => {c_tok_s:.2f} tok/s")
print(f"==========================================")
