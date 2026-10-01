import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

payload = {
    "model": "qwen3.8-flash-next-coder-iq1_m",
    "messages": [
        {"role": "system", "content": "You are a professional software engineer. Be concise."},
        {"role": "user", "content": "Write a thread-safe singleton in modern C++ with explanation in two bullet points."}
    ],
    "max_tokens": 160,
    "reasoning_budget_tokens": 50,
    "stream": True
}

print("[TEST] Sending completion request to Strata server...")
start = time.time()
first_token = None
tokens_generated = 0
reasoning_text = ""
content_text = ""

req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    for raw in resp:
        line = raw.decode("utf-8").strip()
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
            print(f"[TEST] Time to first token (TTFT): {first_token - start:.2f}s")
        
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        r = delta.get("reasoning_content")
        c = delta.get("content")
        if r:
            reasoning_text += r
            print(r, end="", flush=True)
            tokens_generated += 1
        if c:
            content_text += c
            print(c, end="", flush=True)
            tokens_generated += 1

end = time.time()
total_time = end - (first_token or start)
tok_s = (tokens_generated / total_time) if total_time > 0 else 0
print(f"\n\n[TEST SUMMARY]")
print(f"Total tokens generated: {tokens_generated}")
print(f"Generation time: {total_time:.2f}s")
print(f"Throughput: {tok_s:.2f} tok/s")
