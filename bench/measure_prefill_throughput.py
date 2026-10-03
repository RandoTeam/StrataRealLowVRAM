import time
import json
import urllib.request
import sys
import psutil

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def test_prompt_prefill(prompt_len_tokens=1000):
    # Construct a repeated text prompt of approximately prompt_len_tokens
    # Each repetition of a 20-word phrase is ~30 tokens
    base_phrase = "In a modern high performance computer system, memory bandwidth and tensor cores dictate the throughput of deep neural network architectures. "
    repeats = max(1, prompt_len_tokens // 25)
    prompt_text = base_phrase * repeats
    
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1,  # We only want TTFT / prefill speed
        "temperature": 0.0,
        "stream": False
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    
    print(f"[*] Sending fresh prompt of ~{len(prompt_text.split())} words (~{prompt_len_tokens} tokens)...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            elapsed = time.time() - t0
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            prompt_toks = usage.get("prompt_tokens", prompt_len_tokens)
            speed = prompt_toks / elapsed if elapsed > 0 else 0
            print(f"[+] COMPLETED in {elapsed:.2f}s!")
            print(f"    - Prompt Tokens: {prompt_toks}")
            print(f"    - Prefill Wall Speed: {speed:.1f} tok/s")
            return speed, prompt_toks, elapsed
    except Exception as e:
        print(f"[!] Request failed: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    test_prompt_prefill(n)
