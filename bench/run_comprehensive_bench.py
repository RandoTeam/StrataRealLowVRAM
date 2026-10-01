import time
import json
import urllib.request
import urllib.error

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

TEST_PROMPTS = [
    {
        "name": "1. Short Prompt (Logic & Math)",
        "thinking": "medium",
        "system": "You are a concise, precise AI assistant.",
        "user": "Explain the difference between a mutex and a semaphore in concurrent programming in 3 short bullet points. Provide a one-line C++ code example for each."
    },
    {
        "name": "2. Medium Prompt (Algorithm & Coding)",
        "thinking": "high",
        "system": "You are an expert Python engineer and computer scientist.",
        "user": """Implement an LRU (Least Recently Used) cache in Python without using `collections.OrderedDict`.
Requirements:
1. Use a doubly linked list and a hash map for O(1) get and put operations.
2. Provide complete class implementation (`LRUCache`) with `get(key)` and `put(key, value)`.
3. Include clear docstrings, type annotations, and a short test verifying eviction when capacity is exceeded."""
    },
    {
        "name": "3. Long Prompt (Refactoring & Code Review)",
        "thinking": "high",
        "system": "You are a senior systems engineer reviewing performance-critical backend code.",
        "user": """Analyze and optimize the following Python database synchronization worker. Identify 4 performance bottlenecks, race conditions, or memory leaks, and provide the fully refactored, robust production version:

```python
import time, threading

CACHE = {}
LOCK = threading.Lock()

def fetch_user_data(user_id):
    # Simulated slow database call
    time.sleep(0.05)
    return {"id": user_id, "score": user_id * 10, "timestamp": time.time()}

def worker(user_ids):
    results = []
    for uid in user_ids:
        if uid in CACHE:
            results.append(CACHE[uid])
        else:
            data = fetch_user_data(uid)
            with LOCK:
                CACHE[uid] = data
            results.append(data)
    return results

def sync_batch(batches):
    threads = []
    all_results = []
    for batch in batches:
        t = threading.Thread(target=lambda: all_results.extend(worker(batch)))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return all_results
```
Show your reasoning step-by-step in Russian, followed by the clean production Python code."""
    }
]

def run_test(prompt_info):
    payload = {
        "model": "qwen3.8-flash-next-coder-iq1_m",
        "messages": [
            {"role": "system", "content": prompt_info["system"]},
            {"role": "user", "content": prompt_info["user"]}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 1024,
        "reasoning_effort": prompt_info["thinking"]
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            res_body = response.read().decode("utf-8")
            elapsed = time.time() - start_time
            parsed = json.loads(res_body)
            
            usage = parsed.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            choice = parsed.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            
            speed = completion_tokens / elapsed if elapsed > 0 else 0
            
            return {
                "name": prompt_info["name"],
                "success": True,
                "elapsed_sec": round(elapsed, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "speed_tok_sec": round(speed, 2),
                "reasoning_len": len(reasoning),
                "content_len": len(content),
                "sample_output": content[:300],
                "sample_reasoning": reasoning[:200] if reasoning else "N/A"
            }
    except Exception as e:
        return {
            "name": prompt_info["name"],
            "success": False,
            "error": str(e),
            "elapsed_sec": round(time.time() - start_time, 2)
        }

if __name__ == "__main__":
    print(f"=== Starting Comprehensive Benchmark across {len(TEST_PROMPTS)} prompts ===")
    results = []
    for p in TEST_PROMPTS:
        print(f"\nRunning: {p['name']} (Thinking: {p['thinking']})...")
        res = run_test(p)
        results.append(res)
        if res.get("success"):
            print(f"  -> Finished in {res['elapsed_sec']}s: {res['completion_tokens']} tokens generated ({res['speed_tok_sec']} tok/s)")
            print(f"  -> Prompt tokens: {res['prompt_tokens']}")
            print(f"  -> Reasoning snippet: {res['sample_reasoning'][:120]}...")
            print(f"  -> Content snippet: {res['sample_output'][:120]}...")
        else:
            print(f"  -> Failed: {res.get('error')}")
        time.sleep(2)
        
    with open("bench/benchmark_results_192_experts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== Benchmark Complete! Saved to bench/benchmark_results_192_experts.json ===")
