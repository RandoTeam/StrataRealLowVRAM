import urllib.request
import json
import time

url = "http://127.0.0.1:8080/v1/chat/completions"
req_data = {
    "model": "qwen3.8-flash-next-q2_0",
    "messages": [
        {"role": "user", "content": "Реши задачу на Python: дан массив целых чисел, найди длину наибольшей строго возрастающей подпоследовательности за время O(n log n). Напиши код с подробным объяснением."}
    ],
    "max_tokens": 180,
    "temperature": 0.6
}

data = json.dumps(req_data).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("[benchmark] Sending request to Strata server...")
t0 = time.time()
try:
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        t1 = time.time()
        res = json.loads(body)
        content = res["choices"][0]["message"]["content"]
        usage = res.get("usage", {})
        total_time = t1 - t0
        print(f"[benchmark] Response received in {total_time:.2f} s")
        print(f"[benchmark] Usage: {usage}")
        print("\n--- Response ---")
        print(content)
        print("----------------")
except Exception as e:
    print(f"Error: {e}")
