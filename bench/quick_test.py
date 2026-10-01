import urllib.request
import json
import time

data = json.dumps({
    'model': 'qwen3.8-flash-next-coder-iq1_m',
    'messages': [{'role': 'user', 'content': 'Write a 2-line Python function to check if a number is prime and explain it briefly.'}],
    'temperature': 0.6,
    'max_tokens': 128
}).encode('utf-8')

t0 = time.time()
req = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as res:
    el = time.time() - t0
    b = json.loads(res.read().decode())
    usage = b.get('usage', {})
    c = usage.get('completion_tokens', 0)
    p = usage.get('prompt_tokens', 0)
    print(f"Prompt tokens: {p}, Completion tokens: {c}, Elapsed: {el:.2f}s, Speed: {c/el:.2f} tok/s")
    print("Content preview:")
    print(b['choices'][0]['message']['content'][:300])
