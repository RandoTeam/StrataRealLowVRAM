import time
import requests
import json

url = "http://127.0.0.1:8080/v1/chat/completions"

headers = {"Content-Type": "application/json"}

# Prompt designed to test structured reasoning and code generation
payload = {
    "model": "default",
    "messages": [
        {"role": "user", "content": "Write a Python function to check if a binary tree is symmetric, with a brief step-by-step reasoning."}
    ],
    "max_tokens": 150,
    "temperature": 0.6,
    "stream": True
}

print("Sending request to Qwen3.6-35B-A3B at 131k context (512 cache slots)...")
t0 = time.time()
response = requests.post(url, headers=headers, json=payload, stream=True)

first_token_time = None
tokens = []
reasoning_tokens = []

for line in response.iter_lines():
    if not line:
        continue
    line_str = line.decode('utf-8')
    if line_str.startswith("data: "):
        data_str = line_str[6:].strip()
        if data_str == "[DONE]":
            break
        data = json.loads(data_str)
        choice = data["choices"][0]
        delta = choice.get("delta", {})
        
        now = time.time()
        if first_token_time is None:
            first_token_time = now
            print(f"Time to first token (prefill): {(first_token_time - t0)*1000:.1f} ms")
            
        r_text = delta.get("reasoning_content")
        c_text = delta.get("content")
        if r_text:
            reasoning_tokens.append(r_text)
            print(r_text, end="", flush=True)
        if c_text:
            tokens.append(c_text)
            print(c_text, end="", flush=True)

t_end = time.time()
total_tokens = len(reasoning_tokens) + len(tokens)
decode_time = t_end - first_token_time if first_token_time else 0.001
tok_per_sec = total_tokens / decode_time if decode_time > 0 else 0

print("\n" + "="*50)
print(f"Total chunks/tokens emitted: {total_tokens}")
print(f"Decode duration: {decode_time:.2f} s")
print(f"Effective generation speed: {tok_per_sec:.2f} chunks/sec")
print("="*50)
