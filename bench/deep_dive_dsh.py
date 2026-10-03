import zstandard as zstd
import json
from pathlib import Path
from datetime import datetime

SESSION_PATH = Path(r"C:\Users\Ilia V\.dsh\sessions\--C-DeepSeak~0020Harness-Test~00201--\session-73f94640-d151-4d00-aac4-83736e9e865e\session.jsonl.zstd")

def run():
    dctx = zstd.ZstdDecompressor()
    with open(SESSION_PATH, 'rb') as f:
        reader = dctx.stream_reader(f)
        text_stream = reader.read().decode('utf-8', errors='replace')
    
    events = [json.loads(line) for line in text_stream.strip().split('\n') if line.strip()]
    
    # 1. User prompts
    print("=================== USER PROMPTS ===================")
    for ev in events:
        if ev.get("type") == "user/message":
            ts = datetime.fromtimestamp(ev.get("time", 0)/1000).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] {json.dumps(ev.get('data', {}), ensure_ascii=False)[:500]}")
            
    # 2. Compaction events
    print("\n=================== COMPACTION EVENTS ===================")
    for ev in events:
        if "compaction" in ev.get("type", ""):
            ts = datetime.fromtimestamp(ev.get("time", 0)/1000).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] type={ev.get('type')}: {json.dumps(ev.get('data', {}), ensure_ascii=False)}")
            
    # 3. LLM retries or errors
    print("\n=================== LLM RETRIES / ERRORS ===================")
    for ev in events:
        if "retry" in ev.get("type", "") or "error" in ev.get("type", ""):
            ts = datetime.fromtimestamp(ev.get("time", 0)/1000).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] type={ev.get('type')}: {json.dumps(ev.get('data', {}), ensure_ascii=False)}")
            
    # 4. Context token progression
    print("\n=================== CONTEXT TOKEN PROGRESSION (Sampled) ===================")
    step_usages = []
    for ev in events:
        if ev.get("type") == "assistant/chunk" and ev.get("data", {}).get("chunk", {}).get("type") == "usage":
            u = ev["data"]["chunk"]["usage"]
            step = ev["data"].get("step", 0)
            ts = datetime.fromtimestamp(ev.get("time", 0)/1000).strftime("%H:%M:%S")
            step_usages.append((ts, step, u))
            
    for ts, step, u in step_usages[::3]:  # sample every 3 steps
        inp = u.get("inputTokens", 0)
        out = u.get("outputTokens", 0)
        cache = u.get("cacheReadTokens", 0)
        tot = inp + cache
        print(f"[{ts}] Step {step:2d}: input={inp:5d}, cache={cache:5d}, total_ctx={tot:5d}, out={out:4d}")
    if step_usages:
        ts, step, u = step_usages[-1]
        inp = u.get("inputTokens", 0)
        out = u.get("outputTokens", 0)
        cache = u.get("cacheReadTokens", 0)
        print(f"[{ts}] Final Step {step:2d}: input={inp:5d}, cache={cache:5d}, total_ctx={inp+cache:5d}, out={out:4d}")

    # 5. Tool call summary
    print("\n=================== TOOLS CALLED ===================")
    tool_counts = {}
    for ev in events:
        if ev.get("type") == "tool/call":
            name = ev.get("data", {}).get("name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1
    for name, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {cnt} calls")

if __name__ == "__main__":
    run()
