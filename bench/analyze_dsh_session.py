import zstandard as zstd
import json
import sys
from pathlib import Path
from datetime import datetime

SESSION_PATH = Path(r"C:\Users\Ilia V\.dsh\sessions\--C-DeepSeak~0020Harness-Test~00201--\session-73f94640-d151-4d00-aac4-83736e9e865e\session.jsonl.zstd")

def analyze():
    dctx = zstd.ZstdDecompressor()
    with open(SESSION_PATH, 'rb') as f:
        reader = dctx.stream_reader(f)
        text_stream = reader.read().decode('utf-8', errors='replace')
    
    lines = text_stream.strip().split('\n')
    print(f"Total lines in session: {len(lines)}")
    
    events = []
    for i, line in enumerate(lines):
        try:
            ev = json.loads(line)
            events.append(ev)
        except Exception as e:
            print(f"Line {i} parse error: {e}")
            
    print(f"Parsed {len(events)} events.")
    
    # Analyze start and end
    first_ev = events[0]
    last_ev = events[-1]
    
    print("\n--- FIRST EVENT ---")
    print(json.dumps(first_ev, indent=2, ensure_ascii=False)[:1000])
    
    print("\n--- LAST EVENT ---")
    print(json.dumps(last_ev, indent=2, ensure_ascii=False)[:2000])
    
    # Event types summary
    types = {}
    for ev in events:
        t = ev.get("type", ev.get("role", "unknown"))
        types[t] = types.get(t, 0) + 1
    print("\n--- EVENT TYPES ---")
    for k, v in types.items():
        print(f"  {k}: {v}")
        
    # Check last 10 events
    print("\n--- LAST 10 EVENTS SUMMARY ---")
    for ev in events[-10:]:
        t = ev.get("type", ev.get("role", "unknown"))
        ts = ev.get("timestamp", ev.get("created_at", ev.get("time", "")))
        content = str(ev.get("content", ev.get("message", ev.get("data", ""))))[:150]
        print(f"[{ts}] type={t}: {content}")

if __name__ == "__main__":
    analyze()
