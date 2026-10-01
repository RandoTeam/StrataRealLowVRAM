import os
import json
import glob

BRAIN_DIR = r"C:\Users\Ilia V\.gemini\antigravity\brain"
REPORTS_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata\bench\tournament2\reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

subagent_map = {
    "S01": ("84d6b49a-737a-4745-a392-bb9c18c9bd5d", "S01_VRAM_Liberation.md"),
    "S02": ("f9b9858f-2cd9-445d-b7de-2705c91a202e", "S02_KV_Streaming.md"),
    "S03": ("cfa5f07f-475d-4419-b276-0ae83cd15719", "S03_Context_Crusher.md"),
    "S04": ("1de4fd6c-ad26-4cc5-b00d-20d29da8af61", "S04_Expert_Profile_Master.md"),
    "S05": ("4d2df264-18e3-4ab9-bce4-c37b2425bf18", "S05_Huge_Pages_Engineer.md"),
    "S06": ("7ce358df-6f46-4c65-9819-1a4f3eee3671", "S06_SSD_Direct_Oracle.md"),
    "S07": ("67f63ca0-0efb-4dba-9bf6-c8c402d2aea9", "S07_MTP_Kill_MaxVRAM.md"),
    "S08": ("0e20be9a-715f-4591-891b-8b0ec91356cc", "S08_Quant_Explorer.md"),
    "S09": ("4394bb80-590f-4c9f-b7f7-cea9fa9feffd", "S09_CPU_Pool_Architect.md"),
    "S10": ("9fa53887-3015-4a6d-80ab-834f0f880cf1", "S10_PCIe_Frac_Scientist.md"),
    "S11": ("019b5a33-6ccc-4ab5-8299-ed85945d2d8a", "S11_Speculation_Mathematician.md"),
    "S12": ("059d11be-191b-4d01-b622-d886ed433d85", "S12_StrataGP_Hunter.md"),
    "S13": ("a7e9f389-9f27-4578-80f8-d391e706fc0a", "S13_Reddit_Scout_Fresh.md"),
    "S14": ("9a2eed2d-c611-441c-aa3a-dad56cd9e4f9", "S14_OS_Kernel_Tuner.md"),
    "S15": ("af5bf150-9b73-42e7-8e98-30b1ccb5bd12", "S15_PLE_IO_Optimizer.md"),
    "S16": ("c55a82e4-4c19-4e1a-858a-f6444fe38af6", "S16_Attention_Geometry.md"),
    "S17": ("b9f57312-c9da-478d-b3a8-03fe40862cb1", "S17_Graph_Capture_Expert.md"),
    "S18": ("96e76b1c-7b23-43c5-aceb-040e5d4c8c36", "S18_Native_Shard_Pioneer.md"),
    "S19": ("ca36b2a2-20b7-4880-a3a6-c6c6b4c4fcf4", "S19_Hybrid_Frankenstein.md"),
    "S20": ("fb48a1cf-b9a5-4479-870a-1768d3e9d3f3", "S20_Cross_Engine_Analyst.md"),
}

for sid, (conv_id, fname) in subagent_map.items():
    target_file = os.path.join(REPORTS_DIR, fname)
    if os.path.exists(target_file) and os.path.getsize(target_file) > 2000:
        print(f"[OK] {sid} already exists: {fname}")
        continue
    
    # Read from full transcript first, then compact
    t_full = os.path.join(BRAIN_DIR, conv_id, ".system_generated", "logs", "transcript_full.jsonl")
    t_compact = os.path.join(BRAIN_DIR, conv_id, ".system_generated", "logs", "transcript.jsonl")
    
    t_path = t_full if os.path.exists(t_full) else t_compact
    if not os.path.exists(t_path):
        print(f"[MISSING TRANSCRIPT] {sid} {conv_id}")
        continue
        
    report_content = None
    with open(t_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # check tool calls for send_message
                for tc in obj.get("tool_calls", []):
                    if tc.get("name") == "send_message":
                        msg = tc.get("args", {}).get("Message", "")
                        if "Исследовательский вектор" in msg or "Отчёт субагента" in msg:
                            report_content = msg
            except Exception:
                pass
                
    if report_content:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[EXTRACTED & SAVED] {sid} -> {fname} ({len(report_content)} bytes)")
    else:
        print(f"[FAILED TO EXTRACT] {sid}")
