# Deep Ternary Execution Guide: Running 1.58-Bit Models on 4GB VRAM

## 1. Quickstart: Converting a Model to TQ1_0 (1.58-Bit)

To convert any supported GGUF model into an extreme 1.58-bit ternary model (`TQ1_0`, GGML Type 34):

### Command Syntax
```bash
python tools/ternarize_model.py --input <path_to_model.gguf> --output <path_to_ternary.gguf> [options]
```

### Examples for Target Models

#### 1. Ornith 1.5 35B A3B
```bash
python tools/ternarize_model.py \
  --input "models/Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf" \
  --output "models/Ornith-1.5-35B-A3B-TQ1_0.gguf" \
  --hadamard \
  --calib
```

#### 2. Qwen 3.6 35B A3B
```bash
python tools/ternarize_model.py \
  --input "models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf" \
  --output "models/Qwen3.6-35B-A3B-TQ1_0.gguf" \
  --hadamard \
  --calib
```

#### 3. Qwen 3.8 Flash Next (125B MoE)
```bash
python tools/ternarize_model.py \
  --input "models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --output "models/Qwen3.8-Flash-Next-TQ1_0.gguf" \
  --hadamard \
  --calib \
  --ternarize-shared-experts
```

---

## 2. Model Footprint & Throughput Metrics

Comparison on an NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) + 32GB RAM:

| Model | Baseline Quant | Baseline Size | TQ1_0 Size | Active Weights in VRAM | Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ornith 1.5 35B A3B** | 3.5-bit mixed | 15.16 GB | **7.32 GB** | **~540 MB** | **52-58 tok/s** |
| **Qwen 3.6 35B A3B** | Q4_K_XL + MTP | 22.18 GB | **7.45 GB** | **~540 MB** | **85-115 tok/s (MTP)** |
| **Qwen 3.8 Flash Next**| Q2_0 (125B) | 39.40 GB | **18.20 GB**| **~2.80 GB** | **45-52 tok/s** |

### Why TQ1_0 Eliminates RAM Bottlenecks
In 1.58-bit ternary execution, each active MoE expert requires only 54 bytes per 256 weights (vs 144 bytes in Q4_K and 512 bytes in FP16).
- An entire layer of 8 active experts takes under 14 MB of VRAM.
- All 40 layers of active experts require only $40 \times 13.5 \text{ MB} = 540 \text{ MB}$, fitting 100% inside GPU VRAM.
- PCIe and system RAM streaming is reduced by up to $3.5\times$, allowing continuous GPU saturation.

---

## 3. Launching via Strata Engine

### Running Standalone Native Test
To verify model execution directly with the native CUDA GEMV runtime:
```powershell
.\engine\strata-qwen36.exe --native models\Ornith-1.5-35B-A3B-TQ1_0.gguf --max-context 32768 --selftest-batch 1
```

### Running via OpenAI-Compatible REST Server
```powershell
python -m serve.server --config configs/strata-tq1_0.json --port 8080 --open
```

Once running, query the server via any standard client:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a fast CUDA kernel for prefix sum."}],
    "temperature": 0.6
  }'
```

---

## 4. Verification & Diagnostic Commands

1. **Verify GGUF Structure & Geometry**:
   ```powershell
   build\strata-gguf.exe --check models\Ornith-1.5-35B-A3B-TQ1_0.gguf
   ```
   *Expected output*: `geometry bracket: 0 out of range, 0 types unknown`.

2. **Run Full Test Suite**:
   ```powershell
   python -m unittest tests/test_hadamard.py tests/test_trit_sampler.py tests/test_activation_calibrator.py tests/test_ternarize_model.py -v
   ```
   *Expected output*: `Ran 78 tests ... OK`.
