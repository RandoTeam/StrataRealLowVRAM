# Deep Ternary Quantization Pipeline Specification (TQ1_0 / 1.58-Bit)

## 1. System Overview

The Strata Deep Ternary Quantization Pipeline converts high-parameter Mixture-of-Experts (MoE) and dense language models into extreme 1.58-bit ternary representations (`GGML_TYPE_TQ1_0`, Type ID 34). This yields a 1.6875 bits-per-weight (bpw) footprint while maintaining execution exclusively on the GPU through native CUDA integer DP4A accumulation.

### Target Architectures
1. **Qwen 3.8 Flash Next (125B MoE)**: 512 total experts, 8 routed ($k=8$), $d_{in}=4096$, $d_{ffn}=2048$. Active weights compressed from ~14 GB to 2.8 GB, fitting entirely within GPU VRAM.
2. **Qwen 3.6 35B A3B**: 256 total experts, 8 routed ($k=8$), $d_{in}=2048$, $d_{ffn}=512$. Model file compressed from 19.7 GB to ~7.3 GB, active experts $<600$ MB.
3. **Ornith 1.5 35B A3B**: 256 total experts, 8 routed ($k=8$), $d_{in}=2048$, $d_{ffn}=512$. Model file compressed from 19.7 GB to ~7.3 GB, active experts $<600$ MB.

---

## 2. Mathematical Foundation

### 2.1. Fast Walsh-Hadamard Transform (FWHT) & Outlier Dispersion
To eliminate high-magnitude activation and weight outliers without altering linear projection invariants ($Y = X W$):
$$Y = X W = X (Q^T Q) W = (X Q^T) (Q W) = X_{rot} W_{rot}$$
where $Q = \tilde{H}_N R$, with $\tilde{H}_N = \frac{1}{\sqrt{N}} H_N$ being the orthonormal Sylvester Hadamard matrix and $R = \text{diag}(\pm 1)$ being a randomized Rademacher sign matrix.

- **Orthogonality**: $\tilde{H}_N \tilde{H}_N^T = I_N$ (error $< 10^{-16}$ in float64).
- **Norm Preservation**: $\|\tilde{H}_N x\|_2 = \|x\|_2$ (relative error $< 10^{-7}$ in float32).
- **Theoretical Outlier Suppression Bound**:
  $$\|\tilde{H}_N x\|_\infty \le \frac{1}{\sqrt{N}} \|x\|_1$$
  - For $N = 512$ (expert intermediate size): $8.5\times - 11.0\times$ outlier reduction.
  - For $N = 2048$ (hidden dimension): $15.4\times - 20.5\times$ outlier reduction.

### 2.2. MSE-Optimal Trit Thresholding & Scaling
Weights $W \in \mathbb{R}^{256}$ within each block are mapped to ternary trits $T \in \{-1, 0, +1\}$:
$$\text{trit}(w, \Delta) = \begin{cases} +1 & w > \Delta \\ -1 & w < -\Delta \\ 0 & |w| \le \Delta \end{cases}$$

1. **Optimal Scaling Factor $\alpha^*$**:
   $$\alpha^*(\Delta) = \frac{\sum_{i: |w_i| > \Delta} |w_i|}{N_{\text{active}}}$$
2. **Optimal Critical Boundary Condition**:
   $$\Delta^* = \frac{1}{2} \alpha^*(\Delta^*)$$
3. **Analytical Distribution-Optimal Threshold**:
   $$\Delta^* \approx \frac{2}{3} \mathbb{E}[|W|]$$
   - Gaussian distribution: zero-trit sparsity = $40.5\%$, normalized MSE = $0.1934$ (within $1.7\%$ of unconstrained minimum).
   - Laplace distribution: zero-trit sparsity = $48.7\%$, strictly within target $[35\%, 50\%]$.

### 2.3. MoE Router-Weighted Activation Calibration
In MoE architectures, expert weights are calibrated using the routing-weighted activation Hessian:
$$h_{e, j} = \sum_{i=1}^N g_e(x_i)^2 X_{i, j}^2 \implies H_{\text{diag}} = G_{\text{sq}}^T S_{\text{act}} \in \mathbb{R}^{E \times d_{\text{in}}}$$
For rarely fired experts ($M_e = \sum_i g_e(x_i) \ll N$), Bayesian shrinkage prevents overfitting:
$$h_e^{\text{reg}} = \frac{h_e + \lambda \bar{h}}{M_e + \lambda}$$

---

## 3. Bit-Packing Geometry: GGML Type 34 (`TQ1_0`)

Strata's `TQ1_0` format encodes 256 weights into 54 bytes (1.6875 bpw):
- **48 bytes `qs`**: 240 trits encoded in Radix-3 groups of 5 trits per byte ($3^5 = 243 \le 256$):
  $$q = t_0 + 3 t_1 + 9 t_2 + 27 t_3 + 81 t_4 \quad (\text{with bias } +1 \implies t_i \in \{0, 1, 2\})$$
- **4 bytes `qh`**: 16 trits encoded in Radix-3 groups of 4 trits per byte ($3^4 = 81 \le 256$):
  $$q_h = t_0 + 3 t_1 + 9 t_2 + 27 t_3$$
- **2 bytes `d`**: IEEE 754 half-precision FP16 scaling factor.

### Division-Free CUDA Unpack
Unpacking on CUDA cores uses the precomputed integer power constant `POW3_PACKED = 0xF3511B090301ULL`:
```cuda
const unsigned p3 = (0xF3511B090301ULL >> (t * 8)) & 0xFF;
const unsigned q = (q_byte * p3) & 0xFF;
const int trit = int((q * 3) >> 8) - 1; // {-1, 0, +1}
```

---

## 4. Selective Quantization Policy

To avoid reasoning and linguistic degradation, the model is split into sensitive (protected) and compressible (ternarized) tensors:

| Tensor Pattern | Target Type | Rationale |
| :--- | :--- | :--- |
| `token_embd.weight` | FP16 / Original | High sensitivity; vocabulary embeddings dictate token geometry. |
| `output.weight` | FP16 / Original | Final logits distribution determines sampling sharpness. |
| `*norm*.weight` | FP32 / FP16 | Normalization scales are 1D vectors; negligible memory footprint. |
| `*ffn_gate_inp*.weight` | FP16 / Q8_0 | Expert router weights decide routing paths; must remain exact. |
| `*ffn_gate_exps*.weight` | **TQ1_0 (Type 34)** | MoE expert projections; 70-85% total model weight mass. |
| `*ffn_up_exps*.weight` | **TQ1_0 (Type 34)** | MoE expert projections; 70-85% total model weight mass. |
| `*ffn_down_exps*.weight` | **TQ1_0 (Type 34)** | MoE expert projections; 70-85% total model weight mass. |
| `*attn_q*, *attn_k*, *attn_v*` | **TQ1_0 (Type 34)** | Attention projections; outlier-dispersed via FWHT. |
| `*attn_output*.weight` | **TQ1_0 (Type 34)** | Attention output; outlier-dispersed via FWHT. |

---

## 5. Verification & Parity Test Suite

The pipeline is verified by a 78-test test suite:
- `tests/test_hadamard.py` (22 tests): Orthogonality, Parseval isometry, Cauchy outlier suppression.
- `tests/test_trit_sampler.py` (20 tests): Mathematical threshold proofs, Radix-3 roundtrip, bitwise CUDA parity.
- `tests/test_activation_calibrator.py` (26 tests): Simultaneous 512-expert GEMM Hessian estimation, router weighting, Bayesian shrinkage.
- `tests/test_ternarize_model.py` (10 tests): End-to-end GGUF generation, 32-byte alignment, Strata C++ reader verification.
