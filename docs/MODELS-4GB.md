# Which model for 4 GB of VRAM

The reference PC: **RTX 3050 Laptop (4 GB), Ryzen 5 5500U, 32 GB DDR4-3200, Windows 11, `sm_86`**. What it measured:
[BENCH-4GB.md](BENCH-4GB.md). What the same models do on an 8 GB laptop and a 32 GB workstation:
[the model pages](models/) and the README.

How to read the scores: vendor numbers are self-reported and their harnesses differ, so they compare within a row,
not across rows. Where a model publishes nothing for a test, this page says so. The community notes come from the
HuggingFace cards and the projects' own posts, with links; no review site was used.

## The twelve models and their sizes

| Model | Parameters | Sizes (download) | RAM beside a 4 GB card | Context cost | Images | License |
| --- | --- | --- | --- | --- | --- | --- |
| **LFM2.5-24B-A2B** (Liquid AI) | 24B / 2B active | UD-Q4_K_XL 15 GB, UD-Q3_K_XL 11, QAD-Q4_0 14, UD-Q6_K_XL 22, Q8_0 25 | 16-31 GB | 128K = +2 GB over 32K | no | LFM Open v1.0 (commercial use under $10M revenue) |
| **Qwen3.6-35B-A3B** (Qwen) | 35B / 3B active | UD-Q3_K_M 17 GB, UD-Q2_K_XL 13, UD-Q4_K_M 23 | 17-27 GB | tested to 32K | yes, +1 GB | Apache 2.0 |
| **Ornith 1.5-35B-A3B** (ornith-ai) | 35B / 3B active, Qwen3.6's architecture | APEX-I-COMPACT 17 GB, MINI 14, QUALITY 24 | 18-28 GB | tested to 32K | yes, MTP draft | its own (check the model page) |
| **Tiel Coder 35B-A3B** (peculiar-ragdoll) | 35B / 3B active; **not a fine-tune**: Ornith 1.5 requantized with its own imatrix and Sharp template, weights unchanged | UD-Q3_K_XL 17 GB, UD-IQ3_XXS 14, UD-Q4_K_XL 23 | 18-27 GB | tested to 32K | yes, +1 GB | as its base |
| **Clover-35B-A3B-Preview** (DiffuSpace) | 35B / 3B active, **diffusion** language model | Q3_K_M 17 GB, APEX-I-NANOPLUS 13, Q4_K_M 21 | 16-25 GB | tested at 32K | no, MTP off | Apache 2.0 |
| **Flint 40B-A2.5B** (Recursive) | 40B / 2.5B active | UD-Q3_K_XL 18 GB, UD-Q4_K_XL 25, UD-Q6_K_XL 36, Q8_0 42 | 22-46 GB | 128K = +2.6 GB over 32K | no | Apache 2.0 |
| **Gemma 4 26B-A4B** (Google) | 26B / 3.8B active | UD-Q3_K_M 13 GB, UD-Q2_K_XL 11, UD-Q4_K_M 17, QAT-UD-Q4_K_XL 15 | 15-21 GB | tested past 10K | yes, Google's draft | Gemma (check the terms) |
| **MiniCPM-V 4.7 35B-A3B** (OpenBMB) | 35B / ~3B active, Qwen3.6's architecture | Q4_K_M 21 GB, mix_q3gud_im 16, Q6_K 29 | 19-31 GB | tested past 10K | yes, no draft | Apache 2.0 |
| **Kolibri 1** (Aleph Alpha) | 78B / 3.5B active, German and English | Q2_K 29 GB, Q3_K_M 38, Q4_K_M 48 | 33-52 GB | 1M claimed, 262K recommended | no | Apache 2.0 |
| **GLM 5.3 Flash** (Zhipu) | 320B / 18B active | 3.0BIT 114 GB, UD-IQ2_XXS 102, REAP50-Q3_K_M 79 | 60-96 GB, and a 32 GB card | tested past 29K | no, Linux only | MIT |
| **DeepSeek V4 Flash** (DeepSeek) | 284B / 13B active | UD-IQ2_XXS 91 GB | 72+ GB, and a 32 GB card, Linux only | 128K = +0.7 GB over 32K | no | MIT |
| **Qwen3.8-Flash-Next** (Qwen; runs on the Strata project's engine) | 125B / 6B active + 51B n-gram | IQ2_XS 68 GB to Q8_0 188 GB, Coder IQ1_M 58 | 32-64 GB, and a 12 GB card | trained at 262K, up to 524K | yes | Qwen Community 1.0 |

The RAM column is what the model's file leaves for the rest of the system; a smaller size needs less. The installer
picks a size from your RAM.

## Releases, scores, community notes

### LFM2.5-24B-A2B, 12 October 2026
- <https://huggingface.co/LiquidAI/LFM2.5-24B-A2B>
- Scores (the model card, vLLM): SWE-Verified 51.0, GPQA Diamond 74.42, MATH-500 96.04, IFEval 96.75, LiveClaw 48.09,
  BFCLv4 65.01. AIME and LiveCodeBench: not published.
- Community: 588 likes, 28K downloads a month. Discussions about tool calling against Qwen3.6, and about VRAM.
- For 4 GB: the smallest active part of the twelve (2.45B), so the fastest here, and the lowest SWE score of its own
  table. 128K context costs little.

### Qwen3.6-35B-A3B, April 2026
- <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Scores (the model card): SWE-Verified 73.4, SWE-Multilingual 67.2, SWE-Pro 49.5, Terminal-Bench 2.0 51.5,
  LiveCodeBench v6 80.4, GPQA 86.0, AIME26 92.7, MMLU-Pro 85.2.
- Community: a Q3 quant on an RTX 3090 writes about 120 tokens/s on real data-science tasks
  ([thread](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/37)); Q4_K_M on a laptop's 8 GB RTX 4060 writes
  33-34 tokens/s ([thread](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/58)). 936 derivative quants.
- For 4 GB: the safest pick. Its architecture is the one this engine was written for, it has run on 8 GB laptops, and
  the agentic reports are strong.

### Ornith 1.5-35B-A3B, 23 August 2026
- <https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B>, <https://ornith.ai/ornith_1_5.html>
- Scores (card and post): SWE-Verified 79.0, SWE-Pro 59.6, SWE-Multilingual 71.4, Terminal-Bench 2.1 67.8-68.5,
  GPQA Diamond 89.2, HLE 25.6/33.4, ClawEval 72.5. AIME and LiveCodeBench: not published.
- Community: fewer loops than Qwen3.6 in an independent Terminal-Bench run (0.6-0.7 against 0.3-0.4,
  [thread](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/discussions/23)); against that, accusations of
  benchmark tuning and complaints about tool calls and loops. Reproducibility is disputed.
- For 4 GB: the best coding scores of the 35B models, and the most disputed in practice.

### Tiel Coder 35B-A3B, 19-24 August 2026
- <https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF>
- Scores (the author's, one run, so author-reported): SWE-bench-Live 12/25 (level with Opus 4.6 medium), Claw-Eval
  67.2, MMLU-Pro 73.7. Its base, Ornith 1.5, is above.
- Community: the MTP GGUF has 2.9M downloads a month, the most of any coding build here. The author's own framing:
  a fast coder, bad at trivia.
- For 4 GB: the two Cyber-Tiel-Coder sizes in the README's 4 GB table are this model's abliterated variant, and they
  are the sizes that were measured at 131K on this laptop.

### Clover-35B-A3B-Preview, 28-30 October 2026
- <https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview>, <https://diffuspace.ai/blog/clover-35b/>
- Scores (the launch post): SWE-Verified 73.6, Terminal-Bench 2.1 51.7, GPQA Diamond 83.8, MMLU-Pro 84.8,
  ZebraLogic 95.8, BFCL-v4 67.9.
- Community: 7 discussions, questions about GGUF, Vulkan and ROCm; about 100 tokens/s on an RTX 3090 with a Q4 file
  and a patched llama.cpp.
- For 4 GB: it runs on this engine at 32K (see [BENCH-4GB.md](BENCH-4GB.md)), writes 26 to 30 tokens/s here, and is
  the newest of the twelve, so the least proven.

### Flint 40B-A2.5B, 21 October 2026
- <https://huggingface.co/RecursiveSI/flint>
- Scores (the vendor): Terminal-Bench 2.1 54.9, SWE-Pro v2 89.7, SWE-Verified 74.2. AIME and GPQA: not published.
- Community: 1,251 likes, 19K downloads a month. The card asks for one B300 or H200 and about 80 GB of download.
- For 4 GB: the UD-Q3_K_XL file (17.2 GB) loads and serves at 32K on this laptop; its speed is not measured yet. Most
  of its layers look back only 513 tokens, so a long conversation costs little VRAM.

### Gemma 4 26B-A4B, 11 March 2026
- <https://huggingface.co/google/gemma-4-26B-A4B-it>
- Scores (Google): MMLU-Pro 82.6, AIME 2026 88.3, LiveCodeBench v6 77.1, Codeforces 1718, GPQA Diamond 82.3. SWE:
  not published.
- Community: 1,719 likes, 6.3M downloads a month, the most of the twelve. Unsloth's GGUF: 1.32M downloads.
- For 4 GB: the best score per active parameter (3.8B active), but no official SWE numbers, and 4 GB is untested.

### MiniCPM-V 4.7 35B-A3B, 12 October 2026
- <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>
- Scores (the vendor, vision): MMBench 91.7, MMStar 80.7, MMMU 80.2, OCRBench 91.2. SWE, AIME and GPQA: not
  published.
- Community: thin reports ("independently verified? not yet"), about 37.5 tokens/s with an image.
- For 4 GB: made for weak hardware (3 GB on a phone), but its vision path on 4 GB is not measured, and the Q4 file is
  21 GB, which is tight beside 32 GB of RAM.

### Kolibri 1, 3 October 2026
- <https://huggingface.co/Aleph-Alpha/Kolibri-1>
- Scores (the vendor): AIME 2025 96.9, AIME 2026 96.0, GPQA 84.3, LiveCodeBench v6 85.9, SWE-Verified 66.4,
  Terminal-Bench 2.1 27.7.
- Hardware: at least two A100 80 GB, about 78 GB in FP8; hands-on reports about 140 GB of VRAM.
- For 4 GB: the smallest file is 29 GB, so it needs more RAM than this laptop has to spare.

### GLM 5.3 Flash, 26 August 2026 (GLM-5.3 itself, 14 August 2026)
- <https://huggingface.co/zai-org/GLM-5.3-Flash>
- Scores (Z.ai): Terminal-Bench 2.1 84.3, DeepSWE 63.4, HLE with tools 55.3. Independently, on another harness:
  SWE-Verified 92.0, GPQA 90.2.
- For 4 GB: 320B parameters and a 79 GB file at the smallest size. It needs a 32 GB card and 96 GB of RAM.

### DeepSeek V4 Flash, 31 July 2026 (the line is tangled: V4.1-Flash, 10 September 2026)
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- Scores (the vendor, Max mode): SWE-Verified 79.0, SWE-Multilingual 73.3, Terminal-Bench 2.0 56.9, LiveCodeBench
  91.6, GPQA 88.1.
- For 4 GB: 284B parameters and a 91 GB file. It needs a 32 GB card and 72 GB of RAM.

### Qwen3.8-Flash-Next, 26-27 August 2026
- <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- Scores (the model card): DeepSWE 58.7, SWE-Pro 62.5, SWE-Multilingual 81.0, GPQA Diamond 91.7, LiveCodeBench v6
  91.9.
- Community: praised on hard tasks, complained about after tool calls (hallucinated results); the top request is a
  9B or 35B version.
- For 4 GB: 125B plus a 51B n-gram model, and 68 GB at the smallest size. It runs on the Strata project's engine and
  needs a 12 GB card and 32 GB of RAM.

## What fits on this PC

**Fits:** LFM2.5 (from 11 GB), Qwen3.6 (from 13 GB), Ornith (from 14 GB), Tiel Coder (from 14 GB), Clover (from
13 GB), Gemma 4 (from 11 GB), MiniCPM-V (from 16 GB; the Q4 file at 21 GB is tight), Flint (from 18 GB; tight).

**Does not fit:** Kolibri 1 (from 29 GB), GLM 5.3 Flash (from 79 GB), DeepSeek V4 Flash (91 GB),
Qwen3.8-Flash-Next (from 68 GB).

Measured here so far: the two Cyber-Tiel-Coder sizes, Clover APEX-I-NANOPLUS and Flint UD-Q3_K_XL. At 32K to 131K
they write 27 to 42 tokens/s; [the numbers, their logs and the settings measured one at a time](BENCH-4GB.md).

An order that costs little: Tiel Coder (re-measure the release's 131K numbers) then LFM2.5 (the fastest to load) then
Qwen3.6 (the reference for the engine) then Ornith, Clover and Gemma 4.
