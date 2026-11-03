# Where the bytes live: the three memory tiers on a 4 GB card

A 4 GB card, 32 GB of RAM, one SSD. Each tier is several times slower than the one above it, and the engine's speed
is decided by how many bytes of a model each token has to pull through them. This page does the arithmetic for the two
models measured here, from their own GGUF metadata and from runs on the laptop.

The machine: RTX 3050 Laptop (4 GB, `sm_86`), Ryzen 5 5500U, 32 GB DDR4-3200, Windows 11. Engine: `strata-fourgbvram`
at [PolyStrata](https://github.com/VecSzn/PolyStrata) v0.1.45-poly.7, CUDA 13.

## The tiers

| Tier | What it holds here | Speed | Where that number comes from |
| --- | --- | ---: | --- |
| **VRAM** | dense weights, attention caches, the output head, the cached experts, staging buffers | about 150 GB/s | the card's memory bus (128-bit GDDR6); not measured in this project |
| **RAM** | the model file in the OS file cache, parked conversations, the expert profile | about 45 GB/s | DDR4-3200 dual channel; matched by the engine's own CPU-expert time below |
| **PCIe 3.0 x8** | what moves between the two | 6 to 8 GB/s | the engine's start line ("up to ~8 GB/s to the GPU") |
| **SSD** | the model file itself, read when it is not in the page cache | the drive's, not measured here | — |

## What one token costs, per model

Computed from each file's metadata (`general.architecture` fields) and its tensor sizes:

| | Clover APEX-I-NANOPLUS | Flint UD-Q3_K_XL |
| --- | ---: | ---: |
| layers | 40, all mixture-of-experts | 49 (2 dense, 47 MoE) |
| experts per layer | 256 | 256 |
| experts used per token | **8** | **6** |
| routed experts a token touches | **320** | **282** |
| bytes per expert | 1.06 MB | 1.37 MB |
| **routed-expert bytes per token** | **338 MB** | **388 MB** |
| dense weights (attention, embeddings, head, router, shared experts) | 1.75 GB | 1.96 GB |
| attention cache (K and V, fp16) per token | 40 KiB | 47 KiB |
| the same at 16K of context | 0.62 GiB | 0.73 GiB |
| the same at 32K of context | **1.25 GiB** | **1.47 GiB** |
| the same with `--kv q8_0` per token (Clover only) | 21 KiB | — |
| the same with `--kv q8_0` at 32K | **0.66 GiB** | — |

Two consequences, and they are the whole story of this machine:

1. **A token needs about a third of a gigabyte of expert weights.** From RAM at 45 GB/s that is 7.5 ms, which is
   exactly what the engine reports as "CPU experts 7.6 ms/token" on Clover. That part of decode is RAM-bound, not
   compute-bound.
2. **One token's expert set is 320 or 282 slots, and the card holds 421 or 712.** So VRAM holds *one to two tokens'
   worth* of experts, no more; the next token routes to a different set, and most of it comes from RAM.

## The VRAM budget, by context length

The card is 4 GB. Subtract what is not expert cache on Clover:

| At 32K of context | GiB |
| --- | ---: |
| dense weights | 1.75 |
| attention caches | 1.25 |
| staging and working buffers | ~0.4 |
| **left for experts** | **~0.5** |
| which is, at 1.06 MB an expert | **421 slots**, 1.3 tokens' worth |

At 16K the attention caches drop to 0.62 GiB and the expert cache could roughly double; at 4K it could be three times
what it is at 32K. **The attention cache is what steals VRAM from the experts as a conversation grows**, which is the
mechanism behind the decode curve in [BENCH-4GB.md](BENCH-4GB.md#context-and-speed) (38.7 tokens/s at a 2K prompt,
28.5 at 16K, 19.5 at 32K).

**`--kv q8_0` takes that back, and setup now writes it for Clover on a card of 6 GB or less.** The attention caches
become 0.66 GiB at 32K instead of 1.25, and the 0.6 GiB goes to the expert cache — measured at 24K of context: 39.0
against 30.9 written tokens/s, and a 24K prompt read at 676 against 245 tokens/s, because a prefill with almost no
cached experts re-reads every expert of every layer from RAM
([BENCH-4GB.md](BENCH-4GB.md#the-attention-caches-8-bit-at-depth-half-the-vram-half-the-wait)). Flint keeps fp16: the
engine's 8-bit cache is `clover_moe`'s and the qwen3.5/3.6 family's, not Flint's.

## What each tier actually does in a run

- **VRAM**: the dense weights, the head, the attention caches, and 421 (Clover) or 712 (Flint) experts — the hottest
  4 to 6% of the model. Every expert that is here is computed on the GPU.
- **RAM**: the rest of the file, read by the page cache. The CPU computes those experts at about 45 GB/s of reads; this
  is the 7.6 ms per token the engine prints.
- **PCIe**: the experts that move between RAM and the VRAM slots while an answer is written — about 30 MB per token
  when the cache turns over, overlapped with compute.
- **SSD**: cold start only. Both files (12.6 and 18.5 GB) fit in 32 GB of RAM, so after the first read the SSD is not
  in the loop. That is also why `--ram-experts-mib`, which holds a private RAM copy beside the page cache, made things
  25 to 50% slower instead of faster: it duplicated what the OS already had.

## What this says about speed, and what it does not

- **More VRAM for experts is the highest-leverage change, and the lever is the attention cache.** It was measured:
  `--kv q8_0` gives back 0.6 GiB at 32K, and at 24K of context that is 39.0 written tokens/s against 30.9 — a
  quarter more, from a setting, not from a kernel. Halving the context does the same thing more bluntly.
- **Pulling the last few percent of experts into VRAM is not where the time goes.** If every expert a token needs were
  in VRAM, 338 MB at 150 GB/s would be 2.2 ms; the token takes 26 ms. The rest is GPU work per layer, of which the
  cached experts are a part — and **how much of the 26 ms is compute, launch overhead and waiting is not separated yet**
  (the next measurement: `--no-gpu-experts` against the default, and a per-layer timing).
- **The window is what amortises the RAM traffic.** With 8-token windows and a draft acceptance around 68% (Clover),
  one staging pass and one RAM read serve several tokens, which is why the draft is worth its verify pass. Turn it off
  and the same 338 MB is paid per token.

## The next numbers to measure

1. Where the ~26 ms per token goes: `--no-gpu-experts` (all experts on the CPU), the default, and a per-layer timing.
2. What one staging pass serves: tokens/s with `--window 8` against `--window 1`, draft acceptance for each.
3. The prompt batch's scratch is 429 MiB at `--prompt-batch 1024` and 88 MiB at 64; a layout that shrinks it between
   requests would be worth about 340 MiB of experts at a short context, and is not built.

Partly measured already; the rest is what the plan in the repository's local notes puts first.
