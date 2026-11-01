# Measured on a 4 GB card

What this project's engine does on the laptop it is made for. Every number here comes from a log on that laptop;
nothing is estimated.

**The laptop:** NVIDIA GeForce RTX 3050 Laptop, 4 GB of VRAM, `sm_86`; AMD Ryzen 5 5500U, 6 cores and 12 threads;
32 GB of DDR4-3200; Windows 11. The engine is built from this repository with CUDA 12.8
(`engine-fourgbvram/strata-fourgbvram.exe`, `kSnapPlaces = 7`).

**What the server prints.** Each request adds one line to the log:

```text
strata serve: prompt 495 tokens = 12 reused + 483 read in 2151 ms (224.6 tok/s), 9148 generated in 349079 ms (26.2 tok/s), drafts accepted 1009 of 1492
```

- *prompt N tokens* is what the request carried; *reused* is what the conversation cache already held, *read* is
  what the engine had to take in, and the speed is for the read part.
- *generated* is the answer, and its speed counts from the first token to the last.
- *drafts accepted* is the draft head's guesses the model kept; the rest it replaced.

## Clover-35B-A3B-Preview APEX-I-NANOPLUS

`Clover-35B-A3B-Preview.APEX-I-NanoPlus-V2.1.gguf`, 11.69 GB, 32K context, 2026-10-31. The run config is
`strata-clover-apex-i-nanoplus.json`: the engine's defaults plus `--expert-profile` and
`--conversation-cache-mib auto`. The whole log:

```text
[strata] 2026-10-31 01:50:09 engine started: ...\strata-fourgbvram.exe --serve --model ...\Clover-35B-A3B-Preview.APEX-I-NanoPlus-V2.1.gguf --max-context 32768 --expert-profile ...\expert-profile.txt --conversation-cache-mib auto
strata-fourgbvram: STRATA_FATTN: prompt attention through llama.cpp's flash-attention kernel (STRATA_FATTN=0 turns it off)
strata-fourgbvram: STRATA_FDEC: decode attention through the tensor-core decode kernel from 4,096 tokens of context (STRATA_FDEC=0 turns it off)
strata-fourgbvram: loaded in 1.8 s; 244 of 10240 routed experts in VRAM (0.2 GiB), 12 CPU threads
strata-fourgbvram: conversation cache sized from the free RAM: 8192 MiB for up to 4 parked conversations (--conversation-cache-mib N sets it, 0 turns it off)
strata serve: prompt 13 tokens = 0 reused + 13 read in 938 ms (13.9 tok/s), 43 generated in 1453 ms (29.6 tok/s), drafts accepted 0 of 2
strata-fourgbvram: the expert cache 244 -> 337 experts (0.34 GiB) for the VRAM free now, 127 MiB left
strata serve: prompt 495 tokens = 12 reused + 483 read in 2151 ms (224.6 tok/s), 9148 generated in 349079 ms (26.2 tok/s), drafts accepted 1009 of 1492
```

What it says:

| | |
| --- | ---: |
| Load, 32K context | 1.8 s |
| Experts in VRAM at the start | 244 of 10,240 (0.2 GiB) |
| Experts in VRAM after the first request | 337 of 10,240 (0.34 GiB), 127 MiB of VRAM left |
| Writes a short answer (43 tokens) | 29.6 tokens/s |
| Writes a long answer (9,148 tokens) | 26.2 tokens/s |
| Reads a 483-token prompt | 224.6 tokens/s |
| Draft guesses kept | 1,009 of 1,492 (68%) |

The first request's 13-token read took 938 ms, which is the engine warming up rather than a read speed: at 13 tokens
there is nothing to measure.

## Flint 40B-A2.5B UD-Q3_K_XL

`flint-UD-Q3_K_XL.gguf`, 17.22 GB, 32K context, 2026-10-31. The whole log:

```text
[strata] 2026-10-31 03:00:41 engine started: ...\strata-fourgbvram.exe --serve --model ...\flint-UD-Q3_K_XL.gguf --max-context 32768 --expert-profile ...\expert-profile.txt --conversation-cache-mib auto
strata-fourgbvram: loaded in 1.2 s; 0 of 12032 routed experts in VRAM (0.0 GiB), 12 CPU threads
strata-fourgbvram: conversation cache sized from the free RAM: 7168 MiB for up to 4 parked conversations (--conversation-cache-mib N sets it, 0 turns it off)
```

It loads and serves at 32K, and it starts with an empty expert cache: its 12,032 routed experts are all outside VRAM.
No request was sent, so there is no speed to report. The expert cache fills while the model works, as Clover's did
(244 to 337); how far it gets with Flint on this card is not measured.

## Not measured

- Every other model on 4 GB of VRAM. The tables in the README for them come from an 8 GB laptop and a 32 GB
  workstation.
- Any setting that would change speed: `--vram-reserve-mib`, `--threads`, `--tail-skip`, `--kv`, `--prompt-batch`
  and the expert-cache size. The engine's defaults are what the two runs above used.
- Memory use of the rest of the PC, so the numbers hold only on a laptop with nothing else on the GPU.

## Measuring your own

The server prints the same lines for your requests, so a chat and a long prompt are enough. For a run with numbers
side by side, `python fourgbvramstrata.py benchmark --model <name> --prompt-tokens 24000`
([what it sends and prints](BACKENDS.md#benchmark)) and `BENCH.bat` (`./bench.sh`) write a report you can share
([how](COMMUNITY_BENCHMARKS.md#run-the-benchmark)).
