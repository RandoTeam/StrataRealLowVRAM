# Measured on a 4 GB card

What this project's engine does on the laptop it is made for. Every number here comes from a run on that laptop;
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

`flint-UD-Q3_K_XL.gguf`, 17.22 GB, 32K context. The log of the first start, 2026-10-31:

```text
[strata] 2026-10-31 03:00:41 engine started: ...\strata-fourgbvram.exe --serve --model ...\flint-UD-Q3_K_XL.gguf --max-context 32768 --expert-profile ...\expert-profile.txt --conversation-cache-mib auto
strata-fourgbvram: loaded in 1.2 s; 0 of 12032 routed experts in VRAM (0.0 GiB), 12 CPU threads
strata-fourgbvram: conversation cache sized from the free RAM: 7168 MiB for up to 4 parked conversations (--conversation-cache-mib N sets it, 0 turns it off)
```

That run got no expert into VRAM, and no request was sent in it. The same file started with the card to itself
reports **342 of 12,032 experts (0.4 GiB)**, so the difference was what else held the card: Clover's server had been
serving since 01:50 and kept its own share. One model at a time on 4 GB.

With the card free, a 2,000-token prompt is read at 379 tokens/s and 128 tokens are written at 24.5 tokens/s
(median of three runs, engine defaults). `--prompt-batch 512` writes 31.5 tokens/s and `--prompt-batch 2048` reads
453 tokens/s; [the table above](#settings-measured-one-at-a-time) has both.

The first request of a fresh process is slower than the rest: on Flint it read the same prompt at 131 tokens/s where
the next runs read it at 379. The engine records a layer's kernels once (CUDA graphs), and the expert cache has no
history yet.

## Not measured

- Every other model on 4 GB of VRAM. The tables in the README for them come from an 8 GB laptop and a 32 GB
  workstation.
- Memory use of the rest of the PC, so the numbers hold only on a laptop with nothing else on the GPU.

## Settings, measured one at a time

One engine process per run, the model's own tokenizer, a 2,000-token prompt, 128 new tokens, 32K context, greedy
sampling, nothing else on the GPU. The median of three runs, except where a row says otherwise. "Experts at load" is
what the engine reports before the first request.

Clover-35B-A3B-Preview APEX-I-NANOPLUS:

| Setting | Reads a 2,000-token prompt | Writes 128 tokens | Experts at load |
| --- | ---: | ---: | ---: |
| the engine's defaults | 505 tokens/s | 31.3 tokens/s | 247 |
| `--tail-skip 6` | 506 tokens/s | 36.3 tokens/s | 247 |
| `--prompt-batch 2048` | 566 tokens/s | 30.8 tokens/s | 0 |
| `--prompt-batch 512` | 409 tokens/s | 29.9 tokens/s | 454 |
| `--threads 6` | 507 tokens/s | 32.2 tokens/s | 247 |
| `--window 4` | 511 tokens/s | 31.5 tokens/s | 495 |
| `--kv-whole` (one run) | 509 tokens/s | 30.8 tokens/s | 0 |
| `--kv q8_0` (one run) | 513 tokens/s | 31.6 tokens/s | 243 |

Flint 40B-A2.5B UD-Q3_K_XL:

| Setting | Reads a 2,000-token prompt | Writes 128 tokens | Experts at load |
| --- | ---: | ---: | ---: |
| the engine's defaults | 379 tokens/s | 24.5 tokens/s | 342 |
| `--prompt-batch 512` | 335 tokens/s | 31.5 tokens/s | 490 |
| `--prompt-batch 2048` (one run) | 453 tokens/s | 28.4 tokens/s | 51 |
| `--threads 6` (one run) | 374 tokens/s | 23.9 tokens/s | 342 |
| `--stage-mib 1024` (one run) | 380 tokens/s | 24.6 tokens/s | 342 |

What the numbers say:

- **`--tail-skip 6` is the one clear gain on Clover: +16% written tokens.** The engine leaves out an expert outside
  VRAM that every token of a window ranks at place 7 or later; the answers move a little, and how much was not
  measured here. Setup gives Qwen3.6 6 and leaves the other families at 0.
- **`--kv-whole` is wrong on a 4 GB card:** the whole 32K of attention caches leaves no room for a single expert.
  The engine's default, growing the caches with the conversation, is what makes 32K work here at all.
- **`--kv q8_0` changes nothing measurable on Clover** (243 against 247 experts), so its KV format is not the one the
  flag covers; the engine's help lists `qwen35moe` and `kolibri1`.
- **`--prompt-batch` cuts both ways.** On Clover 2,048 reads a prompt 12% faster and costs the expert cache at load;
  on Flint 512 writes 28% faster and 2,048 reads 19% faster. Neither is a default this project should pick for every
  model; both are worth a try on yours.
- **Six CPU threads are not faster than twelve** on either model, so the old configs' `--threads 6` only cost speed.

## The engine this release ships

The zip's engine is built with CUDA 13.0 for sm_75, 80, 86, 89 and 120 (RTX 20 to RTX 50) with `STRATA_PORTABLE`, from
v0.1.45-poly.7 (slots sized by each layer's experts, read-ahead of every token's guess, page work queued behind the
reads). The same harness, the same prompt, the same settings — only these four were re-measured, the median of three
runs, except Flint's defaults: six runs, because its first two in a fresh process read at 112 and 258 tokens/s while
the card's clocks settled.

| Model and setting | Reads a 2,000-token prompt | Writes 128 tokens | Experts at load |
| --- | ---: | ---: | ---: |
| Clover, engine defaults | 522 tokens/s | 33.4 tokens/s | 421 (0.4 GiB) |
| Clover, `--tail-skip 6` | 519 tokens/s | 38.3 tokens/s | 421 (0.4 GiB) |
| Flint, engine defaults | 420 tokens/s | 38.5 tokens/s | 561 (0.7 GiB) |
| Flint, `--prompt-batch 512` | 372 tokens/s | 39.9 tokens/s | 710 (0.9 GiB) |

Against poly.6 (505 / 31.3, 506 / 36.3, 379 / 24.5, 335 / 31.5): the answers are 6% faster on Clover (5% with
`--tail-skip 6`) and 27 to 57% faster on Flint, and the expert cache at load more than doubled on both (Clover 247 →
421, Flint 342 → 561), because the engine now gives each layer's experts the slot size they need instead of one size
for all.

The same source built with CUDA 12.8 on the development PC measured 30.5, 36.4, 33.8 and 36.7 written tokens/s with
the same settings, so the CUDA 13 build is faster on all four. How much of that is codegen and how much the card
being warm was not separated. The tables above are poly.6 numbers; every setting there other than these four was not
measured again.

## How many experts swap per answer

`--expert-swaps` is a concurrency cap: while an answer is written, up to N experts are on their way into VRAM in place
of cached ones the answer uses less. The default is 64 and the engine refuses more. Two requests in one process — the
answer, then the same prompt again so the cache has settled — on 2026-11-02:

Clover APEX-I-NANOPLUS with `--tail-skip 6`, median of three runs:

| `--expert-swaps` | First answer | Second answer |
| ---: | ---: | ---: |
| 0 (cache frozen at load) | 28.4 | 31.5 |
| 32 | 37.5 | 40.3 |
| 64 (default) | **41.8** | **45.3** |

Flint UD-Q3_K_XL with `--prompt-batch 512`, 64 and 32 interleaved in one session (64, 32, 64, 32, 64, 32), median of
three each: 64 gives **27.8** first and **37.5** settled, 32 gives 27.5 and 36.1.

What it says:

- **Swapping is what makes the cache useful.** Freezing it at load (`--expert-swaps 0`) costs 22 to 30%: on a 4 GB
  card 421 or 710 slots are 4 to 6% of the experts, and the answer keeps needing others.
- **The most the engine takes is the best here.** 64 wins on both models, on both answers, so the default stands.
- **A first batch of Flint runs put 32 ahead of 64 on the settled answer (38.0 against 29.0).** It did not reproduce:
  those runs compared two engine builds in two sessions, not two settings. Interleaving the two settings in one
  session reversed it. An `--expert-swaps-after` prototype that ran 64 for the first answer and cooled to 32 measured
  **32.4** — below both fixed counts — and was dropped.
- **The lesson for the next measurement:** compare settings interleaved, in one session, on one engine build, with
  the expert profile fixed; a batch of one setting and then a batch of another is not an A/B.

## Settings that did not help

Clover APEX-I-NANOPLUS, `--tail-skip 6` unless the row says otherwise, three interleaved rounds, 2026-11-02. "First /
second" are written tokens/s of the first answer and of the same prompt asked again:

| Setting | What it did |
| --- | --- |
| `--ram-experts-mib 6144` | 21.4 / 18.7 against 38.4 / 40.9: 25 to 50% slower |
| `--ram-experts-mib 8192` | 27.9 / 28.2: slower too |
| `--stage-ahead 2048` | prefill 265 against 514: halved |
| `--cpu-share 0` | prefill 517 against 514: nothing |
| `--stage-mib 2048` | 34.2 / 32.7: slower |
| `--prompt-batch 2048` | prefill 520 against 515, and the cache held 17 experts instead of 421: 30.5 written |
| `--prompt-batch 4096` | 0 experts at load |
| `--lookup 0` | 30.5 / 31.4: the repeat-based guesses pay for themselves |
| `--no-draft` | 36.5 / 37.3, the same as the default |
| `--window 4` against 8 | no difference: five interleaved pairs, 46.3 against 45.6, `--window 4` ahead in 2 of 5 |

## The machine varies more than the settings

The same configuration (`--tail-skip 6`, nothing else) measured **35 to 38** written tokens/s early in one session and
**45 to 48** an hour later — same binary, same prompt, same expert profile, the same 421 experts at load. That is a
±25% swing, wider than every effect in the table above.

What it changes:

- **Only interleaved pairs inside one session are comparable.** Absolute numbers from two sessions are not, and a
  batch of one setting followed by a batch of another is not an A/B (that mistake cost a reverted feature; see the
  section on `--expert-swaps`).
- **The 45 tokens/s line has been crossed here**: 45.3 to 48.0 in the last batch of pairs, with the shipped settings.
  Whether a given run lands at 35 or 48 is not decided by any flag measured so far.
- **Unseparated:** the card's clocks and temperature, the laptop's power budget, and Windows moving the process to the
  efficiency cores. The engine calls `no_power_throttling()` at start and the server starts it through
  `serve/winjob.py`; this harness starts the engine directly. Which of the three moves the number is the next thing to
  measure.

## Measuring your own

The server prints the same lines for your requests, so a chat and a long prompt are enough. For a run with numbers
side by side, `python fourgbvramstrata.py benchmark --model <name> --prompt-tokens 24000`
([what it sends and prints](BACKENDS.md#benchmark)) and `BENCH.bat` (`./bench.sh`) write a report you can share
([how](COMMUNITY_BENCHMARKS.md#run-the-benchmark)).
