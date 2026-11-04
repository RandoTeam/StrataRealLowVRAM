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

## The number that matters now (2026-11-03, evening): a chat request on a saved prompt

Everything below this section was measured before there was a saved prompt, and much of it in the engine's raw token-id
mode — the model continues the text instead of answering a question. That is a different task, and its speed is not
what a person sees. Both are now measured side by side, with the same saved prompt (`bench/speed/prompt.txt`, about
1,900 tokens, a question at the end), the same engine, eight runs each, 128 new tokens, 32K context, on a quiet machine:

| Model | Task | Written tokens/s, median (range) | Prompt read | Draft guesses kept |
| --- | --- | ---: | ---: | ---: |
| Clover APEX-I-NANOPLUS | **chat through the server** | **38.1 (32.6-39.1)** | 449 tokens/s | 22-51 of 41-74 |
| Clover APEX-I-NANOPLUS | raw continuation | 31.1 (29.5-34.4) | 496 tokens/s | 32-53 of 55-81 |
| Flint UD-Q3_K_XL | **chat through the server** | **21.9 (20.5-22.4)** | 331 tokens/s | 1-8 of 5-19 |
| Flint UD-Q3_K_XL | raw continuation | 21.6 (21.2-23.3) | 326 tokens/s | 28 of 38-40 |

The engine was `strata-fourgbvram.exe` with SHA-256 `a37654f5f940047f…`, built from the commit this page ships with;
the settings are the ones setup writes (Clover `--tail-skip 6 --no-graphs --kv q8_0`; Flint `--prompt-batch 512
--expert-swaps 32 --no-graphs`). The rows are `bench/speed/results/*.json`, the appended log is `bench/speed/runs.log`,
and the harness is `bench/speed/run.py`.

**The earlier table — 47.5 and 40.7 written tokens/s — is not reproduced.** Its prompt was not saved and its logs were
lost, so it cannot be re-run; the same request today writes at 38.1 and 21.9. It is kept further down with that mark,
because a number that did not reproduce is part of the record, not something to erase.

Two things stand out:

- **Flint's draft head keeps almost none of its guesses on a question-answering prompt** (1 to 8 of 5 to 19). That is
  where its time goes: the earlier 40.7 was written on a task where the guesses were kept.
- **The raw continuation is not always faster than chat.** With a question at the end of the text, continuing the
  document is the *harder* task (Clover 31.1 against 38.1), which is the opposite of what a prose prompt shows.

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
- **`--kv q8_0` used to change nothing measurable on Clover**, and that was measured at load, where the attention
  caches are still tiny (243 against 247 experts). At depth it is the largest single win in this document: see
  [the section above](#the-attention-caches-8-bit-at-depth-half-the-vram-half-the-wait). The engine takes q8_0 for
  `clover_moe`, `qwen35moe` and `kolibri1`; Flint rejects it.
- **`--prompt-batch` cuts both ways, and the 8-bit caches killed the smaller batch's case.** With fp16 caches, 512
  gave Clover +7% written tokens for −13% prompt reading at 8K (and +38% expert slots); with `--kv q8_0` the same
  comparison at 8K is 55.1 against 56.3 written — no gain, inside the noise — and at 24K only +4% (40.9 against
  39.2), both for −10 to −12% prompt reading. Once the caches are 8-bit the expert cache is no longer the shortage, so
  Clover keeps 1,024. Flint is the other way round: 512 writes 28% faster and 2,048 reads 19% faster, so setup writes
  512 there.
- **`--tail-skip 6` still pays on the bigger cache:** at 24K with `--kv q8_0`, 37.4 against 35.3 written tokens/s
  (3 of 3 rounds), prefill unchanged. Setup now writes it for Clover too; before this it did not, so a Clover install
  from setup was slower than every number in this document.
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

**The absolute figures in this table are not reproduced either** (same cause as the section at the top: the prompt was
not saved). What the section claims is the *comparison* — poly.6 against poly.7 — and that held: the same relative
gains were seen again in later sessions.

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

Flint UD-Q3_K_XL with `--prompt-batch 512`, the release engine, 32 and 64 interleaved in one session (four pairs):
**32 gives 38.0 written tokens/s on the settled answer against 30.6 for 64**, and 32 was ahead in all four pairs
(by 4.7 to 8.0).

What it says:

- **Swapping is what makes the cache useful.** Freezing it at load (`--expert-swaps 0`) costs 22 to 30%: on a 4 GB
  card 421 or 710 slots are 4 to 6% of the experts, and the answer keeps needing others.
- **The two families want different counts.** Clover does best at the engine's maximum (64); Flint does best at 32.
  Setup writes the measured value per family: a Flint config now carries `--expert-swaps 32`, Clover's nothing (the
  default).
- **The build matters as much as the setting.** The same A/B on the development build (CUDA 12.8, one architecture)
  put 64 ahead on Flint, 37.5 against 36.1. The release binary is the one users get, so its result is the one setup
  follows — a settings comparison measured on a dev build is not quoted as a property of the release.
- **An `--expert-swaps-after` prototype** (64 for the first answer, cooled to 32 after) measured 32.4 on the settled
  answer in the development build — below both fixed counts — and was dropped rather than shipped.
- **The lesson for the next measurement:** compare settings interleaved, in one session, on the binary that ships,
  with the expert profile fixed; a batch of one setting and then a batch of another is not an A/B.

## CUDA graphs cost more than they save here

`--no-graphs` launches a token's kernels one by one instead of replaying a CUDA graph recorded per layer. On this card
the replay is the more expensive of the two. Measured 2026-11-03, both models with their shipped settings (Clover
`--tail-skip 6`, Flint `--prompt-batch 512 --expert-swaps 32`), eight runs each in one session, written tokens/s:

| Model | With graphs, first / settled | `--no-graphs`, first / settled | Expert slots |
| --- | ---: | ---: | ---: |
| Clover | 38.4 / 38.8 | 38.5 / **47.5** (32.7-55.3) | 420 |
| Flint | 27.8 / 29.6 | 27.8 / **40.7** (38.3-45.1) | 518 |

**These settled figures are not reproduced (2026-11-03, evening).** The comparison itself — graphs against
`--no-graphs` — held in every later session (the first answer unchanged, later answers 10 to 25% faster without
graphs), but the absolute numbers came from a prompt that was not saved; on the saved prompt the same settings write
38.1 and 21.9 tokens/s. The *relative* result is what this section claims.

A three-request check on Flint, two runs each: with graphs 29.8 / 39.1 / 33.9 and 28.9 / 38.6 / 33.7; without,
29.9 / 44.4 / 44.3 and 27.7 / 39.3 / 42.7. The first answer is unchanged, every later one gains 10 to 25%.

Prefill does not move (Clover 519 against 514 on a 2,000-token prompt). The answers differ in their last bits, as any
change in cache timing does; the eight-prompt check below gives the same correct answers both ways.

Setup writes `--no-graphs` for the Clover and Flint families from these measurements; every other family keeps the
default. Why it wins is one hypothesis and not yet verified: a per-layer graph replay costs more than the handful of
launches it replaces when each layer's kernels are this small. A kernel-level profile is the next step.

## The attention caches 8-bit, at depth: half the VRAM, half the wait

Clover's attention cache is 40 KiB a token in fp16, so at 32K it is 1.25 GiB of a 4 GB card and the expert cache is
left with ~0.5 GiB. `--kv q8_0` stores K and V in ggml's Q8_0 blocks (34 bytes a 32 values, 0.53 of fp16). Measured
on Clover APEX-I-NANOPLUS, `--tail-skip 6 --no-graphs`, 32K context, a fresh engine per run, 128 tokens written, one
prompt per depth, median of 3 rounds alternating the two formats (2026-11-03):

| Prompt | fp16 reads | q8_0 reads | fp16 writes | q8_0 writes |
| --- | ---: | ---: | ---: | ---: |
| 2,000 tokens | 524 tokens/s | 526 tokens/s | 45.6 tokens/s | 47.1 tokens/s |
| 4,000 tokens | 679 | 677 | 61.5 | 61.3 |
| 8,000 tokens | 650 | **693** | 53.1 | **56.2** |
| 24,000 tokens | 245 | **676** | 30.9 | **39.0** |
| 32,000 tokens (one run each) | 180 | **663** | 25.7 | **33.9** |

The shape is the point: **up to ~4K the two are the same, and from 16K the fp16 cache is what starves the expert
cache.** At 24K the fp16 run reads at 245 tokens/s — a 24K prompt in 98 s — because almost no expert is cached and
every layer re-reads its experts from RAM; the q8_0 run reads the same prompt in 35 s. On the shipped engine
(`engine-fourgbvram\strata-fourgbvram.exe`, the zip's binary) the same 24K prompt: 245.6 against 679.4 tokens/s read,
30.9 against 39.1 written.

Quality: the sixteen-prompt suite passes 16 of 16 every attempt in both formats
([QUALITY-4GB.md](QUALITY-4GB.md)), and a needle placed at 15% of a 20,033-token prompt ("the calibration code for bay
seven is ZUBROVKA-7741") came back exactly right twice in both formats — the fp16 run took 79.0 s to its first answer,
the q8_0 run 38.5 s.

**Setup writes `--kv q8_0` for Clover on a card of 6 GB or less**, the card class it was measured on. A larger card
keeps fp16: the win there is untested, and the answers of a quantized cache are not byte-identical. `--kv fp16` in the
config turns it off.

## What a token costs, after the CUDA graphs came off

Clover with `--tail-skip 6 --no-graphs`, the engine's own counters, 96-token answers, 2026-11-03:

| Run | Tokens/s | Experts from VRAM | CPU experts | Router wait | Drafts kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| the shipped settings | **45 to 48** | 21.6% | 0.7-0.8 ms | 0.9-1.0 ms | 75-76 of 77-89 (97%) |
| `--no-gpu-experts` | 32 to 36 | 0 | 21-22 ms | 3.1 ms | 97% |
| `--window 1` | 32 to 40 | 44-78% | 0.3-0.6 ms | 1.2-1.3 ms | none |

- **The expert cache is not the bottleneck any more.** With it holding a fifth of the experts, the CPU's share of a
  token is under a millisecond.
- **The split is still worth it:** all experts on the CPU triples that to 21 ms and costs a third of the speed.
- **The draft window is what pays.** `--window 1` lifts the VRAM hit rate to 44-78% and is still slower, because every
  pass then serves one token instead of eight; 97% of the guesses are kept.
- **`--tail-skip` 7 and 8 are slower than 6** (41 and 35-40 tokens/s against 47): the elbow is at 6.
- **What is left — about 20 ms a token — is GPU work per layer:** attention, the output head, the cached experts'
  MMVQ, and the waits around them. A kernel-level profile is the next step; no flag has moved it so far.
- **The card is not downclocking.** A 512-token answer ran at **50.6 tokens/s** with the GPU steady at 1,749-1,762 MHz,
  about 55% busy on average (30-93% between samples) and 35 W: half the time the card waits, so the headroom is in
  overlap and batch size, not in clocks. Longer answers also score higher than the 128-token table (50.6 against
  45-48): the same warm-up spread over more tokens.

## Parallel requests on a 4 GB card: a small gain in total, a loss per answer

A second sequence's attention cache is 0.59 GiB at 32K (0.27 at 8K), and `setup.parallel_recommend` says **0 slots** on
a 4 GB card at any context. Measured anyway, Clover at 8K with `"parallel": 2`, two 128-token answers (2026-11-03):

| | Wall time for 256 tokens | Total | Each answer |
| --- | ---: | ---: | ---: |
| one after the other | 11.6 s | 22.1 tokens/s | 6.4 s and 5.2 s |
| at the same time | 9.0 s | **28.4 tokens/s** | 9.0 s each |

So two at once is +28% in total and about 40% slower per answer, and it costs the 32K context to afford the second
slot. For one person at the keyboard that is a loss; for two people sharing a server it is a small gain. Left off: the
4 GB card is expert-starved, and setup's own rule already says so.

## Does `--tail-skip` cost quality?

`--tail-skip 6` leaves out an expert outside VRAM that every token of a window ranks at place 7 or later, so the
answers can move. Measured on 2026-11-02, Clover APEX-I-NANOPLUS, the server's OpenAI endpoint, greedy
(`temperature 0`), the same eight prompts with `--tail-skip 6` and with it off:

| Prompt | `--tail-skip 6` | Without |
| --- | --- | --- |
| 137 × 29 | 3973 | 3973 |
| five primes between 100 and 200 | 101, 103, 107, 109, 113 | the same |
| `is_palindrome(s)` in Python | correct, in a code fence | correct, no fence |
| Kyiv as JSON | `{"city":"Kyiv","country":"Ukraine"}` | the same |
| three reasons a laptop is slow, in Russian | three correct reasons | the same three, other wording |
| capital of Australia | Canberra | Canberra |
| one-sentence summary of MoE | correct sentence | correct sentence, other wording |
| first 8 Fibonacci numbers | `1,1,2,3,5,8,13,21` | `1, 1, 2, 3, 5, 8, 13, 21` |

**Every answer is correct in both runs.** Four of the eight are byte-identical; the other four differ in formatting or
wording, not in content. The reasoning block differs every time, which is expected: the expert cache changes where
each expert is computed, and the last bits of the sums move with it.

What this does **not** say: eight short prompts are not a quality gate. It says the flag does not obviously lobotomise
the model on these tasks. A wider set — long context, tool calls, code that has to run — is still to come.

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
| `--stage-read-mib 0` | prefill 343 against 514: the read-time staging the default does is worth 50% |
| `--stage-read-mib 4096` | prefill 515: nothing more (it is capped at a layer's experts) |
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

## Context and speed

One process, Clover with `--tail-skip 6`, one conversation grown in five requests: the same text cut at 2K, 4K, 8K,
16K and 32K tokens, 128 tokens answered at each depth (2026-11-03):

| Prompt | Reused | Read | Prefill | Decode | Expert cache |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2,000 | 0 | 2,000 | 511 | 38.7 | 420 → 314 experts |
| 4,000 | 1,999 | 2,001 | 457 | 40.3 | 310 → 186 experts |
| 8,000 | 4,127 | 3,873 | 487 | 39.1 | — |
| 16,000 | 8,127 | 7,873 | 238 | 28.5 | — |
| 32,000 | 16,127 | 15,873 | 116 | 19.5 | — |

- **Decode holds to 8K, then falls:** 38.7 to 39.1 up to 8K, −27% at 16K, −50% at 32K.
- **Prefill of new tokens at depth falls with it:** about 490 up to 8K, 238 at 16K, 116 at 32K. A long prompt arriving
  in a long conversation is the slowest thing this engine does.
- **The expert cache shrinks as the attention caches grow:** 420 experts at a 2K prompt, 186 by 4K. That is the
  mechanism behind both curves, and it is where the work on the 8-16K zone goes.

So the two numbers in the README's table are a 2,000-token prompt at a 32K context *setting*, not a 32K-deep
conversation; the last row above is what the latter costs.

## Prefill: where the 800 tokens/s is

A 2,000-token prompt is read at 515 tokens/s, an 8,000-token one at 649: the first second of any read is the engine
warming up (the first 249-token piece runs at 441, the rest at about 650). `--prompt-batch` is the lever that moves it:

| 8,000-token prompt | Prefill | Expert slots at load |
| --- | ---: | ---: |
| `--prompt-batch 1024` (default) | 649 tokens/s | 421 |
| `--prompt-batch 2048` | **765 tokens/s** | **17** |
| `--prompt-batch 4096` | 522 tokens/s | 0 |

So 2,048-token batches buy 18% of prefill, and cost the expert cache: the batch's buffers take about 2.8 GB of VRAM,
which on a 4 GB card leaves 17 slots instead of 421, and decode collapses. `--stage-read-mib 0` (the read-time staging
off) costs 33% of prefill, so the default there is already the good one.

That is the shape of the next engine change: **lend the prompt batch its VRAM for the read and give it back to the
expert cache before the answer starts**, the way `--stage-read-mib` already lends the staging buffer. Then a long
prompt reads at 765 with 421 slots for the answer, instead of one or the other.

## Prompt size: the short prompt is the slow one

A fresh prompt, one request per process, Clover with `--tail-skip 6` (2026-11-03):

| Prompt | Time to first token | "Prefill" | After the first batch |
| ---: | ---: | ---: | ---: |
| 300 tokens | 2.97 s | 112 tokens/s | — |
| 600 | 3.34 s | 181 | — |
| 1,200 | 3.43 s | 351 | — |
| 2,400 | 4.55 s | 530 | — |
| 4,800 | 6.92 s | 694 | — |
| 8,000 | 12.46 s | 649 | ~650 tokens/s for every later piece |

Read the second and fourth columns together: the *first* batch of a prompt costs about 2.5 seconds, whatever its size,
and every later batch runs at 650 to 1,000 tokens per second. A 600-token message is therefore 2.5 seconds of expert
staging plus 0.6 seconds of tokens — 181 "tokens/s", though the tokens themselves were read at about 1,000.

The first batch of a fresh prompt routes to nearly every expert in the model (599 tokens × 8 experts × 48 layers), and
the ones that are not in the card's cache come from RAM over PCIe 3.0 x8: about 11 GB at roughly 6 GB/s. That is the
floor for a new prompt, and it is why "prefill tokens/s" is a long-prompt figure. For a short prompt the number that
matters is **time to first token**, about 2.5 to 3 seconds.

What moves it:

- `--stage-ahead` 512 or 256: about 15% (187 to 213 tokens/s cold, 223 to 240 warm).
- `--cpu-share 0.5`: about 6%.
- A *smaller* `--prompt-batch` is worse, not better: 512 gives 142 tokens/s, 256 gives 101 — every piece pays the
  staging again, so one batch is right for a short prompt.
- A smaller model file stages less; a continuation of the same conversation pays nothing (a 2,000-token follow-up:
  1,999 tokens reused, 1 read, 50 ms).

So **800 tokens/s is a long-prompt figure**: 600 tokens at 800 would be 0.75 s, while the staging alone is about 2.5 s
on this link. The target for a short prompt is time to first token near 2 seconds, reached by staging less, not by
reading faster.

## Measuring your own

The server prints the same lines for your requests, so a chat and a long prompt are enough. For a run with numbers
side by side, `python fourgbvramstrata.py benchmark --model <name> --prompt-tokens 24000`
([what it sends and prints](BACKENDS.md#benchmark)) and `BENCH.bat` (`./bench.sh`) write a report you can share
([how](COMMUNITY_BENCHMARKS.md#run-the-benchmark)).
