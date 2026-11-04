# Documentation map

There are two kinds of pages in this directory, and mixing them up is how a wrong number gets published. Read this
first, then the page that owns the question.

## 1. This fork's own evidence — measured on the reference laptop

RTX 3050 Laptop 4 GB, Ryzen 5 5500U, 32 GB, Windows 11. Only these pages carry numbers that were measured *here*.

| Page | What it answers |
| --- | --- |
| [BENCH-4GB.md](BENCH-4GB.md) | How fast, with which settings, on which prompt — and what did not reproduce |
| [QUALITY-4GB.md](QUALITY-4GB.md) | The 22-prompt gate, how it runs, and what it caught |
| [MEMORY-4GB.md](MEMORY-4GB.md) | Where the bytes live: VRAM, RAM and SSD per model and per token |
| [MODELS-4GB.md](MODELS-4GB.md) | Which of the twelve models fits this card, and what each one scores |
| [models/clover.md](models/clover.md), [models/flint.md](models/flint.md) | The two models this fork is about |

The measurements live in the repository as well, with their conditions: `bench/speed/` (speed, with a saved prompt and
an append-only log) and `bench/quality/` (the gate). Nothing here is estimated; a number that did not reproduce is
marked as such beside the number that replaced it.

## 2. Installing and running

| Page | What it answers |
| --- | --- |
| [INSTALL.md](INSTALL.md) | Every install question, offline installs, other drives |
| [AI_SETUP.md](AI_SETUP.md) | Handing the install to an AI assistant, step by step |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | What to do when a start fails |
| [MCP_SERVER.md](MCP_SERVER.md) | The same setup steps as MCP tools for an assistant |
| [FAQ.md](FAQ.md) | Short answers to the questions that come back |

## 3. Inherited from the donors — **not evidence for this laptop**

The rest of `docs/` comes from PolyStrata and Strata. It describes how the engine works, which flags exist and which
model families it can run, and its numbers were measured on an RTX 5090 (32 GB), an 8 GB laptop, AMD cards, Intel Arc
or a Mac. Useful for mechanisms and flags; **never quote its speeds as ours**.

The big ones: [DETAILS.md](DETAILS.md) (all settings, the API, the engine's design), [BACKENDS.md](BACKENDS.md) (every
model the engine runs), [AMD_HIP.md](AMD_HIP.md), [INTEL.md](INTEL.md), [MAC.md](MAC.md),
[MULTI_GPU.md](MULTI_GPU.md), [BATCHING.md](BATCHING.md), [NVFP4.md](NVFP4.md), [Q8_KV.md](Q8_KV.md), and the
per-family pages ([GLM.md](GLM.md), [ORNITH.md](ORNITH.md), [DEEPSEEK.md](DEEPSEEK.md), [FLINT.md](FLINT.md),
[LFM.md](LFM.md), [QWEN36.md](QWEN36.md)).

## 4. Archived

`_attic/` holds quarantined working files that the release zip leaves out. It is not documentation; nothing there is
current.
