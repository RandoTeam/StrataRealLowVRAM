<h1 align="center">4gbVramStrata</h1>

<p align="center"><b>Run 24B to 40B mixture-of-experts models on a laptop with a 4 GB graphics card.</b><br>
No cloud, no API key, nothing leaves the machine.</p>

<p align="center">
  <a href="#measured-on-a-4-gb-laptop">Measured</a> ·
  <a href="#the-models">Models</a> ·
  <a href="#install">Install</a> ·
  <a href="#use-it">Use it</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="docs/BENCH-4GB.md">Benchmarks</a>
</p>

---

A 35B mixture-of-experts model has about 35 billion parameters but uses only 3 billion of them for any one token. The
rest sit still. That is the opening this project takes: keep the whole model in the PC's memory, keep the few experts
that the next token needs in the graphics card, and let the CPU compute what does not fit. A 4 GB card is enough,
because it never has to hold the model — only the hottest slice of it.

This is a fork of [PolyStrata](https://github.com/VecSzn/PolyStrata), which is a fork of
[Strata](https://github.com/Niko1221/Strata). It is tuned, measured and documented on one machine: an **RTX 3050
Laptop with 4 GB of VRAM**, a Ryzen 5 5500U and 32 GB of RAM, on Windows 11.

## Measured on a 4 GB laptop

Reference machine: RTX 3050 Laptop (4 GB, `sm_86`), Ryzen 5 5500U (6 cores, 12 threads), 32 GB DDR4-3200, Windows 11.
The engine is built from this repository with CUDA 13 for `sm_75` to `sm_120`. A 2,000-token prompt, 128 new tokens,
32K context, the median of three runs:

| Model | File | Reads the prompt | Writes the answer | Experts in VRAM |
| --- | ---: | ---: | ---: | ---: |
| Clover-35B-A3B-Preview APEX-I-NANOPLUS | 11.7 GB | 522 tokens/s | 33.4 tokens/s (38.3 with `--tail-skip 6`) | 421 of 10,240 |
| Flint 40B-A2.5B UD-Q3_K_XL | 17.2 GB | 420 tokens/s | 38.5 tokens/s (39.9 with `--prompt-batch 512`) | 561 of 12,032 |

Two files of 12 and 17 GB decode at 33 to 39 tokens per second here, though neither fits in the card at all: the load
takes about two seconds, and after that the card runs the experts it holds while the CPU runs the rest. That is a
fresh 2,000-token prompt; a conversation grown to 32K is about half as fast
([the curve](docs/BENCH-4GB.md#context-and-speed)).

Every number, its log, the settings measured one at a time, what a conversation costs as it grows to 32K, and what
was **not** measured: **[docs/BENCH-4GB.md](docs/BENCH-4GB.md)**.

## The models

Twelve models, each from the vendor's own GGUF or a well-known quantizer's. Two are measured on 4 GB here; the rest
are listed with what the installer estimates for this card. Sizes are the smallest download; details, quant sizes,
vendor scores and sources: **[docs/MODELS-4GB.md](docs/MODELS-4GB.md)**.

| Model | Parameters | Smallest file | On a 4 GB card |
| --- | --- | ---: | --- |
| **Clover-35B-A3B-Preview** (DiffuSpace) | 35B / 3B active | 13 GB | **measured**, 33.4 tokens/s |
| **Flint 40B-A2.5B** (Recursive) | 40B / 2.5B active | 18 GB | **measured**, 38.5 tokens/s |
| **LFM2.5-24B-A2B** (Liquid AI) | 24B / 2B active | 11 GB | should start; not run here |
| **Qwen3.6-35B-A3B** (Qwen) | 35B / 3B active | 13 GB | no: 4.2 GB before any expert |
| **Ornith 1.5-35B-A3B** (ornith-ai) | 35B / 3B active | 14 GB | the smallest size should start; not run here |
| **Tiel Coder 35B-A3B** (peculiar-ragdoll) | 35B / 3B active | 14 GB | no: 4.4 GB before any expert |
| **Gemma 4 26B-A4B** (Google) | 26B / 3.8B active | 11 GB | no: 4.1 GB before any expert |
| **MiniCPM-V 4.7 35B-A3B** (OpenBMB) | 35B / 3B active | 16 GB | should start; not run here |
| **Kolibri 1** (Aleph Alpha) | 78B / 3.5B active | 29 GB | no: 4.1 GB before any expert |
| **GLM 5.3 Flash** (Zhipu) | 320B / 18B active | 79 GB | no: needs a 32 GB card |
| **DeepSeek V4 Flash** (DeepSeek) | 284B / 13B active | 91 GB | no: needs a 32 GB card |
| **Qwen3.8-Flash-Next** (Qwen) | 125B / 6B + 51B n-gram | 58 GB | no: needs a 12 GB card and 32 GB of RAM |

"Before any expert" is what a size takes in VRAM for its dense weights, attention caches and buffers. A card that has
that much left over starts, with a smaller expert cache; **[docs/BENCH-4GB.md](docs/BENCH-4GB.md#the-engine-this-release-ships)**
has what that costs.

## Install

Windows 10 or 11, 64-bit. An NVIDIA RTX 20 series card or newer, driver 580 or newer. 16 GB of RAM runs the smaller
models, 32 GB the two measured above. Python 3.10 or newer: setup installs it for your user account if it is missing,
no administrator needed.

1. Download **[the latest release](https://github.com/RandoTeam/4gbVramStrata/releases/latest)** and unzip it anywhere
   with room for the model you want (11 to 92 GB).
2. Run `START-HERE.bat`. On the first run it checks the PC, offers the models that fit, downloads one, writes its run
   config and starts the server.
3. Later runs start what is installed. `SETUP.bat` asks again — another model, another size, another context.
   Setup writes a `run-<model>.bat` per installed model: one click, no questions.

The first run downloads the Python packages (a few hundred MB) and the model. Nothing needs a compiler: the engine is
already inside the release, built for every RTX 20 to RTX 50 card.

## Use it

The server speaks both API dialects at once, so most tools work by changing one address:

| | |
| --- | --- |
| OpenAI | `http://127.0.0.1:8080/v1` — `/chat/completions`, `/completions`, `/models`, tool calls, streaming |
| Anthropic | `http://127.0.0.1:8080/v1/messages` — what Claude Code speaks |
| Web app | `http://127.0.0.1:8080` — chat, model switch, live monitor |
| MCP | `docs/MCP_SERVER.md` — install, start, stop and switch from an AI assistant |

Any API key works (a local server checks none). Point an agent at the address and it behaves like a hosted model, with
one difference: it is slower on the first token of a long prompt, and it lives on your desk.

## How it works

The engine that makes this project (`strata-fourgbvram`, in `src/fourgbvram/`) reads the model's GGUF file in place.
The dense weights, the attention caches and one slot per cached expert live in VRAM; every other expert stays in the
file, which the operating system caches in RAM. Each token routes to a handful of experts per layer. The engine copies
the ones it does not hold into the slots of the ones it uses least, over PCIe, while the answer is being written —
that copy is the price of a small card, and most of this project's work is making it cheaper and spending it on the
right experts.

A second engine, `strata` (`src/`, `include/`), runs Qwen3.8-Flash-Next and comes unchanged from Strata.

## Something went wrong?

- **The card seems to be the problem.** Another program is using the VRAM: close browsers with hardware acceleration,
  other model servers, anything with a GPU view. One model at a time on 4 GB.
- **The answer is much slower than the table above.** The expert cache started empty (a card just under the size) or
  another process took the VRAM. The start prints how many experts it holds; the log of the last run is
  `strata-<model>.log` beside the program.
- **The start failed.** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) has the table; the first lines of
  `strata-<model>.log` usually name the cause.

## Docs

- **[docs/BENCH-4GB.md](docs/BENCH-4GB.md)** — what this hardware measured, with the logs.
- **[docs/MODELS-4GB.md](docs/MODELS-4GB.md)** — the twelve models, sizes, scores, sources.
- **[docs/INSTALL.md](docs/INSTALL.md)** — every install question, offline installs, other drives.
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — what to do when a start fails.
- **[docs/AI_SETUP.md](docs/AI_SETUP.md)** — hand the install to an AI assistant.
- **[docs/BACKENDS.md](docs/BACKENDS.md)** — the engine, its flags, and every model it runs.

## Status

The engine, the installer and the server are used daily on the reference laptop. What that means honestly:

- **Measured on 4 GB:** Clover APEX-I-NANOPLUS and Flint UD-Q3_K_XL, at 32K. Not 131K, not the other ten models.
- **131K context** is what the engine and its build support, and what the release is named after; only 32K has been
  measured here.
- **New speed settings** are measured against the previous ones in the same session, and a setting that does not
  reproduce is dropped, not documented as a win.

## Credits and license

Built on [PolyStrata](https://github.com/VecSzn/PolyStrata) by VecSzn, itself built on
[Strata](https://github.com/Niko1221/Strata) by Niko1221. The Qwen engine, the installer, the server and the web app
come from Strata; the 4 GB engine, its measurements and this documentation are this fork's. Models keep their own
licenses — each page in [docs/models/](docs/models/) names the one that applies.

MIT, see [LICENSE](LICENSE).
