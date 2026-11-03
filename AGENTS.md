# AGENTS.md

> **This branch is RandoTeam/4gbVramStrata `main` (4gbVramStrata line).** Release script: `tools/package_release_131k.py`
> (`4gbVramStrata-v0.1.45-131k-win-x64`). Origin is canonical (`RandoTeam/4gbVramStrata`: public, a fork of
> [Strata](https://github.com/Niko1221/Strata); until 2026-11-01 it was called `StrataRealLowVRAM`, which GitHub
> redirects). `upstream` is the primary donor (`VecSzn/PolyStrata`), `strata-upstream` the secondary
> (`Niko1221/Strata`). `RandoTeam/4gbVramStrata-private` is an archived copy of this branch, kept for its issue and
> pull-request history; nothing is pushed there.
> The experimental 0.2.0-based port lives on `feature/strata-020-fourgbvram-131k` (`tools/package_portable.py` there).

4gbVramStrata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

A second engine, `strata-fourgbvram` (`src/fourgbvram/`), runs GLM 5.3 Flash, Ornith 1.5 (35B and 122B), DeepSeek V4
Flash and other GGUF models from their GGUF files, on Linux or Windows: [docs/GLM.md](docs/GLM.md),
[docs/ORNITH.md](docs/ORNITH.md), [docs/DEEPSEEK.md](docs/DEEPSEEK.md), [docs/BACKENDS.md](docs/BACKENDS.md). The Qwen
engine, the installer, the server and the web app come from [Strata](https://github.com/Niko1221/Strata).

## Installing 4gbVramStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, 4gbVramStrata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Text taken from the Strata project names that project. After merging from it,
  `python tools/brand_names.py` writes 4gbVramStrata again; its first lines say what it leaves alone.
  (The donor's `tools/polystrata_names.py` is kept for merges.)
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Keeping the plan and the documents current

Work here is not finished when the code runs. Three things have to be true as well: the plan says where we are, the
public pages say what was measured and under what conditions, and the release in `dist/` was rebuilt from the commit
that did it.

- **Read `.local/ROADMAP.md` before starting, and update it in the same sitting as the work.** It holds the goals, the
  measured baselines, everything tried, the queue and the calendar. It is git-ignored on purpose: plans are not
  published, and nothing from it is quoted in a public file.
- **After every experiment, write down** what was tried, the numbers with their conditions, what it means, and what
  goes next. A finding that is not in the plan or in a page did not happen.
- **Public measurements go to the page that owns the topic** (`docs/BENCH-4GB.md` for this hardware), with machine,
  engine build, prompt, settings and number of runs. A number that did not reproduce is published as not reproduced,
  with what it was measured against.
- **A settings comparison is interleaved, in one session, on the binary that ships, with the expert profile fixed.** A
  batch of one setting and then a batch of another is not an A/B; that mistake cost a reverted feature already.
- **A speed win ships only with its quality check.** No lossy flag or engine change is called an improvement before a
  same-prompt comparison says the answers did not get worse.
- **Documents and the release move together.** A change that alters a documented number rebuilds
  `tools/package_release_131k.py`, uploads the zip with `gh release upload … --clobber`, and keeps the release notes'
  table in step.
- **New knowledge goes where it will be found:** the model's page in `docs/models/`, the topic page in `docs/`, a row in
  `docs/BENCH-4GB.md`, or a line in the local plan when it is a plan or an untested hunch.

**Before a task is called done:** the plan lists the result; the owning page has the numbers with their conditions (or
says nothing changed); the build and the tests that fit ran; the release zip was rebuilt if a documented number moved;
and the next step is written down for whoever picks it up.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
