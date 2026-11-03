# Quality on a 4 GB card

Speed is not worth much if the answers get worse. This page is the smoke test the project runs after a change that
touches speed: sixteen prompts across seven domains, greedy, scored pass or fail. It is deliberately shallow — it
catches an answer that is empty, off-topic, in the wrong language, refuses or repeats the question. It is not a
benchmark and it does not score quality.

## How it runs

```sh
# the server must already run with the model loaded
python bench/quality/run.py --model local --repeat 2 --out bench/quality/results/clover.json
```

The prompts are `bench/quality/prompts.json`; the checks are in `bench/quality/run.py` (`contains`, `json_keys`,
`fib8`, `comma_count`, `word_count`, `russian_list`, `palindrome_code`). `--max-tokens` defaults to 1200 because both
models write a reasoning block first, and a small budget ends the answer inside it. `--repeat 2` asks every prompt
twice: greedy answers can still move, because the expert cache moves. A row is `ok` only if every attempt passes;
`FLAKY` means one attempt failed and one passed.

## Result, 2026-11-03

The shipped settings (`--tail-skip 6 --no-graphs` on Clover, `--prompt-batch 512 --expert-swaps 32 --no-graphs` on
Flint), 32K context, two attempts per prompt:

| Model | Passed every attempt | Domains |
| --- | ---: | --- |
| Clover APEX-I-NANOPLUS | **16 of 16** | arithmetic ×3, logic ×2, code ×2, JSON and tool call ×2, Russian, German, instruction ×2, summary, fact, Fibonacci |
| Flint UD-Q3_K_XL | **15 of 16** | the same, but the Russian instruction |

**The one failure is real and it is Flint's.** Asked in Russian to list three reasons a laptop can run slowly, Flint
repeated the instruction back (both attempts this run). Earlier runs of the same prompt, same engine, same settings,
gave a correct numbered list once and a greeting ("Привет, дорогой, друг!") once. Three runs, three behaviours: **Flint's
Russian instruction-following is unstable.** Clover answered it correctly in every run.

Everything else passed for both: arithmetic with exact results, the logic traps (roses, shortest), a palindrome
function written correctly (two different idioms), a list-comprehension result, JSON with the asked keys, a JSON tool
call, German translation, an eight-word sentence, exactly five primes, a one-sentence summary, the capital of
Australia, and the Fibonacci sequence in either convention (0- or 1-based).

## What this does and does not cover

- **Covers:** does the model still answer the question, in the right language, with the asked shape.
- **Does not cover:** long context, tool-call loops that have to run, code that has to execute, reasoning depth,
  multi-turn consistency, and anything where a wrong answer looks plausible. Those are what a fuller gate would add.
- **A failure is a prompt to look, not a verdict:** two of the first runs "failed" only because a correct palindrome
  used `split()` instead of `replace()` and because Fibonacci was 0-based. The checks were widened; the answers were
  right.

## Next

- Run this suite after every speed change and compare the pass counts, per model.
- Add the domains it lacks: a 20K-token needle, a tool loop over three calls, a Python snippet that must run.
- Give Flint's Russian prompt a second form to see whether the instability is the language or the instruction shape.
