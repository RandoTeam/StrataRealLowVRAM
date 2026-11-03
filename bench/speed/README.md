# Speed measurements

Two tasks, measured with the same saved prompt (`prompt.txt`, about 1,900 tokens, a question at the end):

- **chat** — a chat request through the server. This is what a person sees.
- **raw** — the engine's `--tokens` mode with the prompt's ids and no chat template. The model *continues* the text
  instead of answering it. Kept for comparison with the numbers this project published before 2026-11-03, never as the
  headline.

```sh
python bench/speed/run.py --config strata-clover-apex-i-nanoplus.json --model clover --task chat --runs 8
python bench/speed/run.py --config strata-flint-ud-q3_k_xl.json --model flint --task chat --runs 8
```

The config is the one setup wrote for that model; its flags are the settings, and they are copied into the result JSON
together with the engine's SHA-256 and the prompt's token count. A run appends one line to `runs.log`; nothing is
rewritten or deleted. Results land in `results/<model>-<task>.json`.

Measure on a quiet machine: a CUDA build, a browser or this agent's own work on the same cores moves the number by a
quarter or more (measured: the same gate read 27.5 tokens/s on a quiet machine and 15.9 during a build).
